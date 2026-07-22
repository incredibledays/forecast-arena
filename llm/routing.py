"""ModelRouter — task-aware, budget-aware, cache-aware tier selection.

The router is a pure decision layer. It does NOT call any provider; it
returns a `RoutingResult` describing which logical tier / concrete
provider+model / batch mode a caller should use, with the reasoning and
budget/cache flags attached. Dispatch, retry execution, and actual
token spend belong to the caller (or a later dispatcher phase).

Pipeline for `route()`:

  1. Validate the task type — deterministic operations (LMSR, Kelly,
     ActionPolicy, portfolio valuation, ...) raise `NonLLMTaskError`.
  2. Score complexity + importance from the context.
  3. Pick a base tier from the default rules (task type + scores),
     honoring an explicit `requested_tier` and SPECIALIST/LOCAL intent.
  4. Apply failure escalation (bounded by `max_escalations`).
  5. Apply budget fallback: if the chosen tier is unaffordable, try a
     cached result, then step down cheaper tiers, then a deterministic
     rule-based fallback marked `degraded`.
  6. Choose a batch mode from task type + urgency.
  7. Resolve the tier to a concrete provider/model from config.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from llm.budget import BudgetManager
from llm.router_config import RouterConfig, load_router_config
from llm.tasks import TaskType, validate_task_type
from llm.tiers import (
    BatchMode,
    Tier,
    Urgency,
    next_cheaper,
    next_stronger,
)


# --- scoring thresholds (tunable, not model-specific) ----------------
_CONFLICT_STRONG = 0.6      # evidence_conflict_score above this ⇒ STRONG-worthy
_CONFLICT_BALANCED = 0.3    # above this ⇒ at least BALANCED
_IMPACT_STRONG = 0.7        # information_impact_score above this ⇒ STRONG-worthy


# Task types that are inherently cheap/mechanical → FAST by default.
_FAST_TASKS = frozenset({
    TaskType.EVIDENCE_CLASSIFICATION,
    TaskType.EVIDENCE_SUMMARY,
    TaskType.JSON_REPAIR,
    TaskType.PERSONA_VARIATION,
    TaskType.MEMORY_SUMMARY,
})

# Task types that default to BALANCED.
_BALANCED_TASKS = frozenset({
    TaskType.ROUTINE_BELIEF_UPDATE,
    TaskType.ARCHETYPE_GENERATION,
    TaskType.PERSONA_ARCHETYPE_GENERATION,
    TaskType.EVIDENCE_CONFLICT_ANALYSIS,
})

# Task types that default to STRONG.
_STRONG_TASKS = frozenset({
    TaskType.MAJOR_BELIEF_UPDATE,
    TaskType.DECISION_AUDIT,
    TaskType.CODE_OR_SYSTEM_ANALYSIS,
})

# Task types eligible for ASYNC_BATCH by nature.
_ASYNC_BATCH_TASKS = frozenset({
    TaskType.PERSONA_VARIATION,
    TaskType.ARCHETYPE_GENERATION,
    TaskType.PERSONA_ARCHETYPE_GENERATION,
    TaskType.MEMORY_SUMMARY,
    TaskType.DECISION_AUDIT,
})

# Task types that micro-batch well when not urgent.
_MICRO_BATCH_TASKS = frozenset({
    TaskType.ROUTINE_BELIEF_UPDATE,
    TaskType.EVIDENCE_SUMMARY,
    TaskType.EVIDENCE_CLASSIFICATION,
})

# High-volume bulk tasks that prefer LOCAL when it is configured.
_LOCAL_PREFERRED_TASKS = frozenset({
    TaskType.EVIDENCE_CLASSIFICATION,
    TaskType.ROUTINE_BELIEF_UPDATE,
})


@dataclass
class TaskRoutingContext:
    """Everything the router needs to decide, supplied by the caller."""

    task_type: Any  # TaskType | str — validated in route()
    estimated_input_tokens: int = 0
    expected_output_tokens: int = 0
    evidence_count: int = 0
    evidence_conflict_score: float = 0.0      # 0..1
    information_impact_score: float = 0.0     # 0..1
    urgency: Urgency = Urgency.NORMAL
    maximum_latency_ms: Optional[int] = None
    structured_output_required: bool = False
    previous_failures: int = 0
    cache_available: bool = False
    batch_eligible: bool = True
    market_id: Optional[int] = None
    evidence_bundle_id: Optional[str] = None
    requested_tier: Optional[Any] = None       # Tier | str | None
    task_metadata: Dict[str, Any] = field(default_factory=dict)

    def coerced_urgency(self) -> Urgency:
        if isinstance(self.urgency, Urgency):
            return self.urgency
        try:
            return Urgency(str(self.urgency).strip().upper())
        except ValueError:
            return Urgency.NORMAL


@dataclass
class RoutingResult:
    """The router's decision. Serializable, non-secret."""

    tier: str
    provider: Optional[str]
    model: Optional[str]
    batch_mode: str
    reason: str
    complexity_score: float
    importance_score: float
    budget_allowed: bool
    cache_eligible: bool
    allow_fallback: bool
    degraded: bool = False
    escalation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "provider": self.provider,
            "model": self.model,
            "batch_mode": self.batch_mode,
            "reason": self.reason,
            "complexity_score": self.complexity_score,
            "importance_score": self.importance_score,
            "budget_allowed": self.budget_allowed,
            "cache_eligible": self.cache_eligible,
            "allow_fallback": self.allow_fallback,
            "degraded": self.degraded,
            "escalation_count": self.escalation_count,
        }


class ModelRouter:
    """Task-aware router over logical tiers.

    Stateless w.r.t. decisions; holds a `RouterConfig` and a shared
    `BudgetManager`. `route()` is safe to call from many threads (the
    budget manager serializes its own state).
    """

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        budget_manager: Optional[BudgetManager] = None,
    ):
        self.config = config or load_router_config()
        self.budget = budget_manager or BudgetManager(self.config.budget)

    # ------------------------------------------------------------------
    # Scoring

    def _complexity_score(self, ctx: TaskRoutingContext, task: TaskType) -> float:
        """0..1 estimate of how hard the task is."""
        score = 0.0
        if task in _STRONG_TASKS:
            score += 0.6
        elif task in _BALANCED_TASKS:
            score += 0.35
        else:
            score += 0.1
        # More evidence and more conflict → harder.
        score += min(0.2, ctx.evidence_count / 100.0)
        score += 0.2 * max(0.0, min(1.0, ctx.evidence_conflict_score))
        return round(max(0.0, min(1.0, score)), 4)

    def _importance_score(self, ctx: TaskRoutingContext, task: TaskType) -> float:
        """0..1 estimate of how much the outcome matters."""
        score = 0.0
        score += 0.5 * max(0.0, min(1.0, ctx.information_impact_score))
        score += 0.3 * max(0.0, min(1.0, ctx.evidence_conflict_score))
        if task in _STRONG_TASKS:
            score += 0.3
        elif task in _BALANCED_TASKS:
            score += 0.15
        if ctx.coerced_urgency() == Urgency.HIGH:
            score += 0.1
        return round(max(0.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------
    # Base tier selection (default routing rules)

    def _coerce_tier(self, value) -> Optional[Tier]:
        if value is None:
            return None
        if isinstance(value, Tier):
            return value
        try:
            return Tier(str(value).strip().upper())
        except ValueError:
            return None

    def _base_tier(self, ctx: TaskRoutingContext, task: TaskType) -> "tuple[Tier, str]":
        """Pick the pre-budget, pre-escalation tier + a reason."""
        # 0. SPECIALIST only on explicit metadata request.
        wants_specialist = bool(
            ctx.task_metadata.get("specialist")
            or ctx.task_metadata.get("requires_specialist")
        )
        req = self._coerce_tier(ctx.requested_tier)
        if req == Tier.SPECIALIST or wants_specialist:
            return Tier.SPECIALIST, "explicit specialist request"

        # 1. LOCAL: high-volume/low-latency bulk tasks when configured.
        # LOCAL is an opt-in preference, NOT an automatic destination:
        # the caller must signal bulk/low-latency intent (requested_tier
        # LOCAL, prefer_local, or a bulk flag) AND the task must be one
        # of the bulk-friendly types. Otherwise the default FAST/BALANCED
        # rules apply — so ordinary classification still routes to FAST.
        wants_local = (
            req == Tier.LOCAL
            or bool(ctx.task_metadata.get("prefer_local"))
            or bool(ctx.task_metadata.get("bulk"))
        )
        local_by_nature = wants_local and task in _LOCAL_PREFERRED_TASKS
        if (req == Tier.LOCAL or local_by_nature) and self.config.is_tier_configured(Tier.LOCAL):
            return Tier.LOCAL, "local provider configured for bulk/low-latency task"
        if wants_local and not self.config.is_tier_configured(Tier.LOCAL):
            # Never silently route to LOCAL when unconfigured — fall through
            # to the cost-axis rules and note it.
            pass

        # 2. Explicit cost-axis request wins over defaults.
        if req in (Tier.FAST, Tier.BALANCED, Tier.STRONG):
            return req, f"explicit {req.value} request"

        # 3. Escalate to STRONG on strong signals regardless of base task.
        if (
            ctx.information_impact_score >= _IMPACT_STRONG
            or ctx.evidence_conflict_score >= _CONFLICT_STRONG
            or ctx.previous_failures >= 2
            or bool(ctx.task_metadata.get("complex"))
            or bool(ctx.task_metadata.get("official") and ctx.information_impact_score >= 0.5)
        ):
            return Tier.STRONG, "high impact / disagreement / repeated failure / complex"

        # 4. Default by task type.
        if task in _STRONG_TASKS:
            return Tier.STRONG, f"{task.value} defaults to STRONG"
        if task in _BALANCED_TASKS:
            # Moderate conflict nudges conflict-analysis to BALANCED (already is).
            return Tier.BALANCED, f"{task.value} defaults to BALANCED"
        if task in _FAST_TASKS:
            # A FAST task with moderate conflict bumps to BALANCED.
            if ctx.evidence_conflict_score >= _CONFLICT_BALANCED and task == TaskType.EVIDENCE_SUMMARY:
                return Tier.BALANCED, "evidence summary with moderate conflict"
            return Tier.FAST, f"{task.value} defaults to FAST"

        return Tier.BALANCED, "no rule matched; defaulting to BALANCED"

    # ------------------------------------------------------------------
    # Batch mode

    def _batch_mode(self, ctx: TaskRoutingContext, task: TaskType, tier: Tier) -> BatchMode:
        urgency = ctx.coerced_urgency()

        # REALTIME triggers: urgent, latency-bound, high impact, or the
        # caller says the task isn't batch-eligible.
        realtime = (
            urgency == Urgency.HIGH
            or not ctx.batch_eligible
            or ctx.information_impact_score >= _IMPACT_STRONG
            or (ctx.maximum_latency_ms is not None and ctx.maximum_latency_ms <= 2000)
            or bool(ctx.task_metadata.get("realtime"))
            or bool(ctx.task_metadata.get("market_closing"))
            or bool(ctx.task_metadata.get("portfolio_risk"))
        )
        if realtime:
            return BatchMode.REALTIME

        if task in _ASYNC_BATCH_TASKS and urgency == Urgency.LOW:
            return BatchMode.ASYNC_BATCH
        if task in _ASYNC_BATCH_TASKS and not ctx.market_id:
            return BatchMode.ASYNC_BATCH
        if task in _MICRO_BATCH_TASKS:
            return BatchMode.MICRO_BATCH
        # Default: micro-batch when eligible, else realtime.
        return BatchMode.MICRO_BATCH if ctx.batch_eligible else BatchMode.REALTIME

    # ------------------------------------------------------------------
    # Budget-aware resolution with fallback chain

    def _affordable(self, ctx: TaskRoutingContext, tier: Tier) -> bool:
        decision = self.budget.can_afford(
            tier=tier,
            input_tokens=ctx.estimated_input_tokens,
            output_tokens=ctx.expected_output_tokens,
            market_id=ctx.market_id,
            bundle_id=ctx.evidence_bundle_id,
        )
        return decision.allowed

    # ------------------------------------------------------------------
    # Public API

    def route(self, ctx: TaskRoutingContext) -> RoutingResult:
        """Return a `RoutingResult` for `ctx`. Raises on non-LLM tasks."""
        task = validate_task_type(ctx.task_type)

        complexity = self._complexity_score(ctx, task)
        importance = self._importance_score(ctx, task)

        tier, reason = self._base_tier(ctx, task)

        # --- Failure escalation (bounded) ---
        escalations = 0
        max_esc = self.config.budget.max_escalations
        if ctx.previous_failures > 0 and tier in (Tier.FAST, Tier.BALANCED):
            while (
                escalations < max_esc
                and ctx.previous_failures > escalations
            ):
                stronger = next_stronger(tier)
                if stronger is None:
                    break
                tier = stronger
                escalations += 1
                reason = f"escalated after {ctx.previous_failures} failure(s)"

        cache_eligible = bool(ctx.cache_available)

        # --- Budget fallback chain ---
        degraded = False
        budget_allowed = True
        allow_fallback = True

        if not self._affordable(ctx, tier):
            budget_allowed = False
            # 1. Reuse an acceptable cached result if available.
            if ctx.cache_available:
                reason = f"budget exhausted for {tier.value}; serving cached result"
                cache_eligible = True
                degraded = False
                # Cache serve doesn't spend budget; keep the tier label
                # but flag budget_allowed False so caller uses cache.
                return self._finalize(
                    ctx, task, tier, reason, complexity, importance,
                    budget_allowed=False, cache_eligible=True,
                    allow_fallback=True, degraded=False, escalations=escalations,
                )
            # 2. Fall back to cheaper tiers.
            candidate = tier
            stepped = False
            while True:
                cheaper = next_cheaper(candidate)
                if cheaper is None:
                    break
                candidate = cheaper
                if self._affordable(ctx, candidate):
                    tier = candidate
                    stepped = True
                    reason = f"budget fallback to cheaper tier {tier.value}"
                    budget_allowed = True
                    break
            if not stepped:
                # 3. Deterministic rule-based fallback; mark degraded.
                # 4. Never retry indefinitely — we return a single degraded
                #    result rather than looping.
                degraded = True
                budget_allowed = False
                allow_fallback = False
                reason = (
                    "budget exhausted at all tiers; deterministic rule-based "
                    "fallback (degraded)"
                )
                return self._finalize(
                    ctx, task, tier, reason, complexity, importance,
                    budget_allowed=False, cache_eligible=cache_eligible,
                    allow_fallback=False, degraded=True, escalations=escalations,
                )

        return self._finalize(
            ctx, task, tier, reason, complexity, importance,
            budget_allowed=budget_allowed, cache_eligible=cache_eligible,
            allow_fallback=allow_fallback, degraded=degraded,
            escalations=escalations,
        )

    def _finalize(
        self, ctx, task, tier, reason, complexity, importance,
        *, budget_allowed, cache_eligible, allow_fallback, degraded, escalations,
    ) -> RoutingResult:
        settings = self.config.tier_settings(tier)
        provider = settings.provider or self.config.default_provider
        model = settings.model

        # If the resolved tier isn't configured, note it in the reason but
        # still return the decision — the caller decides whether to fall
        # back to the legacy client. LOCAL/SPECIALIST are never selected
        # unconfigured by _base_tier, but a budget step could land here.
        if model is None:
            reason = f"{reason} [tier {tier.value} not configured]"

        batch_mode = self._batch_mode(ctx, task, tier)

        return RoutingResult(
            tier=tier.value,
            provider=provider,
            model=model,
            batch_mode=batch_mode.value,
            reason=reason,
            complexity_score=complexity,
            importance_score=importance,
            budget_allowed=budget_allowed,
            cache_eligible=cache_eligible,
            allow_fallback=allow_fallback,
            degraded=degraded,
            escalation_count=escalations,
        )

    # ------------------------------------------------------------------
    # Legacy compatibility

    def route_legacy(self, note: str = "legacy client call") -> RoutingResult:
        """Route an untyped legacy call to BALANCED with a structured warning.

        Preserves the old single-client behavior (compatibility rule #9)
        while flagging the call for later migration to a typed task.
        """
        print(
            f"[ModelRouter][legacy] {note}: untyped LLM call routed to "
            f"BALANCED. Migrate to a TaskType-tagged route(). ",
            file=sys.stderr,
        )
        settings = self.config.tier_settings(Tier.BALANCED)
        return RoutingResult(
            tier=Tier.BALANCED.value,
            provider=settings.provider or self.config.default_provider,
            model=settings.model,
            batch_mode=BatchMode.REALTIME.value,
            reason="legacy untyped call defaulted to BALANCED (migrate to TaskType)",
            complexity_score=0.0,
            importance_score=0.0,
            budget_allowed=True,
            cache_eligible=False,
            allow_fallback=True,
            degraded=False,
            escalation_count=0,
        )


__all__ = [
    "ModelRouter",
    "TaskRoutingContext",
    "RoutingResult",
]
