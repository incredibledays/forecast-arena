"""MemoryService — compact incremental stats + sparse important episodes.

Two data flows keep the action hot path fast:

  1. Stats  — `AgentMemoryStats` has ONE row per Agent, updated
     incrementally on every wake-up / trade / resolution. The action
     policy reads this row and NEVER scans `trades` / `agent_decisions`.

  2. Episodes — sparse `AgentMemoryEpisode` rows created only for
     genuinely important events. Routine HOLDs and small trades don't
     create episodes. A pruner enforces a per-Agent cap while preserving
     type + category diversity so rare failures don't age out.

Bounded adjustments (`compute_effective_persona`) apply memory to the
Agent's decision knobs deterministically and bounded — no single outcome
permanently dominates parameters. Every multiplier and offset is clamped.

The LLM is used ONLY for optional narrative summaries of important
episodes (`summarize_episode_llm`) and NEVER in the incremental update
path. Summaries route ordinary→FAST, unusual→BALANCED, never STRONG by
default.
"""

from __future__ import annotations

import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

from sqlalchemy.orm.attributes import flag_modified

from models import (
    Agent, AgentMemoryEpisode, AgentMemoryStats, db,
    EPISODE_LARGE_GAIN, EPISODE_LARGE_LOSS, EPISODE_HIGH_CONFIDENCE_ERROR,
    EPISODE_SUCCESSFUL_REVERSAL, EPISODE_FAILED_REVERSAL,
    EPISODE_EARLY_SIGNAL, EPISODE_SEVERE_DRAWDOWN,
    EPISODE_TYPES,
    LARGE_PNL_FRACTION, HIGH_CONFIDENCE_ERROR_THRESHOLD,
    SEVERE_DRAWDOWN_FRACTION, DEFAULT_EPISODE_CAP,
)
from services._schema_cache import ensure_created as _ensure_schema_cached


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---- effective persona (memory-adjusted, bounded) -------------------

@dataclass
class MemoryAdjustments:
    """The bounded, diagnostic breakdown of memory's effect on parameters.

    All fields have hard clamps applied at construction so no single
    outcome can permanently dominate.
    """

    # Belief-side adjustments
    probability_shrink_toward_half: float = 0.0   # ∈ [0, 0.4]
    confidence_multiplier: float = 1.0            # ∈ [0.4, 1.0]

    # Sizing / Kelly adjustments
    kelly_multiplier: float = 1.0                 # ∈ [0.2, 1.0]
    size_multiplier: float = 1.0                  # ∈ [0.3, 1.0]
    calibration_multiplier: float = 1.0           # ∈ [0.5, 1.5]

    # Threshold adjustments
    entry_threshold_add: float = 0.0              # ∈ [0, 0.10]
    reversal_threshold_add: float = 0.0           # ∈ [0, 0.05]

    # Suggested natural wake-up rate multiplier (for the scheduler).
    wakeup_rate_multiplier: float = 1.0           # ∈ [0.3, 1.5]

    # Diagnostics (why the adjustments look this way).
    reasons: Dict[str, float] = field(default_factory=dict)


@dataclass
class EffectivePersona:
    """The memory-adjusted persona ActionPolicy consumes.

    Base fields come from the Agent row (unmodified — deterministic
    replay). `effective_*` fields have the memory adjustments applied and
    clamped. Adapters and the policy read the effective fields; nothing
    downstream should re-apply the adjustments.
    """

    # Identity + base (unmodified) knobs
    agent_id: int
    strategy_type: str
    kelly_fraction: float
    max_event_exposure: float
    max_total_exposure: float
    max_drawdown_tolerance: float
    entry_edge_threshold: float
    exit_edge_threshold: float
    reversal_edge_threshold: float
    minimum_trade_notional: float
    category_expertise: float

    # Memory-adjusted (final)
    effective_entry_threshold: float
    effective_exit_threshold: float
    effective_reversal_threshold: float
    effective_kelly_fraction: float          # kelly × kelly_multiplier
    effective_size_multiplier: float         # 0.3..1.0
    confidence_multiplier: float             # 0.4..1.0
    probability_shrink_toward_half: float    # 0..0.4
    calibration_multiplier: float            # 0.5..1.5

    memory_adjustments: MemoryAdjustments


class MemoryService:
    """Incremental compact memory + bounded, diagnostic adjustments."""

    # ==================================================================
    # Bootstrap
    # ==================================================================

    @staticmethod
    def ensure_schema() -> None:
        _ensure_schema_cached()

    @classmethod
    def ensure_stats(cls, agent_id: int, initial_cash: float = 0.0) -> AgentMemoryStats:
        row = db.session.get(AgentMemoryStats, agent_id)
        if row is None:
            row = AgentMemoryStats(
                agent_id=agent_id,
                portfolio_value=initial_cash,
                high_water_mark=max(0.0, initial_cash),
                category_stats={}, strategy_stats={}, source_reliability_stats={},
            )
            db.session.add(row)
            db.session.flush()
        return row

    # ==================================================================
    # Incremental wake-up / trade / resolution updates
    # ==================================================================

    @classmethod
    def increment_after_wakeup(
        cls, agent_id: int, traded: bool, initial_cash: float = 0.0,
    ) -> AgentMemoryStats:
        row = cls.ensure_stats(agent_id, initial_cash=initial_cash)
        row.wake_up_count = (row.wake_up_count or 0) + 1
        if not traded:
            row.wake_ups_without_trade = (row.wake_ups_without_trade or 0) + 1
        row.update_version = (row.update_version or 0) + 1
        db.session.flush()
        return row

    @classmethod
    def increment_after_trade(
        cls, agent_id: int, *, notional: float, realized_pnl_delta: float,
        unrealized_pnl_delta: float, category: Optional[str] = None,
        strategy_type: Optional[str] = None, initial_cash: float = 0.0,
        create_episode_hint: Optional[str] = None,
    ) -> AgentMemoryStats:
        """Fold a completed trade into the compact stats (no history scan).

        Callers (an executor / test) hand us the P&L deltas — we DO NOT
        walk the Trade table. This is the invariant that keeps the hot
        path O(1). Optionally records an episode when the trade is
        genuinely important.
        """
        row = cls.ensure_stats(agent_id, initial_cash=initial_cash)
        row.trade_count = (row.trade_count or 0) + 1
        row.realized_pnl = (row.realized_pnl or 0.0) + float(realized_pnl_delta)
        row.unrealized_pnl = (row.unrealized_pnl or 0.0) + float(unrealized_pnl_delta)
        pnl_this = float(realized_pnl_delta) + float(unrealized_pnl_delta)
        if pnl_this > 0:
            row.profitable_trade_count = (row.profitable_trade_count or 0) + 1
            row.win_streak = (row.win_streak or 0) + 1
            row.loss_streak = 0
        elif pnl_this < 0:
            row.loss_streak = (row.loss_streak or 0) + 1
            row.win_streak = 0

        # Category / strategy compact stats (JSON dicts).
        if category:
            row.category_stats = cls._bump_slice(row.category_stats, category, pnl_this)
            flag_modified(row, "category_stats")
        if strategy_type:
            row.strategy_stats = cls._bump_slice(row.strategy_stats, strategy_type, pnl_this)
            flag_modified(row, "strategy_stats")

        # Portfolio update from PnL. In this phase the caller passes the
        # deltas; a portfolio snapshot could refine this later.
        prior_pv = float(row.portfolio_value or 0.0)
        row.portfolio_value = prior_pv + pnl_this
        if row.portfolio_value > row.high_water_mark:
            row.high_water_mark = row.portfolio_value
        # current_drawdown clamped to [0, 1]
        hwm = max(1e-9, row.high_water_mark)
        row.current_drawdown = max(0.0, (row.high_water_mark - row.portfolio_value) / hwm)
        row.max_drawdown = max(row.max_drawdown or 0.0, row.current_drawdown)

        row.update_version = (row.update_version or 0) + 1
        db.session.flush()

        # Sparse episode creation (importance-gated, never for routine).
        cls.maybe_create_episode_from_trade(
            row, initial_cash=initial_cash, pnl_this=pnl_this, notional=notional,
            category=category, kind_hint=create_episode_hint,
        )
        return row

    @classmethod
    def _bump_slice(cls, slice_json, key: str, pnl: float) -> Dict[str, Any]:
        """Fold one trade into a `{key: {count,pnl_sum,...}}` compact dict.

        Deep-copies the incoming dict so mutations never alias the ORM's
        stored value — the caller reassigns + flag_modified to force
        SQLAlchemy's JSON change tracking.
        """
        d = deepcopy(slice_json) if slice_json else {}
        cur = dict(d.get(key) or {"count": 0, "pnl_sum": 0.0, "win_count": 0})
        cur["count"] = int(cur.get("count", 0)) + 1
        cur["pnl_sum"] = float(cur.get("pnl_sum", 0.0)) + float(pnl)
        if pnl > 0:
            cur["win_count"] = int(cur.get("win_count", 0)) + 1
        d[key] = cur
        return d

    @classmethod
    def record_resolution(
        cls, agent_id: int, *, predicted_probability_yes: float,
        actual_yes: bool, confidence: float,
        category: Optional[str] = None, initial_cash: float = 0.0,
        event_id: Optional[int] = None,
    ) -> AgentMemoryStats:
        """Fold ONE resolved prediction into calibration stats.

        Computes Brier + log-loss incrementally — no history scan.
        Records HIGH_CONFIDENCE_ERROR episodes when confidence is high AND
        the prediction was clearly wrong.
        """
        row = cls.ensure_stats(agent_id, initial_cash=initial_cash)
        p = _clamp(float(predicted_probability_yes), 1e-6, 1.0 - 1e-6)
        y = 1.0 if actual_yes else 0.0

        row.resolved_prediction_count = (row.resolved_prediction_count or 0) + 1
        row.brier_running_sum = (row.brier_running_sum or 0.0) + (p - y) ** 2
        row.brier_average = row.brier_running_sum / max(1, row.resolved_prediction_count)
        # Log-loss: -[y·log(p) + (1-y)·log(1-p)]
        ll = -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
        row.log_loss_running_sum = (row.log_loss_running_sum or 0.0) + ll
        row.log_loss_average = row.log_loss_running_sum / max(1, row.resolved_prediction_count)

        predicted_side_correct = (p >= 0.5) == bool(actual_yes)
        n = row.resolved_prediction_count
        # Running average of (accuracy, confidence).
        row.empirical_accuracy = (
            (row.empirical_accuracy or 0.0) * (n - 1) + (1.0 if predicted_side_correct else 0.0)
        ) / n
        c = _clamp(float(confidence), 0.0, 1.0)
        row.average_confidence = (
            (row.average_confidence or 0.0) * (n - 1) + c
        ) / n
        row.overconfidence_score = _clamp(
            row.average_confidence - row.empirical_accuracy, -1.0, 1.0
        )

        # Category-brier fold-in.
        if category:
            d = deepcopy(row.category_stats) if row.category_stats else {}
            cur = dict(d.get(category) or {"count": 0, "brier_sum": 0.0})
            cur["brier_count"] = int(cur.get("brier_count", 0)) + 1
            cur["brier_sum"] = float(cur.get("brier_sum", 0.0)) + (p - y) ** 2
            d[category] = cur
            row.category_stats = d
            flag_modified(row, "category_stats")

        # HIGH_CONFIDENCE_ERROR: high conf on a badly-wrong prediction.
        brier_this = (p - y) ** 2
        if c >= 0.6 and brier_this >= HIGH_CONFIDENCE_ERROR_THRESHOLD:
            cls.create_episode(
                agent_id=agent_id,
                episode_type=EPISODE_HIGH_CONFIDENCE_ERROR,
                importance=_clamp(c * brier_this, 0.0, 1.0),
                magnitude=brier_this,
                category=category, event_id=event_id,
                concise_summary=(
                    f"high-confidence error: predicted p={p:.2f} conf={c:.2f}, "
                    f"actual={'YES' if actual_yes else 'NO'}, Brier={brier_this:.2f}"
                ),
            )

        row.update_version = (row.update_version or 0) + 1
        db.session.flush()
        return row

    # ==================================================================
    # Episode creation + retention
    # ==================================================================

    @classmethod
    def maybe_create_episode_from_trade(
        cls, row: AgentMemoryStats, *, initial_cash: float, pnl_this: float,
        notional: float, category: Optional[str], kind_hint: Optional[str],
    ) -> Optional[AgentMemoryEpisode]:
        """Sparse: only truly notable trades produce an episode.

        LARGE_GAIN / LARGE_LOSS threshold = 5% of initial cash (or of the
        Agent's current portfolio, whichever is smaller). SEVERE_DRAWDOWN
        fires when current drawdown crosses 80% of the Agent's tolerance.
        SUCCESSFUL_REVERSAL / FAILED_REVERSAL / EARLY_SIGNAL are opt-in
        via `kind_hint` from the caller — they need context we don't have.
        """
        baseline = max(1.0, min(initial_cash or 0.0, row.portfolio_value or 0.0))
        pnl_fraction = abs(pnl_this) / baseline if baseline > 0 else 0.0

        # SEVERE_DRAWDOWN — cross the fraction of tolerance for the first time.
        # We store tolerance externally; a simple heuristic: cross 30% dd.
        if row.current_drawdown >= 0.3 and row.max_drawdown == row.current_drawdown:
            cls.create_episode(
                agent_id=row.agent_id,
                episode_type=EPISODE_SEVERE_DRAWDOWN,
                importance=_clamp(row.current_drawdown, 0.0, 1.0),
                magnitude=-row.current_drawdown,
                category=category,
                concise_summary=f"drawdown crossed {row.current_drawdown:.2%}",
            )

        # LARGE_GAIN / LARGE_LOSS gate.
        if pnl_fraction >= LARGE_PNL_FRACTION:
            kind = EPISODE_LARGE_GAIN if pnl_this > 0 else EPISODE_LARGE_LOSS
            return cls.create_episode(
                agent_id=row.agent_id, episode_type=kind,
                importance=_clamp(pnl_fraction, 0.0, 1.0),
                magnitude=pnl_this, category=category,
                concise_summary=f"{kind}: pnl={pnl_this:+.2f} ({pnl_fraction:.1%} of base)",
            )

        # Explicit reversal / early-signal hints from the caller.
        if kind_hint in (
            EPISODE_SUCCESSFUL_REVERSAL, EPISODE_FAILED_REVERSAL, EPISODE_EARLY_SIGNAL,
        ) and pnl_fraction >= LARGE_PNL_FRACTION / 2:
            return cls.create_episode(
                agent_id=row.agent_id, episode_type=kind_hint,
                importance=_clamp(max(pnl_fraction, 0.4), 0.0, 1.0),
                magnitude=pnl_this, category=category,
                concise_summary=f"{kind_hint}: pnl={pnl_this:+.2f}",
            )

        return None  # routine — NO episode

    @classmethod
    def create_episode(
        cls, *, agent_id: int, episode_type: str, importance: float,
        magnitude: float, category: Optional[str] = None,
        event_id: Optional[int] = None, market_id: Optional[int] = None,
        concise_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cap: int = DEFAULT_EPISODE_CAP,
    ) -> AgentMemoryEpisode:
        ep = AgentMemoryEpisode(
            agent_id=agent_id, event_id=event_id, market_id=market_id,
            category=category, episode_type=episode_type,
            importance=_clamp(float(importance), 0.0, 1.0),
            magnitude=float(magnitude),
            concise_summary=(concise_summary or "")[:400] or None,
            episode_metadata=metadata or None,
        )
        db.session.add(ep)
        db.session.flush()
        cls.prune_episodes(agent_id, cap=cap)
        return ep

    @classmethod
    def prune_episodes(cls, agent_id: int, cap: int = DEFAULT_EPISODE_CAP) -> int:
        """Cap per-Agent episodes while preserving type + category diversity.

        Rule: never simply keep the newest N. First reserve one slot per
        (episode_type) and one per (category) so rare failures survive,
        then fill the rest by (importance, recency). Returns delete count.
        """
        rows = (
            AgentMemoryEpisode.query.filter_by(agent_id=agent_id)
            .order_by(AgentMemoryEpisode.created_at.desc())
            .all()
        )
        if len(rows) <= cap:
            return 0

        # Score = importance + recency-decay (newest gets +0.1).
        n_total = len(rows)
        def score(i: int, ep):
            recency = 0.1 * (1.0 - i / max(1, n_total - 1))
            return float(ep.importance or 0.0) + recency

        # 1. reserve one per type (highest-importance representative).
        keep: Dict[int, AgentMemoryEpisode] = {}
        best_by_type: Dict[str, AgentMemoryEpisode] = {}
        best_by_cat: Dict[str, AgentMemoryEpisode] = {}
        for i, ep in enumerate(rows):
            t = ep.episode_type
            if t not in best_by_type or (ep.importance or 0) > (best_by_type[t].importance or 0):
                best_by_type[t] = ep
            c = ep.category or "_none_"
            if c not in best_by_cat or (ep.importance or 0) > (best_by_cat[c].importance or 0):
                best_by_cat[c] = ep
        for ep in list(best_by_type.values()) + list(best_by_cat.values()):
            keep[ep.id] = ep
            if len(keep) >= cap:
                break

        # 2. fill remaining slots by combined score.
        remaining = sorted(
            ((i, ep) for i, ep in enumerate(rows) if ep.id not in keep),
            key=lambda ie: score(ie[0], ie[1]),
            reverse=True,
        )
        for _, ep in remaining:
            if len(keep) >= cap:
                break
            keep[ep.id] = ep

        to_delete = [ep.id for ep in rows if ep.id not in keep]
        if to_delete:
            AgentMemoryEpisode.query.filter(AgentMemoryEpisode.id.in_(to_delete)).delete(
                synchronize_session=False
            )
            db.session.flush()
        return len(to_delete)

    # ==================================================================
    # Bounded adjustments → EffectivePersona
    # ==================================================================

    @classmethod
    def compute_effective_persona(
        cls, agent_id: int, category: Optional[str] = None,
    ) -> EffectivePersona:
        """Build the memory-adjusted persona ActionPolicy consumes.

        Every adjustment is bounded so no single outcome dominates. The
        base persona fields are copied unmodified for deterministic
        replay; only the `effective_*` fields carry the applied deltas.
        """
        cls.ensure_schema()
        agent = db.session.get(Agent, agent_id)
        if agent is None:
            raise ValueError(f"agent {agent_id} not found")
        row = cls.ensure_stats(agent_id, initial_cash=float(agent.initial_cash or 0.0))

        adj = MemoryAdjustments()
        reasons: Dict[str, float] = {}

        # 1. Overconfidence — shrink toward 0.5, drop confidence.
        overconf = max(0.0, float(row.overconfidence_score or 0.0))
        if overconf > 0.05:
            shrink = _clamp(overconf * 0.6, 0.0, 0.4)
            adj.probability_shrink_toward_half = shrink
            adj.confidence_multiplier = _clamp(1.0 - overconf * 0.5, 0.4, 1.0)
            adj.kelly_multiplier = _clamp(
                adj.kelly_multiplier * (1.0 - overconf * 0.4), 0.2, 1.0
            )
            reasons["overconfidence"] = overconf

        # 2. Drawdown — reduce size + Kelly.
        dd = _clamp(float(row.current_drawdown or 0.0), 0.0, 1.0)
        if dd > 0.05:
            adj.size_multiplier = _clamp(1.0 - dd * 2.0, 0.3, 1.0)
            adj.kelly_multiplier = _clamp(
                adj.kelly_multiplier * (1.0 - dd * 1.5), 0.2, 1.0
            )
            reasons["drawdown"] = dd
            if dd >= 0.5:
                # Suggest scheduler prioritize this Agent for risk wakes.
                adj.wakeup_rate_multiplier = _clamp(1.0 + dd, 0.3, 1.5)

        # 3. Loss streak — raise entry + reversal thresholds.
        ls = int(row.loss_streak or 0)
        if ls > 0:
            adj.entry_threshold_add += _clamp(ls * 0.005, 0.0, 0.05)
            adj.reversal_threshold_add += _clamp(ls * 0.003, 0.0, 0.03)
            reasons["loss_streak"] = ls

        # 4. Poor category performance — raise entry, drop category conf.
        cat_stats = row.category_stats or {}
        cat_brier = None
        if category and category in cat_stats:
            slice_ = cat_stats[category]
            n = int(slice_.get("brier_count") or 0)
            if n >= 3:
                cat_brier = float(slice_["brier_sum"]) / n
                if cat_brier > 0.25:
                    adj.entry_threshold_add += _clamp((cat_brier - 0.25) * 0.2, 0.0, 0.05)
                    adj.confidence_multiplier = _clamp(
                        adj.confidence_multiplier * (1.0 - (cat_brier - 0.25)),
                        0.4, 1.0,
                    )
                    reasons["poor_category"] = cat_brier

        # 5. Calibration — reward accurate agents with a bounded boost.
        if row.resolved_prediction_count >= 5:
            calib = (row.empirical_accuracy or 0.0) - (row.average_confidence or 0.0)
            adj.calibration_multiplier = _clamp(1.0 + calib * 0.5, 0.5, 1.5)
            reasons["calibration"] = calib

        # 6. Repeated wake-ups without action → lower wake rate suggestion.
        w = int(row.wake_up_count or 0)
        idle = int(row.wake_ups_without_trade or 0)
        if w >= 10 and (idle / max(1, w)) > 0.8:
            adj.wakeup_rate_multiplier = _clamp(
                adj.wakeup_rate_multiplier * 0.6, 0.3, 1.5
            )
            reasons["idle_wakeups"] = idle / max(1, w)

        adj.reasons = reasons

        # Re-clamp all bounded fields (defensive).
        adj.probability_shrink_toward_half = _clamp(adj.probability_shrink_toward_half, 0.0, 0.4)
        adj.confidence_multiplier = _clamp(adj.confidence_multiplier, 0.4, 1.0)
        adj.kelly_multiplier = _clamp(adj.kelly_multiplier, 0.2, 1.0)
        adj.size_multiplier = _clamp(adj.size_multiplier, 0.3, 1.0)
        adj.calibration_multiplier = _clamp(adj.calibration_multiplier, 0.5, 1.5)
        adj.entry_threshold_add = _clamp(adj.entry_threshold_add, 0.0, 0.10)
        adj.reversal_threshold_add = _clamp(adj.reversal_threshold_add, 0.0, 0.05)
        adj.wakeup_rate_multiplier = _clamp(adj.wakeup_rate_multiplier, 0.3, 1.5)

        # Category expertise (from persona overrides).
        cat_expertise = 0.5
        if category and isinstance(agent.persona_overrides_json, dict):
            exp = (agent.persona_overrides_json or {}).get("expertise") or {}
            try:
                cat_expertise = float(exp.get(category, 0.5))
            except (TypeError, ValueError):
                cat_expertise = 0.5

        # Assemble the effective persona.
        entry = float(agent.entry_edge_threshold or 0.08) + adj.entry_threshold_add
        exit_ = float(agent.exit_edge_threshold or 0.03)
        reversal = float(agent.reversal_edge_threshold or 0.13) + adj.reversal_threshold_add
        eff_kelly = _clamp(
            float(agent.kelly_fraction or 0.4) * adj.kelly_multiplier, 0.02, 1.0
        )
        persona = EffectivePersona(
            agent_id=agent_id,
            strategy_type=str(agent.strategy_type or "adaptive"),
            kelly_fraction=float(agent.kelly_fraction or 0.4),
            max_event_exposure=float(agent.max_event_exposure or 0.15),
            max_total_exposure=float(agent.max_total_exposure or 0.6),
            max_drawdown_tolerance=float(agent.max_drawdown_tolerance or 0.35),
            entry_edge_threshold=float(agent.entry_edge_threshold or 0.08),
            exit_edge_threshold=exit_,
            reversal_edge_threshold=float(agent.reversal_edge_threshold or 0.13),
            minimum_trade_notional=float(agent.minimum_trade_notional or 20.0),
            category_expertise=cat_expertise,
            effective_entry_threshold=entry,
            effective_exit_threshold=exit_,
            effective_reversal_threshold=reversal,
            effective_kelly_fraction=eff_kelly,
            effective_size_multiplier=adj.size_multiplier,
            confidence_multiplier=adj.confidence_multiplier,
            probability_shrink_toward_half=adj.probability_shrink_toward_half,
            calibration_multiplier=adj.calibration_multiplier,
            memory_adjustments=adj,
        )
        return persona

    # ==================================================================
    # Optional LLM episode summarization (never in the hot path)
    # ==================================================================

    @classmethod
    def summarize_episode_llm(
        cls, episode_id: int, llm_client=None, router=None,
        importance_threshold: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        """Summarize an episode via a routed LLM call. Optional.

        Uses ModelRouter: ordinary important episode → FAST, unusual
        multifactor → BALANCED. STRONG is NOT the default. Skips when
        importance is below the threshold or the router refuses the
        budget. Never called in the incremental update path.
        """
        ep = db.session.get(AgentMemoryEpisode, episode_id)
        if ep is None:
            return None
        if (ep.importance or 0.0) < importance_threshold:
            return {"skipped": True, "reason": "importance below threshold"}
        if llm_client is None or router is None:
            return {"skipped": True, "reason": "no llm/router configured"}

        from llm import TaskRoutingContext, TaskType
        # Ordinary important episode → FAST. If it's a compound (has metadata
        # implying multiple factors), let router see mild conflict so it may
        # pick BALANCED. STRONG stays off by construction (impact/conflict
        # signals kept low).
        multifactor = bool((ep.episode_metadata or {}).get("multifactor"))
        ctx = TaskRoutingContext(
            task_type=TaskType.MEMORY_SUMMARY,
            estimated_input_tokens=200,
            expected_output_tokens=120,
            structured_output_required=True,
            evidence_conflict_score=0.35 if multifactor else 0.0,
            information_impact_score=0.0,          # keep STRONG off
        )
        decision = router.route(ctx)
        if not decision.budget_allowed and not decision.cache_eligible:
            return {"skipped": True, "reason": "budget exhausted"}
        try:
            payload = llm_client.chat_json(
                [
                    {"role": "system", "content": (
                        "Summarize this trading-agent episode in one sentence. "
                        "STRICT JSON: {\"summary\": string}. No CoT."
                    )},
                    {"role": "user", "content": (
                        f"type={ep.episode_type}, importance={ep.importance:.2f}, "
                        f"magnitude={ep.magnitude}, category={ep.category}, "
                        f"summary_hint={ep.concise_summary or ''}"
                    )},
                ], max_tokens=180,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[memory] episode summarize failed: {exc}", file=sys.stderr)
            return {"skipped": True, "reason": f"llm failure: {exc}"}
        summary = (isinstance(payload, dict) and str(payload.get("summary") or "").strip()) or ""
        if summary:
            ep.llm_summary = summary[:600]
            ep.llm_summary_tier = decision.tier
            ep.llm_summary_model = decision.model
            db.session.flush()
        return {
            "summary": summary, "tier": decision.tier, "model": decision.model,
            "batch_mode": decision.batch_mode,
        }


__all__ = [
    "MemoryService",
    "MemoryAdjustments",
    "EffectivePersona",
]
