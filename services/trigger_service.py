"""TriggerService — sparse event-driven wake-ups (scheduling + selection).

This phase implements ONLY scheduling and candidate selection. No LLM
belief updates, no trading. Given an event (information / price /
portfolio / lifecycle), it selects a bounded set of relevant Agents from
the sparse candidate maps, scores + ranks them, applies budgets and
deterministic timing jitter, and writes deduplicated `WakeUpTask` rows.

Everything is deterministic in `(simulation_seed, agent_id, event_id,
sequence)` so a run is replayable. Virtual time comes from the existing
`SchedulerService` clock — nothing sleeps.

Key guarantees (tested):
  * A normal event never scans the full Agent table (uses CandidateService).
  * Holders / experts rank above generic watchers.
  * Per-event wake-up budgets + tiers are enforced.
  * Jitter desynchronizes scheduled times.
  * Duplicate triggers for the same (agent, event, bucket) MERGE atomically.
  * Price triggers respect cooldown, bounded cascade depth, per-market cap.
  * RESOLUTION and critical PORTFOLIO_RISK survive load shedding.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from models import (
    TIER_DELAYED,
    TIER_NORMAL,
    TIER_URGENT,
    TRIGGER_PRIORITY,
    TriggerCooldown,
    TriggerType,
    WakeUpTask,
    STATUS_PENDING,
    STATUS_SHED,
    db,
    make_dedup_key,
    time_bucket,
)
from services._schema_cache import ensure_created as _ensure_schema_cached
from services.candidate_service import CandidateService
from services.scheduler_service import SchedulerService


# ---- jitter windows (seconds) by impact class -----------------------
_JITTER_MAJOR = (0.0, 300.0)          # major official: 0–5 min
_JITTER_NORMAL = (300.0, 7200.0)      # normal: 5–120 min
_JITTER_LOW = (7200.0, 86400.0)       # low-impact: 2–24 h

# Default price-trigger controls.
_DEFAULT_COOLDOWN_S = 3600.0          # 1 virtual hour between price wakes
_DEFAULT_MIN_MOVE = 0.03              # 3 percentage-point minimum move
_DEFAULT_MAX_CASCADE_DEPTH = 3
_DEFAULT_MAX_PER_MARKET = 200         # per market, this trigger call


@dataclass
class EventBudget:
    """Per-information-event wake-up budget."""

    max_candidates: int = 5000
    max_urgent: int = 50
    max_normal: int = 500
    max_delayed: int = 2000
    total_budget: int = 2000

    def tier_cap(self, tier: str) -> int:
        return {
            TIER_URGENT: self.max_urgent,
            TIER_NORMAL: self.max_normal,
            TIER_DELAYED: self.max_delayed,
        }.get(tier, self.max_normal)


@dataclass
class PriceTriggerConfig:
    """Controls for a price-driven trigger."""

    cooldown_s: float = _DEFAULT_COOLDOWN_S
    min_move: float = _DEFAULT_MIN_MOVE
    max_cascade_depth: int = _DEFAULT_MAX_CASCADE_DEPTH
    max_per_market: int = _DEFAULT_MAX_PER_MARKET


def _u01(*parts: Any) -> float:
    """Deterministic uniform in [0,1) from a blake2b of the parts."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    bits = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") >> 11
    return bits / float(1 << 53)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_wake_score(
    relevance: float, information_impact: float,
    position_relevance: float, event_sensitivity: float,
) -> float:
    """The information-event wake score (clamped to [0,1])."""
    score = (
        0.35 * _clamp01(relevance)
        + 0.30 * _clamp01(information_impact)
        + 0.20 * _clamp01(position_relevance)
        + 0.15 * _clamp01(event_sensitivity)
    )
    return _clamp01(score)


class TriggerService:
    """Create sparse, deduplicated, jittered wake-up tasks from events."""

    @staticmethod
    def ensure_schema() -> None:
        _ensure_schema_cached()

    # ==================================================================
    # Deterministic timing jitter
    # ==================================================================

    @staticmethod
    def _jitter_window(impact_class: str):
        if impact_class == "major":
            return _JITTER_MAJOR
        if impact_class == "low":
            return _JITTER_LOW
        return _JITTER_NORMAL

    @classmethod
    def _jittered_time(cls, base_time: float, impact_class: str,
                       sim_seed: int, agent_id: int, event_id, seq: int) -> float:
        lo, hi = cls._jitter_window(impact_class)
        u = _u01(sim_seed, agent_id, event_id, seq, "jitter")
        return base_time + lo + u * (hi - lo)

    # ==================================================================
    # Agent-level parameter hydration (bounded id IN, no full scan)
    # ==================================================================

    @staticmethod
    def _hydrate_agents(agent_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch the few Agent columns scoring needs, by id IN batches.

        Bounded index lookups on the primary key — NOT a full-table scan.
        Returns {agent_id: {event_sensitivity, price_sensitivity,
        portfolio_sensitivity, max_event_exposure, max_total_exposure,
        max_drawdown_tolerance}}.
        """
        out: Dict[int, Dict[str, Any]] = {}
        if not agent_ids:
            return out
        from models import Agent
        CHUNK = 900  # keep the IN list under SQLite's variable limit
        for i in range(0, len(agent_ids), CHUNK):
            chunk = agent_ids[i:i + CHUNK]
            rows = (
                db.session.query(
                    Agent.id, Agent.event_sensitivity, Agent.price_sensitivity,
                    Agent.portfolio_sensitivity, Agent.max_event_exposure,
                    Agent.max_total_exposure, Agent.max_drawdown_tolerance,
                )
                .filter(Agent.id.in_(chunk))
                .all()
            )
            for r in rows:
                out[r[0]] = {
                    "event_sensitivity": r[1] or 0.0,
                    "price_sensitivity": r[2] or 0.0,
                    "portfolio_sensitivity": r[3] or 0.0,
                    "max_event_exposure": r[4] or 0.0,
                    "max_total_exposure": r[5] or 0.0,
                    "max_drawdown_tolerance": r[6] or 0.0,
                }
        db.session.expunge_all()
        return out

    # ==================================================================
    # Information event
    # ==================================================================

    @classmethod
    def information_event(
        cls,
        event_id: int,
        information_impact: float,
        relevance: float = 0.6,
        official: bool = False,
        budget: Optional[EventBudget] = None,
        sim_seed: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Select + schedule wake-ups for an information event.

        Sparse candidate union → deterministic per-Agent wake_score →
        tiering (urgent/normal/delayed) → budget enforcement → jitter →
        deduplicated task upsert. Returns metrics.
        """
        cls.ensure_schema()
        budget = budget or EventBudget()
        if now is None:
            now = SchedulerService.now()
        if sim_seed is None:
            sim_seed = SchedulerService.get_clock().simulation_seed
        impact_class = "major" if official and information_impact >= 0.6 else (
            "low" if information_impact < 0.3 else "normal"
        )

        category = CandidateService.event_category(event_id)
        candidates = CandidateService.collect_candidate_ids(
            event_id, category=category, max_candidates=budget.max_candidates,
        )
        if not candidates:
            return {"event_id": event_id, "candidates": 0, "scheduled": 0,
                    "by_tier": {}, "wake_reason": TriggerType.INFORMATION.value}

        params = cls._hydrate_agents(list(candidates.keys()))

        # Score every candidate deterministically.
        scored: List[Dict[str, Any]] = []
        for aid, meta in candidates.items():
            p = params.get(aid, {})
            # position_relevance: holders get the interest weight (1.0),
            # others a mild base; event_sensitivity from the Agent.
            position_relevance = meta["weight"] if meta["role"] == "holder" else 0.2
            ev_sens = p.get("event_sensitivity", 0.0)
            # Deterministic per-agent relevance perturbation so identical
            # inputs still spread (keeps ranking stable + reproducible).
            rel = _clamp01(relevance + (_u01(sim_seed, aid, event_id, "rel") - 0.5) * 0.1)
            score = compute_wake_score(rel, information_impact, position_relevance, ev_sens)
            scored.append({
                "agent_id": aid, "role": meta["role"], "wake_score": score,
                "relevance": rel, "position_relevance": position_relevance,
                "event_sensitivity": ev_sens, "source_rank": meta["source_rank"],
            })

        # Rank: priority is fixed for INFORMATION, so rank by
        # (position_relevance, source_rank/expertise, wake_score).
        scored.sort(
            key=lambda s: (s["position_relevance"], s["source_rank"], s["wake_score"]),
            reverse=True,
        )

        # Assign tiers by wake_score, then enforce per-tier + total budgets.
        by_tier = {TIER_URGENT: 0, TIER_NORMAL: 0, TIER_DELAYED: 0}
        scheduled = 0
        priority = TRIGGER_PRIORITY[TriggerType.INFORMATION.value]

        for s in scored:
            if scheduled >= budget.total_budget:
                break
            tier = (
                TIER_URGENT if s["wake_score"] >= 0.7
                else TIER_NORMAL if s["wake_score"] >= 0.4
                else TIER_DELAYED
            )
            if by_tier[tier] >= budget.tier_cap(tier):
                # Tier full — try to demote to a lower tier with room.
                demoted = False
                for lower in (TIER_NORMAL, TIER_DELAYED):
                    if lower != tier and by_tier[lower] < budget.tier_cap(lower):
                        tier = lower
                        demoted = True
                        break
                if not demoted:
                    continue

            seq = by_tier[tier]
            sched_at = cls._jittered_time(now, impact_class, sim_seed, s["agent_id"], event_id, seq)
            cls._upsert_task(
                agent_id=s["agent_id"], event_id=event_id, market_id=None,
                trigger_type=TriggerType.INFORMATION, priority=priority, tier=tier,
                scheduled_at=sched_at, now=now,
                relevance=s["relevance"], information_impact=information_impact,
                position_relevance=s["position_relevance"], expertise=0.0,
                portfolio_risk=0.0, wake_score=s["wake_score"],
            )
            by_tier[tier] += 1
            scheduled += 1

        db.session.commit()
        db.session.expunge_all()
        return {
            "event_id": event_id, "candidates": len(candidates),
            "scheduled": scheduled, "by_tier": by_tier, "impact_class": impact_class,
            "wake_reason": TriggerType.INFORMATION.value,
        }

    # ==================================================================
    # Price event
    # ==================================================================

    @classmethod
    def price_event(
        cls,
        event_id: int,
        market_id: int,
        kind: str,                       # abs_move|rel_move|extreme|volume_spike|fair_value|stop_loss|take_profit
        movement: float = 0.0,
        config: Optional[PriceTriggerConfig] = None,
        cascade_id: Optional[str] = None,
        cascade_depth: int = 0,
        trigger_trade_id: Optional[int] = None,
        sim_seed: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Schedule PRICE wake-ups for holders/watchers of a moved market.

        Enforces: minimum movement, per-(agent,market) cooldown, bounded
        cascade depth, and a per-market cap on wake-ups this call. Records
        cascade id/depth and the triggering trade id.
        """
        cls.ensure_schema()
        config = config or PriceTriggerConfig()
        if now is None:
            now = SchedulerService.now()
        if sim_seed is None:
            sim_seed = SchedulerService.get_clock().simulation_seed

        result = {
            "event_id": event_id, "market_id": market_id, "kind": kind,
            "scheduled": 0, "skipped_cooldown": 0, "skipped_min_move": 0,
            "cascade_id": cascade_id, "cascade_depth": cascade_depth,
            "capped": False, "wake_reason": TriggerType.PRICE.value,
        }

        # Cascade depth bound — refuse to schedule beyond the max depth.
        if cascade_depth > config.max_cascade_depth:
            result["cascade_stopped"] = True
            return result

        # Minimum movement gate (volume/extreme/fair_value/stop/take may
        # pass movement=inf-ish or their own logic; abs/rel honor min_move).
        if kind in ("abs_move", "rel_move") and abs(movement) < config.min_move:
            result["skipped_min_move"] = 1
            return result

        if cascade_id is None:
            cascade_id = uuid.uuid4().hex

        category = CandidateService.event_category(event_id)
        candidates = CandidateService.collect_candidate_ids(event_id, category=category)
        # Price triggers care about holders + watchers/subscribers, not
        # generic archetype/expert interest — filter to position/interest.
        candidates = {
            aid: meta for aid, meta in candidates.items()
            if meta["role"] in ("holder", "watcher", "subscriber")
        }
        if not candidates:
            return result

        params = cls._hydrate_agents(list(candidates.keys()))
        priority = TRIGGER_PRIORITY[TriggerType.PRICE.value]

        # Rank holders first, then by price_sensitivity.
        ranked = sorted(
            candidates.items(),
            key=lambda kv: (kv[1]["source_rank"],
                            params.get(kv[0], {}).get("price_sensitivity", 0.0)),
            reverse=True,
        )

        scheduled = 0
        for aid, meta in ranked:
            if scheduled >= config.max_per_market:
                result["capped"] = True
                break
            # Cooldown gate (atomic upsert of last_fired).
            if not cls._check_and_set_cooldown(aid, market_id, TriggerType.PRICE.value,
                                               now, config.cooldown_s):
                result["skipped_cooldown"] += 1
                continue

            p = params.get(aid, {})
            price_sens = p.get("price_sensitivity", 0.0)
            position_relevance = 1.0 if meta["role"] == "holder" else 0.3
            wake_score = _clamp01(0.5 * min(1.0, abs(movement) / 0.2)
                                  + 0.3 * position_relevance + 0.2 * price_sens)
            # Small deterministic jitter (normal window) so price wakes on
            # one market don't all land on the same timestamp.
            sched_at = cls._jittered_time(now, "normal", sim_seed, aid, event_id, scheduled)
            cls._upsert_task(
                agent_id=aid, event_id=event_id, market_id=market_id,
                trigger_type=TriggerType.PRICE, priority=priority, tier=TIER_NORMAL,
                scheduled_at=sched_at, now=now,
                relevance=min(1.0, abs(movement) / 0.2), information_impact=0.0,
                position_relevance=position_relevance, expertise=0.0,
                portfolio_risk=0.0, wake_score=wake_score,
                cascade_id=cascade_id, cascade_depth=cascade_depth,
                trigger_trade_id=trigger_trade_id,
            )
            scheduled += 1

        db.session.commit()
        db.session.expunge_all()
        result["scheduled"] = scheduled
        result["cascade_id"] = cascade_id
        return result

    # ==================================================================
    # Portfolio + lifecycle triggers (urgent, protected from shedding)
    # ==================================================================

    @classmethod
    def portfolio_risk_event(
        cls, agent_id: int, reason: str, market_id: Optional[int] = None,
        event_id: Optional[int] = None, risk: float = 1.0,
        critical: bool = True, now: Optional[float] = None,
        sim_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Urgently schedule a single Agent for a portfolio-risk reason.

        Reasons: event_exposure / total_exposure / drawdown /
        unrealized_loss / opposing_information. Critical risk is scheduled
        at `now` (no jitter delay) and is protected from load shedding.
        """
        cls.ensure_schema()
        if now is None:
            now = SchedulerService.now()
        priority = TRIGGER_PRIORITY[TriggerType.PORTFOLIO_RISK.value]
        cls._upsert_task(
            agent_id=agent_id, event_id=event_id, market_id=market_id,
            trigger_type=TriggerType.PORTFOLIO_RISK, priority=priority,
            tier=TIER_URGENT, scheduled_at=now, now=now,
            relevance=1.0, information_impact=0.0, position_relevance=1.0,
            expertise=0.0, portfolio_risk=_clamp01(risk), wake_score=1.0,
            extra_reason=reason,
        )
        db.session.commit()
        db.session.expunge_all()
        return {"agent_id": agent_id, "reason": reason, "critical": critical,
                "wake_reason": TriggerType.PORTFOLIO_RISK.value}

    @classmethod
    def market_closing_event(
        cls, event_id: int, market_id: int, now: Optional[float] = None,
        sim_seed: Optional[int] = None, budget: Optional[EventBudget] = None,
    ) -> Dict[str, Any]:
        """Schedule holders of a near-closing market (urgent, minimal jitter)."""
        cls.ensure_schema()
        if now is None:
            now = SchedulerService.now()
        if sim_seed is None:
            sim_seed = SchedulerService.get_clock().simulation_seed
        holders = CandidateService._holder_ids(event_id)
        priority = TRIGGER_PRIORITY[TriggerType.MARKET_CLOSING.value]
        scheduled = 0
        for aid in holders:
            sched_at = cls._jittered_time(now, "major", sim_seed, aid, event_id, 0)
            cls._upsert_task(
                agent_id=aid, event_id=event_id, market_id=market_id,
                trigger_type=TriggerType.MARKET_CLOSING, priority=priority,
                tier=TIER_URGENT, scheduled_at=sched_at, now=now,
                relevance=1.0, information_impact=0.0, position_relevance=1.0,
                expertise=0.0, portfolio_risk=0.0, wake_score=0.9,
            )
            scheduled += 1
        db.session.commit()
        db.session.expunge_all()
        return {"event_id": event_id, "scheduled": scheduled,
                "wake_reason": TriggerType.MARKET_CLOSING.value}

    @classmethod
    def resolution_event(
        cls, event_id: int, market_id: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Schedule all holders on a resolved event (top priority, protected)."""
        cls.ensure_schema()
        if now is None:
            now = SchedulerService.now()
        holders = CandidateService._holder_ids(event_id)
        priority = TRIGGER_PRIORITY[TriggerType.RESOLUTION.value]
        scheduled = 0
        for aid in holders:
            cls._upsert_task(
                agent_id=aid, event_id=event_id, market_id=market_id,
                trigger_type=TriggerType.RESOLUTION, priority=priority,
                tier=TIER_URGENT, scheduled_at=now, now=now,
                relevance=1.0, information_impact=0.0, position_relevance=1.0,
                expertise=0.0, portfolio_risk=0.0, wake_score=1.0,
            )
            scheduled += 1
        db.session.commit()
        db.session.expunge_all()
        return {"event_id": event_id, "scheduled": scheduled,
                "wake_reason": TriggerType.RESOLUTION.value}

    # ==================================================================
    # Cooldown
    # ==================================================================

    @classmethod
    def _check_and_set_cooldown(
        cls, agent_id: int, market_id: int, trigger_type: str,
        now: float, cooldown_s: float,
    ) -> bool:
        """True if allowed (and records now); False if within cooldown.

        Uses the unique (agent, market, type) row. Reads last_fired_at,
        and if outside the window, updates it. Single-statement update
        keeps it transactionally safe under the surrounding commit.
        """
        row = (
            TriggerCooldown.query
            .filter_by(agent_id=agent_id, market_id=market_id, trigger_type=trigger_type)
            .one_or_none()
        )
        if row is None:
            db.session.add(TriggerCooldown(
                agent_id=agent_id, market_id=market_id,
                trigger_type=trigger_type, last_fired_at=now,
            ))
            db.session.flush()
            return True
        if now - row.last_fired_at < cooldown_s:
            return False
        row.last_fired_at = now
        db.session.flush()
        return True

    # ==================================================================
    # Deduplicated task upsert (transactionally safe)
    # ==================================================================

    @classmethod
    def _upsert_task(
        cls, *, agent_id, event_id, market_id, trigger_type: TriggerType,
        priority: int, tier: str, scheduled_at: float, now: float,
        relevance: float, information_impact: float, position_relevance: float,
        expertise: float, portfolio_risk: float, wake_score: float,
        cascade_id: Optional[str] = None, cascade_depth: int = 0,
        trigger_trade_id: Optional[int] = None, extra_reason: Optional[str] = None,
    ) -> WakeUpTask:
        """Insert a WakeUpTask or MERGE into an existing one for the bucket.

        Merge rules when a row already exists for (agent, event, bucket):
          * priority = max(existing, new)
          * scheduled_at = min (earliest urgent time wins)
          * wake_reasons = union of trigger types
          * ranking signals = max of each
          * trigger_type/tier promoted to the higher-priority one
          * latest evidence/market versions kept (max)
          * cascade metadata copied if the new one has it
        The unique dedup_key + a re-query under the same transaction makes
        this safe against concurrent duplicate triggers.
        """
        bucket = time_bucket(scheduled_at)
        dedup_key = make_dedup_key(agent_id, event_id, bucket)
        reason = trigger_type.value + (f":{extra_reason}" if extra_reason else "")

        existing = WakeUpTask.query.filter_by(dedup_key=dedup_key).one_or_none()
        if existing is None:
            task = WakeUpTask(
                agent_id=agent_id, event_id=event_id, market_id=market_id,
                trigger_type=trigger_type.value, priority=priority, tier=tier,
                status=STATUS_PENDING, scheduled_at=scheduled_at, time_bucket=bucket,
                relevance=relevance, information_impact=information_impact,
                position_relevance=position_relevance, expertise=expertise,
                portfolio_risk=portfolio_risk, wake_score=wake_score,
                wake_reasons=reason, dedup_key=dedup_key,
                cascade_id=cascade_id, cascade_depth=cascade_depth,
                trigger_trade_id=trigger_trade_id,
            )
            db.session.add(task)
            try:
                db.session.flush()
                return task
            except Exception:  # unique race — fall through to merge
                db.session.rollback()
                existing = WakeUpTask.query.filter_by(dedup_key=dedup_key).one_or_none()
                if existing is None:
                    raise

        # ---- merge ----
        if priority > existing.priority:
            existing.priority = priority
            existing.trigger_type = trigger_type.value
            existing.tier = tier
        existing.scheduled_at = min(existing.scheduled_at, scheduled_at)
        existing.relevance = max(existing.relevance, relevance)
        existing.information_impact = max(existing.information_impact, information_impact)
        existing.position_relevance = max(existing.position_relevance, position_relevance)
        existing.expertise = max(existing.expertise, expertise)
        existing.portfolio_risk = max(existing.portfolio_risk, portfolio_risk)
        existing.wake_score = max(existing.wake_score, wake_score)
        reasons = set((existing.wake_reasons or "").split(",")) if existing.wake_reasons else set()
        reasons.discard("")
        reasons.add(reason)
        existing.wake_reasons = ",".join(sorted(reasons))
        if market_id is not None and existing.market_id is None:
            existing.market_id = market_id
        if cascade_id is not None:
            existing.cascade_id = cascade_id
            existing.cascade_depth = max(existing.cascade_depth, cascade_depth)
        if trigger_trade_id is not None:
            existing.trigger_trade_id = trigger_trade_id
        db.session.flush()
        return existing

    # ==================================================================
    # Load shedding
    # ==================================================================

    @classmethod
    def shed_load(cls, keep: int, now: Optional[float] = None) -> Dict[str, Any]:
        """Shed the lowest-priority pending tasks down to `keep` rows.

        RESOLUTION and PORTFOLIO_RISK tasks are NEVER shed. Among the
        sheddable remainder, the lowest (priority, wake_score) go first.
        Marks shed rows STATUS_SHED (audit trail), doesn't delete them.
        """
        cls.ensure_schema()
        protected_types = (TriggerType.RESOLUTION.value, TriggerType.PORTFOLIO_RISK.value)

        pending_total = (
            db.session.query(db.func.count(WakeUpTask.id))
            .filter(WakeUpTask.status == STATUS_PENDING).scalar() or 0
        )
        protected = (
            db.session.query(db.func.count(WakeUpTask.id))
            .filter(
                WakeUpTask.status == STATUS_PENDING,
                WakeUpTask.trigger_type.in_(protected_types),
            ).scalar() or 0
        )
        if pending_total <= keep:
            return {"pending": pending_total, "shed": 0, "protected": protected, "kept": pending_total}

        # How many sheddable rows to keep = keep minus the protected ones
        # (which we always keep). Never shed a protected row.
        keep_sheddable = max(0, keep - protected)
        sheddable = (
            db.session.query(WakeUpTask.id)
            .filter(
                WakeUpTask.status == STATUS_PENDING,
                ~WakeUpTask.trigger_type.in_(protected_types),
            )
            .order_by(WakeUpTask.priority.asc(), WakeUpTask.wake_score.asc(),
                      WakeUpTask.scheduled_at.desc())
            .all()
        )
        shed_ids = [r[0] for r in sheddable[keep_sheddable:]]
        shed_count = 0
        if shed_ids:
            CHUNK = 900
            for i in range(0, len(shed_ids), CHUNK):
                chunk = shed_ids[i:i + CHUNK]
                db.session.query(WakeUpTask).filter(WakeUpTask.id.in_(chunk)).update(
                    {WakeUpTask.status: STATUS_SHED}, synchronize_session=False
                )
                shed_count += len(chunk)
            db.session.commit()
        db.session.expunge_all()
        return {
            "pending": pending_total, "shed": shed_count, "protected": protected,
            "kept": pending_total - shed_count,
        }

    # ==================================================================
    # Due query (bounded, ordered by scheduled time)
    # ==================================================================

    @classmethod
    def due_events(cls, limit: int = 100, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return up to `limit` pending event-driven tasks due at `now`.

        Ordered by (priority desc, scheduled_at asc). Bounded index scan
        on (status, scheduled_at). Read-only — a later worker phase claims.
        """
        cls.ensure_schema()
        if now is None:
            now = SchedulerService.now()
        rows = db.session.execute(
            text(
                "SELECT id, agent_id, event_id, market_id, trigger_type, priority, "
                "tier, scheduled_at, wake_score, wake_reasons, cascade_id, cascade_depth "
                "FROM wakeup_tasks "
                "WHERE status = :pending AND scheduled_at <= :now "
                "ORDER BY priority DESC, scheduled_at ASC "
                "LIMIT :limit"
            ),
            {"pending": STATUS_PENDING, "now": now, "limit": limit},
        ).fetchall()
        db.session.expunge_all()
        return [
            {"id": r[0], "agent_id": r[1], "event_id": r[2], "market_id": r[3],
             "trigger_type": r[4], "priority": r[5], "tier": r[6],
             "scheduled_at": r[7], "wake_score": r[8], "wake_reasons": r[9],
             "cascade_id": r[10], "cascade_depth": r[11]}
            for r in rows
        ]


__all__ = [
    "TriggerService",
    "EventBudget",
    "PriceTriggerConfig",
    "compute_wake_score",
]
