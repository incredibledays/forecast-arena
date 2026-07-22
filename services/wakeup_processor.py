"""AgentWakeupProcessor — the worker pipeline that turns wake-up tasks
into decisions and (when justified) executed trades.

This is the integration layer. It orchestrates the components built in
earlier phases without changing their core math:

    SchedulerService   — virtual clock, natural-wakeup rescheduling
    TriggerService     — price-change trigger publishing (bounded cascade)
    BeliefService      — archetype belief updates + lazy agent beliefs
    MemoryService      — compact stats, effective persona, incremental update
    ActionPolicy       — pure-code decision (NO LLM, NO history scan)
    MarketService      — strict LMSR quote + execution
    MarketExecutor     — per-market serialization + AgentDecision audit

Per-task flow (spec §1):
    claim -> load agent/event -> relevance -> evidence bundle ->
    (belief fresh? skip LLM) -> derive AgentBelief -> memory stats ->
    position + portfolio -> market state -> ActionPolicy -> AgentDecision;
    HOLD completes; else quote -> revalidate edge after impact ->
    execute (atomic, stale-safe) -> update memory -> publish price
    trigger (bounded) -> schedule next natural wake-up -> complete.

Model routing (spec §2): the LLM is only ever touched when the archetype
belief is STALE for the current EvidenceBundle version — routed through
BeliefService (BALANCED routine / STRONG major-conflict). A fresh belief
means zero LLM calls. ActionPolicy and LMSR execution never call an LLM.

Robustness (spec §6): one agent's failure never crashes the worker; the
task records a concise error + retry_count, and invalid business
decisions are not retried indefinitely.
"""

from __future__ import annotations

import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from models import (
    Agent, AgentBelief, ArchetypeBelief, Event, EventType, Market,
    MarketStatus, Position, Trade, WakeUpTask,
    STATUS_PENDING, STATUS_CLAIMED, STATUS_DONE, STATUS_SHED, STATUS_FAILED,
    db,
)
from services.action_policy import (
    ActionPolicy, BeliefInput, PortfolioSummary,
)
from services.belief_service import BeliefService
from services.market_executor import MarketExecutor
from services.market_service import MarketService, MarketError, StaleQuoteError
from services.memory_service import MemoryService
from services.scheduler_service import SchedulerService, sample_wait_seconds
from services.trigger_service import TriggerService, PriceTriggerConfig


# ---- backpressure / batching defaults (all configurable) ------------
DEFAULT_MICRO_BATCH = 100
DEFAULT_MAX_PENDING_PER_MARKET = 5000
DEFAULT_MAX_PENDING_TOTAL = 5000
DEFAULT_MAX_EXECUTIONS_PER_MARKET_PER_MIN = 600
DEFAULT_MAX_PRICE_CASCADE_DEPTH = 3
MAX_RETRIES = 3                       # after this a task is FAILED, not retried
MAX_STALE_REQUOTES = 2                # inner requote attempts per execution
# Price move (abs Δ in YES price) below which we don't publish a trigger.
PRICE_TRIGGER_MIN_MOVE = 0.03
# Post-impact edge floor: if the marginal price moved past belief, cancel.
_EDGE_CANCEL_EPSILON = 1e-4


@dataclass
class ProcessorConfig:
    micro_batch: int = DEFAULT_MICRO_BATCH
    max_pending_per_market: int = DEFAULT_MAX_PENDING_PER_MARKET
    max_pending_total: int = DEFAULT_MAX_PENDING_TOTAL
    max_executions_per_market_per_min: int = DEFAULT_MAX_EXECUTIONS_PER_MARKET_PER_MIN
    max_price_cascade_depth: int = DEFAULT_MAX_PRICE_CASCADE_DEPTH
    price_trigger_max_per_trade: int = 5
    max_retries: int = MAX_RETRIES
    sim_seed: int = 0
    publish_price_triggers: bool = True
    fair_queue: bool = False


@dataclass
class ProcessorMetrics:
    claimed: int = 0
    completed: int = 0
    holds: int = 0
    trades: int = 0
    deferred: int = 0
    failed: int = 0
    skipped_irrelevant: int = 0
    skipped_stale: int = 0
    belief_llm_updates: int = 0
    reused_fresh_belief: int = 0
    price_triggers_published: int = 0
    triggers_suppressed_backpressure: int = 0
    tier_distribution: Dict[str, int] = field(default_factory=dict)
    batches: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TaskSnapshot:
    """Plain-value snapshot of a claimed WakeUpTask.

    The processor calls downstream services (belief, memory) that clear
    the SQLAlchemy session (`expunge_all`), which would detach a held
    WakeUpTask ORM object. We snapshot the fields we need at claim time
    and re-fetch the row by id only when writing status — so no ORM task
    instance is ever read across a session clear.
    """

    id: int
    agent_id: int
    event_id: Optional[int]
    market_id: Optional[int]
    priority: int
    scheduled_at: float
    cascade_id: Optional[str]
    cascade_depth: int

    @classmethod
    def of(cls, task: WakeUpTask) -> "TaskSnapshot":
        return cls(
            id=int(task.id), agent_id=int(task.agent_id),
            event_id=(int(task.event_id) if task.event_id is not None else None),
            market_id=(int(task.market_id) if task.market_id is not None else None),
            priority=int(task.priority or 0),
            scheduled_at=float(task.scheduled_at or 0.0),
            cascade_id=task.cascade_id,
            cascade_depth=int(task.cascade_depth or 0),
        )


class AgentWakeupProcessor:
    """Drains pending WakeUpTasks into decisions / trades. No web, no sleep."""

    def __init__(self, config: Optional[ProcessorConfig] = None,
                 llm_client=None, router=None):
        self.config = config or ProcessorConfig()
        self._policy = ActionPolicy()
        self._belief = BeliefService(llm_client=llm_client, router=router)
        self._llm = llm_client
        self._router = router
        # Rolling per-market execution timestamps for rate limiting.
        self._exec_times: Dict[int, List[float]] = defaultdict(list)

    # ==================================================================
    # Public entry points
    # ==================================================================

    def run_once(self, limit: int = 100, now: Optional[float] = None) -> ProcessorMetrics:
        """Claim and process up to `limit` due tasks, micro-batched by market.

        Tasks are grouped by market, sorted by (priority desc, scheduled
        time asc), and processed in micro-batches; the market price is
        re-read between tasks so a batch never executes against a stale
        in-memory snapshot, and each execution passes the freshly-read
        version so stale quotes are rejected + requoted.
        """
        metrics = ProcessorMetrics()
        if now is None:
            now = SchedulerService.now()

        claimed = self._claim_due(limit=limit, now=now)
        metrics.claimed = len(claimed)
        if not claimed:
            return metrics

        # Group by market and order within each market by priority/time.
        by_market: Dict[Optional[int], List[TaskSnapshot]] = defaultdict(list)
        for t in claimed:
            by_market[t.market_id].append(t)
        for market_id, tasks in by_market.items():
            tasks.sort(key=lambda t: (-int(t.priority or 0), float(t.scheduled_at or 0.0)))
            self._process_market_tasks(market_id, tasks, now, metrics)
        return metrics

    def run_loop(self, batch_limit: int = 100, max_batches: Optional[int] = None,
                 idle_sleep_ticks: int = 0) -> ProcessorMetrics:
        """Repeatedly `run_once` until the pending queue drains.

        Virtual-time friendly: does NOT sleep on real wall-clock. Stops
        when a pass claims nothing or `max_batches` is reached. Callers
        that want a daemon can wrap this + advance the clock.
        """
        total = ProcessorMetrics()
        n = 0
        while True:
            m = self.run_once(limit=batch_limit)
            self._merge(total, m)
            total.batches += 1
            n += 1
            if m.claimed == 0:
                break
            if max_batches is not None and n >= max_batches:
                break
        return total

    # ==================================================================
    # Claim (atomic compare-and-swap on status)
    # ==================================================================

    def _claim_due(self, limit: int, now: float) -> List[TaskSnapshot]:
        """Claim up to `limit` due pending tasks (PENDING -> CLAIMED).

        Uses a bounded SELECT of ids ordered by (priority, scheduled_at),
        then a guarded UPDATE per id (compare-and-swap on status) so two
        workers can't claim the same task. Returns plain-value snapshots
        (not ORM rows) so downstream session clears can't detach them.
        """
        if self.config.fair_queue:
            # Live/web mode: don't let a burst of high-priority PRICE tasks
            # on one event starve older NATURAL wakeups on other events.
            # Priority still breaks ties inside the same time slice.
            order_by = "scheduled_at ASC, priority DESC, wake_score DESC"
        else:
            order_by = "priority DESC, scheduled_at ASC"

        rows = db.session.execute(
            text(
                "SELECT id FROM wakeup_tasks "
                "WHERE status = :pending AND scheduled_at <= :now "
                f"ORDER BY {order_by} LIMIT :limit"
            ),
            {"pending": STATUS_PENDING, "now": now, "limit": int(limit)},
        ).fetchall()
        claimed: List[TaskSnapshot] = []
        for (task_id,) in rows:
            res = db.session.execute(
                text(
                    "UPDATE wakeup_tasks SET status = :claimed "
                    "WHERE id = :id AND status = :pending"
                ),
                {"claimed": STATUS_CLAIMED, "id": task_id, "pending": STATUS_PENDING},
            )
            if res.rowcount == 1:
                task = db.session.get(WakeUpTask, task_id)
                if task is not None:
                    claimed.append(TaskSnapshot.of(task))
        db.session.commit()
        db.session.expunge_all()
        return claimed

    # ==================================================================
    # Per-market micro-batch loop
    # ==================================================================

    def _process_market_tasks(self, market_id, tasks: List[TaskSnapshot], now,
                              metrics: ProcessorMetrics):
        batch_size = max(1, int(self.config.micro_batch))
        for start in range(0, len(tasks), batch_size):
            metrics.batches += 1
            batch = tasks[start:start + batch_size]
            for task in batch:
                # Refresh market price/version BETWEEN tasks so a long
                # batch never trades on a stale snapshot (spec §4). The
                # per-task pipeline re-reads market state itself.
                try:
                    self._process_one(task, now, metrics)
                except Exception as exc:  # noqa: BLE001 — never crash the worker
                    self._handle_failure(task, exc, metrics)

    # ==================================================================
    # Single-task pipeline (spec §1, steps 1–19)
    # ==================================================================

    def _process_one(self, task: TaskSnapshot, now: float, metrics: ProcessorMetrics):
        # (1) already claimed. (2) load agent + event; capture PRIMITIVES
        # up front. Downstream belief/memory services call expunge_all(),
        # which detaches ORM objects — so we never read an ORM attribute
        # across such a call; we re-fetch by id right before use instead.
        agent = db.session.get(Agent, task.agent_id)
        event = db.session.get(Event, task.event_id) if task.event_id else None
        if agent is None or event is None:
            return self._complete(task, metrics, reason="missing agent/event",
                                  counter="skipped_irrelevant")

        # (3) verify task relevance: event must be OPEN and tradeable.
        market = self._market_for(task, event)
        if market is None or market.status != MarketStatus.OPEN:
            return self._complete(task, metrics, reason="event not open",
                                  counter="skipped_irrelevant")

        agent_id = int(agent.id)
        event_id = int(event.id)
        market_id = int(market.id)
        category = event.category
        strategy_type = agent.strategy_type
        initial_cash = float(agent.initial_cash or 0.0)
        arch_id = agent.archetype_id

        # (4) latest EvidenceBundle + (5) archetype belief freshness.
        from models import EvidenceBundle
        latest_bundle = (
            EvidenceBundle.query.filter_by(event_id=event_id)
            .order_by(EvidenceBundle.version.desc()).first()
        )
        bundle_version = latest_bundle.version if latest_bundle else 0

        belief_fresh = self._archetype_belief_fresh(arch_id, event_id, bundle_version)
        if not belief_fresh and arch_id is not None:
            # (5) update archetype belief — the ONLY place an LLM may be
            # called. Routed inside BeliefService (BALANCED/STRONG).
            m = self._belief.update_archetype_beliefs(
                event_id, archetype_ids=[arch_id],
            )
            metrics.belief_llm_updates += int(m.get("archetype_llm_requests", 0))
            for tier, n in (m.get("tier_distribution") or {}).items():
                metrics.tier_distribution[tier] = metrics.tier_distribution.get(tier, 0) + n
        else:
            metrics.reused_fresh_belief += 1

        # (6) derive AgentBelief (pure code — reconstruct, no persistence).
        belief_dict = self._belief.reconstruct_agent_belief(
            agent_id, event_id, sim_seed=self.config.sim_seed,
        )
        if belief_dict is None:
            return self._complete(task, metrics, reason="no belief available",
                                  counter="skipped_irrelevant")
        arch_belief = self._current_archetype_belief(arch_id, event_id, bundle_version)
        belief = BeliefInput(
            calibrated_probability=belief_dict["calibrated_probability"],
            confidence=(arch_belief.confidence if arch_belief else 0.5),
        )

        # (7) memory stats + effective persona (these call expunge_all()).
        persona = MemoryService.compute_effective_persona(agent_id, category=category)
        memory = MemoryService.ensure_stats(agent_id, initial_cash=initial_cash)

        # --- RE-FETCH ORM objects now that all expunging services ran. ---
        agent = db.session.get(Agent, agent_id)
        market = db.session.get(Market, market_id)

        # (8) position + portfolio value.
        position = (
            Position.query.filter_by(agent_id=agent_id, market_id=market_id).one_or_none()
        )
        portfolio = self._portfolio_summary(agent, market, position)

        # (9) latest MarketState (price + version).
        price = MarketService.get_current_price(market_id)
        market_version = int(market.version or 0)
        market_state = {
            "market_id": market_id,
            "yes_price": price["yes_price"],
            "no_price": price["no_price"],
        }

        # (10) ActionPolicy — pure code, no LLM.
        decision = self._policy.decide(
            persona=persona, belief=belief, memory=memory, position=position,
            portfolio=portfolio, market_state=market_state,
            quote_fn=MarketService.quote_buy,
        )
        action = decision.recommended_action
        edge = decision.policy_factors.get("edge_yes")

        # (11)+(12) HOLD path: record decision, no trade, complete.
        if action == "HOLD":
            MarketExecutor.execute(
                agent_id=agent_id, market_id=market_id, action="HOLD",
                event_id=event_id, probability_yes=belief.calibrated_probability,
                confidence=belief.confidence, edge=edge, urgency=decision.urgency,
                outcome_side=decision.side, requested_notional=0.0,
                reasoning_summary=decision.reasoning_summary,
                policy_factors=decision.policy_factors,
            )
            metrics.holds += 1
            MemoryService.increment_after_wakeup(agent_id, traded=False,
                                                 initial_cash=initial_cash)
            self._schedule_next_natural(agent_id, now)
            return self._complete(task, metrics, reason="hold")

        # (13)+(14) detailed quote + post-impact edge revalidation, then
        # (15) execute transactionally with stale-safe inner requote.
        trade = self._execute_with_revalidation(
            agent_id=agent_id, event_id=event_id, market_id=market_id,
            decision=decision, belief=belief, metrics=metrics,
            event_analysis_summary=(
                arch_belief.reasoning_summary if arch_belief else None
            ),
        )
        if trade is None:
            metrics.holds += 1
            MemoryService.increment_after_wakeup(agent_id, traded=False,
                                                 initial_cash=initial_cash)
            self._schedule_next_natural(agent_id, now)
            return self._complete(task, metrics, reason="edge gone / stale-cancel")

        metrics.trades += 1
        # (16) update memory statistics incrementally (no history scan).
        notional = float(getattr(trade, "amount", 0.0) or 0.0)
        MemoryService.increment_after_trade(
            agent_id=agent_id, notional=notional,
            realized_pnl_delta=0.0, unrealized_pnl_delta=0.0,
            category=category, strategy_type=strategy_type,
            initial_cash=initial_cash,
        )
        MemoryService.increment_after_wakeup(agent_id, traded=True,
                                             initial_cash=initial_cash)

        # (17) publish price-change trigger (bounded cascade).
        self._maybe_publish_price_trigger(task, event_id, market_id, trade, now, metrics)

        # (18) schedule the agent's next natural wake-up.
        self._schedule_next_natural(agent_id, now)

        # (19) complete task.
        return self._complete(task, metrics, reason="trade")

    # ==================================================================
    # Execution with post-impact edge revalidation + stale requote
    # ==================================================================

    def _execute_with_revalidation(self, *, agent_id, event_id, market_id, decision,
                                   belief, metrics, event_analysis_summary=None):
        action = decision.recommended_action
        outcome_side = decision.side if decision.side in ("YES", "NO") else (
            "YES" if action.endswith("YES") else "NO"
        )
        # SELL/FLIP legs: size by fraction (policy set requested_notional as a
        # dollar value only for BUYs). We route BUY via amount, SELL via
        # fraction=1.0-ish; the executor + LMSR handle the rest.
        is_buy = action in ("BUY_YES", "BUY_NO")
        amount = float(decision.requested_notional or 0.0) if is_buy else 0.0
        fraction = None if is_buy else 1.0

        for attempt in range(MAX_STALE_REQUOTES + 1):
            # Re-read fresh market state each attempt (spec §6 stale flow).
            market = db.session.get(Market, market_id)
            if market is None or market.status != MarketStatus.OPEN:
                return None
            cur_version = int(market.version or 0)
            p_yes = float(belief.calibrated_probability)

            # (14) post-impact edge validation for BUYs: the trade is
            # profitable in expectation iff p_agent > AVERAGE_EXECUTION_PRICE
            # (not marginal_after — marginal is the last unit's price,
            # average is what we actually pay). Cancelling on marginal
            # produces false HOLDs in low-b markets where slippage is
            # steep. Comparing to average matches the LMSR audit spec §4.
            if is_buy and amount > 0:
                quote = MarketService.quote_buy(market_id, outcome_side, amount)
                shares = float(quote.get("shares", 0.0) or 0.0)
                if shares <= 0:
                    return None
                avg_price = float(quote.get("effective_price_per_share")
                                  or (amount / max(shares, 1e-9)))
                fair = p_yes if outcome_side == "YES" else (1.0 - p_yes)
                edge_at_avg = fair - avg_price
                if edge_at_avg <= _EDGE_CANCEL_EPSILON:
                    # Trade would lose in expectation at the average
                    # execution price — record the cancelled intent as HOLD.
                    MarketExecutor.execute(
                        agent_id=agent_id, market_id=market_id, action="HOLD",
                        event_id=event_id, probability_yes=p_yes,
                        confidence=belief.confidence, edge=edge_at_avg,
                        urgency=decision.urgency, outcome_side=decision.side,
                        requested_notional=amount,
                        reasoning_summary="[edge gone at avg-execution price] "
                                          + (decision.reasoning_summary or ""),
                        policy_factors=decision.policy_factors,
                    )
                    return None

            try:
                result = MarketExecutor.execute(
                    agent_id=agent_id, market_id=market_id, action=action,
                    amount=amount, fraction=fraction,
                    expected_market_state_version=cur_version,
                    event_id=event_id, probability_yes=p_yes,
                    confidence=belief.confidence,
                    edge=decision.policy_factors.get("edge_yes"),
                    urgency=decision.urgency, outcome_side=outcome_side,
                    requested_notional=(amount if is_buy else None),
                    reasoning_summary=decision.reasoning_summary,
                    event_analysis_summary=event_analysis_summary,
                    policy_factors=decision.policy_factors,
                )
            except StaleQuoteError:
                metrics.skipped_stale += 1
                continue  # requote loop (bounded by MAX_STALE_REQUOTES)
            except MarketError:
                # Business error (insufficient cash, nothing to sell, ...).
                # Not retryable — record nothing further, return None.
                return None

            # A HoldResult (e.g. FLIP no-op) counts as no trade.
            if getattr(result, "action", None) is not None and \
                    str(getattr(result.action, "value", result.action)) == "HOLD":
                return None
            return result

        # Exhausted requote attempts.
        return None

    # ==================================================================
    # Price-change trigger (bounded cascade + backpressure)
    # ==================================================================

    def _maybe_publish_price_trigger(self, task: TaskSnapshot, event_id, market_id,
                                     trade, now, metrics):
        if not self.config.publish_price_triggers:
            return
        move = abs(float(getattr(trade, "price_after", 0.0) or 0.0)
                   - float(getattr(trade, "price_before", 0.0) or 0.0))
        if move < PRICE_TRIGGER_MIN_MOVE:
            return
        # Cascade bound: never let price triggers chain past the max depth.
        parent_depth = int(task.cascade_depth or 0)
        if parent_depth >= self.config.max_price_cascade_depth:
            metrics.triggers_suppressed_backpressure += 1
            return
        # Backpressure: per-market pending cap.
        pending_total = (
            db.session.query(db.func.count(WakeUpTask.id))
            .filter(WakeUpTask.status == STATUS_PENDING).scalar() or 0
        )
        if pending_total >= self.config.max_pending_total:
            metrics.triggers_suppressed_backpressure += 1
            return
        pending = (
            db.session.query(db.func.count(WakeUpTask.id))
            .filter(WakeUpTask.market_id == market_id,
                    WakeUpTask.status == STATUS_PENDING).scalar() or 0
        )
        if pending >= self.config.max_pending_per_market:
            metrics.triggers_suppressed_backpressure += 1
            return
        try:
            res = TriggerService.price_event(
                event_id=event_id, market_id=market_id, kind="abs_move",
                movement=move,
                config=PriceTriggerConfig(
                    min_move=PRICE_TRIGGER_MIN_MOVE,
                    max_cascade_depth=self.config.max_price_cascade_depth,
                    max_per_market=max(0, int(self.config.price_trigger_max_per_trade)),
                ),
                cascade_id=task.cascade_id,
                cascade_depth=parent_depth + 1,
                trigger_trade_id=getattr(trade, "id", None),
                now=now,
            )
            metrics.price_triggers_published += int(res.get("scheduled", 0))
        except Exception as exc:  # noqa: BLE001 — trigger failure must not abort the trade
            print(f"[processor] price trigger publish failed: {exc}", file=sys.stderr)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _market_for(self, task: TaskSnapshot, event: Event) -> Optional[Market]:
        if task.market_id is not None:
            return db.session.get(Market, task.market_id)
        # BINARY events: primary market.
        return (
            Market.query.filter_by(event_id=event.id)
            .order_by(Market.id.asc()).first()
        )

    def _archetype_belief_fresh(self, arch_id, event_id, bundle_version) -> bool:
        if arch_id is None:
            return True  # no archetype → nothing to update via LLM
        return db.session.query(ArchetypeBelief.id).filter_by(
            archetype_id=arch_id, event_id=event_id,
            evidence_bundle_version=bundle_version,
        ).first() is not None

    def _current_archetype_belief(self, arch_id, event_id, bundle_version):
        if arch_id is None:
            return None
        return ArchetypeBelief.query.filter_by(
            archetype_id=arch_id, event_id=event_id,
            evidence_bundle_version=bundle_version,
        ).one_or_none()

    def _portfolio_summary(self, agent, market, position) -> PortfolioSummary:
        cash = float(agent.virtual_cash or 0.0)
        event_exposure = 0.0
        if position is not None:
            price = MarketService.get_current_price(market.id)
            event_exposure = (
                float(position.yes_shares or 0.0) * price["yes_price"]
                + float(position.no_shares or 0.0) * price["no_price"]
            )
        # Portfolio value ≈ cash + this event's marked exposure. A full
        # cross-market mark is a heavier query we avoid in the hot path;
        # the policy only needs bankroll + this-event exposure for sizing.
        return PortfolioSummary(
            virtual_cash=cash,
            portfolio_value=cash + event_exposure,
            total_exposure_notional=event_exposure,
            open_event_exposure_notional=event_exposure,
            open_positions_count=(1 if position is not None else 0),
        )

    def _schedule_next_natural(self, agent_id: int, now: float):
        """Reschedule the agent's next natural wake-up via its schedule row.

        Reuses the deterministic exponential sampler; leaves the natural
        scheduler as the source of truth. Best-effort: never raises into
        the task pipeline.
        """
        try:
            from models import AgentScheduleState
            st = db.session.get(AgentScheduleState, agent_id)
            if st is None:
                return
            seq = int(st.natural_wakeup_sequence or 0)
            wait = sample_wait_seconds(
                rate_per_day=float(st.base_wakeup_rate_per_day or 1.0),
                sim_seed=int(st.agent_random_seed or agent_id),
                agent_seed=int(agent_id), sequence=seq,
                version=int(st.scheduler_version or 1),
            )
            st.natural_wakeup_sequence = seq + 1
            st.last_natural_wakeup_at = now
            st.next_natural_wakeup_at = now + wait
            db.session.flush()
        except Exception as exc:  # noqa: BLE001
            print(f"[processor] reschedule failed for agent {agent_id}: {exc}",
                  file=sys.stderr)

    # ==================================================================
    # Completion / failure
    # ==================================================================

    def _complete(self, task: TaskSnapshot, metrics: ProcessorMetrics,
                  reason: str = "", counter: Optional[str] = None):
        # Re-fetch by id: the snapshot is detached from the session.
        row = db.session.get(WakeUpTask, task.id)
        if row is not None:
            row.status = STATUS_DONE
            row.last_error = None
            db.session.commit()
        metrics.completed += 1
        if counter == "skipped_irrelevant":
            metrics.skipped_irrelevant += 1
        return reason

    def _handle_failure(self, task: TaskSnapshot, exc: Exception,
                        metrics: ProcessorMetrics):
        """One agent's failure never crashes the worker (spec §6).

        Rolls back the poisoned transaction, records a concise error +
        bumps retry_count. Below the retry cap the task is re-queued
        (PENDING) for another pass; at/above the cap it is marked FAILED
        and never retried again.
        """
        db.session.rollback()
        concise = f"{type(exc).__name__}: {exc}"[:200]
        try:
            fresh = db.session.get(WakeUpTask, task.id)
            if fresh is None:
                return
            fresh.retry_count = int(fresh.retry_count or 0) + 1
            fresh.last_error = concise
            if fresh.retry_count >= self.config.max_retries:
                fresh.status = STATUS_FAILED
                metrics.failed += 1
            else:
                fresh.status = STATUS_PENDING   # re-queue for a later pass
                metrics.deferred += 1
            db.session.commit()
        except Exception as inner:  # noqa: BLE001 — last-ditch; don't crash
            db.session.rollback()
            print(f"[processor] failure bookkeeping error on task {task.id}: "
                  f"{inner}", file=sys.stderr)
        print(f"[processor] task {task.id} agent {task.agent_id} failed: "
              f"{concise}", file=sys.stderr)

    @staticmethod
    def _merge(total: ProcessorMetrics, m: ProcessorMetrics):
        for k, v in m.as_dict().items():
            if k == "tier_distribution":
                for tier, n in v.items():
                    total.tier_distribution[tier] = total.tier_distribution.get(tier, 0) + n
            elif isinstance(v, int):
                setattr(total, k, getattr(total, k) + v)


__all__ = [
    "AgentWakeupProcessor",
    "ProcessorConfig",
    "ProcessorMetrics",
]
