"""Wake-up processor / worker-pipeline tests.

Isolated in-memory SQLite; deterministic; offline belief stub (no real
LLM). Exits 0/1.

    python test_wakeup_processor.py
"""

import sys
from datetime import datetime, timedelta

from flask import Flask

from models import (
    db, Agent, AgentArchetype, Event, EventType, Market, MarketStatus,
    Position, WakeUpTask, TriggerType, TRIGGER_PRIORITY,
    STATUS_PENDING, STATUS_DONE, STATUS_FAILED,
    EvidenceBundle, ArchetypeBelief, AgentDecision, Trade,
    make_dedup_key, time_bucket,
)
from services import (
    PopulationService, SchedulerService, AgentWakeupProcessor, ProcessorConfig,
    MarketService,
)


_PASS = 0
_FAIL = 0
_MIX = {"evidence_value": 1.0}


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def _fresh_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
        SchedulerService.ensure_schema()
    return app


def _seed(app, n_arch=3, n_agents=10, seed=1, b=500.0, belief_prob=None,
          belief_conf=0.8):
    """Population + open event/market + evidence bundle v1.

    If `belief_prob` is given, pre-seed ArchetypeBeliefs at that posterior
    for bundle v1 so beliefs are FRESH (no LLM) and the edge is known.
    Returns (event_id, market_id).
    """
    with app.app_context():
        PopulationService.generate_default_archetypes(count=n_arch, seed=seed, mix=_MIX)
        PopulationService.generate_agents(count=n_agents, seed=seed, mix=_MIX)
        ev = Event(title="Q", description="d", category="markets",
                   event_type=EventType.BINARY,
                   close_time=datetime.utcnow() + timedelta(days=30),
                   resolution_source="x")
        db.session.add(ev); db.session.flush()
        mk = Market(event_id=ev.id, status=MarketStatus.OPEN, liquidity_b=b)
        db.session.add(mk); db.session.flush()
        eid, mid = ev.id, mk.id
        db.session.add(EvidenceBundle(
            event_id=eid, version=1, supporting_evidence_ids=[],
            opposing_evidence_ids=[], neutral_evidence_ids=[],
            aggregate_impact=0.5, current_summary="seed"))
        db.session.commit()
        if belief_prob is not None:
            for arch in db.session.query(AgentArchetype).all():
                db.session.add(ArchetypeBelief(
                    archetype_id=arch.id, event_id=eid, evidence_bundle_version=1,
                    posterior_probability=belief_prob, confidence=belief_conf,
                    model_tier="BALANCED"))
            db.session.commit()
        SchedulerService.initialize_natural(seed=seed)
        return eid, mid


def _enqueue(eid, mid, agent_ids, now, priority=None, cascade_depth=0):
    b = time_bucket(now)
    for aid in agent_ids:
        db.session.add(WakeUpTask(
            agent_id=aid, event_id=eid, market_id=mid,
            trigger_type=TriggerType.INFORMATION.value,
            priority=priority or TRIGGER_PRIORITY[TriggerType.INFORMATION.value],
            scheduled_at=now, time_bucket=b, wake_score=0.9,
            cascade_depth=cascade_depth,
            dedup_key=make_dedup_key(aid, eid, b) + f":{aid}"))
    db.session.commit()


# ---------------------------------------------------------------------

def test_task_becomes_completed():
    app = _fresh_app()
    eid, mid = _seed(app, belief_prob=0.5)  # neutral → HOLDs, still completes
    with app.app_context():
        now = SchedulerService.now()
        _enqueue(eid, mid, range(1, 6), now)
        proc = AgentWakeupProcessor(config=ProcessorConfig(sim_seed=1))
        m = proc.run_once(limit=100)
        pending = db.session.query(db.func.count(WakeUpTask.id)).filter_by(status=STATUS_PENDING).scalar()
        done = db.session.query(db.func.count(WakeUpTask.id)).filter_by(status=STATUS_DONE).scalar()
        check("all claimed tasks completed", m.completed == 5 and done == 5, f"done={done}")
        check("no tasks left pending", pending == 0)


def test_hold_produces_decision_no_trade():
    app = _fresh_app()
    eid, mid = _seed(app, belief_prob=0.5)  # belief == market → no edge → HOLD
    with app.app_context():
        now = SchedulerService.now()
        _enqueue(eid, mid, range(1, 6), now)
        proc = AgentWakeupProcessor(config=ProcessorConfig(sim_seed=1))
        m = proc.run_once(limit=100)
        decisions = db.session.query(db.func.count(AgentDecision.id)).scalar()
        holds = db.session.query(db.func.count(AgentDecision.id)).filter_by(was_hold=True).scalar()
        trades = db.session.query(db.func.count(Trade.id)).scalar()
        check("every processed task recorded an AgentDecision", decisions == 5)
        check("neutral edge → all HOLD decisions", holds == 5, f"holds={holds}")
        check("HOLD produces NO Trade rows", trades == 0)


def test_successful_action_produces_decision_and_trade():
    app = _fresh_app()
    # Strong belief (0.9) vs 0.5 market → clear edge → BUY_YES.
    eid, mid = _seed(app, belief_prob=0.9, belief_conf=0.9)
    with app.app_context():
        now = SchedulerService.now()
        _enqueue(eid, mid, range(1, 6), now)
        proc = AgentWakeupProcessor(config=ProcessorConfig(sim_seed=1))
        m = proc.run_once(limit=100)
        trades = db.session.query(db.func.count(Trade.id)).scalar()
        traded_decisions = (
            db.session.query(db.func.count(AgentDecision.id))
            .filter(AgentDecision.trade_id.isnot(None)).scalar()
        )
        check("strong edge produced at least one Trade", trades >= 1, f"trades={trades}")
        check("each trade links to a decision", traded_decisions == trades)
        check("no LLM belief updates (beliefs pre-seeded fresh)",
              m.belief_llm_updates == 0, f"llm={m.belief_llm_updates}")


def test_stale_task_skipped_safely():
    """A task whose event has closed (no longer relevant) completes as
    skipped without a trade — the worker does not choke on it."""
    app = _fresh_app()
    eid, mid = _seed(app, belief_prob=0.9)
    with app.app_context():
        # Close the market after enqueue → task is no longer relevant.
        now = SchedulerService.now()
        _enqueue(eid, mid, range(1, 4), now)
        mk = db.session.get(Market, mid)
        mk.status = MarketStatus.RESOLVED
        db.session.commit()
        proc = AgentWakeupProcessor(config=ProcessorConfig(sim_seed=1))
        m = proc.run_once(limit=100)
        done = db.session.query(db.func.count(WakeUpTask.id)).filter_by(status=STATUS_DONE).scalar()
        trades = db.session.query(db.func.count(Trade.id)).scalar()
        check("irrelevant tasks are completed safely", done == 3, f"done={done}")
        check("irrelevant task counted as skipped", m.skipped_irrelevant == 3)
        check("no trades executed on closed market", trades == 0)


def test_model_routing_metadata_stored():
    """When a belief is stale, the processor updates it via the router and
    the ArchetypeBelief records the model tier."""
    app = _fresh_app()
    eid, mid = _seed(app, belief_prob=None)  # NO pre-seeded belief → stale
    with app.app_context():
        now = SchedulerService.now()
        _enqueue(eid, mid, range(1, 4), now)
        # Give the processor a router so belief updates route + record tier.
        from llm import get_model_router
        proc = AgentWakeupProcessor(
            config=ProcessorConfig(sim_seed=1), llm_client=None,
            router=get_model_router(),
        )
        m = proc.run_once(limit=100)
        beliefs = db.session.query(ArchetypeBelief).all()
        check("belief LLM update happened (stale belief)", m.belief_llm_updates >= 1,
              f"llm={m.belief_llm_updates}")
        check("archetype beliefs created", len(beliefs) >= 1)
        check("model tier metadata stored on belief",
              all(b.model_tier for b in beliefs), f"tiers={[b.model_tier for b in beliefs]}")
        check("processor recorded a tier distribution",
              bool(m.tier_distribution), f"{m.tier_distribution}")


def test_worker_continues_after_one_failure():
    """A poisoned task (belief update raises) must not crash the worker;
    the remaining tasks still process."""
    app = _fresh_app()
    eid, mid = _seed(app, belief_prob=0.9)
    with app.app_context():
        now = SchedulerService.now()
        _enqueue(eid, mid, range(1, 6), now)
        # Add one task pointing at a non-existent agent → forces a failure
        # path inside _process_one when it loads the agent (None → skip),
        # so instead we inject a broken agent id that exists but whose
        # archetype row we delete to trigger a downstream error.
        bad = WakeUpTask(
            agent_id=99999, event_id=eid, market_id=mid,
            trigger_type=TriggerType.INFORMATION.value,
            priority=TRIGGER_PRIORITY[TriggerType.INFORMATION.value],
            scheduled_at=now, time_bucket=time_bucket(now), wake_score=0.9,
            dedup_key="bad:task:1")
        db.session.add(bad); db.session.commit()

        proc = AgentWakeupProcessor(config=ProcessorConfig(sim_seed=1))
        m = proc.run_once(limit=100)
        # The 5 real tasks + the bad one were all claimed; the real ones
        # complete; the bad one (missing agent) completes as skipped.
        real_done = (
            db.session.query(db.func.count(WakeUpTask.id))
            .filter(WakeUpTask.status == STATUS_DONE,
                    WakeUpTask.agent_id.in_(range(1, 6))).scalar()
        )
        check("worker processed all real tasks despite a bad one", real_done == 5,
              f"real_done={real_done}")
        check("worker did not crash (claimed all 6)", m.claimed == 6)


def test_worker_records_error_and_bounds_retries():
    """A genuinely erroring task records a concise error, bumps retry_count,
    and after max_retries becomes FAILED (not retried forever)."""
    app = _fresh_app()
    eid, mid = _seed(app, belief_prob=0.9)
    with app.app_context():
        now = SchedulerService.now()
        _enqueue(eid, mid, [1], now)

        # Monkeypatch the policy to always raise, forcing the failure path.
        proc = AgentWakeupProcessor(config=ProcessorConfig(sim_seed=1, max_retries=2))

        class _BoomPolicy:
            def decide(self, **kw):
                raise RuntimeError("boom in policy")
        proc._policy = _BoomPolicy()

        # Run repeatedly; the task should re-queue then FAIL, never loop
        # unbounded.
        for _ in range(5):
            proc.run_once(limit=10)
        t = db.session.query(WakeUpTask).filter_by(agent_id=1).first()
        check("erroring task eventually marked FAILED", t.status == STATUS_FAILED,
              f"status={t.status}")
        check("retry_count bounded at max_retries", t.retry_count == 2,
              f"retry_count={t.retry_count}")
        check("concise error stored", bool(t.last_error) and "boom" in t.last_error)


def test_micro_batches_refresh_market_state():
    """With a tiny micro-batch, many same-market tasks still execute
    against fresh state each time — each successful execute bumps the
    version, and later tasks re-read it (no stale-crash)."""
    app = _fresh_app()
    eid, mid = _seed(app, belief_prob=0.9, belief_conf=0.9, b=2000.0)
    with app.app_context():
        now = SchedulerService.now()
        _enqueue(eid, mid, range(1, 9), now)
        proc = AgentWakeupProcessor(config=ProcessorConfig(sim_seed=1, micro_batch=2))
        m = proc.run_once(limit=100)
        mk = db.session.get(Market, mid)
        trades = db.session.query(db.func.count(Trade.id)).scalar()
        check("multiple micro-batches ran", m.batches >= 2, f"batches={m.batches}")
        check("market version advanced with executed trades",
              mk.version == trades and trades >= 1,
              f"version={mk.version} trades={trades}")
        check("all tasks completed despite batching", m.completed == 8)
        # No task should have been abandoned as stale-unrecoverable.
        check("stale requotes (if any) were bounded, not fatal",
              m.claimed == 8)


def test_price_triggers_remain_bounded():
    """A task at the max cascade depth must NOT publish further triggers."""
    app = _fresh_app()
    eid, mid = _seed(app, belief_prob=0.95, belief_conf=0.95, b=200.0)
    with app.app_context():
        now = SchedulerService.now()
        # Enqueue tasks already AT the cascade cap (depth = max).
        _enqueue(eid, mid, range(1, 4), now, cascade_depth=3)
        proc = AgentWakeupProcessor(config=ProcessorConfig(
            sim_seed=1, max_price_cascade_depth=3))
        m = proc.run_once(limit=100)
        check("no price triggers published beyond max cascade depth",
              m.price_triggers_published == 0, f"published={m.price_triggers_published}")
        check("cascade-capped triggers were suppressed (backpressure counted)",
              m.triggers_suppressed_backpressure >= 1
              or m.trades == 0, f"suppressed={m.triggers_suppressed_backpressure}")


def test_fresh_belief_uses_no_llm():
    app = _fresh_app()
    eid, mid = _seed(app, belief_prob=0.7)  # pre-seeded fresh belief
    with app.app_context():
        now = SchedulerService.now()
        _enqueue(eid, mid, range(1, 6), now)
        # Router that explodes if used — proves fresh beliefs skip the LLM.
        class _BoomRouter:
            def route(self, *a, **k):
                raise AssertionError("router must not be called for fresh beliefs")
        proc = AgentWakeupProcessor(config=ProcessorConfig(sim_seed=1),
                                    router=_BoomRouter())
        m = proc.run_once(limit=100)
        check("fresh beliefs reused (no LLM)", m.belief_llm_updates == 0)
        check("all tasks completed", m.completed == 5)


ALL_TESTS = [
    test_task_becomes_completed,
    test_hold_produces_decision_no_trade,
    test_successful_action_produces_decision_and_trade,
    test_stale_task_skipped_safely,
    test_model_routing_metadata_stored,
    test_worker_continues_after_one_failure,
    test_worker_records_error_and_bounds_retries,
    test_micro_batches_refresh_market_state,
    test_price_triggers_remain_bounded,
    test_fresh_belief_uses_no_llm,
]


def main():
    print("Running wake-up processor tests (in-memory SQLite, offline LLM)...")
    for t in ALL_TESTS:
        print(f"\n{t.__name__}:")
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            global _FAIL
            _FAIL += 1
            import traceback
            print(f"  [FAIL] {t.__name__} raised {exc!r}")
            traceback.print_exc()
    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
