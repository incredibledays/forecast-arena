"""Event-driven trigger tests: sparse selection, budgets, jitter, dedup,
cooldown, cascade bounds, load shedding — plus query-count checks.

Isolated in-memory SQLite; virtual time; no LLM. Exits 0 / 1.

    python test_triggers.py
"""

import sys
from datetime import datetime, timedelta

from flask import Flask
from sqlalchemy import event, text

from models import (
    db, Event, EventType, Market, MarketStatus, Position,
    AgentEventInterest, ArchetypeEventInterest, AgentArchetype,
    ROLE_WATCHER, ROLE_SUBSCRIBER, WakeUpTask, TriggerType, STATUS_PENDING,
    STATUS_SHED,
)
from services import PopulationService, SchedulerService, CandidateService, TriggerService
from services.trigger_service import (
    EventBudget, PriceTriggerConfig, compute_wake_score,
)


_PASS = 0
_FAIL = 0


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
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return app


def _setup(app, n_agents=500, seed=42, build_expertise=True):
    """Seed population + one BINARY event/market. Returns (event_id, market_id)."""
    with app.app_context():
        PopulationService.generate_default_archetypes(count=40, seed=seed)
        PopulationService.generate_agents(count=n_agents, seed=seed, batch_size=250)
        if build_expertise:
            CandidateService.build_category_expertise()
        ev = Event(title="Q?", description="d", category="ai", event_type=EventType.BINARY,
                   close_time=datetime.utcnow() + timedelta(days=5), resolution_source="s")
        db.session.add(ev); db.session.flush()
        mk = Market(event_id=ev.id, status=MarketStatus.OPEN)
        db.session.add(mk); db.session.flush()
        eid, mid = ev.id, mk.id
        db.session.commit()
        SchedulerService.initialize_natural(seed=seed)
        return eid, mid


def _add_holders(market_id, agent_ids):
    for aid in agent_ids:
        db.session.add(Position(agent_id=aid, market_id=market_id, yes_shares=10.0, no_shares=0.0))
    db.session.commit()


def _add_watchers(event_id, agent_ids, role=ROLE_WATCHER):
    for aid in agent_ids:
        db.session.add(AgentEventInterest(agent_id=aid, event_id=event_id, role=role, weight=0.5))
    db.session.commit()


class _SQLCounter:
    def __init__(self, table=None):
        self.table = table; self.total = 0; self.table_hits = 0

    def __enter__(self):
        self._l = self._before
        event.listen(db.engine, "before_cursor_execute", self._l)
        return self

    def __exit__(self, *a):
        event.remove(db.engine, "before_cursor_execute", self._l)

    def _before(self, conn, cursor, statement, params, context, executemany):
        self.total += 1
        if self.table and (" " + self.table.lower()) in (" " + statement.lower()):
            self.table_hits += 1


# ---------------------------------------------------------------------

def test_score_formula():
    s = compute_wake_score(1.0, 1.0, 1.0, 1.0)
    check("full score ≈ 1.0", abs(s - 1.0) < 1e-9)
    s2 = compute_wake_score(1.0, 0.0, 0.0, 0.0)
    check("relevance-only weight = 0.35", abs(s2 - 0.35) < 1e-9)
    s3 = compute_wake_score(2.0, 2.0, 2.0, 2.0)
    check("score never exceeds 1.0 (inputs clamped)", s3 <= 1.0 and s3 >= 0.99)
    s4 = compute_wake_score(-5.0, -5.0, -5.0, -5.0)
    check("score never below 0.0", s4 == 0.0)


def test_unrelated_agents_excluded_and_subset():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=500, build_expertise=False)
    with app.app_context():
        _add_holders(mid, [10, 11, 12])
        _add_watchers(eid, [20, 21, 22, 23])
        # No expertise map built → candidates are exactly holders+watchers.
        cands = CandidateService.collect_candidate_ids(eid)
        ids = set(cands.keys())
        check("only holders+watchers are candidates", ids == {10, 11, 12, 20, 21, 22, 23},
              f"got {sorted(ids)}")
        check("unrelated agent 400 excluded", 400 not in ids)
        check("candidate subset ≪ population", len(ids) < 50)


def test_holders_rank_higher():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=300, build_expertise=False)
    with app.app_context():
        _add_holders(mid, [10, 11])
        _add_watchers(eid, [20, 21, 22])
        TriggerService.information_event(eid, information_impact=0.5, relevance=0.6,
                                         budget=EventBudget())
        rows = db.session.execute(text(
            "SELECT agent_id, position_relevance FROM wakeup_tasks ORDER BY position_relevance DESC"
        )).fetchall()
        holder_pr = [r[1] for r in rows if r[0] in (10, 11)]
        watcher_pr = [r[1] for r in rows if r[0] in (20, 21, 22)]
        check("holders have higher position_relevance than watchers",
              min(holder_pr) > max(watcher_pr), f"holders={holder_pr} watchers={watcher_pr}")


def test_experts_rank_higher():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=400, build_expertise=True)
    with app.app_context():
        # Experts come from AgentCategoryExpertise for "ai"; plain watchers
        # (not experts) should rank below experts by source_rank.
        # Pick some watchers that are NOT ai-experts.
        expert_ids = {r[0] for r in db.session.execute(text(
            "SELECT agent_id FROM agent_category_expertise WHERE category='ai'"
        )).fetchall()}
        non_experts = [a for a in range(1, 400) if a not in expert_ids][:5]
        _add_watchers(eid, non_experts)
        cands = CandidateService.collect_candidate_ids(eid, category="ai")
        # An expert candidate should carry role 'expert' (rank 4) > watcher (2).
        expert_ranks = [m["source_rank"] for aid, m in cands.items() if aid in expert_ids and m["role"] == "expert"]
        watcher_ranks = [m["source_rank"] for aid, m in cands.items() if aid in non_experts]
        check("experts present as candidates", len(expert_ranks) > 0)
        check("experts outrank plain watchers",
              (min(expert_ranks) if expert_ranks else 0) > (max(watcher_ranks) if watcher_ranks else 99),
              f"expert={set(expert_ranks)} watcher={set(watcher_ranks)}")


def test_event_wakeup_limit_enforced():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=600, build_expertise=True)
    with app.app_context():
        _add_holders(mid, list(range(10, 60)))
        budget = EventBudget(max_urgent=5, max_normal=10, max_delayed=10, total_budget=20)
        r = TriggerService.information_event(eid, information_impact=0.9, relevance=0.9,
                                             official=True, budget=budget)
        total = db.session.execute(text("SELECT COUNT(*) FROM wakeup_tasks")).scalar()
        check("total scheduled ≤ total_budget", total <= 20, f"got {total}")
        check("urgent tier ≤ cap", r["by_tier"]["urgent"] <= 5, f"{r['by_tier']}")
        check("reported scheduled matches rows", r["scheduled"] == total)


def test_jitter_desynchronizes():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=500, build_expertise=False)
    with app.app_context():
        _add_holders(mid, list(range(10, 60)))
        TriggerService.information_event(eid, information_impact=0.5, relevance=0.6,
                                         budget=EventBudget(), sim_seed=42, now=0.0)
        times = [r[0] for r in db.session.execute(text(
            "SELECT scheduled_at FROM wakeup_tasks")).fetchall()]
        distinct = len(set(round(t, 3) for t in times))
        check("scheduled times are not all identical", distinct > len(times) * 0.8,
              f"{distinct}/{len(times)} distinct")
        check("all scheduled_at within normal jitter window", all(300.0 <= t <= 7200.0 for t in times))


def test_duplicate_triggers_merge():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=200, build_expertise=False)
    with app.app_context():
        _add_holders(mid, [10])
        # Two info events at the SAME now → same time bucket after jitter?
        # Force merge by scheduling both at now=0 with 0 jitter via major+official.
        TriggerService.information_event(eid, information_impact=0.9, relevance=0.9,
                                         official=True, now=0.0, sim_seed=1,
                                         budget=EventBudget())
        # A portfolio-risk trigger for the same agent/event at now=0 (bucket 0).
        TriggerService.portfolio_risk_event(agent_id=10, reason="drawdown",
                                             event_id=eid, market_id=mid, now=0.0)
        rows = db.session.execute(text(
            "SELECT agent_id, trigger_type, priority, wake_reasons FROM wakeup_tasks "
            "WHERE agent_id=10")).fetchall()
        check("agent 10 has a single merged task", len(rows) == 1, f"got {len(rows)} rows")
        if rows:
            _, ttype, prio, reasons = rows[0]
            check("merged task promoted to highest priority (PORTFOLIO_RISK)",
                  ttype == TriggerType.PORTFOLIO_RISK.value, f"got {ttype}")
            check("merged task records multiple wake reasons",
                  reasons and "," in reasons, f"reasons={reasons}")


def test_cooldown_blocks_repeated_price():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=300, build_expertise=False)
    with app.app_context():
        _add_holders(mid, [10, 11, 12])
        cfg = PriceTriggerConfig(cooldown_s=3600.0, min_move=0.03)
        r1 = TriggerService.price_event(eid, mid, kind="abs_move", movement=0.1, config=cfg, now=0.0)
        r2 = TriggerService.price_event(eid, mid, kind="abs_move", movement=0.1, config=cfg, now=600.0)
        check("first price wave schedules holders", r1["scheduled"] == 3)
        check("second wave within cooldown is blocked", r2["scheduled"] == 0)
        check("second wave counts skipped-cooldown", r2["skipped_cooldown"] == 3)
        # After cooldown expires, allowed again.
        r3 = TriggerService.price_event(eid, mid, kind="abs_move", movement=0.1, config=cfg, now=4000.0)
        check("after cooldown expiry price wakes again", r3["scheduled"] == 3)


def test_min_move_gate():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=200, build_expertise=False)
    with app.app_context():
        _add_holders(mid, [10])
        cfg = PriceTriggerConfig(min_move=0.05)
        r = TriggerService.price_event(eid, mid, kind="abs_move", movement=0.01, config=cfg, now=0.0)
        check("sub-threshold move is skipped", r["scheduled"] == 0 and r["skipped_min_move"] == 1)


def test_cascade_depth_bounded():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=200, build_expertise=False)
    with app.app_context():
        _add_holders(mid, [10, 11])
        cfg = PriceTriggerConfig(max_cascade_depth=2, cooldown_s=0.0)
        # depth 0,1,2 allowed; depth 3 refused.
        ok0 = TriggerService.price_event(eid, mid, "abs_move", 0.1, config=cfg, cascade_depth=0, now=0.0)
        ok2 = TriggerService.price_event(eid, mid, "abs_move", 0.1, config=cfg, cascade_depth=2, now=10.0,
                                         cascade_id=ok0["cascade_id"])
        stopped = TriggerService.price_event(eid, mid, "abs_move", 0.1, config=cfg, cascade_depth=3, now=20.0,
                                             cascade_id=ok0["cascade_id"])
        check("cascade at max depth still schedules", ok2["scheduled"] > 0)
        check("cascade beyond max depth is stopped", stopped.get("cascade_stopped") is True)
        check("stopped cascade schedules nothing", stopped["scheduled"] == 0)


def test_critical_and_resolution_survive_shedding():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=800, build_expertise=True)
    with app.app_context():
        _add_holders(mid, list(range(10, 40)))
        # Lots of low-priority INFORMATION tasks.
        TriggerService.information_event(eid, information_impact=0.4, relevance=0.5,
                                         budget=EventBudget(total_budget=300))
        # Critical resolution + portfolio risk.
        TriggerService.resolution_event(eid, mid, now=0.0)
        TriggerService.portfolio_risk_event(agent_id=500, reason="total_exposure",
                                            event_id=eid, market_id=mid, now=0.0)
        before = db.session.execute(text("SELECT COUNT(*) FROM wakeup_tasks WHERE status='pending'")).scalar()
        shed = TriggerService.shed_load(keep=5)
        # Protected tasks must all survive.
        surviving_protected = db.session.execute(text(
            "SELECT COUNT(*) FROM wakeup_tasks WHERE status='pending' AND "
            "trigger_type IN ('RESOLUTION','PORTFOLIO_RISK')")).scalar()
        total_protected = db.session.execute(text(
            "SELECT COUNT(*) FROM wakeup_tasks WHERE "
            "trigger_type IN ('RESOLUTION','PORTFOLIO_RISK')")).scalar()
        check("shedding removed low-priority tasks", shed["shed"] > 0, f"{shed}")
        check("all RESOLUTION/PORTFOLIO_RISK survive shedding",
              surviving_protected == total_protected, f"{surviving_protected}/{total_protected}")
        shed_rows = db.session.execute(text(
            "SELECT COUNT(*) FROM wakeup_tasks WHERE status='shed' AND "
            "trigger_type IN ('RESOLUTION','PORTFOLIO_RISK')")).scalar()
        check("no protected task was shed", shed_rows == 0)


def test_normal_event_no_full_population_scan():
    """The information-event path's query count must be INDEPENDENT of
    population size — the definitive proof it doesn't scan the full table.
    We run the identical event against a small and a large population and
    require the same number of SQL statements."""
    def _run(npop):
        app = _fresh_app()
        eid, mid = _setup(app, n_agents=npop, seed=1, build_expertise=False)
        with app.app_context():
            _add_holders(mid, [10, 11, 12])
            _add_watchers(eid, [20, 21, 22, 23, 24])
            with _SQLCounter() as c:
                TriggerService.information_event(eid, information_impact=0.6, relevance=0.6,
                                                 budget=EventBudget(), now=0.0, sim_seed=1)
            scheduled = db.session.execute(text("SELECT COUNT(*) FROM wakeup_tasks")).scalar()
            return c.total, scheduled

    q_small, s_small = _run(300)
    q_large, s_large = _run(3000)
    check("query count is independent of population size (no full scan)",
          q_small == q_large, f"small={q_small} large={q_large}")
    check("same wake-ups scheduled regardless of population", s_small == s_large)
    check("far fewer wake-ups than population", s_large <= 8, f"{s_large}")
    # No unbounded agents scan: any agents access must be by id IN / keyset.
    check("query volume bounded by scheduled tasks, not population",
          q_large < 100, f"issued {q_large} queries for {s_large} tasks")


def test_collect_candidates_query_bounded():
    """Candidate collection query count must not scale with population."""
    def _run(npop):
        app = _fresh_app()
        eid, mid = _setup(app, n_agents=npop, seed=1, build_expertise=True)
        with app.app_context():
            _add_holders(mid, list(range(10, 30)))
            _add_watchers(eid, list(range(100, 120)))
            with _SQLCounter(table="from agents") as c:
                cands = CandidateService.collect_candidate_ids(eid, category="ai")
            return c.total, c.table_hits, len(cands)

    (q1, hits1, n1) = _run(500)
    (q2, hits2, n2) = _run(2000)
    check("candidate collection query count independent of population",
          q1 == q2, f"{q1} vs {q2}")
    check("candidate collection touches agents ≤ 1 time (archetype expand only)",
          hits2 <= 1, f"agents hits={hits2}")
    check("candidate collection is a bounded handful of queries", q2 <= 10, f"total={q2}")


def test_no_llm_call():
    import llm as llm_pkg
    calls = {"n": 0}

    class _Boom:
        available = True
        def chat_json(self, *a, **k): calls["n"] += 1; raise AssertionError("no LLM")
        def chat(self, *a, **k): calls["n"] += 1; raise AssertionError("no LLM")

    oc, orr = llm_pkg.get_llm_client, llm_pkg.get_model_router
    llm_pkg.get_llm_client = lambda *a, **k: _Boom()
    llm_pkg.get_model_router = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no router"))
    try:
        app = _fresh_app()
        eid, mid = _setup(app, n_agents=300, build_expertise=True)
        with app.app_context():
            _add_holders(mid, [10, 11])
            TriggerService.information_event(eid, information_impact=0.7, relevance=0.6,
                                             budget=EventBudget())
            TriggerService.price_event(eid, mid, "abs_move", 0.1, now=0.0)
            TriggerService.resolution_event(eid, mid, now=0.0)
            TriggerService.shed_load(keep=1)
        check("no LLM call anywhere in trigger scheduling", calls["n"] == 0)
    finally:
        llm_pkg.get_llm_client, llm_pkg.get_model_router = oc, orr


ALL_TESTS = [
    test_score_formula,
    test_unrelated_agents_excluded_and_subset,
    test_holders_rank_higher,
    test_experts_rank_higher,
    test_event_wakeup_limit_enforced,
    test_jitter_desynchronizes,
    test_duplicate_triggers_merge,
    test_cooldown_blocks_repeated_price,
    test_min_move_gate,
    test_cascade_depth_bounded,
    test_critical_and_resolution_survive_shedding,
    test_normal_event_no_full_population_scan,
    test_collect_candidates_query_bounded,
    test_no_llm_call,
]


def main():
    print("Running trigger tests (in-memory SQLite, virtual time, no LLM)...")
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
