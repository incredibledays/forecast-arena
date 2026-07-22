"""Natural wake-up scheduler tests + query-count / memory checks.

Isolated in-memory SQLite; virtual time (no sleeping); no LLM. Plain
script style — exits 0 on success, 1 on first failure.

    python test_scheduler.py
"""

import statistics
import sys
import tracemalloc

from flask import Flask
from sqlalchemy import event, text

from models import (
    Agent, AgentScheduleState, SchedulerClock, db,
    OBJECTIVE_MAXIMIZE_WEALTH,
)
from services import PopulationService, SchedulerService
from services.scheduler_service import sample_wait_seconds, SECONDS_PER_DAY


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


def _fresh_app() -> Flask:
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return app


def _populate(app, n_arch=50, n_agents=1000, seed=42):
    with app.app_context():
        PopulationService.generate_default_archetypes(count=n_arch, seed=seed)
        PopulationService.generate_agents(count=n_agents, seed=seed, batch_size=500)


class _SQLCounter:
    """Context manager counting SQL statements + statements hitting a table."""

    def __init__(self, table=None):
        self.table = table
        self.total = 0
        self.table_hits = 0

    def __enter__(self):
        self._listener = self._before
        event.listen(db.engine, "before_cursor_execute", self._listener)
        return self

    def __exit__(self, *exc):
        event.remove(db.engine, "before_cursor_execute", self._listener)

    def _before(self, conn, cursor, statement, params, context, executemany):
        self.total += 1
        if self.table and self.table.lower() in statement.lower():
            self.table_hits += 1


# ---------------------------------------------------------------------

def test_pure_sampling_deterministic():
    a = sample_wait_seconds(5.0, sim_seed=42, agent_seed=123, sequence=0, version=1)
    b = sample_wait_seconds(5.0, sim_seed=42, agent_seed=123, sequence=0, version=1)
    c = sample_wait_seconds(5.0, sim_seed=42, agent_seed=123, sequence=1, version=1)
    d = sample_wait_seconds(5.0, sim_seed=99, agent_seed=123, sequence=0, version=1)
    check("same inputs → identical wait", a == b)
    check("different sequence → different wait", a != c)
    check("different sim seed → different wait", a != d)
    check("wait is positive", a > 0)


def test_wakeups_not_synchronized():
    app = _fresh_app()
    _populate(app, seed=42)
    with app.app_context():
        SchedulerService.initialize_natural(seed=42)
        rows = db.session.execute(text(
            "SELECT next_natural_wakeup_at FROM agent_schedule_state"
        )).fetchall()
        times = [r[0] for r in rows]
        distinct = len(set(round(t, 3) for t in times))
        check("wake-ups are not all identical", distinct > len(times) * 0.9,
              f"{distinct} distinct of {len(times)}")
        check("wake-ups have meaningful spread", statistics.pstdev(times) > 1.0,
              f"stdev={statistics.pstdev(times):.2f}")


def test_same_seed_reproduces():
    a1 = _fresh_app(); _populate(a1, seed=42)
    with a1.app_context():
        SchedulerService.initialize_natural(seed=42)
        t1 = {r[0]: round(r[1], 6) for r in db.session.execute(text(
            "SELECT agent_id, next_natural_wakeup_at FROM agent_schedule_state"
        )).fetchall()}
    a2 = _fresh_app(); _populate(a2, seed=42)
    with a2.app_context():
        SchedulerService.initialize_natural(seed=42)
        t2 = {r[0]: round(r[1], 6) for r in db.session.execute(text(
            "SELECT agent_id, next_natural_wakeup_at FROM agent_schedule_state"
        )).fetchall()}
    check("same seed reproduces identical timestamps", t1 == t2)


def test_different_seed_changes():
    a1 = _fresh_app(); _populate(a1, seed=42)
    with a1.app_context():
        SchedulerService.initialize_natural(seed=42)
        t1 = sorted(round(r[0], 6) for r in db.session.execute(text(
            "SELECT next_natural_wakeup_at FROM agent_schedule_state")).fetchall())
    # Same population, different SCHEDULER seed → different wake-ups.
    a2 = _fresh_app(); _populate(a2, seed=42)
    with a2.app_context():
        SchedulerService.initialize_natural(seed=777)
        t2 = sorted(round(r[0], 6) for r in db.session.execute(text(
            "SELECT next_natural_wakeup_at FROM agent_schedule_state")).fetchall())
    check("different scheduler seed changes timestamps", t1 != t2)


def test_one_next_wakeup_per_agent():
    app = _fresh_app(); _populate(app, n_agents=500, seed=5)
    with app.app_context():
        SchedulerService.initialize_natural(seed=5)
        n_rows = db.session.execute(text(
            "SELECT COUNT(*) FROM agent_schedule_state")).scalar()
        n_agents = db.session.execute(text("SELECT COUNT(*) FROM agents")).scalar()
        # exactly one schedule row per agent, ever.
        check("one schedule row per agent", n_rows == n_agents, f"{n_rows} vs {n_agents}")
        dup = db.session.execute(text(
            "SELECT COUNT(*) FROM (SELECT agent_id FROM agent_schedule_state "
            "GROUP BY agent_id HAVING COUNT(*) > 1)")).scalar()
        check("no agent has more than one wake-up row", dup == 0)
        # After processing a big drain, still one row each.
        SchedulerService.advance_time(days=10)
        SchedulerService.due(limit=100000, batch_size=500)
        n_rows2 = db.session.execute(text(
            "SELECT COUNT(*) FROM agent_schedule_state")).scalar()
        check("still one row per agent after processing", n_rows2 == n_agents)


def test_next_later_than_processed():
    app = _fresh_app(); _populate(app, n_agents=500, seed=7)
    with app.app_context():
        SchedulerService.initialize_natural(seed=7)
        SchedulerService.advance_time(days=5)
        work = SchedulerService.due(limit=100000, batch_size=500)
        check("some wake-ups were due", len(work) > 0)
        ok = all(w["next_natural_wakeup_at"] > w["wakeup_at"] for w in work)
        check("next wake-up is strictly later than processed one", ok)
        after = all(w["next_natural_wakeup_at"] > w["processed_at"] for w in work)
        check("next wake-up is after processing time", after)


def test_due_does_not_load_all_agents():
    app = _fresh_app(); _populate(app, n_agents=1000, seed=3)
    with app.app_context():
        SchedulerService.initialize_natural(seed=3)
        SchedulerService.advance_time(days=3)
        # The due path must not SELECT from the agents table at all.
        with _SQLCounter(table="from agents") as c:
            work = SchedulerService.due(limit=200, batch_size=100)
        check("due() never selects from agents table", c.table_hits == 0,
              f"agents SELECTs={c.table_hits}")
        check("due() produced work", len(work) > 0)
        # And it must not materialize ORM Agent objects.
        resident_agents = sum(
            1 for o in db.session.identity_map.values() if isinstance(o, Agent)
        )
        check("no ORM Agent objects resident after due()", resident_agents == 0)


def test_claim_is_safe_no_double_process():
    app = _fresh_app(); _populate(app, n_agents=800, seed=11)
    with app.app_context():
        SchedulerService.initialize_natural(seed=11)
        SchedulerService.advance_time(days=7)
        seen = set()
        total = 0
        # Drain in several bounded calls; an agent must never appear twice.
        for _ in range(5):
            work = SchedulerService.due(limit=1000, batch_size=200)
            for w in work:
                total += 1
                seen.add((w["agent_id"], w["sequence"]))
        check("no (agent, sequence) processed twice", len(seen) == total,
              f"{total} items, {len(seen)} unique")


def test_batches_release_session():
    app = _fresh_app(); _populate(app, n_agents=3000, seed=9)
    with app.app_context():
        SchedulerService.initialize_natural(seed=9, batch_size=500)
        # identity map empty right after init.
        check("identity map empty after init", len(list(db.session.identity_map.values())) == 0)
        SchedulerService.advance_time(days=10)
        SchedulerService.due(limit=100000, batch_size=500)
        resident = len(list(db.session.identity_map.values()))
        check("identity map empty after large due drain", resident == 0, f"resident={resident}")


def test_inactive_agents_do_not_wake():
    app = _fresh_app(); _populate(app, n_agents=400, seed=13)
    with app.app_context():
        SchedulerService.initialize_natural(seed=13)
        # Mark a chunk inactive directly on the schedule state.
        db.session.execute(text(
            "UPDATE agent_schedule_state SET status='inactive' "
            "WHERE agent_id IN (SELECT agent_id FROM agent_schedule_state "
            "ORDER BY agent_id LIMIT 100)"))
        db.session.commit()
        SchedulerService.advance_time(days=30)
        work = SchedulerService.due(limit=100000, batch_size=200)
        woke_ids = {w["agent_id"] for w in work}
        inactive_ids = {r[0] for r in db.session.execute(text(
            "SELECT agent_id FROM agent_schedule_state WHERE status='inactive'"
        )).fetchall()}
        check("no inactive agent woke", woke_ids.isdisjoint(inactive_ids),
              f"overlap={len(woke_ids & inactive_ids)}")
        check("active agents still woke", len(work) > 0)


def test_no_llm_call():
    import llm as llm_pkg
    calls = {"n": 0}

    class _Boom:
        available = True
        def chat_json(self, *a, **k):
            calls["n"] += 1; raise AssertionError("scheduler must not call LLM")
        def chat(self, *a, **k):
            calls["n"] += 1; raise AssertionError("scheduler must not call LLM")

    oc, orr = llm_pkg.get_llm_client, llm_pkg.get_model_router
    llm_pkg.get_llm_client = lambda *a, **k: _Boom()
    llm_pkg.get_model_router = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("scheduler must not build a router"))
    try:
        app = _fresh_app(); _populate(app, n_agents=500, seed=1)
        with app.app_context():
            m = SchedulerService.initialize_natural(seed=1)
            SchedulerService.advance_time(days=5)
            SchedulerService.due(limit=10000, batch_size=250)
            SchedulerService.inspect()
        check("no LLM call during schedule + due + inspect", calls["n"] == 0)
        check("init metrics report 0 LLM requests", m["llm_request_count"] == 0)
    finally:
        llm_pkg.get_llm_client, llm_pkg.get_model_router = oc, orr


def test_activity_groups_affect_rates():
    """Higher activity band ⇒ higher mean wake-up rate ⇒ earlier first wake."""
    app = _fresh_app(); _populate(app, n_agents=3000, seed=21)
    with app.app_context():
        SchedulerService.initialize_natural(seed=21)
        # Mean seconds-until-first-wake by band; ultra_active should be the
        # smallest (highest rate), very_low_frequency the largest.
        rows = db.session.execute(text(
            "SELECT a.activity_group, AVG(s.next_natural_wakeup_at) "
            "FROM agent_schedule_state s JOIN agents a ON a.id = s.agent_id "
            "GROUP BY a.activity_group")).fetchall()
        means = {r[0]: r[1] for r in rows}
        check("ultra_active wakes sooner than normal on average",
              means.get("ultra_active", 1e9) < means.get("normal", 0),
              f"{means}")
        check("very_low_frequency wakes later than normal on average",
              means.get("very_low_frequency", 0) > means.get("normal", 1e9),
              f"{means}")


def test_memory_flat_across_batches():
    app = _fresh_app()
    with app.app_context():
        PopulationService.generate_default_archetypes(count=100, seed=42)
        PopulationService.generate_agents(count=10000, seed=42, batch_size=1000)
        tracemalloc.start()
        SchedulerService.initialize_natural(seed=42, batch_size=1000)
        _, peak_init = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        SchedulerService.advance_time(days=2)
        work = SchedulerService.due(limit=100000, batch_size=1000)
        _, peak_due = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        resident = len(list(db.session.identity_map.values()))
        print(f"      init peak={peak_init/1024:.0f}KB  due peak={peak_due/1024:.0f}KB  "
              f"claimed={len(work)}  resident={resident}")
        check("init peak memory bounded (<50MB for 10k)", peak_init < 50 * 1024 * 1024)
        check("due peak memory bounded (<50MB for 10k)", peak_due < 50 * 1024 * 1024)
        check("no ORM objects resident after 10k due", resident == 0)


ALL_TESTS = [
    test_pure_sampling_deterministic,
    test_wakeups_not_synchronized,
    test_same_seed_reproduces,
    test_different_seed_changes,
    test_one_next_wakeup_per_agent,
    test_next_later_than_processed,
    test_due_does_not_load_all_agents,
    test_claim_is_safe_no_double_process,
    test_batches_release_session,
    test_inactive_agents_do_not_wake,
    test_no_llm_call,
    test_activity_groups_affect_rates,
    test_memory_flat_across_batches,
]


def main():
    print("Running scheduler tests (in-memory SQLite, virtual time, no LLM)...")
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
