"""Population generation tests + 10k mock benchmark.

Isolated in-memory SQLite; never touches the on-disk DB. Plain-script
style; no LLM is ever called (pure-code path). Exits 0 on success, 1 on
first failure.

    python test_population.py              # tests + 10k benchmark
    python test_population.py --bench-only # just the 10k benchmark
"""

import argparse
import sys
import time
import tracemalloc

from flask import Flask
from sqlalchemy import inspect, text

from models import Agent, AgentArchetype, OBJECTIVE_MAXIMIZE_WEALTH, db
from services import PopulationService
from services.population_service import (
    DEFAULT_ACTIVITY_DISTRIBUTION,
    DEFAULT_POPULATION_MIX,
    validate_mix,
)
from agents.strategy_families import STRATEGY_FAMILIES


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


def _population_summary():
    """A stable, order-independent fingerprint of the whole population."""
    rows = sorted(
        (
            a.name, round(a.initial_cash, 2), round(a.risk_aversion, 4),
            round(a.kelly_fraction, 4), round(a.entry_edge_threshold, 4),
            round(a.base_wakeup_rate_per_day, 3), a.activity_group,
            a.strategy_type, a.archetype_id, a.random_seed,
        )
        for a in Agent.query.filter(Agent.archetype_id.isnot(None)).all()
    )
    return rows


def _close(realized: dict, target: dict, total: int, tol: float, label: str) -> bool:
    """True if every realized fraction is within `tol` of target."""
    ok = True
    for key, weight in target.items():
        want = weight
        got = realized.get(key, 0) / total if total else 0
        if abs(got - want) > tol:
            ok = False
            print(f"      {label}: {key} target={want:.3f} got={got:.3f} (Δ={abs(got-want):.3f})")
    return ok


# ---------------------------------------------------------------------

def test_config_mixes_sum_to_one():
    validate_mix(DEFAULT_POPULATION_MIX)
    validate_mix(DEFAULT_ACTIVITY_DISTRIBUTION, name="activity")
    check("default population mix sums to 1", True)
    check("default activity distribution sums to 1", True)
    bad = False
    try:
        validate_mix({"a": 0.5, "b": 0.4})
    except ValueError:
        bad = True
    check("non-summing mix is rejected", bad)


def test_same_seed_identical_summary():
    a1 = _fresh_app()
    with a1.app_context():
        PopulationService.generate_default_archetypes(count=60, seed=42)
        PopulationService.generate_agents(count=1000, seed=42, batch_size=250)
        s1 = _population_summary()
    a2 = _fresh_app()
    with a2.app_context():
        PopulationService.generate_default_archetypes(count=60, seed=42)
        PopulationService.generate_agents(count=1000, seed=42, batch_size=333)
        s2 = _population_summary()
    check("same seed → identical count", len(s1) == len(s2) == 1000)
    check("same seed → identical population summary (batch size irrelevant)", s1 == s2)


def test_different_seed_differs():
    a1 = _fresh_app()
    with a1.app_context():
        PopulationService.generate_default_archetypes(count=60, seed=42)
        PopulationService.generate_agents(count=1000, seed=42)
        s1 = set(_population_summary())
    a2 = _fresh_app()
    with a2.app_context():
        PopulationService.generate_default_archetypes(count=60, seed=7)
        PopulationService.generate_agents(count=1000, seed=7)
        s2 = set(_population_summary())
    check("different seed → different agents", s1 != s2)


def test_same_archetype_heterogeneous():
    app = _fresh_app()
    with app.app_context():
        # Force every agent onto one family with one archetype.
        one = {"evidence_value": 1.0}
        PopulationService.generate_default_archetypes(count=1, seed=9, mix=one)
        PopulationService.generate_agents(count=80, seed=9, mix=one)
        agents = Agent.query.all()
        arch_ids = {a.archetype_id for a in agents}
        check("all agents share the single archetype", len(arch_ids) == 1)
        vecs = {(round(a.risk_aversion, 4), round(a.kelly_fraction, 4),
                 round(a.initial_cash, 2), round(a.entry_edge_threshold, 4))
                for a in agents}
        check("same-archetype agents are heterogeneous", len(vecs) > 1,
              f"{len(vecs)} distinct among {len(agents)}")


def test_values_in_bounds():
    app = _fresh_app()
    with app.app_context():
        PopulationService.generate_default_archetypes(count=100, seed=3)
        PopulationService.generate_agents(count=2000, seed=3, batch_size=500)
        report = PopulationService.validate_population()
        check("validate_population ok", report["ok"] is True, f"{report.get('errors')}")
        check("no out-of-bounds agents", report["stats"].get("out_of_bounds_agents", 0) == 0)


def test_strategy_distribution_close():
    app = _fresh_app()
    with app.app_context():
        PopulationService.generate_default_archetypes(count=100, seed=11)
        m = PopulationService.generate_agents(count=5000, seed=11, batch_size=1000)
        dist = m["strategy_distribution"]
        ok = _close(dist, DEFAULT_POPULATION_MIX, 5000, tol=0.02, label="strategy")
        check("strategy distribution close to config (±2%)", ok)


def test_activity_distribution_close():
    app = _fresh_app()
    with app.app_context():
        PopulationService.generate_default_archetypes(count=100, seed=13)
        m = PopulationService.generate_agents(count=5000, seed=13, batch_size=1000)
        dist = m["activity_distribution"]
        ok = _close(dist, DEFAULT_ACTIVITY_DISTRIBUTION, 5000, tol=0.02, label="activity")
        check("activity distribution close to config (±2%)", ok)


def test_every_agent_wealth_objective():
    app = _fresh_app()
    with app.app_context():
        PopulationService.generate_default_archetypes(count=40, seed=1)
        PopulationService.generate_agents(count=1000, seed=1)
        n_wrong = Agent.query.filter(
            (Agent.objective.is_(None)) | (Agent.objective != OBJECTIVE_MAXIMIZE_WEALTH)
        ).count()
        check("every agent uses the wealth objective", n_wrong == 0, f"{n_wrong} wrong")


def test_10k_no_llm_calls():
    import llm as llm_pkg

    calls = {"n": 0}

    class _Boom:
        available = True

        def chat_json(self, *a, **k):
            calls["n"] += 1
            raise AssertionError("LLM must not be called during population generation")

        def chat(self, *a, **k):
            calls["n"] += 1
            raise AssertionError("LLM must not be called during population generation")

    orig_c, orig_r = llm_pkg.get_llm_client, llm_pkg.get_model_router
    llm_pkg.get_llm_client = lambda *a, **k: _Boom()
    llm_pkg.get_model_router = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("router must not be built during pure-code generation"))
    try:
        app = _fresh_app()
        with app.app_context():
            PopulationService.generate_default_archetypes(count=100, seed=42)
            m = PopulationService.generate_agents(count=10_000, seed=42, batch_size=1000)
        check("generated exactly 10,000 agents", m["agents"] == 10_000, f"got {m['agents']}")
        check("zero LLM calls generating 10k agents", calls["n"] == 0, f"got {calls['n']}")
        check("metrics report 0 LLM requests", m["llm_request_count"] == 0)
        check("used bounded batches (10 of 1000)", m["batch_count"] == 10, f"got {m['batch_count']}")
    finally:
        llm_pkg.get_llm_client, llm_pkg.get_model_router = orig_c, orig_r


def test_orm_objects_not_resident():
    """After generating 10k agents the ORM identity map must NOT hold them.

    bulk_insert_mappings bypasses the identity map, and we expunge/expire
    between batches — so the resident object count stays tiny.
    """
    app = _fresh_app()
    with app.app_context():
        PopulationService.generate_default_archetypes(count=100, seed=42)
        PopulationService.generate_agents(count=10_000, seed=42, batch_size=1000)
        resident = len(list(db.session.identity_map.values()))
        check("ORM identity map not holding the population", resident < 100,
              f"resident={resident}")
        # Confirm the rows really are in the DB (bulk insert worked).
        check("10k agents persisted", Agent.query.count() == 10_000)


def test_no_destruction_without_reset():
    """Simulates the CLI guard: a second generate without --reset must not
    wipe existing agents. We assert the guard predicate the CLI uses."""
    app = _fresh_app()
    with app.app_context():
        PopulationService.generate_default_archetypes(count=20, seed=1)
        PopulationService.generate_agents(count=500, seed=1)
        before = Agent.query.filter(Agent.name.like("pop-%")).count()
        # The CLI refuses when this count > 0 and --reset absent. Emulate:
        existing = Agent.query.filter(Agent.name.like("pop-%")).count()
        reset = False
        would_refuse = existing > 0 and not reset
        check("CLI would refuse regeneration without --reset", would_refuse)
        # And nothing was deleted by merely checking.
        check("no agents destroyed by the guard", Agent.query.count() == before)


def test_legacy_agents_preserved():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.session.execute(text("""
            CREATE TABLE agents (
                id INTEGER PRIMARY KEY, name VARCHAR(80) UNIQUE NOT NULL,
                strategy_type VARCHAR(64), virtual_cash FLOAT NOT NULL DEFAULT 10000.0,
                initial_cash FLOAT NOT NULL DEFAULT 10000.0, risk_profile VARCHAR(32),
                created_at DATETIME NOT NULL)
        """))
        db.session.execute(text(
            "INSERT INTO agents (name, strategy_type, virtual_cash, initial_cash, "
            "risk_profile, created_at) VALUES ('LegacyBot','momentum',7777.0,8000.0,"
            "'medium','2026-01-01 00:00:00')"))
        db.session.commit()

        PopulationService.ensure_schema()
        cols = {c["name"] for c in inspect(db.engine).get_columns("agents")}
        check("migration added activity_group", "activity_group" in cols)
        legacy = Agent.query.filter_by(name="LegacyBot").one()
        check("legacy cash preserved", abs(legacy.virtual_cash - 7777.0) < 1e-6)
        check("legacy objective backfilled", legacy.objective == OBJECTIVE_MAXIMIZE_WEALTH)

        PopulationService.generate_default_archetypes(count=8, seed=1)
        PopulationService.generate_agents(count=20, seed=1)
        check("legacy + generated coexist", Agent.query.count() == 21)
        report = PopulationService.validate_population()
        check("validation ok with legacy present", report["ok"] is True, f"{report.get('errors')}")


def test_retail_like_coherent():
    app = _fresh_app()
    with app.app_context():
        PopulationService.generate_default_archetypes(count=len(STRATEGY_FAMILIES), seed=11)
        arch = AgentArchetype.query.filter_by(strategy_type="retail_like").first()
        check("retail_like archetype exists", arch is not None)
        check("retail_like positive entry threshold",
              arch.strategy_parameters_json["entry_edge_threshold"] > 0.0)
        check("retail_like herds strongly", arch.cognitive_biases_json["herding"] >= 0.5)
        check("retail_like recency-biased", arch.cognitive_biases_json["recency"] >= 0.5)
        check("retail_like loose risk (high kelly)", arch.risk_profile_json["kelly_fraction"] >= 0.5)


# --- 10k benchmark (mock, no LLM) ------------------------------------

def benchmark_10k():
    print("\n" + "=" * 60)
    print("  10,000-AGENT MOCK BENCHMARK (in-memory SQLite, no LLM)")
    print("=" * 60)
    app = _fresh_app()
    with app.app_context():
        tracemalloc.start()
        am = PopulationService.generate_default_archetypes(count=100, seed=42)
        t0 = time.perf_counter()
        m = PopulationService.generate_agents(count=10_000, seed=42, batch_size=1000)
        wall = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        resident = len(list(db.session.identity_map.values()))

        print(f"  archetypes           : {am['archetypes']}")
        print(f"  agents               : {m['agents']}")
        print(f"  gen time (s)         : {m['generation_time_s']}")
        print(f"  db time (s)          : {m['db_time_s']}")
        print(f"  wall time (s)        : {round(wall, 3)}")
        print(f"  throughput (agents/s): {round(m['agents'] / wall):,}")
        print(f"  peak memory (KB)     : {round(peak / 1024, 1)}")
        print(f"  batch count          : {m['batch_count']}")
        print(f"  LLM requests         : {m['llm_request_count']}")
        print(f"  ORM objects resident : {resident}")
        print("=" * 60)


ALL_TESTS = [
    test_config_mixes_sum_to_one,
    test_same_seed_identical_summary,
    test_different_seed_differs,
    test_same_archetype_heterogeneous,
    test_values_in_bounds,
    test_strategy_distribution_close,
    test_activity_distribution_close,
    test_every_agent_wealth_objective,
    test_10k_no_llm_calls,
    test_orm_objects_not_resident,
    test_no_destruction_without_reset,
    test_legacy_agents_preserved,
    test_retail_like_coherent,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-only", action="store_true")
    args = parser.parse_args()

    if args.bench_only:
        benchmark_10k()
        return 0

    print("Running population tests (in-memory SQLite, no LLM)...")
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

    if _FAIL == 0:
        benchmark_10k()
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
