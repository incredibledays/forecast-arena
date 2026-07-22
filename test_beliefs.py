"""Belief-layer tests: archetype-level updates + lazy Agent materialization.

Isolated in-memory SQLite; deterministic; no real LLM (a mock client or
the offline stub). Exits 0 / 1.

    python test_beliefs.py
"""

import sys
from datetime import datetime, timedelta

from flask import Flask

from models import (
    db, Agent, Event, EventType, Market, MarketStatus, Position,
    AgentEventInterest, ROLE_WATCHER, ArchetypeBelief, AgentBelief,
    BELIEF_STATUS_MATERIALIZED,
)
from services import PopulationService, SchedulerService, EvidenceService, BeliefService
from services.belief_service import (
    personalize_one, personalize_batch, stable_logit, stable_sigmoid,
)
from llm import get_model_router


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
        SchedulerService.get_clock()
    return app


class MockProvider:
    enabled = True

    def __init__(self, items):
        self._items = items

    def set_items(self, items):
        self._items = items

    def search(self, q, max_results=5):
        return list(self._items)


_ONE_SIDED = [
    {"title": "Officials confirmed and approved the plan",
     "url": "https://federalreserve.gov/a",
     "content_summary": "Officials confirmed and approved; on track, agreed.",
     "relevance_score": 0.9, "published_date": "2026-07-15"},
]
_CONTESTED = _ONE_SIDED + [
    {"title": "Reuters: plan cancelled and denied",
     "url": "https://reuters.com/b",
     "content_summary": "Officials denied, cancelled and halted the plan; will not proceed.",
     "relevance_score": 0.9, "published_date": "2026-07-15"},
]


def _setup(app, n_agents=1000, n_arch=20, seed=42, holders=10, watchers=10,
           evidence=_CONTESTED, mix=None):
    with app.app_context():
        PopulationService.generate_default_archetypes(count=n_arch, seed=seed, mix=mix)
        PopulationService.generate_agents(count=n_agents, seed=seed, batch_size=500, mix=mix)
        ev = Event(title="Will the plan proceed?", description="d", category="markets",
                   event_type=EventType.BINARY,
                   close_time=datetime.utcnow() + timedelta(days=5),
                   resolution_source="federalreserve.gov")
        db.session.add(ev); db.session.flush()
        mk = Market(event_id=ev.id, status=MarketStatus.OPEN)
        db.session.add(mk); db.session.flush()
        eid, mid = ev.id, mk.id
        for a in range(1, holders + 1):
            db.session.add(Position(agent_id=a, market_id=mid, yes_shares=5.0, no_shares=0.0))
        for a in range(holders + 1, holders + watchers + 1):
            db.session.add(AgentEventInterest(agent_id=a, event_id=eid, role=ROLE_WATCHER, weight=0.5))
        db.session.commit()
        SchedulerService.initialize_natural(seed=seed)
        EvidenceService(search_provider=MockProvider(evidence), router=get_model_router()).refresh(eid)
        return eid, mid


# ---------------------------------------------------------------------

def test_pure_math():
    check("logit/sigmoid round-trip", abs(stable_sigmoid(stable_logit(0.7)) - 0.7) < 1e-9)
    lo, hi = stable_sigmoid(-800), stable_sigmoid(800)
    check("sigmoid stable + monotonic at extremes (no overflow)",
          0.0 <= lo < hi <= 1.0, f"lo={lo} hi={hi}")


def test_vector_matches_scalar():
    # Same rows through scalar reference and vectorized batch must agree.
    rows = [
        (10, 1, 111, 0.8, 0.7, 0.6, 0.2, 0.1),
        (11, 1, 222, 0.3, 0.4, 0.3, 0.5, -0.2),
        (12, 2, 333, 0.5, 0.5, 0.25, 0.25, 0.0),
    ]
    p_by_arch = {1: 0.62, 2: 0.4}
    batch = personalize_batch(p_by_arch, rows, sim_seed=42, event_id=7, bundle_version=3)
    for row, (cal_b, raw_b) in zip(rows, batch):
        cal_s = personalize_one(p_by_arch[row[1]], row, sim_seed=42, event_id=7, bundle_version=3)
        check(f"vector==scalar for agent {row[0]}", abs(cal_b - cal_s) < 1e-12,
              f"{cal_b} vs {cal_s}")


def test_probabilities_in_range():
    rows = [(i, 1, i * 7, 1.0, 1.0, 1.0, 0.0, 5.0) for i in range(50)]  # extreme inputs
    out = personalize_batch({1: 0.999}, rows, sim_seed=1, event_id=1, bundle_version=1)
    check("all calibrated probs within [0.01, 0.99]",
          all(0.01 <= c <= 0.99 for c, _ in out))


def test_one_archetype_belief_reused_by_many():
    app = _fresh_app()
    # Single-family mix so n_arch=1 is valid (one archetype for everyone).
    eid, mid = _setup(app, n_agents=500, n_arch=1, holders=30, watchers=0,
                      mix={"evidence_value": 1.0})
    with app.app_context():
        bs = BeliefService(router=get_model_router())
        m = bs.update_archetype_beliefs(eid)
        check("only one archetype belief created", m["relevant_archetypes"] == 1)
        check("one LLM request for the single archetype", m["archetype_llm_requests"] == 1)
        mm = bs.materialize_agent_beliefs(eid, sim_seed=42)
        n_arch = db.session.query(db.func.count(ArchetypeBelief.id)).scalar()
        check("30 agents share ONE archetype belief row", n_arch == 1 and mm["beliefs_persisted"] == 30,
              f"arch={n_arch} persisted={mm['beliefs_persisted']}")


def test_10k_no_per_agent_llm():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=10000, n_arch=100, holders=250, watchers=250)
    with app.app_context():
        bs = BeliefService(router=get_model_router())
        m = bs.update_archetype_beliefs(eid)
        mm = bs.materialize_agent_beliefs(eid, sim_seed=42)
        check("archetype LLM requests ≪ agents", m["archetype_llm_requests"] <= 100,
              f"{m['archetype_llm_requests']}")
        check("zero per-agent LLM calls (update)", m["individual_agent_llm_requests"] == 0)
        check("zero per-agent LLM calls (materialize)", mm["individual_agent_llm_requests"] == 0)
        check("batched into 10–30 archetype groups", m["batches"] <= 10, f"batches={m['batches']}")


def test_same_archetype_heterogeneous():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=500, n_arch=1, holders=40, watchers=0,
                      mix={"evidence_value": 1.0})
    with app.app_context():
        bs = BeliefService(router=get_model_router())
        bs.update_archetype_beliefs(eid)
        bs.materialize_agent_beliefs(eid, sim_seed=42)
        probs = [r[0] for r in db.session.query(AgentBelief.calibrated_probability).all()]
        check("same-archetype agents have heterogeneous beliefs",
              len(set(round(p, 6) for p in probs)) > 1, f"{len(set(probs))} distinct")


def test_same_seed_reproduces():
    app1 = _fresh_app(); e1, _ = _setup(app1, n_agents=300, n_arch=10, seed=42)
    with app1.app_context():
        bs = BeliefService(router=get_model_router())
        bs.update_archetype_beliefs(e1)
        bs.materialize_agent_beliefs(e1, sim_seed=42)
        p1 = {r[0]: round(r[1], 8) for r in db.session.query(
            AgentBelief.agent_id, AgentBelief.calibrated_probability).all()}
    app2 = _fresh_app(); e2, _ = _setup(app2, n_agents=300, n_arch=10, seed=42)
    with app2.app_context():
        bs = BeliefService(router=get_model_router())
        bs.update_archetype_beliefs(e2)
        bs.materialize_agent_beliefs(e2, sim_seed=42)
        p2 = {r[0]: round(r[1], 8) for r in db.session.query(
            AgentBelief.agent_id, AgentBelief.calibrated_probability).all()}
    check("same seed reproduces identical beliefs", p1 == p2)


def test_inactive_agents_no_rows():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=2000, n_arch=20, holders=15, watchers=15)
    with app.app_context():
        bs = BeliefService(router=get_model_router())
        bs.update_archetype_beliefs(eid)
        mm = bs.materialize_agent_beliefs(eid, sim_seed=42)
        n_rows = db.session.query(db.func.count(AgentBelief.id)).scalar()
        check("only eligible agents get belief rows (not all 2000)",
              n_rows == 30 and mm["eligible_agents"] == 30, f"rows={n_rows}")
        check("inactive-not-stored count is large", (2000 - n_rows) == 1970)


def test_materialized_matches_reconstructed():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=500, n_arch=15, holders=20, watchers=0)
    with app.app_context():
        bs = BeliefService(router=get_model_router())
        bs.update_archetype_beliefs(eid)
        bs.materialize_agent_beliefs(eid, sim_seed=42)
        mismatches = 0
        for aid in range(1, 21):
            row = db.session.query(AgentBelief).filter_by(agent_id=aid, event_id=eid).first()
            rec = bs.reconstruct_agent_belief(aid, eid, sim_seed=42)
            if row is None or rec is None:
                mismatches += 1
                continue
            if abs(row.calibrated_probability - rec["calibrated_probability"]) > 1e-9:
                mismatches += 1
        check("materialized == reconstructed within tolerance", mismatches == 0,
              f"{mismatches} mismatches")


def test_reconstruct_inactive_agent():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=500, n_arch=10, holders=5, watchers=0)
    with app.app_context():
        bs = BeliefService(router=get_model_router())
        bs.update_archetype_beliefs(eid)
        bs.materialize_agent_beliefs(eid, sim_seed=42)
        # An agent that is NOT eligible → no stored row, but reconstructable.
        inactive = 400
        stored = db.session.query(AgentBelief).filter_by(agent_id=inactive, event_id=eid).first()
        rec = bs.reconstruct_agent_belief(inactive, eid, sim_seed=42)
        check("inactive agent has no stored belief", stored is None)
        check("inactive agent belief is still reconstructable", rec is not None
              and 0.01 <= rec["calibrated_probability"] <= 0.99)


def test_batch_retry_only_failed():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=300, n_arch=8, holders=8, watchers=0,
                      mix={"evidence_value": 1.0})

    class FlakyLLM:
        """First call returns valid JSON for all but 2 archetypes; the
        repair call returns valid JSON for everyone. Records call count."""
        available = True

        def __init__(self):
            self.calls = 0

        def chat_json(self, messages, max_tokens=None):
            self.calls += 1
            # Parse the requested archetype ids out of the user message.
            user = messages[-1]["content"]
            import re
            ids = [int(x) for x in re.findall(r"\d+", user.split("archetype id:")[-1])]
            beliefs = []
            for i, aid in enumerate(ids):
                # On the FIRST attempt, drop the last two (invalid → retried).
                if self.calls == 1 and i >= len(ids) - 2:
                    beliefs.append({"archetype_id": aid, "garbage": True})  # invalid
                else:
                    beliefs.append({
                        "archetype_id": aid,
                        "posterior_probability_yes": 0.55,
                        "confidence": 0.6,
                        "reasoning_summary": "ok",
                        "key_evidence": [], "risk_factors": [],
                    })
            return {"beliefs": beliefs}

    with app.app_context():
        flaky = FlakyLLM()
        bs = BeliefService(llm_client=flaky, router=get_model_router())
        m = bs.update_archetype_beliefs(eid, batch_size=30)
        # 1 initial batch call + at least 1 repair; NOT one call per archetype.
        check("retry happened", m["retries"] >= 1, f"retries={m['retries']}")
        check("all archetypes ended with a belief", m["relevant_archetypes"] ==
              db.session.query(db.func.count(ArchetypeBelief.id)).scalar())
        check("no per-archetype fan-out (bounded calls)", flaky.calls <= 3,
              f"llm calls={flaky.calls}")
        check("no degraded (repair succeeded)", m["degraded"] == 0, f"degraded={m['degraded']}")


def test_routine_uses_fast_or_balanced():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=300, n_arch=10, holders=10, watchers=0,
                      evidence=_ONE_SIDED)  # one-sided → routine
    with app.app_context():
        bs = BeliefService(router=get_model_router())
        m = bs.update_archetype_beliefs(eid)
        tiers = set(m["tier_distribution"].keys())
        check("routine belief update uses FAST or BALANCED (not STRONG)",
              "STRONG" not in tiers, f"tiers={tiers}")


def test_major_conflict_uses_strong():
    app = _fresh_app()
    # Strong support + strong oppose, high impact → contested + high impact.
    strong = [
        {"title": "Fed officially confirmed and approved the rate cut",
         "url": "https://federalreserve.gov/press", "relevance_score": 0.98,
         "content_summary": "Officially confirmed and approved; on track, agreed, reached.",
         "published_date": "2026-07-16"},
        {"title": "Reuters: cut cancelled and denied",
         "url": "https://reuters.com/cut", "relevance_score": 0.97,
         "content_summary": "Officials denied, cancelled and halted the cut; will not proceed.",
         "published_date": "2026-07-16"},
    ]
    eid, mid = _setup(app, n_agents=300, n_arch=10, holders=10, watchers=0, evidence=strong)
    with app.app_context():
        bs = BeliefService(router=get_model_router())
        m = bs.update_archetype_beliefs(eid)
        check("major high-impact contradiction can use STRONG",
              "STRONG" in m["tier_distribution"], f"tiers={m['tier_distribution']}")


def test_full_orm_population_not_loaded():
    app = _fresh_app()
    eid, mid = _setup(app, n_agents=5000, n_arch=50, holders=100, watchers=100)
    with app.app_context():
        bs = BeliefService(router=get_model_router())
        bs.update_archetype_beliefs(eid)
        bs.materialize_agent_beliefs(eid, sim_seed=42)
        resident_agents = sum(1 for o in db.session.identity_map.values() if isinstance(o, Agent))
        check("no full ORM Agent population resident", resident_agents == 0,
              f"resident agents={resident_agents}")


ALL_TESTS = [
    test_pure_math,
    test_vector_matches_scalar,
    test_probabilities_in_range,
    test_one_archetype_belief_reused_by_many,
    test_10k_no_per_agent_llm,
    test_same_archetype_heterogeneous,
    test_same_seed_reproduces,
    test_inactive_agents_no_rows,
    test_materialized_matches_reconstructed,
    test_reconstruct_inactive_agent,
    test_batch_retry_only_failed,
    test_routine_uses_fast_or_balanced,
    test_major_conflict_uses_strong,
    test_full_orm_population_not_loaded,
]


def main():
    print("Running belief tests (in-memory SQLite, mock LLM, deterministic)...")
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
