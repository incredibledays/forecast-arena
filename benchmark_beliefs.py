"""Belief-pipeline benchmark (mock LLM, in-memory, deterministic).

Builds a population, an event with mock evidence, runs the archetype-level
belief update and lazy Agent materialization, and reports the model-call
metrics the phase requires — proving zero per-Agent LLM calls at scale.

    python benchmark_beliefs.py --agents 10000 --archetypes 100 --mock-llm --seed 42

`--mock-llm` (default on) uses the deterministic offline belief stub so no
network/model is needed. Everything runs against a private in-memory DB.
"""

import argparse
import time
import tracemalloc
from datetime import datetime, timedelta

from flask import Flask

from models import (
    db, Event, EventType, Market, MarketStatus, Position,
    AgentEventInterest, ROLE_WATCHER, ArchetypeBelief, AgentBelief,
)
from services import PopulationService, SchedulerService, EvidenceService, BeliefService
from llm import get_model_router


class _MockProvider:
    enabled = True

    def __init__(self):
        self._items = [
            {"title": "Official confirmed and approved the plan",
             "url": "https://federalreserve.gov/press",
             "content_summary": "Officials confirmed and approved; on track.",
             "relevance_score": 0.95, "published_date": "2026-07-15"},
            {"title": "Report: plan delayed and denied",
             "url": "https://reuters.com/story",
             "content_summary": "Sources say the plan was delayed and denied.",
             "relevance_score": 0.85, "published_date": "2026-07-14"},
        ]

    def search(self, q, max_results=5):
        return list(self._items)


def _fresh_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return app


def main():
    ap = argparse.ArgumentParser(description="Belief pipeline benchmark.")
    ap.add_argument("--agents", type=int, default=10000)
    ap.add_argument("--archetypes", type=int, default=100)
    ap.add_argument("--mock-llm", action="store_true", default=True)
    ap.add_argument("--holders", type=int, default=500,
                    help="how many agents hold/watch the event (eligible set)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    app = _fresh_app()
    with app.app_context():
        t0 = time.perf_counter()
        PopulationService.generate_default_archetypes(count=args.archetypes, seed=args.seed)
        PopulationService.generate_agents(count=args.agents, seed=args.seed, batch_size=1000)
        pop_t = time.perf_counter() - t0

        ev = Event(title="Benchmark: will the plan proceed?", description="d",
                   category="markets", event_type=EventType.BINARY,
                   close_time=datetime.utcnow() + timedelta(days=5),
                   resolution_source="federalreserve.gov")
        db.session.add(ev); db.session.flush()
        mk = Market(event_id=ev.id, status=MarketStatus.OPEN)
        db.session.add(mk); db.session.flush()
        eid, mid = ev.id, mk.id
        # Make a subset eligible (holders + watchers).
        half = max(1, args.holders // 2)
        for a in range(1, half + 1):
            db.session.add(Position(agent_id=a, market_id=mid, yes_shares=5.0, no_shares=0.0))
        for a in range(half + 1, args.holders + 1):
            db.session.add(AgentEventInterest(agent_id=a, event_id=eid, role=ROLE_WATCHER, weight=0.5))
        db.session.commit()
        SchedulerService.initialize_natural(seed=args.seed)

        EvidenceService(search_provider=_MockProvider(), router=get_model_router()).refresh(eid)

        # Mock LLM: pass no client → deterministic offline stub (counts as
        # archetype-level requests, never per-agent).
        bs = BeliefService(llm_client=None, router=get_model_router())

        tracemalloc.start()
        t1 = time.perf_counter()
        arch_metrics = bs.update_archetype_beliefs(eid)
        arch_t = time.perf_counter() - t1

        t2 = time.perf_counter()
        mat_metrics = bs.materialize_agent_beliefs(eid, sim_seed=args.seed)
        mat_t = time.perf_counter() - t2
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_agents = args.agents
        n_arch_beliefs = db.session.query(db.func.count(ArchetypeBelief.id)).scalar()
        n_agent_beliefs = db.session.query(db.func.count(AgentBelief.id)).scalar()
        resident = len(list(db.session.identity_map.values()))

        print("\n" + "=" * 64)
        print("  BELIEF PIPELINE BENCHMARK (mock LLM, in-memory)")
        print("=" * 64)
        print(f"  agents                       : {args.agents}")
        print(f"  archetypes                   : {args.archetypes}")
        print(f"  eligible agents              : {mat_metrics['eligible_agents']}")
        print("  --- model-call metrics ---")
        print(f"  relevant archetypes          : {arch_metrics['relevant_archetypes']}")
        print(f"  ARCHETYPE LLM requests       : {arch_metrics['archetype_llm_requests']}")
        print(f"  individual Agent LLM requests: {arch_metrics['individual_agent_llm_requests'] + mat_metrics['individual_agent_llm_requests']}")
        print(f"  individual beliefs calculated: {mat_metrics['individual_beliefs_calculated']}")
        print(f"  beliefs persisted            : {mat_metrics['beliefs_persisted']}")
        print(f"  beliefs reconstructed-not-stored (inactive): {args.agents - mat_metrics['beliefs_persisted']}")
        print(f"  cache eligible               : {arch_metrics['cache_eligible']}")
        print(f"  model-tier distribution      : {arch_metrics['tier_distribution']}")
        print(f"  batches / retries / degraded : {arch_metrics['batches']} / {arch_metrics['retries']} / {arch_metrics['degraded']}")
        print("  --- persistence ---")
        print(f"  ArchetypeBelief rows         : {n_arch_beliefs}")
        print(f"  AgentBelief rows             : {n_agent_beliefs}")
        print("  --- performance ---")
        print(f"  population build (s)         : {round(pop_t, 3)}")
        print(f"  archetype update (s)         : {round(arch_t, 3)}")
        print(f"  materialization (s)          : {round(mat_t, 3)}")
        print(f"  peak memory (KB)             : {round(peak / 1024, 1)}")
        print(f"  ORM objects resident         : {resident}")
        print("=" * 64)
        per_agent = arch_metrics['individual_agent_llm_requests'] + mat_metrics['individual_agent_llm_requests']
        assert per_agent == 0, "per-agent LLM calls must be zero"
        print("  OK: zero per-Agent LLM calls.")


if __name__ == "__main__":
    main()
