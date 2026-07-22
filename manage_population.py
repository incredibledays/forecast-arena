"""Manage the Agent population: archetypes + generated agents at scale.

Usage:
    python manage_population.py generate-archetypes --count 100 --seed 42
    python manage_population.py generate-archetypes --count 100 --seed 42 --use-llm
    python manage_population.py generate-agents --count 10000 --seed 42 --batch-size 1000
    python manage_population.py validate

Destructive regeneration is REFUSED unless you pass --reset:
  * generate-archetypes --reset wipes archetypes AND generated (pop-*)
    agents (generated agents reference archetypes).
  * generate-agents --reset wipes only generated (pop-*) agents.
Legacy (non-population) agents are never touched by --reset.

Every command prints a metrics block: counts, generation/database time,
peak memory (when tracemalloc is available), strategy + activity
distribution, LLM request count, and batch count.
"""

import argparse
import json
import sys
import time
import tracemalloc

from app import app  # Flask + DB context
from models import Agent, AgentArchetype, db
from services import PopulationService
from services.population_service import (
    DEFAULT_ACTIVITY_DISTRIBUTION,
    DEFAULT_POPULATION_MIX,
)


def _reset_archetypes():
    n_agents = Agent.query.filter(Agent.name.like("pop-%")).delete(synchronize_session=False)
    n_arch = AgentArchetype.query.delete(synchronize_session=False)
    db.session.commit()
    print(f"[reset] deleted {n_arch} archetype(s) and {n_agents} generated agent(s)")


def _reset_agents():
    n_agents = Agent.query.filter(Agent.name.like("pop-%")).delete(synchronize_session=False)
    db.session.commit()
    print(f"[reset] deleted {n_agents} generated agent(s)")


def _print_metrics(title, metrics, peak_kb, wall_s):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  archetype count      : {metrics.get('archetypes', AgentArchetype.query.count())}")
    print(f"  agent count          : {metrics.get('agents', '-')}")
    print(f"  generation time (s)  : {metrics.get('generation_time_s', '-')}")
    print(f"  database time (s)    : {metrics.get('db_time_s', '-')}")
    print(f"  wall time (s)        : {round(wall_s, 4)}")
    print(f"  peak memory (KB)     : {peak_kb if peak_kb is not None else 'n/a'}")
    print(f"  LLM request count    : {metrics.get('llm_request_count', 0)}")
    if "llm_fallback_count" in metrics:
        print(f"  LLM fallback count   : {metrics['llm_fallback_count']}")
    print(f"  batch count          : {metrics.get('batch_count', '-')}")
    print(f"  batch size           : {metrics.get('batch_size', '-')}")
    if metrics.get("strategy_distribution"):
        print("  strategy distribution:")
        for k, v in sorted(metrics["strategy_distribution"].items()):
            print(f"      {k:<18} {v}")
    if metrics.get("activity_distribution"):
        print("  activity distribution:")
        for k, v in sorted(metrics["activity_distribution"].items()):
            print(f"      {k:<20} {v}")
    print("=" * 60)


def _run_with_metrics(title, fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    metrics = fn()
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _print_metrics(title, metrics, round(peak / 1024, 1), wall)
    return metrics


def cmd_generate_archetypes(args):
    PopulationService.ensure_schema()
    existing = AgentArchetype.query.count()
    if existing > 0 and not args.reset:
        print(f"error: {existing} archetype(s) already exist. Re-run with --reset.",
              file=sys.stderr)
        return 2
    if args.reset:
        _reset_archetypes()

    if args.use_llm:
        fn = lambda: PopulationService.generate_llm_archetypes(count=args.count, seed=args.seed)
    else:
        fn = lambda: PopulationService.generate_default_archetypes(count=args.count, seed=args.seed)
    _run_with_metrics(f"ARCHETYPES (use_llm={args.use_llm})", fn)

    rep = PopulationService.validate_archetypes()
    if not rep["ok"]:
        print("archetype validation FAILED:", rep["errors"], file=sys.stderr)
        return 1
    return 0


def cmd_generate_agents(args):
    PopulationService.ensure_schema()
    existing = Agent.query.filter(Agent.name.like("pop-%")).count()
    if existing > 0 and not args.reset:
        print(f"error: {existing} generated agent(s) already exist. Re-run with --reset.",
              file=sys.stderr)
        return 2
    if args.reset:
        _reset_agents()

    if AgentArchetype.query.count() == 0:
        print("error: no archetypes exist. Run generate-archetypes first.", file=sys.stderr)
        return 2

    _run_with_metrics(
        "AGENTS",
        lambda: PopulationService.generate_agents(
            count=args.count, seed=args.seed, batch_size=args.batch_size,
            cash_base=args.cash_base,
        ),
    )
    return 0


def cmd_validate(args):
    PopulationService.ensure_schema()
    report = PopulationService.validate_population()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description="Manage the Agent population.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_arch = sub.add_parser("generate-archetypes")
    p_arch.add_argument("--count", type=int, default=100)
    p_arch.add_argument("--seed", type=int, default=42)
    p_arch.add_argument("--use-llm", action="store_true")
    p_arch.add_argument("--reset", action="store_true")
    p_arch.set_defaults(func=cmd_generate_archetypes)

    p_ag = sub.add_parser("generate-agents")
    p_ag.add_argument("--count", type=int, default=10000)
    p_ag.add_argument("--seed", type=int, default=42)
    p_ag.add_argument("--batch-size", type=int, default=1000)
    p_ag.add_argument("--cash-base", type=float, default=10000.0)
    p_ag.add_argument("--reset", action="store_true")
    p_ag.set_defaults(func=cmd_generate_agents)

    p_val = sub.add_parser("validate")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    with app.app_context():
        return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
