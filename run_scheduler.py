"""Drive the natural wake-up scheduler (this phase: natural wake-ups only).

Usage:
    # Sample the first natural wake-up for every active Agent (seed-driven):
    python run_scheduler.py initialize-natural --seed 42

    # Claim up to N due wake-ups at the current virtual time, printing the
    # work items (agent_id, sequence, wakeup_at, next_natural_wakeup_at):
    python run_scheduler.py due --limit 100

    # Advance the virtual simulation clock (no real sleeping):
    python run_scheduler.py advance-time --hours 12

    # Print scheduler state (clock, counts, earliest wake-up, sample):
    python run_scheduler.py inspect

Re-running initialize-natural is REFUSED unless --reset is given, since it
overwrites every Agent's schedule state.

The scheduler NEVER calls an LLM and NEVER loads the full population.
"""

import argparse
import json
import sys

from app import app  # Flask + DB context
from models import AgentScheduleState, db
from services import SchedulerService


def cmd_initialize_natural(args):
    SchedulerService.ensure_schema()
    existing = db.session.query(db.func.count(AgentScheduleState.agent_id)).scalar() or 0
    if existing > 0 and not args.reset:
        print(
            f"error: {existing} schedule row(s) already exist. Re-run with "
            f"--reset to overwrite.",
            file=sys.stderr,
        )
        return 2
    if args.reset and existing > 0:
        db.session.query(AgentScheduleState).delete(synchronize_session=False)
        db.session.commit()
        print(f"[reset] cleared {existing} schedule row(s)")

    metrics = SchedulerService.initialize_natural(
        seed=args.seed, batch_size=args.batch_size
    )
    print("\n" + "=" * 60)
    print("  INITIALIZE NATURAL WAKE-UPS")
    print("=" * 60)
    for k in ("scheduled_agents", "seed", "scheduler_version", "batch_count",
              "batch_size", "generation_time_s", "db_time_s", "llm_request_count"):
        print(f"  {k:<22}: {metrics[k]}")
    print("=" * 60)
    return 0


def cmd_due(args):
    SchedulerService.ensure_schema()
    work = SchedulerService.due(limit=args.limit, batch_size=args.batch_size)
    print(f"claimed {len(work)} due natural wake-up(s) at "
          f"virtual_time={SchedulerService.now():.1f}s")
    for item in work[: args.show]:
        print(
            f"  agent {item['agent_id']:>7} | seq {item['sequence']:>4} | "
            f"woke@ {item['wakeup_at']:.1f}s | next@ {item['next_natural_wakeup_at']:.1f}s"
        )
    if len(work) > args.show:
        print(f"  ... and {len(work) - args.show} more")
    return 0


def cmd_advance_time(args):
    SchedulerService.ensure_schema()
    new_t = SchedulerService.advance_time(
        seconds=args.seconds, hours=args.hours, days=args.days
    )
    print(f"virtual time advanced to {new_t:.1f}s ({new_t / 3600.0:.3f}h)")
    return 0


def cmd_inspect(args):
    SchedulerService.ensure_schema()
    print(json.dumps(SchedulerService.inspect(sample=args.sample), indent=2, default=str))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Natural wake-up scheduler.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("initialize-natural")
    p_init.add_argument("--seed", type=int, default=42)
    p_init.add_argument("--batch-size", type=int, default=1000)
    p_init.add_argument("--reset", action="store_true")
    p_init.set_defaults(func=cmd_initialize_natural)

    p_due = sub.add_parser("due")
    p_due.add_argument("--limit", type=int, default=100)
    p_due.add_argument("--batch-size", type=int, default=1000)
    p_due.add_argument("--show", type=int, default=20, help="how many items to print")
    p_due.set_defaults(func=cmd_due)

    p_adv = sub.add_parser("advance-time")
    p_adv.add_argument("--hours", type=float, default=0.0)
    p_adv.add_argument("--days", type=float, default=0.0)
    p_adv.add_argument("--seconds", type=float, default=0.0)
    p_adv.set_defaults(func=cmd_advance_time)

    p_ins = sub.add_parser("inspect")
    p_ins.add_argument("--sample", type=int, default=5)
    p_ins.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    with app.app_context():
        return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
