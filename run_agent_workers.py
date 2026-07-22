"""Agent worker CLI — drains WakeUpTasks through the AgentWakeupProcessor.

Usage:
    # Single pass, up to 100 due tasks:
    python run_agent_workers.py --once --limit 100

    # Drain the queue in a loop (one worker):
    python run_agent_workers.py --loop --workers 1

    # Mock-LLM stress pass (no real model calls), 1000 tasks:
    python run_agent_workers.py --mock-llm --once --limit 1000

Options:
    --once / --loop        one pass, or keep going until the queue drains
    --limit N              max tasks claimed per pass (default 100)
    --batch-size N         micro-batch size per market (default 100)
    --workers N            number of worker threads (default 1)
    --mock-llm             never build a real LLM client/router; belief
                           updates use the deterministic offline stub
    --sim-seed N           deterministic personalization seed (default 0)
    --max-batches N        (loop only) cap on passes

SQLite note: SQLite serializes writers on a single database lock. Running
--workers > 1 against SQLite will contend and may raise "database is
locked". For real multi-worker throughput use PostgreSQL (the executor's
per-market lock becomes SELECT ... FOR UPDATE there). The CLI prints a
warning when --workers > 1 on a sqlite URL.
"""

import argparse
import os
import sys
import threading

from app import app  # Flask + DB context
from services import AgentWakeupProcessor, ProcessorConfig, SchedulerService


def _is_sqlite() -> bool:
    uri = app.config.get("SQLALCHEMY_DATABASE_URI", "") or os.getenv("DATABASE_URL", "")
    return uri.startswith("sqlite")


def _build_processor(args) -> AgentWakeupProcessor:
    llm_client = None
    router = None
    if not args.mock_llm:
        # Only wire real LLM handles when NOT in mock mode. A missing key
        # degrades gracefully inside BeliefService (offline stub).
        try:
            from llm import get_llm_client, get_model_router
            llm_client = get_llm_client()
            router = get_model_router()
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] LLM handles unavailable ({exc}); using offline stub",
                  file=sys.stderr)
    config = ProcessorConfig(
        micro_batch=args.batch_size,
        max_price_cascade_depth=args.max_cascade_depth,
        max_retries=args.max_retries,
        sim_seed=args.sim_seed,
    )
    return AgentWakeupProcessor(config=config, llm_client=llm_client, router=router)


def _print_metrics(title, m):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for k, v in m.as_dict().items():
        print(f"  {k:<34}: {v}")
    print("=" * 60)


def _run_once(args):
    with app.app_context():
        SchedulerService.ensure_schema()
        proc = _build_processor(args)
        metrics = proc.run_once(limit=args.limit)
        _print_metrics(f"WORKER --once (limit={args.limit})", metrics)
        return 0


def _run_loop(args):
    with app.app_context():
        SchedulerService.ensure_schema()
    # Each worker thread gets its own app context + processor.
    results = []
    lock = threading.Lock()

    def worker(idx):
        with app.app_context():
            proc = _build_processor(args)
            m = proc.run_loop(batch_limit=args.limit, max_batches=args.max_batches)
            with lock:
                results.append((idx, m))

    n = max(1, int(args.workers))
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for idx, m in sorted(results):
        _print_metrics(f"WORKER #{idx} --loop", m)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Drain agent wake-up tasks.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="single pass")
    mode.add_argument("--loop", action="store_true", help="drain until empty")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--sim-seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-cascade-depth", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    if args.workers > 1 and _is_sqlite():
        print(
            "[worker][WARN] --workers > 1 on SQLite: SQLite serializes "
            "writers on a single lock and may raise 'database is locked'. "
            "Use PostgreSQL for real multi-worker concurrency.",
            file=sys.stderr,
        )

    if args.loop:
        return _run_loop(args)
    # Default to --once when neither flag is given.
    return _run_once(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
