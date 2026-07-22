"""Create tables, optionally wipe them, and seed starter data.

Usage:
    python init_db.py           # create tables + seed only if empty
    python init_db.py --reset   # drop + recreate all tables, then seed
"""

import argparse
import random
import sys
from datetime import datetime, timedelta

from flask import Flask
from sqlalchemy import inspect

from models import (
    Agent,
    Event,
    EventType,
    Market,
    MarketOutcome,
    MarketStatus,
    PriceHistory,
    db,
    init_app,
)


SEED_EVENTS = [
    # ---------------------------------------------------------------- AI
    # BINARY: a single YES/NO question.
    {
        "type": "BINARY",
        "title": "Will OpenAI publicly release a model branded 'GPT-6' before 2027-01-01?",
        "description": (
            "Resolves YES if OpenAI publishes an official announcement, blog "
            "post, or API entry for a generally-available model whose public "
            "name contains 'GPT-6' before 2027-01-01 00:00 UTC. A research "
            "preview, limited alpha, or unbranded 'next-generation' model "
            "does NOT count. Resolves NO otherwise."
        ),
        "category": "ai",
        "close_days": 175,   # trading stops ~1 week before deadline
        "resolution_days": 183,
        "resolution_source": "openai.com and OpenAI's official channels",
    },
    # CATEGORICAL: pick one of N mutually-exclusive candidates. Each
    # candidate becomes an independent binary Market under one Event.
    {
        "type": "CATEGORICAL",
        "title": "Which frontier AI lab ships GPT-5-equivalent first in 2026?",
        "description": (
            "Resolves to whichever lab first publicly releases a "
            "generally-available model matching or exceeding GPT-5-class "
            "capabilities in 2026. \"Other\" resolves if none of the "
            "listed labs is first."
        ),
        "category": "ai",
        "close_days": 175,
        "resolution_days": 183,
        "resolution_source": "public announcements from the lab in question",
        "candidates": ["OpenAI", "Anthropic", "Google DeepMind", "Meta", "Other"],
    },
    # SCALAR: numeric-value question with bucketed binary markets.
    # Each (lo, hi) becomes one binary Market; the bucket containing the
    # observed value on resolution day wins.
    {
        "type": "SCALAR",
        "title": "BTC/USD closing price on 2026-12-31 (UTC)?",
        "description": (
            "Coinbase spot BTC/USD price at 24h close on 2026-12-31 UTC. "
            "Resolves to whichever bucket contains that price. Buckets are "
            "half-open [lo, hi) except the last one, which is closed. "
            "Tail buckets `<X` and `X+` are supported."
        ),
        "category": "markets",
        "close_days": 175,
        "resolution_days": 183,
        "resolution_source": "Coinbase official 24h close",
        "scalar_unit": "USD",
        "buckets": [
            (None, 50_000),     # <50,000 tail
            (50_000, 80_000),
            (80_000, 100_000),
            (100_000, 120_000),
            (120_000, 150_000),
            (150_000, None),    # 150,000+ tail
        ],
    },
    # GROUPED: N binary markets that resolve independently. Any mix of
    # YES/NO across sub-markets is possible on resolution day.
    {
        "type": "GROUPED",
        "title": "2026 Fed rate cuts (each meeting independent)",
        "description": (
            "Independent YES/NO on whether the Fed cuts the target rate "
            "at each of the four listed meetings. Multiple sub-markets "
            "can resolve YES."
        ),
        "category": "markets",
        "close_days": 175,
        "resolution_days": 183,
        "resolution_source": "FOMC statements",
        "candidates": ["March", "June", "September", "December"],
    },
    # CONDITIONAL: a single binary market gated by a parent. The parent
    # is identified here by title; init_db does a second pass to link
    # the primary_market by id once the parents exist.
    {
        "type": "CONDITIONAL",
        "title": "If GPT-6 ships before 2027-01-01, will it be multimodal by default?",
        "description": (
            "Resolves YES if the released GPT-6 supports vision + audio "
            "by default (no add-on toggle). Refunded automatically if the "
            "parent GPT-6 market resolves NO."
        ),
        "category": "ai",
        "close_days": 175,
        "resolution_days": 183,
        "resolution_source": "OpenAI model card / release notes",
        "parent_seed_title": "Will OpenAI publicly release a model branded 'GPT-6' before 2027-01-01?",
        "parent_required_outcome": "YES",
    },
]


SEED_AGENTS = [
    # All 8 agents run the news_research strategy: retrieve via Tavily,
    # forecast via LLM, trade only when |p_llm - p_market| exceeds a
    # tier-dependent edge threshold. The tiers below map to risk_profile
    # inside NewsResearchAgent (_TIERS) — same code path, different
    # aggression. Rough shape: 3 aggressive / 3 balanced / 2 conservative.

    # High risk: edge=0.05, cash_frac=0.15, max_evid=8 — jumps on thin edges.
    {"name": "BreakingBull",  "strategy_type": "news_research", "risk_profile": "high"},
    {"name": "TabloidTrader", "strategy_type": "news_research", "risk_profile": "high"},
    {"name": "NewsHawk",      "strategy_type": "news_research", "risk_profile": "high"},

    # Medium risk: edge=0.08, cash_frac=0.10, max_evid=6 — the default profile.
    {"name": "SignalReader",  "strategy_type": "news_research", "risk_profile": "medium"},
    {"name": "MarketScribe",  "strategy_type": "news_research", "risk_profile": "medium"},
    {"name": "DeepReader",    "strategy_type": "news_research", "risk_profile": "medium"},

    # Low risk: edge=0.12, cash_frac=0.06, min_trade=$15, max_evid=5.
    {"name": "SlowThinker",   "strategy_type": "news_research", "risk_profile": "low"},
    {"name": "ContraQuant",   "strategy_type": "news_research", "risk_profile": "low"},
]


def _make_app():
    app = Flask(__name__)
    init_app(app)
    return app


def _assert_lmsr_schema_or_die():
    """Refuse to run against a pre-LMSR (or pre-retrieval-enrichment) schema.

    Two upgrades to detect here:
      * LMSR (added `liquidity_b` / `q_yes` / `q_no` on markets, `shares`
        on trades). Pre-LMSR trade/position data can't be safely mapped
        onto aggregate LMSR state.
      * Retrieval enrichment (added `published_date`, `source_domain`,
        `stance`, `stance_confidence`, `final_score` on evidence). Old
        rows lack the metadata by design — a `--reset` isn't strictly
        required to keep working, but future writes need the columns.

    In both cases we bail with a clear message rather than silently
    running against a stale schema.
    """
    inspector = inspect(db.engine)
    tables = set(inspector.get_table_names())
    if "markets" not in tables:
        return  # fresh install; create_all() will lay down the current schema
    market_cols = {col["name"] for col in inspector.get_columns("markets")}
    if "liquidity_b" not in market_cols:
        print(
            "[init_db] LMSR upgrade detected but existing markets table "
            "lacks `liquidity_b`. Re-run with --reset to wipe and re-seed:\n"
            "    python init_db.py --reset\n"
            "The pre-LMSR trade/position data cannot be safely mapped onto "
            "LMSR aggregate q_yes/q_no state; a clean re-seed is required.",
            file=sys.stderr,
        )
        sys.exit(2)

    if "evidence" in tables:
        ev_cols = {col["name"] for col in inspector.get_columns("evidence")}
        missing = {"published_date", "source_domain", "stance",
                   "stance_confidence", "final_score"} - ev_cols
        if missing:
            print(
                "[init_db] retrieval-enrichment upgrade detected but the "
                f"evidence table is missing columns: {sorted(missing)}.\n"
                "Re-run with --reset to pick up the new schema:\n"
                "    python init_db.py --reset\n"
                "Old evidence rows are safe to drop — they will be "
                "re-populated on the next `run_agents.py` round.",
                file=sys.stderr,
            )
            sys.exit(2)


def seed(rng=None):
    """Insert starter events, agents, and initial 50/50 price history."""
    rng = rng or random.Random(0xF0CACC1A)  # fixed seed for reproducible cash
    now = datetime.utcnow()

    # Events + Markets. Type-specific market fan-out:
    #   BINARY:      1 unlabeled market
    #   CATEGORICAL: 1 market per candidate
    #   GROUPED:     1 market per candidate (same shape as CATEGORICAL;
    #                the difference is in resolution semantics)
    #   SCALAR:      1 market per bucket, bucket_lo/hi filled
    #   CONDITIONAL: 1 unlabeled market; parent_market_id patched in a
    #                second pass below once the parent event exists
    conditional_pending = []  # [(child_market, parent_seed_title, req)]
    for spec in SEED_EVENTS:
        et = EventType(spec.get("type", "BINARY"))
        ev = Event(
            title=spec["title"],
            description=spec["description"],
            category=spec["category"],
            event_type=et,
            close_time=now + timedelta(days=spec["close_days"]),
            resolution_source=spec["resolution_source"],
            scalar_unit=spec.get("scalar_unit") if et == EventType.SCALAR else None,
        )
        db.session.add(ev)
        db.session.flush()  # get ev.id
        resolution_time = (
            now + timedelta(days=spec["resolution_days"])
            if spec.get("resolution_days") is not None
            else None
        )
        if et == EventType.BINARY:
            market_specs = [(None, None, None)]  # (label, lo, hi)
        elif et in (EventType.CATEGORICAL, EventType.GROUPED):
            labels = list(spec.get("candidates") or [])
            if len(labels) < 2:
                raise ValueError(
                    f"{et.value} seed {spec['title']!r} needs >=2 candidates"
                )
            market_specs = [(lab, None, None) for lab in labels]
        elif et == EventType.SCALAR:
            buckets = list(spec.get("buckets") or [])
            if len(buckets) < 2:
                raise ValueError(
                    f"SCALAR seed {spec['title']!r} needs >=2 buckets"
                )
            market_specs = []
            for lo, hi in buckets:
                # Tail buckets: (None, X) → "<X"; (X, None) → "≥X".
                if lo is None and hi is not None:
                    lbl = f"<{hi:,g}"
                elif hi is None and lo is not None:
                    lbl = f"≥{lo:,g}"
                else:
                    lbl = f"{lo:,g}–{hi:,g}"
                market_specs.append((lbl, lo, hi))
        else:  # CONDITIONAL — single market, parent linked in pass 2
            market_specs = [(None, None, None)]

        created_markets = []
        for label, bucket_lo, bucket_hi in market_specs:
            m = Market(
                event_id=ev.id,
                label=label,
                bucket_lo=bucket_lo,
                bucket_hi=bucket_hi,
                status=MarketStatus.OPEN,
                outcome=None,
                resolution_time=resolution_time,
            )
            db.session.add(m)
            created_markets.append(m)

        if et == EventType.CONDITIONAL:
            conditional_pending.append((
                created_markets[0],
                spec["parent_seed_title"],
                MarketOutcome(spec.get("parent_required_outcome", "YES")),
            ))
    db.session.flush()  # get market IDs

    # Second pass for CONDITIONAL: link each child's parent_market_id to
    # the primary_market of the event whose title matches.
    for child, parent_title, required in conditional_pending:
        parent_event = Event.query.filter_by(title=parent_title).first()
        if parent_event is None:
            raise ValueError(
                f"CONDITIONAL seed points at parent title "
                f"{parent_title!r} but no seed event has that title"
            )
        parent_market = parent_event.primary_market
        if parent_market is None:
            raise ValueError(
                f"parent event {parent_event.id} has no markets"
            )
        child.parent_market_id = parent_market.id
        child.parent_required_outcome = required
    db.session.flush()

    # Agents
    for spec in SEED_AGENTS:
        cash = round(rng.uniform(8000.0, 12000.0), 2)
        agent = Agent(
            name=spec["name"],
            strategy_type=spec["strategy_type"],
            virtual_cash=cash,
            initial_cash=cash,
            risk_profile=spec["risk_profile"],
        )
        db.session.add(agent)

    # Initial price history: 50/50 for every market
    for m in Market.query.all():
        db.session.add(
            PriceHistory(
                market_id=m.id,
                yes_price=0.5,
                no_price=0.5,
                timestamp=now,
            )
        )

    db.session.commit()


def main():
    parser = argparse.ArgumentParser(description="Initialize ForecastArena DB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables before seeding.",
    )
    args = parser.parse_args()

    app = _make_app()
    with app.app_context():
        if args.reset:
            print("[init_db] --reset: dropping all tables")
            db.drop_all()
        else:
            _assert_lmsr_schema_or_die()

        db.create_all()

        existing_events = Event.query.count()
        existing_agents = Agent.query.count()

        if args.reset or (existing_events == 0 and existing_agents == 0):
            print("[init_db] seeding starter events + agents")
            seed()
        else:
            print(
                f"[init_db] tables already populated "
                f"(events={existing_events}, agents={existing_agents}); "
                f"skipping seed. Use --reset to wipe."
            )

        print("[init_db] done.")
        print(f"  events: {Event.query.count()}")
        print(f"  markets: {Market.query.count()}")
        print(f"  agents: {Agent.query.count()}")
        print(f"  price_history rows: {PriceHistory.query.count()}")


if __name__ == "__main__":
    main()
