"""Database bootstrap.

Exposes the shared `db` object and an `init_app(app)` helper that wires
SQLAlchemy into a Flask app using DATABASE_URL from environment / .env,
falling back to a local SQLite file.
"""

import os

from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

load_dotenv()

DEFAULT_DATABASE_URL = "sqlite:///forecast_arena.db"

db = SQLAlchemy()


def init_app(app):
    """Configure `app` for SQLAlchemy and bind the shared `db` to it."""
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)
    if database_url.startswith("sqlite"):
        engine_options = dict(app.config.get("SQLALCHEMY_ENGINE_OPTIONS") or {})
        connect_args = dict(engine_options.get("connect_args") or {})
        connect_args.setdefault(
            "timeout", float(os.getenv("SQLITE_BUSY_TIMEOUT", "30"))
        )
        engine_options["connect_args"] = connect_args
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = engine_options
    db.init_app(app)
    return db


# Composite indexes we want on EXISTING databases that predate the
# `__table_args__` declarations on the models. `db.create_all()` won't
# add indexes to already-created tables, so we CREATE INDEX IF NOT EXISTS
# them explicitly. Fresh DBs get them via the model declarations anyway.
_PERF_INDEXES = [
    ("ix_price_history_market_ts", "price_history", "(market_id, timestamp)"),
    ("ix_trades_market_created",   "trades",        "(market_id, created_at)"),
    ("ix_trades_agent_market_created", "trades",    "(agent_id, market_id, created_at)"),
    ("ix_evidence_event_retrieved", "evidence",     "(event_id, retrieved_at)"),
    ("ix_evidence_trade_id", "evidence", "(trade_id)"),
    ("ix_wakeup_status_updated", "wakeup_tasks", "(status, updated_at)"),
    ("ix_agent_decisions_created", "agent_decisions", "(created_at)"),
]

# Derived counters we maintain incrementally on the write path (see
# MarketService._record_trade). Existing DBs won't have these columns —
# we add them here with ALTER TABLE and backfill from actuals so the
# reads that use them are correct from the first request.
#
# Each entry: (table, column, ddl_add, backfill_sql).
_DERIVED_COLUMNS = [
    (
        "events", "total_volume",
        "ALTER TABLE events ADD COLUMN total_volume FLOAT NOT NULL DEFAULT 0.0",
        # Sum trades.amount joined through markets, grouped by event.
        """
        UPDATE events SET total_volume = COALESCE((
            SELECT SUM(t.amount)
            FROM trades t JOIN markets m ON t.market_id = m.id
            WHERE m.event_id = events.id
        ), 0.0)
        """,
    ),
    (
        "markets", "total_volume",
        "ALTER TABLE markets ADD COLUMN total_volume FLOAT NOT NULL DEFAULT 0.0",
        "UPDATE markets SET total_volume = COALESCE((SELECT SUM(amount) FROM trades WHERE trades.market_id = markets.id), 0.0)",
    ),
    (
        "agents", "total_trades",
        "ALTER TABLE agents ADD COLUMN total_trades INTEGER NOT NULL DEFAULT 0",
        "UPDATE agents SET total_trades = COALESCE((SELECT COUNT(*) FROM trades WHERE trades.agent_id = agents.id), 0)",
    ),
    (
        "agents", "non_hold_trades",
        "ALTER TABLE agents ADD COLUMN non_hold_trades INTEGER NOT NULL DEFAULT 0",
        """
        UPDATE agents SET non_hold_trades = COALESCE((
            SELECT COUNT(*) FROM trades
            WHERE trades.agent_id = agents.id AND trades.action != 'HOLD'
        ), 0)
        """,
    ),
    (
        "evidence", "trade_id",
        "ALTER TABLE evidence ADD COLUMN trade_id INTEGER",
        "UPDATE evidence SET trade_id = NULL WHERE 0",
    ),
]


def _column_exists(table: str, column: str) -> bool:
    """SQLite-only column-existence check via PRAGMA table_info."""
    rows = db.session.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return any(r[1] == column for r in rows)


def ensure_perf_indexes():
    """Migrate existing DBs to the indexes + counters the hot pages need.

    Idempotent — safe to call on every startup:
      * CREATE INDEX IF NOT EXISTS for the composite indexes on
        price_history and trades that turn full scans into range reads.
      * ALTER TABLE + backfill for the incrementally-maintained
        counters on `events` and `agents`. If the column already exists
        we skip both the ADD and the backfill (the write path keeps
        them fresh).

    SQLite-only for the column check (`PRAGMA table_info`). Postgres
    installations would need a separate migration path — but the app
    ships pointed at SQLite by default.
    """
    existing_tables = {
        row[0] for row in db.session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
    }
    # Derived columns first: some performance indexes target columns added
    # here (e.g. evidence.trade_id), so old DBs must ALTER before CREATE INDEX.
    # Only backfill on first migration — subsequent runs are no-ops because
    # the column already exists.
    for table, column, add_ddl, backfill_sql in _DERIVED_COLUMNS:
        if table not in existing_tables:
            continue
        if _column_exists(table, column):
            continue
        db.session.execute(text(add_ddl))
        db.session.execute(text(backfill_sql))

    for idx_name, table, cols in _PERF_INDEXES:
        if table not in existing_tables:
            continue
        db.session.execute(
            text(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} {cols}")
        )

    db.session.commit()
