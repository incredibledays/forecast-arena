"""Idempotent + fast schema-creation guard.

`db.create_all()` on every service call fires a `has_table` PRAGMA per
model in the metadata (SQLite) or an `information_schema` lookup
(Postgres). Under a hot-path drain (10k+ tasks), the profiler shows
`create_all` reflection consuming ~25% of wall time in a real run —
purely wasted, since the schema does not change during a run.

`ensure_created(bind_key=None)` is a memoizing wrapper that runs
`db.create_all()` at most once per (bind, engine) — subsequent calls are
zero-cost. Services swap `db.create_all()` for this in their
`ensure_schema()` bootstraps.
"""

from __future__ import annotations

import threading
from typing import Set, Tuple

from models.database import db


_lock = threading.Lock()
_done: Set[Tuple[int, object]] = set()


def ensure_created() -> None:
    """Run db.create_all() at most once per (engine) per process.

    Cheap when the schema is already up: a lock check + set membership.
    Safe to call from any service's `ensure_schema()`; the actual DDL
    fires only on the FIRST call for a given engine.
    """
    try:
        engine = db.engine
    except Exception:  # noqa: BLE001 — no app context yet
        return
    key = (id(engine), None)
    if key in _done:
        return
    with _lock:
        if key in _done:
            return
        db.create_all()
        _done.add(key)


def reset_for_tests() -> None:
    """Forget the memoized state — used by test fixtures that swap engines."""
    with _lock:
        _done.clear()


__all__ = ["ensure_created", "reset_for_tests"]
