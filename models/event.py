"""Event model: a topic container for one or more binary Markets.

An Event describes the real-world question ("Will X happen?", "Who wins
the 2028 election?", "What will BTC close at on Dec 31?"). Each Event
carries one or more Markets that are actually traded and resolved:

    BINARY       — 1 Market, label=None, UI shows plain YES/NO
    CATEGORICAL  — N Markets, one per candidate, mutually exclusive (P1)
    SCALAR       — N Markets, bucketed by value range (P2)
    GROUPED      — N Markets, independent resolution (P3)
    CONDITIONAL  — Markets with a parent dependency (P3)

For BINARY events, `event.status`, `event.outcome`, and
`event.primary_market` are derived helpers that keep the templates and
services largely unchanged while the underlying primitive migrates.
"""

import enum
from datetime import datetime

from models.database import db
from models.market import MarketStatus, MarketOutcome


class EventType(str, enum.Enum):
    BINARY = "BINARY"
    CATEGORICAL = "CATEGORICAL"
    SCALAR = "SCALAR"
    GROUPED = "GROUPED"
    CONDITIONAL = "CONDITIONAL"


class Event(db.Model):
    __tablename__ = "events"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    category = db.Column(db.String(64), nullable=True)
    event_type = db.Column(
        db.Enum(EventType, native_enum=False, length=16),
        default=EventType.BINARY,
        nullable=False,
    )
    close_time = db.Column(db.DateTime, nullable=True)
    resolution_source = db.Column(db.String(255), nullable=True)
    # Only populated for SCALAR events — display unit for the numeric
    # variable (e.g. "USD", "%", "cuts", "°C"). Not used in resolution.
    scalar_unit = db.Column(db.String(32), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Running sum of every Trade.amount on this event's markets. Maintained
    # incrementally by MarketService._record_trade so the index page's
    # per-event volume readout doesn't have to full-scan the trades table
    # on every hit. Backfilled from truth by `ensure_perf_indexes()` on
    # startup so DBs predating this column still get correct values.
    total_volume = db.Column(db.Float, default=0.0, nullable=False)

    # Relationships — trading state (trades/positions/price_history) hangs
    # off Market now, not Event directly. Evidence is topic-level and
    # stays on Event.
    markets = db.relationship(
        "Market",
        backref="event",
        lazy="dynamic",
        cascade="all, delete-orphan",
        order_by="Market.id",
    )
    evidence = db.relationship(
        "Evidence", backref="event", lazy="dynamic", cascade="all, delete-orphan"
    )

    # ---- Derived helpers (BINARY-friendly compatibility layer) ---------

    @property
    def primary_market(self):
        """The first Market on this event.

        Convenience for BINARY events (which have exactly one). For
        multi-market event types, callers should iterate `markets`
        explicitly — this property is only meaningful when the event is
        effectively binary.
        """
        return self.markets.order_by(None).order_by("id").first()

    @property
    def status(self):
        """Aggregate status of the event.

        RESOLVED  — every market is resolved
        CLOSED    — any market is closed but not all resolved
        OPEN      — otherwise (including the no-markets edge case)

        Returned as a `MarketStatus` so existing callers that read
        `event.status.value` (e.g. templates) keep working.
        """
        markets = list(self.markets)
        if not markets:
            return MarketStatus.OPEN
        if all(m.status == MarketStatus.RESOLVED for m in markets):
            return MarketStatus.RESOLVED
        if any(m.status == MarketStatus.CLOSED for m in markets):
            return MarketStatus.CLOSED
        return MarketStatus.OPEN

    @property
    def outcome(self):
        """BINARY: the primary market's outcome; other types: None.

        Templates/services that read `event.outcome.value` (e.g.
        `event_detail.html`) get the BINARY behavior they had before the
        refactor. Multi-market event types will surface per-market
        outcomes directly instead.
        """
        if self.event_type != EventType.BINARY:
            return None
        pm = self.primary_market
        return pm.outcome if pm is not None else None

    @property
    def winning_market(self):
        """CATEGORICAL / SCALAR: the (single) Market that resolved YES.

        GROUPED has no single winner (each market resolves independently),
        so it returns None. CONDITIONAL has one market — this returns it
        only if it resolved YES.
        """
        if self.event_type == EventType.GROUPED:
            return None
        if self.event_type == EventType.BINARY:
            return None
        for m in self.markets:
            if m.outcome == MarketOutcome.YES:
                return m
        return None

    @property
    def numeric_summary(self):
        """SCALAR: {min, max, unit, n_buckets, unbounded_below, unbounded_above};
        else None.

        Aggregates bucket bounds across all markets. Tail buckets are
        modelled as bucket_lo=None (unbounded below) or bucket_hi=None
        (unbounded above); `min` is the smallest finite bucket_lo (or
        None if there is a `<X` tail), `max` symmetrical.
        """
        if self.event_type != EventType.SCALAR:
            return None
        los = [m.bucket_lo for m in self.markets if m.bucket_lo is not None]
        his = [m.bucket_hi for m in self.markets if m.bucket_hi is not None]
        n_buckets = self.markets.count()
        unbounded_below = any(
            m.bucket_lo is None and m.bucket_hi is not None
            for m in self.markets
        )
        unbounded_above = any(
            m.bucket_hi is None and m.bucket_lo is not None
            for m in self.markets
        )
        return {
            "min": min(los) if los else None,
            "max": max(his) if his else None,
            "unit": self.scalar_unit,
            "n_buckets": n_buckets,
            "unbounded_below": unbounded_below,
            "unbounded_above": unbounded_above,
        }

    @property
    def resolution_time(self):
        """BINARY: primary market's resolution_time; else None."""
        if self.event_type != EventType.BINARY:
            return None
        pm = self.primary_market
        return pm.resolution_time if pm is not None else None

    def __repr__(self):
        return (
            f"<Event {self.id} {self.title!r} "
            f"type={self.event_type.value} status={self.status.value}>"
        )
