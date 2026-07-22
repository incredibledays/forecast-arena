"""Market model: a single binary YES/NO tradeable unit.

An Event is a topic container; each Event has one or more Markets. For
BINARY events there is exactly one Market with an empty `label` and the
UI renders it as plain YES/NO. Future event types (CATEGORICAL, SCALAR,
GROUPED, CONDITIONAL) hang N Markets off one Event with meaningful
labels and, for SCALAR, `bucket_lo` / `bucket_hi` bounds.

All trading state (shares, prices, outcome) lives here, NOT on Event.
Trade / Position / PriceHistory all FK to `markets.id`.
"""

import enum
from datetime import datetime

from models.database import db


class MarketStatus(str, enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"      # trading halted, awaiting resolution
    RESOLVED = "RESOLVED"  # outcome known


class MarketOutcome(str, enum.Enum):
    YES = "YES"
    NO = "NO"
    UNRESOLVED = "UNRESOLVED"
    REFUNDED = "REFUNDED"  # CONDITIONAL child auto-refunded (parent went opposite)


class Market(db.Model):
    __tablename__ = "markets"

    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(
        db.Integer, db.ForeignKey("events.id"), nullable=False, index=True
    )
    # For BINARY events, label is NULL and the UI shows plain YES/NO.
    # For CATEGORICAL/GROUPED, e.g. "Trump" or "降息 25bp".
    # For SCALAR buckets, e.g. "90k–100k".
    label = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)

    # Populated only for SCALAR bucket markets — inclusive lower / exclusive
    # upper bound of the numeric range this market covers.
    bucket_lo = db.Column(db.Float, nullable=True)
    bucket_hi = db.Column(db.Float, nullable=True)

    status = db.Column(
        db.Enum(MarketStatus, native_enum=False, length=16),
        default=MarketStatus.OPEN,
        nullable=False,
    )
    outcome = db.Column(
        db.Enum(MarketOutcome, native_enum=False, length=16),
        default=None,
        nullable=True,
    )
    resolution_time = db.Column(db.DateTime, nullable=True)

    # --- LMSR (Hanson Logarithmic Market Scoring Rule) state ---------
    # Hanson LMSR prices a binary market from `q_yes` / `q_no` (total
    # outstanding shares of each side across all agents) and the fixed
    # liquidity parameter `b`. See `services/market_service.py` for the
    # closed-form pricing and inversion math.
    #
    # Invariants:
    #   p_yes = 1 / (1 + exp((q_no - q_yes) / b))            # never 0 or 1
    #   sum(pos.yes_shares for pos in market.positions) == q_yes
    #   sum(pos.no_shares  for pos in market.positions) == q_no
    #
    # `liquidity_b` is per-market so future seed logic can dial liquidity
    # by category (higher b for stable macro markets, lower b for
    # niche / illiquid ones). Default 5000: with ~$10k agent cash and
    # trade sizes of $600–1500 (10–15% of cash), a single trade moves
    # yes_price by ~5–15% from 50/50 — visible but not saturating, which
    # keeps `edge = p_llm - yes_price` a meaningful signal for the next
    # agent instead of always slamming to 0/1.
    liquidity_b = db.Column(db.Float, default=5000.0, nullable=False)
    q_yes = db.Column(db.Float, default=0.0, nullable=False)
    q_no = db.Column(db.Float, default=0.0, nullable=False)

    # CONDITIONAL child: pointer to the parent Market whose resolution
    # gates this one. When the parent resolves to
    # `parent_required_outcome`, this market continues trading /
    # resolving normally. If the parent resolves to the opposite side,
    # this market is auto-refunded (all buys unwound at their original
    # amounts). NULL for non-conditional markets.
    parent_market_id = db.Column(
        db.Integer, db.ForeignKey("markets.id"), nullable=True, index=True
    )
    # Which side the parent must resolve to for this child to be
    # honored. Only YES or NO — REFUNDED / UNRESOLVED are invalid here.
    parent_required_outcome = db.Column(
        db.Enum(MarketOutcome, native_enum=False, length=16),
        default=None,
        nullable=True,
    )

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Running sum of every Trade.amount on this market. Same rationale as
    # Event.total_volume — the per-market volume readout on multi-market
    # event pages (CATEGORICAL / SCALAR) otherwise runs a `SUM WHERE
    # market_id = X` full scan per sub-market on every page hit.
    total_volume = db.Column(db.Float, default=0.0, nullable=False)

    # ---- MarketState versioning ----------------------------------------
    # Monotonically incremented after every non-HOLD trade that mutates
    # q_yes/q_no/status. Quotes carry `market_state_version = version` and
    # the executor refuses to execute a quote whose version no longer
    # matches — see `services.market_service.execute_trade`. `state_updated_at`
    # is a wall-clock breadcrumb (the virtual clock lives in SchedulerClock).
    version = db.Column(db.Integer, default=0, nullable=False)
    state_updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships — all trading rows hang off the Market, not the Event.
    trades = db.relationship(
        "Trade", backref="market", lazy="dynamic", cascade="all, delete-orphan"
    )
    positions = db.relationship(
        "Position", backref="market", lazy="dynamic", cascade="all, delete-orphan"
    )
    price_history = db.relationship(
        "PriceHistory", backref="market", lazy="dynamic", cascade="all, delete-orphan"
    )
    # Self-referential: children[] are markets that pointed their
    # parent_market_id back here. Used by the settlement cascade to find
    # what to refund/enable when this market resolves.
    children = db.relationship(
        "Market",
        backref=db.backref("parent_market", remote_side="Market.id"),
        foreign_keys=[parent_market_id],
        lazy="dynamic",
    )

    def __repr__(self):
        return (
            f"<Market {self.id} event={self.event_id} "
            f"label={self.label!r} status={self.status.value}>"
        )
