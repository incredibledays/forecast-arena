"""Trade model: one buy/hold decision by an agent on a market.

Trades are per-Market, not per-Event: a multi-market Event (e.g. a
CATEGORICAL race with N candidates) accumulates one Trade per decision
per candidate market.
"""

import enum
from datetime import datetime

from models.database import db


class TradeAction(str, enum.Enum):
    """Agent-facing actions plus one service-generated bookkeeping row.

    Agents may emit BUY_YES / BUY_NO / SELL_YES / SELL_NO / FLIP_YES /
    FLIP_NO / HOLD. AUTO_MERGE is written only by MarketService when a
    position that ends up holding both YES and NO shares gets the
    complementary pair merged back into cash (Polymarket CTF semantics).
    """

    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    SELL_YES = "SELL_YES"
    SELL_NO = "SELL_NO"
    FLIP_YES = "FLIP_YES"       # sell all NO, then buy YES
    FLIP_NO = "FLIP_NO"         # sell all YES, then buy NO
    HOLD = "HOLD"
    AUTO_MERGE = "AUTO_MERGE"   # service-only: complementary pair → cash


class Trade(db.Model):
    __tablename__ = "trades"

    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(
        db.Integer, db.ForeignKey("agents.id"), nullable=False, index=True
    )
    market_id = db.Column(
        db.Integer, db.ForeignKey("markets.id"), nullable=False, index=True
    )
    action = db.Column(
        db.Enum(TradeAction, native_enum=False, length=16), nullable=False
    )
    # For BUY_*: cash spent. For SELL_*/AUTO_MERGE: cash received. For
    # FLIP_*: cash "extra" appended to the target-side leg (may be 0).
    amount = db.Column(db.Float, default=0.0, nullable=False)
    # SELL_*/FLIP_* only — fraction of the closing side sold, in (0, 1].
    # BUY_*/HOLD/AUTO_MERGE leave this NULL.
    fraction = db.Column(db.Float, nullable=True)
    # FLIP_* is stored as two separate Trade rows (a SELL leg then a BUY
    # leg) sharing this UUID so the UI can fold them back together. NULL
    # for atomic actions.
    trade_group_id = db.Column(db.String(32), nullable=True, index=True)
    # LMSR share delta actually moved on this trade. BUY_*: positive.
    # SELL_*: positive (shares surrendered, not signed). FLIP legs each
    # record their own leg's shares. AUTO_MERGE: pairs burned. HOLD:
    # NULL. This column mirrors the position delta so the audit log is
    # self-contained without joining Position history.
    shares = db.Column(db.Float, nullable=True)
    price_before = db.Column(db.Float, nullable=True)                  # YES price pre-trade
    price_after = db.Column(db.Float, nullable=True)                   # YES price post-trade
    probability_yes = db.Column(db.Float, nullable=True)               # agent's belief
    confidence = db.Column(db.Float, nullable=True)                    # 0..1
    reasoning_summary = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        # Two hot access patterns need composite indexes:
        #   1. event_detail: latest N trades on this event (join through
        #      markets, filter by event, order by created_at desc). The
        #      (market_id, created_at) index lets SQLite walk backwards
        #      instead of scanning + sorting.
        #   2. brier scoring: latest trade per (agent, market) with a
        #      non-null probability_yes — (agent_id, market_id, created_at)
        #      lets that query use an index range instead of a full scan.
        db.Index("ix_trades_market_created", "market_id", "created_at"),
        db.Index(
            "ix_trades_agent_market_created", "agent_id", "market_id", "created_at"
        ),
    )

    def __repr__(self):
        return (
            f"<Trade {self.id} agent={self.agent_id} market={self.market_id} "
            f"{self.action.value} amount={self.amount:.2f}>"
        )
