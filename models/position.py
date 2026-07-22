"""Position model: per-(agent, market) YES/NO share holdings.

Uniqueness is on (agent_id, market_id) — an agent can hold shares in
several markets under one Event (e.g. YES on Trump *and* YES on Harris,
which the market will punish via arbitrage but is legal to express).
"""

from datetime import datetime
from typing import Tuple

from models.database import db


class Position(db.Model):
    __tablename__ = "positions"

    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(
        db.Integer, db.ForeignKey("agents.id"), nullable=False, index=True
    )
    market_id = db.Column(
        db.Integer, db.ForeignKey("markets.id"), nullable=False, index=True
    )
    yes_shares = db.Column(db.Float, default=0.0, nullable=False)
    no_shares = db.Column(db.Float, default=0.0, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        db.UniqueConstraint("agent_id", "market_id", name="uix_position_agent_market"),
    )

    def __repr__(self):
        return (
            f"<Position agent={self.agent_id} market={self.market_id} "
            f"yes={self.yes_shares:.2f} no={self.no_shares:.2f}>"
        )

    def net_side(self) -> Tuple[str, float]:
        """Return the directional side of the position.

        MarketService auto-merges complementary pairs after every trade,
        so in the steady state only one side is nonzero. When called
        mid-trade (before auto-merge fires) this reports the *net*
        exposure — whichever side is larger — with the merged remainder
        as its size. Returns ``("FLAT", 0.0)`` when the two sides are
        equal (including both zero).
        """
        yes = float(self.yes_shares or 0.0)
        no = float(self.no_shares or 0.0)
        if yes > no:
            return ("YES", yes - no)
        if no > yes:
            return ("NO", no - yes)
        return ("FLAT", 0.0)
