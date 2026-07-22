"""PriceHistory model: time-series snapshots of YES/NO prices per market."""

from datetime import datetime

from models.database import db


class PriceHistory(db.Model):
    __tablename__ = "price_history"

    id = db.Column(db.Integer, primary_key=True)
    market_id = db.Column(
        db.Integer, db.ForeignKey("markets.id"), nullable=False, index=True
    )
    yes_price = db.Column(db.Float, nullable=False)
    no_price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        # Composite index for the hot access pattern:
        #   PriceHistory.query.filter_by(market_id=X).order_by(timestamp DESC).first()
        # (used on the index page for every event) and
        #   .order_by(timestamp ASC).all()
        # (used on event_detail for the chart). Without this, once
        # price_history reaches ~20k rows every event_detail load spends
        # hundreds of ms sorting.
        db.Index("ix_price_history_market_ts", "market_id", "timestamp"),
    )

    def __repr__(self):
        return (
            f"<PriceHistory market={self.market_id} "
            f"yes={self.yes_price:.3f} no={self.no_price:.3f} @ {self.timestamp}>"
        )
