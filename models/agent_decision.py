"""AgentDecision — one row per action attempt (HOLD or trade).

The audit trail for what a policy decided vs what the executor did. HOLDs
land here (never in `trades`, per the LMSR engine spec); non-HOLD trades
also land here and link to the persisted `Trade` row via `trade_id`.

Executed fields (`actual_notional`, `average_execution_price`, marginal
prices, version-seen vs version-at-execution, `was_stale`) let a later
worker/audit layer reconstruct any decision without touching the LMSR
math or the Trade table.
"""

from datetime import datetime

from models.database import db


class AgentDecision(db.Model):
    __tablename__ = "agent_decisions"

    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=False, index=True)
    event_id = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=True, index=True)
    market_id = db.Column(db.Integer, db.ForeignKey("markets.id"), nullable=True, index=True)

    # --- what the policy asked for ---
    recommended_action = db.Column(db.String(16), nullable=False)  # BUY_YES/NO/SELL_/FLIP_/HOLD
    was_hold = db.Column(db.Boolean, default=False, nullable=False)
    outcome_side = db.Column(db.String(8), nullable=True)  # YES / NO / NEUTRAL / None
    urgency = db.Column(db.String(8), nullable=True)       # LOW / NORMAL / HIGH

    probability_yes = db.Column(db.Float, nullable=True)   # Agent's fair prob
    confidence = db.Column(db.Float, nullable=True)
    edge = db.Column(db.Float, nullable=True)              # (p_agent − price) or symmetric

    requested_notional = db.Column(db.Float, nullable=True)
    actual_notional = db.Column(db.Float, nullable=True)   # what actually consumed/refunded

    # --- execution outcome (nullable for HOLD) ---
    average_execution_price = db.Column(db.Float, nullable=True)
    marginal_price_before = db.Column(db.Float, nullable=True)
    marginal_price_after = db.Column(db.Float, nullable=True)

    market_state_version_seen = db.Column(db.Integer, nullable=True)      # quote's version
    market_state_version_at_execution = db.Column(db.Integer, nullable=True)  # post-execute version
    was_stale = db.Column(db.Boolean, default=False, nullable=False)

    trade_id = db.Column(db.Integer, db.ForeignKey("trades.id"), nullable=True)

    reasoning_summary = db.Column(db.String(400), nullable=True)
    policy_factors_json = db.Column(db.JSON, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        db.Index("ix_decision_agent_created", "agent_id", "created_at"),
        db.Index("ix_decision_event_created", "event_id", "created_at"),
        db.Index("ix_agent_decisions_created", "created_at"),
    )

    def __repr__(self):
        return (
            f"<AgentDecision {self.id} agent={self.agent_id} "
            f"{self.recommended_action} hold={self.was_hold} "
            f"trade={self.trade_id} stale={self.was_stale}>"
        )


__all__ = ["AgentDecision"]
