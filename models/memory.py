"""Compact statistical Agent memory + sparse important episodes.

Two rows drive the memory subsystem:

  AgentMemoryStats  — ONE compact row per Agent, updated INCREMENTALLY
      on every wake-up and trade. The action hot path reads this row
      only — never scans the Trade / AgentDecision tables.

  AgentMemoryEpisode — sparse; created ONLY for genuinely important
      events (large PnL, high-confidence errors, successful/failed
      reversals, early signals, severe drawdowns). Routine HOLDs and
      small trades do NOT create episodes.

JSON columns hold small dicts (category / strategy / source stats) so
schema evolution is additive. The `update_version` field is bumped by
callers using compare-and-swap when they need atomicity across workers.
"""

from datetime import datetime

from models.database import db


# --- episode taxonomy ------------------------------------------------
EPISODE_LARGE_GAIN = "LARGE_GAIN"
EPISODE_LARGE_LOSS = "LARGE_LOSS"
EPISODE_HIGH_CONFIDENCE_ERROR = "HIGH_CONFIDENCE_ERROR"
EPISODE_SUCCESSFUL_REVERSAL = "SUCCESSFUL_REVERSAL"
EPISODE_FAILED_REVERSAL = "FAILED_REVERSAL"
EPISODE_EARLY_SIGNAL = "EARLY_SIGNAL"
EPISODE_SEVERE_DRAWDOWN = "SEVERE_DRAWDOWN"

EPISODE_TYPES = (
    EPISODE_LARGE_GAIN, EPISODE_LARGE_LOSS, EPISODE_HIGH_CONFIDENCE_ERROR,
    EPISODE_SUCCESSFUL_REVERSAL, EPISODE_FAILED_REVERSAL,
    EPISODE_EARLY_SIGNAL, EPISODE_SEVERE_DRAWDOWN,
)

# Thresholds for episode creation (in the units they're compared to).
# Tuned so routine activity generates NO episodes.
LARGE_PNL_FRACTION = 0.05        # ≥5% of initial cash → large
HIGH_CONFIDENCE_ERROR_THRESHOLD = 0.6   # Brier > 0.6 at conf > 0.6
SEVERE_DRAWDOWN_FRACTION = 0.8   # crossing 80% of Agent's tolerance

# Retention: default cap per Agent; the prune step protects type/category
# diversity so rare important failures aren't just aged out.
DEFAULT_EPISODE_CAP = 50

MEMORY_ALGO_VERSION = 1


class AgentMemoryStats(db.Model):
    """One compact row per Agent — updated INCREMENTALLY, never rescanned."""

    __tablename__ = "agent_memory_stats"

    agent_id = db.Column(
        db.Integer, db.ForeignKey("agents.id"), primary_key=True
    )

    # Prediction calibration (running).
    resolved_prediction_count = db.Column(db.Integer, default=0, nullable=False)
    brier_running_sum = db.Column(db.Float, default=0.0, nullable=False)
    brier_average = db.Column(db.Float, default=0.0, nullable=False)
    log_loss_running_sum = db.Column(db.Float, default=0.0, nullable=False)
    log_loss_average = db.Column(db.Float, default=0.0, nullable=False)
    empirical_accuracy = db.Column(db.Float, default=0.0, nullable=False)  # correct-side hit rate
    average_confidence = db.Column(db.Float, default=0.0, nullable=False)
    overconfidence_score = db.Column(db.Float, default=0.0, nullable=False)  # conf − accuracy

    # PnL + portfolio.
    realized_pnl = db.Column(db.Float, default=0.0, nullable=False)
    unrealized_pnl = db.Column(db.Float, default=0.0, nullable=False)
    portfolio_value = db.Column(db.Float, default=0.0, nullable=False)
    high_water_mark = db.Column(db.Float, default=0.0, nullable=False)
    current_drawdown = db.Column(db.Float, default=0.0, nullable=False)  # (hwm-pv)/hwm ∈ [0,1]
    max_drawdown = db.Column(db.Float, default=0.0, nullable=False)

    win_streak = db.Column(db.Integer, default=0, nullable=False)
    loss_streak = db.Column(db.Integer, default=0, nullable=False)

    # Activity counts.
    wake_up_count = db.Column(db.Integer, default=0, nullable=False)
    wake_ups_without_trade = db.Column(db.Integer, default=0, nullable=False)
    trade_count = db.Column(db.Integer, default=0, nullable=False)
    profitable_trade_count = db.Column(db.Integer, default=0, nullable=False)

    # Compact per-slice stats. Each dict maps slice-name → {count, brier_sum,
    # pnl_sum} so an adjustment can pull "how do I do on category=ai" in O(1).
    category_stats = db.Column(db.JSON, nullable=True)
    strategy_stats = db.Column(db.JSON, nullable=True)
    source_reliability_stats = db.Column(db.JSON, nullable=True)

    update_version = db.Column(db.Integer, default=0, nullable=False)  # optimistic lock
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self):
        return (
            f"<AgentMemoryStats agent={self.agent_id} pv={self.portfolio_value:.2f} "
            f"dd={self.current_drawdown:.3f} wake={self.wake_up_count} "
            f"trades={self.trade_count}>"
        )


class AgentMemoryEpisode(db.Model):
    """Sparse important-event record. Created only when it matters."""

    __tablename__ = "agent_memory_episodes"

    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=False)
    event_id = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=True)
    market_id = db.Column(db.Integer, db.ForeignKey("markets.id"), nullable=True)
    category = db.Column(db.String(64), nullable=True)

    episode_type = db.Column(db.String(32), nullable=False)
    importance = db.Column(db.Float, default=0.0, nullable=False)     # [0,1]
    magnitude = db.Column(db.Float, default=0.0, nullable=False)      # signed PnL / effect

    # Compact metadata + a concise summary (never chain-of-thought).
    episode_metadata = db.Column(db.JSON, nullable=True)
    concise_summary = db.Column(db.String(400), nullable=True)

    # Optional LLM-summarized narrative + its routing metadata. Populated
    # only when MemoryService.summarize_episode_llm is called.
    llm_summary = db.Column(db.String(600), nullable=True)
    llm_summary_tier = db.Column(db.String(16), nullable=True)
    llm_summary_model = db.Column(db.String(128), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        db.Index("ix_episode_agent_created", "agent_id", "created_at"),
        db.Index("ix_episode_agent_type", "agent_id", "episode_type"),
    )

    def __repr__(self):
        return (
            f"<AgentMemoryEpisode {self.id} agent={self.agent_id} "
            f"{self.episode_type} imp={self.importance:.2f} "
            f"mag={self.magnitude:+.2f}>"
        )


__all__ = [
    "AgentMemoryStats",
    "AgentMemoryEpisode",
    "EPISODE_LARGE_GAIN", "EPISODE_LARGE_LOSS", "EPISODE_HIGH_CONFIDENCE_ERROR",
    "EPISODE_SUCCESSFUL_REVERSAL", "EPISODE_FAILED_REVERSAL",
    "EPISODE_EARLY_SIGNAL", "EPISODE_SEVERE_DRAWDOWN",
    "EPISODE_TYPES",
    "LARGE_PNL_FRACTION", "HIGH_CONFIDENCE_ERROR_THRESHOLD",
    "SEVERE_DRAWDOWN_FRACTION", "DEFAULT_EPISODE_CAP",
    "MEMORY_ALGO_VERSION",
]
