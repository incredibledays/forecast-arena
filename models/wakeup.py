"""Event-driven wake-up + trigger-cooldown models.

This phase adds *sparse* event-driven wake-ups on top of the natural
scheduler. Natural wake-ups stay in `AgentScheduleState` (one next
wake-up per Agent); event-driven wake-ups are bounded per event, so they
live as explicit `WakeUpTask` rows that a later worker phase will drain.

`WakeUpTask` carries a `dedup_key` — `agent_id:event_id:time_bucket` —
with a UNIQUE constraint so concurrent triggers for the same Agent/event
in the same time bucket MERGE into one row (see TriggerService._upsert)
rather than creating duplicates. Ranking fields (priority, relevances,
scheduled time) let the drain order work; cascade fields bound price
cascades.

No LLM is involved anywhere here.
"""

import enum
from datetime import datetime

from models.database import db


class TriggerType(str, enum.Enum):
    """Why an Agent is being woken. NATURAL is preserved from the prior phase."""

    NATURAL = "NATURAL"
    INFORMATION = "INFORMATION"
    PRICE = "PRICE"
    PORTFOLIO_RISK = "PORTFOLIO_RISK"
    MARKET_CLOSING = "MARKET_CLOSING"
    RESOLUTION = "RESOLUTION"


# Priority ladder (higher = more important). Drives ranking, dedup
# (highest wins), and load shedding (lowest shed first). RESOLUTION and
# PORTFOLIO_RISK sit at the top and are never shed.
TRIGGER_PRIORITY = {
    TriggerType.NATURAL.value: 10,
    TriggerType.INFORMATION.value: 40,
    TriggerType.PRICE.value: 50,
    TriggerType.MARKET_CLOSING.value: 70,
    TriggerType.PORTFOLIO_RISK.value: 90,
    TriggerType.RESOLUTION.value: 100,
}

# Wake urgency tiers (used for information-event budgets + scheduling).
TIER_URGENT = "urgent"
TIER_NORMAL = "normal"
TIER_DELAYED = "delayed"

# Statuses a WakeUpTask moves through. This phase only ever writes
# PENDING (scheduling + selection); a later worker phase consumes them.
STATUS_PENDING = "pending"
STATUS_CLAIMED = "claimed"
STATUS_DONE = "done"
STATUS_SHED = "shed"
STATUS_FAILED = "failed"

# Time-bucket width (seconds) for dedup. Triggers for the same
# agent/event landing in the same bucket merge into one task.
DEFAULT_BUCKET_SECONDS = 300.0


def time_bucket(virtual_time_s: float, width_s: float = DEFAULT_BUCKET_SECONDS) -> int:
    """Floor virtual time into an integer bucket index for dedup keys."""
    if width_s <= 0:
        width_s = DEFAULT_BUCKET_SECONDS
    return int(virtual_time_s // width_s)


def make_dedup_key(agent_id: int, event_id, bucket: int) -> str:
    """Stable dedup identity: one task per (agent, event, time bucket)."""
    return f"{int(agent_id)}:{'_' if event_id is None else int(event_id)}:{int(bucket)}"


class WakeUpTask(db.Model):
    __tablename__ = "wakeup_tasks"

    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=False)
    # Event-scoped for INFORMATION/PRICE/CLOSING/RESOLUTION; may be NULL
    # for a pure portfolio-risk wake that isn't tied to one event.
    event_id = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=True)
    market_id = db.Column(db.Integer, db.ForeignKey("markets.id"), nullable=True)

    trigger_type = db.Column(db.String(20), nullable=False)
    priority = db.Column(db.Integer, default=0, nullable=False)
    tier = db.Column(db.String(12), default=TIER_NORMAL, nullable=False)
    status = db.Column(db.String(12), default=STATUS_PENDING, nullable=False)

    # Virtual time (seconds) the Agent should actually wake — the jittered
    # scheduled time, NOT the trigger time.
    scheduled_at = db.Column(db.Float, nullable=False)
    time_bucket = db.Column(db.Integer, nullable=False)

    # Ranking signals (all in [0,1] except priority above).
    relevance = db.Column(db.Float, default=0.0, nullable=False)
    information_impact = db.Column(db.Float, default=0.0, nullable=False)
    position_relevance = db.Column(db.Float, default=0.0, nullable=False)
    expertise = db.Column(db.Float, default=0.0, nullable=False)
    portfolio_risk = db.Column(db.Float, default=0.0, nullable=False)
    wake_score = db.Column(db.Float, default=0.0, nullable=False)

    # Merged wake reasons (comma-joined trigger types) + dedup identity.
    wake_reasons = db.Column(db.String(128), nullable=True)
    dedup_key = db.Column(db.String(64), nullable=False)

    # Cascade / provenance metadata (price triggers).
    cascade_id = db.Column(db.String(40), nullable=True)
    cascade_depth = db.Column(db.Integer, default=0, nullable=False)
    trigger_trade_id = db.Column(db.Integer, nullable=True)

    # Version pointers a later belief/market phase will read. Stored now
    # so dedup can keep the latest; unused for compute here.
    evidence_bundle_version = db.Column(db.String(32), nullable=True)
    market_state_version = db.Column(db.String(32), nullable=True)

    # --- worker-pipeline fields (this phase) ---
    # Incremented every time the worker defers or fails the task. Once a
    # task exceeds `MAX_RETRIES` in the processor it moves to STATUS_FAILED
    # with the concise error captured on `last_error` — never retried
    # unbounded.
    retry_count = db.Column(db.Integer, default=0, nullable=False)
    last_error = db.Column(db.String(200), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        # Transactionally-safe dedup: one task per (agent, event, bucket).
        db.UniqueConstraint("dedup_key", name="uix_wakeup_dedup"),
        # Drain / due query: pending tasks ordered by when they fire.
        db.Index("ix_wakeup_status_scheduled", "status", "scheduled_at"),
        # Per-market cap counting + cooldown lookups.
        db.Index("ix_wakeup_market_type", "market_id", "trigger_type"),
        db.Index("ix_wakeup_agent", "agent_id"),
        db.Index("ix_wakeup_status_updated", "status", "updated_at"),
    )

    def __repr__(self):
        return (
            f"<WakeUpTask {self.id} agent={self.agent_id} event={self.event_id} "
            f"{self.trigger_type} prio={self.priority} @ {self.scheduled_at:.1f}>"
        )


class TriggerCooldown(db.Model):
    """Per (agent, market, trigger_type) cooldown gate for price triggers.

    Records the virtual time of the last accepted trigger so a burst of
    price moves can't wake the same Agent on the same market repeatedly
    inside the cooldown window.
    """

    __tablename__ = "trigger_cooldowns"

    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, nullable=False)
    market_id = db.Column(db.Integer, nullable=True)
    trigger_type = db.Column(db.String(20), nullable=False)
    last_fired_at = db.Column(db.Float, nullable=False)  # virtual seconds

    __table_args__ = (
        db.UniqueConstraint(
            "agent_id", "market_id", "trigger_type", name="uix_cooldown"
        ),
        db.Index("ix_cooldown_lookup", "agent_id", "market_id", "trigger_type"),
    )

    def __repr__(self):
        return (
            f"<TriggerCooldown agent={self.agent_id} market={self.market_id} "
            f"{self.trigger_type} @ {self.last_fired_at:.1f}>"
        )
