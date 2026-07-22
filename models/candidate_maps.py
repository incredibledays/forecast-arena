"""Sparse candidate-selection maps.

Event-driven wake-ups must select candidates from a UNION of sparse sets
without scanning the full Agent table. These tables are the sparse sets:

  AgentEventInterest    — an Agent is a holder / watcher / subscriber of a
                          specific event (role distinguishes them).
  AgentCategoryExpertise— an Agent has expertise in a category (e.g. "ai"),
                          derived once from its archetype/persona; used
                          when an event names a category.
  ArchetypeEventInterest— an Archetype is interested in an event; expanded
                          to its member Agents at selection time.

Every lookup column is indexed so candidate collection is a handful of
bounded index scans, not an O(population) sweep. Holder membership for an
event is derived from Positions on that event's Markets (also indexed).
"""

from datetime import datetime

from models.database import db


# Interest roles on an event.
ROLE_HOLDER = "holder"
ROLE_WATCHER = "watcher"
ROLE_SUBSCRIBER = "subscriber"


class AgentEventInterest(db.Model):
    """An Agent's interest in a specific event, with a role."""

    __tablename__ = "agent_event_interest"

    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=False)
    event_id = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=False)
    role = db.Column(db.String(16), default=ROLE_WATCHER, nullable=False)
    # Optional per-interest weight in [0,1] the score can read.
    weight = db.Column(db.Float, default=0.5, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("agent_id", "event_id", "role", name="uix_agent_event_role"),
        # Primary lookup: "who is interested in event e" — index by event.
        db.Index("ix_interest_event_role", "event_id", "role"),
        db.Index("ix_interest_agent", "agent_id"),
    )

    def __repr__(self):
        return f"<AgentEventInterest agent={self.agent_id} event={self.event_id} {self.role}>"


class AgentCategoryExpertise(db.Model):
    """An Agent's expertise in a topical category (sparse: only where high)."""

    __tablename__ = "agent_category_expertise"

    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=False)
    category = db.Column(db.String(64), nullable=False)
    expertise = db.Column(db.Float, default=0.0, nullable=False)  # [0,1]
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("agent_id", "category", name="uix_agent_category"),
        # Primary lookup: "who are the experts in category c" — index by
        # (category, expertise) so a threshold scan is bounded.
        db.Index("ix_expertise_category", "category", "expertise"),
        db.Index("ix_expertise_agent", "agent_id"),
    )

    def __repr__(self):
        return (
            f"<AgentCategoryExpertise agent={self.agent_id} "
            f"{self.category}={self.expertise:.2f}>"
        )


class ArchetypeEventInterest(db.Model):
    """An Archetype's interest in an event (expanded to members at selection)."""

    __tablename__ = "archetype_event_interest"

    id = db.Column(db.Integer, primary_key=True)
    archetype_id = db.Column(
        db.Integer, db.ForeignKey("agent_archetypes.id"), nullable=False
    )
    event_id = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=False)
    weight = db.Column(db.Float, default=0.5, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("archetype_id", "event_id", name="uix_arch_event"),
        db.Index("ix_arch_interest_event", "event_id"),
    )

    def __repr__(self):
        return (
            f"<ArchetypeEventInterest arch={self.archetype_id} event={self.event_id}>"
        )


__all__ = [
    "AgentEventInterest",
    "AgentCategoryExpertise",
    "ArchetypeEventInterest",
    "ROLE_HOLDER",
    "ROLE_WATCHER",
    "ROLE_SUBSCRIBER",
]
