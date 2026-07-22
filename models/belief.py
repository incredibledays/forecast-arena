"""Belief models: Archetype-level beliefs + lazy individual Agent beliefs.

Two tiers, mirroring the population design:

  ArchetypeBelief — ONE LLM-produced belief per (archetype, event,
      EvidenceBundle version). A whole population of Agents sharing an
      archetype reads from this single row; the LLM is called per
      archetype (or per compatible batch), never per Agent.

  AgentBelief — an individual Agent's belief, derived in PURE CODE from
      its ArchetypeBelief plus per-Agent personalization (expertise /
      source-trust / bias / memory / deterministic noise). These rows are
      MATERIALIZED LAZILY — only for Agents that hold, watch, subscribe,
      were woken, or need an auditable snapshot. Sleeping Agents do not
      all acquire rows; their belief is reconstructed on demand from the
      ArchetypeBelief + a compact projection + the deterministic seed.

Only concise reasoning is ever stored — never chain-of-thought.
"""

from datetime import datetime

from models.database import db


# AgentBelief lifecycle / provenance.
BELIEF_STATUS_MATERIALIZED = "materialized"   # persisted row
BELIEF_STATUS_RECONSTRUCTED = "reconstructed"  # computed, not stored
BELIEF_STATUS_STALE = "stale"                  # bundle moved on

# Bump when the personalization math changes so stored rows are
# distinguishable and reconstruction stays honest.
PERSONALIZATION_ALGO_VERSION = 1


class ArchetypeBelief(db.Model):
    __tablename__ = "archetype_beliefs"

    id = db.Column(db.Integer, primary_key=True)
    archetype_id = db.Column(
        db.Integer, db.ForeignKey("agent_archetypes.id"), nullable=False, index=True
    )
    event_id = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=False, index=True)
    evidence_bundle_version = db.Column(db.Integer, nullable=False)

    prior_probability = db.Column(db.Float, nullable=True)
    posterior_probability = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, default=0.5, nullable=False)

    # Concise reasoning ONLY — never chain-of-thought.
    reasoning_summary = db.Column(db.Text, nullable=True)
    key_evidence = db.Column(db.JSON, nullable=True)     # list of short citations
    risk_factors = db.Column(db.JSON, nullable=True)     # list of short strings

    # Model provenance / routing metadata.
    model_provider = db.Column(db.String(64), nullable=True)
    model = db.Column(db.String(128), nullable=True)
    model_tier = db.Column(db.String(16), nullable=True)   # FAST/BALANCED/STRONG/...
    batch_mode = db.Column(db.String(16), nullable=True)
    prompt_version = db.Column(db.String(32), nullable=True)
    cache_metadata = db.Column(db.JSON, nullable=True)     # {stable_prefix_hash, ...}
    degraded = db.Column(db.Boolean, default=False, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        db.UniqueConstraint(
            "archetype_id", "event_id", "evidence_bundle_version",
            name="uix_archbelief_arch_event_ver",
        ),
        db.Index("ix_archbelief_event_ver", "event_id", "evidence_bundle_version"),
    )

    def __repr__(self):
        return (
            f"<ArchetypeBelief a={self.archetype_id} e={self.event_id} "
            f"v{self.evidence_bundle_version} p={self.posterior_probability:.3f} "
            f"tier={self.model_tier}{' DEGRADED' if self.degraded else ''}>"
        )


class AgentBelief(db.Model):
    __tablename__ = "agent_beliefs"

    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=False, index=True)
    event_id = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=False, index=True)
    archetype_belief_id = db.Column(
        db.Integer, db.ForeignKey("archetype_beliefs.id"), nullable=True
    )

    prior_probability = db.Column(db.Float, nullable=True)
    raw_probability = db.Column(db.Float, nullable=False)         # pre-calibration
    calibrated_probability = db.Column(db.Float, nullable=False)  # clamped [0.01,0.99]
    confidence = db.Column(db.Float, default=0.5, nullable=False)

    belief_status = db.Column(db.String(16), default=BELIEF_STATUS_MATERIALIZED, nullable=False)
    last_evidence_bundle_version = db.Column(db.Integer, nullable=True)
    last_market_price_seen = db.Column(db.Float, nullable=True)
    next_review_time = db.Column(db.Float, nullable=True)  # virtual seconds

    # The pure-code adjustment breakdown (compact JSON) — lets an audit
    # reproduce the number and a UI explain it.
    personalization_components = db.Column(db.JSON, nullable=True)
    personalization_algo_version = db.Column(
        db.Integer, default=PERSONALIZATION_ALGO_VERSION, nullable=False
    )

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        db.UniqueConstraint("agent_id", "event_id", name="uix_agentbelief_agent_event"),
        db.Index("ix_agentbelief_event", "event_id"),
    )

    def __repr__(self):
        return (
            f"<AgentBelief agent={self.agent_id} e={self.event_id} "
            f"p={self.calibrated_probability:.3f} {self.belief_status}>"
        )


__all__ = [
    "ArchetypeBelief",
    "AgentBelief",
    "BELIEF_STATUS_MATERIALIZED",
    "BELIEF_STATUS_RECONSTRUCTED",
    "BELIEF_STATUS_STALE",
    "PERSONALIZATION_ALGO_VERSION",
]
