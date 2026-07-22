"""Shared real-time evidence layer models.

Four tables implement the "retrieve once, store once, version, delta"
design:

  SourceContent    — canonical source text stored ONCE, keyed by
                     content_hash. Many InformationEvents (across many
                     markets/events) reference the same row; we never
                     duplicate full source text per market.
  InformationEvent — one appraisal of a SourceContent for a specific
                     forecast Event: relevance / credibility / freshness /
                     novelty / stance / impact + prompt-injection risk, and
                     the historical-fairness timestamps (published_at,
                     retrieved_at, available_to_agents_at).
  EvidenceBundle   — a versioned, per-Event snapshot of the current
                     evidence set: added/removed/superseded ids, the S/O/N/
                     official reference lists, what changed, contradiction /
                     uncertainty / aggregate-impact summaries, a compact
                     current summary, and model metadata. Versions are
                     monotonically increasing per Event.
  EvidenceDelta    — the compact change between two bundles: only newly
                     added / removed facts, changed contradictions, changed
                     uncertainty, impact delta, and a prior-probability
                     context reference. This is what routine updates send
                     instead of the full history.

JSON columns use `db.JSON` (TEXT on SQLite, JSONB later). No LLM here.
"""

from datetime import datetime

from models.database import db


# Coarse source classifications used for ranking + official-source
# preservation. Free-form string so new types don't need a migration.
SOURCE_TYPE_OFFICIAL = "official"
SOURCE_TYPE_WIRE = "wire"
SOURCE_TYPE_NEWS = "news"
SOURCE_TYPE_BLOG = "blog"
SOURCE_TYPE_SOCIAL = "social"
SOURCE_TYPE_OTHER = "other"

STANCE_SUPPORT = "SUPPORT"
STANCE_REFUTE = "REFUTE"
STANCE_NEUTRAL = "NEUTRAL"


class SourceContent(db.Model):
    """Canonical source text, stored ONCE and deduplicated by content_hash."""

    __tablename__ = "source_content"

    id = db.Column(db.Integer, primary_key=True)
    # SHA-256 (hex) of the cleaned text — the dedup key. Two retrievals of
    # the same article (even via different URLs) collapse to one row.
    content_hash = db.Column(db.String(64), unique=True, nullable=False, index=True)
    canonical_url = db.Column(db.String(1024), nullable=True)
    title = db.Column(db.String(512), nullable=True)
    source_domain = db.Column(db.String(128), nullable=True, index=True)
    source_type = db.Column(db.String(16), default=SOURCE_TYPE_OTHER, nullable=False)
    published_at = db.Column(db.DateTime, nullable=True)
    retrieved_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    # Compact extracted / cleaned content (boilerplate stripped, capped).
    cleaned_text = db.Column(db.Text, nullable=True)
    language = db.Column(db.String(8), nullable=True)
    content_metadata = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return (
            f"<SourceContent {self.id} {self.content_hash[:10]} "
            f"{self.source_domain} {self.source_type}>"
        )


class InformationEvent(db.Model):
    """One appraisal of a SourceContent for a specific forecast Event."""

    __tablename__ = "information_events"

    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=False, index=True)
    source_content_id = db.Column(
        db.Integer, db.ForeignKey("source_content.id"), nullable=False, index=True
    )

    # Appraisal scores, all in [0, 1].
    relevance = db.Column(db.Float, default=0.0, nullable=False)
    credibility = db.Column(db.Float, default=0.0, nullable=False)
    freshness = db.Column(db.Float, default=0.0, nullable=False)
    novelty = db.Column(db.Float, default=0.0, nullable=False)
    impact = db.Column(db.Float, default=0.0, nullable=False)
    stance = db.Column(db.String(16), default=STANCE_NEUTRAL, nullable=False)
    # Prompt-injection risk in [0,1] + the raw flags (JSON list).
    prompt_injection_risk = db.Column(db.Float, default=0.0, nullable=False)
    injection_flags = db.Column(db.JSON, nullable=True)

    # --- historical fairness: three distinct clocks ---
    # published_at: when the world learned it (from the source).
    # retrieved_at: when we fetched it.
    # available_to_agents_at: the virtual time an Agent may first USE it.
    published_at = db.Column(db.DateTime, nullable=True)
    retrieved_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    available_to_agents_at = db.Column(db.Float, nullable=False, index=True)  # virtual seconds

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    source = db.relationship("SourceContent", lazy="joined")

    __table_args__ = (
        # Fetch this event's evidence available by a given virtual time.
        db.Index("ix_infoevent_event_available", "event_id", "available_to_agents_at"),
        # One appraisal per (event, source) — re-refresh updates in place.
        db.UniqueConstraint("event_id", "source_content_id", name="uix_event_source"),
    )

    def __repr__(self):
        return (
            f"<InformationEvent {self.id} event={self.event_id} "
            f"src={self.source_content_id} stance={self.stance} "
            f"impact={self.impact:.2f}>"
        )


class EvidenceBundle(db.Model):
    """A versioned, per-Event snapshot of the current evidence set."""

    __tablename__ = "evidence_bundles"

    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=False, index=True)
    # Monotonically increasing per event, starting at 1.
    version = db.Column(db.Integer, nullable=False)
    previous_bundle_id = db.Column(
        db.Integer, db.ForeignKey("evidence_bundles.id"), nullable=True
    )

    # Change sets (JSON lists of InformationEvent ids).
    added_evidence_ids = db.Column(db.JSON, nullable=True)
    removed_evidence_ids = db.Column(db.JSON, nullable=True)
    superseded_evidence_ids = db.Column(db.JSON, nullable=True)

    # Categorized reference lists (JSON lists of InformationEvent ids).
    supporting_evidence_ids = db.Column(db.JSON, nullable=True)
    opposing_evidence_ids = db.Column(db.JSON, nullable=True)
    neutral_evidence_ids = db.Column(db.JSON, nullable=True)
    official_evidence_ids = db.Column(db.JSON, nullable=True)

    # Narrative / analytic summaries (compact, LLM-free in this phase).
    what_changed = db.Column(db.Text, nullable=True)
    contradiction_changes = db.Column(db.JSON, nullable=True)
    uncertainty_summary = db.Column(db.Text, nullable=True)
    aggregate_impact = db.Column(db.Float, default=0.0, nullable=False)
    current_summary = db.Column(db.Text, nullable=True)

    # Which model/tier/route produced any LLM-assisted fields (JSON). Empty
    # in this phase since summaries are computed, not generated.
    model_metadata = db.Column(db.JSON, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("event_id", "version", name="uix_bundle_event_version"),
        db.Index("ix_bundle_event_version", "event_id", "version"),
    )

    def __repr__(self):
        return (
            f"<EvidenceBundle {self.id} event={self.event_id} v{self.version} "
            f"impact={self.aggregate_impact:.2f}>"
        )


class EvidenceDelta(db.Model):
    """The compact change between two bundles — what routine updates send."""

    __tablename__ = "evidence_deltas"

    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey("events.id"), nullable=False, index=True)
    bundle_id = db.Column(
        db.Integer, db.ForeignKey("evidence_bundles.id"), nullable=False, index=True
    )
    from_version = db.Column(db.Integer, nullable=True)   # None for the first bundle
    to_version = db.Column(db.Integer, nullable=False)

    # Compact change payload — NOT the full history.
    added_facts = db.Column(db.JSON, nullable=True)          # [{id, title, stance, impact}]
    removed_facts = db.Column(db.JSON, nullable=True)        # [id, ...]
    changed_contradictions = db.Column(db.JSON, nullable=True)
    uncertainty_change = db.Column(db.Float, default=0.0, nullable=False)
    impact_delta = db.Column(db.Float, default=0.0, nullable=False)
    # Reference (not a copy) to the prior probability context — a bundle
    # version the belief phase can look up. Kept as a small pointer.
    previous_probability_ref = db.Column(db.String(64), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return (
            f"<EvidenceDelta {self.id} event={self.event_id} "
            f"v{self.from_version}->v{self.to_version} "
            f"impactΔ={self.impact_delta:+.2f}>"
        )


__all__ = [
    "SourceContent",
    "InformationEvent",
    "EvidenceBundle",
    "EvidenceDelta",
    "SOURCE_TYPE_OFFICIAL",
    "SOURCE_TYPE_WIRE",
    "SOURCE_TYPE_NEWS",
    "SOURCE_TYPE_BLOG",
    "SOURCE_TYPE_SOCIAL",
    "SOURCE_TYPE_OTHER",
    "STANCE_SUPPORT",
    "STANCE_REFUTE",
    "STANCE_NEUTRAL",
]
