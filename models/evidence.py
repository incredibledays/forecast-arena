"""Evidence model: retrieved snippets an agent used to form a belief.

Populated by the retrieval pipeline (see `services/retrieval_service.py`
and `retrieval/`) and persisted from `run_agents._persist_evidence`.

Column additions in the retrieval-layer overhaul are all nullable so
older rows written before the upgrade keep loading cleanly. An existing
DB needs `python init_db.py --reset` to add the columns (this repo
uses `db.create_all()` rather than Alembic).
"""

from datetime import datetime

from models.database import db


class Evidence(db.Model):
    __tablename__ = "evidence"

    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(
        db.Integer, db.ForeignKey("agents.id"), nullable=False, index=True
    )
    event_id = db.Column(
        db.Integer, db.ForeignKey("events.id"), nullable=False, index=True
    )
    # Present only for agent-generated event-analysis rows tied to an actual
    # executed trade. Retrieved/news evidence leaves this NULL.
    trade_id = db.Column(db.Integer, db.ForeignKey("trades.id"), nullable=True, index=True)
    query = db.Column(db.String(512), nullable=True)
    title = db.Column(db.String(255), nullable=True)
    url = db.Column(db.String(1024), nullable=True)
    content_summary = db.Column(db.Text, nullable=True)
    relevance_score = db.Column(db.Float, nullable=True)
    retrieved_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # --- Retrieval-pipeline enrichment (all nullable) ---
    # `published_date` is what the article says about itself; missing on
    # a lot of Tavily results (Reuters embargo pages, PDFs, etc).
    published_date = db.Column(db.DateTime, nullable=True)
    # Bare hostname (no `www.`, no port) — useful for grouping / weighting
    # by source without re-parsing the URL each time.
    source_domain = db.Column(db.String(128), nullable=True)
    # Stance vs the event's YES outcome. One of SUPPORT / REFUTE /
    # NEUTRAL (matches retrieval/stance.py labels). NULL when the stance
    # classifier didn't run (no LLM, or item skipped).
    stance = db.Column(db.String(16), nullable=True)
    stance_confidence = db.Column(db.Float, nullable=True)
    # Composite score in [0, 1] combining relevance, freshness, and
    # source weight — the value the pipeline sorted on when picking top-N.
    final_score = db.Column(db.Float, nullable=True)

    __table_args__ = (
        # Hot event page query: latest evidence/agent-analysis rows for one
        # event, ordered newest first. SQLite can scan this index backwards.
        db.Index("ix_evidence_event_retrieved", "event_id", "retrieved_at"),
    )

    def __repr__(self):
        return f"<Evidence agent={self.agent_id} event={self.event_id} url={self.url!r}>"
