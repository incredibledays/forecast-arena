"""Agent model: a virtual AI trader.

Every Agent's ultimate objective is `maximize_long_term_virtual_wealth`
(stored on `objective`). Agents are generated from an `AgentArchetype`
(see `models.archetype` and `services.population_service`): the Agent
copies the archetype's parameter blocks and then perturbs them with a
per-Agent seeded draw, so agents from the same archetype are related but
never identical.

Schema note: the original Agent had only
`id/name/strategy_type/virtual_cash/initial_cash/risk_profile/created_at`.
The population columns below are ALL nullable / defaulted so an existing
DB keeps loading; `services.population_service.ensure_agent_schema()`
adds the missing columns in place via ALTER TABLE (the repo has no
Alembic). Existing rows keep their cash and get sensible defaults for
the new fields.
"""

from datetime import datetime

from models.database import db

# Re-exported for callers that want the constant without importing the
# archetype module. Kept in sync with models.archetype.
OBJECTIVE_MAXIMIZE_WEALTH = "maximize_long_term_virtual_wealth"


class Agent(db.Model):
    __tablename__ = "agents"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    strategy_type = db.Column(db.String(64), nullable=True)
    virtual_cash = db.Column(db.Float, default=10000.0, nullable=False)
    initial_cash = db.Column(db.Float, default=10000.0, nullable=False)
    risk_profile = db.Column(db.String(32), nullable=True)  # e.g. low/med/high
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Trade counters — maintained incrementally by MarketService._record_trade.
    # The leaderboard used to compute these with a `GROUP BY agent_id` full
    # table scan on every hit; on a warm DB with hundreds of thousands of
    # trades that was one of the two remaining hot spots (the other being
    # per-event volume on the index page). Backfilled on startup by
    # `ensure_perf_indexes()` so existing DBs keep working.
    total_trades = db.Column(db.Integer, default=0, nullable=False)
    non_hold_trades = db.Column(db.Integer, default=0, nullable=False)

    # --- population / archetype linkage (all nullable for back-compat) ---
    archetype_id = db.Column(
        db.Integer, db.ForeignKey("agent_archetypes.id"), nullable=True, index=True
    )
    objective = db.Column(
        db.String(64), default=OBJECTIVE_MAXIMIZE_WEALTH, nullable=True
    )
    # Per-Agent deterministic seed — the draw that produced this Agent's
    # perturbations. Storing it makes the population fully reproducible /
    # auditable.
    random_seed = db.Column(db.BigInteger, nullable=True)
    status = db.Column(db.String(16), default="active", nullable=True)

    # --- risk / sizing parameters (perturbed from archetype) ---
    risk_aversion = db.Column(db.Float, nullable=True)
    kelly_fraction = db.Column(db.Float, nullable=True)
    max_event_exposure = db.Column(db.Float, nullable=True)
    max_total_exposure = db.Column(db.Float, nullable=True)
    max_drawdown_tolerance = db.Column(db.Float, nullable=True)

    # --- trade-decision thresholds ---
    entry_edge_threshold = db.Column(db.Float, nullable=True)
    exit_edge_threshold = db.Column(db.Float, nullable=True)
    reversal_edge_threshold = db.Column(db.Float, nullable=True)
    minimum_trade_notional = db.Column(db.Float, nullable=True)

    # --- attention / wake-up shape (consumed by a LATER scheduling phase;
    # stored now so the population is complete) ---
    base_wakeup_rate_per_day = db.Column(db.Float, nullable=True)
    event_sensitivity = db.Column(db.Float, nullable=True)
    price_sensitivity = db.Column(db.Float, nullable=True)
    portfolio_sensitivity = db.Column(db.Float, nullable=True)
    information_delay_seconds = db.Column(db.Float, nullable=True)
    # Coarse activity band (ultra_active / active / normal / low_frequency /
    # very_low_frequency). Stored as a label alongside the continuous
    # base_wakeup_rate_per_day so the scheduler can bucket cheaply. We
    # store RATES, not fixed intervals — the band is a summary of the rate.
    activity_group = db.Column(db.String(24), nullable=True, index=True)

    # Per-Agent persona tweaks layered over the archetype's persona.
    persona_overrides_json = db.Column(db.JSON, nullable=True)

    # Relationships
    trades = db.relationship(
        "Trade", backref="agent", lazy="dynamic", cascade="all, delete-orphan"
    )
    positions = db.relationship(
        "Position", backref="agent", lazy="dynamic", cascade="all, delete-orphan"
    )
    evidence = db.relationship(
        "Evidence", backref="agent", lazy="dynamic", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Agent {self.id} {self.name!r} cash={self.virtual_cash:.2f}>"
