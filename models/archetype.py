"""AgentArchetype model: the heterogeneity template for a population.

An archetype is a *shared* behavioral blueprint. A population of 10,000
Agents is generated from ~100 archetypes: each Agent copies its
archetype's parameters and then perturbs them with a per-Agent seeded
draw (see `services.population_service`). This is what lets us build a
heterogeneous 10k-Agent population from at most ~100 LLM calls (and zero
when `use_llm=False`) — the LLM, when used at all, only ever writes
archetypes, never individual Agents.

Every archetype shares the same ultimate objective —
`maximize_long_term_virtual_wealth` — stored on `objective`. Difference
between archetypes lives entirely in the JSON parameter blocks:

    expertise_json         — {category: skill in [0,1]}, forecasting edge
    cognitive_biases_json  — {bias_name: strength in [0,1]}
    source_trust_json      — {source_domain_or_class: trust in [0,1]}
    attention_profile_json — what wakes this archetype & how strongly
    risk_profile_json      — risk aversion, Kelly fraction, exposure caps
    strategy_parameters_json — edge thresholds, sizing, family knobs

The JSON blocks are stored as TEXT via `db.JSON` (SQLite renders it as
TEXT; Postgres as JSONB later) so the schema is portable and additive.
"""

from datetime import datetime

from models.database import db

# The single ultimate objective every archetype and Agent shares.
OBJECTIVE_MAXIMIZE_WEALTH = "maximize_long_term_virtual_wealth"


class AgentArchetype(db.Model):
    __tablename__ = "agent_archetypes"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    # Strategy family (see agents.strategy_families) — evidence_value,
    # momentum, contrarian, market_following, specialist, mean_reversion,
    # adaptive, retail_like.
    strategy_type = db.Column(db.String(64), nullable=False)
    # Always OBJECTIVE_MAXIMIZE_WEALTH for now, but stored explicitly so a
    # future multi-objective experiment doesn't need a migration.
    objective = db.Column(
        db.String(64), default=OBJECTIVE_MAXIMIZE_WEALTH, nullable=False
    )

    # --- heterogeneity parameter blocks (all JSON) ---
    expertise_json = db.Column(db.JSON, nullable=True)
    cognitive_biases_json = db.Column(db.JSON, nullable=True)
    source_trust_json = db.Column(db.JSON, nullable=True)
    attention_profile_json = db.Column(db.JSON, nullable=True)
    risk_profile_json = db.Column(db.JSON, nullable=True)
    strategy_parameters_json = db.Column(db.JSON, nullable=True)

    # Version tag for the persona prompt that produced/decorates this
    # archetype — lets the cache-metadata layer rotate the stable prefix
    # when the prompt contract changes.
    persona_prompt_version = db.Column(db.String(32), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Reverse side of Agent.archetype (Agent defines the FK). lazy=dynamic
    # so a huge population doesn't eagerly load every Agent when an
    # archetype is touched.
    agents = db.relationship(
        "Agent",
        backref="archetype",
        lazy="dynamic",
    )

    def __repr__(self):
        return (
            f"<AgentArchetype {self.id} {self.name!r} "
            f"strategy={self.strategy_type}>"
        )
