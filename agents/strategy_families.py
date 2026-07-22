"""Strategy families + their archetype parameter templates.

A *strategy family* is the coarse behavioral bucket an archetype belongs
to. Each family ships a base template of the six heterogeneity blocks an
`AgentArchetype` carries (expertise / cognitive_biases / source_trust /
attention_profile / risk_profile / strategy_parameters). The population
service starts from a family template, then jitters it per-archetype and
again per-Agent so no two rows are identical.

Every family shares the ONE ultimate objective:

    maximize_long_term_virtual_wealth

Families differ only in *how* they pursue it:

  evidence_value    — trades the gap between researched belief and price;
                      disciplined risk, high research expertise.
  momentum          — follows recent price direction.
  contrarian        — fades crowded extremes.
  market_following  — trusts the market price; trades small deviations.
  specialist        — deep edge in a narrow category, weak elsewhere.
  mean_reversion    — bets prices revert toward a prior/anchor.
  adaptive          — blends signals; moderate everything, wider attention.
  retail_like       — genuinely tries to profit, but with weaker
                      calibration, stronger herding + recency bias, and
                      looser risk control. NOT a random/noise trader:
                      it still buys the side it believes and sizes by a
                      (miscalibrated) edge — it just does so badly.

The templates below are deliberately deterministic and LLM-free: a full
10k population can be built from these with zero model calls. When
`use_llm=True`, the LLM only *enriches* archetype descriptions/personas
— the numeric parameter blocks still come from (or are validated
against) these templates so bounds hold.
"""

from __future__ import annotations

from typing import Dict, List

# The single ultimate objective (kept in sync with models.archetype).
OBJECTIVE_MAXIMIZE_WEALTH = "maximize_long_term_virtual_wealth"

# Current persona-prompt contract version for cache-key rotation.
PERSONA_PROMPT_VERSION = "persona-v1"

# Canonical expertise categories. Archetypes carry a skill in [0,1] per
# category; the population perturbs these per-Agent.
EXPERTISE_CATEGORIES: List[str] = [
    "ai", "markets", "politics", "sports", "climate", "tech", "macro",
]

# Canonical cognitive biases, strength in [0,1] (0 = unbiased).
COGNITIVE_BIASES: List[str] = [
    "overconfidence", "herding", "recency", "anchoring",
    "confirmation", "loss_aversion",
]

# Canonical source-trust classes, trust in [0,1].
SOURCE_TRUST_CLASSES: List[str] = [
    "official", "wire", "mainstream", "social", "blog",
]


STRATEGY_FAMILIES: List[str] = [
    "evidence_value",
    "momentum",
    "contrarian",
    "market_following",
    "specialist",
    "mean_reversion",
    "adaptive",
    "retail_like",
]


def _mk(
    *,
    strategy_type: str,
    description: str,
    expertise: Dict[str, float],
    biases: Dict[str, float],
    source_trust: Dict[str, float],
    attention: Dict[str, float],
    risk: Dict[str, float],
    strategy_params: Dict[str, float],
) -> Dict:
    """Assemble one family template with the shared objective attached."""
    return {
        "strategy_type": strategy_type,
        "objective": OBJECTIVE_MAXIMIZE_WEALTH,
        "description": description,
        "persona_prompt_version": PERSONA_PROMPT_VERSION,
        "expertise_json": dict(expertise),
        "cognitive_biases_json": dict(biases),
        "source_trust_json": dict(source_trust),
        "attention_profile_json": dict(attention),
        "risk_profile_json": dict(risk),
        "strategy_parameters_json": dict(strategy_params),
    }


# --- per-family base templates ---------------------------------------
# Values are the archetype *centers*; the population service jitters
# around them. All numbers are in documented ranges (skills/biases/trust
# in [0,1]; risk/threshold knobs in their natural units) and are clamped
# again after perturbation so no Agent leaves bounds.

_FAMILY_TEMPLATES: Dict[str, Dict] = {
    "evidence_value": _mk(
        strategy_type="evidence_value",
        description=(
            "Researches evidence and trades the gap between a "
            "well-calibrated belief and the market price. Disciplined "
            "risk, high research skill, trusts primary/official sources."
        ),
        expertise={"ai": 0.7, "markets": 0.7, "macro": 0.65, "politics": 0.55,
                   "tech": 0.6, "climate": 0.5, "sports": 0.4},
        biases={"overconfidence": 0.2, "herding": 0.15, "recency": 0.2,
                "anchoring": 0.25, "confirmation": 0.2, "loss_aversion": 0.3},
        source_trust={"official": 0.9, "wire": 0.8, "mainstream": 0.65,
                      "social": 0.3, "blog": 0.25},
        attention={"event": 0.8, "price": 0.4, "portfolio": 0.6, "base_rate": 3.0},
        risk={"risk_aversion": 0.5, "kelly_fraction": 0.4,
              "max_event_exposure": 0.15, "max_total_exposure": 0.6,
              "max_drawdown_tolerance": 0.35},
        strategy_params={"entry_edge_threshold": 0.08, "exit_edge_threshold": 0.03,
                         "reversal_edge_threshold": 0.15,
                         "minimum_trade_notional": 25.0},
    ),
    "momentum": _mk(
        strategy_type="momentum",
        description=(
            "Trades the direction of recent price movement; enters on "
            "trends, exits when momentum fades."
        ),
        expertise={"markets": 0.6, "tech": 0.55, "ai": 0.5, "macro": 0.45,
                   "politics": 0.4, "climate": 0.35, "sports": 0.45},
        biases={"overconfidence": 0.35, "herding": 0.5, "recency": 0.6,
                "anchoring": 0.2, "confirmation": 0.35, "loss_aversion": 0.25},
        source_trust={"official": 0.6, "wire": 0.6, "mainstream": 0.6,
                      "social": 0.55, "blog": 0.4},
        attention={"event": 0.5, "price": 0.9, "portfolio": 0.4, "base_rate": 6.0},
        risk={"risk_aversion": 0.4, "kelly_fraction": 0.5,
              "max_event_exposure": 0.18, "max_total_exposure": 0.7,
              "max_drawdown_tolerance": 0.4},
        strategy_params={"entry_edge_threshold": 0.06, "exit_edge_threshold": 0.03,
                         "reversal_edge_threshold": 0.12,
                         "minimum_trade_notional": 20.0},
    ),
    "contrarian": _mk(
        strategy_type="contrarian",
        description=(
            "Fades crowded extremes — buys the unpopular side when the "
            "price looks overreacted."
        ),
        expertise={"markets": 0.6, "macro": 0.55, "politics": 0.5, "ai": 0.5,
                   "tech": 0.5, "climate": 0.45, "sports": 0.4},
        biases={"overconfidence": 0.4, "herding": 0.1, "recency": 0.2,
                "anchoring": 0.45, "confirmation": 0.3, "loss_aversion": 0.35},
        source_trust={"official": 0.75, "wire": 0.65, "mainstream": 0.5,
                      "social": 0.35, "blog": 0.35},
        attention={"event": 0.55, "price": 0.85, "portfolio": 0.5, "base_rate": 4.0},
        risk={"risk_aversion": 0.55, "kelly_fraction": 0.35,
              "max_event_exposure": 0.14, "max_total_exposure": 0.6,
              "max_drawdown_tolerance": 0.35},
        strategy_params={"entry_edge_threshold": 0.1, "exit_edge_threshold": 0.04,
                         "reversal_edge_threshold": 0.1,
                         "minimum_trade_notional": 25.0},
    ),
    "market_following": _mk(
        strategy_type="market_following",
        description=(
            "Treats the market price as the best estimate; trades only "
            "small, low-conviction deviations. Low variance."
        ),
        expertise={"markets": 0.55, "macro": 0.5, "ai": 0.45, "tech": 0.45,
                   "politics": 0.4, "climate": 0.4, "sports": 0.4},
        biases={"overconfidence": 0.15, "herding": 0.6, "recency": 0.3,
                "anchoring": 0.5, "confirmation": 0.25, "loss_aversion": 0.3},
        source_trust={"official": 0.7, "wire": 0.65, "mainstream": 0.6,
                      "social": 0.4, "blog": 0.3},
        attention={"event": 0.4, "price": 0.7, "portfolio": 0.5, "base_rate": 3.0},
        risk={"risk_aversion": 0.6, "kelly_fraction": 0.3,
              "max_event_exposure": 0.1, "max_total_exposure": 0.5,
              "max_drawdown_tolerance": 0.3},
        strategy_params={"entry_edge_threshold": 0.05, "exit_edge_threshold": 0.02,
                         "reversal_edge_threshold": 0.2,
                         "minimum_trade_notional": 15.0},
    ),
    "specialist": _mk(
        strategy_type="specialist",
        description=(
            "Deep edge in one narrow category and near-blind elsewhere. "
            "Trades aggressively inside its domain, abstains outside it."
        ),
        # One category is set high by the population service per-archetype;
        # baseline here is a modest generalist floor.
        expertise={"ai": 0.4, "markets": 0.4, "politics": 0.4, "sports": 0.4,
                   "climate": 0.4, "tech": 0.4, "macro": 0.4},
        biases={"overconfidence": 0.45, "herding": 0.2, "recency": 0.25,
                "anchoring": 0.3, "confirmation": 0.45, "loss_aversion": 0.3},
        source_trust={"official": 0.85, "wire": 0.75, "mainstream": 0.55,
                      "social": 0.4, "blog": 0.45},
        attention={"event": 0.85, "price": 0.35, "portfolio": 0.5, "base_rate": 3.5},
        risk={"risk_aversion": 0.45, "kelly_fraction": 0.5,
              "max_event_exposure": 0.2, "max_total_exposure": 0.6,
              "max_drawdown_tolerance": 0.4},
        strategy_params={"entry_edge_threshold": 0.07, "exit_edge_threshold": 0.03,
                         "reversal_edge_threshold": 0.14,
                         "minimum_trade_notional": 30.0},
    ),
    "mean_reversion": _mk(
        strategy_type="mean_reversion",
        description=(
            "Bets that prices revert toward a prior/anchor after moves; "
            "sells strength, buys weakness within a band."
        ),
        expertise={"markets": 0.6, "macro": 0.55, "ai": 0.5, "tech": 0.5,
                   "politics": 0.45, "climate": 0.45, "sports": 0.4},
        biases={"overconfidence": 0.3, "herding": 0.2, "recency": 0.15,
                "anchoring": 0.55, "confirmation": 0.3, "loss_aversion": 0.4},
        source_trust={"official": 0.75, "wire": 0.7, "mainstream": 0.55,
                      "social": 0.35, "blog": 0.35},
        attention={"event": 0.5, "price": 0.8, "portfolio": 0.55, "base_rate": 4.0},
        risk={"risk_aversion": 0.55, "kelly_fraction": 0.35,
              "max_event_exposure": 0.13, "max_total_exposure": 0.55,
              "max_drawdown_tolerance": 0.33},
        strategy_params={"entry_edge_threshold": 0.09, "exit_edge_threshold": 0.03,
                         "reversal_edge_threshold": 0.1,
                         "minimum_trade_notional": 20.0},
    ),
    "adaptive": _mk(
        strategy_type="adaptive",
        description=(
            "Blends value, momentum, and market signals; moderate on "
            "every axis with broad attention. A robust generalist."
        ),
        expertise={"ai": 0.55, "markets": 0.6, "politics": 0.5, "sports": 0.45,
                   "climate": 0.5, "tech": 0.55, "macro": 0.55},
        biases={"overconfidence": 0.25, "herding": 0.3, "recency": 0.3,
                "anchoring": 0.3, "confirmation": 0.25, "loss_aversion": 0.3},
        source_trust={"official": 0.8, "wire": 0.72, "mainstream": 0.6,
                      "social": 0.4, "blog": 0.35},
        attention={"event": 0.65, "price": 0.65, "portfolio": 0.6, "base_rate": 5.0},
        risk={"risk_aversion": 0.5, "kelly_fraction": 0.4,
              "max_event_exposure": 0.15, "max_total_exposure": 0.6,
              "max_drawdown_tolerance": 0.38},
        strategy_params={"entry_edge_threshold": 0.07, "exit_edge_threshold": 0.03,
                         "reversal_edge_threshold": 0.13,
                         "minimum_trade_notional": 20.0},
    ),
    "retail_like": _mk(
        strategy_type="retail_like",
        description=(
            "A retail trader genuinely trying to profit but poorly "
            "calibrated: strong herding and recency bias, overconfident, "
            "loose risk control and thin research. Still directional and "
            "edge-seeking — not a random/noise trader."
        ),
        expertise={"ai": 0.35, "markets": 0.35, "politics": 0.35, "sports": 0.4,
                   "climate": 0.3, "tech": 0.4, "macro": 0.3},
        biases={"overconfidence": 0.7, "herding": 0.75, "recency": 0.7,
                "anchoring": 0.4, "confirmation": 0.6, "loss_aversion": 0.55},
        source_trust={"official": 0.5, "wire": 0.5, "mainstream": 0.6,
                      "social": 0.75, "blog": 0.65},
        attention={"event": 0.5, "price": 0.75, "portfolio": 0.3, "base_rate": 7.0},
        # Loose risk control: higher Kelly, higher per-event exposure,
        # higher drawdown tolerance, lower risk aversion.
        risk={"risk_aversion": 0.3, "kelly_fraction": 0.7,
              "max_event_exposure": 0.25, "max_total_exposure": 0.85,
              "max_drawdown_tolerance": 0.5},
        # Low entry threshold (trades on thin/weak edges) but still a real
        # positive threshold — it acts on belief, it just believes badly.
        strategy_params={"entry_edge_threshold": 0.03, "exit_edge_threshold": 0.05,
                         "reversal_edge_threshold": 0.08,
                         "minimum_trade_notional": 10.0},
    ),
}


def family_template(strategy_type: str) -> Dict:
    """Return a deep-ish copy of the family template for `strategy_type`."""
    key = (strategy_type or "").strip().lower()
    if key not in _FAMILY_TEMPLATES:
        known = ", ".join(STRATEGY_FAMILIES)
        raise ValueError(f"unknown strategy family {strategy_type!r}; known: {known}")
    tmpl = _FAMILY_TEMPLATES[key]
    # Copy nested dicts so callers can mutate freely.
    return {
        k: (dict(v) if isinstance(v, dict) else v)
        for k, v in tmpl.items()
    }


def all_family_templates() -> Dict[str, Dict]:
    return {fam: family_template(fam) for fam in STRATEGY_FAMILIES}


__all__ = [
    "STRATEGY_FAMILIES",
    "EXPERTISE_CATEGORIES",
    "COGNITIVE_BIASES",
    "SOURCE_TRUST_CLASSES",
    "OBJECTIVE_MAXIMIZE_WEALTH",
    "PERSONA_PROMPT_VERSION",
    "family_template",
    "all_family_templates",
]
