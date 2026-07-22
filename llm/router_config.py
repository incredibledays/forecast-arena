"""Router + budget configuration, loaded from environment.

Kept separate from `llm.config.LLMConfig` so the existing
provider-agnostic client config is untouched (compatibility rule #9).
`LLMConfig` still describes the single legacy client; `RouterConfig`
adds the tier→(provider, model) mapping, batch defaults, and budgets
the router needs.

Tiers never name a model in code — the mapping is entirely env-driven:

    LLM_TIER_FAST_PROVIDER / LLM_TIER_FAST_MODEL
    LLM_TIER_BALANCED_PROVIDER / LLM_TIER_BALANCED_MODEL
    LLM_TIER_STRONG_PROVIDER / LLM_TIER_STRONG_MODEL
    LLM_TIER_SPECIALIST_PROVIDER / LLM_TIER_SPECIALIST_MODEL
    LLM_TIER_LOCAL_PROVIDER / LLM_TIER_LOCAL_MODEL

A tier with no configured model is "unconfigured": the router will not
route to it silently (critical for LOCAL and SPECIALIST).

Budgets (all optional; unset ⇒ unlimited):

    LLM_BUDGET_DAILY_INPUT_TOKENS
    LLM_BUDGET_DAILY_OUTPUT_TOKENS
    LLM_BUDGET_DAILY_STRONG_TOKENS
    LLM_BUDGET_PER_MARKET_DAILY_TOKENS
    LLM_BUDGET_PER_BUNDLE_TOKENS
    LLM_MAX_RETRIES              (default 2)
    LLM_MAX_ESCALATIONS          (default 1)
    LLM_MAX_CONCURRENT_STRONG    (default 4)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from llm.tiers import Tier


def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class TierSettings:
    """Concrete provider/model bound to one logical tier.

    `configured` is False when no model is set — the router treats such
    a tier as unavailable and will not select it silently.
    """

    tier: Tier
    provider: Optional[str] = None
    model: Optional[str] = None

    @property
    def configured(self) -> bool:
        return bool(self.model)

    def public_dict(self) -> Dict[str, object]:
        return {
            "tier": self.tier.value,
            "provider": self.provider,
            "model": self.model,
            "configured": self.configured,
        }


@dataclass(frozen=True)
class BudgetConfig:
    """Daily / scoped token budgets and bounded-work limits.

    A `None` budget means unlimited. Bounds (retries, escalations,
    concurrency) always have safe integer defaults so nothing is ever
    "retry forever".
    """

    daily_input_tokens: Optional[int] = None
    daily_output_tokens: Optional[int] = None
    daily_strong_tokens: Optional[int] = None
    per_market_daily_tokens: Optional[int] = None
    per_bundle_tokens: Optional[int] = None

    max_retries: int = 2
    max_escalations: int = 1
    max_concurrent_strong: int = 4

    def public_dict(self) -> Dict[str, object]:
        return {
            "daily_input_tokens": self.daily_input_tokens,
            "daily_output_tokens": self.daily_output_tokens,
            "daily_strong_tokens": self.daily_strong_tokens,
            "per_market_daily_tokens": self.per_market_daily_tokens,
            "per_bundle_tokens": self.per_bundle_tokens,
            "max_retries": self.max_retries,
            "max_escalations": self.max_escalations,
            "max_concurrent_strong": self.max_concurrent_strong,
        }


@dataclass
class RouterConfig:
    """Everything the router needs beyond the legacy client config."""

    tiers: Dict[Tier, TierSettings] = field(default_factory=dict)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    # Default provider used when a tier omits its own provider — falls
    # back to the legacy LLM_PROVIDER so single-provider deploys "just
    # work" by only setting per-tier models.
    default_provider: str = "openai"

    def tier_settings(self, tier: Tier) -> TierSettings:
        return self.tiers.get(tier, TierSettings(tier=tier))

    def is_tier_configured(self, tier: Tier) -> bool:
        return self.tier_settings(tier).configured

    def public_dict(self) -> Dict[str, object]:
        return {
            "default_provider": self.default_provider,
            "tiers": {t.value: s.public_dict() for t, s in self.tiers.items()},
            "budget": self.budget.public_dict(),
        }


# Sensible fallbacks so a deployment that sets ONLY a legacy LLM_MODEL
# still gets a working FAST/BALANCED/STRONG mapping (all pointing at
# that one model). LOCAL/SPECIALIST stay unconfigured unless set.
def _tier_from_env(tier: Tier, default_provider: str, legacy_model: Optional[str]) -> TierSettings:
    provider = _env(f"LLM_TIER_{tier.value}_PROVIDER") or default_provider
    model = _env(f"LLM_TIER_{tier.value}_MODEL")
    if model is None and tier in (Tier.FAST, Tier.BALANCED, Tier.STRONG):
        # Fall back to the legacy single model for the cost-axis tiers
        # only — never for LOCAL/SPECIALIST, which must be explicit.
        model = legacy_model
    return TierSettings(tier=tier, provider=provider if model else provider, model=model)


def load_router_config() -> RouterConfig:
    """Build a `RouterConfig` from environment variables.

    Never raises: missing tiers are simply "unconfigured". The legacy
    `LLM_PROVIDER` / `LLM_MODEL` (and `OPENAI_MODEL`) supply the default
    provider and the cost-axis fallback model.
    """
    default_provider = _env("LLM_PROVIDER") or "openai"
    legacy_model = _env("LLM_MODEL") or _env("OPENAI_MODEL")

    tiers: Dict[Tier, TierSettings] = {}
    for tier in Tier:
        tiers[tier] = _tier_from_env(tier, default_provider, legacy_model)

    budget = BudgetConfig(
        daily_input_tokens=_env_int("LLM_BUDGET_DAILY_INPUT_TOKENS", None),
        daily_output_tokens=_env_int("LLM_BUDGET_DAILY_OUTPUT_TOKENS", None),
        daily_strong_tokens=_env_int("LLM_BUDGET_DAILY_STRONG_TOKENS", None),
        per_market_daily_tokens=_env_int("LLM_BUDGET_PER_MARKET_DAILY_TOKENS", None),
        per_bundle_tokens=_env_int("LLM_BUDGET_PER_BUNDLE_TOKENS", None),
        max_retries=_env_int("LLM_MAX_RETRIES", 2) or 0,
        max_escalations=_env_int("LLM_MAX_ESCALATIONS", 1) or 0,
        max_concurrent_strong=_env_int("LLM_MAX_CONCURRENT_STRONG", 4) or 1,
    )

    return RouterConfig(
        tiers=tiers, budget=budget, default_provider=default_provider
    )


__all__ = [
    "TierSettings",
    "BudgetConfig",
    "RouterConfig",
    "load_router_config",
]
