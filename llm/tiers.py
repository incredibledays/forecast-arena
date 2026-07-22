"""Logical model tiers, batch modes, and urgency levels.

These are *logical* tiers — they never name a commercial model in
business logic. The mapping from a tier to a concrete provider/model
lives entirely in configuration (see `llm.config.RouterConfig` and the
`LLM_TIER_*` env vars). Router code and callers reason in tiers; only
the config layer knows that (say) BALANCED means "gpt-4o-mini on
openai" for one deployment and "llama3.1:70b on a local vLLM" for
another.
"""

from __future__ import annotations

import enum


class Tier(str, enum.Enum):
    """Logical capability/cost tiers, cheapest → strongest.

    LOCAL is orthogonal to the fast→strong axis: it means "prefer a
    configured local provider for high-volume low-latency work". The
    router only ever selects LOCAL when a local provider is actually
    configured (never silently).
    """

    FAST = "FAST"
    BALANCED = "BALANCED"
    STRONG = "STRONG"
    SPECIALIST = "SPECIALIST"
    LOCAL = "LOCAL"


# Escalation order used by budget/failure fallback. LOCAL and SPECIALIST
# are intentionally absent from the escalation ladder: escalation walks
# the cost/capability axis (FAST → BALANCED → STRONG), while LOCAL and
# SPECIALIST are selected by explicit intent, not by climbing.
ESCALATION_LADDER = (Tier.FAST, Tier.BALANCED, Tier.STRONG)

# The reverse — cheaper alternatives to fall *back* to when a budget is
# exhausted. Ordered strongest → cheapest so `de_escalate` can find the
# next-cheaper tier.
DE_ESCALATION_LADDER = (Tier.STRONG, Tier.BALANCED, Tier.FAST)


class BatchMode(str, enum.Enum):
    """How a request should be dispatched.

    REALTIME     — send immediately; latency-critical.
    MICRO_BATCH  — coalesce with sibling requests over a short window.
    ASYNC_BATCH  — defer to an offline / async batch queue.

    The router only *labels* the request with a mode; actual batching is
    the dispatcher's job (not implemented in this phase).
    """

    REALTIME = "REALTIME"
    MICRO_BATCH = "MICRO_BATCH"
    ASYNC_BATCH = "ASYNC_BATCH"


class Urgency(str, enum.Enum):
    """Caller-declared urgency, independent of task type.

    Used as an override signal: a HIGH-urgency routine task is pulled
    into REALTIME even if its task type would normally micro-batch.
    """

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


def next_stronger(tier: Tier) -> "Tier | None":
    """Return the next tier up the escalation ladder, or None at the top."""
    if tier not in ESCALATION_LADDER:
        return None
    idx = ESCALATION_LADDER.index(tier)
    if idx + 1 >= len(ESCALATION_LADDER):
        return None
    return ESCALATION_LADDER[idx + 1]


def next_cheaper(tier: Tier) -> "Tier | None":
    """Return the next cheaper tier for budget fallback, or None at the floor."""
    if tier not in DE_ESCALATION_LADDER:
        # SPECIALIST / LOCAL fall back onto the cost axis at BALANCED.
        return Tier.BALANCED
    idx = DE_ESCALATION_LADDER.index(tier)
    if idx + 1 >= len(DE_ESCALATION_LADDER):
        return None
    return DE_ESCALATION_LADDER[idx + 1]


__all__ = [
    "Tier",
    "BatchMode",
    "Urgency",
    "ESCALATION_LADDER",
    "DE_ESCALATION_LADDER",
    "next_stronger",
    "next_cheaper",
]
