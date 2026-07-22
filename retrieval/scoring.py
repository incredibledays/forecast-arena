"""Composite evidence scoring: relevance + time decay + source weight.

Tavily's own `relevance_score` captures topical similarity but misses
two things that matter for a real-time prediction market:
  * time-of-publication — a two-day-old scoop dominates a two-year-old
    background piece even if the older piece is nominally more relevant;
  * source quality — a Reuters wire is worth more than an unknown blog.

We combine the three into a single `final_score` in [0, 1] used for
sorting and top-N selection.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from retrieval.utils import domain_of


# Weights: pinned in code so behavior is auditable without env config.
# They sum to 1.0 for readability but the composite is already clamped.
_W_RELEVANCE = 0.5
_W_TIME = 0.3
_W_SOURCE = 0.2

# Default half-life-ish parameter for the exponential decay, in hours.
# 72h is a sweet spot for weekly-cycle news; individual callers can pass
# a smaller tau for markets that close soon.
_DEFAULT_TAU_HOURS = 72.0

# When we have no published_date, don't punish the item — treat it as a
# neutral 0.5. Missing dates are common in Tavily results.
_TIME_UNKNOWN = 0.5


# ~30 domains where "if Reuters says it, take it seriously" applies.
# Weights are hand-picked in [0.3, 1.0]; anything not listed scores 0.0
# (no penalty, no bonus). Add domains here — no schema change needed.
SOURCE_WEIGHTS = {
    # Global wires
    "reuters.com": 1.0,
    "apnews.com": 1.0,
    "bloomberg.com": 1.0,
    "ft.com": 0.9,
    "wsj.com": 0.9,
    "nytimes.com": 0.85,
    "washingtonpost.com": 0.8,
    "economist.com": 0.85,
    # Tech-specific but widely-cited
    "theverge.com": 0.6,
    "arstechnica.com": 0.6,
    "techcrunch.com": 0.55,
    "wired.com": 0.6,
    # Broadcast
    "bbc.com": 0.85,
    "bbc.co.uk": 0.85,
    "cnbc.com": 0.7,
    "cnn.com": 0.65,
    "aljazeera.com": 0.65,
    # Markets
    "marketwatch.com": 0.65,
    "seekingalpha.com": 0.45,
    "coindesk.com": 0.55,
    # Official / primary
    "sec.gov": 1.0,
    "federalreserve.gov": 1.0,
    "whitehouse.gov": 0.9,
    "openai.com": 0.8,
    "anthropic.com": 0.8,
    "deepmind.google": 0.8,
    "ai.meta.com": 0.75,
    # Aggregators / meta
    "polymarket.com": 0.5,
    "kalshi.com": 0.5,
    "manifold.markets": 0.4,
}


def source_weight(url: str) -> float:
    """Return the domain's weight in [0, 1] or 0.0 if unlisted."""
    d = domain_of(url)
    if not d:
        return 0.0
    # Try full match first, then progressively shorter suffixes so
    # `edition.cnn.com` maps to `cnn.com`.
    if d in SOURCE_WEIGHTS:
        return SOURCE_WEIGHTS[d]
    parts = d.split(".")
    for i in range(1, len(parts) - 1):
        suffix = ".".join(parts[i:])
        if suffix in SOURCE_WEIGHTS:
            return SOURCE_WEIGHTS[suffix]
    return 0.0


def time_decay(
    published: Optional[datetime],
    now: Optional[datetime] = None,
    tau_hours: float = _DEFAULT_TAU_HOURS,
) -> float:
    """Exponential decay by hours since publication, capped to [0, 1].

    Returns 0.5 (`_TIME_UNKNOWN`) when the date is missing so that stale
    metadata doesn't get punished — an item without a date is treated
    the same as a 50-hour-old one under tau=72.
    """
    if published is None:
        return _TIME_UNKNOWN
    now = now or datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    delta = (now - published).total_seconds() / 3600.0
    if delta < 0:                                   # future-dated: treat as fresh
        return 1.0
    tau = max(1e-6, float(tau_hours))
    return max(0.0, min(1.0, math.exp(-delta / tau)))


def composite_score(
    relevance: float,
    published: Optional[datetime],
    url: str,
    now: Optional[datetime] = None,
    tau_hours: float = _DEFAULT_TAU_HOURS,
) -> float:
    """Weighted sum of relevance / freshness / source weight.

    Every component is clamped to [0, 1]; the final value is clamped
    again for safety. Returns a plain float — callers stash it on the
    evidence dict as `final_score`.
    """
    try:
        r = float(relevance)
    except (TypeError, ValueError):
        r = 0.0
    r = max(0.0, min(1.0, r))
    t = time_decay(published, now=now, tau_hours=tau_hours)
    s = source_weight(url)
    return max(0.0, min(1.0, _W_RELEVANCE * r + _W_TIME * t + _W_SOURCE * s))


__all__ = [
    "SOURCE_WEIGHTS",
    "source_weight",
    "time_decay",
    "composite_score",
]
