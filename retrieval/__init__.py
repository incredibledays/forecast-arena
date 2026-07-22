"""Real-time search / retrieval providers.

Import order matters: `base` first so `TavilyProvider` can import its
schema helpers.
"""

from retrieval.base import (
    EVIDENCE_KEYS,
    EVIDENCE_OPTIONAL_KEYS,
    SearchProvider,
    is_valid_evidence,
)
from retrieval.scoring import (
    SOURCE_WEIGHTS,
    composite_score,
    source_weight,
    time_decay,
)
from retrieval.tavily_provider import TavilyProvider
from retrieval.utils import (
    clean_snippet,
    domain_of,
    normalize_url,
    parse_published_date,
    title_key,
)

__all__ = [
    # Core types
    "SearchProvider",
    "TavilyProvider",
    "EVIDENCE_KEYS",
    "EVIDENCE_OPTIONAL_KEYS",
    "is_valid_evidence",
    # Utils
    "normalize_url",
    "domain_of",
    "clean_snippet",
    "title_key",
    "parse_published_date",
    # Scoring
    "SOURCE_WEIGHTS",
    "source_weight",
    "time_decay",
    "composite_score",
]
