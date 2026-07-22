"""SearchProvider interface.

Any concrete provider (Tavily, Serper, Bing, a mock, ...) must return a
list of dicts matching the shared evidence schema. Consumers can then
treat evidence uniformly regardless of source.

Schema evolution: the original four keys (`title`, `url`,
`content_summary`, `relevance_score`) are still required — that's the
back-compat surface. Newer keys added by the retrieval pipeline
(`published_date`, `source_domain`, `stance`, `stance_confidence`,
`final_score`) are optional and validated only *when present*.
"""

from datetime import datetime
from typing import List


# Required evidence keys — every provider MUST populate these four.
EVIDENCE_KEYS = ("title", "url", "content_summary", "relevance_score")

# Optional enrichment keys populated by the retrieval pipeline. Kept
# here so a single import (`from retrieval.base import ...`) tells you
# the full evidence surface.
EVIDENCE_OPTIONAL_KEYS = (
    "published_date",       # datetime | None
    "source_domain",        # str | ''
    "url_normalized",       # str — dedup key, NOT for display
    "stance",               # "SUPPORT" | "REFUTE" | "NEUTRAL" | None
    "stance_confidence",    # float in [0, 1] | None
    "final_score",          # float in [0, 1] | None
    "query",                # the query that produced this item, for audit
)


class SearchProvider:
    """Base class for real-time search backends."""

    #: Short identifier for logs / provenance columns (e.g. "tavily").
    name: str = "base"

    def search(self, query: str, max_results: int = 5) -> List[dict]:
        raise NotImplementedError


def is_valid_evidence(item: dict) -> bool:
    """True if `item` matches the shared schema.

    Required keys must be present with plausible types. Optional keys
    are validated only when the caller set them — a bare four-field item
    from a mock provider still passes.
    """
    if not isinstance(item, dict):
        return False
    if not all(k in item for k in EVIDENCE_KEYS):
        return False
    if not isinstance(item["title"], str) or not isinstance(item["url"], str):
        return False
    if not isinstance(item["content_summary"], str):
        return False
    try:
        float(item["relevance_score"])
    except (TypeError, ValueError):
        return False

    # Optional-key type spot-checks — cheap and catches upstream bugs
    # where someone stashes the wrong type on an evidence dict.
    if "published_date" in item and item["published_date"] is not None:
        if not isinstance(item["published_date"], datetime):
            return False
    for k in ("source_domain", "url_normalized", "stance", "query"):
        if k in item and item[k] is not None and not isinstance(item[k], str):
            return False
    for k in ("stance_confidence", "final_score"):
        if k in item and item[k] is not None:
            try:
                float(item[k])
            except (TypeError, ValueError):
                return False
    return True


__all__ = [
    "SearchProvider",
    "EVIDENCE_KEYS",
    "EVIDENCE_OPTIONAL_KEYS",
    "is_valid_evidence",
]
