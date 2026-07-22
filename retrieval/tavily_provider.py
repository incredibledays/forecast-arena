"""Tavily-backed SearchProvider.

Behavior:
    * Missing TAVILY_API_KEY  → warn once, `search()` returns [].
    * Missing tavily-python   → warn once, `search()` returns [].
    * Any Tavily error        → warn, return [] — never raises to the caller.

Results are normalized to the schema in retrieval/base.py so downstream
code doesn't need to know Tavily's raw field names. The normalize step
also populates the optional enrichment fields the retrieval pipeline
uses (published_date, source_domain, url_normalized) — later stages
add stance / final_score on top.
"""

import os
import sys
from typing import List

from dotenv import load_dotenv

from retrieval.base import EVIDENCE_KEYS, SearchProvider
from retrieval.utils import (
    clean_snippet,
    domain_of,
    normalize_url,
    parse_published_date,
)

load_dotenv()

# Import the SDK lazily-but-eagerly-once, so a missing package is a clean
# warning rather than an ImportError at every call site.
try:
    from tavily import TavilyClient  # type: ignore
    _TAVILY_IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001 — surface any import problem the same way
    TavilyClient = None  # type: ignore
    _TAVILY_IMPORT_ERROR = _exc


# Cap the summary so we don't stash entire pages in DB / logs. The clean
# pass runs BEFORE the cut so we don't waste characters on boilerplate.
_SUMMARY_MAX_CHARS = 1000


class TavilyProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str = None, search_depth: str = "basic"):
        """Initialize the provider.

        `api_key` defaults to the TAVILY_API_KEY env var. If the key is
        missing OR the tavily-python package isn't importable, the
        provider stays in a disabled state and `search()` returns [].
        """
        self._api_key = api_key or os.getenv("TAVILY_API_KEY") or None
        self._search_depth = search_depth
        self._client = None
        self._disabled_reason = None

        if TavilyClient is None:
            self._disabled_reason = (
                f"tavily-python not installed ({_TAVILY_IMPORT_ERROR!r}); "
                "run `pip install tavily-python`"
            )
        elif not self._api_key:
            self._disabled_reason = (
                "TAVILY_API_KEY is not set; retrieval disabled"
            )
        else:
            try:
                self._client = TavilyClient(api_key=self._api_key)
            except Exception as exc:  # noqa: BLE001
                self._disabled_reason = f"failed to init TavilyClient: {exc}"

        if self._disabled_reason:
            print(f"[TavilyProvider] {self._disabled_reason}", file=sys.stderr)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def search(self, query: str, max_results: int = 5) -> List[dict]:
        if not self.enabled:
            return []
        if not query or not str(query).strip():
            return []

        max_results = max(1, min(int(max_results), 20))
        try:
            raw = self._client.search(
                query=str(query),
                max_results=max_results,
                search_depth=self._search_depth,
            )
        except Exception as exc:  # noqa: BLE001 — retrieval must not crash callers
            print(
                f"[TavilyProvider] search failed for {query!r}: {exc}",
                file=sys.stderr,
            )
            return []

        return _normalize(raw)


def _normalize(raw) -> List[dict]:
    """Coerce Tavily's response into the shared evidence schema.

    Populates both the required four keys and the optional enrichment
    keys the pipeline uses (`published_date`, `source_domain`,
    `url_normalized`). Later stages fill in `stance` / `final_score`.
    """
    if not isinstance(raw, dict):
        return []
    items = raw.get("results") or []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        # Tavily typically returns a `content` snippet; fall back to
        # `raw_content` (truncated) if content is absent. Clean BEFORE
        # truncating so we don't spend our char budget on boilerplate.
        raw_summary = item.get("content") or item.get("raw_content") or ""
        summary = clean_snippet(str(raw_summary))[:_SUMMARY_MAX_CHARS].strip()
        score = item.get("score")
        try:
            score = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0

        if not (title or url or summary):
            continue

        published = parse_published_date(item.get("published_date"))

        out.append(
            {
                "title": title,
                "url": url,
                "content_summary": summary,
                "relevance_score": score,
                # --- optional enrichment fields ---
                "published_date": published,
                "source_domain": domain_of(url),
                # Dedup key. Keep the original `url` for display/audit.
                "url_normalized": normalize_url(url),
                # Placeholders — filled by later pipeline stages. Present
                # so downstream code can rely on `.get()` returning None
                # instead of a missing key on non-enriched items.
                "stance": None,
                "stance_confidence": None,
                "final_score": None,
            }
        )
    # Ensure every returned item has all the required keys.
    return [x for x in out if all(k in x for k in EVIDENCE_KEYS)]
