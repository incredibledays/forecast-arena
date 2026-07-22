"""RetrievalService — the retrieval pipeline for one trading round.

Pipeline (per event):
    expand queries → search each → dedup → clean → stance → score → cache

The service is instantiated once per round in `run_agents.py` and
shared across every NewsResearchAgent. That gives us two properties
the previous "each agent hits Tavily independently" design lacked:

  * **Fairness.** All agents on the same event see the same evidence,
    so cross-agent P&L differences reflect strategy, not stochastic
    Tavily ordering.
  * **Cost.** One expand + one round of Tavily calls + one stance call
    per event, regardless of how many agents are trading.

The public API is deliberately narrow — one method, `get_evidence`.
Callers hand in an event object and a max_items hint; the service
handles cache lookup, pipeline execution, and top-N trimming.

Failure policy matches the rest of the retrieval layer: never raise.
An LLM outage skips expansion + stance but returns raw Tavily results.
A Tavily outage returns []; the news agent falls back to no-evidence
forecasting.
"""

from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from llm.client import LLMClient
from retrieval import is_valid_evidence
from retrieval.base import SearchProvider
from retrieval.query_expander import expand_queries
from retrieval.scoring import composite_score
from retrieval.stance import classify_stances
from retrieval.utils import title_key


# Cap on how many raw results we pull PER expanded query before dedup /
# scoring. 4 × 4 queries → ~16 raw → ~10 after dedup, which comfortably
# fills the top-N candidate pool without over-paying Tavily.
_PER_QUERY_MAX = 4

# Absolute ceiling on how many items we ever hand back — LLM prompt
# size dominates cost from ~10 items up. The pipeline always fetches
# up to this many so a single cache slot per event covers every tier
# (high requests 8, medium 6, low 5 — all just slice from the same list).
_HARD_MAX_ITEMS = 12

# Parallelism for the Tavily fan-out. 4 threads match `num_queries=4`
# so every query starts immediately; the SDK is sync but network I/O
# releases the GIL, so wall-clock drops proportionally.
_SEARCH_WORKERS = 4

# Default TTL for the per-event cache in seconds. 10 minutes lets a
# single `run_agents.py` round reuse the same evidence across all
# agents, but drops stale results between rounds if the operator runs
# the runner infrequently. Callers can override in __init__.
_DEFAULT_TTL_SECONDS = 600

# How many event entries to keep in the cache before evicting LRU.
_DEFAULT_CACHE_SIZE = 64


def _cache_key(event_id: int) -> tuple:
    """One slot per event.

    The pipeline always fetches up to `_HARD_MAX_ITEMS`; callers that
    want fewer items just slice the returned list. Prior versions
    bucketed by `max_items` (2 slots per event: high tier at 8, medium
    /low at 6), which forced two full pipeline runs per event. Now every
    tier lands on the same slot — 1 Tavily fan-out + 1 stance call per
    event, regardless of how many agents run.
    """
    return (int(event_id),)


class RetrievalService:
    """Orchestrates the retrieval pipeline with a per-event TTL cache.

    Args:
        search_provider:  the SearchProvider (e.g. TavilyProvider). If
            disabled, `get_evidence` always returns []. Required.
        llm_client:  optional LLMClient. When present it drives query
            expansion and stance classification. When None (or its
            `.available` is False) those stages are skipped and the
            pipeline degrades to "run one Tavily search, dedup, score".
        ttl_seconds:  cache entry lifetime.
        max_cache_size:  LRU size.
        num_queries:  how many queries to ask the expander for.
    """

    def __init__(
        self,
        search_provider: SearchProvider,
        llm_client: Optional[LLMClient] = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_cache_size: int = _DEFAULT_CACHE_SIZE,
        num_queries: int = 4,
    ):
        self._search = search_provider
        self._llm = llm_client
        self._ttl = max(1, int(ttl_seconds))
        self._cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
        self._cache_max = max(1, int(max_cache_size))
        self._num_queries = max(1, min(int(num_queries), 6))
        self._lock = threading.Lock()
        # Counters for the debug endpoint / `stats()`.
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True iff the underlying search provider is enabled."""
        return bool(getattr(self._search, "enabled", False))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._cache),
                "search_enabled": self.enabled,
                "llm_available": bool(
                    self._llm is not None
                    and getattr(self._llm, "available", False)
                ),
            }

    # ------------------------------------------------------------------
    # Public API

    def get_evidence(self, event, max_items: int = 8) -> List[Dict[str, Any]]:
        """Return a scored, stance-labeled evidence list for `event`.

        Cache-first: repeat calls for the same event within the TTL get
        the pre-computed list (sliced to `max_items`) without any I/O.
        The cache stores up to `_HARD_MAX_ITEMS` per event, so every
        tier — high/medium/low with max_items 8/6/5 — hits the same
        slot and the pipeline only runs once per event per round.
        Never raises — worst case returns [].
        """
        event_id = getattr(event, "id", None)
        if event_id is None:
            return []
        max_items = max(1, min(int(max_items), _HARD_MAX_ITEMS))

        cached = self._cache_lookup(event_id)
        if cached is not None:
            return cached[:max_items]

        if not self.enabled:
            # No provider — nothing to fetch, but still cache the empty
            # result so we don't retry every agent.
            self._cache_store(event_id, [])
            return []

        try:
            # Always fetch the full pool. Slicing to `max_items` happens
            # at return time so tiers share one cache entry.
            items = self._run_pipeline(event, target=_HARD_MAX_ITEMS)
        except Exception as exc:  # noqa: BLE001 — pipeline must not crash the round
            print(
                f"[RetrievalService] pipeline failed for event "
                f"{event_id}: {exc}",
                file=sys.stderr,
            )
            items = []

        self._cache_store(event_id, items)
        return items[:max_items]

    # ------------------------------------------------------------------
    # Pipeline

    def _run_pipeline(self, event, target: int) -> List[Dict[str, Any]]:
        title = getattr(event, "title", "") or ""
        description = getattr(event, "description", "") or ""

        # 1. Query expansion (LLM; falls back to `[title+description]`).
        queries = expand_queries(self._llm, title, description, n=self._num_queries)
        # De-dupe query list first — no point burning a thread on it.
        seen_q, unique_queries = set(), []
        for q in queries:
            if q and q not in seen_q:
                seen_q.add(q)
                unique_queries.append(q)
        if not unique_queries:
            return []

        # 2. Fan-out Tavily searches in parallel. tavily-python is sync
        # but each call is network-bound, so a small thread pool gives
        # near-linear speedup with negligible CPU cost. Errors per query
        # come back as [] from the provider — one bad query can't taint
        # the rest.
        raw: List[Dict[str, Any]] = []
        workers = max(1, min(len(unique_queries), _SEARCH_WORKERS))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_query = {
                pool.submit(self._search.search, q, _PER_QUERY_MAX): q
                for q in unique_queries
            }
            for fut in as_completed(future_to_query):
                q = future_to_query[fut]
                try:
                    results = fut.result() or []
                except Exception as exc:  # noqa: BLE001 — provider shouldn't raise, but belt-and-suspenders
                    print(
                        f"[RetrievalService] search worker failed "
                        f"for {q!r}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                for item in results:
                    if isinstance(item, dict):
                        # Stamp the query for audit — persistence uses this too.
                        item.setdefault("query", q)
                        raw.append(item)

        if not raw:
            return []

        # 3. Dedup by normalized URL, then by title-key (fallback for
        # aggregator + primary with different URLs but same headline).
        deduped = self._dedupe(raw)

        # 4. Validate — drop anything mangled before the LLM sees it.
        deduped = [x for x in deduped if is_valid_evidence(x)]
        if not deduped:
            return []

        # 5. Composite score (relevance + time decay + source weight).
        # We take a slightly larger candidate pool INTO stance
        # classification than we return, so stance doesn't pay for the
        # long-tail results we would drop anyway.
        now = datetime.now(timezone.utc)
        for item in deduped:
            item["final_score"] = composite_score(
                relevance=item.get("relevance_score", 0.0),
                published=item.get("published_date"),
                url=item.get("url", ""),
                now=now,
            )
        deduped.sort(key=lambda x: x.get("final_score") or 0.0, reverse=True)
        candidates = deduped[: min(_HARD_MAX_ITEMS, max(target, 8))]

        # 6. Stance (LLM; falls back to NEUTRAL / 0.0 for every item).
        labeled = classify_stances(self._llm, title, candidates)

        # 7. Return top-N. Sort stayed stable across the stance step
        # because classify_stances preserves input order, and each item
        # keeps its final_score.
        return labeled

    @staticmethod
    def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """URL-normalized then title-based dedup. Keeps the first hit.

        Preserving order matters: the first query was the primary-entity
        one, so we keep the wire copy of a story over an aggregator's.
        """
        seen_urls = set()
        seen_titles = set()
        out = []
        for item in items:
            url = item.get("url_normalized") or item.get("url") or ""
            if url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
            tk = title_key(item.get("title") or "")
            if tk:
                if tk in seen_titles:
                    continue
                seen_titles.add(tk)
            out.append(item)
        return out

    # ------------------------------------------------------------------
    # Cache

    def _cache_lookup(self, event_id: int) -> Optional[List[Dict[str, Any]]]:
        key = _cache_key(event_id)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if self._is_expired(entry):
                self._cache.pop(key, None)
                self._misses += 1
                return None
            # LRU touch.
            self._cache.move_to_end(key)
            self._hits += 1
            return list(entry["items"])

    def _cache_store(self, event_id: int, items: List[Dict[str, Any]]) -> None:
        key = _cache_key(event_id)
        with self._lock:
            self._cache[key] = {
                "items": list(items),
                "stored_at": datetime.now(timezone.utc),
            }
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        stored = entry.get("stored_at")
        if not isinstance(stored, datetime):
            return True
        age = (datetime.now(timezone.utc) - stored).total_seconds()
        return age > self._ttl


__all__ = ["RetrievalService"]
