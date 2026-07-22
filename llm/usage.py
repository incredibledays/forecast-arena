"""LLM usage records — secret-safe telemetry for routed requests.

`LLMUsage` is a plain dataclass (no DB coupling in this phase — the
audit found there is no Alembic and `create_all` can't add columns to
existing tables, so we keep this a persistence-agnostic record that a
later phase can map to a table). `UsageRecorder` is an in-memory,
thread-safe sink with aggregate helpers for the budget layer and tests.

Hard rule (enforced by `_scrub` + tests): a usage record NEVER holds
API keys, authorization headers, raw chain-of-thought, or full
sensitive prompts. Only hashes, versions, counts, and coarse outcome
flags are recorded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# Substrings that must never appear as keys in recorded metadata. If a
# caller stuffs one of these into `task_metadata`, `_scrub` drops it.
_FORBIDDEN_META_KEYS = (
    "api_key", "apikey", "authorization", "auth", "bearer", "token",
    "secret", "password", "cookie", "prompt", "messages", "content",
    "chain_of_thought", "cot", "reasoning",
)


def _scrub(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of `meta` with any secret-ish keys removed."""
    if not meta:
        return {}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        lk = str(k).lower()
        if any(bad in lk for bad in _FORBIDDEN_META_KEYS):
            continue
        out[k] = v
    return out


@dataclass
class LLMUsage:
    """One routed LLM request's telemetry. All fields are non-secret."""

    task_type: str
    selected_tier: str
    provider: str
    model: str
    batch_mode: str
    timestamp: float

    market_id: Optional[int] = None
    evidence_bundle_id: Optional[str] = None

    estimated_input_tokens: int = 0
    actual_input_tokens: Optional[int] = None
    output_tokens: int = 0

    cache_eligible: bool = False
    cache_hit: Optional[bool] = None

    retry_count: int = 0
    escalation_count: int = 0
    fallback_used: bool = False
    degraded: bool = False

    latency_ms: Optional[float] = None
    success: bool = True

    # Non-secret extras only; scrubbed on construction.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Belt-and-suspenders: scrub even if the caller bypassed the
        # recorder helper and built the record directly.
        self.metadata = _scrub(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UsageRecorder:
    """Thread-safe in-memory sink for `LLMUsage` records.

    Aggregate helpers back the budget layer and tests. A production
    deployment can subclass / wrap this to also persist rows.
    """

    def __init__(self, max_records: int = 100_000):
        self._records: List[LLMUsage] = []
        self._max = max(1, int(max_records))
        self._lock = threading.Lock()

    def record(self, usage: LLMUsage) -> LLMUsage:
        usage.metadata = _scrub(usage.metadata)
        with self._lock:
            self._records.append(usage)
            # Bound memory: drop oldest if we exceed the cap.
            if len(self._records) > self._max:
                overflow = len(self._records) - self._max
                del self._records[:overflow]
        return usage

    def all(self) -> List[LLMUsage]:
        with self._lock:
            return list(self._records)

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def total_output_tokens(self) -> int:
        with self._lock:
            return sum(r.output_tokens or 0 for r in self._records)

    def total_input_tokens(self) -> int:
        with self._lock:
            return sum(
                (r.actual_input_tokens if r.actual_input_tokens is not None
                 else r.estimated_input_tokens) or 0
                for r in self._records
            )

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            recs = list(self._records)
        by_tier: Dict[str, int] = {}
        by_batch: Dict[str, int] = {}
        degraded = 0
        for r in recs:
            by_tier[r.selected_tier] = by_tier.get(r.selected_tier, 0) + 1
            by_batch[r.batch_mode] = by_batch.get(r.batch_mode, 0) + 1
            if r.degraded:
                degraded += 1
        return {
            "records": len(recs),
            "by_tier": by_tier,
            "by_batch_mode": by_batch,
            "degraded": degraded,
            "total_output_tokens": sum(r.output_tokens or 0 for r in recs),
        }


__all__ = ["LLMUsage", "UsageRecorder"]
