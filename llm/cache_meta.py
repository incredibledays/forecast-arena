"""Provider-neutral prompt cache metadata.

Prompt caching (OpenAI automatic prefix caching, Anthropic explicit
cache_control, vLLM prefix cache, ...) all reward a *stable prefix*
followed by a *dynamic suffix*. This module computes that split in a
provider-neutral way: it produces two deterministic hashes and lets the
caller assemble the actual prompt however their provider prefers.

Design rules (enforced by tests):

  * The STABLE PREFIX hash is a pure function of durable inputs only —
    system-prompt version, JSON-schema version, market-definition hash,
    resolution-rule hash, archetype hash, evidence-bundle version. It
    NEVER folds in volatile data (UUIDs, request timestamps, wall-clock,
    or random ordering). Two logically-identical requests always hash
    the same, which is what makes the cache hit.

  * The DYNAMIC SUFFIX hash covers the parts that legitimately change
    per request: prior probability, the evidence *delta* hash, the
    current time bucket, and (optionally) current market state.

Changing any stable input (e.g. bumping the schema version or editing
the resolution rule) rotates the prefix hash — which is correct: a new
prefix means a new cache entry, so stale cached completions are never
reused across a contract change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Bump these when the corresponding text/contract changes so cached
# entries computed against the old version are naturally invalidated.
DEFAULT_SYSTEM_PROMPT_VERSION = "sys-v1"
DEFAULT_SCHEMA_VERSION = "schema-v1"


def _stable_hash(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serializable object.

    `sort_keys=True` removes dict-ordering nondeterminism; `default=str`
    keeps it from throwing on stray non-JSON values. Truncated to 16
    hex chars — 64 bits is ample for cache-key collision avoidance and
    keeps logs readable.
    """
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def hash_text(text: Optional[str]) -> str:
    """Hash an arbitrary text blob (market definition, resolution rules...)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def evidence_delta_hash(prev_bundle_version, new_bundle_version, added_ids=None) -> str:
    """Hash the *change* between two evidence-bundle states.

    Deliberately order-insensitive on `added_ids` (sorted before
    hashing) so the same set of new evidence produces the same delta
    hash regardless of retrieval ordering — retrieval order is volatile
    and must not leak into cache identity.
    """
    payload = {
        "prev": str(prev_bundle_version or ""),
        "new": str(new_bundle_version or ""),
        "added": sorted(str(x) for x in (added_ids or [])),
    }
    return _stable_hash(payload)


@dataclass
class CacheMetadata:
    """The stable/dynamic split plus the derived cache identity.

    Callers build one per request from durable + per-request inputs.
    `stable_prefix_hash` is the cache key prefix; `cache_key` combines
    both halves for a full-request identity when needed.
    """

    # --- stable inputs ---
    system_prompt_version: str = DEFAULT_SYSTEM_PROMPT_VERSION
    schema_version: str = DEFAULT_SCHEMA_VERSION
    market_definition_hash: Optional[str] = None
    resolution_rule_hash: Optional[str] = None
    archetype_hash: Optional[str] = None
    evidence_bundle_version: Optional[str] = None

    # --- dynamic inputs ---
    prior_probability: Optional[float] = None
    evidence_delta_hash: Optional[str] = None
    time_bucket: Optional[str] = None
    market_state_hash: Optional[str] = None

    # Free-form, non-secret extras that should participate in the
    # dynamic half only (never the stable prefix).
    dynamic_extra: Dict[str, Any] = field(default_factory=dict)

    def stable_prefix_payload(self) -> Dict[str, Any]:
        """The durable-only dict the stable prefix hashes over.

        Intentionally excludes prior probability, evidence delta, time,
        and market state — anything that changes per request.
        """
        return {
            "system_prompt_version": self.system_prompt_version,
            "schema_version": self.schema_version,
            "market_definition_hash": self.market_definition_hash,
            "resolution_rule_hash": self.resolution_rule_hash,
            "archetype_hash": self.archetype_hash,
            "evidence_bundle_version": self.evidence_bundle_version,
        }

    def dynamic_suffix_payload(self) -> Dict[str, Any]:
        """The per-request dict the dynamic suffix hashes over."""
        payload = {
            "prior_probability": self.prior_probability,
            "evidence_delta_hash": self.evidence_delta_hash,
            "time_bucket": self.time_bucket,
            "market_state_hash": self.market_state_hash,
        }
        if self.dynamic_extra:
            payload["extra"] = self.dynamic_extra
        return payload

    @property
    def stable_prefix_hash(self) -> str:
        return _stable_hash(self.stable_prefix_payload())

    @property
    def dynamic_suffix_hash(self) -> str:
        return _stable_hash(self.dynamic_suffix_payload())

    @property
    def cache_key(self) -> str:
        """Full-request cache identity: `<prefix>:<suffix>`."""
        return f"{self.stable_prefix_hash}:{self.dynamic_suffix_hash}"

    def public_dict(self) -> Dict[str, Any]:
        """Log-safe view — only hashes/versions, no prompt content."""
        return {
            "stable_prefix_hash": self.stable_prefix_hash,
            "dynamic_suffix_hash": self.dynamic_suffix_hash,
            "cache_key": self.cache_key,
            "system_prompt_version": self.system_prompt_version,
            "schema_version": self.schema_version,
            "evidence_bundle_version": self.evidence_bundle_version,
        }


def build_cache_metadata(
    *,
    system_prompt_version: str = DEFAULT_SYSTEM_PROMPT_VERSION,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    market_definition: Optional[str] = None,
    resolution_rules: Optional[str] = None,
    archetype_definition: Optional[str] = None,
    evidence_bundle_version: Optional[str] = None,
    prior_probability: Optional[float] = None,
    evidence_delta: Optional[str] = None,
    time_bucket: Optional[str] = None,
    market_state: Optional[str] = None,
    dynamic_extra: Optional[Dict[str, Any]] = None,
) -> CacheMetadata:
    """Convenience builder: hashes the raw text fields for you.

    Pass raw strings for market definition / resolution rules /
    archetype definition; this hashes them into the stable prefix. Pass
    a pre-computed `evidence_delta` hash (see `evidence_delta_hash`) or
    a raw string — either way it lands only in the dynamic suffix.
    """
    return CacheMetadata(
        system_prompt_version=system_prompt_version,
        schema_version=schema_version,
        market_definition_hash=hash_text(market_definition) if market_definition is not None else None,
        resolution_rule_hash=hash_text(resolution_rules) if resolution_rules is not None else None,
        archetype_hash=hash_text(archetype_definition) if archetype_definition is not None else None,
        evidence_bundle_version=(str(evidence_bundle_version) if evidence_bundle_version is not None else None),
        prior_probability=prior_probability,
        evidence_delta_hash=(evidence_delta if evidence_delta is None or len(str(evidence_delta)) <= 32 else hash_text(evidence_delta)),
        time_bucket=time_bucket,
        market_state_hash=hash_text(market_state) if market_state is not None else None,
        dynamic_extra=dict(dynamic_extra or {}),
    )


__all__ = [
    "CacheMetadata",
    "build_cache_metadata",
    "hash_text",
    "evidence_delta_hash",
    "DEFAULT_SYSTEM_PROMPT_VERSION",
    "DEFAULT_SCHEMA_VERSION",
]
