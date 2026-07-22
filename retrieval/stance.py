"""LLM-driven stance classification for retrieved evidence.

For each snippet, the classifier labels how it bears on the event's
YES outcome:

    SUPPORT   — evidence increases the probability of YES
    REFUTE    — evidence decreases the probability of YES
    NEUTRAL   — background, off-topic, or ambiguous

Stance is metadata for the forecast LLM (surfaced in
`llm/prompts.format_evidence`) — we do NOT rebalance the evidence list
to force 50/50 support/refute. When the event is genuinely one-sided,
we want that to show up.

The classifier runs as a single batched call: one JSON object out for
the whole list, one field per input index. That keeps the LLM cost
roughly constant regardless of how many items we retrieved.

Never raises: if the LLM is missing / errors / returns bad JSON, every
item is tagged NEUTRAL with confidence 0.0 so downstream code has a
consistent shape to read.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from llm.client import LLMClient


_VALID_LABELS = ("SUPPORT", "REFUTE", "NEUTRAL")


_SYSTEM_MSG = (
    "You classify how each evidence snippet bears on a YES/NO prediction "
    "market question. For each numbered item, output one of:\n"
    "  SUPPORT  — snippet raises the probability of YES\n"
    "  REFUTE   — snippet lowers the probability of YES\n"
    "  NEUTRAL  — background, off-topic, or ambiguous\n"
    "Also give a confidence in [0.0, 1.0]. Base the label ONLY on the "
    "snippet content — do not invent facts. Respond with strict JSON only."
)


def _fallback_labels(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tag every item NEUTRAL / 0.0 — the safe default when we can't classify."""
    out = []
    for item in items:
        merged = dict(item)
        merged["stance"] = "NEUTRAL"
        merged["stance_confidence"] = 0.0
        out.append(merged)
    return out


def _render_items_block(items: List[Dict[str, Any]], max_chars: int = 400) -> str:
    lines = []
    for i, item in enumerate(items):
        title = str(item.get("title", "")).strip()[:200]
        summary = str(item.get("content_summary", "")).strip()[:max_chars]
        lines.append(f"[{i}] {title}\n    {summary}")
    return "\n".join(lines)


def _build_user_msg(event_title: str, items: List[Dict[str, Any]]) -> str:
    return (
        f"Prediction-market question: {event_title}\n\n"
        f"Evidence items:\n{_render_items_block(items)}\n\n"
        "Respond with JSON of exactly this shape (one entry per input "
        "item, indexed by `i`):\n"
        "{\n"
        '  "stances": [\n'
        '    {"i": 0, "label": "SUPPORT|REFUTE|NEUTRAL", "confidence": 0.0-1.0},\n'
        '    {"i": 1, "label": "...", "confidence": ...}\n'
        "  ]\n"
        "}"
    )


def classify_stances(
    llm: Optional[LLMClient],
    event_title: str,
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return `items` with `stance` / `stance_confidence` populated on each.

    Input items are NOT mutated; the function returns a new list of
    shallow copies. The order of the input is preserved.
    """
    if not items:
        return []
    if llm is None or not getattr(llm, "available", False):
        return _fallback_labels(items)

    messages = [
        {"role": "system", "content": _SYSTEM_MSG},
        {"role": "user", "content": _build_user_msg(event_title, items)},
    ]

    try:
        payload = llm.chat_json(messages)
    except Exception as exc:  # noqa: BLE001
        print(f"[stance] LLM call failed: {exc}", file=sys.stderr)
        return _fallback_labels(items)

    stances = payload.get("stances") if isinstance(payload, dict) else None
    if not isinstance(stances, list):
        return _fallback_labels(items)

    # Build an index → (label, conf) map so out-of-order responses still
    # land on the right item. Missing indices default to NEUTRAL / 0.0.
    by_index: Dict[int, Dict[str, Any]] = {}
    for entry in stances:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("i"))
        except (TypeError, ValueError):
            continue
        label = str(entry.get("label", "")).strip().upper()
        if label not in _VALID_LABELS:
            label = "NEUTRAL"
        try:
            conf = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        by_index[idx] = {"stance": label, "stance_confidence": conf}

    out = []
    for i, item in enumerate(items):
        merged = dict(item)
        labeled = by_index.get(i, {"stance": "NEUTRAL", "stance_confidence": 0.0})
        merged["stance"] = labeled["stance"]
        merged["stance_confidence"] = labeled["stance_confidence"]
        out.append(merged)
    return out


def stance_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """Small helper for callers that want the S/R/N tally.

    Only items whose stance is one of the three canonical labels count;
    items without a stance or with None are ignored.
    """
    counts = {"SUPPORT": 0, "REFUTE": 0, "NEUTRAL": 0}
    for item in items or []:
        s = item.get("stance") if isinstance(item, dict) else None
        if s in counts:
            counts[s] += 1
    return counts


__all__ = ["classify_stances", "stance_counts"]
