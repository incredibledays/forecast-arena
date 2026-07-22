"""Prompt builders + JSON schema constants for forecast-arena.

Kept separate from the client so callers can reuse or override just
the prompt without touching transport code. All prompts here forbid
chain-of-thought and require strict-JSON responses.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


# The JSON contract the forecast prompts return. Duplicated as a plain
# dict so callers (and tests) can validate against it without importing
# jsonschema.
FORECAST_JSON_SHAPE: Dict[str, Any] = {
    "probability_yes": "<float 0.0-1.0>",
    "confidence": "<float 0.0-1.0>",
    "reasoning_summary": "<one short sentence, <=200 chars>",
    "key_evidence": ["<citation 1>", "<citation 2>"],
    "risk_factors": ["<risk 1>", "<risk 2>"],
}


FORECAST_SYSTEM_MSG = (
    "You are a probabilistic forecaster for a binary YES/NO prediction "
    "market. Use only the retrieved evidence and general knowledge to "
    "estimate the probability that the YES outcome occurs. Each evidence "
    "item may include metadata: stance (SUPPORT/REFUTE/NEUTRAL vs the "
    "YES outcome), age, and source domain — weigh these when items "
    "conflict. Respond with a single JSON object matching the requested "
    "schema — no prose outside the JSON, no chain-of-thought, no "
    "analysis notes."
)


def _age_hint(published: Optional[datetime], now: Optional[datetime]) -> str:
    """Return a compact age string like `2d` / `6h` / `just now`, or ''."""
    if published is None:
        return ""
    now = now or datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    delta = (now - published).total_seconds()
    if delta < 0:
        return "future"
    if delta < 3600:
        return "just now"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def _meta_bracket(item: Dict[str, Any], now: Optional[datetime]) -> str:
    """Return a `(stance, age, domain)` bracket, only for present fields.

    Elides the whole parenthetical when nothing useful is set — that's
    the pre-enrichment path (bare Tavily results) which just renders
    title / url / summary.
    """
    stance = (item.get("stance") or "").strip().upper() or None
    age = _age_hint(item.get("published_date"), now)
    domain = (item.get("source_domain") or "").strip() or None
    bits = [b for b in (stance, age, domain) if b]
    return f" ({', '.join(bits)})" if bits else ""


def format_evidence(
    evidence: Iterable[Dict[str, Any]],
    max_items: int = 6,
    max_chars: int = 500,
    now: Optional[datetime] = None,
) -> str:
    """Render an evidence list into a stable, size-capped block.

    Now includes stance / age / domain metadata when the retrieval
    pipeline populated them; falls back to the previous shape when the
    caller passed bare evidence (e.g. a stub in tests).
    """
    now = now or datetime.now(timezone.utc)
    lines: List[str] = []
    for i, item in enumerate(list(evidence)[:max_items], start=1):
        summary = str(item.get("content_summary", ""))[:max_chars]
        meta = _meta_bracket(item, now)
        lines.append(
            f"[{i}]{meta} {item.get('title','')}\n"
            f"    url: {item.get('url','')}\n"
            f"    summary: {summary}"
        )
    return "\n".join(lines) if lines else "(no evidence retrieved)"


def build_forecast_messages(
    title: str,
    description: str,
    evidence: Iterable[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Return chat-completions ``messages`` for a strict-JSON forecast."""
    evidence_block = format_evidence(evidence)
    user_msg = (
        f"Event title: {title}\n"
        f"Event description: {description}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Respond with JSON of exactly this shape:\n"
        "{\n"
        '  "probability_yes": <float 0.0-1.0>,\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "reasoning_summary": "<one short sentence, <=200 chars>",\n'
        '  "key_evidence": ["<citation 1>", "<citation 2>", ...],\n'
        '  "risk_factors": ["<risk 1>", "<risk 2>", ...]\n'
        "}"
    )
    return [
        {"role": "system", "content": FORECAST_SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]
