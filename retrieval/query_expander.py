"""LLM-driven query expansion for the retrieval pipeline.

Turns a prediction-market event (title + description) into a small,
complementary set of search queries. The two things this buys us over
the previous one-shot `title + description` query:

  * **Vocabulary coverage** — event descriptions use compliance /
    policy phrasing; news uses different words. One LLM call maps the
    former to the latter.
  * **Confirmation-bias hedge** — at least one query is generated with
    reverse phrasing (`denied`, `postponed`, `withdrawn`, `cancelled`)
    so we surface refuting evidence alongside supporting.

Failure is always fatal-silent: if the LLM isn't available, the call
fails, or the JSON is malformed, we return the single default query
so the pipeline keeps working.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from llm.client import LLMClient


_SYSTEM_MSG = (
    "You generate concise web search queries for a real-time prediction "
    "market. Return a small set of complementary queries covering "
    "different angles of the event. Respond with strict JSON only — no "
    "prose."
)


def _default_query(title: str, description: str) -> str:
    """The fallback query when expansion fails (matches pre-upgrade behavior)."""
    return (f"{title or ''} {description or ''}").strip()[:400]


def _build_user_msg(title: str, description: str, n: int) -> str:
    return (
        f"Event title: {title}\n"
        f"Event description: {description}\n\n"
        f"Generate {n} short web-search queries (each 3-10 words). "
        "Coverage requirements:\n"
        "  1. One query on the primary entity + action described.\n"
        "  2. One query using REVERSE phrasing (e.g. 'denied', "
        "'postponed', 'withdrawn', 'cancelled', 'fails') to surface "
        "refuting evidence.\n"
        "  3. One query anchored on the relevant date / deadline / "
        "quarter, when applicable.\n"
        "  4. Additional queries can cover related parties, regulators, "
        "or competitors.\n\n"
        "Do NOT include quotation marks around the whole query, and do "
        "NOT copy the description verbatim. Respond with JSON of exactly "
        "this shape:\n"
        "{\"queries\": [\"query one\", \"query two\", ...]}"
    )


def expand_queries(
    llm: Optional[LLMClient],
    title: str,
    description: str,
    n: int = 4,
) -> List[str]:
    """Return a de-duped list of 1..n query strings.

    Always returns at least one query (the fallback title+description
    string). Never raises — the caller can iterate the result without
    guarding.
    """
    n = max(1, min(int(n), 8))
    fallback = [_default_query(title, description)]
    fallback = [q for q in fallback if q]

    if llm is None or not getattr(llm, "available", False):
        return fallback or [""]

    messages = [
        {"role": "system", "content": _SYSTEM_MSG},
        {"role": "user", "content": _build_user_msg(title, description, n)},
    ]

    try:
        payload = llm.chat_json(messages)
    except Exception as exc:  # noqa: BLE001 — never take down the retrieval round
        print(f"[query_expander] LLM call failed: {exc}", file=sys.stderr)
        return fallback or [""]

    queries = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(queries, list):
        return fallback or [""]

    # Clean & dedupe, preserving order so the caller's first Tavily call
    # is still the primary entity query.
    seen = set()
    out: List[str] = []
    for q in queries:
        if not isinstance(q, str):
            continue
        cleaned = q.strip().strip('"').strip("'")[:200]
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= n:
            break

    if not out:
        return fallback or [""]

    # Always include the raw title+description query as an anchor —
    # cheap insurance against the LLM going too abstract.
    anchor = _default_query(title, description)
    if anchor and anchor.lower() not in seen:
        out.append(anchor)

    return out


__all__ = ["expand_queries"]
