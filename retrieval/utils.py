"""Pure, dependency-free helpers for the retrieval pipeline.

These functions are network-free and LLM-free so the service layer can
mix them into the pipeline without any I/O concerns. Keep them cheap:
every retrieved snippet passes through several of these.

Nothing here raises for bad input — the retrieval layer must survive
mangled Tavily responses without taking down a trading round.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


# Query params we always want to strip from a URL. Anything starting
# with `utm_` is stripped too via the prefix check below.
_TRACKING_PARAMS = frozenset({
    "fbclid", "gclid", "yclid", "msclkid", "mc_cid", "mc_eid",
    "igshid", "twclid", "ref", "ref_src", "ref_url",
    "_hsenc", "_hsmi", "hsCtaTracking",
    "spm", "share", "source",
})


def normalize_url(url: str) -> str:
    """Return a canonicalized URL used for dedup only.

    - lowercase scheme + host
    - drop the fragment
    - drop utm_*, fbclid, gclid, and friends
    - keep the remaining query params in their original order
    - collapse a lone trailing slash on a non-root path

    We keep the ORIGINAL url intact elsewhere (display, persistence);
    this canonical form is a dedup key, not a replacement.
    """
    if not url or not isinstance(url, str):
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()

    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()

    kept = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(kept, doseq=True)

    path = parts.path or ""
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    return urlunsplit((scheme, netloc, path, query, ""))


def domain_of(url: str) -> str:
    """Return the bare hostname (no port, no `www.`) or ''."""
    if not url or not isinstance(url, str):
        return ""
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return ""
    if not host:
        return ""
    if "@" in host:                                 # strip userinfo, just in case
        host = host.rsplit("@", 1)[1]
    if ":" in host:                                 # strip port
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host[:128]                               # matches the DB column width


# Boilerplate patterns commonly tacked onto news snippets. Each is
# case-insensitive; the trailing `.*` swallows the rest of the line so we
# don't leave a dangling "…newsletter" fragment. Kept short and
# additive — future domains just append here.
_BOILERPLATE_RE = re.compile(
    r"(?im)"
    r"("
    r"\s*subscribe to our newsletter.*$"
    r"|\s*sign up (?:for|to) (?:our )?newsletter.*$"
    r"|\s*this article (?:is|was) (?:for|available to) subscribers.*$"
    r"|\s*(?:read|continue reading|read more|learn more).{0,4}(?:→|»|\.\.\.).*$"
    r"|\s*share (?:this|on) (?:facebook|twitter|linkedin|whatsapp).*$"
    r"|\s*follow us on (?:facebook|twitter|linkedin|instagram|x).*$"
    r"|\s*by (?:clicking|using) (?:this|our) (?:site|website).*cookies.*$"
    r"|\s*we (?:use|and our partners use) cookies.*$"
    r"|\s*please (?:enable|disable) (?:javascript|ad ?block).*$"
    r"|\s*©\s*\d{4}[^\n]*all rights reserved.*$"
    r"|\s*advertisement\s*$"
    r"|\s*sponsored content\s*$"
    r")"
)

# Collapse runs of whitespace introduced by the strip.
_WS_RE = re.compile(r"\s+")


def clean_snippet(text: str) -> str:
    """Strip common boilerplate from a snippet and normalize whitespace.

    Non-destructive: if nothing matches, we only whitespace-normalize.
    """
    if not text or not isinstance(text, str):
        return ""
    stripped = _BOILERPLATE_RE.sub(" ", text)
    stripped = _WS_RE.sub(" ", stripped).strip()
    return stripped


_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def title_key(title: str) -> str:
    """Return a stable dedup key derived from a title.

    Lowercase, strip everything non-alphanumeric, cap at 60 chars. Two
    articles whose titles differ only in punctuation / prefix (`"Reuters:
    …"` vs `"Reuters | …"`) collapse to the same key.

    Empty input returns ''; callers should treat that as "no dedup".
    """
    if not title or not isinstance(title, str):
        return ""
    return _ALNUM_RE.sub("", title.lower())[:60]


def parse_published_date(raw) -> Optional[datetime]:
    """Best-effort date parse. Returns tz-aware UTC datetime or None.

    Accepts: ISO-8601 (Tavily's default), RFC-2822, or a plain
    `YYYY-MM-DD`. Anything else → None (callers fall back to the
    neutral 0.5 in the time-decay scorer).
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None

    # ISO-8601 — accept both `Z` and `+00:00`.
    iso = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    # Bare date.
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    # RFC-2822 (some feeds still serve this).
    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass

    return None


__all__ = [
    "normalize_url",
    "domain_of",
    "clean_snippet",
    "title_key",
    "parse_published_date",
]
