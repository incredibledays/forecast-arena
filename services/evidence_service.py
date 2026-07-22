"""EvidenceService — the shared, optimized real-time evidence layer.

This phase implements retrieval, normalization, dedup, versioned
EvidenceBundles, compact EvidenceDeltas, source selection, and
cache-friendly prompt-context assembly. It does NOT update individual
Agent beliefs or trade — that is a later phase.

Design (all "once per market refresh", never per Agent):

    refresh(event_id)
        1. Generate 5 queries (primary / official / recent / contradiction
           / resolution) and search ONCE per query via the existing
           SearchProvider (a shared cache keeps repeats free).
        2. Normalize URLs (strip tracking params), dedup by exact URL /
           canonical URL / content hash / near-identical title.
        3. Store each distinct source's cleaned text ONCE in SourceContent
           (keyed by content_hash). Many events reference the same row.
        4. Appraise each source for THIS event → InformationEvent rows with
           relevance/credibility/freshness/novelty/stance/impact +
           prompt-injection risk + the three fairness clocks.
        5. If the new set is meaningfully novel vs the latest bundle, cut a
           new EvidenceBundle (version+1) and an EvidenceDelta. No novelty
           ⇒ no new version.

    select_sources(...)     — token-budgeted top-K with support/oppose/
                              official preservation.
    build_prompt_context(...) — stable prefix (hashed, deterministic) +
                              dynamic suffix (delta only) + the mandatory
                              untrusted-content preamble.

No LLM is required: stance/impact are computed heuristically here (the
router is consulted only to LABEL which tier a real belief update WOULD
use, honoring budgets — it does not itself call a model in this phase).
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from llm import (
    UNTRUSTED_PREAMBLE,
    build_cache_metadata,
    evidence_delta_hash,
    hash_text,
    scan_for_injection,
)
from llm.evidence_security import injection_risk_score
from models import (
    EvidenceBundle,
    EvidenceDelta,
    Event,
    InformationEvent,
    SourceContent,
    STANCE_NEUTRAL,
    STANCE_REFUTE,
    STANCE_SUPPORT,
    SOURCE_TYPE_BLOG,
    SOURCE_TYPE_NEWS,
    SOURCE_TYPE_OFFICIAL,
    SOURCE_TYPE_SOCIAL,
    SOURCE_TYPE_WIRE,
    db,
)
from services._schema_cache import ensure_created as _ensure_schema_cached
from retrieval.scoring import source_weight, time_decay
from retrieval.utils import (
    clean_snippet,
    domain_of,
    normalize_url,
    parse_published_date,
    title_key,
)
from services.scheduler_service import SchedulerService


# Approx chars-per-token for the token-size estimate (English prose ≈ 4).
_CHARS_PER_TOKEN = 4.0
_DEFAULT_TOKEN_BUDGET = 2000
_DEFAULT_TOP_K = 8
_MAX_CLEANED_CHARS = 1200

# Novelty gate: a refresh cuts a new bundle only if the added/removed
# fraction or the aggregate-impact shift clears these thresholds.
_NOVELTY_MIN_CHANGED = 1          # at least one added/removed source, AND
_NOVELTY_MIN_IMPACT_SHIFT = 0.02  # OR a >2% aggregate-impact move

# Official/primary domains that make a source "official" for preservation.
_OFFICIAL_DOMAINS = {
    "sec.gov", "federalreserve.gov", "whitehouse.gov", "openai.com",
    "anthropic.com", "deepmind.google", "ai.meta.com", "europa.eu",
    "bls.gov", "treasury.gov",
}
_WIRE_DOMAINS = {"reuters.com", "apnews.com", "bloomberg.com"}
_MAX_BUNDLE_SUMMARY_CHARS = 12000
_MAX_DELTA_FACTS = 50


def _limit_text(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 32].rstrip() + " ... [truncated]"


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    return int(len(text or "") / _CHARS_PER_TOKEN) + 1


def _classify_source_type(domain: str) -> str:
    if domain in _OFFICIAL_DOMAINS or domain.endswith(".gov"):
        return SOURCE_TYPE_OFFICIAL
    if domain in _WIRE_DOMAINS:
        return SOURCE_TYPE_WIRE
    if any(s in domain for s in ("twitter.", "x.com", "reddit.", "facebook.", "t.me")):
        return SOURCE_TYPE_SOCIAL
    if "blog" in domain or domain.endswith(".medium.com"):
        return SOURCE_TYPE_BLOG
    return SOURCE_TYPE_NEWS


# Simple lexical stance heuristic (LLM-free for this phase). Support/refute
# keyword hits vs the event's YES framing; NEUTRAL otherwise.
_SUPPORT_WORDS = ("confirmed", "announced", "will", "agreed", "approved",
                  "launched", "yes", "on track", "expected to", "reached")
_REFUTE_WORDS = ("denied", "delayed", "cancelled", "canceled", "postponed",
                 "will not", "won't", "rejected", "no plans", "unlikely",
                 "fails", "failed", "halted")


class EvidenceService:
    """Shared per-event evidence pipeline. One instance per refresh round."""

    def __init__(self, search_provider=None, router=None):
        self._search = search_provider
        self._router = router

    @staticmethod
    def ensure_schema() -> None:
        _ensure_schema_cached()

    # ==================================================================
    # Query generation
    # ==================================================================

    @staticmethod
    def build_queries(event) -> Dict[str, str]:
        """The 5 query variants for one market refresh. Deterministic."""
        title = (getattr(event, "title", "") or "").strip()
        desc = (getattr(event, "description", "") or "").strip()
        base = f"{title} {desc}".strip()[:300]
        src = (getattr(event, "resolution_source", "") or "").strip()
        return {
            "primary": base,
            "official": (f"{title} official announcement statement {src}").strip()[:300],
            "recent": (f"{title} latest news update today").strip()[:300],
            "contradiction": (f"{title} delay denied cancelled dispute").strip()[:300],
            "resolution": (f"{title} result outcome resolved {src}").strip()[:300],
        }

    # ==================================================================
    # Refresh — the once-per-market-update entry point
    # ==================================================================

    def refresh(
        self, event_id: int, max_per_query: int = 4, now: Optional[float] = None,
        availability_delay_s: float = 0.0,
    ) -> Dict[str, Any]:
        """Retrieve → normalize → dedup → store → appraise → version.

        Returns metrics incl. how many searches ran, distinct sources
        stored, whether a new bundle version was cut, and the delta.
        """
        self.ensure_schema()
        event = db.session.get(Event, event_id)
        if event is None:
            raise ValueError(f"event {event_id} not found")
        if now is None:
            now = SchedulerService.now()

        queries = self.build_queries(event)

        # --- 1. Search ONCE per query variant. ---
        raw_items: List[Dict[str, Any]] = []
        search_calls = 0
        if self._search is not None and getattr(self._search, "enabled", False):
            for qkind, q in queries.items():
                if not q:
                    continue
                try:
                    results = self._search.search(q, max_results=max_per_query) or []
                except Exception as exc:  # noqa: BLE001 — retrieval must not crash
                    print(f"[evidence] search failed ({qkind}): {exc}", file=sys.stderr)
                    results = []
                search_calls += 1
                for item in results:
                    if isinstance(item, dict):
                        item.setdefault("query_kind", qkind)
                        raw_items.append(item)

        # --- 2. Normalize + dedup (URL / canonical / content hash / title). ---
        deduped = self._dedupe(raw_items)

        # --- 3 & 4. Store sources once + appraise for this event. ---
        info_events, new_source_count = self._store_and_appraise(
            event, deduped, now=now, availability_delay_s=availability_delay_s
        )

        # --- 5. Version gate. ---
        version_result = self._maybe_new_bundle(event_id, info_events, now=now)

        db.session.commit()
        result = {
            "event_id": event_id,
            "search_calls": search_calls,
            "queries": list(queries.keys()),
            "raw_items": len(raw_items),
            "deduped_items": len(deduped),
            "new_sources_stored": new_source_count,
            "information_events": len(info_events),
            **version_result,
        }
        db.session.expunge_all()
        return result

    # ------------------------------------------------------------------
    # Dedup

    @staticmethod
    def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dedup by exact URL, canonical URL, content hash, and title key."""
        seen_url, seen_canon, seen_hash, seen_title = set(), set(), set(), set()
        out = []
        for item in items:
            url = str(item.get("url") or "").strip()
            canon = normalize_url(url)
            summary = clean_snippet(str(item.get("content_summary") or item.get("content") or ""))
            chash = _sha256(summary) if summary else ""
            tkey = title_key(item.get("title") or "")

            if url and url in seen_url:
                continue
            if canon and canon in seen_canon:
                continue
            if chash and chash in seen_hash:
                continue
            if tkey and tkey in seen_title:
                continue

            if url:
                seen_url.add(url)
            if canon:
                seen_canon.add(canon)
            if chash:
                seen_hash.add(chash)
            if tkey:
                seen_title.add(tkey)

            item["_canonical_url"] = canon
            item["_content_hash"] = chash
            item["_cleaned"] = summary
            out.append(item)
        return out

    # ------------------------------------------------------------------
    # Store + appraise

    def _store_and_appraise(
        self, event, deduped: List[Dict[str, Any]], now: float,
        availability_delay_s: float,
    ) -> Tuple[List[InformationEvent], int]:
        info_events: List[InformationEvent] = []
        new_sources = 0
        title = getattr(event, "title", "") or ""
        now_dt = datetime.now(timezone.utc)

        for item in deduped:
            cleaned = (item.get("_cleaned") or "")[:_MAX_CLEANED_CHARS]
            if not cleaned:
                continue
            chash = item.get("_content_hash") or _sha256(cleaned)
            url = str(item.get("url") or "")
            domain = domain_of(url)
            published = parse_published_date(item.get("published_date"))

            # --- SourceContent: store ONCE by content hash. ---
            source = SourceContent.query.filter_by(content_hash=chash).one_or_none()
            if source is None:
                source = SourceContent(
                    content_hash=chash,
                    canonical_url=item.get("_canonical_url") or url,
                    title=(item.get("title") or "")[:512] or None,
                    source_domain=domain or None,
                    source_type=_classify_source_type(domain),
                    published_at=published,
                    retrieved_at=now_dt,
                    cleaned_text=cleaned,
                    language=item.get("language") or "en",
                    content_metadata={"query_kind": item.get("query_kind")},
                )
                db.session.add(source)
                db.session.flush()
                new_sources += 1

            # --- Appraisal scores (heuristic, LLM-free). ---
            flags = scan_for_injection(cleaned)
            inj_risk = injection_risk_score(cleaned)
            relevance = self._relevance(title, cleaned, item.get("relevance_score"))
            credibility = source_weight(url) or self._credibility_floor(source.source_type)
            freshness = time_decay(published, now=now_dt)
            stance = self._stance(cleaned)
            impact = self._impact(relevance, credibility, freshness, stance)
            # Suspicious content is impact-penalized so it can't dominate.
            if inj_risk > 0:
                impact *= (1.0 - 0.5 * inj_risk)

            available_at = float(now) + float(availability_delay_s)

            # --- InformationEvent: one per (event, source), upsert. ---
            ie = (
                InformationEvent.query
                .filter_by(event_id=event.id, source_content_id=source.id)
                .one_or_none()
            )
            is_new = ie is None
            if ie is None:
                ie = InformationEvent(event_id=event.id, source_content_id=source.id)
                db.session.add(ie)
            ie.relevance = relevance
            ie.credibility = credibility
            ie.freshness = freshness
            ie.novelty = 1.0 if is_new else 0.0
            ie.impact = impact
            ie.stance = stance
            ie.prompt_injection_risk = inj_risk
            ie.injection_flags = flags or None
            ie.published_at = published
            ie.retrieved_at = now_dt
            if is_new or ie.available_to_agents_at is None:
                ie.available_to_agents_at = available_at
            db.session.flush()
            info_events.append(ie)

        return info_events, new_sources

    # ------------------------------------------------------------------
    # Appraisal heuristics

    @staticmethod
    def _relevance(title: str, text: str, provider_score) -> float:
        try:
            base = float(provider_score)
            if base > 0:
                return max(0.0, min(1.0, base))
        except (TypeError, ValueError):
            pass
        # Lexical overlap fallback.
        t_words = {w for w in title.lower().split() if len(w) > 3}
        if not t_words:
            return 0.5
        hits = sum(1 for w in t_words if w in text.lower())
        return max(0.1, min(1.0, hits / max(1, len(t_words))))

    @staticmethod
    def _credibility_floor(source_type: str) -> float:
        return {
            SOURCE_TYPE_OFFICIAL: 0.9, SOURCE_TYPE_WIRE: 0.85,
            SOURCE_TYPE_NEWS: 0.5, SOURCE_TYPE_BLOG: 0.3,
            SOURCE_TYPE_SOCIAL: 0.25,
        }.get(source_type, 0.4)

    @staticmethod
    def _stance(text: str) -> str:
        low = text.lower()
        sup = sum(1 for w in _SUPPORT_WORDS if w in low)
        ref = sum(1 for w in _REFUTE_WORDS if w in low)
        if sup > ref:
            return STANCE_SUPPORT
        if ref > sup:
            return STANCE_REFUTE
        return STANCE_NEUTRAL

    @staticmethod
    def _impact(relevance, credibility, freshness, stance) -> float:
        directional = 0.0 if stance == STANCE_NEUTRAL else 1.0
        return max(0.0, min(1.0,
            0.4 * relevance + 0.3 * credibility + 0.2 * freshness + 0.1 * directional))

    # ==================================================================
    # Versioning + delta
    # ==================================================================

    def _latest_bundle(self, event_id: int) -> Optional[EvidenceBundle]:
        return (
            EvidenceBundle.query.filter_by(event_id=event_id)
            .order_by(EvidenceBundle.version.desc())
            .first()
        )

    def _maybe_new_bundle(
        self, event_id: int, info_events: List[InformationEvent], now: float,
    ) -> Dict[str, Any]:
        """Cut a new bundle+delta only if the evidence set is meaningfully novel."""
        current_ids = sorted({ie.id for ie in info_events})
        latest = self._latest_bundle(event_id)

        prev_ids = set()
        prev_impact = 0.0
        prev_version = None
        if latest is not None:
            prev_ids = set(
                (latest.supporting_evidence_ids or [])
                + (latest.opposing_evidence_ids or [])
                + (latest.neutral_evidence_ids or [])
            )
            prev_impact = latest.aggregate_impact
            prev_version = latest.version

        cur_set = set(current_ids)
        added = sorted(cur_set - prev_ids)
        removed = sorted(prev_ids - cur_set)

        # Aggregate impact of the CURRENT set.
        agg_impact = round(
            sum(ie.impact for ie in info_events) / max(1, len(info_events)), 4
        ) if info_events else 0.0
        impact_shift = abs(agg_impact - prev_impact)

        novel = (
            latest is None
            or len(added) + len(removed) >= _NOVELTY_MIN_CHANGED
            or impact_shift >= _NOVELTY_MIN_IMPACT_SHIFT
        )
        if not novel:
            return {
                "version": prev_version, "new_version": False,
                "added": 0, "removed": 0, "aggregate_impact": agg_impact,
                "delta_id": None,
            }

        # Categorize by stance / source type.
        support, oppose, neutral, official = [], [], [], []
        for ie in info_events:
            if ie.stance == STANCE_SUPPORT:
                support.append(ie.id)
            elif ie.stance == STANCE_REFUTE:
                oppose.append(ie.id)
            else:
                neutral.append(ie.id)
            if ie.source and ie.source.source_type == SOURCE_TYPE_OFFICIAL:
                official.append(ie.id)

        new_version = 1 if latest is None else latest.version + 1
        contradiction = self._contradiction_summary(support, oppose)

        bundle = EvidenceBundle(
            event_id=event_id, version=new_version,
            previous_bundle_id=(latest.id if latest else None),
            added_evidence_ids=added, removed_evidence_ids=removed,
            superseded_evidence_ids=removed,
            supporting_evidence_ids=support, opposing_evidence_ids=oppose,
            neutral_evidence_ids=neutral, official_evidence_ids=official,
            what_changed=(
                f"+{len(added)} / -{len(removed)} sources; "
                f"impact {prev_impact:.2f}→{agg_impact:.2f}"
            ),
            contradiction_changes=contradiction,
            uncertainty_summary=(
                f"support={len(support)} oppose={len(oppose)} neutral={len(neutral)}"
            ),
            aggregate_impact=agg_impact,
            current_summary=_limit_text(
                self._compact_summary(info_events), _MAX_BUNDLE_SUMMARY_CHARS
            ),
            model_metadata={},
        )
        db.session.add(bundle)
        db.session.flush()

        delta = EvidenceDelta(
            event_id=event_id, bundle_id=bundle.id,
            from_version=prev_version, to_version=new_version,
            added_facts=[
                {"id": ie.id, "title": (ie.source.title if ie.source else None),
                 "stance": ie.stance, "impact": round(ie.impact, 3)}
                for ie in info_events if ie.id in set(added)
            ][:_MAX_DELTA_FACTS],
            removed_facts=removed,
            changed_contradictions=contradiction,
            uncertainty_change=round(
                self._uncertainty(support, oppose) -
                (self._prev_uncertainty(latest)), 4
            ),
            impact_delta=round(agg_impact - prev_impact, 4),
            previous_probability_ref=(f"bundle:{latest.id}" if latest else None),
        )
        db.session.add(delta)
        db.session.flush()

        return {
            "version": new_version, "new_version": True,
            "added": len(added), "removed": len(removed),
            "aggregate_impact": agg_impact, "delta_id": delta.id,
            "bundle_id": bundle.id,
        }

    @staticmethod
    def _contradiction_summary(support_ids, oppose_ids) -> Dict[str, Any]:
        contested = bool(support_ids) and bool(oppose_ids)
        return {
            "contested": contested,
            "support": len(support_ids),
            "oppose": len(oppose_ids),
        }

    @staticmethod
    def _uncertainty(support_ids, oppose_ids) -> float:
        s, o = len(support_ids), len(oppose_ids)
        total = s + o
        if total == 0:
            return 0.5
        return round(min(s, o) / total, 4)  # 0 = one-sided, 0.5 = split

    def _prev_uncertainty(self, latest: Optional[EvidenceBundle]) -> float:
        if latest is None:
            return 0.5
        return self._uncertainty(
            latest.supporting_evidence_ids or [], latest.opposing_evidence_ids or []
        )

    @staticmethod
    def _compact_summary(info_events: List[InformationEvent], max_items: int = 5) -> str:
        top = sorted(info_events, key=lambda e: e.impact, reverse=True)[:max_items]
        bits = []
        for ie in top:
            t = (ie.source.title if ie.source else None) or "(untitled)"
            bits.append(f"[{ie.stance[:1]}] {t[:80]}")
        return " | ".join(bits) if bits else "(no evidence)"

    # ==================================================================
    # Source selection (token-budgeted top-K with preservation)
    # ==================================================================

    def select_sources(
        self, event_id: int, top_k: int = _DEFAULT_TOP_K,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
        as_of: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Pick a token-budgeted evidence subset for a prompt.

        Ranks by utility (impact·credibility·freshness), then GUARANTEES at
        least one strong SUPPORT and one strong OPPOSE source (when they
        exist) and prioritizes official/resolution-relevant sources.
        Respects `as_of` historical fairness: only evidence whose
        `available_to_agents_at <= as_of` is eligible.
        """
        self.ensure_schema()
        if as_of is None:
            as_of = SchedulerService.now()

        q = InformationEvent.query.filter(
            InformationEvent.event_id == event_id,
            InformationEvent.available_to_agents_at <= as_of,
        )
        candidates = q.all()
        if not candidates:
            db.session.expunge_all()
            return {"selected": [], "token_estimate": 0, "dropped": 0,
                    "has_support": False, "has_oppose": False}

        def utility(ie: InformationEvent) -> float:
            off = 1.0 if (ie.source and ie.source.source_type == SOURCE_TYPE_OFFICIAL) else 0.0
            return (0.5 * ie.impact + 0.2 * ie.credibility + 0.15 * ie.freshness
                    + 0.15 * off)

        ranked = sorted(candidates, key=utility, reverse=True)

        selected: List[InformationEvent] = []
        used_tokens = 0

        def _try_add(ie: InformationEvent) -> bool:
            nonlocal used_tokens
            if ie in selected:
                return False
            text = (ie.source.cleaned_text if ie.source else "") or ""
            tok = _estimate_tokens(text)
            if selected and used_tokens + tok > token_budget:
                return False
            selected.append(ie)
            used_tokens += tok
            return True

        # 1. Preserve one strong SUPPORT + one strong OPPOSE first (highest
        #    utility of each stance), so budget pressure can't erase a side.
        strong_support = next((ie for ie in ranked if ie.stance == STANCE_SUPPORT), None)
        strong_oppose = next((ie for ie in ranked if ie.stance == STANCE_REFUTE), None)
        for pin in (strong_support, strong_oppose):
            if pin is not None:
                _try_add(pin)

        # 2. Fill by utility up to top_k / budget.
        for ie in ranked:
            if len(selected) >= top_k:
                break
            _try_add(ie)

        has_support = any(ie.stance == STANCE_SUPPORT for ie in selected)
        has_oppose = any(ie.stance == STANCE_REFUTE for ie in selected)
        result = {
            "selected": [self._evidence_view(ie) for ie in selected],
            "token_estimate": used_tokens,
            "token_budget": token_budget,
            "dropped": len(candidates) - len(selected),
            "has_support": has_support,
            "has_oppose": has_oppose,
        }
        db.session.expunge_all()
        return result

    @staticmethod
    def _evidence_view(ie: InformationEvent) -> Dict[str, Any]:
        s = ie.source
        return {
            "information_event_id": ie.id,
            "title": (s.title if s else None),
            "source_domain": (s.source_domain if s else None),
            "source_type": (s.source_type if s else None),
            "stance": ie.stance,
            "impact": round(ie.impact, 3),
            "credibility": round(ie.credibility, 3),
            "injection_risk": round(ie.prompt_injection_risk, 3),
            "injection_flags": ie.injection_flags or [],
            # Structured facts preferred over raw prose; we hand a capped
            # cleaned snippet, never the full page.
            "text": ((s.cleaned_text if s else "") or "")[:500],
        }

    # ==================================================================
    # Prompt context (stable prefix + dynamic suffix, deterministic hashes)
    # ==================================================================

    def build_prompt_context(
        self, event_id: int, previous_probability: Optional[float] = None,
        archetype_definition: Optional[str] = None,
        json_schema_version: str = "forecast-v1",
        system_prompt_version: str = "sys-v1",
        as_of: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Assemble a cache-friendly prompt context for a belief update.

        STABLE PREFIX (hashed, deterministic — no UUID / wall-clock /
        volatile ordering): system instruction, security instruction, JSON
        schema, market definition, resolution rules, archetype placeholder.

        DYNAMIC SUFFIX: previous probability, Evidence Delta, current
        simulation time, current uncertainty.

        Returns the assembled text blocks + the component hashes + the
        router decision (which tier a belief update WOULD use).
        """
        self.ensure_schema()
        event = db.session.get(Event, event_id)
        if event is None:
            raise ValueError(f"event {event_id} not found")
        if as_of is None:
            as_of = SchedulerService.now()

        market_definition = f"{event.title}\n{event.description or ''}".strip()
        resolution_rules = event.resolution_source or "(resolution source unspecified)"

        # --- stable prefix components ---
        stable_components = {
            "system_instruction": "You are a probabilistic forecaster for a binary market.",
            "security_instruction": UNTRUSTED_PREAMBLE,
            "json_schema": '{"probability_yes": float, "confidence": float, "reasoning_summary": string}',
            "market_definition": market_definition,
            "resolution_rules": resolution_rules,
            "archetype_placeholder": archetype_definition or "{{ARCHETYPE_DEFINITION}}",
        }
        # Deterministic per-component hashes (order-independent).
        component_hashes = {k: hash_text(v) for k, v in stable_components.items()}

        # Reuse the cache-metadata helper for the canonical stable/dynamic
        # split hashes (never folds in UUID/time — enforced by that module).
        latest = self._latest_bundle(event_id)
        bundle_version = str(latest.version) if latest else "0"
        latest_delta = None
        if latest is not None:
            latest_delta = (
                EvidenceDelta.query.filter_by(event_id=event_id, to_version=latest.version)
                .first()
            )
        delta_hash = evidence_delta_hash(
            prev_bundle_version=(latest.previous_bundle_id if latest else None),
            new_bundle_version=bundle_version,
            added_ids=(latest_delta.removed_facts if latest_delta else None) or
                      ([f["id"] for f in (latest_delta.added_facts or [])] if latest_delta else []),
        )

        selection = self.select_sources(event_id, as_of=as_of)
        uncertainty = self._uncertainty(
            [e for e in (latest.supporting_evidence_ids or [])] if latest else [],
            [e for e in (latest.opposing_evidence_ids or [])] if latest else [],
        )

        cache_meta = build_cache_metadata(
            system_prompt_version=system_prompt_version,
            schema_version=json_schema_version,
            market_definition=market_definition,
            resolution_rules=resolution_rules,
            archetype_definition=archetype_definition,
            evidence_bundle_version=bundle_version,
            prior_probability=previous_probability,
            evidence_delta=delta_hash,
            time_bucket=str(int(as_of // 300)),  # 5-min bucket, deterministic
        )

        # --- route (label the tier a belief update WOULD use) ---
        route = self._route_for_bundle(event_id, latest, latest_delta, selection)

        # --- assemble text ---
        stable_prefix_text = "\n\n".join(
            f"### {k}\n{v}" for k, v in stable_components.items()
        )
        dynamic_suffix_text = self._render_dynamic_suffix(
            previous_probability, latest_delta, as_of, uncertainty, selection
        )

        result = {
            "event_id": event_id,
            "stable_prefix": stable_prefix_text,
            "dynamic_suffix": dynamic_suffix_text,
            "stable_component_hashes": component_hashes,
            "stable_prefix_hash": cache_meta.stable_prefix_hash,
            "dynamic_suffix_hash": cache_meta.dynamic_suffix_hash,
            "cache_key": cache_meta.cache_key,
            "bundle_version": bundle_version,
            "selection": selection,
            "route": route,
            "untrusted_preamble_present": UNTRUSTED_PREAMBLE in stable_prefix_text,
        }
        db.session.expunge_all()
        return result

    def _render_dynamic_suffix(
        self, prev_prob, delta, as_of, uncertainty, selection,
    ) -> str:
        lines = [
            f"previous_probability: {prev_prob if prev_prob is not None else 'n/a'}",
            f"current_simulation_time_s: {as_of}",
            f"current_uncertainty: {uncertainty}",
        ]
        if delta is not None:
            lines.append(
                f"evidence_delta: +{len(delta.added_facts or [])} facts, "
                f"-{len(delta.removed_facts or [])} facts, "
                f"impactΔ={delta.impact_delta:+.3f}, "
                f"contradictions={delta.changed_contradictions}"
            )
        else:
            lines.append("evidence_delta: (initial — no prior version)")
        lines.append("--- EVIDENCE (untrusted external content) ---")
        for v in selection["selected"]:
            flag = f" [FLAGGED: {','.join(v['injection_flags'])}]" if v["injection_flags"] else ""
            lines.append(f"- ({v['stance']}, {v['source_type']}){flag} {v['title']}: {v['text'][:200]}")
        return "\n".join(lines)

    def _route_for_bundle(self, event_id, latest, latest_delta, selection) -> Dict[str, Any]:
        """Ask the ModelRouter which tier a belief update on this bundle
        would use — classification/routine → FAST, contested → BALANCED,
        major high-impact contradiction → STRONG. Honors budgets. Does NOT
        call any model."""
        if self._router is None:
            return {"routed": False, "reason": "no router supplied"}
        from llm import TaskRoutingContext, TaskType

        contested = False
        impact = 0.0
        conflict = 0.0
        if latest is not None:
            cc = latest.contradiction_changes or {}
            contested = bool(cc.get("contested"))
            impact = latest.aggregate_impact
            s, o = len(latest.supporting_evidence_ids or []), len(latest.opposing_evidence_ids or [])
            conflict = (min(s, o) / (s + o)) if (s + o) else 0.0

        impact_delta = abs(latest_delta.impact_delta) if latest_delta else 0.0

        # Choose the task type by situation.
        if contested and impact >= 0.6 and impact_delta >= 0.1:
            task = TaskType.MAJOR_BELIEF_UPDATE           # → STRONG
        elif contested:
            task = TaskType.EVIDENCE_CONFLICT_ANALYSIS    # → BALANCED
        else:
            task = TaskType.EVIDENCE_SUMMARY              # → FAST (routine)

        # Impact/conflict signals only escalate a CONTESTED update. A
        # routine one-sided summary must stay FAST even when the lone
        # source is high-impact — STRONG is reserved for high-impact
        # *contradiction*, not high-impact agreement. So we only feed the
        # router the escalation signals when the evidence is contested.
        routed_impact = impact if contested else 0.0
        routed_conflict = conflict if contested else 0.0

        ctx = TaskRoutingContext(
            task_type=task,
            evidence_count=len(selection["selected"]),
            evidence_conflict_score=routed_conflict,
            information_impact_score=routed_impact,
            structured_output_required=True,
            market_id=None,
            evidence_bundle_id=str(getattr(latest, "id", "") or ""),
            batch_eligible=not (contested and impact >= 0.6),
        )
        decision = self._router.route(ctx)
        return {
            "routed": True,
            "task_type": task.value,
            "tier": decision.tier,
            "batch_mode": decision.batch_mode,
            "budget_allowed": decision.budget_allowed,
            "reason": decision.reason,
        }

    # ==================================================================
    # Read helpers for the CLI
    # ==================================================================

    def inspect(self, event_id: int) -> Dict[str, Any]:
        self.ensure_schema()
        latest = self._latest_bundle(event_id)
        n_sources = (
            db.session.query(db.func.count(InformationEvent.id))
            .filter(InformationEvent.event_id == event_id).scalar() or 0
        )
        n_bundles = (
            db.session.query(db.func.count(EvidenceBundle.id))
            .filter(EvidenceBundle.event_id == event_id).scalar() or 0
        )
        out = {
            "event_id": event_id,
            "information_events": n_sources,
            "bundles": n_bundles,
            "latest_version": (latest.version if latest else None),
            "aggregate_impact": (latest.aggregate_impact if latest else None),
            "current_summary": (latest.current_summary if latest else None),
            "support": (len(latest.supporting_evidence_ids or []) if latest else 0),
            "oppose": (len(latest.opposing_evidence_ids or []) if latest else 0),
            "official": (len(latest.official_evidence_ids or []) if latest else 0),
        }
        db.session.expunge_all()
        return out

    def get_delta(self, event_id: int, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        self.ensure_schema()
        q = EvidenceDelta.query.filter_by(event_id=event_id)
        if version is None:
            delta = q.order_by(EvidenceDelta.to_version.desc()).first()
        else:
            delta = q.filter_by(to_version=version).first()
        if delta is None:
            db.session.expunge_all()
            return None
        out = {
            "event_id": event_id,
            "from_version": delta.from_version,
            "to_version": delta.to_version,
            "added_facts": delta.added_facts,
            "removed_facts": delta.removed_facts,
            "changed_contradictions": delta.changed_contradictions,
            "uncertainty_change": delta.uncertainty_change,
            "impact_delta": delta.impact_delta,
            "previous_probability_ref": delta.previous_probability_ref,
        }
        db.session.expunge_all()
        return out


__all__ = ["EvidenceService"]
