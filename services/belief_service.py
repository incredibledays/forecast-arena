"""BeliefService — Archetype-level LLM belief updates + lazy Agent beliefs.

Two-tier belief system that keeps LLM cost O(archetypes), never O(agents):

  update_archetype_beliefs(event_id)
      For each RELEVANT archetype, produce ONE belief from the latest
      EvidenceBundle/Delta via a routed, batched LLM call. Compatible
      archetypes (same event + bundle version + schema + tier) are
      grouped into batches of 10–30 and answered in a single request;
      each result is validated independently and only failed items are
      retried (with bounded repair/escalation). Stored: concise reasoning
      only — never chain-of-thought.

  materialize_agent_beliefs(event_id, agent_ids)
      Individual beliefs are computed in PURE CODE (no LLM, ever) by
      personalizing the archetype posterior per Agent. Only ELIGIBLE
      agents (holders / watchers / subscribers / woken / needs-audit) get
      a persisted AgentBelief row — sleeping agents do not. The batch path
      uses compact columnar projections (no ORM object / big dict per
      Agent, no per-Agent archetype query, one bulk upsert).

  reconstruct_agent_belief(agent_id, event_id)
      Recompute a non-materialized belief from the ArchetypeBelief + the
      Agent projection + deterministic seed. Matches the persisted value
      within floating-point tolerance (same math).

Personalization (additive in logit space):

    logit(p_agent) = logit(p_arch)
                   + expertise_adjustment
                   + source_trust_adjustment
                   + bias_adjustment
                   + memory_calibration_adjustment
                   + deterministic_noise
    p_agent = clamp(sigmoid(logit(p_agent)), 0.01, 0.99)

Deterministic noise depends on (sim_seed, agent_seed, event_id,
bundle_version, personalization_algo_version).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models import (
    Agent,
    AgentArchetype,
    AgentBelief,
    AgentEventInterest,
    ArchetypeBelief,
    BELIEF_STATUS_MATERIALIZED,
    BELIEF_STATUS_RECONSTRUCTED,
    Event,
    EvidenceBundle,
    Market,
    PERSONALIZATION_ALGO_VERSION,
    Position,
    ROLE_SUBSCRIBER,
    ROLE_WATCHER,
    db,
)
from services._schema_cache import ensure_created as _ensure_schema_cached
from services.scheduler_service import SchedulerService


# --- personalization coefficients (pinned; algo version guards changes) ---
_K_EXPERTISE = 0.35     # experts lean further into the archetype's read
_K_TRUST = 0.20         # trust in sources amplifies the signal
_K_OVERCONF = 0.40      # overconfidence amplifies deviation from 0.5
_K_HERDING = 0.30       # herding dampens toward the crowd (0.5 here)
_K_MEMORY = 0.50        # additive calibration offset from memory
_NOISE_SCALE = 0.30     # width of the deterministic idiosyncratic noise

_PROB_FLOOR = 0.01
_PROB_CEIL = 0.99

_PROMPT_VERSION = "belief-v1"
_DEFAULT_BATCH = 20         # archetypes per compatible batch (10–30)
_MIN_BATCH = 10
_MAX_BATCH = 30
_MAX_LLM_CONTEXT_CHARS = 12000


def _bounded_context(text: str, max_chars: int = _MAX_LLM_CONTEXT_CHARS) -> str:
    text = " ".join(str(text or "").split())
    if not text:
        return "(no evidence)"
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 80].rstrip() + " ... [truncated to fit context budget]"


# ==================================================================
# Pure math (stable, dependency-free, unit-testable)
# ==================================================================

def stable_logit(p: float) -> float:
    """logit with clamping so it never blows up at the boundaries."""
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def stable_sigmoid(x: float) -> float:
    """Numerically-stable logistic."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _det_uniform(sim_seed, agent_seed, event_id, bundle_version, algo_version) -> float:
    """Deterministic uniform in [0,1) from the seed tuple (blake2b, no salt)."""
    key = f"{int(sim_seed)}|{int(agent_seed)}|{int(event_id)}|{int(bundle_version)}|{int(algo_version)}".encode()
    bits = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") >> 11
    return bits / float(1 << 53)


# A compact projection row is a tuple — NOT an ORM object or a dict per agent:
#   (agent_id, archetype_id, agent_seed, expertise, avg_trust,
#    overconfidence, herding, memory_calibration)
ProjectionRow = Tuple[int, int, int, float, float, float, float, float]


def personalize_one(
    p_archetype: float,
    row: ProjectionRow,
    *,
    sim_seed: int,
    event_id: int,
    bundle_version: int,
    algo_version: int = PERSONALIZATION_ALGO_VERSION,
    return_components: bool = False,
):
    """Scalar REFERENCE personalization for one projection row.

    Returns the calibrated probability, or (calibrated, raw, components)
    when `return_components`. This is the ground truth the vectorized
    batch path must reproduce exactly.
    """
    (_aid, _arch, agent_seed, expertise, avg_trust,
     overconfidence, herding, memory_cal) = row

    le = stable_logit(p_archetype)
    expertise_adj = _K_EXPERTISE * (expertise - 0.5) * le
    trust_adj = _K_TRUST * (avg_trust - 0.5) * le
    bias_adj = (_K_OVERCONF * overconfidence - _K_HERDING * herding) * le
    memory_adj = _K_MEMORY * memory_cal
    u = _det_uniform(sim_seed, agent_seed, event_id, bundle_version, algo_version)
    noise = _NOISE_SCALE * (u - 0.5)

    raw_logit = le + expertise_adj + trust_adj + bias_adj + memory_adj + noise
    raw_p = stable_sigmoid(raw_logit)
    calibrated = min(_PROB_CEIL, max(_PROB_FLOOR, raw_p))

    if return_components:
        components = {
            "logit_archetype": round(le, 6),
            "expertise_adj": round(expertise_adj, 6),
            "trust_adj": round(trust_adj, 6),
            "bias_adj": round(bias_adj, 6),
            "memory_adj": round(memory_adj, 6),
            "noise": round(noise, 6),
        }
        return calibrated, raw_p, components
    return calibrated


def personalize_batch(
    p_archetype_by_arch: Dict[int, float],
    rows: Sequence[ProjectionRow],
    *,
    sim_seed: int,
    event_id: int,
    bundle_version: int,
    algo_version: int = PERSONALIZATION_ALGO_VERSION,
) -> List[Tuple[float, float]]:
    """Vectorized-style batch personalization over columnar projection rows.

    Processes rows without building an ORM object or a per-Agent dict, and
    without re-querying the archetype (posteriors handed in as a small
    map). Returns a list of (calibrated, raw) aligned with `rows`. Produces
    identical numbers to `personalize_one` (same formula, applied per
    element).
    """
    out: List[Tuple[float, float]] = []
    # Cache logit(posterior) per archetype so we don't recompute per row.
    le_cache: Dict[int, float] = {}
    for row in rows:
        arch_id = row[1]
        p_arch = p_archetype_by_arch.get(arch_id)
        if p_arch is None:
            # No archetype belief for this row → neutral fallback.
            out.append((0.5, 0.5))
            continue
        le = le_cache.get(arch_id)
        if le is None:
            le = stable_logit(p_arch)
            le_cache[arch_id] = le
        (_aid, _arch, agent_seed, expertise, avg_trust,
         overconfidence, herding, memory_cal) = row
        raw_logit = (
            le
            + _K_EXPERTISE * (expertise - 0.5) * le
            + _K_TRUST * (avg_trust - 0.5) * le
            + (_K_OVERCONF * overconfidence - _K_HERDING * herding) * le
            + _K_MEMORY * memory_cal
            + _NOISE_SCALE * (_det_uniform(sim_seed, agent_seed, event_id, bundle_version, algo_version) - 0.5)
        )
        raw_p = stable_sigmoid(raw_logit)
        out.append((min(_PROB_CEIL, max(_PROB_FLOOR, raw_p)), raw_p))
    return out


# ==================================================================
# Strict-JSON schema for the archetype belief
# ==================================================================

_REQUIRED_KEYS = ("posterior_probability_yes", "confidence", "reasoning_summary")


def _validate_belief_json(obj: Any) -> Optional[Dict[str, Any]]:
    """Coerce/validate one archetype-belief JSON object. None if invalid."""
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in _REQUIRED_KEYS):
        return None
    try:
        p = float(obj["posterior_probability_yes"])
        c = float(obj["confidence"])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= p <= 1.0) or not (0.0 <= c <= 1.0):
        return None
    reasoning = str(obj.get("reasoning_summary") or "").strip()
    if not reasoning:
        return None
    key_ev = obj.get("key_evidence")
    key_ev = [str(x)[:120] for x in key_ev][:6] if isinstance(key_ev, list) else []
    risks = obj.get("risk_factors")
    risks = [str(x)[:120] for x in risks][:6] if isinstance(risks, list) else []
    return {
        "posterior_probability_yes": p,
        "confidence": c,
        # Concise reasoning ONLY — capped, never chain-of-thought.
        "reasoning_summary": reasoning[:400],
        "key_evidence": key_ev,
        "risk_factors": risks,
    }


class BeliefService:
    """Archetype belief updates + lazy Agent belief materialization."""

    def __init__(self, llm_client=None, router=None):
        self._llm = llm_client
        self._router = router

    @staticmethod
    def ensure_schema() -> None:
        _ensure_schema_cached()

    # ==================================================================
    # Relevant archetypes + eligibility (sparse — no full population load)
    # ==================================================================

    @staticmethod
    def _market_ids(event_id: int) -> List[int]:
        return [m[0] for m in db.session.query(Market.id).filter(Market.event_id == event_id).all()]

    @classmethod
    def relevant_archetype_ids(cls, event_id: int) -> List[int]:
        """Archetypes with at least one eligible agent on this event.

        Derived from holders + watchers/subscribers (sparse), mapped to
        their archetypes with a bounded id-IN query — never a full scan.
        """
        agent_ids = cls.eligible_agent_ids(event_id)
        if not agent_ids:
            return []
        arch_ids = set()
        ids = list(agent_ids)
        CHUNK = 900
        for i in range(0, len(ids), CHUNK):
            rows = (
                db.session.query(Agent.archetype_id)
                .filter(Agent.id.in_(ids[i:i + CHUNK]),
                        Agent.archetype_id.isnot(None))
                .distinct().all()
            )
            arch_ids.update(r[0] for r in rows)
        return sorted(arch_ids)

    @classmethod
    def eligible_agent_ids(
        cls, event_id: int, woken_ids: Optional[Sequence[int]] = None,
    ) -> set:
        """Sparse union of agents eligible for a materialized belief.

        holders (Position on the event's markets) ∪ watchers ∪ subscribers
        ∪ explicitly woken. Sleeping agents with no stake are excluded, so
        they never all acquire AgentBelief rows.
        """
        ids: set = set()
        mids = cls._market_ids(event_id)
        if mids:
            for (aid,) in (
                db.session.query(Position.agent_id)
                .filter(Position.market_id.in_(mids)).distinct().all()
            ):
                ids.add(aid)
        for (aid,) in (
            db.session.query(AgentEventInterest.agent_id)
            .filter(AgentEventInterest.event_id == event_id,
                    AgentEventInterest.role.in_((ROLE_WATCHER, ROLE_SUBSCRIBER)))
            .distinct().all()
        ):
            ids.add(aid)
        if woken_ids:
            ids.update(int(a) for a in woken_ids)
        return ids

    # ==================================================================
    # Archetype belief update (LLM, batched, per archetype not per agent)
    # ==================================================================

    def update_archetype_beliefs(
        self, event_id: int, archetype_ids: Optional[Sequence[int]] = None,
        batch_size: int = _DEFAULT_BATCH, now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Produce one belief per relevant archetype from the latest bundle.

        Groups compatible archetypes (same event/bundle/schema/tier) into
        batches, issues ONE routed LLM request per batch, validates each
        result independently, and retries only failed items with bounded
        repair/escalation. Returns metrics (incl. tier distribution and
        LLM request count).
        """
        self.ensure_schema()
        if now is None:
            now = SchedulerService.now()

        latest = (
            EvidenceBundle.query.filter_by(event_id=event_id)
            .order_by(EvidenceBundle.version.desc()).first()
        )
        bundle_version = latest.version if latest else 0
        prior_prob, conflict, impact, contested = self._bundle_signals(latest)

        if archetype_ids is None:
            archetype_ids = self.relevant_archetype_ids(event_id)
        archetype_ids = list(archetype_ids)

        metrics = {
            "event_id": event_id,
            "bundle_version": bundle_version,
            "relevant_archetypes": len(archetype_ids),
            "archetype_llm_requests": 0,
            "individual_agent_llm_requests": 0,   # invariant: always 0
            "tier_distribution": {},
            "batches": 0,
            "retries": 0,
            "degraded": 0,
            "cache_eligible": False,
        }
        if not archetype_ids:
            return metrics

        batch_size = max(_MIN_BATCH, min(_MAX_BATCH, int(batch_size)))

        # Route once for this event/bundle — all archetypes share the same
        # evidence, so they share a tier ⇒ they are batch-compatible.
        route = self._route(event_id, latest, conflict, impact, contested,
                            n_archetypes=len(archetype_ids))
        tier = route["tier"]
        metrics["cache_eligible"] = route.get("cache_eligible", False)

        # Skip archetypes already computed for this exact bundle version.
        todo = [
            aid for aid in archetype_ids
            if not self._has_current_belief(aid, event_id, bundle_version)
        ]

        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            metrics["batches"] += 1
            results, n_requests, n_retries, n_degraded = self._process_batch(
                batch, event_id, bundle_version, prior_prob, tier, route, latest,
            )
            metrics["archetype_llm_requests"] += n_requests
            metrics["retries"] += n_retries
            metrics["degraded"] += n_degraded
            for aid, parsed, degraded in results:
                self._store_archetype_belief(
                    aid, event_id, bundle_version, prior_prob, parsed, route, degraded
                )
                metrics["tier_distribution"][tier] = metrics["tier_distribution"].get(tier, 0) + 1

        db.session.commit()
        db.session.expunge_all()
        return metrics

    def _has_current_belief(self, archetype_id, event_id, bundle_version) -> bool:
        return db.session.query(ArchetypeBelief.id).filter_by(
            archetype_id=archetype_id, event_id=event_id,
            evidence_bundle_version=bundle_version,
        ).first() is not None

    @staticmethod
    def _bundle_signals(latest: Optional[EvidenceBundle]):
        if latest is None:
            return None, 0.0, 0.0, False
        s = len(latest.supporting_evidence_ids or [])
        o = len(latest.opposing_evidence_ids or [])
        conflict = (min(s, o) / (s + o)) if (s + o) else 0.0
        contested = bool((latest.contradiction_changes or {}).get("contested"))
        return None, conflict, latest.aggregate_impact, contested

    def _route(self, event_id, latest, conflict, impact, contested, n_archetypes):
        """Route the belief update: routine→FAST, contested→BALANCED,
        major high-impact contradiction→STRONG. Honors budgets."""
        if self._router is None:
            return {"tier": "BALANCED", "batch_mode": "MICRO_BATCH",
                    "provider": None, "model": None, "cache_eligible": True,
                    "budget_allowed": True, "reason": "no router; default BALANCED"}
        from llm import TaskRoutingContext, TaskType
        if contested and impact >= 0.6:
            task = TaskType.MAJOR_BELIEF_UPDATE
            routed_impact, routed_conflict = impact, conflict
        elif contested:
            task = TaskType.EVIDENCE_CONFLICT_ANALYSIS
            routed_impact, routed_conflict = 0.0, conflict
        else:
            task = TaskType.ROUTINE_BELIEF_UPDATE
            routed_impact, routed_conflict = 0.0, 0.0
        ctx = TaskRoutingContext(
            task_type=task,
            evidence_count=n_archetypes,
            evidence_conflict_score=routed_conflict,
            information_impact_score=routed_impact,
            structured_output_required=True,
            cache_available=True,
            evidence_bundle_id=str(getattr(latest, "id", "") or ""),
            batch_eligible=not (contested and impact >= 0.6),
        )
        d = self._router.route(ctx)
        return {
            "tier": d.tier, "batch_mode": d.batch_mode, "provider": d.provider,
            "model": d.model, "cache_eligible": d.cache_eligible,
            "budget_allowed": d.budget_allowed, "task_type": task.value,
            "reason": d.reason,
        }

    def _process_batch(
        self, batch: List[int], event_id, bundle_version, prior_prob, tier, route, latest,
    ):
        """Issue ONE LLM request for the batch; validate each; retry failures.

        Returns (results, n_requests, n_retries, n_degraded) where results
        is [(archetype_id, parsed_dict, degraded_bool), ...].
        """
        summary = _bounded_context((latest.current_summary if latest else "") or "(no evidence)")
        pending = list(batch)
        parsed_by_arch: Dict[int, Dict[str, Any]] = {}
        degraded_arch: set = set()

        n_requests = 0
        n_retries = 0
        max_repairs = 2  # bounded repair/escalation

        attempt = 0
        while pending and attempt <= max_repairs:
            raw = self._call_llm_batch(pending, event_id, summary, tier, repair=(attempt > 0))
            n_requests += 1
            if attempt > 0:
                n_retries += 1
            attempt += 1

            got = self._parse_batch_response(raw, pending)
            still_failed = []
            for aid in pending:
                parsed = got.get(aid)
                if parsed is not None:
                    parsed_by_arch[aid] = parsed
                else:
                    still_failed.append(aid)
            pending = still_failed

        # Anything still unparsed after bounded repair → deterministic
        # degraded fallback (keeps the pipeline alive; flagged degraded).
        for aid in pending:
            parsed_by_arch[aid] = self._degraded_belief(prior_prob)
            degraded_arch.add(aid)

        results = [(aid, parsed_by_arch[aid], aid in degraded_arch) for aid in batch]
        return results, n_requests, n_retries, len(degraded_arch)

    # ---- LLM adapter (mock-friendly; never per-agent) ----

    def _call_llm_batch(self, archetype_ids, event_id, summary, tier, repair=False):
        """Call the LLM once for a batch. Returns the raw parsed object.

        When no LLM client is configured, uses a deterministic offline
        stub so tests/benchmarks run mock. The stub still returns strict
        JSON so the validation path is exercised.
        """
        from llm import UNTRUSTED_PREAMBLE
        if self._llm is None or not getattr(self._llm, "available", False):
            return self._offline_batch(archetype_ids, event_id, summary)

        system = (
            "You are a probabilistic forecaster. " + UNTRUSTED_PREAMBLE +
            ' Respond with STRICT JSON: {"beliefs":[{"archetype_id":int,'
            '"posterior_probability_yes":float,"confidence":float,'
            '"reasoning_summary":string,"key_evidence":[],"risk_factors":[]}]}.'
            " reasoning_summary must be one short natural-language sentence analyzing the event itself, not a trading action. No prose, no chain-of-thought."
            + (" Your previous reply was invalid JSON; return valid JSON only." if repair else "")
        )
        user = (
            f"Event {event_id}. Evidence (untrusted): {summary}\n"
            f"Produce one belief for each archetype id: {list(archetype_ids)}"
        )
        try:
            return self._llm.chat_json(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                max_tokens=800,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[belief] LLM batch call failed: {exc}", file=sys.stderr)
            return {}

    @staticmethod
    def _offline_batch(archetype_ids, event_id, summary):
        """Deterministic strict-JSON stub — one belief per archetype."""
        beliefs = []
        for aid in archetype_ids:
            u = _det_uniform(0, aid, event_id, 0, 0)
            p = round(0.35 + 0.3 * u, 4)   # spread in [0.35, 0.65]
            beliefs.append({
                "archetype_id": aid,
                "posterior_probability_yes": p,
                "confidence": round(0.5 + 0.3 * u, 3),
                "reasoning_summary": (
                    "The available evidence is limited, so this archetype keeps "
                    "the event probability near neutral."
                ),
                "key_evidence": [], "risk_factors": [],
            })
        return {"beliefs": beliefs}

    @staticmethod
    def _parse_batch_response(raw, archetype_ids) -> Dict[int, Dict[str, Any]]:
        """Map a batch response to {archetype_id: validated_belief}.

        Tolerates a `{"beliefs":[...]}` array (keyed by archetype_id or by
        positional order). Each entry is validated independently.
        """
        out: Dict[int, Dict[str, Any]] = {}
        beliefs = None
        if isinstance(raw, dict):
            beliefs = raw.get("beliefs")
            # A single-object response (batch of 1) is also accepted.
            if beliefs is None and any(k in raw for k in _REQUIRED_KEYS) and len(archetype_ids) == 1:
                beliefs = [dict(raw, archetype_id=archetype_ids[0])]
        if not isinstance(beliefs, list):
            return out
        by_pos = list(archetype_ids)
        for idx, entry in enumerate(beliefs):
            parsed = _validate_belief_json(entry)
            if parsed is None:
                continue
            aid = None
            if isinstance(entry, dict) and "archetype_id" in entry:
                try:
                    aid = int(entry["archetype_id"])
                except (TypeError, ValueError):
                    aid = None
            if aid is None and idx < len(by_pos):
                aid = by_pos[idx]
            if aid in archetype_ids:
                out[aid] = parsed
        return out

    @staticmethod
    def _degraded_belief(prior_prob):
        return {
            "posterior_probability_yes": 0.5 if prior_prob is None else float(prior_prob),
            "confidence": 0.2,
            "reasoning_summary": "degraded: LLM produced no valid belief; neutral fallback.",
            "key_evidence": [], "risk_factors": ["degraded_result"],
        }

    def _store_archetype_belief(self, archetype_id, event_id, bundle_version,
                                prior_prob, parsed, route, degraded):
        from llm import build_cache_metadata
        cache_meta = build_cache_metadata(
            evidence_bundle_version=str(bundle_version),
            prior_probability=prior_prob,
        )
        existing = ArchetypeBelief.query.filter_by(
            archetype_id=archetype_id, event_id=event_id,
            evidence_bundle_version=bundle_version,
        ).one_or_none()
        if existing is None:
            existing = ArchetypeBelief(
                archetype_id=archetype_id, event_id=event_id,
                evidence_bundle_version=bundle_version,
            )
            db.session.add(existing)
        existing.prior_probability = prior_prob
        existing.posterior_probability = parsed["posterior_probability_yes"]
        existing.confidence = parsed["confidence"]
        existing.reasoning_summary = parsed["reasoning_summary"]
        existing.key_evidence = parsed["key_evidence"]
        existing.risk_factors = parsed["risk_factors"]
        existing.model_provider = route.get("provider")
        existing.model = route.get("model")
        existing.model_tier = route.get("tier")
        existing.batch_mode = route.get("batch_mode")
        existing.prompt_version = _PROMPT_VERSION
        existing.cache_metadata = {
            "stable_prefix_hash": cache_meta.stable_prefix_hash,
            "cache_key": cache_meta.cache_key,
            "cache_eligible": route.get("cache_eligible", False),
        }
        existing.degraded = bool(degraded)
        db.session.flush()
        return existing

    # ==================================================================
    # Agent projections (compact columnar tuples — no ORM/dict per agent)
    # ==================================================================

    @staticmethod
    def _event_category(event_id: int) -> Optional[str]:
        row = db.session.query(Event.category).filter(Event.id == event_id).one_or_none()
        return row[0] if row else None

    @classmethod
    def _projections(
        cls, agent_ids: Sequence[int], category: Optional[str],
        memory_projection: Optional[Dict[int, float]] = None,
    ) -> List[ProjectionRow]:
        """Build compact projection tuples for `agent_ids`.

        Streams by id-IN chunks; extracts only the scalars personalization
        needs from `persona_overrides_json`. Never instantiates an ORM
        Agent object or a per-Agent dict that outlives the loop.
        """
        memory_projection = memory_projection or {}
        rows: List[ProjectionRow] = []
        ids = list(agent_ids)
        CHUNK = 900
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i:i + CHUNK]
            q = (
                db.session.query(
                    Agent.id, Agent.archetype_id, Agent.random_seed,
                    Agent.persona_overrides_json,
                )
                .filter(Agent.id.in_(chunk))
            )
            for aid, arch_id, seed, overrides in q:
                expertise, avg_trust, overconf, herding = cls._extract_scalars(overrides, category)
                rows.append((
                    int(aid), int(arch_id) if arch_id is not None else -1,
                    int(seed if seed is not None else aid),
                    expertise, avg_trust, overconf, herding,
                    float(memory_projection.get(aid, 0.0)),
                ))
        return rows

    @staticmethod
    def _extract_scalars(overrides, category):
        expertise, avg_trust, overconf, herding = 0.5, 0.5, 0.25, 0.25
        if isinstance(overrides, dict):
            exp = overrides.get("expertise") or {}
            if category and category in exp:
                try:
                    expertise = float(exp[category])
                except (TypeError, ValueError):
                    pass
            elif exp:
                try:
                    expertise = sum(float(v) for v in exp.values()) / len(exp)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            trust = overrides.get("source_trust") or {}
            if trust:
                try:
                    avg_trust = sum(float(v) for v in trust.values()) / len(trust)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            biases = overrides.get("biases") or {}
            try:
                overconf = float(biases.get("overconfidence", 0.25))
                herding = float(biases.get("herding", 0.25))
            except (TypeError, ValueError):
                pass
        return expertise, avg_trust, overconf, herding

    # ==================================================================
    # Materialization (lazy, eligible-only, bulk upsert)
    # ==================================================================

    def materialize_agent_beliefs(
        self, event_id: int, agent_ids: Optional[Sequence[int]] = None,
        woken_ids: Optional[Sequence[int]] = None, sim_seed: int = 0,
        now: Optional[float] = None, market_price: Optional[float] = None,
        memory_projection: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        """Persist AgentBelief rows for ELIGIBLE agents only (pure code).

        If `agent_ids` is None, uses the sparse eligibility set. Computes
        beliefs via the vectorized batch path and bulk-upserts. Never calls
        an LLM. Returns metrics including how many were persisted vs. how
        many eligible.
        """
        self.ensure_schema()
        if now is None:
            now = SchedulerService.now()
        if agent_ids is None:
            agent_ids = sorted(self.eligible_agent_ids(event_id, woken_ids=woken_ids))
        else:
            agent_ids = list(agent_ids)

        metrics = {
            "event_id": event_id, "eligible_agents": len(agent_ids),
            "individual_beliefs_calculated": 0, "beliefs_persisted": 0,
            "individual_agent_llm_requests": 0,   # invariant: 0
        }
        if not agent_ids:
            return metrics

        latest = (
            EvidenceBundle.query.filter_by(event_id=event_id)
            .order_by(EvidenceBundle.version.desc()).first()
        )
        bundle_version = latest.version if latest else 0

        # Archetype posteriors for this bundle — a small map, one query.
        p_by_arch = self._archetype_posteriors(event_id, bundle_version)
        belief_id_by_arch = self._archetype_belief_ids(event_id, bundle_version)

        category = self._event_category(event_id)
        rows = self._projections(agent_ids, category, memory_projection)
        personalized = personalize_batch(
            p_by_arch, rows, sim_seed=sim_seed, event_id=event_id,
            bundle_version=bundle_version,
        )
        metrics["individual_beliefs_calculated"] = len(personalized)

        # Existing beliefs for these agents (one query) → update vs insert.
        existing_ids = {
            r[0] for r in db.session.query(AgentBelief.agent_id).filter(
                AgentBelief.event_id == event_id,
                AgentBelief.agent_id.in_(agent_ids),
            ).all()
        }

        inserts: List[Dict[str, Any]] = []
        updates: List[Dict[str, Any]] = []
        for row, (calibrated, raw) in zip(rows, personalized):
            aid, arch_id = row[0], row[1]
            payload = {
                "agent_id": aid, "event_id": event_id,
                "archetype_belief_id": belief_id_by_arch.get(arch_id),
                "prior_probability": None,
                "raw_probability": raw,
                "calibrated_probability": calibrated,
                "confidence": 0.5,
                "belief_status": BELIEF_STATUS_MATERIALIZED,
                "last_evidence_bundle_version": bundle_version,
                "last_market_price_seen": market_price,
                "next_review_time": None,
                "personalization_components": None,
                "personalization_algo_version": PERSONALIZATION_ALGO_VERSION,
            }
            if aid in existing_ids:
                updates.append(payload)
            else:
                inserts.append(payload)

        if inserts:
            db.session.bulk_insert_mappings(AgentBelief, inserts)
        if updates:
            # Bulk update needs the PK; fetch id map in one query.
            id_map = {
                r[0]: r[1] for r in db.session.query(
                    AgentBelief.agent_id, AgentBelief.id
                ).filter(AgentBelief.event_id == event_id,
                         AgentBelief.agent_id.in_([u["agent_id"] for u in updates])).all()
            }
            for u in updates:
                u["id"] = id_map.get(u["agent_id"])
            db.session.bulk_update_mappings(
                AgentBelief, [u for u in updates if u.get("id")]
            )
        db.session.commit()
        db.session.expunge_all()

        metrics["beliefs_persisted"] = len(inserts) + len(updates)
        return metrics

    def _archetype_posteriors(self, event_id, bundle_version) -> Dict[int, float]:
        rows = db.session.query(
            ArchetypeBelief.archetype_id, ArchetypeBelief.posterior_probability
        ).filter_by(event_id=event_id, evidence_bundle_version=bundle_version).all()
        return {r[0]: r[1] for r in rows}

    def _archetype_belief_ids(self, event_id, bundle_version) -> Dict[int, int]:
        rows = db.session.query(
            ArchetypeBelief.archetype_id, ArchetypeBelief.id
        ).filter_by(event_id=event_id, evidence_bundle_version=bundle_version).all()
        return {r[0]: r[1] for r in rows}

    # ==================================================================
    # Reconstruction (non-materialized, must match persisted within tol)
    # ==================================================================

    def reconstruct_agent_belief(
        self, agent_id: int, event_id: int, sim_seed: int = 0,
        memory_projection: Optional[Dict[int, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Recompute a belief WITHOUT persisting it.

        Uses the same math as materialization, so a persisted row and its
        reconstruction agree within floating-point tolerance. Returns None
        if the agent or its archetype belief is missing.
        """
        self.ensure_schema()
        latest = (
            EvidenceBundle.query.filter_by(event_id=event_id)
            .order_by(EvidenceBundle.version.desc()).first()
        )
        bundle_version = latest.version if latest else 0
        category = self._event_category(event_id)
        rows = self._projections([agent_id], category, memory_projection)
        if not rows:
            return None
        row = rows[0]
        arch_id = row[1]
        p_arch = self._archetype_posteriors(event_id, bundle_version).get(arch_id)
        if p_arch is None:
            return None
        calibrated, raw, components = personalize_one(
            p_arch, row, sim_seed=sim_seed, event_id=event_id,
            bundle_version=bundle_version, return_components=True,
        )
        belief_id = self._archetype_belief_ids(event_id, bundle_version).get(arch_id)
        db.session.expunge_all()
        return {
            "agent_id": agent_id, "event_id": event_id,
            "archetype_belief_id": belief_id,
            "raw_probability": raw, "calibrated_probability": calibrated,
            "belief_status": BELIEF_STATUS_RECONSTRUCTED,
            "last_evidence_bundle_version": bundle_version,
            "personalization_components": components,
            "personalization_algo_version": PERSONALIZATION_ALGO_VERSION,
        }


__all__ = [
    "BeliefService",
    "stable_logit",
    "stable_sigmoid",
    "personalize_one",
    "personalize_batch",
]
