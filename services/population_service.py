"""PopulationService — scalable, deterministic Archetype + Agent generation.

Design goal: stand up 50–300 Archetypes and 10,000+ heterogeneous Agents
WITHOUT one LLM call per Agent, reproducibly, and without holding the
whole population in the ORM identity map.

Two tiers
---------
    Archetypes  — 50–300 rows. Numeric parameter blocks always come from
                  `agents.strategy_families` templates + per-archetype
                  seeded jitter (so bounds always hold). Descriptions may
                  optionally be LLM-enriched (one routed call PER ARCHETYPE,
                  never per Agent), with strict-JSON schema validation and
                  a deterministic-template fallback.
    Agents      — thousands+, generated in PURE CODE by sampling around an
                  archetype's parameters with an independent per-Agent RNG
                  stream. Written in bounded batches; the session is
                  expunged/expired between batches so memory stays flat.

Determinism
-----------
Everything derives from one master `seed` via blake2b (NOT Python's salted
`hash()`, and NOT global `random`). Independent streams keep concerns from
bleeding into each other:

    _rng(seed, "archetype", i)      — archetype i's jitter
    _rng(seed, "family_plan")       — which family each Agent slot gets
    _rng(seed, "activity_plan")     — which activity band each slot gets
    _rng(seed, "agent", g)          — Agent g's parameter draws
    _rng(seed, "archsel", g)        — Agent g's archetype pick within family

Same seed ⇒ identical population summary; different seed ⇒ different
Agents; same-archetype Agents still differ (their per-Agent stream differs).
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, inspect, text

from models import Agent, AgentArchetype, OBJECTIVE_MAXIMIZE_WEALTH, db
from agents.strategy_families import (
    EXPERTISE_CATEGORIES,
    PERSONA_PROMPT_VERSION,
    STRATEGY_FAMILIES,
    family_template,
)
from services._schema_cache import ensure_created as _ensure_schema_cached


# ---- configurable population mix (must sum to 1) --------------------
DEFAULT_POPULATION_MIX: Dict[str, float] = {
    "evidence_value": 0.25,
    "specialist": 0.15,
    "momentum": 0.12,
    "market_following": 0.12,
    "contrarian": 0.10,
    "retail_like": 0.10,
    "mean_reversion": 0.08,
    "adaptive": 0.08,
}

# ---- activity bands (must sum to 1). Each maps to a multiplier applied
# to the archetype's base wake-up RATE (we store rates, not intervals).
DEFAULT_ACTIVITY_DISTRIBUTION: Dict[str, float] = {
    "ultra_active": 0.02,
    "active": 0.13,
    "normal": 0.50,
    "low_frequency": 0.25,
    "very_low_frequency": 0.10,
}
_ACTIVITY_RATE_MULTIPLIER: Dict[str, float] = {
    "ultra_active": 6.0,
    "active": 2.5,
    "normal": 1.0,
    "low_frequency": 0.4,
    "very_low_frequency": 0.1,
}


# ---- bounds (every generated value is clamped into these) -----------
_BOUNDS = {
    "initial_cash": (500.0, 1_000_000.0),
    "risk_aversion": (0.05, 0.95),
    "kelly_fraction": (0.05, 1.0),
    "max_event_exposure": (0.02, 0.5),
    "max_total_exposure": (0.1, 1.0),
    "max_drawdown_tolerance": (0.1, 0.9),
    "entry_edge_threshold": (0.005, 0.5),
    "exit_edge_threshold": (0.005, 0.5),
    "reversal_edge_threshold": (0.01, 0.6),
    "minimum_trade_notional": (1.0, 1000.0),
    "base_wakeup_rate_per_day": (0.05, 100.0),
    "event_sensitivity": (0.0, 1.0),
    "price_sensitivity": (0.0, 1.0),
    "portfolio_sensitivity": (0.0, 1.0),
    "information_delay_seconds": (0.0, 86_400.0),
    "unit_interval": (0.0, 1.0),
}

# Agent columns added by the population phases (name → SQLite DDL type).
_NEW_AGENT_COLUMNS = {
    "total_trades": "INTEGER NOT NULL DEFAULT 0",
    "non_hold_trades": "INTEGER NOT NULL DEFAULT 0",
    "archetype_id": "INTEGER",
    "objective": "VARCHAR(64)",
    "random_seed": "BIGINT",
    "status": "VARCHAR(16)",
    "risk_aversion": "FLOAT",
    "kelly_fraction": "FLOAT",
    "max_event_exposure": "FLOAT",
    "max_total_exposure": "FLOAT",
    "max_drawdown_tolerance": "FLOAT",
    "entry_edge_threshold": "FLOAT",
    "exit_edge_threshold": "FLOAT",
    "reversal_edge_threshold": "FLOAT",
    "minimum_trade_notional": "FLOAT",
    "base_wakeup_rate_per_day": "FLOAT",
    "event_sensitivity": "FLOAT",
    "price_sensitivity": "FLOAT",
    "portfolio_sensitivity": "FLOAT",
    "information_delay_seconds": "FLOAT",
    "activity_group": "VARCHAR(24)",
    "persona_overrides_json": "JSON",
}

_DEFAULT_BATCH_SIZE = 1000


# ---- deterministic RNG helpers --------------------------------------

def _seed_int(*parts: Any) -> int:
    joined = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(joined, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _rng(*parts: Any) -> random.Random:
    return random.Random(_seed_int(*parts))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _perturb_map(rng: random.Random, base: Dict[str, float], sigma: float,
                 lo: float = 0.0, hi: float = 1.0) -> Dict[str, float]:
    out = {}
    for k, v in (base or {}).items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        out[k] = round(_clamp(fv + rng.gauss(0.0, sigma), lo, hi), 4)
    return out


def _allocate(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Largest-remainder integer allocation of `total` across `weights`.

    Guarantees the counts sum to exactly `total`. Every weight > 0 gets
    at least a fair share; a floor of 0 is possible for tiny weights on
    small totals (acceptable — callers ensure archetype coverage).
    """
    if total <= 0 or not weights:
        return {k: 0 for k in weights}
    raw = {k: total * w for k, w in weights.items()}
    floors = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total - sum(floors.values())
    order = sorted(weights.keys(), key=lambda k: (raw[k] - floors[k]), reverse=True)
    for k in order[: max(0, remainder)]:
        floors[k] += 1
    return floors


def validate_mix(mix: Dict[str, float], name: str = "population mix") -> Dict[str, float]:
    """Validate a distribution sums to 1 and has known keys. Returns it."""
    if not mix:
        raise ValueError(f"{name} is empty")
    total = sum(mix.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to 1.0, got {total:.6f}")
    if any(v < 0 for v in mix.values()):
        raise ValueError(f"{name} has a negative weight")
    return mix


class PopulationService:
    """Generate archetypes + agents deterministically and at scale."""

    # ==================================================================
    # Schema
    # ==================================================================

    @staticmethod
    def ensure_schema() -> None:
        """Create archetype table + add new Agent columns in place.

        Idempotent; preserves existing Agent rows (new columns nullable).
        Backfills the shared objective / status on legacy rows — metadata
        only, never cash/trades.
        """
        _ensure_schema_cached()

        inspector = inspect(db.engine)
        if "agents" not in inspector.get_table_names():
            return
        existing = {c["name"] for c in inspector.get_columns("agents")}
        missing = [(n, t) for n, t in _NEW_AGENT_COLUMNS.items() if n not in existing]
        if missing:
            with db.engine.begin() as conn:
                for name, ddl_type in missing:
                    conn.execute(text(f"ALTER TABLE agents ADD COLUMN {name} {ddl_type}"))
            print(
                f"[population] migrated agents table: added {len(missing)} column(s)",
                file=sys.stderr,
            )
        with db.engine.begin() as conn:
            conn.execute(text(
                "UPDATE agents SET objective = :obj WHERE objective IS NULL"
            ), {"obj": OBJECTIVE_MAXIMIZE_WEALTH})
            conn.execute(text(
                "UPDATE agents SET status = 'active' WHERE status IS NULL"
            ))

    # ==================================================================
    # Archetypes — deterministic (no LLM)
    # ==================================================================

    @classmethod
    def generate_default_archetypes(
        cls, count: int, seed: int = 0,
        mix: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Create `count` archetypes spread across families by `mix`.

        Pure code. Returns a metrics dict. Every family present in the
        mix gets at least one archetype (borrowing from the largest
        family if largest-remainder floored it to zero).
        """
        t0 = time.perf_counter()
        if count <= 0:
            return cls._archetype_metrics([], seed, 0.0, 0.0, 0, 0)
        cls.ensure_schema()
        mix = validate_mix(dict(mix or DEFAULT_POPULATION_MIX))

        per_family = cls._archetype_family_counts(count, mix)
        specs = cls._build_all_archetype_specs(per_family, seed)

        db_t0 = time.perf_counter()
        cls._bulk_insert_archetypes(specs)
        db_time = time.perf_counter() - db_t0

        gen_time = (time.perf_counter() - t0) - db_time
        return cls._archetype_metrics(specs, seed, gen_time, db_time, 0, 0)

    @classmethod
    def generate_llm_archetypes(
        cls, count: int, seed: int = 0,
        mix: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Create `count` archetypes, LLM-enriching each DESCRIPTION.

        Numeric parameter blocks are ALWAYS the deterministic
        template+jitter values — the LLM only proposes prose, validated
        against a strict schema, with a template fallback. One routed
        call per archetype (never per Agent). Ordinary generation →
        BALANCED; archetypes flagged complex → STRONG; batch mode is the
        router's ASYNC/MICRO batch. Falls back entirely to deterministic
        templates if no LLM is configured.
        """
        t0 = time.perf_counter()
        if count <= 0:
            return cls._archetype_metrics([], seed, 0.0, 0.0, 0, 0)
        cls.ensure_schema()
        mix = validate_mix(dict(mix or DEFAULT_POPULATION_MIX))

        per_family = cls._archetype_family_counts(count, mix)
        specs = cls._build_all_archetype_specs(per_family, seed)

        router, client = cls._llm_handles()
        llm_calls = 0
        llm_fallbacks = 0
        if router is not None and client is not None:
            for spec in specs:
                rng = _rng(seed, "archetype_llm", spec["name"])
                made_call, ok = cls._enrich_archetype_description(spec, router, client, rng)
                llm_calls += 1 if made_call else 0
                llm_fallbacks += 0 if ok else 1
        else:
            llm_fallbacks = len(specs)  # no LLM ⇒ all deterministic

        db_t0 = time.perf_counter()
        cls._bulk_insert_archetypes(specs)
        db_time = time.perf_counter() - db_t0

        gen_time = (time.perf_counter() - t0) - db_time
        return cls._archetype_metrics(specs, seed, gen_time, db_time, llm_calls, llm_fallbacks)

    # ---- archetype internals -----------------------------------------

    @staticmethod
    def _archetype_family_counts(count: int, mix: Dict[str, float]) -> Dict[str, int]:
        per_family = _allocate(count, mix)
        # Guarantee ≥1 archetype for every family in the mix so every
        # Agent family has an archetype to sample from.
        deficit = [f for f, n in per_family.items() if n == 0 and mix.get(f, 0) > 0]
        for f in deficit:
            donor = max(per_family, key=lambda k: per_family[k])
            if per_family[donor] > 1:
                per_family[donor] -= 1
                per_family[f] += 1
        return per_family

    @classmethod
    def _build_all_archetype_specs(
        cls, per_family: Dict[str, int], seed: int
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        global_i = 0
        for family in STRATEGY_FAMILIES:
            n = per_family.get(family, 0)
            for k in range(n):
                rng = _rng(seed, "archetype", global_i)
                specs.append(cls._build_archetype_spec(family, global_i, k, rng))
                global_i += 1
        return specs

    @classmethod
    def _build_archetype_spec(
        cls, family: str, global_index: int, family_k: int, rng: random.Random
    ) -> Dict[str, Any]:
        tmpl = family_template(family)

        expertise = _perturb_map(rng, tmpl["expertise_json"], sigma=0.08)
        if family == "specialist":
            focus = EXPERTISE_CATEGORIES[family_k % len(EXPERTISE_CATEGORIES)]
            for cat in expertise:
                expertise[cat] = round(_clamp(0.2 + rng.gauss(0, 0.05), 0.0, 0.4), 4)
            expertise[focus] = round(_clamp(0.9 + rng.gauss(0, 0.05), 0.6, 1.0), 4)

        biases = _perturb_map(rng, tmpl["cognitive_biases_json"], sigma=0.06)
        trust = _perturb_map(rng, tmpl["source_trust_json"], sigma=0.06)

        att = dict(tmpl["attention_profile_json"])
        att_out = {
            "event": round(_clamp(att.get("event", 0.5) + rng.gauss(0, 0.06), 0, 1), 4),
            "price": round(_clamp(att.get("price", 0.5) + rng.gauss(0, 0.06), 0, 1), 4),
            "portfolio": round(_clamp(att.get("portfolio", 0.5) + rng.gauss(0, 0.06), 0, 1), 4),
            "base_rate": round(_clamp(att.get("base_rate", 4.0) * rng.uniform(0.8, 1.2),
                                      *_BOUNDS["base_wakeup_rate_per_day"]), 3),
        }

        r = dict(tmpl["risk_profile_json"])
        risk_out = {
            "risk_aversion": round(_clamp(r["risk_aversion"] + rng.gauss(0, 0.05), *_BOUNDS["risk_aversion"]), 4),
            "kelly_fraction": round(_clamp(r["kelly_fraction"] + rng.gauss(0, 0.05), *_BOUNDS["kelly_fraction"]), 4),
            "max_event_exposure": round(_clamp(r["max_event_exposure"] + rng.gauss(0, 0.02), *_BOUNDS["max_event_exposure"]), 4),
            "max_total_exposure": round(_clamp(r["max_total_exposure"] + rng.gauss(0, 0.05), *_BOUNDS["max_total_exposure"]), 4),
            "max_drawdown_tolerance": round(_clamp(r["max_drawdown_tolerance"] + rng.gauss(0, 0.04), *_BOUNDS["max_drawdown_tolerance"]), 4),
        }

        sp = dict(tmpl["strategy_parameters_json"])
        sp_out = {
            "entry_edge_threshold": round(_clamp(sp["entry_edge_threshold"] + rng.gauss(0, 0.01), *_BOUNDS["entry_edge_threshold"]), 4),
            "exit_edge_threshold": round(_clamp(sp["exit_edge_threshold"] + rng.gauss(0, 0.01), *_BOUNDS["exit_edge_threshold"]), 4),
            "reversal_edge_threshold": round(_clamp(sp["reversal_edge_threshold"] + rng.gauss(0, 0.015), *_BOUNDS["reversal_edge_threshold"]), 4),
            "minimum_trade_notional": round(_clamp(sp["minimum_trade_notional"] * rng.uniform(0.8, 1.2), *_BOUNDS["minimum_trade_notional"]), 2),
        }

        return {
            "name": f"{family}-{global_index:04d}",
            "description": tmpl["description"],
            "strategy_type": family,
            "objective": OBJECTIVE_MAXIMIZE_WEALTH,
            "expertise_json": expertise,
            "cognitive_biases_json": biases,
            "source_trust_json": trust,
            "attention_profile_json": att_out,
            "risk_profile_json": risk_out,
            "strategy_parameters_json": sp_out,
            "persona_prompt_version": PERSONA_PROMPT_VERSION,
        }

    @staticmethod
    def _bulk_insert_archetypes(specs: List[Dict[str, Any]]) -> None:
        if not specs:
            return
        db.session.bulk_insert_mappings(AgentArchetype, specs)
        db.session.commit()
        db.session.expunge_all()

    @staticmethod
    def _archetype_metrics(specs, seed, gen_time, db_time, llm_calls, llm_fallbacks):
        from collections import Counter
        dist = Counter(s["strategy_type"] for s in specs)
        return {
            "archetypes": len(specs),
            "seed": seed,
            "generation_time_s": round(gen_time, 4),
            "db_time_s": round(db_time, 4),
            "llm_request_count": llm_calls,
            "llm_fallback_count": llm_fallbacks,
            "strategy_distribution": dict(dist),
        }

    # ---- optional LLM enrichment -------------------------------------

    @staticmethod
    def _llm_handles():
        try:
            from llm import get_model_router, get_llm_client
            client = get_llm_client()
            if not client.available:
                print(
                    "[population] use_llm requested but no LLM configured; "
                    "archetypes fall back to deterministic templates.",
                    file=sys.stderr,
                )
                return None, None
            return get_model_router(), client
        except Exception as exc:  # noqa: BLE001
            print(f"[population] LLM handles unavailable: {exc}", file=sys.stderr)
            return None, None

    @staticmethod
    def _valid_llm_archetype(payload: Any) -> bool:
        """Strict schema check for an LLM archetype-description response."""
        if not isinstance(payload, dict):
            return False
        desc = payload.get("description")
        if not isinstance(desc, str) or not desc.strip():
            return False
        if len(desc) > 800:
            return False
        return True

    @classmethod
    def _enrich_archetype_description(
        cls, spec, router, client, rng
    ) -> Tuple[bool, bool]:
        """Route one call and fold a validated description in.

        Returns (made_call, ok). Never raises; never touches numeric
        blocks. On any routing/parse/schema failure, keeps the template
        description and reports ok=False.
        """
        from llm import TaskRoutingContext, TaskType

        complex_flag = rng.random() < 1.0 / 6.0
        ctx = TaskRoutingContext(
            task_type=TaskType.PERSONA_ARCHETYPE_GENERATION,
            estimated_input_tokens=350,
            expected_output_tokens=180,
            structured_output_required=True,
            batch_eligible=True,
            task_metadata={"complex": complex_flag},
        )
        try:
            decision = router.route(ctx)
        except Exception as exc:  # noqa: BLE001
            print(f"[population] router failed: {exc}", file=sys.stderr)
            return False, False
        if not decision.budget_allowed and not decision.cache_eligible:
            return False, False  # degraded/exhausted → keep template

        messages = [
            {"role": "system", "content": (
                "Write a one-sentence trader-persona description for a "
                "prediction-market simulation. STRICT JSON only: "
                '{"description":"<one sentence>"}. No prose, no CoT.'
            )},
            {"role": "user", "content": (
                f"Strategy family: {spec['strategy_type']}. "
                f"Objective: {spec['objective']}. "
                f"Base: {spec['description']}"
            )},
        ]
        try:
            payload = client.chat_json(messages, max_tokens=200)
        except Exception as exc:  # noqa: BLE001
            print(f"[population] archetype enrich failed: {exc}", file=sys.stderr)
            return True, False
        if not cls._valid_llm_archetype(payload):
            return True, False
        spec["description"] = payload["description"].strip()[:500]
        return True, True

    @classmethod
    def validate_archetypes(cls) -> Dict[str, Any]:
        """Validate every archetype: objective, family, blocks, bounds."""
        report = {"ok": True, "errors": [], "warnings": [], "stats": {}}
        archetypes = AgentArchetype.query.all()
        report["stats"]["archetypes"] = len(archetypes)
        if not archetypes:
            report["ok"] = False
            report["errors"].append("no archetypes exist")
            return report

        for a in archetypes:
            if a.objective != OBJECTIVE_MAXIMIZE_WEALTH:
                report["ok"] = False
                report["errors"].append(f"archetype {a.id} wrong objective {a.objective!r}")
            if a.strategy_type not in STRATEGY_FAMILIES:
                report["ok"] = False
                report["errors"].append(f"archetype {a.id} unknown family {a.strategy_type!r}")
            for block in ("expertise_json", "cognitive_biases_json", "source_trust_json",
                          "attention_profile_json", "risk_profile_json",
                          "strategy_parameters_json"):
                if not getattr(a, block):
                    report["ok"] = False
                    report["errors"].append(f"archetype {a.id} missing {block}")
            if not a.persona_prompt_version:
                report["warnings"].append(f"archetype {a.id} has no persona_prompt_version")

        from collections import Counter
        report["stats"]["by_family"] = dict(Counter(a.strategy_type for a in archetypes))
        return report

    # ==================================================================
    # Agents — pure code, bulk, batched
    # ==================================================================

    @classmethod
    def generate_agents(
        cls, count: int, seed: int = 0, batch_size: int = _DEFAULT_BATCH_SIZE,
        cash_base: float = 10_000.0,
        mix: Optional[Dict[str, float]] = None,
        activity: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Bulk-generate `count` Agents in pure code around the archetypes.

        Agents are assigned families per `mix` and activity bands per
        `activity` (both largest-remainder allocated, then shuffled with
        independent RNG streams). Written in batches of `batch_size`;
        the session is expunged/expired after each batch so the ORM
        identity map never holds the whole population. Returns metrics.
        """
        t0 = time.perf_counter()
        if count <= 0:
            return cls._agent_metrics(0, seed, 0.0, 0.0, 0, batch_size, {}, {})
        cls.ensure_schema()
        mix = validate_mix(dict(mix or DEFAULT_POPULATION_MIX))
        activity = validate_mix(dict(activity or DEFAULT_ACTIVITY_DISTRIBUTION),
                                name="activity distribution")
        batch_size = max(1, int(batch_size))

        # Load archetypes ONCE, grouped by family into plain dicts, then
        # expunge so no archetype ORM objects linger (no per-Agent query).
        arch_by_family = cls._load_archetypes_by_family()
        db.session.expunge_all()
        missing_fam = [f for f, w in mix.items() if w > 0 and not arch_by_family.get(f)]
        if missing_fam:
            raise ValueError(
                f"no archetypes for families {missing_fam} — run "
                f"generate-archetypes first"
            )

        # Plan family + activity per slot (cheap string lists), shuffled
        # with independent deterministic streams so bands/families aren't
        # clustered by insertion order.
        family_plan = cls._build_plan(count, mix, _rng(seed, "family_plan"))
        activity_plan = cls._build_plan(count, activity, _rng(seed, "activity_plan"))

        existing_pop = Agent.query.filter(Agent.name.like("pop-%")).count()

        from collections import Counter
        fam_realized: Counter = Counter()
        act_realized: Counter = Counter()

        buffer: List[Dict[str, Any]] = []
        db_time = 0.0
        batch_count = 0
        created = 0

        for j in range(count):
            gi = existing_pop + j
            family = family_plan[j]
            act = activity_plan[j]
            fam_realized[family] += 1
            act_realized[act] += 1

            archs = arch_by_family[family]
            sel_rng = _rng(seed, "archsel", gi)
            arch = archs[sel_rng.randrange(len(archs))]

            agent_rng = _rng(seed, "agent", gi)
            buffer.append(cls._build_agent_mapping(arch, gi, seed, agent_rng, cash_base, act))
            created += 1

            if len(buffer) >= batch_size:
                db_time += cls._flush_batch(buffer)
                batch_count += 1

        if buffer:
            db_time += cls._flush_batch(buffer)
            batch_count += 1

        gen_time = (time.perf_counter() - t0) - db_time
        return cls._agent_metrics(
            created, seed, gen_time, db_time, batch_count, batch_size,
            dict(fam_realized), dict(act_realized),
        )

    @staticmethod
    def _load_archetypes_by_family() -> Dict[str, List[Dict[str, Any]]]:
        """Group archetypes by family as plain dicts (id + param blocks)."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        for a in AgentArchetype.query.order_by(AgentArchetype.id.asc()).all():
            out.setdefault(a.strategy_type, []).append({
                "id": a.id,
                "strategy_type": a.strategy_type,
                "expertise_json": a.expertise_json or {},
                "cognitive_biases_json": a.cognitive_biases_json or {},
                "source_trust_json": a.source_trust_json or {},
                "attention_profile_json": a.attention_profile_json or {},
                "risk_profile_json": a.risk_profile_json or {},
                "strategy_parameters_json": a.strategy_parameters_json or {},
            })
        return out

    @staticmethod
    def _build_plan(count: int, weights: Dict[str, float], rng: random.Random) -> List[str]:
        """A shuffled length-`count` list of keys allocated by `weights`."""
        alloc = _allocate(count, weights)
        plan: List[str] = []
        for key, n in alloc.items():
            plan.extend([key] * n)
        # Pad/trim to exactly count (allocation already sums to count, but
        # guard against float edge cases).
        if len(plan) < count:
            filler = max(weights, key=lambda k: weights[k])
            plan.extend([filler] * (count - len(plan)))
        elif len(plan) > count:
            plan = plan[:count]
        rng.shuffle(plan)
        return plan

    @staticmethod
    def _flush_batch(buffer: List[Dict[str, Any]]) -> float:
        """Insert one batch, commit, and clear session state. Returns db time."""
        t0 = time.perf_counter()
        db.session.bulk_insert_mappings(Agent, buffer)
        db.session.commit()
        # Drop identity-map + pending state so memory stays flat batch to
        # batch (bulk_insert_mappings doesn't populate the map, but expire
        # any loaded state to be safe).
        db.session.expunge_all()
        db.session.expire_all()
        buffer.clear()
        return time.perf_counter() - t0

    @classmethod
    def _build_agent_mapping(
        cls, arch: Dict[str, Any], index: int, seed: int,
        rng: random.Random, cash_base: float, activity_group: str,
    ) -> Dict[str, Any]:
        risk = arch["risk_profile_json"]
        sp = arch["strategy_parameters_json"]
        att = arch["attention_profile_json"]

        cash = cash_base * (math.e ** rng.gauss(0.0, 0.4))
        cash = round(_clamp(cash, *_BOUNDS["initial_cash"]), 2)

        risk_aversion = _clamp(risk.get("risk_aversion", 0.5) + rng.gauss(0, 0.06), *_BOUNDS["risk_aversion"])
        kelly = _clamp(risk.get("kelly_fraction", 0.4) + rng.gauss(0, 0.06), *_BOUNDS["kelly_fraction"])
        max_event = _clamp(risk.get("max_event_exposure", 0.15) + rng.gauss(0, 0.02), *_BOUNDS["max_event_exposure"])
        max_total = _clamp(risk.get("max_total_exposure", 0.6) + rng.gauss(0, 0.05), *_BOUNDS["max_total_exposure"])
        max_event = min(max_event, max_total)
        max_dd = _clamp(risk.get("max_drawdown_tolerance", 0.35) + rng.gauss(0, 0.04), *_BOUNDS["max_drawdown_tolerance"])

        entry = _clamp(sp.get("entry_edge_threshold", 0.07) + rng.gauss(0, 0.012), *_BOUNDS["entry_edge_threshold"])
        exit_ = _clamp(sp.get("exit_edge_threshold", 0.03) + rng.gauss(0, 0.008), *_BOUNDS["exit_edge_threshold"])
        reversal = _clamp(sp.get("reversal_edge_threshold", 0.13) + rng.gauss(0, 0.015), *_BOUNDS["reversal_edge_threshold"])
        min_notional = _clamp(sp.get("minimum_trade_notional", 20.0) * rng.uniform(0.8, 1.25), *_BOUNDS["minimum_trade_notional"])

        # Base rate from archetype, then scaled by the activity band's
        # multiplier (with a little jitter). We store the RATE + the band.
        base_rate = att.get("base_rate", 4.0)
        mult = _ACTIVITY_RATE_MULTIPLIER.get(activity_group, 1.0)
        base_wake = _clamp(base_rate * mult * rng.uniform(0.85, 1.15),
                           *_BOUNDS["base_wakeup_rate_per_day"])

        event_sens = _clamp(att.get("event", 0.6) + rng.gauss(0, 0.06), *_BOUNDS["event_sensitivity"])
        price_sens = _clamp(att.get("price", 0.6) + rng.gauss(0, 0.06), *_BOUNDS["price_sensitivity"])
        port_sens = _clamp(att.get("portfolio", 0.5) + rng.gauss(0, 0.06), *_BOUNDS["portfolio_sensitivity"])
        delay_base = 600.0 if arch["strategy_type"] == "retail_like" else 120.0
        info_delay = _clamp(delay_base * rng.uniform(0.3, 2.5), *_BOUNDS["information_delay_seconds"])

        persona_overrides = {
            "expertise": _perturb_map(rng, arch["expertise_json"], sigma=0.05),
            "biases": _perturb_map(rng, arch["cognitive_biases_json"], sigma=0.05),
            "source_trust": _perturb_map(rng, arch["source_trust_json"], sigma=0.05),
        }

        return {
            "name": f"pop-{index:07d}",
            "strategy_type": arch["strategy_type"],
            "archetype_id": arch["id"],
            "objective": OBJECTIVE_MAXIMIZE_WEALTH,
            "random_seed": _seed_int(seed, "agent", index),
            "status": "active",
            "virtual_cash": cash,
            "initial_cash": cash,
            "risk_profile": _coarse_risk(risk_aversion),
            "risk_aversion": round(risk_aversion, 4),
            "kelly_fraction": round(kelly, 4),
            "max_event_exposure": round(max_event, 4),
            "max_total_exposure": round(max_total, 4),
            "max_drawdown_tolerance": round(max_dd, 4),
            "entry_edge_threshold": round(entry, 4),
            "exit_edge_threshold": round(exit_, 4),
            "reversal_edge_threshold": round(reversal, 4),
            "minimum_trade_notional": round(min_notional, 2),
            "base_wakeup_rate_per_day": round(base_wake, 3),
            "event_sensitivity": round(event_sens, 4),
            "price_sensitivity": round(price_sens, 4),
            "portfolio_sensitivity": round(port_sens, 4),
            "information_delay_seconds": round(info_delay, 1),
            "activity_group": activity_group,
            "persona_overrides_json": persona_overrides,
            "created_at": datetime.utcnow(),
        }

    @staticmethod
    def _agent_metrics(created, seed, gen_time, db_time, batch_count, batch_size,
                       fam_realized, act_realized):
        return {
            "agents": created,
            "seed": seed,
            "generation_time_s": round(gen_time, 4),
            "db_time_s": round(db_time, 4),
            "batch_count": batch_count,
            "batch_size": batch_size,
            "llm_request_count": 0,  # invariant: agents never call the LLM
            "strategy_distribution": fam_realized,
            "activity_distribution": act_realized,
        }

    # ==================================================================
    # Population validation
    # ==================================================================

    @classmethod
    def validate_population(cls) -> Dict[str, Any]:
        """Aggregate-query validation. No per-Agent joins / N+1."""
        report = {"ok": True, "errors": [], "warnings": [], "stats": {}}

        report["stats"]["archetypes"] = AgentArchetype.query.count()
        report["stats"]["agents"] = Agent.query.count()

        wrong_obj = Agent.query.filter(
            (Agent.objective.is_(None)) | (Agent.objective != OBJECTIVE_MAXIMIZE_WEALTH)
        ).count()
        if wrong_obj:
            report["ok"] = False
            report["errors"].append(f"{wrong_obj} agent(s) missing wealth objective")

        orphan = Agent.query.filter(
            Agent.archetype_id.isnot(None),
            ~Agent.archetype_id.in_(db.session.query(AgentArchetype.id)),
        ).count()
        if orphan:
            report["ok"] = False
            report["errors"].append(f"{orphan} agent(s) point at a missing archetype")

        legacy = Agent.query.filter(Agent.archetype_id.is_(None)).count()
        if legacy:
            report["warnings"].append(f"{legacy} legacy agent(s) have no archetype (preserved)")

        # Bounds — single streaming scan over generated agents only.
        checks = {k: _BOUNDS[k] for k in (
            "initial_cash", "risk_aversion", "kelly_fraction", "max_event_exposure",
            "max_total_exposure", "max_drawdown_tolerance", "entry_edge_threshold",
            "exit_edge_threshold", "reversal_edge_threshold", "minimum_trade_notional",
            "base_wakeup_rate_per_day", "event_sensitivity", "price_sensitivity",
            "portfolio_sensitivity", "information_delay_seconds",
        )}
        out_of_bounds = 0
        for row in Agent.query.filter(Agent.archetype_id.isnot(None)).yield_per(2000):
            for field, (lo, hi) in checks.items():
                val = getattr(row, field, None)
                if val is None:
                    continue
                if val < lo - 1e-9 or val > hi + 1e-9:
                    out_of_bounds += 1
                    if len(report["errors"]) < 20:
                        report["errors"].append(f"agent {row.id} {field}={val} out of [{lo},{hi}]")
                    break
        if out_of_bounds:
            report["ok"] = False
            report["stats"]["out_of_bounds_agents"] = out_of_bounds

        # Distributions via GROUP BY (no N+1).
        fam_rows = (
            db.session.query(Agent.strategy_type, func.count(Agent.id))
            .filter(Agent.archetype_id.isnot(None))
            .group_by(Agent.strategy_type).all()
        )
        act_rows = (
            db.session.query(Agent.activity_group, func.count(Agent.id))
            .filter(Agent.archetype_id.isnot(None))
            .group_by(Agent.activity_group).all()
        )
        report["stats"]["strategy_distribution"] = {k: v for k, v in fam_rows}
        report["stats"]["activity_distribution"] = {k: v for k, v in act_rows}

        arch_fam = (
            db.session.query(AgentArchetype.strategy_type, func.count(AgentArchetype.id))
            .group_by(AgentArchetype.strategy_type).all()
        )
        report["stats"]["archetypes_by_family"] = {k: v for k, v in arch_fam}
        unknown = [k for k, _ in arch_fam if k not in STRATEGY_FAMILIES]
        if unknown:
            report["ok"] = False
            report["errors"].append(f"archetypes with unknown family: {unknown}")

        return report


def _coarse_risk(risk_aversion: float) -> str:
    if risk_aversion >= 0.6:
        return "low"
    if risk_aversion <= 0.35:
        return "high"
    return "medium"


__all__ = [
    "PopulationService",
    "DEFAULT_POPULATION_MIX",
    "DEFAULT_ACTIVITY_DISTRIBUTION",
    "validate_mix",
]
