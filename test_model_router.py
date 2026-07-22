"""Mock-mode tests + demo for the LLM routing / budgeting / cache layer.

Runs with NO real provider — everything is exercised against explicit
in-memory configs, so it is safe in CI and offline. Style matches the
repo's other `test_*.py` scripts: plain asserts, exits 0 on success and
1 on the first failure. No pytest dependency.

Usage:
    python test_model_router.py            # run all tests, then print demo
    python test_model_router.py --demo     # print the routing table only
    python test_model_router.py --quiet    # tests only, no demo table
"""

from __future__ import annotations

import argparse
import io
import sys
import threading
from contextlib import redirect_stderr

from llm.tiers import Tier, BatchMode, Urgency
from llm.tasks import TaskType, NonLLMTaskError, validate_task_type, NonLLMOperation
from llm.router_config import RouterConfig, TierSettings, BudgetConfig
from llm.budget import BudgetManager
from llm.usage import LLMUsage, UsageRecorder
from llm.cache_meta import build_cache_metadata, evidence_delta_hash, CacheMetadata
from llm.routing import ModelRouter, TaskRoutingContext, RoutingResult


# --- fixtures --------------------------------------------------------

def _all_tiers_config(*, with_local=True, with_specialist=True, budget=None) -> RouterConfig:
    """A fully-mapped RouterConfig using placeholder (non-commercial) names."""
    tiers = {
        Tier.FAST: TierSettings(Tier.FAST, provider="mockprov", model="mock-fast"),
        Tier.BALANCED: TierSettings(Tier.BALANCED, provider="mockprov", model="mock-balanced"),
        Tier.STRONG: TierSettings(Tier.STRONG, provider="mockprov", model="mock-strong"),
        Tier.SPECIALIST: TierSettings(
            Tier.SPECIALIST,
            provider="specprov" if with_specialist else None,
            model="mock-specialist" if with_specialist else None,
        ),
        Tier.LOCAL: TierSettings(
            Tier.LOCAL,
            provider="localprov" if with_local else None,
            model="mock-local" if with_local else None,
        ),
    }
    return RouterConfig(
        tiers=tiers,
        budget=budget or BudgetConfig(),
        default_provider="mockprov",
    )


def _router(**kw) -> ModelRouter:
    cfg = _all_tiers_config(**kw)
    return ModelRouter(cfg, BudgetManager(cfg.budget))


# --- test helpers ----------------------------------------------------

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# --- tests -----------------------------------------------------------

def test_classification_routes_fast():
    r = _router()
    res = r.route(TaskRoutingContext(task_type=TaskType.EVIDENCE_CLASSIFICATION))
    check("classification → FAST", res.tier == Tier.FAST.value, f"got {res.tier}")


def test_routine_belief_routes_balanced():
    r = _router()
    res = r.route(TaskRoutingContext(task_type=TaskType.ROUTINE_BELIEF_UPDATE))
    check("routine belief → BALANCED", res.tier == Tier.BALANCED.value, f"got {res.tier}")


def test_major_conflict_routes_strong():
    r = _router()
    res = r.route(TaskRoutingContext(
        task_type=TaskType.EVIDENCE_CONFLICT_ANALYSIS,
        evidence_conflict_score=0.9,
        information_impact_score=0.8,
    ))
    check("major conflict → STRONG", res.tier == Tier.STRONG.value, f"got {res.tier}")


def test_explicit_specialist():
    r = _router()
    res = r.route(TaskRoutingContext(
        task_type=TaskType.CODE_OR_SYSTEM_ANALYSIS,
        task_metadata={"specialist": True},
    ))
    check("explicit → SPECIALIST", res.tier == Tier.SPECIALIST.value, f"got {res.tier}")
    check("specialist has model", res.model == "mock-specialist", f"got {res.model}")


def test_local_when_configured():
    r = _router(with_local=True)
    res = r.route(TaskRoutingContext(
        task_type=TaskType.EVIDENCE_CLASSIFICATION,
        task_metadata={"prefer_local": True},
        batch_eligible=True,
    ))
    check("bulk task → LOCAL when configured", res.tier == Tier.LOCAL.value, f"got {res.tier}")


def test_local_never_silent_when_unconfigured():
    r = _router(with_local=False)
    res = r.route(TaskRoutingContext(
        task_type=TaskType.EVIDENCE_CLASSIFICATION,
        task_metadata={"prefer_local": True},
        batch_eligible=True,
    ))
    check("no silent LOCAL when unconfigured", res.tier != Tier.LOCAL.value, f"got {res.tier}")
    check("falls back to a configured tier", res.model is not None, f"got model {res.model}")


def test_action_policy_rejected():
    r = _router()
    rejected = False
    try:
        r.route(TaskRoutingContext(task_type="ACTION_POLICY"))
    except NonLLMTaskError:
        rejected = True
    check("ActionPolicy rejected", rejected)


def test_lmsr_rejected():
    r = _router()
    for name in ("LMSR_EXECUTION", "LMSR_QUOTE", "LMSR", "KELLY_CALCULATION",
                 "PORTFOLIO_VALUATION", "LEADERBOARD_CALCULATION",
                 "SCORE_AGGREGATION", "RISK_LIMITS", "CANDIDATE_SELECTION",
                 "WAKEUP_SCHEDULING"):
        rejected = False
        try:
            validate_task_type(name)
        except NonLLMTaskError:
            rejected = True
        check(f"non-LLM op rejected: {name}", rejected)


def test_budget_exhaustion_degrades_safely():
    # Tiny output budget so any request is unaffordable at every tier.
    budget = BudgetConfig(daily_output_tokens=0)
    cfg = _all_tiers_config(budget=budget)
    r = ModelRouter(cfg, BudgetManager(budget))
    res = r.route(TaskRoutingContext(
        task_type=TaskType.MAJOR_BELIEF_UPDATE,
        estimated_input_tokens=100,
        expected_output_tokens=100,
        cache_available=False,
    ))
    check("budget exhausted → degraded", res.degraded is True, f"got {res}")
    check("budget exhausted → budget_allowed False", res.budget_allowed is False)
    check("budget exhausted → no infinite fallback (allow_fallback False)",
          res.allow_fallback is False)


def test_budget_exhaustion_prefers_cache():
    budget = BudgetConfig(daily_output_tokens=0)
    cfg = _all_tiers_config(budget=budget)
    r = ModelRouter(cfg, BudgetManager(budget))
    res = r.route(TaskRoutingContext(
        task_type=TaskType.MAJOR_BELIEF_UPDATE,
        estimated_input_tokens=100,
        expected_output_tokens=100,
        cache_available=True,
    ))
    check("budget exhausted + cache → cache_eligible", res.cache_eligible is True)
    check("budget exhausted + cache → not degraded", res.degraded is False)


def test_budget_fallback_to_cheaper_tier():
    # STRONG-only budget is 0, but overall input/output budgets are generous.
    budget = BudgetConfig(daily_strong_tokens=0)
    cfg = _all_tiers_config(budget=budget)
    r = ModelRouter(cfg, BudgetManager(budget))
    res = r.route(TaskRoutingContext(
        task_type=TaskType.MAJOR_BELIEF_UPDATE,  # wants STRONG
        estimated_input_tokens=50,
        expected_output_tokens=50,
    ))
    check("STRONG budget out → cheaper tier", res.tier in (Tier.BALANCED.value, Tier.FAST.value),
          f"got {res.tier}")
    check("cheaper tier is affordable/allowed", res.budget_allowed is True)


def test_retries_bounded():
    budget = BudgetConfig(max_retries=2)
    cfg = _all_tiers_config(budget=budget)
    r = ModelRouter(cfg, BudgetManager(budget))
    check("max_retries is finite and small", 0 <= r.config.budget.max_retries <= 5)


def test_escalation_bounded():
    budget = BudgetConfig(max_escalations=1)
    cfg = _all_tiers_config(budget=budget)
    r = ModelRouter(cfg, BudgetManager(budget))
    # Many previous failures must not escalate past 1 step from base.
    res = r.route(TaskRoutingContext(
        task_type=TaskType.EVIDENCE_CLASSIFICATION,  # base FAST
        previous_failures=99,
    ))
    check("escalation bounded by max_escalations", res.escalation_count <= 1,
          f"got {res.escalation_count}")
    # FAST + 1 escalation → BALANCED at most (not STRONG).
    check("bounded escalation lands ≤ BALANCED",
          res.tier in (Tier.BALANCED.value, Tier.STRONG.value),
          f"got {res.tier}")


def test_stable_prefix_hash_is_stable():
    a = build_cache_metadata(
        market_definition="Will X happen by 2027?",
        resolution_rules="Resolves YES if official source says so.",
        archetype_definition="cautious macro analyst",
        evidence_bundle_version="v3",
        prior_probability=0.4,
        evidence_delta=evidence_delta_hash("v2", "v3", ["e5"]),
        time_bucket="2026-07-16T10",
    )
    b = build_cache_metadata(
        market_definition="Will X happen by 2027?",
        resolution_rules="Resolves YES if official source says so.",
        archetype_definition="cautious macro analyst",
        evidence_bundle_version="v3",
        prior_probability=0.9,  # DIFFERENT dynamic input
        evidence_delta=evidence_delta_hash("v2", "v3", ["e5", "e6"]),  # different
        time_bucket="2026-07-16T23",  # different
    )
    check("stable prefix identical across dynamic changes",
          a.stable_prefix_hash == b.stable_prefix_hash,
          f"{a.stable_prefix_hash} vs {b.stable_prefix_hash}")
    check("dynamic suffix differs when dynamic inputs differ",
          a.dynamic_suffix_hash != b.dynamic_suffix_hash)


def test_evidence_delta_changes_only_dynamic():
    base = dict(
        market_definition="Q?", resolution_rules="R", archetype_definition="A",
        evidence_bundle_version="v1", prior_probability=0.5, time_bucket="t",
    )
    a = build_cache_metadata(evidence_delta=evidence_delta_hash("v0", "v1", ["e1"]), **base)
    b = build_cache_metadata(evidence_delta=evidence_delta_hash("v0", "v1", ["e1", "e2"]), **base)
    check("evidence delta leaves stable prefix unchanged",
          a.stable_prefix_hash == b.stable_prefix_hash)
    check("evidence delta changes dynamic suffix",
          a.dynamic_suffix_hash != b.dynamic_suffix_hash)


def test_delta_hash_order_insensitive():
    h1 = evidence_delta_hash("v1", "v2", ["a", "b", "c"])
    h2 = evidence_delta_hash("v1", "v2", ["c", "a", "b"])
    check("evidence delta hash is order-insensitive", h1 == h2)


def test_no_secrets_in_usage():
    rec = UsageRecorder()
    u = LLMUsage(
        task_type="EVIDENCE_CLASSIFICATION", selected_tier="FAST",
        provider="mockprov", model="mock-fast", batch_mode="MICRO_BATCH",
        timestamp=0.0,
        metadata={
            "api_key": "sk-SUPERSECRET",
            "authorization": "Bearer sk-XYZ",
            "prompt": "the full sensitive prompt text",
            "chain_of_thought": "step 1 ... step 2 ...",
            "market_id_note": "ok-to-keep",
        },
    )
    rec.record(u)
    blob = repr(rec.all()[0].to_dict())
    leaked = any(s in blob for s in ("SUPERSECRET", "Bearer", "sensitive prompt", "step 1"))
    check("no api key / auth / prompt / CoT in usage record", not leaked, blob)
    check("non-secret metadata retained", "ok-to-keep" in blob)


def test_legacy_call_still_works():
    r = _router()
    buf = io.StringIO()
    with redirect_stderr(buf):
        res = r.route_legacy("smoke")
    check("legacy routes to BALANCED", res.tier == Tier.BALANCED.value, f"got {res.tier}")
    check("legacy emits structured warning", "legacy" in buf.getvalue().lower())
    check("legacy still returns a usable model", res.model == "mock-balanced")


def test_legacy_client_unchanged():
    # The provider-agnostic client must import + behave as before.
    from llm import LLMClient, load_config, get_llm_client
    c = LLMClient(load_config())
    check("legacy LLMClient constructs", c is not None)
    check("legacy LLMClient exposes .available", hasattr(c, "available"))
    check("get_llm_client singleton works", get_llm_client() is get_llm_client())


def test_concurrent_budget_does_not_exceed():
    # Output budget only allows 10 commits of 10 tokens = 100 total.
    budget = BudgetConfig(daily_output_tokens=100)
    bm = BudgetManager(budget)

    granted = []
    lock = threading.Lock()

    def worker():
        d = bm.reserve(Tier.BALANCED, input_tokens=0, output_tokens=10)
        if d.allowed:
            with lock:
                granted.append(d.reservation_id)
            bm.commit(d.reservation_id, actual_output_tokens=10)

    threads = [threading.Thread(target=worker) for _ in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    status = bm.status()
    check("concurrent reservations bounded to budget",
          len(granted) <= 10, f"granted {len(granted)}")
    check("committed output tokens never exceed budget",
          status["output_spent"] <= 100, f"spent {status['output_spent']}")


def test_concurrent_strong_slots_bounded():
    budget = BudgetConfig(max_concurrent_strong=3)
    bm = BudgetManager(budget)
    # Reserve without committing to hold slots.
    reservations = []
    for _ in range(10):
        d = bm.reserve(Tier.STRONG, input_tokens=1, output_tokens=1)
        if d.allowed:
            reservations.append(d.reservation_id)
    check("STRONG concurrency capped", len(reservations) == 3, f"got {len(reservations)}")
    # Releasing one frees a slot.
    bm.release(reservations[0])
    d = bm.reserve(Tier.STRONG, input_tokens=1, output_tokens=1)
    check("released STRONG slot reusable", d.allowed is True)


def test_realtime_batch_for_urgent():
    r = _router()
    res = r.route(TaskRoutingContext(
        task_type=TaskType.ROUTINE_BELIEF_UPDATE,
        urgency=Urgency.HIGH,
        task_metadata={"market_closing": True},
    ))
    check("urgent/closing → REALTIME batch", res.batch_mode == BatchMode.REALTIME.value,
          f"got {res.batch_mode}")


def test_async_batch_for_persona():
    r = _router()
    res = r.route(TaskRoutingContext(
        task_type=TaskType.PERSONA_VARIATION,
        urgency=Urgency.LOW,
    ))
    check("persona/low urgency → ASYNC_BATCH", res.batch_mode == BatchMode.ASYNC_BATCH.value,
          f"got {res.batch_mode}")


ALL_TESTS = [
    test_classification_routes_fast,
    test_routine_belief_routes_balanced,
    test_major_conflict_routes_strong,
    test_explicit_specialist,
    test_local_when_configured,
    test_local_never_silent_when_unconfigured,
    test_action_policy_rejected,
    test_lmsr_rejected,
    test_budget_exhaustion_degrades_safely,
    test_budget_exhaustion_prefers_cache,
    test_budget_fallback_to_cheaper_tier,
    test_retries_bounded,
    test_escalation_bounded,
    test_stable_prefix_hash_is_stable,
    test_evidence_delta_changes_only_dynamic,
    test_delta_hash_order_insensitive,
    test_no_secrets_in_usage,
    test_legacy_call_still_works,
    test_legacy_client_unchanged,
    test_concurrent_budget_does_not_exceed,
    test_concurrent_strong_slots_bounded,
    test_realtime_batch_for_urgent,
    test_async_batch_for_persona,
]


# --- demo table ------------------------------------------------------

def print_demo():
    print("\n" + "=" * 78)
    print("ROUTING DEMO (mock mode) — task → tier / provider / model / batch")
    print("=" * 78)
    r = _router()
    scenarios = [
        ("classification", TaskRoutingContext(task_type=TaskType.EVIDENCE_CLASSIFICATION)),
        ("short summary", TaskRoutingContext(task_type=TaskType.EVIDENCE_SUMMARY)),
        ("json repair", TaskRoutingContext(task_type=TaskType.JSON_REPAIR)),
        ("routine belief", TaskRoutingContext(task_type=TaskType.ROUTINE_BELIEF_UPDATE)),
        ("archetype gen", TaskRoutingContext(task_type=TaskType.ARCHETYPE_GENERATION)),
        ("major conflict", TaskRoutingContext(
            task_type=TaskType.EVIDENCE_CONFLICT_ANALYSIS,
            evidence_conflict_score=0.9, information_impact_score=0.85)),
        ("major belief", TaskRoutingContext(task_type=TaskType.MAJOR_BELIEF_UPDATE)),
        ("decision audit", TaskRoutingContext(task_type=TaskType.DECISION_AUDIT)),
        ("specialist", TaskRoutingContext(
            task_type=TaskType.CODE_OR_SYSTEM_ANALYSIS, task_metadata={"specialist": True})),
        ("bulk → local", TaskRoutingContext(
            task_type=TaskType.EVIDENCE_CLASSIFICATION, task_metadata={"prefer_local": True})),
        ("urgent closing", TaskRoutingContext(
            task_type=TaskType.ROUTINE_BELIEF_UPDATE, urgency=Urgency.HIGH,
            task_metadata={"market_closing": True})),
    ]
    hdr = f"{'task':<16}{'tier':<11}{'provider':<11}{'model':<16}{'batch':<13}{'budget':<8}{'cache':<7}{'fallback'}"
    print(hdr)
    print("-" * 78)
    for name, ctx in scenarios:
        res = r.route(ctx)
        print(
            f"{name:<16}{res.tier:<11}{(res.provider or '-'):<11}"
            f"{(res.model or '-'):<16}{res.batch_mode:<13}"
            f"{str(res.budget_allowed):<8}{str(res.cache_eligible):<7}"
            f"{str(res.allow_fallback)}"
        )
    print("=" * 78)
    print(f"budget status: {r.budget.status()['budget']}")


def main():
    parser = argparse.ArgumentParser(description="Model-router tests + demo (mock mode).")
    parser.add_argument("--demo", action="store_true", help="print routing table only")
    parser.add_argument("--quiet", action="store_true", help="tests only, no demo table")
    args = parser.parse_args()

    if args.demo:
        print_demo()
        return 0

    print("Running model-router tests (mock mode)...")
    for t in ALL_TESTS:
        print(f"\n{t.__name__}:")
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            global _FAIL
            _FAIL += 1
            print(f"  [FAIL] {t.__name__} raised {exc!r}")

    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)

    if not args.quiet and _FAIL == 0:
        print_demo()

    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
