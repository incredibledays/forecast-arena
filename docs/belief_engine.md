# Belief Engine

## Two tiers — LLM cost stays O(archetypes), not O(agents)

### `ArchetypeBelief` — one LLM belief per (archetype × event × bundle version)
- Unique key `(archetype_id, event_id, evidence_bundle_version)`.
- Fields: `posterior_probability`, `confidence`, concise `reasoning_summary` (**never chain-of-thought**), `key_evidence`, `risk_factors`, model provenance (`model_provider`, `model`, `model_tier`, `batch_mode`, `prompt_version`, `cache_metadata`), `degraded` flag.
- Compatible archetypes (same event + bundle version + schema + tier) are grouped into 10–30-archetype batches; **one routed LLM call per batch**; each result validated independently; only failed items retried (bounded repair/escalation).
- **Freshness check**: `BeliefService._has_current_belief(arch, event, version)`. If a matching row exists → no LLM. Fresh archetype beliefs make the entire per-agent decision path LLM-free.

### `AgentBelief` — lazy, per-agent, pure code
- Unique key `(agent_id, event_id)`.
- Fields: `prior_probability`, `raw_probability`, `calibrated_probability` (clamped [0.01, 0.99]), `confidence`, `belief_status` (`materialized` / `reconstructed` / `stale`), `last_evidence_bundle_version`, `last_market_price_seen`, `next_review_time`, `personalization_components`, `personalization_algo_version`.

## Personalization math (pure code, deterministic, bounded)
```
logit(p_agent) = logit(p_archetype)
               + K_EXPERTISE   · (expertise    − 0.5) · logit(p_arch)
               + K_TRUST       · (avg_trust    − 0.5) · logit(p_arch)
               + (K_OVERCONF · overconfidence − K_HERDING · herding) · logit(p_arch)
               + K_MEMORY      · memory_calibration
               + NOISE_SCALE   · (U(0,1) − 0.5)
p_agent = clamp(sigmoid(logit(p_agent)), 0.01, 0.99)
```
Coefficients: `K_EXPERTISE=0.35, K_TRUST=0.20, K_OVERCONF=0.40, K_HERDING=0.30, K_MEMORY=0.50, NOISE_SCALE=0.30`.
Noise `U` is `blake2b(sim_seed, agent_seed, event_id, bundle_version, algo_version)` mapped to [0,1). Bumping `PERSONALIZATION_ALGO_VERSION` rotates old rows.

## Vectorized batch (`personalize_batch`)
Consumes **projection tuples** `(agent_id, archetype_id, agent_seed, expertise, avg_trust, overconfidence, herding, memory_cal)` — no ORM Agent object per agent, no per-agent dict, no per-agent archetype query. Same math as `personalize_one` element-wise. Verified: **`vector == scalar within 1e-12`** in `test_beliefs.py`.

## Lazy materialization eligibility
`AgentBelief` rows are persisted only for Agents who:
- hold a `Position` on this event's market
- have an `AgentEventInterest` (watcher / subscriber)
- were selected by a wake-up trigger
- explicitly need an auditable snapshot

Sleeping agents get their belief on demand via `reconstruct_agent_belief` — same math, no row created. Verified in `test_beliefs.py` (persisted count ≪ agent count).

## Batch retry semantics
`_process_batch` sends ONE LLM request for the whole batch. Each item is validated with `_validate_belief_json`. Failed items go into `pending` and are re-batched with a repair prompt. Repair is bounded (`max_repairs = 2`). Items still failing become deterministic degraded fallbacks with `degraded=True`.

## Benchmark numbers
| scenario | archetypes | agents | ARCHETYPE LLM calls | AGENT LLM calls |
|---|---|---|---|---|
| `benchmark_beliefs.py --agents 10000 --archetypes 100` | 100 | 10,000 | **5** (5 batches × ~20 archetypes) | **0** |
| Scenario A of scalability benchmark | 100 | 10,000 | **0** (beliefs pre-seeded fresh) | **0** |

## Files
- `models/belief.py`
- `services/belief_service.py` — `update_archetype_beliefs`, `materialize_agent_beliefs`, `reconstruct_agent_belief`, `eligible_agent_ids`
- `benchmark_beliefs.py`
