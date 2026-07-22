# Model Routing

## Layer summary
The router is a **pure decision layer** — it never calls a provider. It maps a `TaskRoutingContext` (task type, impact, conflict, urgency, budget hints) to a `RoutingResult` (`tier`, `provider`, `model`, `batch_mode`, `budget_allowed`, `cache_eligible`, `allow_fallback`).

Logical tiers → concrete models via **environment variables only**. Business logic never names a commercial model.

## Tier ladder (cost-axis)
`FAST → BALANCED → STRONG` plus two orthogonal tiers:
- `LOCAL` — high-volume bulk work when a local provider is configured; **never silent** (routing refuses if unconfigured).
- `SPECIALIST` — coder/reasoning models when the task metadata explicitly asks.

## Task types (LLM-eligible)
`EVIDENCE_CLASSIFICATION · EVIDENCE_SUMMARY · EVIDENCE_CONFLICT_ANALYSIS · ARCHETYPE_GENERATION · PERSONA_ARCHETYPE_GENERATION · PERSONA_VARIATION · ROUTINE_BELIEF_UPDATE · MAJOR_BELIEF_UPDATE · JSON_REPAIR · MEMORY_SUMMARY · DECISION_AUDIT · CODE_OR_SYSTEM_ANALYSIS`

## Task types that MUST NOT use an LLM
Enforced by `validate_task_type` (raises `NonLLMTaskError`):
- `WAKEUP_SCHEDULING`
- `CANDIDATE_SELECTION`
- `KELLY_CALCULATION`
- `ACTION_POLICY`
- `PORTFOLIO_VALUATION`
- `RISK_LIMITS`
- `LMSR_QUOTE`, `LMSR_EXECUTION`
- `SCORE_AGGREGATION`, `LEADERBOARD_CALCULATION`

## Default routing rules
| task | default tier | escalation |
|---|---|---|
| evidence classification / summary / json-repair | FAST | — |
| routine belief update | BALANCED | contested impact→STRONG |
| ordinary conflict analysis | BALANCED | high impact→STRONG |
| major belief update / high-impact contradiction | STRONG | none |
| persona / memory summary | FAST | ASYNC batch mode |
| specialist (code) | SPECIALIST (only on explicit metadata request) | — |
| high-volume bulk | LOCAL (only if configured) | — |

## Budgeting
`BudgetManager` (thread-safe):
- daily input / output / STRONG-only token caps
- per-market / per-EvidenceBundle token caps
- bounded retries (default 2), bounded escalations (default 1)
- max concurrent STRONG requests (default 4)

Exhaustion order: **cached-hit → cheaper tier → deterministic fallback (`degraded=True`)**. Never unbounded retry.

## Cache metadata
Two-hash split, deterministic:
- **Stable prefix**: system prompt version + schema version + market def hash + resolution rules hash + archetype hash + evidence bundle version. **Never includes UUIDs, wall-clock, or volatile ordering.**
- **Dynamic suffix**: prior probability + evidence delta hash + time bucket + optional market state hash.

Provider-neutral (works with OpenAI prefix caching, Anthropic `cache_control`, or vLLM prefix cache).

## Legacy compatibility
`ModelRouter.route_legacy(note)` routes untyped calls to BALANCED and prints one structured warning per call. Callers migrate to typed `TaskType` at their own pace.

## Usage records
`LLMUsage` dataclass captures task_type, tier, provider, model, batch_mode, market_id, bundle_id, token counts (estimated + actual), cache hit, retry_count, escalation_count, fallback_used, degraded, latency, success. **`_scrub` strips any key containing `api_key / auth / bearer / token / secret / password / prompt / cot / reasoning`** — API keys and raw prompts can never appear in usage records.

## Environment variables
```
LLM_TIER_FAST_MODEL / LLM_TIER_FAST_PROVIDER
LLM_TIER_BALANCED_MODEL / LLM_TIER_BALANCED_PROVIDER
LLM_TIER_STRONG_MODEL / LLM_TIER_STRONG_PROVIDER
LLM_TIER_SPECIALIST_MODEL / LLM_TIER_SPECIALIST_PROVIDER
LLM_TIER_LOCAL_MODEL / LLM_TIER_LOCAL_PROVIDER

LLM_BUDGET_DAILY_INPUT_TOKENS
LLM_BUDGET_DAILY_OUTPUT_TOKENS
LLM_BUDGET_DAILY_STRONG_TOKENS
LLM_BUDGET_PER_MARKET_DAILY_TOKENS
LLM_BUDGET_PER_BUNDLE_TOKENS

LLM_MAX_RETRIES=2
LLM_MAX_ESCALATIONS=1
LLM_MAX_CONCURRENT_STRONG=4
```
