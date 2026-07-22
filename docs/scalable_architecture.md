# Scalable Architecture

## Design in one sentence
Every operation whose cost naïvely scales with the Agent population has been re-shaped so its cost scales with a small, bounded quantity instead — archetypes (~100–300), eligible agents on an event (~hundreds), or configurable batches.

## Component map
```
┌─ population ─┐    ┌─ scheduler ─┐    ┌─ triggers ─┐    ┌─ workers ────────┐
│ Archetypes   │    │ natural     │    │ INFORMATION│    │ AgentWakeup      │
│ Agents (10k+)│    │ wake-ups    │──▶│ PRICE       │──▶│ Processor        │
└──────┬───────┘    │ (one-next)  │    │ RISK/CLOSE │    │  claim → belief  │
       │            └─────────────┘    │ RESOLUTION │    │  → policy → LMSR │
       │                                └──┬─────────┘    │  → memory        │
       │            ┌─ candidates ─┐        │             └──────┬───────────┘
       ▼            │ sparse maps  │◀───────┘                    │
┌─ beliefs ─┐       │ (holder /    │                             ▼
│ Archetype │       │  watcher /   │                    ┌─ market_service ─┐
│  belief   │──▶   │  expert /    │                    │ strict LMSR     │
│ (LLM once │      │  archetype)  │                    │ non-mutating    │
│  per bundle)│    └──────────────┘                    │ quote surface   │
│ Agent     │                                          │ MarketExecutor  │
│  belief   │                                          │ (per-market lock)│
│ (lazy /   │                                          └─────────────────┘
│  code)    │
└───────────┘
```

## Cost model
| operation | naïve cost | actual cost | mechanism |
|---|---|---|---|
| population generation | O(N·LLM) | **O(archetypes)** LLM + O(N) code | archetype templates + seeded jitter |
| candidate selection for event | O(N) scan | **O(sparse-set)** | `AgentEventInterest` / `Position` / `AgentCategoryExpertise` indexes |
| natural scheduling | O(N) sample/step | **O(N)** ONCE, then O(1)/wake | store only next wake-up |
| belief update | O(N·LLM) | **O(archetypes)** LLM + O(eligible) code | `BeliefService.update_archetype_beliefs` + `reconstruct_agent_belief` |
| memory update | O(history scan) | **O(1)** incremental | `AgentMemoryStats` compact row |
| action decision | O(history) | **O(1)** pure code | `ActionPolicy` reads stats + belief |
| LMSR execute | already O(1) | O(1) | `MarketService.execute_trade` |
| worker drain | O(agents) | **O(due tasks)** | `WakeUpTask` queue on indexed `(status, scheduled_at)` |

## Storage footprint (measured at 10k agents)
| table | size / bytes-per-row | notes |
|---|---|---|
| `agents` | ~400 B × N | wide row (all persona fields) |
| `agent_archetypes` | ~300 B × A | JSON blobs |
| `agent_memory_stats` | ~200 B × N | one compact row per agent |
| `agent_schedule_state` | ~80 B × N | one **next** wake per agent |
| `agent_event_interest` | ~80 B × sparse | only where an interest exists |
| `wakeup_tasks` | ~250 B × pending | queue drains |
| `trades` | ~150 B × traded | HOLDs never persist here |
| `agent_decisions` | ~250 B × attempt | HOLDs + successes both land here |

**Measured: 10,000 agents + 100 archetypes + 50 markets + scheduler = 10.6 MB on-disk (SQLite).**

## Determinism seams
- Every RNG stream is seeded from `(sim_seed, agent_seed, sequence, algo_version)` via blake2b — never `random.random()`.
- Wall-clock timestamps live only on audit rows (`created_at`); the simulation clock is `SchedulerClock.virtual_time_s`.
- Trade prices, LMSR q values, and market versions are pure functions of the ordered trade sequence.
- Result: same seed → identical hash of Trade + Market state (verified in `benchmark_results.md`).

## Bulk / batch primitives
- `PopulationService.generate_agents` — `bulk_insert_mappings` in 1k chunks; expunges between chunks. **~1,200 agents/s measured.**
- `SchedulerService.initialize_natural` — id-range keyset scan of Agent, chunk-bulk insert into `agent_schedule_state`. **~7,470 rows/s measured at 100k.**
- `CandidateService.build_category_expertise` — id-range keyset scan of persona JSON, bulk insert of sparse expertise rows.
- `MarketExecutor` — per-market `threading.Lock`; different markets execute concurrently.
