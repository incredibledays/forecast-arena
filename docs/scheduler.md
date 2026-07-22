# Scheduler

## What it schedules (this phase)
Two flows:
1. **Natural wake-ups** — one `AgentScheduleState` row per Agent holds the **next** natural wake-up. Never months of future rows.
2. **Event-driven wake-ups** — sparse `WakeUpTask` rows created by `TriggerService` (information / price / portfolio / market-closing / resolution). Deduplicated by `(agent_id, event_id, time_bucket)`.

## Virtual clock
`SchedulerClock` singleton row holds `virtual_time_s`. Advanced explicitly (`SchedulerService.advance_time(hours=...)`) — nothing sleeps on wall-clock. Simulation seed + scheduler-algo version stored on the same row so runs replay.

## Natural wake-up sampling
Exponential inter-arrival:
```
wait_days = -ln(U) / rate_per_day
U = uniform01 deterministic from blake2b(sim_seed | agent_seed | sequence | version)
```
Same seed → identical timestamps. Verified in `test_scheduler.py`.

Rates come from the Agent's activity band (ultra_active → very_low_frequency) multiplied by a per-Agent jitter, produced at population time.

## Due query
```sql
SELECT ... FROM agent_schedule_state
WHERE status = 'active' AND next_natural_wakeup_at <= :now
ORDER BY agent_id  -- keyset id-range pagination
LIMIT :batch
```
Backed by the covering composite index `ix_sched_status_next(status, next_natural_wakeup_at)`. **Never** joins the `agents` table (denormalized `status`, `base_wakeup_rate_per_day`, `agent_random_seed` columns on the schedule row make it self-contained).

## Atomic claim (natural due)
`SchedulerService.due(limit)` uses compare-and-swap `UPDATE ... WHERE natural_wakeup_sequence = ?` — the sequence increment IS the lock. Two workers can't process the same wake-up twice.

## Event-driven wake-ups (`WakeUpTask`)
Trigger types + priority ladder:
```
NATURAL           10
INFORMATION       40
PRICE             50
MARKET_CLOSING    70
PORTFOLIO_RISK    90  ← never shed
RESOLUTION       100  ← never shed
```

Dedup: `(agent_id, event_id, time_bucket)` UNIQUE. Merge rules on collision: max priority, min scheduled time, union of `wake_reasons`, max of each ranking signal.

Jitter (deterministic, per-Agent per-sequence):
- major official event: 0–5 min
- normal event: 5–120 min
- low-impact: 2–24 h

## Backpressure controls (in `ProcessorConfig` + `TriggerService`)
- `max_pending_per_market`
- `max_price_cascade_depth`
- Load shedding: `TriggerService.shed_load(keep=N)` retains RESOLUTION + PORTFOLIO_RISK unconditionally, sheds lowest (priority, wake_score) among the rest.

## Measured performance
| item | measurement |
|---|---|
| `initialize_natural` for 10,000 agents | 1.22 s |
| `initialize_natural` for 100,000 agents | 13.38 s (≈ 7,470 rows/s) |
| natural due-query on 10k schedule | O(index range scan), <1 ms per page |
| natural wake-ups sampled with wall-clock | zero — pure virtual time |

## Files
- `models/schedule_state.py` — `AgentScheduleState`, `SchedulerClock`
- `models/wakeup.py` — `WakeUpTask` (event-driven), `TriggerCooldown`
- `services/scheduler_service.py` — `sample_wait_seconds`, `initialize_natural`, `due`, `advance_time`
- `services/candidate_service.py` — sparse candidate union
- `services/trigger_service.py` — information / price / risk / closing / resolution triggers
- `run_scheduler.py` — CLI (`initialize-natural`, `due`, `advance-time`, `inspect`)
