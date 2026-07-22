# Operations Runbook

## First-time setup
```bash
# 1. Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Environment
cp .env.example .env
# Set at minimum: DATABASE_URL, and (optional) LLM_TIER_*_MODEL / LLM_BUDGET_*.
# NEVER commit .env — .gitignore already covers it.

# 3. Initialize schema (also seeds a small demo dataset)
python init_db.py --reset
```

## Population generation
```bash
# Deterministic archetypes + agents:
python manage_population.py generate-archetypes --count 100 --seed 42
python manage_population.py generate-agents --count 10000 --seed 42 --batch-size 1000
python manage_population.py validate

# Optional LLM enrichment of archetype descriptions (numbers still pure code):
python manage_population.py generate-archetypes --count 100 --seed 42 --use-llm

# Destructive re-run guard: re-running any generate command REQUIRES --reset.
```

## Scheduler
```bash
# Sample the first natural wake-up for every active Agent (seed-driven):
python run_scheduler.py initialize-natural --seed 42

# Show scheduler + clock status:
python run_scheduler.py inspect

# Advance virtual time (no real sleeping):
python run_scheduler.py advance-time --hours 12

# Claim + report due natural wake-ups (does not itself trade):
python run_scheduler.py due --limit 100
```

## Evidence
```bash
# Refresh evidence for one event (uses TavilyProvider when TAVILY_API_KEY is set):
python manage_evidence.py refresh --event-id 1

# Show current bundle stats:
python manage_evidence.py inspect --event-id 1

# Show the compact delta for a version (or 'latest'):
python manage_evidence.py show-delta --event-id 1 --version latest

# Deterministic offline demo (no network required):
python manage_evidence.py inject-test-event --event-id 1
```

## Workers
```bash
# Single pass, up to 100 due tasks:
python run_agent_workers.py --once --limit 100

# Loop mode (drain until empty):
python run_agent_workers.py --loop --limit 500

# Mock-LLM stress pass — never makes a paid model call:
python run_agent_workers.py --mock-llm --once --limit 1000

# WARNING (printed at startup): --workers > 1 on SQLite will contend on the
# single-writer lock. Use PostgreSQL for real multi-worker parallelism.
```

## Benchmarks
```bash
# Full scalability benchmark (both scenarios):
python benchmark_scalability.py

# Single scenario:
python benchmark_scalability.py --scenario A --a-agents 10000 --a-wakeups 5000 --seed 42
python benchmark_scalability.py --scenario B --b-agents 100000 --seed 42

# Determinism test (seed 42 twice + seed 43 once):
python benchmark_scalability.py --determinism

# Belief-focused benchmark:
python benchmark_beliefs.py --agents 10000 --archetypes 100 --mock-llm --seed 42
```

## Test suites
```bash
python test_lmsr_engine.py               # 55 assertions
python test_dynamic_lmsr_integration.py  # 34 assertions
python test_dynamic_positions.py         # 13 tests
python test_actions.py                   # 30 assertions
python test_beliefs.py                   # 27 assertions
python test_evidence.py                  # 29 assertions
python test_triggers.py                  # 37 assertions
python test_scheduler.py                 # 29 assertions
python test_population.py                # 31 assertions
python test_wakeup_processor.py          # 28 assertions
python test_model_router.py --quiet      # 48 assertions
```

Total: **361 assertions across 11 test suites.**

## Configuration surface (`.env`)
See `.env.example` for the canonical list; the important ones:

| var | meaning |
|---|---|
| `DATABASE_URL` | SQLite for dev, `postgresql://...` for prod |
| `FLASK_PORT` | web dashboard |
| `LLM_PROVIDER` / `LLM_API_KEY` / `LLM_API_BASE` / `LLM_MODEL` | legacy client (still supported) |
| `LLM_TIER_{FAST/BALANCED/STRONG/SPECIALIST/LOCAL}_{PROVIDER,MODEL}` | per-tier concrete models — **the ONLY place model names live** |
| `LLM_BUDGET_DAILY_INPUT_TOKENS` / `_OUTPUT_TOKENS` / `_STRONG_TOKENS` | daily caps |
| `LLM_BUDGET_PER_MARKET_DAILY_TOKENS` / `_PER_BUNDLE_TOKENS` | scoped caps |
| `LLM_MAX_RETRIES` (2) / `LLM_MAX_ESCALATIONS` (1) / `LLM_MAX_CONCURRENT_STRONG` (4) | bounded work |
| `TAVILY_API_KEY` | retrieval provider (absent → evidence refresh is a no-op) |

## SQLite warnings + when to switch to Postgres
- **`--workers > 1` on SQLite** — will hit `database is locked`. Use `--workers 1` locally.
- **>10 GB persisted state** — `price_history` and `trades` grow linearly; SQLite handles it but query planning degrades. Move to Postgres + partition by month.
- **Multi-process producers** (multiple worker processes) — the per-market Python lock only serializes within one process. Move to Postgres with `SELECT ... FOR UPDATE` on the Market row.

## Backup + replay
Every simulation is reproducible from `(schema_version, seed, sequence_of_wakeups)`:
- Deterministic hash over Trade + Market + Agent state is stable per seed (proven in the benchmark).
- Bumping `PERSONALIZATION_ALGO_VERSION` or `DEFAULT_SCHEDULER_VERSION` intentionally rotates the state so old rows can never be silently mis-personalized.

## Failure modes worth alerting on
- `AgentDecision.was_stale=True` count spikes → market state is churning faster than workers can quote.
- `WakeUpTask.status = 'failed'` count > 0 → check `last_error`; retries are bounded.
- `ArchetypeBelief.degraded=True` count grows → LLM budget exhausted; router is doing its job but tighten the alert.
- Peak process RSS growing across `--loop` batches → an ORM-map regression; ORM identity map should stay flat (verified in `test_scheduler.py` and `test_wakeup_processor.py`).
