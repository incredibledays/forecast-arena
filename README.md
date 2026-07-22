# ForecastArena

[中文说明](README.zh-CN.md)

ForecastArena is a Flask + SQLAlchemy virtual prediction-market simulator for AI-agent forecasting experiments. It models each forecast event as one or more binary YES/NO markets, prices trades with Hanson's LMSR market maker, and supports evidence retrieval, LLM-assisted belief updates, scheduled agent wake-ups, trade execution, settlement, and leaderboard scoring.

The project is intended as a **local demo and research prototype**. By default it uses SQLite, and all cash, shares, trades, and payouts are virtual.

## Features

- Create and inspect forecast events from the Web UI.
- Supports binary, categorical, scalar, grouped, and conditional event types.
- LMSR-based automatic market making for `BUY_YES`, `BUY_NO`, `SELL_YES`, `SELL_NO`, `FLIP_YES`, `FLIP_NO`, and `HOLD` decisions.
- Tracks agent cash, YES/NO shares, price history, trades, positions, and `AgentDecision` audit rows.
- Live mode runs a virtual clock, natural wake-ups, evidence refresh, worker processing, and maintenance.
- Optional Tavily integration for external evidence retrieval.
- Optional OpenAI-compatible LLM provider for evidence appraisal, archetype beliefs, and agent decisions.
- `Recent Evidence` shows recently retrieved source evidence only; it does not show agent trade reasons.
- Long-running maintenance keeps high-volume tables bounded so the UI and database do not grow without limit.

## Core Concepts

- `Event`: the forecast question shown to users.
- `Market`: one binary YES/NO tradable unit under an event. Multi-outcome events are represented by multiple markets.
- `Agent`: a virtual trader with cash, strategy settings, risk preferences, memory, and optional archetype linkage.
- `Trade`: a persisted non-HOLD market action.
- `AgentDecision`: the audit record for an agent decision. HOLD decisions are recorded here even though they do not create a `Trade`.
- `Position`: per-agent YES/NO share holdings for a market.
- `SourceContent` / `InformationEvent`: retrieved source content and its event-specific appraisal.
- `EvidenceBundle` / `EvidenceDelta`: versioned evidence summaries used by the belief engine.

## Repository Layout

```text
app.py                    Flask Web UI and local debug endpoints
init_db.py                Database initialization and seed data
run_live.py               Integrated live runner: web + clock + evidence + worker
run_agents.py             Simple batch agent runner
run_agent_workers.py      Drain WakeUpTask worker queue
resolve_event.py          CLI event settlement
models/                   SQLAlchemy models
services/                 Trading, scheduling, evidence, belief, and settlement services
agents/                   Agent strategy implementations
retrieval/                Evidence retrieval, query expansion, scoring, and stance logic
llm/                      OpenAI-compatible providers, routing, budgets, and cache helpers
templates/                Jinja templates
static/                   CSS, Chart.js, and compiled Tailwind assets
docs/                     Architecture notes
test_*.py                 pytest suite
```

## Requirements

- Python 3.10+; the current test environment uses Python 3.12.
- Node.js/npm, only if rebuilding Tailwind or using `npm test`.
- SQLite by default.
- Optional Tavily API key.
- Optional OpenAI-compatible LLM endpoint.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

A minimal local run does not require any API keys. Without Tavily or an LLM provider, the system still runs with mock or heuristic behavior.

## Configuration

Common `.env` settings:

```bash
DATABASE_URL=sqlite:///forecast_arena.db
FLASK_PORT=6006
TAVILY_API_KEY=
LLM_API_KEY=
LLM_API_BASE=
LLM_MODEL=gpt-4o-mini
```

Notes:

- `DATABASE_URL` defaults to SQLite and is suitable for local demos.
- Leave `TAVILY_API_KEY` empty to disable external evidence retrieval.
- `LLM_API_KEY`, `LLM_API_BASE`, and `LLM_MODEL` can point to any OpenAI-compatible endpoint.
- Advanced provider and tier routing configuration is documented in `.env.example` and `llm_providers.example.json`.
- Do not commit `.env`, runtime databases, cache folders, or local logs.

## Initialize the Database

Create tables and seed starter data:

```bash
python init_db.py
```

Reset all data and reseed:

```bash
python init_db.py --reset
```

## Run the Web UI

Start only the Flask Web server:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:6006/
```

If you change Python code, templates, or static assets, restart the server and refresh the browser.

## Run Live Mode

Live mode starts the virtual clock, natural wake-up bridge, evidence refresh loop, worker loop, maintenance loop, and Web UI.

Offline fast demo mode, with no paid LLM calls:

```bash
python run_live.py --mock-llm --speed 60
```

If real LLM and Tavily configuration is available, run:

```bash
python run_live.py --speed 60
```

A conservative Web-friendly configuration:

```bash
python run_live.py --mock-llm --speed 60 \
  --worker 1 \
  --worker-limit 50 \
  --worker-micro-batch 10 \
  --max-pending-total 1000 \
  --evidence-trigger-budget 25
```

Run live mode without the Web server:

```bash
python run_live.py --mock-llm --no-web
```

Start from an existing database without automatic initialization:

```bash
python run_live.py --skip-init --speed 60
```

## Live Mode Options

Common options:

```text
--speed N                         Virtual-clock acceleration factor
--refresh SECONDS                 Evidence refresh interval; default 60
--worker SECONDS                  Worker loop interval; default 3
--worker-limit N                  Maximum tasks processed per worker pass
--worker-micro-batch N            Per-market worker micro-batch size
--max-pending-total N             Global pending WakeUpTask cap
--evidence-trigger-budget N       Maximum agents woken per event evidence refresh
--price-trigger-fanout N          Maximum PRICE agents woken by a price-moving trade
--no-evidence                     Disable evidence refresh
--no-worker                       Do not execute the agent worker
--no-maintenance                  Disable long-run cleanup; not recommended
--information-event-keep-per-event N  Recent InformationEvent rows kept per event; default 1000
```

Show all options:

```bash
python run_live.py --help
```

## Create and Resolve Events

Create events from the UI:

```text
/create-event
```

Newly created events automatically receive a watcher set and initial wake-up tasks, so a running live worker can react.

Resolve events from the event-detail page using the `Resolve event` panel, or via CLI:

```bash
python resolve_event.py --event-id 1 --outcome YES
```

For multi-market event types, inspect the CLI help:

```bash
python resolve_event.py --help
```

## Recent Evidence Behavior

`Recent Evidence` displays recently retrieved external sources:

- Data comes from `InformationEvent` and `SourceContent`.
- The event page displays at most the latest 30 evidence items.
- It does not display agent trade reasons, buy reasons, or LLM operation reasons.
- Live-mode maintenance keeps the latest 1000 `InformationEvent` rows per event by default and removes orphaned `SourceContent` rows.
- The page is server-rendered. Refresh the event page to see the latest evidence.

## Leaderboard

The leaderboard supports multiple sorting metrics. The default sort is by profit; ties are broken by initial capital. Other metrics can be selected from the UI.

## Debug Trade Endpoint

This endpoint is for local debugging only.

Manual buy:

```bash
curl -X POST http://127.0.0.1:6006/dev/trade \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":1,"event_id":1,"action":"BUY_YES","amount":500}'
```

Sell actions require `fraction`:

```bash
curl -X POST http://127.0.0.1:6006/dev/trade \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":1,"event_id":1,"action":"SELL_YES","fraction":0.5}'
```

## Long-Running Performance

Live mode enables maintenance by default to keep the local database bounded:

- Deletes terminal DONE / SHED / FAILED wake-up tasks after a retention window.
- Requeues stale CLAIMED tasks.
- Keeps a recent tail of `PriceHistory` rows per market.
- Keeps a bounded number of `AgentDecision` debug rows.
- Keeps a bounded number of `EvidenceBundle` / `EvidenceDelta` versions per event.
- Keeps a bounded number of `InformationEvent` rows per event and deletes orphaned `SourceContent` rows.

If the logs show:

```text
backpressure: pending=... >= ..., 暂停自然唤醒入队
```

then tasks are being enqueued faster than workers are draining them. You can:

- Increase `--worker-limit` or `--worker-micro-batch`.
- Increase `--max-pending-total`.
- Decrease `--evidence-trigger-budget` or `--price-trigger-fanout`.
- Reduce the number of agents or events.

SQLite is suitable for local single-machine demos. For real multi-process worker concurrency, use PostgreSQL because SQLite serializes writers.

## Static Assets

Tailwind is precompiled into `static/css/tailwind.css`. Rebuild after changing Tailwind classes or CSS inputs:

```bash
npm install
./build-static.sh
```

The build script also creates gzip versions of static CSS and JS assets.

## Tests

Run the full test suite:

```bash
python -m pytest -q
```

Or use npm:

```bash
npm test
```

Recommended pre-release checks:

```bash
python -m pytest -q
python -m compileall -q app.py agents llm models retrieval services run_live.py
```

## Release Checklist

- Confirm `.env` is not committed.
- Confirm `instance/`, `*.db`, `node_modules/`, `__pycache__/`, and `.pytest_cache/` are not committed.
- Run the full test suite.
- Run a local smoke test with `python run_live.py --mock-llm --speed 60`.
- Open the Web UI and check markets, event detail, event creation, leaderboard, and settlement controls.
- If CSS changed, rebuild and commit `static/css/tailwind.css`, `static/css/app.css`, and their `.gz` files.

## License

The project currently follows the repository configuration. Confirm the final license text and ownership details before external publication.
