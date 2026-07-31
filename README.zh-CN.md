# ForecastArena

ForecastArena 是一个基于 Flask + SQLAlchemy 的虚拟预测市场 / AI Agent 交易模拟器。它把一个预测事件建模成一个或多个二元 YES/NO 市场，使用 Hanson LMSR 做自动做市，并支持证据检索、LLM 辅助信念更新、调度唤醒、交易执行、结算和排行榜。

当前定位是**本地演示与研究原型**：默认使用 SQLite，所有资金和交易都是虚拟的。

## 功能概览

- Web UI 创建和查看预测事件。
- 支持事件类型：Binary、Categorical、Scalar、Grouped、Conditional。
- 使用 LMSR 自动定价，支持 `BUY_YES`、`BUY_NO`、`SELL_YES`、`SELL_NO`、`FLIP_YES`、`FLIP_NO`、`HOLD`。
- 维护 agent 现金、YES/NO shares、价格历史、交易历史、AgentDecision 审计行。
- Live 模式运行虚拟时钟、自然唤醒、证据刷新、worker 决策和交易执行。
- 可选 Tavily 检索外部证据；`Recent Evidence` 只展示最近检索到的来源，不展示 agent 买卖理由。
- 可选 OpenAI-compatible LLM provider，用于 archetype belief / evidence appraisal / agent 决策。
- 长时间运行时自动清理高频表，避免 UI 和数据库无限增长。

## 核心概念

- `Event`：用户看到的预测问题。
- `Market`：一个可交易的二元 YES/NO 市场。多结果事件会展开成多个 market。
- `Agent`：虚拟交易者，包含现金、策略、风险偏好、记忆和可选 archetype。
- `Trade`：实际执行的非 HOLD 操作。
- `AgentDecision`：agent 决策审计记录，HOLD 也会记录在这里。
- `Position`：某个 agent 在某个 market 上持有的 YES/NO shares。
- `SourceContent` / `InformationEvent`：检索证据和它对特定事件的 appraisal。
- `EvidenceBundle` / `EvidenceDelta`：供 belief engine 使用的版本化证据摘要。

## 目录结构

```text
app.py                    Flask Web UI 和本地 debug endpoint
init_db.py                建库和 seed 数据
run_live.py               一体化 live runner：web + clock + evidence + worker
run_agents.py             简单 batch agent runner
run_agent_workers.py      单独 drain WakeUpTask worker
resolve_event.py          CLI 结算事件
models/                   SQLAlchemy models
services/                 交易、调度、证据、信念、结算等服务层
agents/                   agent 策略实现
retrieval/                证据检索、query expansion、scoring、stance
llm/                      OpenAI-compatible provider、routing、budget、cache
templates/                Jinja 页面
static/                   CSS、Chart.js、编译后的 Tailwind
docs/                     架构说明
test_*.py                 pytest 测试
```

## 环境要求

- Python 3.10+（当前测试环境使用 Python 3.12）
- Node.js/npm（仅在需要重编译 Tailwind 或运行 `npm test` 时需要）
- SQLite（默认）
- 可选：Tavily API key、OpenAI-compatible LLM endpoint

## 安装

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

复制环境变量模板：

```bash
cp .env.example .env
```

最小本地运行不需要任何 API key。不开 Tavily / LLM 时，系统仍可使用 mock 或启发式逻辑运行。

## 配置

常用 `.env` 项：

```bash
DATABASE_URL=sqlite:///forecast_arena.db
FLASK_PORT=6006
TAVILY_API_KEY=
LLM_API_KEY=
LLM_API_BASE=
LLM_MODEL=gpt-4o-mini
```

说明：

- `DATABASE_URL` 默认是 SQLite；本地演示足够。
- `TAVILY_API_KEY` 为空时不会刷新外部检索证据。
- `LLM_API_KEY` / `LLM_API_BASE` / `LLM_MODEL` 支持任意 OpenAI-compatible endpoint。
- 复杂 provider / tier routing 可参考 `llm_providers.example.json` 和 `.env.example`。
- 发布或提交代码时不要提交 `.env`、数据库文件、缓存目录。

## 初始化数据

创建表并 seed starter 数据：

```bash
python init_db.py
```

清空并重建：

```bash
python init_db.py --reset
```

## 运行 Web UI

只启动 Flask Web：

```bash
python app.py
```

默认打开：

```text
http://127.0.0.1:6006/
```

如果修改了代码或模板，需要重启服务并刷新浏览器。

## 运行 Live 模式

Live 模式会同时启动：虚拟时钟、自然唤醒、证据刷新、worker、maintenance 和 Web UI。

离线快速演示（不调用真实 LLM）：

```bash
python run_live.py --mock-llm --speed 60
```

如果已经配置好真实 LLM / Tavily，也可以直接运行：

```bash
python run_live.py --speed 60
```

更保守的 Web 友好配置：

```bash
python run_live.py --mock-llm --speed 60 \
  --worker 1 \
  --worker-limit 50 \
  --worker-micro-batch 10 \
  --max-pending-total 1000 \
  --evidence-trigger-budget 25
```

只跑后台，不启动 Web：

```bash
python run_live.py --mock-llm --no-web
```

跳过初始化，直接基于现有 DB 启动：

```bash
python run_live.py --skip-init --speed 60
```

## Live 模式常用参数

```text
--speed N                         虚拟时钟加速倍率
--refresh SECONDS                 证据刷新间隔，默认 60
--worker SECONDS                  worker 处理间隔，默认 3
--worker-limit N                  每轮 worker 最多处理任务数
--worker-micro-batch N            worker 内部分 market 微批大小
--max-pending-total N             全局 pending WakeUpTask 上限
--evidence-trigger-budget N       每次证据刷新最多唤醒多少 agent
--price-trigger-fanout N          每笔价格变化最多唤醒多少 PRICE agent
--no-evidence                     禁用证据刷新
--no-worker                       不执行 agent worker
--no-maintenance                  禁用长跑清理，不建议
--information-event-keep-per-event N  每个 event 保留最近多少检索证据，默认 1000
```

查看完整参数：

```bash
python run_live.py --help
```

## 创建和结算事件

从 UI 创建事件：

```text
/create-event
```

新建事件会自动分配 watcher 并加入初始 wake-up task，因此运行中的 live worker 可以开始响应。

从 UI 结算事件：进入事件详情页右侧的 `Resolve event`。

CLI 结算：

```bash
python resolve_event.py --event-id 1 --outcome YES
```

多 market 类型请查看帮助：

```bash
python resolve_event.py --help
```

## Recent Evidence 行为

`Recent Evidence` 显示的是最近检索到的外部证据来源：

- 数据来自 `InformationEvent` / `SourceContent`。
- 页面最多展示最近 30 条。
- 不展示 agent 交易理由、买入理由或 LLM 操作理由。
- 长跑 maintenance 默认每个 event 保留最近 1000 条 `InformationEvent`，并清理不再引用的 `SourceContent`。
- 页面是普通服务端渲染；要看最新变化请刷新事件页。

## 排行榜

Leaderboard 支持不同指标排序。默认按赚钱数量排序；赚钱相同时按初始本金排序。可在 UI 上选择其它排序指标。

## Debug Trade Endpoint

仅用于本地调试。手动买入：

```bash
curl -X POST http://127.0.0.1:6006/dev/trade \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":1,"event_id":1,"action":"BUY_YES","amount":500}'
```

卖出需要 `fraction`：

```bash
curl -X POST http://127.0.0.1:6006/dev/trade \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":1,"event_id":1,"action":"SELL_YES","fraction":0.5}'
```

## 长时间运行和性能

Live mode 默认启动 maintenance，用来控制长期运行的数据库增长：

- 清理 DONE / SHED / FAILED wake-up tasks。
- 重排超时的 CLAIMED tasks。
- 每个 market 保留最近 `PriceHistory` 点。
- 保留有限数量的 `AgentDecision` 调试行。
- 每个 event 保留有限数量的 `EvidenceBundle` / `EvidenceDelta`。
- 每个 event 保留有限数量的 `InformationEvent`，并清理 orphan `SourceContent`。

如果看到：

```text
backpressure: pending=... >= ..., 暂停自然唤醒入队
```

说明入队速度超过处理速度。可以：

- 提高 `--worker-limit` 或 `--worker-micro-batch`。
- 提高 `--max-pending-total`。
- 降低 `--evidence-trigger-budget` 或 `--price-trigger-fanout`。
- 减少 agent / event 数量。

SQLite 适合本地单机演示。真正多进程 worker 并发建议换 PostgreSQL，因为 SQLite 写入会串行化。

## 静态资源

Tailwind 已预编译到 `static/css/tailwind.css`。如需重新生成：

```bash
npm install
./build-static.sh
```

## 测试

运行全部测试：

```bash
python -m pytest -q
```

或：

```bash
npm test
```

发布前建议至少执行：

```bash
python -m pytest -q
python -m compileall -q app.py agents llm models retrieval services run_live.py
```

## 发布前清单

- 确认 `.env` 没有被提交。
- 确认 `instance/`、`*.db`、`node_modules/`、`__pycache__/`、`.pytest_cache/` 没有被提交。
- 运行全量测试。
- 用 `python run_live.py --mock-llm --speed 60` 做一次本地 smoke test。
- 打开 Web UI，检查首页、事件详情、创建事件、排行榜、结算按钮。
- 如果修改了 CSS，确认 `static/css/tailwind.css` / `static/css/app.css` 已更新。

## 许可证

ForecastArena 使用 MIT License 授权。详情见 [LICENSE](LICENSE)。
