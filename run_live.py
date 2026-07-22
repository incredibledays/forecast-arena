"""run_live.py — 一键实时预测市场(初始化 + wall-clock + 真实证据 + web)

一条命令搞定所有事。首次运行会自动完成从零到实时的全部步骤:

  1. 建 schema(db.create_all + scheduler tables + SQLite WAL)
  2. 生成 archetype + agent 种群(如果没有)
  3. 初始化自然唤醒调度(如果没有)
  4. 建 category 专家索引(如果空)
  5. 给一部分 agent 加 watcher 订阅(如果没有)
  6. 创建 demo 事件(如果没有,且没加 --no-demo-events)
  7. --mock-llm 模式时预置分歧的 archetype 信念(让 demo 能出交易)

已经存在的数据保留(除非 --reset)。之后启动 4 个后台线程:

  * tick     — 推进虚拟时钟
  * natural  — 自然唤醒 → WakeUpTask
  * evidence — Tavily 拉新闻 → 切 bundle → 发 INFORMATION 触发
  * worker   — 消费 WakeUpTask → 决策 → LMSR 执行

主线程跑 Flask。浏览器打开 http://localhost:6006 实时看。

用法:
    python run_live.py                                     # 开箱即用(500 agents / 3 events)
    python run_live.py --agents 5000 --archetypes 100     # 大规模
    python run_live.py --reset --agents 200                # 清库重建
    python run_live.py --mock-llm --speed 60                # 离线快速演示(60x 加速)
    python run_live.py --no-web --no-evidence              # 无头 + 无网络

真实用法(有 LLM + Tavily):
    在 .env 里配好 LLM_TIER_*_MODEL 与 TAVILY_API_KEY,然后:
        python run_live.py --agents 5000 --tick 60
    Agent 信念的分歧来自 LLM 对真实新闻的解读,不是预置。

SQLite 提示:自动开 WAL 让读写并发好一些;真正的多进程并发需要
PostgreSQL(把 DATABASE_URL 指向 postgres 即可,应用代码不变)。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app import app
from models import (
    Agent, AgentArchetype, AgentCategoryExpertise, AgentEventInterest,
    AgentMemoryStats, AgentScheduleState, ArchetypeBelief, Event, EventType, EvidenceBundle,
    Market, MarketStatus, Position, ROLE_WATCHER, TRIGGER_PRIORITY, TriggerType,
    STATUS_CLAIMED, STATUS_PENDING, STATUS_SHED, WakeUpTask, db, ensure_perf_indexes,
    make_dedup_key, time_bucket,
)
from services import (
    AgentWakeupProcessor, CandidateService, EvidenceService, PopulationService,
    ProcessorConfig, SchedulerService, TriggerService,
)
from services.trigger_service import EventBudget

# 让 setup / worker 日志在管道场景下也能实时看到 —— 默认 stderr 是全
# 缓冲的,`python ... 2>&1 | tail` 时会挤在退出的一刻才出现。
try:
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except AttributeError:
    pass  # Python < 3.7 或非文本流,忽略


# 优雅关闭:主线程和后台线程共享一个 Event。ctrl-c 触发 handler 后 set。
_STOP = threading.Event()

# SQLite has exactly one writer. Running tick/natural/worker as separate
# threads is fine for wall-clock orchestration, but letting them enter write
# transactions concurrently makes one writer busy-wait while another holds a
# transaction; web requests then appear to hang behind the backlog. Serialize
# the short background write sections in-process. This is intentionally NOT
# held around external network calls (evidence retrieval / LLM setup).
_DB_WRITE_LOCK = threading.Lock()

# 单例的 LLM 句柄(所有线程共享一个 client)——避免每个循环重建。
_LLM_CLIENT = None
_ROUTER = None
_LLM_LOCK = threading.Lock()


def _get_llm(mock: bool):
    """返回 (llm_client, router)。mock=True 时返回 (None, None)。

    None 会让 BeliefService 使用离线 stub —— 无付费调用。
    """
    global _LLM_CLIENT, _ROUTER
    if mock:
        return None, None
    with _LLM_LOCK:
        if _LLM_CLIENT is None:
            try:
                from llm import get_llm_client, get_model_router
                _LLM_CLIENT = get_llm_client()
                _ROUTER = get_model_router()
                if not _LLM_CLIENT.available:
                    print(f"[live] LLM 未配置 ({_LLM_CLIENT.unavailable_reason});"
                          "belief 更新会走 offline stub", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"[live] LLM 句柄不可用 ({exc});用 offline stub", file=sys.stderr)
                _LLM_CLIENT = None
                _ROUTER = None
        return _LLM_CLIENT, _ROUTER


def _try_tavily():
    """返回一个已启用的 SearchProvider,失败返回 None。"""
    try:
        from retrieval import TavilyProvider
        provider = TavilyProvider()
        if getattr(provider, "enabled", False):
            return provider
        print("[live] TAVILY_API_KEY 未设置 —— 证据刷新循环禁用",
              file=sys.stderr)
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[live] Tavily provider 不可用 ({exc})", file=sys.stderr)
        return None


def _enable_wal():
    """SQLite 上开 WAL,允许多读一写并发。其他后端 no-op。"""
    try:
        uri = app.config.get("SQLALCHEMY_DATABASE_URI", "") or ""
        if not uri.startswith("sqlite"):
            return
        db.session.execute(db.text("PRAGMA journal_mode=WAL"))
        busy_timeout_ms = int(float(os.getenv("SQLITE_BUSY_TIMEOUT", "30")) * 1000)
        db.session.execute(db.text(f"PRAGMA busy_timeout={busy_timeout_ms}"))
        db.session.commit()
        print(f"[live] SQLite WAL 已开启,busy_timeout={busy_timeout_ms/1000:g}s",
              file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[live] 无法开 WAL: {exc}", file=sys.stderr)


DEMO_EVENT_TEMPLATES = [
    ("Will AAPL close above $250 by Q4 2026?", "markets"),
    ("Will OpenAI release GPT-6 in 2026?", "ai"),
    ("Will the Fed cut rates in December 2026?", "macro"),
    ("Will a top-10 AI lab merge or be acquired in 2026?", "ai"),
    ("Will NVDA revenue growth exceed 40% in FY2026?", "markets"),
]


def _reset_all_data():
    """--reset:清空所有数据(schema 保留)。幂等,漏表也不报错。"""
    tables = [
        # 依赖顺序:先 FK 深的
        "wakeup_tasks", "trigger_cooldowns", "agent_decisions",
        "trades", "positions", "price_history",
        "agent_beliefs", "archetype_beliefs",
        "evidence_deltas", "evidence_bundles", "information_events",
        "source_content",
        "agent_event_interest", "agent_category_expertise",
        "archetype_event_interest",
        "agent_memory_episodes", "agent_memory_stats",
        "markets", "events",
        "agent_schedule_state", "scheduler_clock",
        "agents", "agent_archetypes",
    ]
    for t in tables:
        try:
            db.session.execute(db.text(f"DELETE FROM {t}"))
        except Exception:
            pass  # 表可能还不存在,忽略
    db.session.commit()
    print("[live][setup] --reset:所有数据表已清空", file=sys.stderr)


def _bootstrap_or_init(args) -> bool:
    """幂等的一键初始化:缺什么补什么。已有的数据保留。

    步骤(每一步都可跳过):
      A. --reset:清库(保留 schema)
      B. schema:db.create_all() —— 已 memoized
      C. Population:archetypes + agents,数量不足时生成
      D. Scheduler:自然唤醒调度状态,空时建
      E. Category expertise 索引:空时 build
      F. Watcher 订阅:开放事件一个订阅都没有时,给 20% agent 加
      G. Demo events:一个开放事件都没有时(且没加 --no-demo-events)
      H. Mock-LLM 预置信念:--mock-llm 且没 ArchetypeBelief 时预置分歧信念

    加 --skip-init 时全部跳过(要求 DB 已就绪)。
    """
    if args.skip_init:
        return _minimal_readiness_check()

    if args.reset:
        _reset_all_data()

    # B. schema
    db.create_all()
    SchedulerService.ensure_schema()
    # 补建性能索引 —— create_all() 不会给已存在的表加新的 composite index,
    # 我们对 price_history / trades 加了 (market_id, timestamp) 之类的
    # 复合索引来救 event_detail / leaderboard 在数据长起来之后的查询。
    ensure_perf_indexes()

    # C. population — archetypes
    n_arch = db.session.query(db.func.count(AgentArchetype.id)).scalar() or 0
    if n_arch < args.archetypes:
        if n_arch > 0:
            print(f"[live][setup] 已有 {n_arch} archetypes(< 目标 {args.archetypes}),"
                  f"沿用现有种群",  file=sys.stderr)
        else:
            print(f"[live][setup] 生成 {args.archetypes} archetypes(seed={args.sim_seed})...",
                  file=sys.stderr)
            m = PopulationService.generate_default_archetypes(
                count=args.archetypes, seed=args.sim_seed)
            print(f"[live][setup]   {m['archetypes']} 个 archetype 建好,耗时 {m['generation_time_s']}s",
                  file=sys.stderr)
    else:
        print(f"[live][setup] {n_arch} archetypes ✓", file=sys.stderr)

    # C. population — agents
    n_agents = db.session.query(db.func.count(Agent.id)).scalar() or 0
    if n_agents < args.agents:
        if n_agents > 0:
            print(f"[live][setup] 已有 {n_agents} agents(< 目标 {args.agents}),沿用",
                  file=sys.stderr)
        else:
            print(f"[live][setup] 生成 {args.agents} agents(seed={args.sim_seed},"
                  f"batch=1000)...", file=sys.stderr)
            m = PopulationService.generate_agents(
                count=args.agents, seed=args.sim_seed, batch_size=1000)
            print(f"[live][setup]   {m['agents']} 个 agent 建好,{m['batch_count']} batches,"
                  f"耗时 {m['generation_time_s']}s",  file=sys.stderr)
    else:
        print(f"[live][setup] {n_agents} agents ✓", file=sys.stderr)

    memory = _ensure_memory_stats()
    if memory["inserted"]:
        print(f"[live][setup] 预建 {memory['inserted']} 条 memory stats "
              f"({memory['agents']} agents)", file=sys.stderr)
    else:
        print(f"[live][setup] {memory['existing']} 条 memory stats ✓", file=sys.stderr)

    recovered = _recover_claimed_tasks()
    if recovered:
        print(f"[live][setup] 恢复 {recovered} 个上次中断遗留的 claimed 任务",
              file=sys.stderr)

    price_shed = _shed_price_backlog(max_per_event=int(args.price_backlog_per_event))
    if price_shed:
        print(f"[live][setup] price backlog fairness: shed {price_shed} 个旧 PRICE 任务",
              file=sys.stderr)

    if int(getattr(args, "max_pending_total", 0) or 0) > 0:
        shed = TriggerService.shed_load(keep=int(args.max_pending_total))
        if int(shed.get("shed") or 0):
            print(f"[live][setup] pending 队列 backpressure: "
                  f"{shed['pending']} → keep {shed['kept']} "
                  f"(shed {shed['shed']})", file=sys.stderr)

    # D. scheduler natural init
    n_sched = db.session.query(db.func.count(AgentScheduleState.agent_id)).scalar() or 0
    if n_sched == 0:
        print(f"[live][setup] 初始化自然唤醒调度(seed={args.sim_seed})...",
              file=sys.stderr)
        m = SchedulerService.initialize_natural(seed=args.sim_seed)
        print(f"[live][setup]   {m['scheduled_agents']} 个 agent 已排下一次自然唤醒,"
              f"耗时 {m['generation_time_s']}s", file=sys.stderr)
    else:
        print(f"[live][setup] {n_sched} 条自然唤醒调度 ✓", file=sys.stderr)

    # E. category expertise 索引
    n_expertise = db.session.query(db.func.count(AgentCategoryExpertise.id)).scalar() or 0
    if n_expertise == 0:
        print(f"[live][setup] 建 category 专家索引...", file=sys.stderr)
        m = CandidateService.build_category_expertise()
        print(f"[live][setup]   {m['expertise_rows']} 行专家索引 "
              f"({m['agents_scanned']} agents 扫描,耗时 {m['build_time_s']}s)",
              file=sys.stderr)
    else:
        print(f"[live][setup] {n_expertise} 行专家索引 ✓", file=sys.stderr)

    # G. demo events(先建,因为 F 会依赖开放事件的存在)
    open_events = [
        r[0] for r in db.session.query(Event.id)
        .join(Market, Market.event_id == Event.id)
        .filter(Market.status == MarketStatus.OPEN).distinct().all()
    ]
    if not open_events and not args.no_demo_events:
        want = max(1, int(args.events))
        picks = DEMO_EVENT_TEMPLATES[:want]
        print(f"[live][setup] 创建 {len(picks)} 个 demo 事件...", file=sys.stderr)
        for title, cat in picks:
            ev = Event(
                title=title, description="", category=cat,
                event_type=EventType.BINARY,
                close_time=datetime.utcnow() + timedelta(days=90),
                resolution_source="demo",
            )
            db.session.add(ev); db.session.flush()
            mk = Market(event_id=ev.id, status=MarketStatus.OPEN, liquidity_b=1000.0)
            db.session.add(mk); db.session.flush()
            db.session.add(EvidenceBundle(
                event_id=ev.id, version=1,
                supporting_evidence_ids=[], opposing_evidence_ids=[],
                neutral_evidence_ids=[], aggregate_impact=0.5,
                current_summary="demo seed"))
            open_events.append(ev.id)
            print(f"[live][setup]   [{cat}] {title}", file=sys.stderr)
        db.session.commit()
    elif not open_events:
        print("[live][setup][WARN] 一个开放事件都没有,而且加了 --no-demo-events。"
              "启动之后 evidence + worker 循环都无事可做。",
              file=sys.stderr)
    else:
        print(f"[live][setup] {len(open_events)} 个开放事件 ✓", file=sys.stderr)

    # F. watcher 订阅(需要开放事件已存在)
    n_interests = 0
    if open_events:
        n_interests = (
            db.session.query(db.func.count(AgentEventInterest.id))
            .filter(AgentEventInterest.event_id.in_(open_events)).scalar() or 0
        )
    if open_events and n_interests == 0:
        agent_ids = [r[0] for r in db.session.query(Agent.id).order_by(Agent.id).all()]
        watchers = agent_ids[::5]  # 20% 的 agent 当 watcher
        rows = [
            {"agent_id": aid, "event_id": eid, "role": ROLE_WATCHER, "weight": 0.5}
            for aid in watchers for eid in open_events
        ]
        if rows:
            db.session.bulk_insert_mappings(AgentEventInterest, rows)
            db.session.commit()
            print(f"[live][setup] 加 {len(rows)} 条 watcher 订阅 "
                  f"({len(watchers)} agents × {len(open_events)} events,20% 采样率)",
                  file=sys.stderr)
    elif open_events:
        print(f"[live][setup] {n_interests} 条 watcher 订阅 ✓", file=sys.stderr)

    # H. mock-LLM demo 用的分歧信念预置
    if args.mock_llm and open_events:
        n_beliefs = db.session.query(db.func.count(ArchetypeBelief.id)).scalar() or 0
        if n_beliefs == 0:
            print(f"[live][setup] --mock-llm:预置分歧 archetype 信念(每 archetype 不同)...",
                  file=sys.stderr)
            arch_ids = [r[0] for r in db.session.query(AgentArchetype.id).all()]
            rows = []
            now = datetime.utcnow()
            for arch_id in arch_ids:
                # 让不同 archetype 有不同信念:一些乐观 (>0.7),一些悲观 (<0.4),
                # 制造真实的市场分歧。用 archetype_id 的哈希做种子,确定性。
                base_p = 0.25 + 0.55 * ((arch_id * 13 + 7) % 11) / 10.0
                for eid in open_events:
                    rows.append({
                        "archetype_id": arch_id, "event_id": eid,
                        "evidence_bundle_version": 1,
                        "posterior_probability": round(base_p, 4),
                        "confidence": 0.8,
                        "model_tier": "BALANCED",
                        "reasoning_summary": "demo pre-seeded belief",
                        "prompt_version": "demo-v1",
                        "created_at": now, "updated_at": now,
                    })
            db.session.bulk_insert_mappings(ArchetypeBelief, rows)
            db.session.commit()
            print(f"[live][setup]   {len(rows)} 条预置信念 "
                  f"(base_p 在 archetype 间从 ~0.25 到 ~0.80 分布)",
                  file=sys.stderr)

    return True


def _minimal_readiness_check() -> bool:
    """--skip-init 模式下的最低就绪检查(不做任何写入)。"""
    n_arch = db.session.query(db.func.count(AgentArchetype.id)).scalar() or 0
    n_agents = db.session.query(db.func.count(Agent.id)).scalar() or 0
    n_events_open = (
        db.session.query(db.func.count(Market.id.distinct()))
        .filter(Market.status == MarketStatus.OPEN).scalar() or 0
    )
    if n_arch == 0 or n_agents == 0 or n_events_open == 0:
        print("=" * 66, file=sys.stderr)
        print(f"[live][BLOCKED] --skip-init 但 DB 未就绪:", file=sys.stderr)
        print(f"  archetypes={n_arch}  agents={n_agents}  open_markets={n_events_open}",
              file=sys.stderr)
        print("去掉 --skip-init 让 run_live.py 自动初始化,或先手动跑:",
              file=sys.stderr)
        print("  python init_db.py --reset && python manage_population.py "
              "generate-archetypes --count 20 --seed 42 && ...", file=sys.stderr)
        print("=" * 66, file=sys.stderr)
        return False
    print(f"[live][setup] --skip-init 就绪检查通过: "
          f"{n_arch} archetypes, {n_agents} agents, {n_events_open} open markets",
          file=sys.stderr)
    return True


def _prepare_web_schema(args) -> bool:
    """Do only the blocking work required for Flask pages to render.

    Full live setup can take seconds on a fresh/reset database because it
    generates agents, schedules wakeups, builds expertise indexes, and may
    create demo events. Before `run_live.py` absorbed the init scripts, the
    web process could start independently; keep that fast path by doing the
    minimal schema/index pass synchronously and moving population setup to a
    background bootstrap thread.
    """
    try:
        _enable_wal()
        if args.reset:
            _reset_all_data()
        db.create_all()
        SchedulerService.ensure_schema()
        ensure_perf_indexes()
        db.session.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        print(f"[live][setup] Web schema preparation failed: {exc}",
              file=sys.stderr)
        return False


def _copy_args(args, **overrides):
    data = vars(args).copy()
    data.update(overrides)
    return argparse.Namespace(**data)


def _ensure_memory_stats() -> Dict[str, int]:
    """Pre-create compact memory rows outside the worker hot path.

    `MemoryService.ensure_stats()` creates missing rows lazily. During live
    mode that means the first wake-up for many agents starts a SQLite write
    transaction before the policy/execution path finishes, which can make web
    reads wait behind the writer after the simulation has been running. Bulk
    inserting the missing rows during setup keeps the worker mostly doing
    short updates instead of schema-like first-touch writes.
    """
    existing = {r[0] for r in db.session.query(AgentMemoryStats.agent_id).all()}
    agents = db.session.query(Agent.id, Agent.initial_cash).all()
    rows = []
    for agent_id, initial_cash in agents:
        if agent_id in existing:
            continue
        cash = float(initial_cash or 0.0)
        rows.append({
            "agent_id": agent_id,
            "resolved_prediction_count": 0,
            "brier_running_sum": 0.0,
            "brier_average": 0.0,
            "log_loss_running_sum": 0.0,
            "log_loss_average": 0.0,
            "empirical_accuracy": 0.0,
            "average_confidence": 0.0,
            "overconfidence_score": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "portfolio_value": cash,
            "high_water_mark": max(0.0, cash),
            "current_drawdown": 0.0,
            "max_drawdown": 0.0,
            "win_streak": 0,
            "loss_streak": 0,
            "wake_up_count": 0,
            "wake_ups_without_trade": 0,
            "trade_count": 0,
            "profitable_trade_count": 0,
            "category_stats": {},
            "strategy_stats": {},
            "source_reliability_stats": {},
            "update_version": 0,
        })
    if rows:
        db.session.bulk_insert_mappings(AgentMemoryStats, rows)
        db.session.commit()
    return {"agents": len(agents), "inserted": len(rows), "existing": len(existing)}


def _recover_claimed_tasks() -> int:
    """Requeue tasks left CLAIMED by a previous interrupted live process."""
    n = db.session.query(WakeUpTask).filter(
        WakeUpTask.status == STATUS_CLAIMED
    ).update({WakeUpTask.status: STATUS_PENDING}, synchronize_session=False)
    if n:
        db.session.commit()
    return int(n or 0)


def _shed_price_backlog(max_per_event: int) -> int:
    """Keep only the earliest PRICE tasks per event in live mode.

    A single early trade can schedule many PRICE follow-ups on that same
    event. Those tasks have higher priority than NATURAL wakeups, so an old
    backlog makes the market look like only one event is alive. Keep a small
    per-event sample and shed the rest; fresh price changes can still enqueue
    more later, bounded by `--price-trigger-fanout`.
    """
    if max_per_event <= 0:
        return 0
    rows = db.session.execute(
        text(
            "SELECT id, event_id FROM wakeup_tasks "
            "WHERE status = :pending AND trigger_type = :price "
            "ORDER BY event_id ASC, scheduled_at ASC, wake_score DESC"
        ),
        {"pending": STATUS_PENDING, "price": TriggerType.PRICE.value},
    ).fetchall()
    kept_by_event: Dict[int, int] = {}
    shed_ids = []
    for task_id, event_id in rows:
        key = int(event_id or -1)
        kept = kept_by_event.get(key, 0)
        if kept < max_per_event:
            kept_by_event[key] = kept + 1
        else:
            shed_ids.append(task_id)
    if not shed_ids:
        return 0
    chunk_size = 900
    for i in range(0, len(shed_ids), chunk_size):
        chunk = shed_ids[i:i + chunk_size]
        db.session.query(WakeUpTask).filter(WakeUpTask.id.in_(chunk)).update(
            {WakeUpTask.status: STATUS_SHED}, synchronize_session=False
        )
    db.session.commit()
    return len(shed_ids)


# ---------------------------------------------------------------------
# 后台线程:maintenance / 长跑保养
# ---------------------------------------------------------------------

def _maintenance_loop(interval_s: float, terminal_keep_hours: float,
                      claimed_timeout_minutes: float,
                      price_history_keep_per_market: int,
                      agent_decision_keep_rows: int,
                      evidence_bundle_keep_per_event: int,
                      information_event_keep_per_event: int):
    """Periodic bounded cleanup for long-running live sessions.

    This keeps the hot demo database from growing without bound while
    preserving business-critical tables (`trades`, `positions`, `agents`,
    `markets`, `events`). The UI only renders recent chart/evidence slices,
    and `AgentDecision` rows are debug/audit telemetry rather than scoring
    truth, so bounded retention is safe for live mode.
    """
    while not _STOP.is_set():
        try:
            with app.app_context():
                with _DB_WRITE_LOCK:
                    stats = _maintenance_once(
                        terminal_keep_hours=terminal_keep_hours,
                        claimed_timeout_minutes=claimed_timeout_minutes,
                        price_history_keep_per_market=price_history_keep_per_market,
                        agent_decision_keep_rows=agent_decision_keep_rows,
                        evidence_bundle_keep_per_event=evidence_bundle_keep_per_event,
                        information_event_keep_per_event=information_event_keep_per_event,
                    )
                changed = {k: v for k, v in stats.items() if int(v or 0) > 0}
                if changed:
                    print(f"[live][maintenance] {changed}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[live][maintenance] 循环错误: {exc}", file=sys.stderr)
        finally:
            try:
                db.session.remove()
            except Exception:
                pass
        _STOP.wait(max(30.0, float(interval_s)))


def _maintenance_once(*, terminal_keep_hours: float,
                      claimed_timeout_minutes: float,
                      price_history_keep_per_market: int,
                      agent_decision_keep_rows: int,
                      evidence_bundle_keep_per_event: int,
                      information_event_keep_per_event: int) -> Dict[str, int]:
    """Run one idempotent maintenance pass; returns row counts."""
    stats = {
        "claimed_requeued": 0,
        "terminal_tasks_deleted": 0,
        "price_history_deleted": 0,
        "event_analysis_deleted": 0,
        "agent_decisions_deleted": 0,
        "old_operation_reasons_deleted": 0,
        "orphan_event_analysis_deleted": 0,
        "evidence_deltas_deleted": 0,
        "evidence_bundles_deleted": 0,
        "information_events_deleted": 0,
        "source_content_deleted": 0,
    }

    # 1) A crash/KeyboardInterrupt can leave tasks CLAIMED forever. Requeue
    # stale claims so the worker can retry them later.
    claimed_timeout = datetime.utcnow() - timedelta(
        minutes=max(1.0, float(claimed_timeout_minutes))
    )
    res = db.session.execute(text(
        "UPDATE wakeup_tasks "
        "SET status = :pending, retry_count = retry_count + 1, "
        "last_error = :err, updated_at = CURRENT_TIMESTAMP "
        "WHERE status = :claimed AND updated_at < :cutoff"
    ), {
        "pending": STATUS_PENDING,
        "claimed": STATUS_CLAIMED,
        "err": "maintenance requeued stale claimed task",
        "cutoff": claimed_timeout,
    })
    stats["claimed_requeued"] = int(res.rowcount or 0)

    # 2) Terminal WakeUpTask rows are operational logs. Keep recent ones for
    # debugging, delete older ones so dedup/status indexes stay small.
    terminal_cutoff = datetime.utcnow() - timedelta(
        hours=max(0.1, float(terminal_keep_hours))
    )
    res = db.session.execute(text(
        "DELETE FROM wakeup_tasks "
        "WHERE status IN ('done', 'shed', 'failed') AND updated_at < :cutoff"
    ), {"cutoff": terminal_cutoff})
    stats["terminal_tasks_deleted"] = int(res.rowcount or 0)

    # 3) The web chart only renders the latest 500 points. Keep a generous
    # tail per market and delete older chart snapshots.
    keep_prices = max(0, int(price_history_keep_per_market))
    if keep_prices > 0:
        market_ids = [row[0] for row in db.session.execute(
            text("SELECT id FROM markets")
        ).fetchall()]
        for market_id in market_ids:
            res = db.session.execute(text(
                "DELETE FROM price_history "
                "WHERE market_id = :market_id AND id NOT IN ("
                "  SELECT id FROM price_history "
                "  WHERE market_id = :market_id "
                "  ORDER BY timestamp DESC, id DESC LIMIT :keep"
                ")"
            ), {"market_id": market_id, "keep": keep_prices})
            stats["price_history_deleted"] += int(res.rowcount or 0)

    # 4) Remove legacy operation-reason rows from older builds. They created
    # duplicate lines like "agent bought NO" + "agent's event analysis".
    res = db.session.execute(text(
        "DELETE FROM evidence WHERE query = 'agent_operation_reason'"
    ))
    stats["old_operation_reasons_deleted"] = int(res.rowcount or 0)

    # 5) Remove legacy LLM-generated event-analysis rows. Recent Evidence is
    # now reserved for retrieved/search evidence from InformationEvent only.
    res = db.session.execute(text(
        "DELETE FROM evidence WHERE query = 'agent_event_analysis'"
    ))
    stats["event_analysis_deleted"] = int(res.rowcount or 0)

    # 6) AgentDecision rows are high-volume telemetry. Trades/positions remain
    # the source of truth for leaderboard and event history.
    keep_decisions = max(0, int(agent_decision_keep_rows))
    if keep_decisions > 0:
        res = db.session.execute(text(
            "DELETE FROM agent_decisions WHERE id NOT IN ("
            "  SELECT id FROM agent_decisions "
            "  ORDER BY created_at DESC, id DESC LIMIT :keep"
            ")"
        ), {"keep": keep_decisions})
        stats["agent_decisions_deleted"] = int(res.rowcount or 0)

    # 7) InformationEvent/SourceContent retention for long-running sessions.
    # The UI renders only the latest 30, but without DB retention a month-long
    # run can accumulate many unique retrieved sources. Keep a generous recent
    # tail per event and then remove SourceContent rows no event references.
    keep_info_events = max(0, int(information_event_keep_per_event))
    if keep_info_events > 0:
        event_ids = [row[0] for row in db.session.execute(
            text("SELECT DISTINCT event_id FROM information_events")
        ).fetchall()]
        for event_id in event_ids:
            keep_ids = [row[0] for row in db.session.execute(text(
                "SELECT id FROM information_events "
                "WHERE event_id = :event_id "
                "ORDER BY retrieved_at DESC, id DESC LIMIT :keep"
            ), {"event_id": event_id, "keep": keep_info_events}).fetchall()]
            if not keep_ids:
                continue
            keep_csv = ",".join(str(int(x)) for x in keep_ids)
            res = db.session.execute(text(
                "DELETE FROM information_events "
                "WHERE event_id = :event_id "
                f"AND id NOT IN ({keep_csv})"
            ), {"event_id": event_id})
            stats["information_events_deleted"] += int(res.rowcount or 0)

        res = db.session.execute(text(
            "DELETE FROM source_content "
            "WHERE id NOT IN ("
            "  SELECT DISTINCT source_content_id FROM information_events"
            ")"
        ))
        stats["source_content_deleted"] = int(res.rowcount or 0)

    # 8) EvidenceBundle/EvidenceDelta retention for month-long runs. Keep the
    # latest N bundle versions per event and delete their old deltas first.
    keep_bundles = max(0, int(evidence_bundle_keep_per_event))
    if keep_bundles > 0:
        event_ids = [row[0] for row in db.session.execute(
            text("SELECT DISTINCT event_id FROM evidence_bundles")
        ).fetchall()]
        for event_id in event_ids:
            keep_ids = [row[0] for row in db.session.execute(text(
                "SELECT id FROM evidence_bundles "
                "WHERE event_id = :event_id "
                "ORDER BY version DESC LIMIT :keep"
            ), {"event_id": event_id, "keep": keep_bundles}).fetchall()]
            if not keep_ids:
                continue
            old_ids = [row[0] for row in db.session.execute(text(
                "SELECT id FROM evidence_bundles "
                "WHERE event_id = :event_id AND id NOT IN ("
                + ",".join(str(int(x)) for x in keep_ids) + ")"
            ), {"event_id": event_id}).fetchall()]
            if not old_ids:
                continue
            old_csv = ",".join(str(int(x)) for x in old_ids)
            db.session.execute(text(
                "UPDATE evidence_bundles SET previous_bundle_id = NULL "
                f"WHERE previous_bundle_id IN ({old_csv})"
            ))
            res = db.session.execute(text(
                f"DELETE FROM evidence_deltas WHERE bundle_id IN ({old_csv})"
            ))
            stats["evidence_deltas_deleted"] += int(res.rowcount or 0)
            res = db.session.execute(text(
                f"DELETE FROM evidence_bundles WHERE id IN ({old_csv})"
            ))
            stats["evidence_bundles_deleted"] += int(res.rowcount or 0)

    db.session.commit()
    db.session.expunge_all()
    return stats

def _make_live_threads(args) -> List[threading.Thread]:
    threads = [
        threading.Thread(target=_tick_loop, args=(args.tick, args.speed),
                         name="tick", daemon=True),
        threading.Thread(target=_natural_bridge_loop,
                         args=(args.natural, args.natural_limit,
                               args.max_pending_total),
                         name="natural", daemon=True),
    ]
    if not args.no_worker:
        threads.append(threading.Thread(
            target=_worker_loop,
            args=(args.worker, args.worker_limit,
                  args.mock_llm, args.sim_seed,
                  args.worker_micro_batch,
                  args.max_pending_per_market,
                  args.max_pending_total,
                  args.price_trigger_fanout,
                  not args.priority_queue),
            name="worker", daemon=True))
    if not args.no_evidence:
        threads.append(threading.Thread(
            target=_evidence_loop,
            args=(args.refresh, args.mock_llm,
                  args.evidence_trigger_budget, args.max_pending_total),
            name="evidence", daemon=True))
    if not args.no_maintenance:
        threads.append(threading.Thread(
            target=_maintenance_loop,
            args=(
                args.maintenance_interval,
                args.terminal_task_keep_hours,
                args.claimed_timeout_minutes,
                args.price_history_keep_per_market,
                args.agent_decision_keep_rows,
                args.evidence_bundle_keep_per_event,
                args.information_event_keep_per_event,
            ),
            name="maintenance", daemon=True))
    return threads


def _start_live_threads(args, threads: List[threading.Thread],
                        threads_lock: threading.Lock):
    new_threads = _make_live_threads(args)
    with threads_lock:
        threads.extend(new_threads)
    for thread in new_threads:
        thread.start()


def _start_bootstrap_thread(args, threads: List[threading.Thread],
                            threads_lock: threading.Lock):
    """Run full init off the request path, then start live loops."""
    app.config["LIVE_INIT_RUNNING"] = True
    app.config["LIVE_INIT_MESSAGE"] = (
        "Agents, schedules, watchers, and demo events are being prepared."
    )

    def _bootstrap_then_start():
        ok = False
        try:
            with app.app_context():
                ok = _bootstrap_or_init(args)
            if ok and not _STOP.is_set():
                app.config["LIVE_INIT_MESSAGE"] = "Live setup complete."
                _start_live_threads(args, threads, threads_lock)
                print("[live][setup] 后台初始化完成,后台循环已启动",
                      file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            app.config["LIVE_INIT_MESSAGE"] = f"Live setup failed: {exc}"
            print(f"[live][setup] 后台初始化失败: {exc}", file=sys.stderr)
        finally:
            if not ok:
                print("[live][setup] 后台初始化未完成,后台循环未启动",
                      file=sys.stderr)
            app.config["LIVE_INIT_RUNNING"] = False

    thread = threading.Thread(target=_bootstrap_then_start,
                              name="bootstrap", daemon=True)
    with threads_lock:
        threads.append(thread)
    thread.start()


# ---------------------------------------------------------------------
# 后台线程 A:tick clock
# ---------------------------------------------------------------------

def _tick_loop(interval_s: float, speed: float = 1.0):
    """每 interval_s 秒把虚拟时钟推进 interval_s * speed 秒。

    speed=1 时是"虚拟时钟锚定 wall-clock"(1x 实时)。demo/加速场景把
    speed 调大即可 —— agent 的 base_wakeup_rate 是"每天",1x 下需要
    真实 hours 才有第一波自然唤醒,页面看起来"打开就没交易",这是
    速率量纲问题而不是逻辑问题。用 --speed 60 就是 60x 加速。
    """
    delta = float(interval_s) * float(speed)
    while not _STOP.is_set():
        try:
            with app.app_context():
                with _DB_WRITE_LOCK:
                    SchedulerService.advance_time(seconds=delta)
        except Exception as exc:  # noqa: BLE001
            print(f"[live][tick] {exc}", file=sys.stderr)
        _STOP.wait(interval_s)


# ---------------------------------------------------------------------
# 后台线程 B:evidence refresh + information trigger
# ---------------------------------------------------------------------

def _natural_bridge_loop(interval_s: float, batch_limit: int,
                         max_pending_total: int):
    """把 SchedulerService.due() 的自然唤醒桥接成 WakeUpTask 行。

    没有这一步的话,自然唤醒只更新 `agent_schedule_state.next_natural_wakeup_at`
    —— Worker 只读 `wakeup_tasks`,无事可做。原始设计里两条路没接;
    对实时演示来说这就是"打开就没交易"的根源。

    这个循环:每 interval_s 秒把到期的自然唤醒 agent 变成 low-priority
    `NATURAL` WakeUpTask 行,针对 agent 有 Position 或 AgentEventInterest
    的每个开放 event 各建一条。dedup_key 唯一 —— 一个 agent 在同一
    time_bucket 里同一 event 上不会重复入队。
    """
    last_backpressure_log = 0.0
    while not _STOP.is_set():
        sleep_after_backpressure = False
        try:
            with app.app_context():
                with _DB_WRITE_LOCK:
                    pending_total = (
                        db.session.query(db.func.count(WakeUpTask.id))
                        .filter(WakeUpTask.status == STATUS_PENDING)
                        .scalar() or 0
                    )
                    if pending_total >= max(1, int(max_pending_total)):
                        now_wall = time.monotonic()
                        if now_wall - last_backpressure_log > 10.0:
                            print(f"[live][natural] backpressure: pending={pending_total} "
                                  f">= {max_pending_total},暂停自然唤醒入队",
                                  file=sys.stderr)
                            last_backpressure_log = now_wall
                        sleep_after_backpressure = True
                    if not sleep_after_backpressure:
                        due_items = SchedulerService.due(limit=batch_limit)
                        if due_items:
                            agent_ids = [w["agent_id"] for w in due_items]
                            n = _bridge_natural_to_tasks(agent_ids, SchedulerService.now())
                            if n:
                                print(f"[live][natural] {len(due_items)} 自然唤醒 → "
                                      f"{n} 个 WakeUpTask 已入队")
        except Exception as exc:  # noqa: BLE001
            print(f"[live][natural] 循环错误: {exc}", file=sys.stderr)
        if sleep_after_backpressure:
            _STOP.wait(interval_s)
            continue
        _STOP.wait(interval_s)


def _bridge_natural_to_tasks(agent_ids: List[int], now: float) -> int:
    """为每个自然唤醒 agent 创建 (agent × event) 的 WakeUpTask 行。

    Events 来源:该 agent 的 Position(持仓)+ AgentEventInterest(订阅)。
    dedup 靠 UNIQUE 约束:已存在的 (agent, event, bucket) key 会被
    SQLite `INSERT OR IGNORE` 静默丢弃。
    """
    if not agent_ids:
        return 0
    bucket = time_bucket(now)
    priority = TRIGGER_PRIORITY[TriggerType.NATURAL.value]

    # 1. Position → 开放市场上的 event
    position_rows = (
        db.session.query(Position.agent_id, Position.market_id, Market.event_id)
        .join(Market, Market.id == Position.market_id)
        .filter(Position.agent_id.in_(agent_ids),
                Market.status == MarketStatus.OPEN).all()
    )
    # 2. AgentEventInterest → 开放市场的 event
    interest_rows = (
        db.session.query(
            AgentEventInterest.agent_id, Market.id.label("market_id"),
            AgentEventInterest.event_id,
        )
        .join(Market, Market.event_id == AgentEventInterest.event_id)
        .filter(AgentEventInterest.agent_id.in_(agent_ids),
                Market.status == MarketStatus.OPEN).all()
    )

    triples = {(aid, eid, mid) for aid, mid, eid in position_rows}
    triples.update((aid, eid, mid) for aid, mid, eid in interest_rows)
    if not triples:
        return 0

    # 3. 组装候选行,按 dedup_key 折叠。dedup_key 里没有 market_id ——
    # CATEGORICAL 事件下同一 (agent, event) 可能有多条 triples (Position
    # 在 market A + AgentEventInterest 命中 market B),两条会撞同一
    # `{aid}:{eid}:{bucket}:nat` key。这里只保留每个 key 的第一条,
    # 否则 bulk_insert 会因批内重复整批 UNIQUE 冲突回滚。
    rows_by_key: Dict[str, Dict[str, Any]] = {}
    for aid, eid, mid in triples:
        dk = make_dedup_key(aid, eid, bucket) + ":nat"
        if dk in rows_by_key:
            continue
        rows_by_key[dk] = {
            "agent_id": aid, "event_id": eid, "market_id": mid,
            "trigger_type": TriggerType.NATURAL.value,
            "priority": priority, "tier": "delayed",
            "status": STATUS_PENDING,
            "scheduled_at": now, "time_bucket": bucket, "wake_score": 0.3,
            "relevance": 0.3, "information_impact": 0.0,
            "position_relevance": 0.5, "expertise": 0.0, "portfolio_risk": 0.0,
            "cascade_depth": 0, "dedup_key": dk,
            "wake_reasons": TriggerType.NATURAL.value,
            "retry_count": 0,
        }

    # 4. 过滤已存在的 dedup_key(避免 IntegrityError 打断整批)
    keys = list(rows_by_key.keys())
    existing = {
        r[0] for r in db.session.query(WakeUpTask.dedup_key)
        .filter(WakeUpTask.dedup_key.in_(keys)).all()
    }
    fresh = [rows_by_key[k] for k in keys if k not in existing]
    if not fresh:
        return 0
    try:
        db.session.bulk_insert_mappings(WakeUpTask, fresh)
        db.session.commit()
        inserted = len(fresh)
    except IntegrityError:
        # 兜底:上面的过滤是 SELECT-then-INSERT,不在同一事务里,evidence
        # loop / 其它进程可能在中间插同 key。回滚后逐行 INSERT OR IGNORE
        # 兼容 SQLite;失败的直接跳过。
        db.session.rollback()
        inserted = 0
        stmt = text(
            "INSERT OR IGNORE INTO wakeup_tasks ("
            "agent_id, event_id, market_id, trigger_type, priority, tier, "
            "status, scheduled_at, time_bucket, relevance, information_impact, "
            "position_relevance, expertise, portfolio_risk, wake_score, "
            "wake_reasons, dedup_key, cascade_depth, retry_count, "
            "created_at, updated_at"
            ") VALUES ("
            ":agent_id, :event_id, :market_id, :trigger_type, :priority, :tier, "
            ":status, :scheduled_at, :time_bucket, :relevance, :information_impact, "
            ":position_relevance, :expertise, :portfolio_risk, :wake_score, "
            ":wake_reasons, :dedup_key, :cascade_depth, :retry_count, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )
        for r in fresh:
            try:
                res = db.session.execute(stmt, r)
                inserted += int(res.rowcount or 0)
            except IntegrityError:
                db.session.rollback()
        db.session.commit()
    db.session.expunge_all()
    return inserted


# ---------------------------------------------------------------------
# 后台线程 C:evidence refresh + information trigger
# ---------------------------------------------------------------------

def _evidence_loop(interval_s: float, mock_llm: bool,
                   trigger_budget: int, max_pending_total: int):
    """每 interval_s 秒:遍历开放事件 → 调 EvidenceService.refresh。

    若某个事件切出了新 EvidenceBundle 版本(说明证据面变了),
    额外发一个 INFORMATION 触发,把对该事件感兴趣的 agent 唤醒到
    WakeUpTask 队列。这样 agent 才有机会对新新闻做交易。
    """
    provider = _try_tavily()
    if provider is None:
        print("[live][evidence] 已禁用(无 Tavily key)", file=sys.stderr)
        return
    llm, router = _get_llm(mock_llm)
    while not _STOP.is_set():
        try:
            with app.app_context():
                event_ids = [
                    r[0] for r in db.session.query(Event.id)
                    .join(Market, Market.event_id == Event.id)
                    .filter(Market.status == MarketStatus.OPEN)
                    .distinct().all()
                ]
                if not event_ids:
                    _STOP.wait(interval_s)
                    continue
                svc = EvidenceService(search_provider=provider, router=router)
                for eid in event_ids:
                    if _STOP.is_set():
                        break
                    try:
                        r = svc.refresh(eid)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[live][evidence] event {eid} refresh 失败: {exc}",
                              file=sys.stderr)
                        continue
                    if not r.get("new_version"):
                        continue
                    # 新证据 → 触发 INFORMATION 唤醒。这里必须尊重全局
                    # pending 上限；否则 3 个 demo event × 每个 200 agent
                    # 会瞬间入队 600 个任务，而默认 worker 还没来得及
                    # 消化，natural bridge 就会长期显示 backpressure。
                    impact = float(r.get("aggregate_impact") or 0.5)
                    added = int(r.get("added") or 0)
                    try:
                        pending_total = (
                            db.session.query(db.func.count(WakeUpTask.id))
                            .filter(WakeUpTask.status == STATUS_PENDING)
                            .scalar() or 0
                        )
                        remaining = max(0, int(max_pending_total) - int(pending_total))
                        per_event_budget = min(
                            max(0, int(trigger_budget)),
                            remaining,
                        )
                        if per_event_budget <= 0:
                            print(f"[live][evidence] event {eid}: "
                                  f"v{r['version']} (+{added} sources, "
                                  f"impact={impact:.2f}) → pending={pending_total} "
                                  f">= {max_pending_total},跳过本轮唤醒",
                                  file=sys.stderr)
                            continue
                        trig = TriggerService.information_event(
                            event_id=eid, information_impact=impact,
                            relevance=0.7,
                            budget=EventBudget(
                                max_urgent=max(1, min(10, per_event_budget)),
                                max_normal=per_event_budget,
                                max_delayed=per_event_budget,
                                total_budget=per_event_budget,
                            ),
                        )
                        n = int(trig.get("scheduled") or 0)
                        print(f"[live][evidence] event {eid}: "
                              f"v{r['version']} (+{added} sources, "
                              f"impact={impact:.2f}) → {n} agent 唤醒")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[live][evidence] event {eid} trigger 失败: {exc}",
                              file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[live][evidence] 循环错误: {exc}", file=sys.stderr)
        _STOP.wait(interval_s)


# ---------------------------------------------------------------------
# 后台线程 C:worker drain
# ---------------------------------------------------------------------

def _worker_loop(interval_s: float, batch_limit: int, mock_llm: bool,
                 sim_seed: int, micro_batch: int,
                 max_pending_per_market: int,
                 max_pending_total: int,
                 price_trigger_fanout: int,
                 fair_queue: bool):
    """每 interval_s 秒跑一次 AgentWakeupProcessor.run_once。"""
    llm, router = _get_llm(mock_llm)
    while not _STOP.is_set():
        t0 = time.monotonic()
        try:
            with app.app_context():
                with _DB_WRITE_LOCK:
                    proc = AgentWakeupProcessor(
                        config=ProcessorConfig(
                            sim_seed=sim_seed,
                            micro_batch=max(1, int(micro_batch)),
                            max_pending_per_market=max(1, int(max_pending_per_market)),
                            max_pending_total=max(1, int(max_pending_total)),
                            price_trigger_max_per_trade=max(0, int(price_trigger_fanout)),
                            fair_queue=bool(fair_queue),
                        ),
                        llm_client=llm, router=router,
                    )
                    m = proc.run_once(limit=batch_limit)
                    shed = {"shed": 0}
                    if max_pending_total > 0 and (
                        m.price_triggers_published or m.claimed
                    ):
                        shed = TriggerService.shed_load(keep=int(max_pending_total))
                if m.claimed:
                    elapsed = time.monotonic() - t0
                    tiers = dict(m.tier_distribution) if m.tier_distribution else {}
                    tier_note = f" tier={tiers}" if tiers else ""
                    shed_note = (
                        f" shed={shed.get('shed')}"
                        if int(shed.get("shed") or 0) else ""
                    )
                    print(f"[live][worker] claimed={m.claimed} "
                          f"completed={m.completed} trades={m.trades} "
                          f"holds={m.holds} llm={m.belief_llm_updates}"
                          f"{tier_note}{shed_note}")
                    if elapsed > max(2.0, interval_s * 0.8):
                        print(f"[live][worker][slow] 本轮耗时 {elapsed:.1f}s; "
                              "若 Web 变慢,可调小 --worker-limit 或加大 --worker",
                              file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[live][worker] 循环错误: {exc}", file=sys.stderr)
        _STOP.wait(interval_s)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def _announce_when_ready(host: str, port: int, timeout: float = 15.0):
    """开个后台线程,轮询端口,真的能连上了才打印 URL。

    Werkzeug 的 `run_simple` 从主线程 bind + listen 通常在 1–2 秒内
    完成,但用户看到 stderr 里的 URL 就手快去打开,可能撞上"服务
    还在启动"—— 浏览器等 TCP accept,体感就是"打开就卡"。这里等
    到真的能 `connect` 成功再刷 URL 出来。超时则不打印(说明起
    失败了,Werkzeug 自己的报错会先出)。
    """
    import socket

    def _wait_and_print():
        deadline = time.monotonic() + timeout
        # 用户在本机浏览器打开,总是要连回本地地址;host="0.0.0.0" 或
        # ""(any) 时我们探 127.0.0.1 就够了。
        probe_host = host if host not in ("", "0.0.0.0") else "127.0.0.1"
        while time.monotonic() < deadline and not _STOP.is_set():
            try:
                with socket.create_connection((probe_host, port), timeout=0.5):
                    display_host = "localhost" if probe_host == "127.0.0.1" else probe_host
                    print(f"[live] ✓ Web 已就绪:  http://{display_host}:{port}",
                          file=sys.stderr)
                    return
            except OSError:
                time.sleep(0.1)

    threading.Thread(target=_wait_and_print, name="ready-probe", daemon=True).start()


def _serve_web(host: str, port: int):
    """启动 web 服务。有 waitress 优先用 waitress,没有回落到 Werkzeug。

    Werkzeug 的 `app.run(threaded=True)` 是开发服务器,keep-alive 处理
    在个别版本 + 特定客户端组合下会 stutter (第二个复用连接的请求
    要 200+ ms 才响应),SSH 隧道下体感放大到"切页也慢"。waitress
    是纯 Python 的生产级 WSGI 服务器,keep-alive / 线程池 / 响应
    缓冲都在,依赖只多这一个包。启动语义(阻塞主线程 + SIGINT)
    跟 Werkzeug 保持一致。

    `use_reloader=False` 无所谓,因为我们只走单进程 —— reloader
    会 fork 让后台四个线程各出现两次,数据库并发直接炸,不能开。
    """
    try:
        from waitress import serve
        # threads=8: 每个 HTTP 请求一个线程,和 Werkzeug threaded=True
        # 一样,让浏览器 6 个并发子资源真的并发处理。
        # ident: 服务端 Server header,不透露版本。
        # channel_timeout: 空闲连接多久回收 —— 300s 让 keep-alive 有
        # 足够长的窗口,SSH 隧道下切页复用同一 TCP 连接。
        print("[live] 使用 waitress 作为 WSGI 服务器(生产级)", file=sys.stderr)
        serve(
            app, host=host, port=port,
            threads=8, ident="ForecastArena",
            channel_timeout=300,
        )
    except ImportError:
        print("[live] waitress 未安装 —— 回落到 Werkzeug 开发服务器"
              "(pip install waitress 可换到更快的路径)", file=sys.stderr)
        # Flask 阻塞主线程。use_reloader=False 关键 —— 否则 Flask 会
        # fork,后台线程复制两份,数据库并发爆炸。
        # threaded=True 让 Werkzeug 每个请求一个线程。
        # SIGINT 由 Werkzeug 自己处理 —— 我们不覆盖它。
        app.run(host=host, port=port, debug=False,
                use_reloader=False, threaded=True)


def main():
    ap = argparse.ArgumentParser(
        description="ForecastArena 实时模式(wall-clock + 真实证据 + web)")
    ap.add_argument("--tick", type=float, default=5.0,
                    help="虚拟时钟推进间隔(秒),默认 5")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="虚拟时钟相对 wall-clock 的加速倍率,默认 1(实时);"
                         "--mock-llm demo 想快点看到交易可以 --speed 60 或更大")
    ap.add_argument("--natural", type=float, default=10.0,
                    help="自然唤醒 → WakeUpTask 桥接间隔(秒),默认 10")
    ap.add_argument("--natural-limit", type=int, default=10,
                    help="每次桥接最多消费的自然唤醒数,默认 10(Web 友好)")
    ap.add_argument("--max-pending-total", type=int, default=1000,
                    help="全局 pending WakeUpTask 上限;达到后暂停自然唤醒/价格触发入队,默认 1000")
    ap.add_argument("--refresh", type=float, default=60.0,
                    help="证据刷新间隔(秒),默认 60")
    ap.add_argument("--worker", type=float, default=3.0,
                    help="Worker 处理间隔(秒),默认 3")
    ap.add_argument("--worker-limit", type=int, default=10,
                    help="每次 Worker 处理的最大任务数,默认 10;页面卡顿可调低,队列积压可调高")
    ap.add_argument("--worker-micro-batch", type=int, default=5,
                    help="Worker 内部分市场微批大小,默认 5")
    ap.add_argument("--evidence-trigger-budget", type=int, default=50,
                    help="每个事件每次证据刷新最多唤醒多少 agent,默认 50")
    ap.add_argument("--max-pending-per-market", type=int, default=500,
                    help="价格触发每个 market 的 pending 上限,默认 500")
    ap.add_argument("--price-trigger-fanout", type=int, default=3,
                    help="每笔触发价格变化的交易最多唤醒多少个 PRICE agent,默认 3")
    ap.add_argument("--price-backlog-per-event", type=int, default=20,
                    help="启动时每个 event 最多保留多少个旧 PRICE pending,默认 20")
    ap.add_argument("--priority-queue", action="store_true",
                    help="worker 严格按 priority 取任务;默认按 scheduled_at 公平轮转")
    ap.add_argument("--no-worker", action="store_true",
                    help="只推进时钟/证据/自然入队,不执行 agent worker")
    ap.add_argument("--sim-seed", type=int, default=42,
                    help="确定性种子,影响 agent 个体化,默认 42")
    ap.add_argument("--mock-llm", action="store_true",
                    help="离线 stub 代替真实 LLM(不产生付费调用),同时预置分歧信念")
    ap.add_argument("--no-evidence", action="store_true",
                    help="不刷新证据(即使 TAVILY_API_KEY 已设置)")
    ap.add_argument("--no-maintenance", action="store_true",
                    help="关闭长跑保养线程(不建议;DB 会持续变大)")
    ap.add_argument("--maintenance-interval", type=float, default=300.0,
                    help="长跑保养间隔秒数,默认 300")
    ap.add_argument("--terminal-task-keep-hours", type=float, default=6.0,
                    help="done/shed/failed WakeUpTask 保留小时数,默认 6")
    ap.add_argument("--claimed-timeout-minutes", type=float, default=10.0,
                    help="CLAIMED 任务超过多久视为卡死并重排,默认 10 分钟")
    ap.add_argument("--price-history-keep-per-market", type=int, default=5000,
                    help="每个 market 保留的 PriceHistory 最近点数,默认 5000")
    ap.add_argument("--agent-decision-keep-rows", type=int, default=200000,
                    help="保留最近多少 AgentDecision 调试行,默认 200000")
    ap.add_argument("--evidence-bundle-keep-per-event", type=int, default=200,
                    help="每个 event 保留的 EvidenceBundle/EvidenceDelta 最近版本数,默认 200")
    ap.add_argument("--information-event-keep-per-event", type=int, default=1000,
                    help="每个 event 保留的检索证据 InformationEvent 最近条数,默认 1000")
    # 一键初始化参数(--skip-init 时全部忽略)
    ap.add_argument("--agents", type=int, default=500,
                    help="目标 agent 数(空库时才建),默认 500")
    ap.add_argument("--archetypes", type=int, default=20,
                    help="目标 archetype 数(空库时才建),默认 20")
    ap.add_argument("--events", type=int, default=3,
                    help="没有开放事件时建几个 demo 事件,默认 3(最多 5)")
    ap.add_argument("--reset", action="store_true",
                    help="清空所有数据表后重建(schema 保留)")
    ap.add_argument("--skip-init", action="store_true",
                    help="不做自动初始化,只做最低就绪检查(要求 DB 已就绪)")
    ap.add_argument("--no-demo-events", action="store_true",
                    help="不自动创建 demo 事件")
    ap.add_argument("--port", type=int, default=None,
                    help="Web 端口,默认读 FLASK_PORT 或 6006")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-web", action="store_true",
                    help="只跑后台循环,不启动 Flask(适合无头运行)")
    args = ap.parse_args()

    threads: List[threading.Thread] = []
    threads_lock = threading.Lock()

    if args.no_web:
        with app.app_context():
            _enable_wal()
            if not _bootstrap_or_init(args):
                return 2
        _start_live_threads(args, threads, threads_lock)
    else:
        if args.skip_init:
            with app.app_context():
                _enable_wal()
                if not _minimal_readiness_check():
                    return 2
            _start_live_threads(args, threads, threads_lock)
        else:
            with app.app_context():
                if not _prepare_web_schema(args):
                    return 2
            # The web server can bind now. The heavier population/scheduler
            # setup continues in the background, then starts tick/natural/
            # worker/evidence loops once the DB is fully ready.
            init_args = _copy_args(args, reset=False)
            _start_bootstrap_thread(init_args, threads, threads_lock)

    port = args.port or int(os.getenv("FLASK_PORT", "6006"))
    print("=" * 66, file=sys.stderr)
    print(f"[live] 实时模式启动", file=sys.stderr)
    print(f"[live]   tick={args.tick}s(speed={args.speed}x)  "
          f"natural_bridge={args.natural}s  "
          f"refresh={args.refresh}s  worker={args.worker}s(limit={args.worker_limit})",
          file=sys.stderr)
    print(f"[live]   sim_seed={args.sim_seed}  mock_llm={args.mock_llm}  "
          f"evidence={'off' if args.no_evidence else 'on'}", file=sys.stderr)
    print(f"[live] Ctrl-C 停止", file=sys.stderr)
    print("=" * 66, file=sys.stderr)

    try:
        if args.no_web:
            # 无头模式:主线程等停止信号。Event.wait() 会被 SIGINT 唤醒
            # (KeyboardInterrupt 从默认 handler 抛出),干净退出。
            while not _STOP.wait(1.0):
                pass
        else:
            # Web 就绪的公告放在一个后台线程里 —— 等到 TCP 端口真的接受
            # 连接了再打印,免得用户看到 URL 就点、但 Werkzeug 还没
            # bind 完(那种"第一次打开就特别慢"的假象)。
            _announce_when_ready(args.host, port)
            _serve_web(args.host, port)
    except KeyboardInterrupt:
        pass
    finally:
        _STOP.set()
        # daemon 线程随主线程死;这里 join 只是等它们打印完 goodbye。
        with threads_lock:
            join_threads = list(threads)
        for t in join_threads:
            t.join(timeout=2.0)
        print("[live] 已停止", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
