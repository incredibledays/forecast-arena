"""ForecastArena Flask app.

Views and one JSON dev endpoint over the seeded database. Trading is
executed via the CLI runner (`run_agents.py`) which drives the same
`MarketService` this app uses to read prices.
"""

import gzip
import io
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    g,
    send_from_directory,
    url_for,
)

from sqlalchemy import func, or_
from sqlalchemy.orm import joinedload

from models import (
    Agent,
    AgentEventInterest,
    Event,
    EventType,
    InformationEvent,
    Market,
    MarketOutcome,
    MarketStatus,
    PriceHistory,
    ROLE_WATCHER,
    STATUS_PENDING,
    TIER_NORMAL,
    TRIGGER_PRIORITY,
    Trade,
    TriggerType,
    WakeUpTask,
    db,
    init_app,
    make_dedup_key,
    time_bucket,
)
from services import MarketService, SchedulerService, ScoringService, SettlementService
from services.market_service import MarketError
from services.settlement_service import SettlementError

load_dotenv()


def _(s, **kwargs):
    """Identity passthrough that used to route through Flask-Babel.

    The UI is English-only now; we keep the `_()` calls in templates and
    Python code so callers don't have to change and the printf-style
    named substitutions (`_("Foo %(bar)s", bar=x)`) keep working.
    """
    return (s % kwargs) if kwargs else s


def create_app():
    app = Flask(__name__)
    # Needed only for flash() messages on the create-event form.
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-not-for-prod")
    # Cache static assets aggressively — tailwind.css / chart.umd.min.js /
    # app.css don't change between requests, and the Werkzeug dev server
    # is single-threaded-ish (see run_live.py's threaded=True). Every
    # avoided 304 round trip is one less request queuing behind the
    # HTML render. One-hour TTL is fine for local dev; a real deploy
    # would fingerprint the URLs and set it to a year.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600
    init_app(app)

    # Make the identity `_` available inside Jinja templates.
    app.jinja_env.globals["_"] = _

    @app.before_request
    def _mark_request_start():
        g._request_started_at = time.perf_counter()

    @app.after_request
    def _log_slow_request(response):
        started = getattr(g, "_request_started_at", None)
        if started is None:
            return response
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        threshold_ms = float(os.getenv("SLOW_REQUEST_MS", "1000"))
        if elapsed_ms >= threshold_ms:
            print(
                f"[web][slow] {request.method} {request.path} "
                f"{response.status_code} {elapsed_ms:.0f}ms",
                file=sys.stderr,
            )
        response.headers["X-Render-Time-ms"] = f"{elapsed_ms:.1f}"
        return response

    @app.after_request
    def _cache_static(response):
        """Tell the browser it can reuse /static/* without revalidating.

        Flask's default sends `Cache-Control: public, max-age=N` but
        the browser still revalidates on hard-reload / new tab because
        no immutability hint is set. Adding `immutable` (safe for the
        1-hour TTL we set above) stops the revalidation round trip.
        """
        if request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=3600, immutable"
        return response

    # Types worth compressing. Images/binaries are already compressed
    # (or too small to matter after HTTP overhead). We include the
    # UMD Chart.js bundle via `application/javascript`.
    _COMPRESSIBLE = {
        "text/html", "text/css", "text/plain", "text/xml",
        "application/json", "application/javascript",
        "application/xml", "image/svg+xml",
    }

    # Content types for pre-gzipped static assets — Flask's mimetype
    # guesser would look at `.css.gz` and say `application/gzip`, which
    # would break the browser. We keep the original type and tell the
    # browser it's just gzip-encoded.
    _STATIC_GZ_TYPES = {
        ".css": "text/css",
        ".js":  "application/javascript",
        ".svg": "image/svg+xml",
        ".json": "application/json",
    }

    # Override Flask's built-in `static` view so it serves the
    # pre-gzipped `.gz` variant next to the asset (see `gzip -k9`
    # step in the build) when the browser sends `Accept-Encoding: gzip`.
    # tailwind.css (73 KB → 10 KB) and chart.umd.min.js (200 KB → 68 KB)
    # both benefit hugely; over an SSH tunnel this is the difference
    # between "3 seconds to first paint" and "instant". Adding a new
    # route with the same rule doesn't work — Flask auto-registers the
    # `static` endpoint in `__init__`, and it wins the dispatch. We
    # keep the endpoint name so `url_for('static', …)` in templates
    # still resolves.
    def _static_gz(filename):
        accept = request.headers.get("Accept-Encoding", "")
        static_dir = app.static_folder
        if "gzip" in accept.lower():
            ext = os.path.splitext(filename)[1].lower()
            if ext in _STATIC_GZ_TYPES:
                gz_path = os.path.join(static_dir, filename + ".gz")
                if os.path.isfile(gz_path):
                    resp = send_from_directory(static_dir, filename + ".gz")
                    resp.headers["Content-Type"] = _STATIC_GZ_TYPES[ext]
                    resp.headers["Content-Encoding"] = "gzip"
                    resp.headers["Vary"] = "Accept-Encoding"
                    return resp
        return send_from_directory(static_dir, filename)

    app.view_functions["static"] = _static_gz

    @app.after_request
    def _gzip_response(response):
        """Compress the response body when the client accepts gzip.

        This is the single biggest win when the browser sits on the
        other side of an SSH port-forward or any slow link — a 2 MB
        `/leaderboard` HTML compresses to ~50 KB. Flask ships no
        compression by default; we do it inline (stdlib `gzip`) to
        avoid a new dependency.

        Static assets take a different path (`_static_gz` above serves
        pre-compressed `.gz` files) — this handler is for dynamic
        text/html/json responses coming out of Jinja / jsonify.

        Skips: too-small bodies (overhead not worth it), already-encoded
        responses (e.g. from `_static_gz`), streaming/direct-passthrough
        responses, and non-text content types.
        """
        accept = request.headers.get("Accept-Encoding", "")
        if "gzip" not in accept.lower():
            return response
        if response.direct_passthrough or response.is_streamed:
            return response
        if response.status_code < 200 or response.status_code >= 300:
            return response
        if response.headers.get("Content-Encoding"):
            return response  # already encoded (e.g. pre-gzipped static)
        ctype = (response.mimetype or "").lower()
        if ctype not in _COMPRESSIBLE:
            return response
        data = response.get_data()
        if len(data) < 1024:
            return response  # < 1 KB: gzip framing overhead is a wash
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz:
            gz.write(data)
        response.set_data(buf.getvalue())
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = str(len(response.get_data()))
        vary = response.headers.get("Vary")
        response.headers["Vary"] = (
            f"{vary}, Accept-Encoding" if vary and "Accept-Encoding" not in vary
            else (vary or "Accept-Encoding")
        )
        return response

    @app.route("/")
    def index():
        # Optional filters via query string. `q` matches title / description
        # / category; `status` filters by aggregate event status. Both
        # normalize to canonical forms so URLs are shareable.
        q = (request.args.get("q") or "").strip()
        status_filter = (request.args.get("status") or "").strip().upper()
        if status_filter not in ("OPEN", "CLOSED", "RESOLVED"):
            status_filter = ""  # treat unknown as "all"

        query = Event.query
        if q:
            pattern = f"%{q}%"
            query = query.filter(
                or_(
                    Event.title.ilike(pattern),
                    Event.description.ilike(pattern),
                    Event.category.ilike(pattern),
                )
            )
        events = query.order_by(Event.created_at.desc()).all()
        event_ids = [ev.id for ev in events]

        # Event.markets is a dynamic relationship, so using helpers like
        # `ev.status` / `ev.primary_market` in a loop issues extra SELECTs
        # per card. The index is the first page people hit through SSH
        # forwarding, so batch-load all markets and derive the per-event
        # maps once.
        markets_by_event = defaultdict(list)
        if event_ids:
            market_rows = (
                Market.query.filter(Market.event_id.in_(event_ids))
                .order_by(Market.event_id.asc(), Market.id.asc())
                .all()
            )
            for market in market_rows:
                markets_by_event[market.event_id].append(market)

        event_status = {}
        primary_markets = {}
        winning_markets = {}
        numeric_summaries = {}
        for ev in events:
            markets = markets_by_event.get(ev.id, [])
            primary_markets[ev.id] = markets[0] if markets else None
            if not markets:
                status = MarketStatus.OPEN
            elif all(m.status == MarketStatus.RESOLVED for m in markets):
                status = MarketStatus.RESOLVED
            elif any(m.status == MarketStatus.CLOSED for m in markets):
                status = MarketStatus.CLOSED
            else:
                status = MarketStatus.OPEN
            event_status[ev.id] = status.value

            if ev.event_type not in (EventType.BINARY, EventType.GROUPED):
                winning_markets[ev.id] = next(
                    (m for m in markets if m.outcome == MarketOutcome.YES), None
                )
            if ev.event_type == EventType.SCALAR:
                los = [m.bucket_lo for m in markets if m.bucket_lo is not None]
                his = [m.bucket_hi for m in markets if m.bucket_hi is not None]
                numeric_summaries[ev.id] = {
                    "min": min(los) if los else None,
                    "max": max(his) if his else None,
                    "unit": ev.scalar_unit,
                    "n_buckets": len(markets),
                    "unbounded_below": any(
                        m.bucket_lo is None and m.bucket_hi is not None
                        for m in markets
                    ),
                    "unbounded_above": any(
                        m.bucket_hi is None and m.bucket_lo is not None
                        for m in markets
                    ),
                }

        if status_filter:
            events = [ev for ev in events if event_status.get(ev.id) == status_filter]
            event_ids = [ev.id for ev in events]

        total_events = Event.query.count()
        # Latest YES/NO price per event, via its primary market — used by
        # BINARY card layout.
        latest_prices = {}
        # For CATEGORICAL cards we need per-market data. Precompute
        # `event_markets[eid] = [(market, latest_price), ...]` sorted by
        # yes_price descending so the card can show top candidates.
        event_markets = {}
        visible_markets = [m for eid in event_ids for m in markets_by_event.get(eid, [])]
        latest_by_market = {}
        if visible_markets:
            market_ids = [m.id for m in visible_markets]
            latest_subq = (
                db.session.query(
                    PriceHistory.market_id.label("market_id"),
                    func.max(PriceHistory.id).label("latest_id"),
                )
                .filter(PriceHistory.market_id.in_(market_ids))
                .group_by(PriceHistory.market_id)
                .subquery()
            )
            latest_rows = (
                PriceHistory.query.join(
                    latest_subq, PriceHistory.id == latest_subq.c.latest_id
                )
                .all()
            )
            latest_by_market = {row.market_id: row for row in latest_rows}

        parent_market_info = {}
        conditional_parents = {}
        parent_ids = [
            primary_markets[ev.id].parent_market_id
            for ev in events
            if ev.event_type == EventType.CONDITIONAL
            and primary_markets.get(ev.id) is not None
            and primary_markets[ev.id].parent_market_id is not None
        ]
        if parent_ids:
            parent_rows = (
                db.session.query(Market, Event)
                .join(Event, Market.event_id == Event.id)
                .filter(Market.id.in_(parent_ids))
                .all()
            )
            conditional_parents = {
                market.id: {"market": market, "event": event}
                for market, event in parent_rows
            }

        for ev in events:
            pm = primary_markets.get(ev.id)
            if pm is None:
                latest_prices[ev.id] = None
            else:
                latest_prices[ev.id] = latest_by_market.get(pm.id)
                if ev.event_type == EventType.CONDITIONAL and pm.parent_market_id:
                    parent = conditional_parents.get(pm.parent_market_id)
                    if parent:
                        parent_market_info[ev.id] = {
                            "title": parent["event"].title,
                            "required": (
                                pm.parent_required_outcome.value
                                if pm.parent_required_outcome else "—"
                            ),
                        }
            if ev.event_type != EventType.BINARY:
                rows = []
                for m in markets_by_event.get(ev.id, []):
                    ph = latest_by_market.get(m.id)
                    yes = ph.yes_price if ph else 0.5
                    rows.append({
                        "market": m,
                        "yes_price": yes,
                    })
                rows.sort(key=lambda r: r["yes_price"], reverse=True)
                event_markets[ev.id] = rows
        agent_count = Agent.query.count()

        # Per-event volume comes straight off the maintained counter
        # (see MarketService._record_trade). Avoids the full-scan
        # GROUP BY over trades that used to run on every index hit.
        event_volume = {ev.id: float(ev.total_volume or 0.0) for ev in events}

        # Dashboard stats. "open_markets" and "resolved_events" derive from
        # the primary market's status for BINARY events; a full multi-market
        # count comes with P1.
        open_markets = Market.query.filter_by(status=MarketStatus.OPEN).count()
        resolved_markets = Market.query.filter_by(status=MarketStatus.RESOLVED).count()
        # Total volume: SUM of the maintained per-event counter — 3–100
        # events, not 200k+ trades.
        total_volume = float(
            db.session.query(func.coalesce(func.sum(Event.total_volume), 0.0)).scalar()
            or 0.0
        )
        stats = {
            "open_markets": open_markets,
            "resolved_events": resolved_markets,
            "active_agents": agent_count,
            "total_volume": total_volume,
        }
        return render_template(
            "index.html",
            events=events,
            latest_prices=latest_prices,
            event_markets=event_markets,
            event_status=event_status,
            primary_markets=primary_markets,
            winning_markets=winning_markets,
            numeric_summaries=numeric_summaries,
            parent_market_info=parent_market_info,
            agent_count=agent_count,
            event_volume=event_volume,
            stats=stats,
            live_init_running=app.config.get("LIVE_INIT_RUNNING", False),
            live_init_message=app.config.get("LIVE_INIT_MESSAGE", ""),
            q=q,
            status_filter=status_filter,
            total_events=total_events,
        )

    @app.route("/agents")
    def agents_list():
        # Merged into /leaderboard (which now offers a table/cards toggle).
        # 302 keeps old bookmarks and inbound links working.
        return redirect(url_for("leaderboard"), code=302)

    @app.route("/leaderboard")
    def leaderboard():
        sort_by = ScoringService.normalize_sort(request.args.get("sort"))
        rows = ScoringService.get_leaderboard(sort_by=sort_by)
        # The cards view needs a few Agent-only fields (created_at,
        # risk_profile, name[:1] for the avatar) that don't appear on the
        # scoring row. Build a lookup keyed by agent id so the template
        # can render either view from the same ranked `rows`.
        agents_by_id = {a.id: a for a in Agent.query.all()}
        return render_template(
            "leaderboard.html",
            rows=rows,
            agents_by_id=agents_by_id,
            sort_by=sort_by,
            sort_options=ScoringService.SORT_OPTIONS,
        )

    @app.route("/events/<int:event_id>")
    def event_detail(event_id):
        ev = Event.query.get(event_id)
        if ev is None:
            abort(404)
        # BINARY: everything hangs off the primary market. CATEGORICAL /
        # SCALAR: iterate `ev.markets`; the template picks the right layout.
        pm = ev.primary_market
        if pm is None:
            # Event exists but has no market row yet — render as empty.
            prices = {"yes_price": 0.5, "no_price": 0.5}
            history = []
            trades = []
        else:
            prices = MarketService.get_current_price(pm.id)
            # Cap the chart to the most recent N price snapshots. Once
            # trading has been running a while there can be tens of
            # thousands of rows per market — pushing all of them through
            # Jinja + Chart.js turns a 100 ms page into 500+ ms and
            # bloats the HTML with data the human eye can't tell apart.
            # DESC + LIMIT + reverse gets us the tail in ascending order
            # cheaply thanks to ix_price_history_market_ts.
            history = list(reversed(
                PriceHistory.query.filter_by(market_id=pm.id)
                .order_by(PriceHistory.timestamp.desc())
                .limit(_CHART_MAX_POINTS)
                .all()
            ))
            # Trades across ALL markets on this event (so CATEGORICAL sees
            # every candidate's activity in one table).
            trades = (
                Trade.query.join(Market, Trade.market_id == Market.id)
                .options(joinedload(Trade.agent))
                .filter(Market.event_id == event_id)
                .order_by(Trade.created_at.desc())
                .limit(50)
                .all()
            )

        # For multi-market events, build a per-candidate view sorted by
        # current YES price (highest first).
        markets_view = []
        if ev.event_type != EventType.BINARY:
            for m in ev.markets:
                ph = (
                    PriceHistory.query.filter_by(market_id=m.id)
                    .order_by(PriceHistory.timestamp.desc())
                    .first()
                )
                yes_price = ph.yes_price if ph else 0.5
                vol = float(m.total_volume or 0.0)
                markets_view.append({
                    "market": m,
                    "yes_price": yes_price,
                    "no_price": 1.0 - yes_price,
                    "volume": vol,
                })
            markets_view.sort(key=lambda r: r["yes_price"], reverse=True)
        # Serialize the price history for the Chart.js line chart in the
        # template. Timestamps go out as ISO strings so JS can parse them.
        # BINARY: single-line chart of the primary market's YES price.
        chart_labels = [p.timestamp.strftime("%Y-%m-%d %H:%M:%S") for p in history]
        chart_values = [round(p.yes_price, 4) for p in history]
        # CATEGORICAL / SCALAR: multi-line chart — one dataset per market,
        # X axis is the union of all timestamps with forward-fill.
        chart_datasets = []
        if ev.event_type != EventType.BINARY:
            chart_labels, chart_datasets = _multi_market_history(ev)
        evidence = (
            db.session.query(InformationEvent)
            .options(joinedload(InformationEvent.source))
            .filter(InformationEvent.event_id == event_id)
            .order_by(InformationEvent.retrieved_at.desc())
            .limit(30)
            .all()
        )
        # Simple "leading/recent agents" for the sticky panel: unique agents
        # from the latest trades on this event, preserving recency order.
        recent_agents = []
        _seen = set()
        for t in trades:
            if t.agent and t.agent.id not in _seen:
                _seen.add(t.agent.id)
                recent_agents.append(t.agent)
            if len(recent_agents) >= 5:
                break
        # Total volume on this event — read the counter maintained on
        # Event.total_volume rather than re-summing every trade join.
        event_volume = float(ev.total_volume or 0.0)
        return render_template(
            "event_detail.html",
            event=ev,
            prices=prices,
            history=history,
            trades=trades,
            evidence=evidence,
            chart_labels=chart_labels,
            chart_values=chart_values,
            chart_datasets=chart_datasets,
            recent_agents=recent_agents,
            event_volume=event_volume,
            markets_view=markets_view,
        )

    # ---- Event creation (no auth; anyone with the URL can create) --------

    @app.route("/create-event", methods=["GET", "POST"])
    def create_event():
        if request.method == "GET":
            return render_template(
                "create_event.html",
                form={},
                errors=[],
                parent_options=_parent_market_options(),
            )

        form = {k: (request.form.get(k) or "").strip() for k in (
            "title", "description", "category",
            "close_time", "resolution_time", "resolution_source",
            "event_type", "candidates", "scalar_unit", "buckets",
            "scalar_min", "scalar_max", "scalar_count",
            "parent_market_id", "parent_required_outcome",
        )}
        errors = []

        if not form["title"]:
            errors.append(_("Title is required."))
        if len(form["title"]) > 255:
            errors.append(_("Title must be 255 characters or fewer."))

        # Type: default BINARY; must be a known enum value.
        raw_type = (form.get("event_type") or "BINARY").upper()
        try:
            event_type = EventType(raw_type)
        except ValueError:
            errors.append(_("Unknown event type: %(t)s", t=raw_type))
            event_type = EventType.BINARY

        close_time = _parse_dt(form["close_time"], "close_time", errors)
        resolution_time = _parse_dt(form["resolution_time"], "resolution_time", errors)
        if close_time and resolution_time and resolution_time < close_time:
            errors.append(_("Resolution time must be at or after close time."))

        # CATEGORICAL / GROUPED: parse candidate labels from the textarea.
        # Both use the same input format — the only server-side difference
        # is that GROUPED markets resolve independently (see settlement).
        candidate_labels = []
        if event_type in (EventType.CATEGORICAL, EventType.GROUPED):
            raw = form.get("candidates") or ""
            seen = set()
            for line in raw.splitlines():
                lab = line.strip()
                if not lab or lab in seen:
                    continue
                seen.add(lab)
                if len(lab) > 255:
                    errors.append(_(
                        "Candidate label too long (max 255 chars): %(lab)s",
                        lab=lab[:64] + "…",
                    ))
                    continue
                candidate_labels.append(lab)
            if len(candidate_labels) < 2:
                errors.append(_("This event type needs at least 2 sub-markets."))
            if len(candidate_labels) > 32:
                errors.append(_("Too many sub-markets (max 32)."))

        # CONDITIONAL: single market with a parent pointer.
        parent_market_id = None
        parent_required = None
        if event_type == EventType.CONDITIONAL:
            pm_raw = form.get("parent_market_id") or ""
            pr_raw = (form.get("parent_required_outcome") or "").upper()
            try:
                parent_market_id = int(pm_raw)
            except ValueError:
                errors.append(_("Choose a parent market."))
            if pr_raw not in ("YES", "NO"):
                errors.append(_("Parent required outcome must be YES or NO."))
            else:
                parent_required = MarketOutcome(pr_raw)
            if parent_market_id is not None and not errors:
                parent = Market.query.get(parent_market_id)
                if parent is None:
                    errors.append(_("Parent market not found."))
                elif parent.status != MarketStatus.OPEN:
                    errors.append(_(
                        "Parent market %(id)s is not OPEN "
                        "(status: %(s)s) and cannot be used.",
                        id=parent.id, s=parent.status.value,
                    ))

        # SCALAR: parse bucket ranges from the textarea, one per line.
        # Alternative: min/max/count for equal-width auto-generation
        # (used only when the textarea is empty).
        bucket_specs = []  # list of (lo, hi, label)
        if event_type == EventType.SCALAR:
            raw = form.get("buckets") or ""
            if raw.strip():
                parsed_lines = []  # [(kind, lo, hi)] where kind ∈ {"lo","closed","hi"}
                for lineno, line in enumerate(raw.splitlines(), start=1):
                    s = line.strip()
                    if not s:
                        continue
                    parsed = _parse_bucket_line(s)
                    if parsed is None:
                        errors.append(_(
                            "Bucket line %(n)s not in `lo-hi`, `<X`, or `X+` form: %(s)r",
                            n=lineno, s=s,
                        ))
                        continue
                    lo, hi = parsed
                    if lo is None:
                        parsed_lines.append(("lo_tail", lo, hi))
                    elif hi is None:
                        parsed_lines.append(("hi_tail", lo, hi))
                    else:
                        if not (lo < hi):
                            errors.append(_(
                                "Bucket line %(n)s: lo must be less than hi (got %(lo)s, %(hi)s)",
                                n=lineno, lo=lo, hi=hi,
                            ))
                            continue
                        parsed_lines.append(("closed", lo, hi))
                # Tail-bucket order rules: at most one lo_tail (must be
                # first), at most one hi_tail (must be last).
                lo_tail_idx = [i for i, (k, _lo, _hi) in enumerate(parsed_lines) if k == "lo_tail"]
                hi_tail_idx = [i for i, (k, _lo, _hi) in enumerate(parsed_lines) if k == "hi_tail"]
                if len(lo_tail_idx) > 1:
                    errors.append(_("At most one `<X` tail bucket allowed."))
                if len(hi_tail_idx) > 1:
                    errors.append(_("At most one `X+` tail bucket allowed."))
                if lo_tail_idx and lo_tail_idx[0] != 0:
                    errors.append(_("The `<X` tail bucket must be the first line."))
                if hi_tail_idx and hi_tail_idx[0] != len(parsed_lines) - 1:
                    errors.append(_("The `X+` tail bucket must be the last line."))
                for kind, lo, hi in parsed_lines:
                    bucket_specs.append((lo, hi, _format_bucket_label(lo, hi)))
            else:
                # No textarea → try min/max/count auto-generation.
                min_s, max_s, cnt_s = form.get("scalar_min"), form.get("scalar_max"), form.get("scalar_count")
                if min_s and max_s and cnt_s:
                    try:
                        smin = _num(min_s)
                        smax = _num(max_s)
                        scnt = int(cnt_s)
                    except ValueError:
                        errors.append(_("scalar_min/max must be numeric, scalar_count must be an integer."))
                        smin = smax = None
                        scnt = 0
                    if smin is not None and smax is not None:
                        if smin >= smax:
                            errors.append(_("scalar_min must be less than scalar_max."))
                        elif not (2 <= scnt <= 32):
                            errors.append(_("scalar_count must be between 2 and 32."))
                        else:
                            width = (smax - smin) / scnt
                            for i in range(scnt):
                                lo = smin + i * width
                                hi = smin + (i + 1) * width if i < scnt - 1 else smax
                                bucket_specs.append((lo, hi, _format_bucket_label(lo, hi)))
                else:
                    errors.append(_(
                        "SCALAR events need either a bucket list OR "
                        "scalar_min + scalar_max + scalar_count."
                    ))
            if not errors and len(bucket_specs) < 2:
                errors.append(_("SCALAR events need at least 2 buckets."))
            if len(bucket_specs) > 32:
                errors.append(_("Too many buckets (max 32)."))

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template(
                "create_event.html",
                form=form,
                errors=errors,
                parent_options=_parent_market_options(),
            ), 400

        # Persist event + market(s) + initial 50/50 price snapshot(s) in
        # one transaction.
        try:
            event = Event(
                title=form["title"],
                description=form["description"] or None,
                category=form["category"] or None,
                event_type=event_type,
                close_time=close_time,
                resolution_source=form["resolution_source"] or None,
                scalar_unit=(form["scalar_unit"] or None) if event_type == EventType.SCALAR else None,
            )
            db.session.add(event)
            db.session.flush()  # get event.id

            if event_type == EventType.BINARY:
                markets_to_create = [(None, None, None, None, None)]
            elif event_type in (EventType.CATEGORICAL, EventType.GROUPED):
                markets_to_create = [
                    (lab, None, None, None, None) for lab in candidate_labels
                ]
            elif event_type == EventType.CONDITIONAL:
                markets_to_create = [(None, None, None, parent_market_id, parent_required)]
            else:  # SCALAR
                markets_to_create = [
                    (lab, lo, hi, None, None) for (lo, hi, lab) in bucket_specs
                ]

            created_market_ids = []
            for label, bucket_lo, bucket_hi, pmid, preq in markets_to_create:
                market = Market(
                    event_id=event.id,
                    label=label,
                    bucket_lo=bucket_lo,
                    bucket_hi=bucket_hi,
                    status=MarketStatus.OPEN,
                    resolution_time=resolution_time,
                    parent_market_id=pmid,
                    parent_required_outcome=preq,
                )
                db.session.add(market)
                db.session.flush()
                created_market_ids.append(market.id)
                db.session.add(
                    PriceHistory(
                        market_id=market.id,
                        yes_price=0.5,
                        no_price=0.5,
                    )
                )
            db.session.commit()
        except Exception as exc:  # noqa: BLE001
            db.session.rollback()
            flash(_("Failed to create event: %(err)s", err=str(exc)), "error")
            return render_template(
                "create_event.html",
                form=form,
                errors=[str(exc)],
                parent_options=_parent_market_options(),
            ), 500

        activation = _activate_new_event_agents(event.id, created_market_ids)
        msg = _("Event #%(id)s created.", id=event.id)
        if activation["watchers"] or activation["tasks"]:
            msg += _(
                " Added %(w)s watcher(s), queued %(t)s wake-up task(s).",
                w=activation["watchers"], t=activation["tasks"],
            )
        flash(msg, "info")
        return redirect(url_for("event_detail", event_id=event.id))

    @app.post("/create-event/ai-suggest")
    def ai_suggest_event():
        """Turn a free-text prompt into a suggested event spec.

        Accepts JSON `{"prompt": "..."}` and returns a shape matching
        the create-event form fields so the client can pre-fill them.
        Falls back to a safe empty response if the LLM isn't configured
        or returns garbage — the form still works without this.
        """
        payload = request.get_json(silent=True) or {}
        user_prompt = (payload.get("prompt") or "").strip()
        if not user_prompt:
            return jsonify({"error": "prompt is required"}), 400
        if len(user_prompt) > 2000:
            return jsonify({"error": "prompt too long (max 2000 chars)"}), 400

        try:
            from llm import get_llm_client
            client = get_llm_client()
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"LLM not available: {exc}"}), 503
        if not client.available:
            return jsonify({
                "error": f"LLM not configured: {client.unavailable_reason}"
            }), 503

        system = (
            "You are an assistant that helps design prediction-market "
            "events. Given a user's rough idea, produce a well-scoped, "
            "objectively resolvable event spec. Reply with STRICT JSON "
            "only — no prose, no markdown. Schema:\n"
            "{\n"
            '  "title": string  # <= 200 chars, phrased as a yes/no or '
            'quantitative question when possible\n'
            '  "description": string  # 1-3 sentences: how it resolves, '
            'data source, edge cases\n'
            '  "category": string  # short lowercase tag (e.g. "markets", '
            '"ai", "sports", "politics", "climate")\n'
            '  "event_type": "BINARY" | "CATEGORICAL" | "SCALAR" | "GROUPED"\n'
            '  "candidates": [string]  # 2-8 items, only if CATEGORICAL or '
            'GROUPED; else []\n'
            '  "scalar_unit": string  # only if SCALAR; else ""\n'
            '  "buckets": [string]  # only if SCALAR: lines like "<50000", '
            '"50000-100000", "100000+"; 2-8 items; else []\n'
            '  "resolution_source": string  # authoritative source, or ""\n'
            '  "close_time": string  # ISO "YYYY-MM-DDTHH:MM" or ""\n'
            '  "resolution_time": string  # ISO "YYYY-MM-DDTHH:MM" or ""\n'
            "}\n"
            "Pick BINARY unless the user's idea clearly needs multiple "
            "mutually-exclusive outcomes (CATEGORICAL), independent "
            "sub-questions (GROUPED), or a numeric range (SCALAR). "
            f"Today's date is {datetime.utcnow().strftime('%Y-%m-%d')}."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        try:
            data = client.chat_json(messages, temperature=0.3, max_tokens=800)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"LLM call failed: {exc}"}), 502

        # Coerce into the shape the client expects — defensively, since
        # the model may omit fields or return wrong types.
        def _s(v):
            return v.strip() if isinstance(v, str) else ""

        def _list_of_str(v):
            if not isinstance(v, list):
                return []
            out = []
            for item in v[:32]:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
            return out

        etype = _s(data.get("event_type")).upper() or "BINARY"
        if etype not in ("BINARY", "CATEGORICAL", "SCALAR", "GROUPED"):
            etype = "BINARY"
        result = {
            "title": _s(data.get("title"))[:255],
            "description": _s(data.get("description")),
            "category": _s(data.get("category"))[:64],
            "event_type": etype,
            "candidates": _list_of_str(data.get("candidates")) if etype in ("CATEGORICAL", "GROUPED") else [],
            "scalar_unit": _s(data.get("scalar_unit"))[:32] if etype == "SCALAR" else "",
            "buckets": _list_of_str(data.get("buckets")) if etype == "SCALAR" else [],
            "resolution_source": _s(data.get("resolution_source"))[:255],
            "close_time": _s(data.get("close_time"))[:16],
            "resolution_time": _s(data.get("resolution_time"))[:16],
        }
        return jsonify(result)

    @app.post("/events/<int:event_id>/resolve")
    def resolve_event_view(event_id):
        """Settle an event from the UI.

        Dispatches on `event_type` to the right SettlementService entry
        point. Form fields:
            BINARY / CONDITIONAL : outcome=YES|NO
            CATEGORICAL          : winner_market_id=<id>
            SCALAR               : actual_value=<float>
            GROUPED              : outcome_<market_id>=YES|NO for each
                                   sub-market you want to settle this call
        No auth in Phase 1 — matches the create-event route's stance.
        """
        ev = Event.query.get(event_id)
        if ev is None:
            abort(404)
        try:
            if ev.event_type == EventType.CATEGORICAL:
                raw = (request.form.get("winner_market_id") or "").strip()
                if not raw:
                    raise SettlementError("choose a winner")
                SettlementService.resolve_categorical(event_id, int(raw))
            elif ev.event_type == EventType.SCALAR:
                raw = (request.form.get("actual_value") or "").strip()
                if not raw:
                    raise SettlementError("enter the actual value")
                SettlementService.resolve_scalar(event_id, float(raw))
            elif ev.event_type == EventType.GROUPED:
                outcomes_map = {}
                for m in ev.markets:
                    v = (request.form.get(f"outcome_{m.id}") or "").upper()
                    if v in ("YES", "NO"):
                        outcomes_map[m.id] = v
                if not outcomes_map:
                    raise SettlementError("pick YES/NO for at least one sub-market")
                SettlementService.resolve_grouped(event_id, outcomes_map)
            else:  # BINARY or CONDITIONAL — single-market events
                outcome = (request.form.get("outcome") or "").upper()
                if outcome not in ("YES", "NO"):
                    raise SettlementError("outcome must be YES or NO")
                if ev.event_type == EventType.BINARY:
                    SettlementService.resolve_event(event_id, outcome)
                else:  # CONDITIONAL: resolve its single market directly
                    pm = ev.primary_market
                    if pm is None:
                        raise SettlementError("event has no market")
                    SettlementService.resolve_market(pm.id, outcome)
            flash(_("Event #%(id)s resolved.", id=event_id), "info")
        except (ValueError, SettlementError) as exc:
            flash(_("Resolve failed: %(err)s", err=str(exc)), "error")
        return redirect(url_for("event_detail", event_id=event_id))

    @app.post("/dev/trade")
    def dev_trade():
        """POST JSON to manually execute a trade against the market.

        Pass either `market_id` directly, or `event_id` (BINARY events
        are resolved to their primary market for you).

        Example:
            curl -X POST http://localhost:6006/dev/trade \\
                 -H 'Content-Type: application/json' \\
                 -d '{"agent_id":1,"event_id":1,"action":"BUY_YES","amount":500}'
        """
        payload = request.get_json(silent=True) or {}
        try:
            market_id = payload.get("market_id")
            if market_id is None:
                event_id = int(payload["event_id"])
                ev = Event.query.get(event_id)
                if ev is None:
                    return jsonify({"error": f"event {event_id} not found"}), 400
                pm = ev.primary_market
                if pm is None:
                    return jsonify({"error": f"event {event_id} has no markets"}), 400
                market_id = pm.id
            trade = MarketService.execute_trade(
                agent_id=int(payload["agent_id"]),
                market_id=int(market_id),
                action=payload["action"],
                amount=float(payload.get("amount", 0) or 0),
                fraction=(
                    float(payload["fraction"])
                    if payload.get("fraction") is not None
                    else None
                ),
                probability_yes=payload.get("probability_yes"),
                confidence=payload.get("confidence"),
                reasoning_summary=payload.get("reasoning_summary"),
            )
        except (KeyError, ValueError) as exc:
            return jsonify({"error": f"bad request: {exc}"}), 400
        except MarketError as exc:
            return jsonify({"error": str(exc)}), 400

        prices = MarketService.get_current_price(trade.market_id)
        return jsonify(
            {
                "trade_id": trade.id,
                "market_id": trade.market_id,
                "action": trade.action.value,
                "amount": trade.amount,
                "price_before": trade.price_before,
                "price_after": trade.price_after,
                "yes_price": prices["yes_price"],
                "no_price": prices["no_price"],
            }
        )

    return app


app = create_app()



def _parse_dt(raw: str, field: str, errors: list):
    """Parse an <input type="datetime-local"> value, or return None.

    Empty string is allowed (both fields are optional). Bad values push
    an error onto the shared errors list and return None so the form
    can round-trip them for the user to fix.
    """
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        errors.append(_("%(field)s: expected YYYY-MM-DDTHH:MM, got %(raw)r",
                        field=field, raw=raw))
        return None


# Matches `<lo> - <hi>` where each side may be int / float / scientific /
# signed. Commas and underscores in numbers are tolerated. Any of
# `-`, `–`, `—`, `to` works as separator.
_BUCKET_RE = re.compile(
    r"^\s*(?P<lo>-?\d[\d_,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"\s*(?:-|–|—|to)\s*"
    r"(?P<hi>-?\d[\d_,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)
# Matches a lower-tail bucket: `<X`, `< X`, `≤X`. Captures X.
_BUCKET_TAIL_LO_RE = re.compile(
    r"^\s*(?:<|<=|≤)\s*(?P<hi>-?\d[\d_,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)
# Matches an upper-tail bucket: `X+`, `>=X`, `≥X`, `>X` (treated as `≥`).
_BUCKET_TAIL_HI_RE = re.compile(
    r"^\s*(?:(?P<lo1>-?\d[\d_,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\+"
    r"|(?:>=|≥|>)\s*(?P<lo2>-?\d[\d_,]*(?:\.\d+)?(?:[eE][+-]?\d+)?))\s*$"
)


def _num(s: str) -> float:
    return float(s.replace(",", "").replace("_", ""))


def _parse_bucket_line(s: str):
    """Parse a bucket spec line. Returns (lo, hi) or None.

    Recognizes three forms:
        "lo - hi"   → closed range         → (lo, hi)
        "<X"        → lower-tail           → (None, X)
        "X+" / ">=X" → upper-tail          → (X, None)
    """
    m = _BUCKET_TAIL_LO_RE.match(s)
    if m:
        try:
            return (None, _num(m.group("hi")))
        except ValueError:
            return None
    m = _BUCKET_TAIL_HI_RE.match(s)
    if m:
        lo_str = m.group("lo1") or m.group("lo2")
        try:
            return (_num(lo_str), None)
        except ValueError:
            return None
    m = _BUCKET_RE.match(s)
    if not m:
        return None
    try:
        return (_num(m.group("lo")), _num(m.group("hi")))
    except ValueError:
        return None


def _format_bucket_label(lo, hi) -> str:
    """Render a bucket range as a human-readable label."""
    if lo is None and hi is not None:
        return f"<{hi:,g}"
    if hi is None and lo is not None:
        return f"≥{lo:,g}"
    return f"{lo:,g}–{hi:,g}"


# Chart palette for the multi-line time chart on CATEGORICAL / SCALAR
# events. Cycles when there are more markets than colors.
_CHART_PALETTE = [
    "#16a34a", "#dc2626", "#2563eb", "#ea580c", "#7c3aed",
    "#0891b2", "#65a30d", "#db2777", "#0d9488", "#a16207",
    "#4f46e5", "#be123c",
]

# Max PriceHistory rows we push through Jinja/Chart.js per market. A
# live market accumulates one row per trade — once trading has been
# running for a few hours that's tens of thousands. Beyond ~500 the
# extra points are visually indistinguishable but blow up wire size
# and render time. We take the tail (most recent) so the "latest
# state" is always accurate.
_CHART_MAX_POINTS = 500


def _multi_market_history(event):
    """Assemble Chart.js labels + datasets for a multi-market event.

    Returns (chart_labels, chart_datasets) where labels is a list of ISO
    timestamp strings (union of every market's PriceHistory rows,
    ascending) and datasets is `[{label, data, color}, ...]` — one per
    market, forward-filled so every dataset has the same length. Missing
    initial samples default to 0.5 (the 50/50 prior).
    """
    from models import PriceHistory  # local to keep helper file-scoped

    per_market = []  # [(market, [(ts, yes_price), ...] asc), ...]
    for m in event.markets:
        # Same tail-only cap as the BINARY chart path — see the comment
        # on the corresponding query in `event_detail`. Uses
        # ix_price_history_market_ts.
        rows = (
            PriceHistory.query.filter_by(market_id=m.id)
            .order_by(PriceHistory.timestamp.desc())
            .limit(_CHART_MAX_POINTS)
            .all()
        )
        rows.reverse()
        per_market.append((m, [(r.timestamp, r.yes_price) for r in rows]))

    all_ts = sorted({ts for _, pairs in per_market for ts, _ in pairs})
    chart_labels = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in all_ts]

    chart_datasets = []
    for i, (m, pairs) in enumerate(per_market):
        by_ts = dict(pairs)
        data = []
        last = 0.5
        for ts in all_ts:
            if ts in by_ts:
                last = by_ts[ts]
            data.append(round(last, 4))
        chart_datasets.append({
            "label": m.label or "(unnamed)",
            "data": data,
            "color": _CHART_PALETTE[i % len(_CHART_PALETTE)],
        })
    return chart_labels, chart_datasets


def _parent_market_options():
    """Options list for the CONDITIONAL parent-market dropdown.

    Returns [{id, label}] for every OPEN market (any event type) in
    creation order. Label combines event title + market label so users
    can pick unambiguously.
    """
    rows = (
        db.session.query(Market, Event)
        .join(Event, Market.event_id == Event.id)
        .filter(Market.status == MarketStatus.OPEN)
        .order_by(Event.id.asc(), Market.id.asc())
        .all()
    )
    options = []
    for m, ev in rows:
        label_bit = m.label or ev.event_type.value
        options.append({
            "id": m.id,
            "label": f"#{ev.id} · {ev.title[:80]} · {label_bit}",
        })
    return options


def _activate_new_event_agents(event_id: int, market_ids, watcher_limit: int = 50):
    """Make a freshly-created event visible to the live agent pipeline.

    Startup seeds watcher subscriptions only for events that already exist.
    Events created later through the UI otherwise have no holders/watchers,
    so natural wake-ups and information triggers may have zero candidates.
    This helper assigns a deterministic 20% watcher slice (capped) and queues
    immediate INFORMATION tasks so a running worker can react without waiting
    for the next natural wake-up.
    """
    market_ids = [int(mid) for mid in (market_ids or []) if mid is not None]
    if not market_ids:
        return {"watchers": 0, "tasks": 0}

    agent_ids = [
        int(row[0]) for row in (
            db.session.query(Agent.id)
            .filter(or_(Agent.status == "active", Agent.status.is_(None)))
            .order_by(Agent.id.asc())
            .all()
        )
    ]
    watcher_ids = agent_ids[::5][:max(0, int(watcher_limit))]
    if not watcher_ids:
        return {"watchers": 0, "tasks": 0}

    existing_watchers = {
        int(row[0]) for row in (
            db.session.query(AgentEventInterest.agent_id)
            .filter(
                AgentEventInterest.event_id == event_id,
                AgentEventInterest.role == ROLE_WATCHER,
                AgentEventInterest.agent_id.in_(watcher_ids),
            )
            .all()
        )
    }
    new_watchers = [aid for aid in watcher_ids if aid not in existing_watchers]
    if new_watchers:
        db.session.bulk_insert_mappings(
            AgentEventInterest,
            [
                {
                    "agent_id": aid,
                    "event_id": event_id,
                    "role": ROLE_WATCHER,
                    "weight": 0.5,
                }
                for aid in new_watchers
            ],
        )

    now = SchedulerService.now()
    bucket = time_bucket(now)
    existing_tasks = {
        row[0] for row in (
            db.session.query(WakeUpTask.dedup_key)
            .filter(
                WakeUpTask.event_id == event_id,
                WakeUpTask.status == STATUS_PENDING,
                WakeUpTask.dedup_key.in_(
                    [make_dedup_key(aid, event_id, bucket) for aid in watcher_ids]
                ),
            )
            .all()
        )
    }
    task_rows = []
    primary_market_id = market_ids[0]
    for aid in watcher_ids:
        dedup_key = make_dedup_key(aid, event_id, bucket)
        if dedup_key in existing_tasks:
            continue
        task_rows.append({
            "agent_id": aid,
            "event_id": event_id,
            "market_id": primary_market_id,
            "trigger_type": TriggerType.INFORMATION.value,
            "priority": TRIGGER_PRIORITY[TriggerType.INFORMATION.value],
            "tier": TIER_NORMAL,
            "status": STATUS_PENDING,
            "scheduled_at": now,
            "time_bucket": bucket,
            "relevance": 0.7,
            "information_impact": 0.5,
            "position_relevance": 0.2,
            "expertise": 0.0,
            "portfolio_risk": 0.0,
            "wake_score": 0.5,
            "wake_reasons": TriggerType.INFORMATION.value,
            "dedup_key": dedup_key,
            "cascade_depth": 0,
        })
    if task_rows:
        db.session.bulk_insert_mappings(WakeUpTask, task_rows)
    db.session.commit()
    return {"watchers": len(new_watchers), "tasks": len(task_rows)}


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "6006"))
    app.run(debug=True, port=port)
