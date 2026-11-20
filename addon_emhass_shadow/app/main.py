"""Panel web server and daily scheduler for the EMHASS Shadow add-on."""

import json
import threading
import time
import traceback
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

import daily
import options_meta
import pipeline
import retrain

DATA = Path("/data")
APP = Path(__file__).resolve().parent
PORT = 8099

STATE = {"running": False, "last_error": None, "last_run": None, "next_run": None}
LOCK = threading.Lock()


def _report_retrain_problems(status: dict) -> None:
    """Raise/clear the retrain error notification from a retrain status dict."""
    models = (status or {}).get("models") or {}
    failed = {
        name: info.get("error", "")
        for name, info in models.items()
        if info.get("status") == "failed"
    }
    if failed:
        details = "; ".join(
            f"{'нагрузка' if name == 'load' else 'солнце'}: {str(err)[:150]}"
            for name, err in failed.items()
        )
        pipeline.notify_problem(
            "retrain",
            "EMHASS Shadow: переобучение не удалось",
            f"Не удалось переобучить модели ({details}). Прогнозы работают на "
            "прежних моделях. Панель → «Переобучить модели» повторит попытку; "
            "уведомление исчезнет после успешного обучения.",
        )
    else:
        pipeline.notify_problem("retrain", "", None)


def run_pipeline(retrain_first: bool = True, notify: str = "always") -> None:
    """One planning run. notify: "always" (daily, manual, grid change) or
    "changes" (hourly replans: notify only when something was written)."""
    with LOCK:
        if STATE["running"]:
            return
        STATE["running"] = True
        STATE["last_error"] = None
    try:
        if not (DATA / "hourly_audit.jsonl").exists():
            # Fresh install: learn from what Home Assistant already holds
            # (recorder detail + months of long-term statistics + outages),
            # train on it, and only then plan.
            try:
                STATE["last_backfill"] = pipeline.bootstrap(pipeline.options())
                _report_retrain_problems(retrain.retrain_if_due(force=True))
            except Exception:
                STATE["last_error"] = traceback.format_exc(limit=5)
        if retrain_first:
            try:
                _report_retrain_problems(retrain.retrain_if_due())
            except Exception:
                # Retraining must never block the daily plan.
                pass
        pipeline.run(notify=notify)
        STATE["last_run"] = datetime.now(UTC).isoformat()
        pipeline.notify_problem("run", "", None)
    except Exception:
        STATE["last_error"] = traceback.format_exc(limit=5)
        pipeline.notify_problem(
            "run",
            "EMHASS Shadow: расчёт не удался",
            "Последний расчёт плана завершился ошибкой:\n"
            + str(STATE["last_error"]).strip().splitlines()[-1][:300]
            + "\nПолный traceback — в панели «Теневой план» и в журнале аддона. "
            "Уведомление исчезнет после первого успешного расчёта.",
        )
    finally:
        STATE["running"] = False


def run_retrain() -> None:
    with LOCK:
        if STATE["running"]:
            return
        STATE["running"] = True
        STATE["last_error"] = None
    try:
        _report_retrain_problems(retrain.retrain_if_due(force=True))
    except Exception:
        STATE["last_error"] = traceback.format_exc(limit=5)
        pipeline.notify_problem(
            "retrain",
            "EMHASS Shadow: переобучение не удалось",
            "Переобучение моделей завершилось ошибкой:\n"
            + str(STATE["last_error"]).strip().splitlines()[-1][:300]
            + "\nПолный traceback — в панели «Теневой план» и в журнале аддона.",
        )
    finally:
        STATE["running"] = False


def run_backfill() -> None:
    """Pull recorder history into the audit, then retrain on it at once."""
    with LOCK:
        if STATE["running"]:
            return
        STATE["running"] = True
        STATE["last_error"] = None
    try:
        opts = pipeline.options()
        site = {
            "latitude": opts["latitude"],
            "longitude": opts["longitude"],
            "time_zone": opts["time_zone"],
            "max_charge_kw": opts.get("max_charge_kw"),
        }
        ent = pipeline.resolve_entities(opts)
        days = float(opts.get("history_backfill_days", 14))
        added = pipeline.backfill_audit(site, ent, days)
        info = {"at": datetime.now(UTC).isoformat(), "recorder_rows": added}
        try:
            info["outage_transitions"] = pipeline.update_outage_log(ent, hours=max(48.0, days * 24))
        except Exception as error:
            info["outage_error"] = str(error)[:200]
        try:
            info["statistics_rows"] = pipeline.backfill_statistics(
                site, ent, float(opts.get("history_statistics_days", 365))
            )
        except Exception as error:
            info["statistics_rows"] = 0
            info["statistics_error"] = str(error)[:200]
        info["added_rows"] = added + info["statistics_rows"]
        STATE["last_backfill"] = info
        if info["added_rows"]:
            retrain.retrain_if_due(force=True)
    except Exception:
        STATE["last_error"] = traceback.format_exc(limit=5)
    finally:
        STATE["running"] = False


def reports() -> list[Path]:
    directory = DATA / "shadow_reports"
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"), reverse=True)


def latest_report() -> dict | None:
    files = reports()
    if not files:
        return None
    return json.loads(files[0].read_text(encoding="utf-8"))


TRIGGER_MIN_INTERVAL = timedelta(minutes=20)  # rate limit for event-driven runs
DEFAULT_REPLAN_MINUTES = 60  # MPC cadence: replan when the last run is older


def replan_interval(opts) -> timedelta:
    try:
        minutes = int(opts.get("replan_minutes", DEFAULT_REPLAN_MINUTES))
    except (TypeError, ValueError):
        minutes = DEFAULT_REPLAN_MINUTES
    return timedelta(minutes=min(240, max(15, minutes)))
TRIGGER_SOC_DEVIATION_PP = 10.0  # replan when SOC leaves the plan by this much
TICK_SECONDS = 600


def _trigger_reason(opts) -> str | None:
    """Why an unscheduled recalculation is due; None when the plan stands.

    Triggers: the grid binary changed state (outage or restore), the plan
    is older than the replan interval (hourly by default — the solve takes
    milliseconds, and a fresh plan keeps the zone values on the measured
    SOC), or the battery SOC deviates from the current hour's optimizer plan
    by more than 10 pp. Triggered runs are rate-limited to one per 20
    minutes so a flapping sensor cannot spin the pipeline.
    """
    last_trigger = STATE.get("last_trigger")
    if last_trigger is not None:
        elapsed = datetime.now(UTC) - datetime.fromisoformat(last_trigger["at"])
        if elapsed < TRIGGER_MIN_INTERVAL:
            return None
    try:
        ent = pipeline.resolve_entities(opts)
        grid_binary = str(ent.get("grid_binary") or "").strip()
        if grid_binary:
            state = str(pipeline.ha_get(f"/api/states/{grid_binary}")["state"])
            previous = STATE.get("grid_state")
            STATE["grid_state"] = state
            if previous in ("on", "off") and state in ("on", "off") and state != previous:
                return f"сеть {previous} -> {state}"
    except Exception:
        pass
    last_run = STATE.get("last_run")
    if last_run is None:
        return "ещё ни одного успешного расчёта"
    if datetime.now(UTC) - datetime.fromisoformat(last_run) >= replan_interval(opts):
        return "плановый пересчёт"
    report = latest_report()
    optimizer = ((report or {}).get("advisory") or {}).get("optimizer") or {}
    soc_opt = optimizer.get("soc_opt")
    starts = ((report or {}).get("hourly") or {}).get("starts_utc") or []
    if soc_opt and starts:
        try:
            now_utc = datetime.now(UTC)
            index = next(
                (
                    i
                    for i, start in enumerate(starts)
                    if datetime.fromisoformat(start)
                    <= now_utc
                    < datetime.fromisoformat(start) + timedelta(hours=1)
                ),
                None,
            )
            if index is not None and index < len(soc_opt):
                ent = pipeline.resolve_entities(opts)
                actual = float(pipeline.ha_get(f"/api/states/{ent['soc']}")["state"])
                planned = float(soc_opt[index]) * 100.0
                if abs(actual - planned) > TRIGGER_SOC_DEVIATION_PP:
                    return f"SOC {actual:.0f}% вне плана {planned:.0f}%"
        except Exception:
            pass
    return None


def scheduler() -> None:
    time.sleep(60)  # let Home Assistant settle after add-on start
    while True:
        try:
            opts = json.loads((DATA / "options.json").read_text())
            tz = ZoneInfo(opts["time_zone"])
            hour, minute = (int(part) for part in opts["run_time"].split(":"))
            now = datetime.now(tz)
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            STATE["next_run"] = target.isoformat()
            today_report = DATA / "shadow_reports" / f"{now:%Y-%m-%d}.json"
            if now >= now.replace(hour=hour, minute=minute, second=0, microsecond=0):
                due = not today_report.exists()
            else:
                due = False
            if due and not STATE["running"]:
                threading.Thread(target=run_pipeline, daemon=True).start()
            # Event-driven recalculations between the scheduled daily runs.
            while datetime.now(tz) < target:
                remaining = (target - datetime.now(tz)).total_seconds()
                time.sleep(max(5, min(TICK_SECONDS, remaining)))
                if datetime.now(tz) >= target:
                    break
                if not STATE["running"]:
                    reason = _trigger_reason(opts)
                    if reason:
                        STATE["last_trigger"] = {
                            "reason": reason,
                            "at": datetime.now(UTC).isoformat(),
                        }
                        # A grid change is news; routine replans speak only
                        # when they wrote something or failed to.
                        notify = "always" if reason.startswith("сеть") else "changes"
                        threading.Thread(
                            target=run_pipeline, kwargs={"notify": notify}, daemon=True
                        ).start()
            threading.Thread(target=run_pipeline, daemon=True).start()
        except Exception:
            STATE["last_error"] = traceback.format_exc(limit=5)
            time.sleep(300)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path in ("/", "/index.html"):
            self._send(200, (APP / "panel.html").read_bytes(), "text/html; charset=utf-8")
        elif path.endswith("/api/latest") or path == "/api/latest":
            report = latest_report()
            if report is None:
                self._json(404, {"error": "no reports yet"})
            else:
                self._json(200, report)
        elif path.endswith("/api/history") or path == "/api/history":
            items = []
            for file in reports()[:30]:
                try:
                    report = json.loads(file.read_text(encoding="utf-8"))
                    items.append(
                        {
                            "date": file.stem,
                            "recommended_floor_soc": report["advisory"]["recommended_floor_soc"],
                            "current_floor_soc": report["advisory"]["current_floor_soc"],
                            "delta_pp": report["advisory"]["delta_pp"],
                            "reserve_kwh": report["reserve"]["reserve_kwh"],
                            "hold_until_local": report["reserve"]["hold_until_local"],
                        }
                    )
                except (OSError, ValueError, KeyError):
                    continue
            self._json(200, items)
        elif path.endswith("/api/daily") or path == "/api/daily":
            try:
                self._json(200, {"days": daily.daily_stats(14)})
            except Exception:
                self._json(200, {"days": [], "error": traceback.format_exc(limit=3)})
        elif path.endswith("/api/metrics") or path == "/api/metrics":
            metrics_path = DATA / "metrics.jsonl"
            days = []
            if metrics_path.exists():
                days = [
                    json.loads(line)
                    for line in metrics_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            recent = [d for d in days[-7:] if d.get("load") or d.get("pv")]

            def avg(key, field):
                values = [d[key][field] for d in recent if d.get(key) and d[key].get(field) is not None]
                return round(sum(values) / len(values), 2) if values else None

            self._json(
                200,
                {
                    "days": days[-14:],
                    "avg7": {
                        "load_mae_kwh": avg("load", "mae_kwh"),
                        "load_coverage_pct": avg("load", "coverage_pct"),
                        "pv_mae_kwh": avg("pv", "mae_kwh"),
                        "pv_coverage_pct": avg("pv", "coverage_pct"),
                        "days": len(recent),
                    },
                },
            )
        elif path.endswith("/api/entities") or path == "/api/entities":
            try:
                self._json(200, {"checks": pipeline.validate_entities()})
            except Exception:
                self._json(200, {"checks": [], "error": traceback.format_exc(limit=3)})
        elif path.endswith("/api/options") or path == "/api/options":
            try:
                self._json(200, {"groups": options_meta.options_view(pipeline.options())})
            except Exception:
                self._json(500, {"error": traceback.format_exc(limit=3)})
        elif path.endswith("/api/entity_ids") or path == "/api/entity_ids":
            try:
                states = pipeline.ha_get("/api/states", timeout=30)
                ids = sorted(s["entity_id"] for s in states if "entity_id" in s)
                self._json(200, {"entity_ids": ids})
            except Exception:
                self._json(200, {"entity_ids": [], "error": traceback.format_exc(limit=3)})
        elif path.endswith("/api/status") or path == "/api/status":
            audit_rows = None
            audit_path = DATA / "hourly_audit.jsonl"
            if audit_path.exists():
                with audit_path.open("rb") as file:
                    audit_rows = sum(1 for _ in file)
            self._json(
                200,
                {
                    "running": STATE["running"],
                    "last_error": STATE["last_error"],
                    "last_run": STATE["last_run"],
                    "next_run": STATE["next_run"],
                    "audit_rows": audit_rows,
                    "retrain": retrain.retrain_status(),
                    "last_backfill": STATE.get("last_backfill"),
                    "last_trigger": STATE.get("last_trigger"),
                    "grid_state": STATE.get("grid_state"),
                },
            )
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("/api/run"):
            if STATE["running"]:
                self._json(409, {"started": False, "reason": "already running"})
            else:
                threading.Thread(target=run_pipeline, daemon=True).start()
                self._json(202, {"started": True})
        elif path.endswith("/api/retrain"):
            if STATE["running"]:
                self._json(409, {"started": False, "reason": "already running"})
            else:
                threading.Thread(target=run_retrain, daemon=True).start()
                self._json(202, {"started": True})
        elif path.endswith("/api/backfill"):
            if STATE["running"]:
                self._json(409, {"started": False, "reason": "already running"})
            else:
                threading.Thread(target=run_backfill, daemon=True).start()
                self._json(202, {"started": True})
        elif path.endswith("/api/config"):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                self._json(200, pipeline.update_options(body.get("options") or {}))
            except ValueError as error:
                self._json(400, {"error": str(error)})
            except Exception:
                self._json(500, {"error": traceback.format_exc(limit=3)})
        else:
            self._json(404, {"error": "not found"})

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    threading.Thread(target=scheduler, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"panel listening on :{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
