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
import pipeline
import retrain

DATA = Path("/data")
APP = Path(__file__).resolve().parent
PORT = 8099

STATE = {"running": False, "last_error": None, "last_run": None, "next_run": None}
LOCK = threading.Lock()


def run_pipeline(retrain_first: bool = True) -> None:
    with LOCK:
        if STATE["running"]:
            return
        STATE["running"] = True
        STATE["last_error"] = None
    try:
        if retrain_first:
            try:
                retrain.retrain_if_due()
            except Exception:
                # Retraining must never block the daily plan.
                pass
        pipeline.run()
        STATE["last_run"] = datetime.now(UTC).isoformat()
    except Exception:
        STATE["last_error"] = traceback.format_exc(limit=5)
    finally:
        STATE["running"] = False


def run_retrain() -> None:
    with LOCK:
        if STATE["running"]:
            return
        STATE["running"] = True
        STATE["last_error"] = None
    try:
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
            while datetime.now(tz) < target:
                time.sleep(30)
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
