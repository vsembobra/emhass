"""Today's and tomorrow's solar forecast against what the panels gave, with
the battery plan — the data behind the panel's two day charts.

Per local hour: the PV forecast the plan used for that hour (the latest run
issued before it started; for future hours the latest run), the measured
hourly mean, whether a full battery throttled the panels (then the measured
value is not a forecast miss), and the battery SOC — measured up to now,
planned by the optimizer from the current hour on. The current hour shows
the live PV power and SOC.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import learning
import pandas as pd
import pipeline


def _forecast(entries: list[dict], hour: pd.Timestamp) -> dict | None:
    """P10/P50/P90 of the latest run issued before ``hour`` that covers it."""
    key = hour.isoformat()
    for entry in reversed(entries):  # the log is in issue order
        if pd.Timestamp(entry["issue_utc"]) > hour:
            continue
        starts = entry.get("starts_utc") or []
        if key not in starts:
            continue
        index = starts.index(key)
        return {
            q: round(float((entry.get(f"pv_{q}_used") or entry["pv"][q])[index]), 3)
            for q in ("p10", "p50", "p90")
        }
    return None


def _soc_plan(report: dict | None) -> dict[str, float]:
    """Planned SOC (%) per hour start: the mean of the hour's two ends."""
    optimizer = ((report or {}).get("advisory") or {}).get("optimizer") or {}
    soc = optimizer.get("soc_opt") or []
    if not soc or not report.get("issue_utc"):
        return {}
    issue = pd.Timestamp(report["issue_utc"])
    plan = {}
    for i, end in enumerate(soc):
        start = soc[i - 1] if i else end
        hour = (issue + pd.Timedelta(hours=i)).isoformat()
        plan[hour] = round((float(start) + float(end)) / 2 * 100, 1)
    return plan


def _live(opts: dict) -> tuple[float | None, float | None]:
    """PV power (kW) and SOC (%) right now; None when HA cannot say."""
    ent = pipeline.resolve_entities(opts)

    def read(entity: str | None, scale: float) -> float | None:
        entity = str(entity or "").strip()
        if not entity:
            return None
        try:
            return round(float(pipeline.ha_get(f"/api/states/{entity}", timeout=10)["state"]) * scale, 3)
        except Exception:
            return None

    pv_entity, pv_scale = ent["mean_sensors"]["pv_measured_kw"]
    return read(pv_entity, pv_scale), read(ent["soc"], 1.0)


def day_charts(report: dict | None, now: datetime | None = None, live: bool = True) -> dict:
    opts = pipeline.options()
    tz = ZoneInfo(opts.get("time_zone", "UTC"))
    now = now or datetime.now(UTC)
    current = pd.Timestamp(now).floor("h")
    try:
        entries = learning.read_forecast_log(pipeline.DATA / "forecast_log.jsonl")
    except OSError:
        entries = []
    try:
        audit = pipeline._read_audit()
    except (OSError, ValueError):
        audit = pd.DataFrame()
    throttled = (
        (audit["pv_filter"] == "possible_curtailment") | pipeline.pv_sink_limited_mask(audit)
        if "pv_filter" in audit
        else pd.Series(dtype=bool)
    )
    plan = _soc_plan(report)
    live_pv, live_soc = _live(opts) if live else (None, None)

    def number(row, column) -> float | None:
        if row is None or column not in row or pd.isna(row[column]):
            return None
        return float(row[column])

    days = []
    today = pd.Timestamp(now).tz_convert(tz).normalize()
    for offset, label in ((0, "Сегодня"), (1, "Завтра")):
        start = today + pd.Timedelta(days=offset)
        local = pd.date_range(start, start + pd.Timedelta(days=1), freq="h", inclusive="left")
        hours = []
        for stamp in local.tz_convert(UTC):
            key = stamp.isoformat()
            row = audit.loc[stamp] if stamp in audit.index else None
            actual = number(row, "pv_measured_kw")
            soc = number(row, "soc_mean")
            is_current = stamp == current
            if is_current:
                actual = live_pv if live_pv is not None else actual
                soc = live_soc / 100.0 if live_soc is not None else soc
            hours.append(
                {
                    "start_utc": key,
                    "label": stamp.tz_convert(tz).strftime("%H"),
                    "forecast": _forecast(entries, stamp),
                    "actual_kw": None if actual is None else round(actual, 3),
                    "throttled": bool(stamp in throttled.index and throttled.loc[stamp]),
                    "soc_pct": None if soc is None else round(soc * 100, 1),
                    "soc_plan_pct": plan.get(key) if stamp >= current else None,
                    "past": stamp < current,
                    "current": is_current,
                }
            )
        p50 = [h["forecast"]["p50"] for h in hours if h["forecast"]]
        done = [h["actual_kw"] for h in hours if h["past"] and h["actual_kw"] is not None]
        ahead = sum(
            h["forecast"]["p50"] * (1 - now.minute / 60 if h["current"] else 1.0)
            for h in hours
            if h["forecast"] and not h["past"]
        )
        days.append(
            {
                "label": label,
                "date": start.strftime("%d.%m"),
                "hours": hours,
                "forecast_kwh": round(sum(p50), 1) if p50 else None,
                "actual_kwh": round(sum(done), 1) if done else None,
                "ahead_kwh": round(ahead, 1),
            }
        )
    return {
        "now_utc": now.isoformat(),
        "now_fraction": round(now.minute / 60, 3),
        "plan_note": None if plan else "нет плана оптимизатора — линия SOC только по факту",
        "days": days,
    }
