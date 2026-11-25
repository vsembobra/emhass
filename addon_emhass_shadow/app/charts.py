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

import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import data_store
import learning
import optimize
import pandas as pd
import pipeline


def _forecast(entries: list[dict], hour: pd.Timestamp, source: str = "pv") -> dict | None:
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
            q: round(float((entry.get(f"{source}_{q}_used") or entry[source][q])[index]), 3)
            for q in ("p10", "p50", "p90")
        }
    return None


def _soc_plan(reports: list[dict]) -> dict[str, float]:
    """Planned SOC (%) per hour start: the mean of the hour's two ends.

    Every hour takes the newest plan issued at or before it, so past hours
    show what was actually planned for them and can be compared against the
    measured SOC. Reports older than any plan covering an hour leave None.
    """
    issued = []
    seen = set()
    for report in reports:
        optimizer = ((report or {}).get("advisory") or {}).get("optimizer") or {}
        soc = [float(v) for v in optimizer.get("soc_opt") or []]
        issue_text = (report or {}).get("issue_utc")
        if soc and issue_text and issue_text not in seen:
            seen.add(issue_text)
            issued.append((pd.Timestamp(issue_text), soc))
    issued.sort(key=lambda item: item[0])
    plan: dict[str, float] = {}
    for k, (issue, soc) in enumerate(issued):
        stop = issued[k + 1][0] if k + 1 < len(issued) else None
        for i, end in enumerate(soc):
            hour = issue + pd.Timedelta(hours=i)
            if stop is not None and hour >= stop:
                break
            start = soc[i - 1] if i else end
            plan[hour.isoformat()] = round((start + end) / 2 * 100, 1)
    return plan


def _hourly_plan(reports: list[dict], values_of) -> dict[str, float]:
    """Per-hour values of the newest plan issued at or before each hour.

    Same stitching as _soc_plan: past hours show what the run covering them
    planned, so the chart can compare plan against fact. ``values_of``
    extracts the per-hour list (one value per hour from the issue) from a
    report, or None when the report does not carry it.
    """
    issued = []
    seen = set()
    for report in reports:
        values = values_of(report or {})
        issue_text = (report or {}).get("issue_utc")
        if values and issue_text and issue_text not in seen:
            seen.add(issue_text)
            issued.append((pd.Timestamp(issue_text), values))
    issued.sort(key=lambda item: item[0])
    plan: dict[str, float] = {}
    for k, (issue, values) in enumerate(issued):
        stop = issued[k + 1][0] if k + 1 < len(issued) else None
        for i, value in enumerate(values):
            hour = issue + pd.Timedelta(hours=i)
            if stop is not None and hour >= stop:
                break
            plan[hour.isoformat()] = round(float(value), 3)
    return plan


def _floor_plan(reports: list[dict]) -> dict[str, float]:
    """Planned SOC floor (%) per hour from the reports' floor_profile."""

    def values(report: dict):
        profile = (report.get("advisory") or {}).get("floor_profile") or []
        floors = [step.get("floor_soc") for step in profile]
        if not floors or any(f is None for f in floors):
            return None
        return [float(f) * 100 for f in floors]

    return _hourly_plan(reports, values)


def _grid_plan(reports: list[dict]) -> dict[str, float]:
    """Planned grid import (kW) per hour from the optimizer's p_grid_kw."""

    def values(report):
        return ((report.get("advisory") or {}).get("optimizer") or {}).get("p_grid_kw")

    return _hourly_plan(reports, values)


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
    # The latest plan covers only the hours since its issue; earlier hours of
    # the day get their plan from the archived reports of previous runs.
    reports = [report] if report else []
    try:
        for path in data_store.json_paths(pipeline.DATA / "shadow_reports")[-3:]:
            try:
                reports.append(data_store.read_json(path))
            except (OSError, json.JSONDecodeError):
                continue
    except OSError:
        pass
    plan = _soc_plan(reports)
    floor_plan = _floor_plan(reports)
    grid_plan = _grid_plan(reports)
    cheap_windows = optimize._cheap_windows(opts) if opts.get("tariff_enabled", False) else []
    live_pv, live_soc = _live(opts) if live else (None, None)

    def number(row, column) -> float | None:
        if row is None or column not in row or pd.isna(row[column]):
            return None
        return float(row[column])

    days = []
    today = pd.Timestamp(now).tz_convert(tz).normalize()
    for offset, label, label_en in ((0, "Сегодня", "Today"), (1, "Завтра", "Tomorrow")):
        start = today + pd.Timedelta(days=offset)
        local = pd.date_range(start, start + pd.Timedelta(days=1), freq="h", inclusive="left")
        hours = []
        for stamp in local.tz_convert(UTC):
            key = stamp.isoformat()
            row = audit.loc[stamp] if stamp in audit.index else None
            actual = number(row, "pv_measured_kw")
            soc = number(row, "soc_mean")
            grid_actual = number(row, "grid_power_kw_raw_sign")
            outage = row is not None and (
                row.get("raw_grid_status") == "outage"
                or row.get("grid_quality") in ("recorded_outage", "low_grid_voltage")
            )
            wall = stamp.tz_convert(tz)
            cheap = bool(cheap_windows) and optimize._is_cheap_minute(
                wall.hour * 60 + wall.minute, cheap_windows
            )
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
                    "load_actual_kw": number(row, "load_kw"),
                    "load_forecast_kw": (_forecast(entries, stamp, "load") or {}).get("p50"),
                    "throttled": bool(stamp in throttled.index and throttled.loc[stamp]),
                    "soc_pct": None if soc is None else round(soc * 100, 1),
                    "soc_plan_pct": plan.get(key),
                    "floor_plan_pct": floor_plan.get(key),
                    "grid_actual_kw": None if grid_actual is None else round(grid_actual, 3),
                    "grid_plan_kw": grid_plan.get(key),
                    "cheap": cheap,
                    "outage": bool(outage),
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
                "label_en": label_en,
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
        "plan_note_en": None if plan else "no optimizer plan — the SOC line is actuals only",
        "days": days,
    }
