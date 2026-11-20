"""Tune the planner on the site's own history: replay, compare, pick.

For every deficit mode the owner's priority allows, the last two weeks are
replayed with :func:`emhass.battery_backtest.simulate`: the forecasts the
add-on actually issued (forecast log) drive an SOC floor policy, and the
measured PV, load, critical load, grid availability and tariff decide what
that policy would have achieved.

Replay windows run from local noon to the next noon, so evening, night and
morning — where the floor matters — are never cut at midnight; each starts
from the SOC measured at its first hour.

The policy mimics a TOU inverter such as Deye: the battery is held at the
zone floor, charges from surplus PV, and is grid-charged up to the floor in
cheap hours (or at any hour without a tariff, or when SOC sits below the
outage reserve). Candidates are ranked exactly in the add-on's priority
order: unserved critical energy during real outages, then curtailed PV,
then cost. A cheaper candidate that would have left the backup circuit dark
for longer always loses.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import optimize
import pandas as pd
import policy

from emhass.battery_backtest import Battery, Dispatch, ReplayStep, simulate
from emhass.battery_planning import ForecastStep, floor_profile
from emhass.reserve import ReserveSettings

TUNING_FILE = "tuning.json"
REPLAY_DAYS = 14
MIN_WINDOWS = 3
UNSERVED_TOLERANCE_KWH = 0.05
SOC_STEP = 0.05
HOUR = pd.Timedelta(hours=1)


def _soc_bounds(opts: dict) -> tuple[float, float]:
    return optimize.soc_bounds(opts)


def battery_from_options(opts: dict, soh: float = 1.0) -> Battery:
    min_soc, max_soc = _soc_bounds(opts)
    return Battery(
        capacity_kwh=float(opts["capacity_kwh"]) * soh,
        min_soc=min_soc,
        max_soc=max_soc,
        max_charge_kw=float(opts["max_charge_kw"]),
        max_discharge_kw=float(opts["max_discharge_kw"]),
        charge_efficiency=float(opts["charge_efficiency"]),
        discharge_efficiency=float(opts["discharge_efficiency"]),
        export_limit_kw=0.0,
    )


def settings_from_options(opts: dict, soh: float = 1.0) -> ReserveSettings:
    min_soc, max_soc = _soc_bounds(opts)
    return ReserveSettings(
        nominal_capacity_kwh=float(opts["capacity_kwh"]),
        soh=soh,
        floor_soc=min_soc,
        max_soc=max_soc,
        discharge_efficiency=float(opts["discharge_efficiency"]),
        cold_start_kw=0.5,
        site_timezone=str(opts.get("time_zone", "UTC")),
    )


def replay_windows(audit: pd.DataFrame, tz: ZoneInfo, since, until) -> list[list[dict]]:
    """Complete noon-to-noon windows of measured hours inside [since, until)."""
    needed = ("pv_measured_kw", "load_kw", "critical_load_kw", "soc_mean")
    if audit.empty or any(column not in audit for column in needed):
        return []
    frame = audit[(audit.index >= since) & (audit.index < until)]
    windows = []
    noons = sorted(
        {stamp.tz_convert(tz).normalize() + pd.Timedelta(hours=12) for stamp in frame.index}
    )
    for noon in noons:
        start = noon.tz_convert(UTC)
        hours = pd.date_range(start, periods=24, freq="h")
        if hours[0] < since or hours[-1] >= until:
            continue
        rows = frame.reindex(hours)
        values = rows[list(needed)].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            continue
        windows.append(
            [
                {
                    "start": hour.to_pydatetime(),
                    "pv_kw": max(0.0, float(row["pv_measured_kw"])),
                    "load_kw": max(0.0, float(row["load_kw"])),
                    "critical_kw": max(
                        0.0, min(float(row["critical_load_kw"]), float(row["load_kw"]))
                    ),
                    "soc": float(row["soc_mean"]),
                    "grid_up": rows.loc[hour].get("raw_grid_status") != "outage",
                }
                for hour, row in values.iterrows()
            ]
        )
    return windows


def issued_forecasts(log: list[dict]) -> list[tuple[pd.Timestamp, dict, pd.DatetimeIndex]]:
    """(issue, entry, starts) for every logged run that carries a reserve."""
    issued = []
    for entry in log:
        if "reserve_kwh_by_hour" not in entry:
            continue
        starts = pd.DatetimeIndex(pd.to_datetime(entry["starts_utc"], utc=True))
        issued.append((pd.Timestamp(entry["issue_utc"]).tz_convert(UTC), entry, starts))
    return sorted(issued, key=lambda item: item[0])


def _forecast_at(issued, at: pd.Timestamp):
    """Newest issued forecast covering ``at``: (entry, starts, step index)."""
    for issue, entry, starts in reversed(issued):
        if issue <= at and at in starts:
            return entry, starts, starts.get_loc(at)
    return None


def floor_at(
    entry: dict, starts: pd.DatetimeIndex, index: int, mode: str, settings: ReserveSettings
) -> tuple[float, float]:
    """(planning floor, outage-reserve floor) at the step, rounded up to 5 %."""
    load = {q: entry["load"][q][index:] for q in ("p10", "p50", "p90")}
    pv = {q: entry["pv"][q][index:] for q in ("p10", "p50", "p90")}
    deficit_load, deficit_pv = policy.deficit_inputs(mode, load, pv)
    starts = starts[index:]
    reserves = entry["reserve_kwh_by_hour"][index:]
    steps = [
        ForecastStep(start.to_pydatetime(), 1.0, float(p), float(l))
        for start, p, l in zip(starts, deficit_pv, deficit_load, strict=True)
    ]
    profile = floor_profile(steps, reserves, settings)
    reserve_floor = settings.floor_for_energy(float(reserves[0]))[0]

    def up(value: float) -> float:
        return math.ceil(value / SOC_STEP - 1e-9) * SOC_STEP

    return up(profile.steps[0].floor_soc), up(reserve_floor)


def make_policy(
    window: list[dict],
    issued: list,
    mode: str,
    battery: Battery,
    settings: ReserveSettings,
    opts: dict,
    tz: ZoneInfo,
):
    by_start = {row["start"]: row for row in window}
    cheap = optimize._cheap_windows(opts) if opts.get("tariff_enabled", False) else None

    def decide(start: datetime, soc: float) -> Dispatch:
        found = _forecast_at(issued, pd.Timestamp(start))
        if found is None:
            return Dispatch(floor_soc=battery.min_soc)
        floor, reserve = floor_at(*found, mode, settings)
        floor = min(max(floor, battery.min_soc), battery.max_soc)
        reserve = min(max(reserve, battery.min_soc), battery.max_soc)
        if soc >= floor - 1e-9:
            return Dispatch(floor_soc=floor)
        local = pd.Timestamp(start).tz_convert(tz)
        minute = local.hour * 60 + local.minute
        allowed = cheap is None or optimize._is_cheap_minute(minute, cheap) or soc < reserve
        if not allowed:
            return Dispatch(floor_soc=floor)
        row = by_start[start]
        surplus = max(0.0, row["pv_kw"] - row["load_kw"])
        topup = (floor - soc) * battery.capacity_kwh / battery.charge_efficiency
        return Dispatch(floor_soc=floor, battery_kw=-min(battery.max_charge_kw, surplus + topup))

    return decide


def replay(
    windows: list[list[dict]],
    issued: list,
    mode: str,
    battery: Battery,
    settings: ReserveSettings,
    opts: dict,
    tz: ZoneInfo,
) -> dict:
    prices = optimize.tariff_prices
    totals = {
        "unserved_critical_kwh": 0.0,
        "curtailed_kwh": 0.0,
        "imported_kwh": 0.0,
        "net_cost": 0.0,
    }
    for window in windows:
        starts = pd.DatetimeIndex([row["start"] for row in window])
        price = prices(starts, tz, opts)
        steps = [
            ReplayStep(
                start=row["start"],
                hours=1.0,
                pv_kw=row["pv_kw"],
                load_kw=row["load_kw"],
                critical_load_kw=row["critical_kw"],
                grid_up=bool(row["grid_up"]),
                import_price=float(p),
                export_price=0.0,
            )
            for row, p in zip(window, price, strict=True)
        ]
        initial = min(max(window[0]["soc"], battery.min_soc), battery.max_soc)
        result = simulate(
            steps, battery, initial, make_policy(window, issued, mode, battery, settings, opts, tz)
        )
        totals["unserved_critical_kwh"] += result.unserved_critical_kwh
        totals["curtailed_kwh"] += result.curtailed_kwh
        totals["imported_kwh"] += result.imported_kwh
        totals["net_cost"] += result.net_cost or 0.0
    return {key: round(value, 3) for key, value in totals.items()}


def rank_key(metrics: dict) -> tuple:
    """Priority order: outage service, own solar, money."""
    unserved = round(metrics["unserved_critical_kwh"] / UNSERVED_TOLERANCE_KWH)
    return (unserved, round(metrics["curtailed_kwh"], 1), round(metrics["net_cost"], 2))


def tune(
    audit: pd.DataFrame, log: list[dict], opts: dict, tz: ZoneInfo, now: datetime, soh: float = 1.0
) -> dict:
    """Replay every allowed deficit mode and pick the best one."""
    issued = issued_forecasts(log)
    until = pd.Timestamp(now).tz_convert(UTC).floor("h")
    since = until - pd.Timedelta(days=REPLAY_DAYS)
    windows = replay_windows(audit, tz, since, until)
    allowed = policy.priority(opts)["deficit_modes"]
    result: dict = {
        "tuned_utc": pd.Timestamp(now).isoformat(),
        "priority_mode": str(opts.get("priority_mode", policy.DEFAULT_PRIORITY)),
        "windows": len(windows),
        "candidates": {},
        "best": None,
    }
    if len(windows) < MIN_WINDOWS:
        result["reason"] = f"нужно минимум {MIN_WINDOWS} полных суток истории с прогнозами"
        return result
    battery = battery_from_options(opts, soh)
    settings = settings_from_options(opts, soh)
    for mode in allowed:
        result["candidates"][mode] = replay(windows, issued, mode, battery, settings, opts, tz)
    result["best"] = min(
        allowed, key=lambda mode: (rank_key(result["candidates"][mode]), allowed.index(mode))
    )
    return result


def load(data_dir: Path) -> dict | None:
    try:
        return json.loads((data_dir / TUNING_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save(data_dir: Path, tuning: dict) -> None:
    (data_dir / TUNING_FILE).write_text(
        json.dumps(tuning, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def due(data_dir: Path, every: timedelta = timedelta(hours=24)) -> bool:
    tuning = load(data_dir)
    if not tuning or not tuning.get("tuned_utc"):
        return True
    return datetime.now(UTC) - datetime.fromisoformat(tuning["tuned_utc"]) >= every


def tuned_mode(opts: dict, tuning: dict | None) -> str:
    """Deficit mode for this run: the tuned one under autotune, if allowed."""
    allowed = policy.priority(opts)["deficit_modes"]
    if opts.get("autotune") and tuning and tuning.get("best") in allowed:
        return str(tuning["best"])
    return policy.default_deficit_mode(opts)
