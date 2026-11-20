"""Periodic full charge so the BMS can re-anchor its state of charge.

A BMS counts charge through a shunt and only re-calibrates its SOC at the
top of the charge. Kept below full for weeks, the counting error grows and
the reported SOC — which every reserve and plan here relies on — drifts.

So when the battery has not been full (SOC >= 99 %) for
``calibration_interval_days`` (14 by default), the next night window
charges it to 100 %: the cheap tariff hours when a tariff is enabled, else
00:00-07:00. Any full charge resets the count, whether from the sun in
summer, an owner automation or a previous calibration — no grid energy is
bought when the battery got full on its own. Optional
(``calibration_charge_enabled``).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import optimize
import pandas as pd

FULL_SOC = 0.99
DEFAULT_INTERVAL_DAYS = 14
DEFAULT_NIGHT = [(0, 7 * 60)]  # 00:00-07:00 when no tariff is configured


def enabled(opts: dict) -> bool:
    return bool(opts.get("calibration_charge_enabled", True))


def interval_days(opts: dict) -> float:
    try:
        return min(
            60.0, max(3.0, float(opts.get("calibration_interval_days", DEFAULT_INTERVAL_DAYS)))
        )
    except (TypeError, ValueError):
        return float(DEFAULT_INTERVAL_DAYS)


def last_full_charge(audit: pd.DataFrame, live_soc: float | None, now: datetime) -> datetime | None:
    """Most recent hour the battery was full (audit soc_max or the live SOC)."""
    if live_soc is not None and live_soc >= FULL_SOC:
        return now
    if "soc_max" not in audit or audit.empty:
        return None
    full = pd.to_numeric(audit["soc_max"], errors="coerce") >= FULL_SOC
    if not full.any():
        return None
    return audit.index[full.to_numpy()][-1].to_pydatetime()


def night_window(starts: pd.DatetimeIndex, tz: ZoneInfo, opts: dict) -> list[int]:
    """Indices of the first night window in the horizon (one contiguous run)."""
    windows = optimize._cheap_windows(opts) if opts.get("tariff_enabled", False) else []
    windows = windows or DEFAULT_NIGHT
    run: list[int] = []
    for i, stamp in enumerate(starts):
        local = stamp.tz_convert(tz)
        inside = optimize._is_cheap_minute(local.hour * 60 + local.minute, windows)
        if inside:
            run.append(i)
        elif run:
            break
    return run


def status(audit: pd.DataFrame, live_soc: float | None, now: datetime, opts: dict) -> dict:
    """Is a calibration charge due? With the facts behind the answer."""
    days = interval_days(opts)
    last = last_full_charge(audit, live_soc, now)
    since = None if last is None else (now - last).total_seconds() / 86400
    info = {
        "enabled": enabled(opts),
        "interval_days": days,
        "last_full_utc": last.isoformat() if last else None,
        "days_since_full": None if since is None else round(since, 1),
        "next_due_utc": (last + timedelta(days=days)).isoformat() if last else None,
    }
    info["due"] = info["enabled"] and (since is None or since >= days)
    return info


def zones_in_window(
    programs: list[dict], starts: pd.DatetimeIndex, tz, window: list[int]
) -> set[int]:
    """TOU zones whose current hours overlap the calibration window."""
    hits = set(window)
    zones = optimize.zone_windows(programs, starts, tz)
    return {
        program["n"]
        for program in programs
        if zones.get(program["n"]) is not None and hits & set(zones[program["n"]].now[0])
    }
