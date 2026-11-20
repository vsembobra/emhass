"""When to run flexible loads (boiler, washer, pool pump) on your own sun.

With feed-in closed, every kWh of PV the battery cannot take is curtailed:
that energy is free. The optimizer's plan says how much it expects to
curtail each hour; a flexible load started there runs on sun that would
otherwise be thrown away. Next best are surplus hours (the sun covers the
load, at the price of charging the battery a bit less).

The add-on only recommends and, when sensor publishing is on, exposes the
recommendation as a sensor; switching the load is left to the owner's
automation (Home Assistant, Node-RED, ...).
"""

from __future__ import annotations

import math
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

MIN_SOLAR_SHARE = 0.3  # below this the best window is reported as "no sun"


def _window_minutes(raw: str | None) -> tuple[int, int] | None:
    if not raw or "-" not in str(raw):
        return None
    begin, end = str(raw).split("-", 1)
    try:
        bh, bm = (int(x) for x in begin.strip().split(":"))
        eh, em = (int(x) for x in end.strip().split(":"))
    except ValueError:
        return None
    return bh * 60 + bm, eh * 60 + em


def _inside(minute: int, window: tuple[int, int] | None) -> bool:
    if window is None:
        return True
    begin, end = window
    return begin <= minute < end if begin < end else minute >= begin or minute < end


def slug(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return text or "load"


def recommend(
    loads: list[dict],
    starts: pd.DatetimeIndex,
    tz: ZoneInfo,
    curtail_kw,
    surplus_kw,
) -> list[dict]:
    """Best start per load inside its window (whole hours, first day only).

    Ranking: most free (curtailed) solar, then most surplus solar, then the
    earliest start. A load longer than its window gets no recommendation.
    """
    curtail = np.maximum(np.asarray(curtail_kw, dtype=float), 0.0)
    surplus = np.maximum(np.asarray(surplus_kw, dtype=float), 0.0)
    local = pd.DatetimeIndex(starts).tz_convert(tz)
    minutes = local.hour * 60 + local.minute
    out = []
    for load in loads:
        power = float(load.get("power_kw") or 0.0)
        hours = float(load.get("hours") or 0.0)
        if power <= 0 or hours <= 0:
            continue
        steps = max(1, math.ceil(hours))
        window = _window_minutes(load.get("window"))
        best = None
        for first in range(len(starts) - steps + 1):
            span = range(first, first + steps)
            if not all(_inside(int(minutes[i]), window) for i in span):
                continue
            # The last step may be partial (e.g. 1.5 h -> 2 steps, last half).
            fractions = [min(1.0, hours - k) for k in range(steps)]
            free = sum(min(power, curtail[i]) * f for i, f in zip(span, fractions, strict=True))
            sunny = sum(
                min(max(0.0, power - curtail[i]), surplus[i]) * f
                for i, f in zip(span, fractions, strict=True)
            )
            key = (round(free, 3), round(sunny, 3), -first)
            if best is None or key > best[0]:
                best = (key, first, free, sunny)
        name = str(load.get("name") or "load")
        if best is None:
            out.append(
                {
                    "name": name,
                    "slug": slug(name),
                    "start_local": None,
                    "reason": "окно короче нагрузки",
                }
            )
            continue
        _, first, free, sunny = best
        free, sunny = float(free), float(sunny)  # plain floats: the report is JSON
        energy = power * hours
        share = (free + sunny) / energy if energy else 0.0
        end = local[first] + pd.Timedelta(hours=hours)
        out.append(
            {
                "name": name,
                "slug": slug(name),
                "start_local": local[first].strftime("%H:%M"),
                "end_local": end.strftime("%H:%M"),
                "start_utc": pd.Timestamp(starts[first]).isoformat(),
                "energy_kwh": round(energy, 2),
                "free_solar_kwh": round(free, 2),
                "surplus_solar_kwh": round(sunny, 2),
                "solar_share": round(share, 3),
                "sunny": bool(share >= MIN_SOLAR_SHARE),
            }
        )
    return out
