"""The night charge target: what the day needs, bought once.

Owner's example: 31 kWh, 15 kWh a day spread evenly, min SOC 20 %, outage
reserve 3 kWh (floor 35 % after the 5 % step), sun 8 kWh median / 4 kWh
pessimistic, cheap 00:00-07:00. By hand: 17 expensive hours ~ 11 kWh + 3 kWh
reserve - 4 kWh sun = 10 kWh ~ 33 % over the 20 % minimum, so about 55-60 %.
The shipped plan bought part of tomorrow's energy tonight (both nights cost
the same) and every night replan put the cloudy-day margin on top again:
70 % at midnight, 85 % by 02:00.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import optimize  # noqa: E402

CAP, KW, ETA, FLOOR = 31.0, 6.1, 0.95, 0.35
ZONES = [
    {"n": n, "from": f, "soc_pct": 50}
    for n, f in enumerate(["00:00", "07:00", "08:00", "11:00", "20:00", "22:00"], 1)
]
OPTS = {
    "time_zone": "UTC",
    "min_soc_pct": 20,
    "max_soc_pct": 100,
    "capacity_kwh": CAP,
    "charge_efficiency": ETA,
    "discharge_efficiency": ETA,
    "max_charge_kw": KW,
    "max_discharge_kw": 17.9,
    "tariff_enabled": True,
    "tariff_normal_price": 2.0,
    "tariff_cheap_price": 1.0,
    "tariff_cheap_windows": "00:00-07:00",
    "day_zone_strategy": "flat",
}


def _sun(hours, total):
    x = np.asarray(hours, dtype=float)
    shape = np.where((x >= 8) & (x < 16), np.sin((x - 7.5) / 8 * np.pi), 0.0)
    return list(shape / shape[:24].sum() * total)


def _plan(issue: str, soc: float) -> tuple[dict, pd.DatetimeIndex]:
    starts = pd.date_range(issue, periods=48, freq="h", tz="UTC")
    result = optimize.run_plan(
        starts,
        _sun(starts.hour, 8.0),
        [15.0 / 24] * 48,
        [FLOOR] * 48,
        soc,
        [dict(z) for z in ZONES],
        OPTS,
        reserve_floors=[FLOOR] * 48,
        pv_p10_kw=_sun(starts.hour, 4.0),
    )
    assert result is not None and not result["floor_relaxed"]
    return result, starts


def _night_zone(result: dict) -> int:
    return next(t for t in result["zone_targets"] if t["n"] == 1)["optimal_soc_pct"]


def test_the_night_charge_covers_the_day_and_no_more():
    result, starts = _plan("2026-11-23 23:00", 0.40)
    assert 55 <= _night_zone(result) <= 60  # by hand ~55 %, the floor's 5 % step on top
    # Nothing carried into tomorrow: the day drains to the floor before the next cheap night.
    day = [
        s for s, t in zip(result["soc_opt"], starts, strict=False) if 7 <= t.hour and t.day == 24
    ]
    assert min(day) <= FLOOR + 0.05


def test_night_replans_do_not_stack_the_cloudy_day_margin():
    soc, targets = 0.40, []
    for hour in (0, 2, 4, 6):
        result, _ = _plan(f"2026-11-24 {hour:02d}:00", soc)
        targets.append(_night_zone(result))
        for _ in range(2):  # the inverter grid-charges toward the zone's Batt%
            soc = min(targets[-1] / 100, soc + KW * ETA / CAP)
    assert max(targets) == targets[0], targets
    assert soc <= 0.60


def test_costs_are_reported_at_the_real_prices():
    result, _ = _plan("2026-11-23 23:00", 0.40)
    bill = sum(g * p for g, p in zip(result["p_grid_kw"], result["prices"], strict=True))
    assert result["prices"][0] == 2.0 and result["prices"][1] == 1.0  # untilted
    assert result["cost_plan"] == pytest.approx(bill, abs=0.05)


def test_the_tie_break_never_reorders_tariffs():
    prices = [2.0] * 24 + [1.0] * 7 + [2.0] * 17
    tilted = optimize.later_is_cheaper(prices)
    assert tilted[-1] == prices[-1]
    assert all(
        t >= p and t <= p * (1 + optimize.TIE_BREAK) for t, p in zip(tilted, prices, strict=True)
    )
    assert tilted[0] > tilted[23] > tilted[47]  # the same tariff a hair cheaper later
    assert max(tilted[24:31]) < min(tilted[:24])  # cheap stays cheaper than normal
