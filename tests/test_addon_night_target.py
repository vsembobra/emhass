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
from zoneinfo import ZoneInfo

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
        pv_scenario_kw=_sun(starts.hour, 4.0),
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


# Owner's report of 23.11 01:00 (Europe/Istanbul): the battery at 95 % covers
# both days, the plan buys nothing for 48 h, sun P50 4.3 kWh, P10 0.
REPORT_LOAD = [
    0.424, 0.3621, 0.3265, 0.3213, 0.3508, 0.3963, 0.4143, 0.4499, 0.6512, 0.7633, 0.7677,
    0.8371, 0.9029, 0.985, 0.9321, 0.8464, 0.7971, 0.7521, 0.6909, 0.6729, 0.6407, 0.5941,
    0.5849, 0.4979,
]  # fmt: skip
REPORT_SUN = [0.0] * 7 + [0.0551, 0.1344, 0.504, 1.6416, 0.9471, 0.6036, 0.3144, 0.115] + [0.0] * 9


def test_a_battery_that_covers_a_dull_day_is_not_held_full_overnight():
    starts = pd.date_range("2026-11-22T22:00Z", periods=48, freq="h")
    floors = ([0.30] * 8 + [0.35] * 9 + [0.30] * 7) * 2
    zones = [
        {"n": n, "from": f, "soc_pct": s}
        for n, (f, s) in enumerate(
            [
                ("00:00", 95),
                ("07:00", 30),
                ("08:00", 35),
                ("11:00", 35),
                ("20:00", 30),
                ("22:00", 30),
            ],
            1,
        )
    ]
    result = optimize.run_plan(
        starts,
        REPORT_SUN + list(np.array(REPORT_SUN) * 11.4 / 4.32),
        REPORT_LOAD * 2,
        floors,
        0.95,
        zones,
        dict(
            OPTS,
            time_zone="Europe/Istanbul",
            min_soc_pct=20,
            capacity_kwh=31.7,
            tariff_cheap_windows="23:00-07:00",  # as the report's prices show
        ),
        reserve_floors=floors,
        pv_scenario_kw=[0.0] * 24 + list(np.array(REPORT_SUN) * 2.92 / 4.32),
    )
    assert sum(result["p_grid_kw"]) == pytest.approx(0.0, abs=0.05)  # nothing to buy
    assert result["night_margin_kwh"] == 0.0  # a dull day is covered anyway
    # The zone holds what the day needs even with no sun (~69 %), not the plan's
    # own 88 % at 07:00: the house runs on the battery at night, whatever it draws.
    # It was held at 95 % before.
    assert _night_zone(result) == 70


def test_dawn_sun_in_the_cheap_window_is_not_a_grid_charge_target():
    # Summer: the battery at its 20 % floor at 23:00, a big sunny day ahead
    # whose first sun (05:00-07:00) falls inside the cheap window. The plan buys
    # nothing; the sun lifts the SOC at dawn. A charge target there made the
    # inverter buy at 00:00 what the sun brings for free.
    starts = pd.date_range("2026-06-20 23:00", periods=48, freq="h", tz="UTC")
    hours = np.asarray(starts.hour, dtype=float)
    sun = np.where((hours >= 5) & (hours < 20), np.sin((hours - 4.5) / 15 * np.pi), 0.0)
    sun = sun / sun[:24].sum() * 40.0
    floors = [0.2] * 48
    result = optimize.run_plan(
        starts, list(sun), [15.0 / 24] * 48, floors, 0.2, [dict(z) for z in ZONES],
        OPTS, reserve_floors=floors,  # no P10: the zone logic alone, no night adjustment
    )  # fmt: skip
    assert result["grid_charge_kwh_cheap"] == pytest.approx(0.0, abs=0.05)
    assert _night_zone(result) == 20


def test_a_replan_in_the_last_calibration_hour_keeps_a_sane_horizon_end():
    # 06:00, the calibration window's last hour: the pipeline lifts floor[0] to
    # 100 %. That floor must not become the end-of-horizon target as well.
    starts = pd.date_range("2026-11-24 06:00", periods=48, freq="h", tz="UTC")
    floors = [1.0] + [FLOOR] * 47
    result = optimize.run_plan(
        starts, _sun(starts.hour, 8.0), [15.0 / 24] * 48, floors, 0.85,
        [dict(z) for z in ZONES], OPTS, reserve_floors=[FLOOR] * 48,
        pv_scenario_kw=_sun(starts.hour, 4.0),
    )  # fmt: skip
    assert not result["floor_relaxed"]
    assert result["soc_opt"][-1] < 0.9  # not held high for two days to refill to 100 %


def _night_for(pv_q: float, load_q: float) -> int:
    """Night zone for the owner's example with the scenario sun and load scaled."""
    starts = pd.date_range("2026-11-23 23:00", periods=48, freq="h", tz="UTC")
    result = optimize.run_plan(
        starts, _sun(starts.hour, 8.0), [15.0 / 24] * 48, [FLOOR] * 48, 0.40,
        [dict(z) for z in ZONES], OPTS, reserve_floors=[FLOOR] * 48,
        pv_scenario_kw=_sun(starts.hour, 8.0 * pv_q),
        load_scenario_kw=[15.0 / 24 * load_q] * 48,
    )  # fmt: skip
    return _night_zone(result)


def test_the_priority_scenario_moves_the_night_charge():
    economy = _night_for(1.5, 0.8)  # sun P90, load P10: an optimistic day
    balanced = _night_for(1.0, 1.0)  # the median day: the plan's own need
    resilience = _night_for(0.5, 1.2)  # sun P10, load P90: a pessimistic day
    assert economy < balanced < resilience
    assert balanced == 50  # ~45 % by hand on the median day, the 5 % step up


def test_a_zone_reaching_into_expensive_hours_gets_no_hold_target_at_23():
    # Owner's panel, 24.11 23:00 (Europe/Istanbul): zone 6 runs 22:00-24:00,
    # cheap from 23:00. In effect at 23:xx its rest is cheap, and it was given
    # the plan's level (80 %) - a value that stays in the inverter through
    # tomorrow's expensive 22:00 hour. Its whole window decides: min SOC plus
    # the reserve, and no grid charge.
    import adapter

    tz = "Europe/Istanbul"
    starts = pd.date_range(pd.Timestamp("2026-11-24 23:00", tz=tz), periods=48, freq="h")
    starts = starts.tz_convert("UTC")
    local = starts.tz_convert(tz).hour
    floors = [0.40 if (h >= 18 or h < 8) else 0.35 for h in local]
    zones = [
        {"n": n, "from": f, "soc_pct": s}
        for n, (f, s) in enumerate(
            [
                ("00:00", 50),
                ("07:00", 35),
                ("08:00", 40),
                ("11:00", 45),
                ("20:00", 40),
                ("22:00", 40),
            ],
            1,
        )
    ]
    opts = dict(
        OPTS, time_zone=tz, min_soc_pct=20, capacity_kwh=31.7, tariff_cheap_windows="23:00-07:00"
    )
    sun = _sun(local, 9.48)
    result = optimize.run_plan(
        starts, sun, [15.27 / 24] * 48, floors, 0.81, [dict(z) for z in zones], opts,
        reserve_floors=floors, pv_scenario_kw=_sun(local, 18.0), load_scenario_kw=[9.16 / 24] * 48,
    )  # fmt: skip
    targets = {t["n"]: t["optimal_soc_pct"] for t in result["zone_targets"]}
    assert targets[6] == 40
    flags = adapter.grid_charge_plan(
        [dict(z) for z in zones], starts[:24], ZoneInfo(tz), opts, floors[:24], sun[:24],
        result["soc_opt"][:24], 0.81,
    )  # fmt: skip
    assert not next(f for f in flags if f["n"] == 6)["desired"]
