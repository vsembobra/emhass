"""Flat day zones at the lowest allowed SOC; night charge sized on PV P10."""

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

UTC = ZoneInfo("UTC")
STARTS = pd.date_range("2026-11-21 00:00", periods=24, freq="h", tz="UTC")
PROGRAMS = [
    {"n": n, "from": f, "soc_pct": 45}
    for n, f in enumerate(["00:00", "07:00", "08:00", "11:00", "20:00", "22:00"], 1)
]
NIGHT = [(0, 7 * 60)]
# Owner's day: charge 30 -> 63 % by 07:00, spend it slowly to 46 % by 24:00.
CHARGE = list(np.linspace(0.35, 0.63, 7))
SPEND = list(np.linspace(0.62, 0.46, 17))


def _zones(strategy, soc=CHARGE + SPEND, soc_init=0.3, cheap=NIGHT, reserve=None):
    out = optimize.zone_targets(
        soc,
        soc_init,
        PROGRAMS,
        STARTS,
        UTC,
        0.1,
        1.0,
        strategy=strategy,
        cheap_windows=cheap,
        reserve_floors=reserve,
    )
    return [t["optimal_soc_pct"] for t in out]


def test_ladder_pins_the_day_to_the_plan_flat_frees_it():
    # The reported staircase: every day zone held above the plan's low.
    assert _zones("ladder") == [65, 65, 60, 50, 50, 50]
    # Flat: the night charge target, then one floor = min SOC for the day.
    assert _zones("flat") == [65, 10, 10, 10, 10, 10]


def test_flat_day_floor_is_the_outage_reserve_while_a_series_lives():
    assert _zones("flat", reserve=[0.4] * 24) == [65, 40, 40, 40, 40, 40]
    # A risk window 20:00-22:00 lifts only the zone that covers it.
    risky = [0.4] * 20 + [0.55] * 2 + [0.4] * 2
    assert _zones("flat", reserve=risky) == [65, 40, 40, 40, 55, 40]


def test_flat_night_zone_without_a_charge_holds_the_battery():
    hold = [0.7] * 7 + list(np.linspace(0.68, 0.4, 17))
    assert _zones("flat", soc=hold, soc_init=0.7)[0] == 70


def test_flat_without_tariff_every_zone_at_the_floor():
    assert _zones("flat", cheap=[]) == [10] * 6


def test_flat_zone_in_effect_still_restores_the_reserve():
    starts = pd.date_range("2026-11-21 12:00", periods=24, freq="h", tz="UTC")
    out = optimize.zone_targets(
        [0.5] * 24,
        0.3,
        PROGRAMS,
        starts,
        UTC,
        0.1,
        1.0,
        strategy="flat",
        cheap_windows=NIGHT,
        reserve_floors=[0.4] * 24,
    )
    assert {t["n"]: t["optimal_soc_pct"] for t in out}[4] == 40  # 11:00 zone, now


def test_cheap_runs_cross_midnight():
    starts = pd.date_range("2026-11-21 21:00", periods=30, freq="h", tz="UTC")
    runs = optimize.cheap_runs(starts, UTC, [(23 * 60, 7 * 60)])
    assert runs == [(2, 9), (26, 29)]


MARGIN_FLOORS = [0.3] * 24
SUN_P50 = [0.0] * 9 + [2.0] * 4 + [0.0] * 11  # 8 kWh from 09:00 to 13:00
SUN_P10 = [0.0] * 9 + [0.5] * 4 + [0.0] * 11  # 6 kWh less = 20 pp of 30 kWh


def _margin(soc_p50, runs=((0, 6),), grid_charging=None):
    dull = [p50 - p10 for p50, p10 in zip(SUN_P50, SUN_P10, strict=True)]  # extra drain, kW
    return optimize.night_charge_adjustment(
        soc_p50, MARGIN_FLOORS, dull, list(runs), 1.0, 30.0, 1.0, grid_charging
    )


def test_margin_covers_the_dip_below_the_floor_on_a_dull_day():
    # The median plan lands on the floor before the next night: without the
    # missing 20 pp the battery would run out and the rest is bought by day.
    p50 = [0.6] * 7 + [0.55, 0.5, 0.55, 0.6, 0.65, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35] + [0.3] * 6
    assert _margin(p50) == {6: pytest.approx(0.2)}


def test_a_battery_that_covers_a_dull_day_is_released_to_the_need():
    # Owner's night 23.11: the median plan stays 35 pp above the floor until
    # the next cheap run; a dull day (20 pp less sun) still leaves 15 pp, so
    # the night zone may go 15 pp below the plan's own end-of-night level.
    p50 = [0.95] * 7 + [0.9, 0.88, 0.9, 0.92, 0.94, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7] + [0.65] * 6
    assert _margin(p50) == {6: pytest.approx(-0.15)}
    # Energy the plan buys for the battery that night may be meant for later: kept.
    assert _margin(p50, grid_charging=[False] * 3 + [True] + [False] * 20) == {}


def test_margin_never_takes_the_median_day_s_room_for_sun():
    p50 = [0.6] * 7 + [0.55, 0.5, 0.7, 0.85, 0.95, 0.95, 0.8, 0.6, 0.45, 0.4, 0.35] + [0.3] * 6
    assert _margin(p50) == {6: pytest.approx(0.05)}  # only 5 pp of headroom left


def test_margin_needs_hours_after_the_run():
    assert _margin([0.5] * 6, runs=[(2, 5)]) == {}


def _plan(pv_day, pv_p10_day, strategy="flat"):
    starts = pd.date_range("2026-11-21 18:00", periods=48, freq="h", tz="UTC")
    hours = np.asarray(starts.hour, dtype=float)
    load = np.full(48, 24.0 / 24)
    bell = np.where((hours >= 8) & (hours < 16), np.sin((hours - 7.5) / 8 * np.pi), 0.0)
    shape = bell / bell[:24].sum()
    opts = {
        "time_zone": "UTC",
        "min_soc_pct": 10,
        "max_soc_pct": 100,
        "capacity_kwh": 24.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "max_charge_kw": 5.0,
        "max_discharge_kw": 5.0,
        "tariff_enabled": True,
        "tariff_normal_price": 1.0,
        "tariff_cheap_price": 0.4,
        "tariff_cheap_windows": "00:00-07:00",
    }
    if strategy is not None:
        opts["day_zone_strategy"] = strategy
    return optimize.run_plan(
        starts,
        list(shape * pv_day),
        list(load),
        [0.1] * 48,
        0.3,
        PROGRAMS,
        opts,
        reserve_floors=[0.1] * 48,
        pv_scenario_kw=list(shape * pv_p10_day),
    )


def test_winter_night_charge_covers_the_pessimistic_sun():
    median = _plan(8.0, 8.0)  # P10 = P50: no extra
    pessimistic = _plan(8.0, 3.0)
    night = {t["n"]: t["optimal_soc_pct"] for t in median["zone_targets"]}[1]
    night_p10 = {t["n"]: t["optimal_soc_pct"] for t in pessimistic["zone_targets"]}[1]
    assert median["night_margin_kwh"] == 0.0
    # 5 kWh less sun -> ~5 kWh / 0.95 more stored overnight (~22 pp of 24 kWh).
    assert pessimistic["night_margin_kwh"] == pytest.approx(5.0 / 0.95 / 0.95, rel=0.15)
    assert night_p10 >= night + 15
    # The day zones stay at the floor either way.
    assert [t["optimal_soc_pct"] for t in pessimistic["zone_targets"]][1:] == [10] * 5
    # The reported trajectory and costs remain the median plan's.
    assert pessimistic["soc_opt"] == median["soc_opt"]
    assert pessimistic["grid_charge_kwh_cheap"] == pytest.approx(
        median["grid_charge_kwh_cheap"] + pessimistic["night_margin_kwh"], abs=0.01
    )


def test_summer_sun_leaves_no_room_for_a_margin():
    result = _plan(45.0, 20.0)
    assert result["night_margin_kwh"] == 0.0


def test_flat_is_the_default_strategy():
    result = _plan(8.0, 8.0, strategy=None)
    assert result["day_zone_strategy"] == "flat"
