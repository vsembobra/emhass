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


def test_margin_is_the_p10_charge_within_the_solar_headroom():
    runs = [(0, 6)]
    p50 = [0.5] * 7 + [0.45] * 5 + [0.7] * 4 + [0.3] * 8  # sun lifts it to 70 %
    p10 = [0.62] * 7 + [0.3] * 17
    assert optimize.night_charge_margin(p50, p10, runs, 1.0) == {6: pytest.approx(0.12)}
    # Median sun would fill the pack to 95 %: only 5 pp of room are left.
    sunny = p50[:12] + [0.95] * 4 + p50[16:]
    assert optimize.night_charge_margin(sunny, p10, runs, 1.0) == {6: pytest.approx(0.05)}
    # Summer: the median plan already reaches max SOC — no margin.
    full = p50[:12] + [1.0] * 4 + p50[16:]
    assert optimize.night_charge_margin(full, p10, runs, 1.0) == {}


def test_margin_excludes_energy_carried_to_the_next_night():
    # The pessimistic plan buys 30 pp more tonight but ends the day 20 pp
    # higher: it only draws 10 pp more over the day; the rest is tomorrow
    # night's purchase made early (both nights cost the same).
    p50 = [0.5] * 7 + [0.45] * 5 + [0.7] * 4 + [0.3] * 8
    p10 = [0.8] * 7 + [0.5] * 17
    assert optimize.night_charge_margin(p50, p10, [(0, 6)], 1.0) == {6: pytest.approx(0.1)}


def test_margin_needs_hours_after_the_run():
    assert optimize.night_charge_margin([0.5] * 6, [0.7] * 6, [(2, 5)], 1.0) == {}


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
        pv_p10_kw=list(shape * pv_p10_day),
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
