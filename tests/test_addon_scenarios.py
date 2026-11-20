"""Owner scenarios as acceptance tests: the plan with the reserve-only floor.

The floor is what pipeline.run() now feeds the optimizer: the outage reserve
alone (zero outside an outage series). Each test states the behaviour the
owner expects for a class of installation.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import optimize  # noqa: E402

from emhass.reserve import ReserveSettings  # noqa: E402

ZONES = [
    {"n": n, "from": f, "soc_pct": 30}
    for n, f in enumerate(["00:00", "06:00", "10:00", "14:00", "18:00", "22:00"], 1)
]


def _bell(hours, total, start, end):
    x = np.asarray(hours, dtype=float)
    shape = np.where(
        (x >= start) & (x < end), np.sin((x - start + 0.5) / (end - start) * np.pi), 0.0
    )
    return shape / shape[:24].sum() * total


def _load(hours, total):
    x = np.asarray(hours, dtype=float)
    shape = 0.6 + 0.6 * np.exp(-((x - 20) ** 2) / 6) + 0.3 * np.exp(-((x - 8) ** 2) / 4)
    return shape / shape[:24].sum() * total


def plan(cap, load_day, pv_day, pv_hours, tariff, soc0, reserve_kwh=0.0, max_kw=5.0):
    starts = pd.date_range("2026-07-01 18:00", periods=48, freq="h", tz="UTC")
    load = _load(starts.hour, load_day)
    pv = _bell(starts.hour, pv_day, *pv_hours)
    opts = {
        "time_zone": "UTC",
        "min_soc_pct": 10,
        "max_soc_pct": 100,
        "capacity_kwh": cap,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "max_charge_kw": max_kw,
        "max_discharge_kw": max_kw,
        "tariff_enabled": tariff,
        "tariff_normal_price": 2.0,
        "tariff_cheap_price": 1.0,
        "tariff_cheap_windows": "23:00-07:00",
    }
    settings = ReserveSettings(cap, 1.0, 0.1, 1.0, 0.95, cold_start_kw=0.5, cold_start_hours=0.0)
    floor = math.ceil(settings.floor_for_energy(reserve_kwh)[0] / 0.05 - 1e-9) * 0.05
    result = optimize.run_plan(
        starts,
        list(pv),
        list(load),
        [floor] * 48,
        soc0,
        ZONES,
        opts,
        reserve_floors=[settings.floor_for_energy(reserve_kwh)[0]] * 48,
    )
    hour = {t.hour: i for i, t in list(enumerate(starts))[:24]}
    return result, starts, load, pv, hour, floor


def test_winter_with_cheap_night_charges_at_night_and_spends_by_day():
    result, starts, load, pv, hour, _ = plan(15, 25, 8, (9, 15), True, 0.3)
    assert not result["floor_relaxed"]
    soc = result["soc_opt"]
    assert soc[hour[6]] > soc[hour[22]] + 0.3  # charged overnight
    # Spent over the expensive 07:00-23:00 (same price all of it, so the
    # optimizer may keep some for the evening peak): low again by 22:00.
    assert soc[24 + hour[22]] < soc[hour[6]] - 0.5
    grid = result["p_grid_kw"]
    night = sum(grid[i] for i in range(24) if starts[i].hour >= 23 or starts[i].hour < 7)
    day = sum(grid[i] for i in range(24)) - night
    assert night > day


def test_winter_with_one_price_never_buys_to_store():
    result, starts, load, pv, hour, _ = plan(15, 25, 8, (9, 15), False, 0.8)
    soc = result["soc_opt"]
    # No PV at night: SOC can only fall or stay; any rise would be grid charging.
    night = [i for i in range(24) if starts[i].hour >= 20 or starts[i].hour < 8]
    assert all(soc[i] <= soc[i - 1] + 1e-6 for i in night if i > 0)
    assert result["grid_charge_kwh_cheap"] == 0.0


def test_small_battery_empties_before_the_sun_and_refills_from_it():
    result, starts, load, pv, hour, floor = plan(4.5, 26, 40, (6, 19), False, 0.6, max_kw=3)
    soc = result["soc_opt"]
    sunrise = next(i for i in range(1, 24) if pv[i] > load[i])
    assert soc[sunrise - 1] <= floor + 0.05  # discharged by the time PV covers the load
    assert max(soc[sunrise:24]) == pytest.approx(1.0, abs=0.01)  # refilled by the sun
    grid = result["p_grid_kw"]
    assert all(grid[i] <= load[i] + 1e-3 for i in range(sunrise))  # never buys to store


def test_small_battery_keeps_the_series_reserve():
    result, starts, load, pv, hour, floor = plan(
        4.5, 26, 40, (6, 19), False, 0.6, reserve_kwh=1.2, max_kw=3
    )
    soc = result["soc_opt"]
    assert floor > 0.1
    assert min(soc) >= floor - 1e-6


def test_summer_big_battery_runs_on_its_own_sun():
    result, starts, load, pv, hour, _ = plan(48, 22, 50, (5, 20), False, 0.6)
    assert sum(result["p_grid_kw"]) == pytest.approx(0.0, abs=0.05)  # 48 h off-grid
    # The battery fills from the sun. Which surplus hour the plan curtails is
    # degenerate here (the pack is full on day 2 either way); a Deye zone
    # SOC does not cap solar charging, so the inverter stores it regardless.
    assert max(result["soc_opt"]) == pytest.approx(1.0, abs=0.01)
