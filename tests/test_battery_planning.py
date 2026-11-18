from datetime import UTC, datetime, timedelta

import pytest

from emhass.battery_planning import ForecastStep, baseline_floor, floor_profile
from emhass.reserve import ReserveSettings

START = datetime(2026, 10, 19, 6, tzinfo=UTC)
SETTINGS = ReserveSettings(12.5, 1.0, 0.1, 0.9, 0.8, 1.0, margin=0.02)


def test_baseline_integrates_positive_deficit_before_the_first_solar_crossover():
    forecast = [
        ForecastStep(START, 0.5, 0, 2),
        ForecastStep(START + timedelta(minutes=30), 1.5, 1, 2),
        ForecastStep(START + timedelta(hours=2), 1, 2, 2),
        ForecastStep(START + timedelta(hours=3), 1, 0, 10),
    ]
    plan = baseline_floor(forecast, reserve_kwh=2.5, settings=SETTINGS)
    assert plan.crossover == START + timedelta(hours=2)
    assert plan.energy_to_crossover_kwh == pytest.approx(3.125)
    assert plan.floor_soc == pytest.approx(0.57)


def test_no_crossover_accounts_for_available_horizon_and_flags_the_shortfall():
    forecast = [ForecastStep(START, 1, 0, 0), ForecastStep(START + timedelta(hours=1), 2, 0, 6)]
    plan = baseline_floor(forecast, reserve_kwh=2.5, settings=SETTINGS)
    assert plan.crossover is None
    assert plan.floor_soc == 0.9
    assert plan.shortfall_kwh == pytest.approx(7.75)


def test_baseline_rejects_gaps_and_signed_load():
    with pytest.raises(ValueError, match="contiguous"):
        baseline_floor(
            [ForecastStep(START, 1, 1, 1), ForecastStep(START + timedelta(hours=2), 1, 1, 1)],
            reserve_kwh=0,
            settings=SETTINGS,
        )
    with pytest.raises(ValueError, match="load"):
        ForecastStep(START, 1, 0, -1)




def _hours_forecast(pv_by_hour, load_by_hour, start=START):
    return [
        ForecastStep(start + timedelta(hours=i), 1.0, pv, load)
        for i, (pv, load) in enumerate(zip(pv_by_hour, load_by_hour, strict=True))
    ]


def test_floor_profile_relaxes_towards_crossover_and_rises_in_the_evening():
    # 30 h horizon: sun covers the load 09:00-16:00 today and again from
    # 09:00 tomorrow, so evening hours see the overnight deficit ahead.
    pv = [0] * 3 + [5] * 8 + [0] * 16 + [5] * 3
    load = [1.0] * 30
    forecast = _hours_forecast(pv, load)
    profile = floor_profile(forecast, reserve_kwh=0.0, settings=SETTINGS)
    floors = [step.floor_soc for step in profile.steps]
    # 06:00 has the largest pre-sun deficit, 08:00 the smallest.
    assert floors[0] > floors[1] > floors[2]
    # Sun hours need no deficit energy: floor is the physical floor plus margin.
    assert all(step.floor_soc == SETTINGS.floor_soc + 0.02 for step in profile.steps[3:11])
    # Evening carries the whole night ahead; the floor then relaxes towards
    # tomorrow morning's crossover.
    assert floors[11] > floors[20] > floors[26]
    assert profile.max_floor_soc == max(floors)
    assert profile.steps[0].crossover == START + timedelta(hours=3)
    assert profile.steps[20].crossover == START + timedelta(hours=27)


def test_floor_profile_per_interval_reserve():
    forecast = _hours_forecast([0.0, 5.0, 5.0], [1.0, 1.0, 1.0])
    flat = floor_profile(forecast, reserve_kwh=1.0, settings=SETTINGS)
    boosted = floor_profile(forecast, reserve_kwh=[3.0, 1.0, 1.0], settings=SETTINGS)
    assert boosted.steps[0].floor_soc > flat.steps[0].floor_soc
    assert boosted.steps[1].floor_soc == flat.steps[1].floor_soc
    with pytest.raises(ValueError, match="match the forecast length"):
        floor_profile(forecast, reserve_kwh=[1.0], settings=SETTINGS)


def test_floor_profile_without_crossover_uses_the_full_horizon():
    forecast = _hours_forecast([0.0, 0.0], [2.0, 2.0])
    profile = floor_profile(forecast, reserve_kwh=0.0, settings=SETTINGS)
    assert profile.steps[0].crossover is None
    assert profile.steps[0].stored_kwh == pytest.approx(4.0 / SETTINGS.discharge_efficiency)
    assert profile.steps[1].stored_kwh == pytest.approx(2.0 / SETTINGS.discharge_efficiency)
