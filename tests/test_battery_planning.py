from datetime import UTC, datetime, timedelta

import pytest

from emhass.battery_planning import ForecastStep, baseline_floor
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
