import json
from datetime import UTC, datetime, timedelta

import pytest

from emhass.battery_backtest import Battery, ReplayStep
from emhass.battery_planning import ForecastStep
from emhass.battery_replay import (
    AnnouncedOutage,
    ForecastRun,
    ObjectivePlan,
    ObjectiveStep,
    main,
    replay_site,
)
from emhass.outage_log import OutageEvent
from emhass.reserve import ReserveSettings

START = datetime(2026, 10, 19, 6, tzinfo=UTC)
NEXT = START + timedelta(hours=1)
BATTERY = Battery(10.0, 0.1, 0.9, 5, 5, 1, 1)
RESERVE = ReserveSettings(10, 1, 0.1, 0.9, 1, 1)
ACTUAL = [ReplayStep(START, 1, 0, 3, 3, True), ReplayStep(NEXT, 1, 0, 3, 3, False)]


def run(**overrides):
    return replay_site(ACTUAL, BATTERY, RESERVE, initial_soc=0.5, fixed_floor_soc=0.2, **overrides)


def test_replay_uses_only_issued_forecasts_and_has_a_matched_hindsight_comparison():
    forecasts = [
        ForecastRun(START, (ForecastStep(START, 1, 3, 3), ForecastStep(NEXT, 1, 0, 3))),
        ForecastRun(START + timedelta(seconds=1), (ForecastStep(START, 1, 0, 3),)),
    ]
    report = run(forecasts=forecasts)
    assert report.fixed.unserved_critical_kwh == pytest.approx(2)
    assert report.baseline.unserved_critical_kwh == pytest.approx(1)
    assert report.hindsight.unserved_critical_kwh == 0
    assert report.hindsight_steps == 2
    assert report.baseline_vs_fixed == "better"
    assert report.forecast_fallback_steps == 0


def test_replay_accepts_only_as_of_objective_setpoints_in_emhass_watts():
    original = ObjectiveStep(START, START, -1000)
    late = ObjectiveStep(START + timedelta(minutes=5), START, 5000)
    report = run(objective_plan=ObjectivePlan("profit", (original, late)))
    assert report.objective == "profit"
    assert report.objective_steps == 1
    assert report.objective_replay.imported_kwh == pytest.approx(4)
    assert report.objective_replay.unserved_critical_kwh == 0
    assert report.objective_vs_fixed is None


def test_scheduled_precharge_depends_on_announcement_time_and_is_released_in_outage():
    scheduled = OutageEvent(NEXT, NEXT + timedelta(hours=5), "scheduled")
    known = run(scheduled=(AnnouncedOutage(START, scheduled),))
    learned_later = run(scheduled=(AnnouncedOutage(NEXT, scheduled),))
    assert known.baseline.imported_kwh == pytest.approx(4)
    assert known.baseline.unserved_critical_kwh == 0
    assert known.missed_precharge_deadlines == 0
    assert learned_later.baseline.unserved_critical_kwh == pytest.approx(1)
    assert learned_later.missed_precharge_deadlines == 1


def test_replay_cli_reports_missing_evidence_instead_of_passing_the_gate(tmp_path, capsys):
    dataset = {
        "reserve": {
            "nominal_capacity_kwh": 10,
            "soh": 1,
            "floor_soc": 0.1,
            "max_soc": 0.9,
            "discharge_efficiency": 1,
            "cold_start_kw": 1,
        },
        "battery": {"max_charge_kw": 5, "max_discharge_kw": 5, "charge_efficiency": 1},
        "initial_soc": 0.5,
        "fixed_floor_soc": 0.2,
        "actuals": [
            {
                "start": START.isoformat(),
                "hours": 1,
                "pv_kw": 0,
                "load_kw": 3,
                "critical_load_kw": 3,
                "grid_up": True,
            }
        ],
    }
    path = tmp_path / "site.json"
    path.write_text(json.dumps(dataset))
    main([str(path)])
    result = json.loads(capsys.readouterr().out)
    assert result["baseline_vs_fixed"] is None
    assert result["hindsight"] is None
    assert result["forecast_fallback_steps"] == 1
    assert result["objective_replay"] is None
