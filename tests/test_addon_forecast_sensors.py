"""Sensors that let the owner compare forecasts and weather with reality in HA."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import learning  # noqa: E402
import pipeline  # noqa: E402

HOUR = pd.Timestamp("2026-11-21 10:00", tz="UTC")


def _entry(issue, pv_raw, pv_used=None):
    starts = pd.date_range(issue, periods=48, freq="h")
    entry = {
        "issue_utc": issue.isoformat(),
        "starts_utc": [s.isoformat() for s in starts],
        "pv": {"p50": [pv_raw] * 48},
        "load": {"p50": [0.5] * 48},
    }
    if pv_used is not None:
        entry["pv_p50_used"] = [pv_used] * 48
        entry["load_p50_used"] = [0.6] * 48
    return entry


def test_day_ahead_takes_the_latest_run_at_least_a_day_before():
    entries = [
        _entry(HOUR - pd.Timedelta(hours=30), 1.0, 1.1),
        _entry(HOUR - pd.Timedelta(hours=24), 2.0, 2.2),  # the one
        _entry(HOUR - pd.Timedelta(hours=3), 3.0, 3.3),  # too fresh
    ]
    ahead = learning.day_ahead(entries, HOUR)
    assert ahead["issue_utc"] == (HOUR - pd.Timedelta(hours=24)).isoformat()
    assert ahead["pv_p50_kw"] == 2.2 and ahead["load_p50_kw"] == 0.6  # what the plan used


def test_day_ahead_falls_back_to_the_raw_forecast_and_to_none():
    assert (
        learning.day_ahead([_entry(HOUR - pd.Timedelta(hours=26), 1.5)], HOUR)["pv_p50_kw"] == 1.5
    )
    assert learning.day_ahead([_entry(HOUR - pd.Timedelta(hours=2), 1.5)], HOUR) is None
    assert learning.day_ahead([], HOUR) is None


def test_weather_hour_reads_the_api_like_the_pv_model():
    stamps = pd.date_range(HOUR, periods=2, freq="h")
    weather = pd.DataFrame(
        {
            "shortwave_radiation": [100.0, 300.0],  # 300 = mean of 10:00-11:00
            "cloud_cover": [40.0, 60.0],
            "temperature_2m": [4.0, 6.0],
        },
        index=stamps,
    )
    assert pipeline.weather_hour(weather, HOUR.to_pydatetime()) == {
        "irradiance_wm2": 300.0,
        "cloud_cover_pct": 50.0,
        "temperature_c": 5.0,
    }
    assert pipeline.weather_hour(None, HOUR) is None


def test_forecast_sensors_in_watts_with_statistics():
    report = {
        "hourly": {
            "starts_utc": [HOUR.isoformat()],
            "pv_p50_kw": [1.234],
            "pv_p10_kw": [0.5],
            "load_p50_kw": [0.8],
        },
        "day_ahead_hour": None,
        "weather_hour": {"irradiance_wm2": 300.0, "cloud_cover_pct": None, "temperature_c": 5.0},
    }
    sensors = {
        entity: (state, attrs) for entity, state, attrs in pipeline._forecast_sensors(report)
    }
    state, attrs = sensors["sensor.emhass_shadow_pv_forecast"]
    assert state == 1234 and attrs["unit_of_measurement"] == "W"
    assert attrs["state_class"] == "measurement" and attrs["hour_start_utc"] == HOUR.isoformat()
    assert sensors["sensor.emhass_shadow_pv_forecast_day_ahead"][0] == "unknown"
    assert sensors["sensor.emhass_shadow_weather_cloud_cover"][0] == "unknown"
    assert sensors["sensor.emhass_shadow_weather_irradiance"][0] == pytest.approx(300.0)
    assert all(a["state_class"] == "measurement" for _, a in sensors.values())


def test_hour_sensors_move_on_between_replans(monkeypatch):
    # A plan issued at 10:00; at 11:00 the sensors take the 11:00 values.
    starts = [(HOUR + pd.Timedelta(hours=i)).isoformat() for i in range(3)]
    report = {
        "hourly": {
            "starts_utc": starts,
            "pv_p50_kw": [1.0, 2.0, 3.0],
            "pv_p10_kw": [0.5, 1.0, 1.5],
            "load_p50_kw": [0.4, 0.5, 0.6],
        }
    }
    posted = {}
    monkeypatch.setattr(pipeline, "ha_post_state", lambda e, s, a: posted.update({e: (s, a)}))
    monkeypatch.setattr(pipeline, "_day_ahead_hour", lambda hour: None)
    monkeypatch.setattr(pipeline, "archived_weather", lambda: None)
    published = pipeline.publish_hour_sensors(report, HOUR + pd.Timedelta(hours=1))
    assert len(published) == 8
    assert posted["sensor.emhass_shadow_pv_forecast"][0] == 2000
    assert posted["sensor.emhass_shadow_load_forecast"][0] == 500
    assert posted["sensor.emhass_shadow_pv_forecast"][1]["hour_start_utc"] == starts[1]
    # An hour the plan does not cover publishes nothing.
    assert pipeline.publish_hour_sensors(report, HOUR + pd.Timedelta(hours=5)) == []
