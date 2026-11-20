"""48 h load forecast: day 2 gets real features, temperature gaps get weather."""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402

from emhass.load_quantiles import LoadFeatureConfig, LoadQuantileBundle, build_features  # noqa: E402

ISSUE = pd.Timestamp("2026-11-15 06:00", tz="UTC")
STARTS_PLAN = pd.date_range(ISSUE, periods=48, freq="h")
P50 = 2.0


class RecordingBundle:
    """Stands in for LoadQuantileBundle; records the features of every call."""

    def __init__(self):
        self.config = LoadFeatureConfig(time_zone="UTC")
        self.calls = []

    def predict(self, grid, *, issue_time=None):
        features = build_features(grid, self.config, issue_time=issue_time)
        self.calls.append((pd.Timestamp(issue_time), features))
        p50 = np.full(len(grid), P50)
        return pd.DataFrame(
            {"load_p10_kw": p50 * 0.5, "load_p50_kw": p50, "load_p90_kw": p50 * 1.5},
            index=grid.index,
        )


def _audit(days=14, temp=None):
    index = pd.date_range(ISSUE - pd.Timedelta(days=days), ISSUE - pd.Timedelta(hours=1), freq="h")
    audit = pd.DataFrame(
        {"load_kw": 1.0 + 0.5 * np.sin(np.arange(len(index)) * 2 * np.pi / 24)}, index=index
    )
    if temp is not None:
        audit["temp_air_observed_c"] = temp
    return audit


def _weather(value=7.0):
    # archive_weather now fetches past_hours=48 + forecast_hours=48.
    index = pd.date_range(ISSUE - pd.Timedelta(hours=48), periods=96, freq="h")
    return pd.DataFrame({"temperature_2m": value}, index=index)


def test_day2_features_use_day1_p50_and_forecast_temperature():
    bundle = RecordingBundle()
    out = pipeline.forecast_load(bundle, _audit(temp=np.nan), STARTS_PLAN, ISSUE, _weather())

    assert list(out.index) == list(STARTS_PLAN)
    (issue1, _), (issue2, features2) = bundle.calls
    assert issue1 == ISSUE and issue2 == ISSUE + pd.Timedelta(hours=24)
    day2 = features2.loc[STARTS_PLAN[24:]]
    # Before the fix these three were NaN for every day-2 hour.
    assert (day2["load_lag_24"] == P50).all()
    assert np.allclose(day2["load_roll_mean_24"], P50)
    assert (day2["temp_lag_24"] == 7.0).all()
    # Lag 168 still reads real history, not the stand-in.
    assert day2["load_lag_168"].notna().all()
    assert not np.allclose(day2["load_lag_168"], P50)


def test_day1_temperature_lag_filled_from_past_weather_hours():
    bundle = RecordingBundle()
    pipeline.forecast_load(bundle, _audit(temp=np.nan), STARTS_PLAN, ISSUE, _weather())
    _, features1 = bundle.calls[0]
    # Before the fix the weather frame started at the issue hour and never
    # overlapped the past audit, so this lag stayed NaN on sensorless sites.
    assert (features1.loc[STARTS_PLAN[:24], "temp_lag_24"] == 7.0).all()


def test_observed_temperature_wins_over_weather():
    bundle = RecordingBundle()
    pipeline.forecast_load(bundle, _audit(temp=3.0), STARTS_PLAN, ISSUE, _weather(7.0))
    _, features1 = bundle.calls[0]
    assert (features1.loc[STARTS_PLAN[:24], "temp_lag_24"] == 3.0).all()


def test_audit_without_temperature_column_and_no_weather():
    bundle = RecordingBundle()
    out = pipeline.forecast_load(bundle, _audit(), STARTS_PLAN, ISSUE, None)
    assert len(out) == 48
    _, features2 = bundle.calls[1]
    assert (features2.loc[STARTS_PLAN[24:], "load_lag_24"] == P50).all()


def test_real_bundle_accepts_both_causal_windows():
    audit = _audit(days=21, temp=np.nan)
    grid = audit.copy()
    grid["temp_air_observed_c"] = 5.0
    bundle = LoadQuantileBundle.fit(grid, LoadFeatureConfig(time_zone="UTC"))
    out = pipeline.forecast_load(bundle, audit, STARTS_PLAN, ISSUE, _weather())
    assert len(out) == 48
    assert np.isfinite(out.to_numpy()).all()
    assert (out["load_p10_kw"] <= out["load_p50_kw"]).all()
    assert (out["load_p50_kw"] <= out["load_p90_kw"]).all()


def test_weather_fetch_includes_past_hours(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    times = [int((now + timedelta(hours=h)).timestamp()) for h in range(-48, 48)]
    body = json.dumps({"hourly": {"time": times, "temperature_2m": [1.0] * len(times)}})
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return body.encode()

    def fake_urlopen(url, timeout=None, context=None):
        seen["url"] = url
        return Response()

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)
    frame, _, stale = pipeline.archive_weather({"latitude": 1.0, "longitude": 2.0})
    assert "past_hours=48" in seen["url"]
    assert not stale
    assert frame.index[0] < pd.Timestamp(now)
