"""Load model: forecast temperature, lead-aware training, site time zone."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402
import retrain  # noqa: E402

from emhass.load_quantiles import (  # noqa: E402
    LoadFeatureConfig,
    LoadQuantileBundle,
    build_features,
    feature_names,
    stacked_features,
)

UTC_CONFIG = LoadFeatureConfig(time_zone="UTC")


def _grid(days=40, seed=3):
    """Heating load driven by the day's (forecast) temperature."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=days * 24, freq="h", tz="UTC")
    daily_temp = rng.normal(5.0, 6.0, days)
    temp = np.repeat(daily_temp, 24)
    hours = index.hour.to_numpy()
    load = 0.6 + 0.3 * np.sin(hours * 2 * np.pi / 24) + 0.12 * np.clip(15.0 - temp, 0, None)
    return pd.DataFrame(
        {"load_kw": load, "temp_air_observed_c": temp, "temp_forecast_c": temp}, index=index
    )


def test_lead_shifts_rolling_features():
    grid = _grid(days=10)
    one = build_features(grid, UTC_CONFIG, lead=1)
    six = build_features(grid, UTC_CONFIG, lead=6)
    assert np.allclose(
        six["load_roll_mean_24"].iloc[30:], one["load_roll_mean_24"].shift(5).iloc[30:]
    )


def test_stacked_features_cover_every_lead():
    config = LoadFeatureConfig(time_zone="UTC", lead_aware=True, training_leads=(1, 12, 24))
    grid = _grid(days=5)
    stacked, origin = stacked_features(grid, config, seasonal=False)
    assert len(stacked) == 3 * len(grid)
    assert sorted(stacked["lead_hours"].unique()) == [1.0, 12.0, 24.0]
    assert origin.tz is not None and (origin[: len(grid)] == grid.index).all()


def test_inference_lead_counts_from_issue():
    config = LoadFeatureConfig(time_zone="UTC", lead_aware=True)
    grid = _grid(days=10)
    issue = grid.index[-24]
    features = build_features(grid, config, issue_time=issue)
    assert list(features.loc[issue:, "lead_hours"]) == [float(h) for h in range(1, 25)]


def test_training_leads_must_stay_causal():
    with pytest.raises(ValueError, match="training_leads"):
        LoadFeatureConfig(lead_aware=True, training_leads=(1, 25))


def test_feature_names_follow_flags():
    names = feature_names(
        LoadFeatureConfig(forecast_temperature=True, lead_aware=True), seasonal=False
    )
    assert names[-2:] == ["temp_forecast_c", "lead_hours"]
    assert "lead_hours" not in feature_names(LoadFeatureConfig(), seasonal=False)


def test_forecast_temperature_beats_lagged_temperature():
    grid = _grid(days=60)
    train, test = grid.iloc[: -24 * 10], grid
    old = LoadQuantileBundle.fit(train, UTC_CONFIG)
    new = LoadQuantileBundle.fit(
        train, LoadFeatureConfig(time_zone="UTC", forecast_temperature=True, lead_aware=True)
    )
    errors = {}
    for name, bundle in (("old", old), ("new", new)):
        absolute = []
        for day in range(10, 0, -1):
            issue = grid.index[-24 * day]
            window = test.loc[: issue + pd.Timedelta(hours=23)]
            p50 = bundle.predict(window, issue_time=issue)["load_p50_kw"].loc[issue:]
            absolute.append(np.abs(p50 - test["load_kw"].reindex(p50.index)))
        errors[name] = float(pd.concat(absolute).mean())
    assert errors["new"] < 0.8 * errors["old"], errors


def test_bundle_roundtrip_keeps_flags_and_old_bundles_load(tmp_path):
    config = LoadFeatureConfig(time_zone="UTC", forecast_temperature=True, lead_aware=True)
    bundle = LoadQuantileBundle.fit(_grid(days=10), config)
    bundle.save(tmp_path / "new")
    loaded = LoadQuantileBundle.load(tmp_path / "new")
    assert loaded.config.forecast_temperature and loaded.config.lead_aware
    # A bundle saved before the options existed keeps the old recipe.
    LoadQuantileBundle.fit(_grid(days=10), UTC_CONFIG).save(tmp_path / "old")
    meta = json.loads((tmp_path / "old" / "bundle.json").read_text())
    for key in ("forecast_temperature", "lead_aware", "training_leads"):
        meta["config"].pop(key)
    (tmp_path / "old" / "bundle.json").write_text(json.dumps(meta))
    old = LoadQuantileBundle.load(tmp_path / "old")
    assert not old.config.forecast_temperature and not old.config.lead_aware


def test_challenger_config_uses_site_time_zone_and_holidays():
    config = retrain.challenger_config(
        None, {"time_zone": "Europe/Berlin", "load_holidays": "12-25, 1-1, bad, 13-40"}
    )
    assert config.time_zone == "Europe/Berlin"
    assert config.holidays == frozenset({"12-25", "01-01"})
    assert config.forecast_temperature and config.lead_aware


def test_forecast_load_feeds_forecast_temperature():
    class Recording:
        config = LoadFeatureConfig(time_zone="UTC", forecast_temperature=True)

        def __init__(self):
            self.features = []

        def predict(self, grid, *, issue_time=None):
            self.features.append(build_features(grid, self.config, issue_time=issue_time))
            p50 = np.ones(len(grid))
            return pd.DataFrame(
                {"load_p10_kw": p50, "load_p50_kw": p50, "load_p90_kw": p50}, index=grid.index
            )

    issue = pd.Timestamp("2026-11-15 06:00", tz="UTC")
    history = pd.date_range(issue - pd.Timedelta(days=10), issue - pd.Timedelta(hours=1), freq="h")
    audit = pd.DataFrame({"load_kw": 1.0}, index=history)
    starts = pd.date_range(issue, periods=48, freq="h")
    weather = pd.DataFrame(
        {"temperature_2m": -3.0},
        index=pd.date_range(issue - pd.Timedelta(hours=48), periods=96, freq="h"),
    )
    bundle = Recording()
    pipeline.forecast_load(bundle, audit, starts, issue, weather)
    assert (bundle.features[0].loc[starts[:24], "temp_forecast_c"] == -3.0).all()
    assert (bundle.features[1].loc[starts[24:], "temp_forecast_c"] == -3.0).all()


def test_retrain_promotes_lead_aware_recipe(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain, "MODELS", tmp_path)
    grid = _grid(days=30)
    audit = grid.assign(load_filter="keep")
    weather = pd.DataFrame({"temperature_2m_previous_day1": grid["temp_forecast_c"]})
    info = retrain._retrain_load(audit, {"time_zone": "UTC"}, weather)
    assert info["status"] == "promoted"
    bundle = LoadQuantileBundle.load(tmp_path / "load_quantiles_v1")
    assert bundle.config.lead_aware and bundle.config.forecast_temperature
    assert bundle.config.time_zone == "UTC"
