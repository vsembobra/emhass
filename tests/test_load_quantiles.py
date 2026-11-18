"""Synthetic-data tests for the opt-in LightGBM load quantile forecaster."""

import importlib
import importlib.util
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from emhass.load_quantiles import (
    LoadFeatureConfig,
    LoadQuantileBundle,
    build_features,
    coverage,
    feature_names,
    pinball_loss,
    try_load,
)

LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None
needs_lightgbm = pytest.mark.skipif(
    not LIGHTGBM_AVAILABLE, reason="lightgbm is an optional 'quantile' extra"
)

FAST_PARAMS = {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 5}


def synthetic_grid(days: int, start: datetime = datetime(2026, 1, 1, tzinfo=UTC)) -> pd.DataFrame:
    index = pd.date_range(start, periods=days * 24, freq="h")
    hours = index.hour
    load = 0.4 + 0.2 * np.sin(2 * np.pi * (hours - 7) / 24) + 0.05 * np.cos(
        2 * np.pi * np.arange(len(index)) / (24 * 7)
    )
    return pd.DataFrame(
        {
            "load_kw": load,
            "temp_air_observed_c": 15.0 + 8.0 * np.sin(2 * np.pi * (hours - 14) / 24),
        },
        index=index,
    )


def test_import_does_not_require_lightgbm():
    saved = {name: module for name, module in sys.modules.items() if name.startswith("lightgbm")}
    for name in saved:
        del sys.modules[name]
    sys.modules.pop("emhass.load_quantiles", None)
    try:
        importlib.import_module("emhass.load_quantiles")
        assert "lightgbm" not in sys.modules
    finally:
        sys.modules.update(saved)
        sys.modules.pop("emhass.load_quantiles", None)
        importlib.import_module("emhass.load_quantiles")


def test_grid_must_be_complete_hourly():
    grid = synthetic_grid(30).drop(synthetic_grid(30).index[100])
    with pytest.raises(ValueError, match="complete hourly"):
        build_features(grid)


def test_features_are_leakage_free_at_issue_time():
    grid = synthetic_grid(30)
    issue = grid.index[400]
    # A forecast issued at `issue` covers the next 24 hourly steps; for those
    # steps every lag still points before the issue instant.
    horizon = (grid.index >= issue) & (grid.index < issue + pd.Timedelta(hours=24))
    shocked_future = grid.copy()
    shocked_future.loc[grid.index >= issue, "load_kw"] = 999.0
    shocked_future.loc[grid.index >= issue, "temp_air_observed_c"] = 999.0
    pd.testing.assert_frame_equal(
        build_features(grid, issue_time=issue).loc[horizon],
        build_features(shocked_future, issue_time=issue).loc[horizon],
    )
    shocked_past = grid.copy()
    shocked_past.loc[grid.index < issue, "load_kw"] = 999.0
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            build_features(grid, issue_time=issue).loc[horizon],
            build_features(shocked_past, issue_time=issue).loc[horizon],
        )


@needs_lightgbm
def test_feature_parity_between_fit_and_predict(tmp_path):
    grid = synthetic_grid(40)
    bundle = LoadQuantileBundle.fit(grid, params=FAST_PARAMS)
    features = build_features(grid, issue_time=grid.index[500])
    assert set(bundle.columns) <= set(features.columns)


@needs_lightgbm
def test_seasonal_features_wait_for_twelve_months():
    short = LoadQuantileBundle.fit(synthetic_grid(300), params=FAST_PARAMS)
    long = LoadQuantileBundle.fit(synthetic_grid(400), params=FAST_PARAMS)
    assert not short.seasonal
    assert "day_of_year_sin" not in short.columns
    assert long.seasonal
    assert "day_of_year_sin" in long.columns
    explicit = LoadQuantileBundle.fit(synthetic_grid(30), seasonal=True, params=FAST_PARAMS)
    assert "day_of_year_sin" in explicit.columns


@needs_lightgbm
def test_predictions_are_finite_ordered_and_nonnegative():
    grid = synthetic_grid(40)
    bundle = LoadQuantileBundle.fit(grid, params=FAST_PARAMS)
    prediction = bundle.predict(grid)
    assert np.isfinite(prediction.to_numpy()).all()
    assert (prediction >= 0).all().all()
    assert (prediction["load_p10_kw"] <= prediction["load_p50_kw"]).all()
    assert (prediction["load_p50_kw"] <= prediction["load_p90_kw"]).all()


@needs_lightgbm
def test_calibration_improves_p90_coverage_on_shifted_slice():
    grid = synthetic_grid(60)
    train, calibration = grid.iloc[: 40 * 24], grid.iloc[40 * 24 :]
    bundle = LoadQuantileBundle.fit(train, params=FAST_PARAMS)
    shifted = calibration.copy()
    shifted["load_kw"] = shifted["load_kw"] * 1.5
    before = (shifted["load_kw"] <= bundle.predict(shifted)["load_p90_kw"]).mean()
    bundle.calibrate(shifted)
    after = (shifted["load_kw"] <= bundle.predict(shifted)["load_p90_kw"]).mean()
    assert after >= before
    assert after >= 0.85
    assert bundle.calibration_offsets["p90"] > 0


@needs_lightgbm
def test_cold_start_and_missing_history_still_predict():
    grid = synthetic_grid(40)
    bundle = LoadQuantileBundle.fit(grid, params=FAST_PARAMS)
    empty = grid.copy()
    empty["load_kw"] = np.nan
    empty["temp_air_observed_c"] = np.nan
    prediction = bundle.predict(empty)
    assert np.isfinite(prediction.to_numpy()).all()
    assert try_load("/nonexistent/bundle") is None


@needs_lightgbm
def test_fit_requires_enough_valid_targets():
    grid = synthetic_grid(2)
    grid["load_kw"] = np.nan
    grid.iloc[:10, grid.columns.get_loc("load_kw")] = 0.5
    with pytest.raises(ValueError, match="at least 24"):
        LoadQuantileBundle.fit(grid, params=FAST_PARAMS)


@needs_lightgbm
def test_save_load_roundtrip_preserves_predictions(tmp_path):
    grid = synthetic_grid(40)
    bundle = LoadQuantileBundle.fit(grid, params=FAST_PARAMS)
    bundle.calibrate(grid.iloc[30 * 24 :])
    bundle.save(tmp_path)
    restored = LoadQuantileBundle.load(tmp_path)
    pd.testing.assert_frame_equal(bundle.predict(grid), restored.predict(grid))
    metadata = json.loads((tmp_path / "bundle.json").read_text())
    assert metadata["model_version"] == "load-quantiles-1"
    assert metadata["feature_names"] == bundle.columns
    assert metadata["trained_from"] and metadata["calibrated_until"]
    assert try_load(tmp_path) is not None


@needs_lightgbm
def test_load_rejects_unknown_model_version(tmp_path):
    grid = synthetic_grid(40)
    LoadQuantileBundle.fit(grid, params=FAST_PARAMS).save(tmp_path)
    metadata = json.loads((tmp_path / "bundle.json").read_text())
    metadata["model_version"] = "future-version"
    (tmp_path / "bundle.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="model version"):
        LoadQuantileBundle.load(tmp_path)
    assert try_load(tmp_path) is None


def test_metric_helpers():
    actual = np.array([1.0, 2.0, 3.0])
    prediction = np.array([1.0, 1.0, 5.0])
    # errors: 0, 1, -2 -> pinball(0.9): (0, 0.9, 0.2) / 3
    assert pinball_loss(actual, prediction, 0.9) == pytest.approx((0.9 + 0.2) / 3)
    assert coverage(actual, np.zeros(3), prediction) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        pinball_loss(actual, prediction[:2], 0.9)


def test_config_validation_and_holidays():
    with pytest.raises(ValueError, match="lag_hours"):
        LoadFeatureConfig(lag_hours=(0,))
    with pytest.raises(ValueError, match="holiday"):
        LoadFeatureConfig(holidays=frozenset({"13-40"}))
    config = LoadFeatureConfig(holidays=frozenset({"01-01"}))
    grid = synthetic_grid(30)
    features = build_features(grid, config)
    jan_first = grid.index.tz_convert("Europe/Istanbul").date.astype(str) == "2026-01-01"
    assert features["is_holiday"][jan_first].eq(1.0).all()
    assert features["is_holiday"][~jan_first].eq(0.0).all()


def test_feature_names_match_config_temperature_lag():
    config = LoadFeatureConfig(temperature_lag_hours=48)
    assert "temp_lag_48" in feature_names(config, seasonal=False)
    assert "temp_lag_24" not in feature_names(config, seasonal=False)
