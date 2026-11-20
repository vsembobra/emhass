"""Synthetic-data tests for the opt-in PV quantile model."""

import importlib
import importlib.util
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from emhass.pv_quantiles import (
    PvGeometry,
    PvQuantileBundle,
    clear_sky_power_kw,
    estimate_geometry,
    plane_power_kw,
    try_load,
)

LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None
needs_lightgbm = pytest.mark.skipif(
    not LIGHTGBM_AVAILABLE, reason="lightgbm is an optional 'quantile' extra"
)

FAST_PARAMS = {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 5}
GEOMETRY = PvGeometry(latitude=45.0, longitude=34.0, tilt_deg=25, azimuth_deg=180, kwp=7.0)


def synthetic_frame(days: int, noise: float = 0.0) -> pd.DataFrame:
    """Measured PV as a fixed share of clear-sky power plus weather columns."""
    index = pd.date_range(datetime(2026, 6, 1, tzinfo=UTC), periods=days * 24, freq="h")
    clearsky = clear_sky_power_kw(GEOMETRY, index)
    rng = np.random.default_rng(7)
    measured = (0.75 + noise * rng.standard_normal(len(index))) * clearsky
    return pd.DataFrame(
        {
            "pv_measured_kw": measured.clip(lower=0),
            "poa_forecast_kw": clearsky.to_numpy() * 0.75,
            "cloud_cover_pct": 30.0,
            "temp_air_c": 20.0,
        },
        index=index,
    )


def test_import_does_not_require_lightgbm():
    saved = {name: module for name, module in sys.modules.items() if name.startswith("lightgbm")}
    for name in saved:
        del sys.modules[name]
    sys.modules.pop("emhass.pv_quantiles", None)
    try:
        importlib.import_module("emhass.pv_quantiles")
        assert "lightgbm" not in sys.modules
    finally:
        sys.modules.update(saved)
        sys.modules.pop("emhass.pv_quantiles", None)
        importlib.import_module("emhass.pv_quantiles")


def test_clear_sky_is_zero_at_night_and_peaks_at_noon():
    times = pd.date_range(datetime(2026, 6, 15, tzinfo=UTC), periods=24, freq="h")
    clearsky = clear_sky_power_kw(GEOMETRY, times)
    assert clearsky.iloc[0] == 0 and clearsky.iloc[23] == 0  # 00:00 and 23:00 UTC
    assert clearsky.idxmax().hour in (9, 10, 11, 12)  # local noon ~ 10 UTC at 34 E
    winter = clear_sky_power_kw(
        GEOMETRY, pd.date_range(datetime(2026, 12, 15, tzinfo=UTC), periods=24, freq="h")
    )
    assert clearsky.max() > winter.max()


def test_plane_power_respects_geometry_orientation():
    times = pd.date_range(datetime(2026, 6, 15, 6, tzinfo=UTC), periods=6, freq="h")
    irradiance = pd.Series(800.0, index=times)
    south = plane_power_kw(GEOMETRY, times, irradiance, irradiance, irradiance * 0.1)
    north = plane_power_kw(
        PvGeometry(45.0, 34.0, 25, 0, 7.0), times, irradiance, irradiance, irradiance * 0.1
    )
    assert south.sum() > north.sum()


def test_estimate_geometry_recovers_synthetic_orientation():
    frame = synthetic_frame(20)
    result = estimate_geometry(frame["pv_measured_kw"], 45.0, 34.0)
    geometry = result["geometry"]
    assert geometry.estimated_from_data
    assert abs(geometry.tilt_deg - 25) <= 10
    assert abs(geometry.azimuth_deg - 180) <= 20
    assert geometry.kwp == pytest.approx(0.75 * 7.0, rel=0.1)


@needs_lightgbm
def test_predictions_bounded_ordered_and_zero_at_night():
    frame = synthetic_frame(40, noise=0.05)
    bundle = PvQuantileBundle.fit(frame, GEOMETRY, params=FAST_PARAMS)
    prediction = bundle.predict(frame)
    assert np.isfinite(prediction.to_numpy()).all()
    assert (prediction["pv_p10_kw"] <= prediction["pv_p50_kw"]).all()
    assert (prediction["pv_p50_kw"] <= prediction["pv_p90_kw"]).all()
    assert (prediction <= GEOMETRY.kwp).all().all()
    night = clear_sky_power_kw(GEOMETRY, frame.index) < bundle.min_clearsky_kw
    assert (prediction.loc[night] == 0).all().all()
    # Ratio model recovers the synthetic 0.75 share near noon.
    noon = frame.index.hour == 10
    assert (prediction.loc[noon, "pv_p50_kw"] / clear_sky_power_kw(GEOMETRY, frame.index[noon])) \
        .mean() == pytest.approx(0.75, abs=0.1)


@needs_lightgbm
def test_night_mask_survives_positive_calibration_offsets():
    """Conformal P10/P90 offsets must not fabricate night generation."""
    frame = synthetic_frame(60, noise=0.1)
    train, calibration = frame.iloc[: 40 * 24], frame.iloc[40 * 24 :]
    bundle = PvQuantileBundle.fit(train, GEOMETRY, params=FAST_PARAMS)
    shifted = calibration.copy()
    shifted["pv_measured_kw"] = shifted["pv_measured_kw"] * 1.3
    bundle.calibrate(shifted)
    assert bundle.calibration_offsets["p90"] > 0
    prediction = bundle.predict(frame)
    night = clear_sky_power_kw(GEOMETRY, frame.index) < bundle.min_clearsky_kw
    assert night.any()
    assert (prediction.loc[night] == 0).all().all()


@needs_lightgbm
def test_excluded_rows_count_censoring():
    frame = synthetic_frame(40)
    frame.iloc[::7, frame.columns.get_loc("pv_measured_kw")] = np.nan
    bundle = PvQuantileBundle.fit(frame, GEOMETRY, params=FAST_PARAMS)
    assert bundle.excluded_rows > 0
    with pytest.raises(ValueError, match="at least 24"):
        PvQuantileBundle.fit(frame.iloc[:24], GEOMETRY, params=FAST_PARAMS)


@needs_lightgbm
def test_calibration_and_save_load_roundtrip(tmp_path):
    frame = synthetic_frame(60, noise=0.1)
    train, calibration = frame.iloc[: 40 * 24], frame.iloc[40 * 24 :]
    bundle = PvQuantileBundle.fit(train, GEOMETRY, params=FAST_PARAMS)
    shifted = calibration.copy()
    shifted["pv_measured_kw"] = shifted["pv_measured_kw"] * 1.3
    bundle.calibrate(shifted)
    assert bundle.calibration_offsets["p90"] > 0
    bundle.save(tmp_path)
    restored = PvQuantileBundle.load(tmp_path)
    pd.testing.assert_frame_equal(bundle.predict(frame), restored.predict(frame))
    metadata = json.loads((tmp_path / "bundle.json").read_text())
    assert metadata["geometry"]["kwp"] == GEOMETRY.kwp
    assert metadata["model_version"] == "pv-quantiles-1"
    assert try_load(tmp_path) is not None
    assert try_load(tmp_path / "missing") is None


@needs_lightgbm
def test_geometry_validation():
    with pytest.raises(ValueError, match="tilt"):
        PvGeometry(45, 34, -5, 180, 7.0)
    with pytest.raises(ValueError, match="kwp"):
        PvGeometry(45, 34, 25, 180, 0.0)
