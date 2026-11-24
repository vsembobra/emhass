"""Corrected panels in the settings reach the PV model.

The PV model's features are the clear-sky and plane-of-array power of the
geometry it was first built on, and later retrains reused it: an owner who
fixed the azimuth (210 -> 170 degrees: peak at 12:40 local where solar noon
is 12:52) kept the old, wrong-way model for good.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402
import retrain  # noqa: E402

from emhass.pv_quantiles import PvGeometry  # noqa: E402

SITE = {
    "latitude": 41.0,
    "longitude": 28.6,
    "panel_tilt_deg": 35.0,
    "panel_azimuth_deg": 170.0,
    "panel_kwp": 7.0,
    "max_charge_kw": 6.1,
}
OLD = PvGeometry(41.0, 28.6, 35.0, 210.0, 7.0, estimated_from_data=True)


def test_geometry_changed_tolerates_rounding_only():
    same = PvGeometry(41.0, 28.6, 35.0, 170.0, 7.0)
    assert not retrain.geometry_changed(same, SITE)
    assert retrain.geometry_changed(OLD, SITE)  # azimuth 210 -> 170
    assert retrain.geometry_changed(same, {**SITE, "panel_tilt_deg": 40.0})
    assert retrain.geometry_changed(same, {**SITE, "panel_kwp": 10.0})
    assert not retrain.geometry_changed(same, {**SITE, "panel_kwp": 7.2})
    north = PvGeometry(41.0, 28.6, 35.0, 358.0, 7.0)
    assert not retrain.geometry_changed(
        north, {**SITE, "panel_azimuth_deg": 2.0}
    )  # 4 degrees apart


def test_the_saved_model_geometry_is_read_from_its_metadata(tmp_path):
    folder = tmp_path / pipeline.PV_MODEL
    folder.mkdir()
    (folder / "bundle.json").write_text(json.dumps({"geometry": OLD.__dict__}))
    assert retrain.pv_geometry_changed(tmp_path, SITE)
    assert not retrain.pv_geometry_changed(tmp_path, {**SITE, "panel_azimuth_deg": 210.0})
    assert not retrain.pv_geometry_changed(tmp_path / "missing", SITE)


def _frame():
    index = pd.date_range("2026-11-20 08:00", periods=6, freq="h", tz="UTC")
    audit = pd.DataFrame(
        {
            "pv_measured_kw": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "pv_filter": ["keep"] * 6,
            "soc_mean": [0.5] * 6,
            "export_kwh": [0.0] * 6,
            "battery_power_kw_raw_sign": [-1.0] * 6,
            "grid_power_kw_raw_sign": [0.0] * 6,
        },
        index=index,
    )
    weather = pd.DataFrame(
        {
            "shortwave_radiation_previous_day1": 600.0,
            "direct_normal_irradiance_previous_day1": 500.0,
            "diffuse_radiation_previous_day1": 100.0,
            "cloud_cover_previous_day1": 20.0,
            "temperature_2m_previous_day1": 10.0,
        },
        index=index.append(index[-1:] + pd.Timedelta(hours=1)),
    )
    return audit, weather


class _Champion:
    geometry = OLD
    ratio_max = 1.2
    min_clearsky_kw = 0.1
    trained_until = None


def test_a_settings_change_rebuilds_the_model_on_the_new_geometry(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain, "MODELS", tmp_path)
    audit, weather = _frame()
    monkeypatch.setattr(retrain, "_try_load", lambda loader, path: _Champion())
    captured = {}

    def fake_fit(frame, geometry, **kwargs):
        captured["geometry"] = geometry
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(retrain.PvQuantileBundle, "fit", fake_fit)
    with pytest.raises(RuntimeError, match="stop after capture"):
        retrain._retrain_pv(audit, SITE, weather)
    assert captured["geometry"].azimuth_deg == 170.0  # the settings, not the old model's 210


def test_unchanged_settings_keep_the_model_geometry(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain, "MODELS", tmp_path)
    audit, weather = _frame()
    monkeypatch.setattr(retrain, "_try_load", lambda loader, path: _Champion())
    result = retrain._retrain_pv(audit, {**SITE, "panel_azimuth_deg": 210.0}, weather)
    assert result["status"] == "kept" and "geometry_changed" not in result
