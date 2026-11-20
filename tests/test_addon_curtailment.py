"""PV censoring on no-export sites: full battery, charge taper, charge saturation."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402
import retrain  # noqa: E402

limited = pipeline.pv_sink_limited


def test_full_battery_rule_kept():
    assert limited(0.97, 0.0, 0.0, 0.0)


def test_charge_taper_near_full():
    # 92 %, charging 1 kW, no import, no export: the BMS caps the current.
    assert limited(0.92, 0.0, -1.0, 0.0)


def test_taper_not_flagged_while_importing():
    assert not limited(0.92, 0.0, -1.0, 0.8)


def test_taper_not_flagged_while_discharging():
    # PV below load: the battery covers the gap, nothing is throttled.
    assert not limited(0.92, 0.0, 0.7, 0.0)


def test_charge_saturation_at_mid_soc():
    assert limited(0.5, 0.0, -4.6, 0.0, max_charge_kw=5.0)
    assert not limited(0.5, 0.0, -3.0, 0.0, max_charge_kw=5.0)
    assert not limited(0.5, 0.0, -4.6, 0.0)  # unknown limit: no saturation rule


def test_export_means_a_sink_existed():
    assert not limited(0.97, 0.5, 0.0, 0.0)
    assert not limited(0.92, 0.5, -1.0, 0.0)


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_unknown_values_never_flag(missing):
    assert not limited(missing, 0.0, -1.0, 0.0)
    assert not limited(0.97, missing, -1.0, 0.0)
    assert not limited(0.92, 0.0, missing, 0.0)


def test_mask_tolerates_missing_columns():
    index = pd.date_range("2026-06-21 08:00", periods=3, freq="h", tz="UTC")
    audit = pd.DataFrame(
        {"soc_mean": [0.92, 0.5, np.nan], "export_kwh": [0.0, 0.0, 0.0]}, index=index
    )
    # No battery/grid columns: only the full-battery rule can fire.
    assert list(pipeline.pv_sink_limited_mask(audit, 5.0)) == [False, False, False]
    audit["battery_power_kw_raw_sign"] = [-1.0, -4.8, -1.0]
    assert list(pipeline.pv_sink_limited_mask(audit, 5.0)) == [True, True, False]


NOON = datetime(2026, 6, 21, 9, 0, tzinfo=UTC)  # 12:00 local at 45N 34E


def _states(value, minutes=-5):
    at = (NOON + timedelta(minutes=minutes)).isoformat()
    return [{"state": str(value), "last_changed": at, "last_updated": at}]


def _audit_row(soc_pct, battery_w, max_charge_kw):
    ent = {
        "mean_sensors": {
            "load_kw": ("sensor.load", 0.001),
            "critical_load_kw": ("sensor.ups", 0.001),
            "pv_measured_kw": ("sensor.pv", 0.001),
            "battery_power_kw_raw_sign": ("sensor.battery", 0.001),
            "grid_power_kw_raw_sign": ("sensor.grid", 0.001),
        },
        "voltages": {},
        "soc": "sensor.soc",
        "export_counter": "sensor.export",
        "grid_binary": "binary_sensor.grid",
    }
    history = {
        "sensor.load": _states(500),
        "sensor.ups": _states(300),
        "sensor.pv": _states(500 - battery_w),
        "sensor.battery": _states(battery_w),
        "sensor.grid": _states(0),
        "sensor.soc": _states(soc_pct, minutes=20),
        "sensor.export": _states(100.0, minutes=-30) + _states(100.0, minutes=70),
        "binary_sensor.grid": _states("on", minutes=-600),
    }
    (row,) = pipeline.build_audit_rows(
        history, [NOON], 45.0, 34.0, ent, max_charge_kw=max_charge_kw
    )
    return row


def test_audit_flags_taper_hour():
    row = _audit_row(92, -1000, 5.0)
    assert row["pv_filter"] == "possible_curtailment"
    assert row["possible_pv_curtailment"] is True


def test_audit_flags_saturated_hour():
    assert _audit_row(50, -4800, 5.0)["pv_filter"] == "possible_curtailment"


def test_audit_keeps_unconstrained_hour():
    assert _audit_row(50, -2000, 5.0)["pv_filter"] == "keep"


def test_retrain_excludes_censored_rows_recorded_before_the_fix(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain, "MODELS", tmp_path)
    index = pd.date_range("2026-06-21 06:00", periods=6, freq="h", tz="UTC")
    audit = pd.DataFrame(
        {
            "pv_measured_kw": [2.0, 3.0, 4.0, 4.0, 3.0, 2.0],
            # Old rows: classified with the SOC >= 95 % rule only.
            "pv_filter": ["keep"] * 6,
            "soc_mean": [0.5, 0.5, 0.92, 0.93, 0.5, 0.5],
            "export_kwh": [0.0] * 6,
            "battery_power_kw_raw_sign": [-1.0, -4.8, -1.0, -0.5, -1.0, -1.0],
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
            "temperature_2m_previous_day1": 20.0,
        },
        index=index,
    )
    captured = {}

    def fake_fit(train, *args, **kwargs):
        captured["train"] = train
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(retrain.PvQuantileBundle, "fit", fake_fit)
    site = {
        "latitude": 45.0,
        "longitude": 34.0,
        "panel_tilt_deg": 26.0,
        "panel_azimuth_deg": 180.0,
        "panel_kwp": 7.0,
        "max_charge_kw": 5.0,
    }
    with pytest.raises(RuntimeError, match="stop after capture"):
        retrain._retrain_pv(audit, site, weather)
    kept = captured["train"]["pv_measured_kw"]
    assert kept.isna().tolist() == [False, True, True, True, False, False]


def test_fill_audit_temperature_creates_missing_column():
    index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    audit = pd.DataFrame({"load_kw": [1.0, 1.0, 1.0]}, index=index)
    weather = pd.DataFrame({"temperature_2m_previous_day1": [4.0, 5.0, 6.0]}, index=index)
    out = retrain._fill_audit_temperature(audit, weather)
    assert list(out["temp_air_observed_c"]) == [4.0, 5.0, 6.0]
    assert retrain._fill_audit_temperature(audit, None)["temp_air_observed_c"].isna().all()
