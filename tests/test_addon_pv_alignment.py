"""PV model inputs aligned to the measured hour; one retrain after the upgrade."""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402
import retrain  # noqa: E402

from emhass.pv_quantiles import PvGeometry, clear_sky_power_kw, plane_power_kw  # noqa: E402

GEOMETRY = PvGeometry(45.0, 34.0, 26.0, 180.0, 7.0)
DAY = pd.date_range("2026-11-21 00:00", periods=24, freq="h", tz="UTC")


def _open_meteo_clear_day(stamps):
    """Clear-sky irradiance the way Open-Meteo stamps it: mean of [T-1h, T)."""
    location = pvlib.location.Location(GEOMETRY.latitude, GEOMETRY.longitude)
    rows = []
    for stamp in stamps:
        inside = pd.date_range(stamp - pd.Timedelta(hours=1), stamp, freq="5min", inclusive="left")
        sky = location.get_clearsky(inside, model="ineichen").mean()
        rows.append(
            {
                "shortwave_radiation": sky["ghi"],
                "direct_normal_irradiance": sky["dni"],
                "diffuse_radiation": sky["dhi"],
            }
        )
    frame = pd.DataFrame(rows, index=stamps)
    frame["cloud_cover"] = np.arange(len(stamps), dtype=float)
    frame["temperature_2m"] = 10.0
    return frame


def test_hour_takes_the_radiation_stamped_one_hour_later():
    stamps = DAY.append(DAY[-1:] + pd.Timedelta(hours=1))
    weather = pd.DataFrame(
        {
            "shortwave_radiation": 0.0,
            "direct_normal_irradiance": 0.0,
            "diffuse_radiation": 0.0,
            "cloud_cover": np.arange(len(stamps), dtype=float),
            "temperature_2m": 5.0,
        },
        index=stamps,
    )
    weather.loc[DAY[10], ["shortwave_radiation", "diffuse_radiation"]] = 300.0  # mean 09-10
    frame = pipeline.pv_weather_frame(GEOMETRY, weather, DAY)
    assert (frame.index == DAY + pd.Timedelta(minutes=30)).all()
    lit = frame["poa_forecast_kw"] > 0
    assert lit.tolist() == [i == 9 for i in range(24)]  # the hour starting 09:00
    assert frame["cloud_cover_pct"].iloc[9] == 9.5  # instants: mean of 09:00 and 10:00
    assert np.isnan(frame["pv_measured_kw"]).all()


def test_last_hour_without_the_next_stamp_keeps_the_instant_values():
    weather = _open_meteo_clear_day(DAY)  # no stamp after the last hour
    frame = pipeline.pv_weather_frame(GEOMETRY, weather, DAY)
    assert np.isnan(frame["poa_forecast_kw"].iloc[-1])
    assert frame["cloud_cover_pct"].iloc[-1] == 23.0


def test_aligned_forecast_follows_a_clear_day_hour_by_hour():
    weather = _open_meteo_clear_day(DAY.append(DAY[-1:] + pd.Timedelta(hours=1)))
    measured = clear_sky_power_kw(GEOMETRY, DAY + pd.Timedelta(minutes=30)).to_numpy()
    aligned = pipeline.pv_weather_frame(GEOMETRY, weather, DAY)["poa_forecast_kw"].to_numpy()
    # Before v2: radiation stamped H (the hour before) with the sun at H.
    lagged = plane_power_kw(
        GEOMETRY,
        DAY,
        weather["shortwave_radiation"].reindex(DAY),
        weather["direct_normal_irradiance"].reindex(DAY),
        weather["diffuse_radiation"].reindex(DAY),
    ).to_numpy()
    error_aligned = np.abs(aligned - measured).sum()
    error_lagged = np.abs(lagged - measured).sum()
    assert error_aligned < 0.2 * error_lagged
    assert np.argmax(aligned) == np.argmax(measured)
    # The first light hour: the old join fed it the dark hour before sunrise.
    first = int(np.argmax(measured > 0.5))
    assert lagged[first] < 0.5 * measured[first] < aligned[first]


def _upgrade_env(monkeypatch, tmp_path, state, options=None):
    monkeypatch.setattr(retrain, "DATA", tmp_path)
    monkeypatch.setattr(retrain, "STATE", tmp_path / "retrain_state.json")
    monkeypatch.setattr(retrain, "MODELS", tmp_path / "models")
    (tmp_path / "options.json").write_text(json.dumps(options or {}), encoding="utf-8")
    (tmp_path / "retrain_state.json").write_text(json.dumps(state), encoding="utf-8")
    calls = []
    audit = pd.DataFrame({"load_kw": [1.0]}, index=DAY[:1])
    monkeypatch.setattr(retrain, "_audit_frame", lambda: audit)
    monkeypatch.setattr(retrain, "_ensure_weather_history", lambda *a: None)
    monkeypatch.setattr(
        retrain, "_retrain_load", lambda *a: calls.append("load") or {"status": "kept"}
    )
    monkeypatch.setattr(retrain, "_retrain_pv", lambda *a: calls.append("pv") or {"status": "kept"})
    return calls


SITE = {"latitude": 45.0, "longitude": 34.0}


@pytest.mark.parametrize("options", [{}, {"retrain_enabled": False}])
def test_upgrade_retrains_the_pv_model_once(monkeypatch, tmp_path, options):
    recent = (datetime.now(UTC) - timedelta(days=1)).isoformat()  # interval not elapsed
    calls = _upgrade_env(monkeypatch, tmp_path, {"last_retrain_utc": recent}, options)
    retrain.retrain_if_due(site=SITE)
    assert calls == ["load", "pv"]
    state = retrain.data_store.read_json(tmp_path / "retrain_state.json")
    assert state["pv_model_tried"] == pipeline.PV_MODEL
    retrain.retrain_if_due(site=SITE)
    assert calls == ["load", "pv"]  # back on the normal schedule


def test_installed_v2_model_needs_no_upgrade_retrain(monkeypatch, tmp_path):
    recent = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    calls = _upgrade_env(monkeypatch, tmp_path, {"last_retrain_utc": recent})
    model = tmp_path / "models" / pipeline.PV_MODEL
    model.mkdir(parents=True)
    (model / "bundle.json").write_text("{}", encoding="utf-8")
    retrain.retrain_if_due(site=SITE)
    assert calls == []
