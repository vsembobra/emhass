"""Hardware learned from the audit: battery, limits, PV geometry, critical profile."""

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import hardware  # noqa: E402

from emhass.pv_quantiles import PvGeometry, clear_sky_power_kw  # noqa: E402

UTC = ZoneInfo("UTC")
OPTS = {
    "capacity_kwh": 32.0,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "max_charge_kw": 5.0,
    "max_discharge_kw": 5.0,
    "latitude": 45.0,
    "longitude": 34.0,
    "time_zone": "UTC",
}


def _battery_audit(capacity_kwh=20.0, eta=0.95, days=20, seed=0):
    """Daily cycle: charge 3 kW for 5 h, rest, discharge 2 kW for 6 h (hourly means)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-10-01", periods=days * 24, freq="h", tz="UTC")
    power, soc = [], []
    stored = 0.3 * capacity_kwh
    for stamp in index:
        hour = stamp.hour
        kw = -3.0 if 9 <= hour < 14 else (2.0 if 18 <= hour < 24 else 0.0)
        kw *= 1 + rng.normal(0, 0.01)
        soc.append(stored / capacity_kwh)
        power.append(kw)
        stored += -kw * eta if kw < 0 else -kw / eta
    return pd.DataFrame({"battery_power_kw_raw_sign": power, "soc_mean": soc}, index=index)


def test_battery_identification_recovers_capacity():
    info = hardware.identify_battery(_battery_audit(), capacity_kwh=22.0)
    assert info["status"] == "ok", info["messages"]
    assert info["capacity_kwh"] == pytest.approx(20.0, rel=0.08)
    assert info["round_trip_efficiency"] == pytest.approx(0.95**2, abs=0.04)


def test_battery_identification_fails_soft_without_cycles():
    index = pd.date_range("2026-10-01", periods=48, freq="h", tz="UTC")
    audit = pd.DataFrame({"battery_power_kw_raw_sign": 0.0, "soc_mean": 0.5}, index=index)
    assert hardware.identify_battery(audit, 32.0)["status"] == "insufficient_data"


def test_observed_limits_and_taper_bands():
    audit = _battery_audit()
    audit["soc_mean"] = audit["soc_mean"].clip(upper=0.85)
    audit.loc[audit.index.hour == 13, "soc_mean"] = 0.96
    audit.loc[audit.index.hour == 13, "battery_power_kw_raw_sign"] = -0.8  # BMS taper
    limits = hardware.observed_limits(audit)
    assert limits["max_charge_kw"] == pytest.approx(3.0, abs=0.1)
    assert limits["max_discharge_kw"] == pytest.approx(2.0, abs=0.1)
    top = limits["charge_by_soc"][-1]
    assert top["charge_p95_kw"] == pytest.approx(0.8, abs=0.05)


def _clear_days(days):
    truth = PvGeometry(45.0, 34.0, 30.0, 180.0, 5.0)
    index = pd.date_range("2026-06-01", periods=24 * days, freq="h", tz="UTC")
    midpoint_power = clear_sky_power_kw(truth, index + pd.Timedelta(minutes=30)).to_numpy()
    return pd.DataFrame({"pv_measured_kw": midpoint_power, "pv_filter": "keep"}, index=index)


def test_geometry_estimate_recovers_orientation():
    geometry = hardware.identify_geometry(_clear_days(25), 45.0, 34.0)
    assert geometry["tilt_deg"] == pytest.approx(30.0, abs=5)
    assert geometry["azimuth_deg"] == pytest.approx(180.0, abs=10)
    assert geometry["kwp"] == pytest.approx(5.0, rel=0.05)
    assert geometry["reliable"] and geometry["relative_error"] < 0.05


def test_geometry_from_too_few_days_is_not_reliable():
    geometry = hardware.identify_geometry(_clear_days(14), 45.0, 34.0)
    assert not geometry["reliable"] and "мало дней" in geometry["reason"]


def test_cloudy_fit_is_not_reliable():
    # Real-site case: ten cloudy November days fitted a 290 deg roof with a
    # 0.9 kW error on a ~3 kW peak.
    audit = _clear_days(25)
    rng = np.random.default_rng(5)
    # Broken clouds: every hour either clear or nearly dark.
    audit["pv_measured_kw"] = audit["pv_measured_kw"] * rng.choice([0.05, 1.0], len(audit))
    geometry = hardware.identify_geometry(audit, 45.0, 34.0)
    assert not geometry["reliable"] and "ошибка подгонки" in geometry["reason"]


def test_big_turn_against_the_configured_roof_needs_confirmation():
    # A 3 h clock offset fits a 290 deg roof with a small error.
    audit = _clear_days(25)
    audit["pv_measured_kw"] = audit["pv_measured_kw"].shift(3).fillna(0.0)
    geometry = hardware.identify_geometry(audit, 45.0, 34.0, 30.0, 180.0)
    assert geometry["relative_error"] < 0.15
    assert not geometry["reliable"] and "проверьте вручную" in geometry["reason"]
    # The same estimate without a configured roof to compare against.
    assert hardware.identify_geometry(audit, 45.0, 34.0)["reliable"]


def test_small_correction_of_the_configured_roof_is_accepted():
    geometry = hardware.identify_geometry(_clear_days(25), 45.0, 34.0, 25.0, 195.0)
    assert geometry["reliable"]


def test_geometry_needs_enough_days():
    index = pd.date_range("2026-06-01", periods=24 * 3, freq="h", tz="UTC")
    audit = pd.DataFrame({"pv_measured_kw": 1.0, "pv_filter": "keep"}, index=index)
    assert hardware.identify_geometry(audit, 45.0, 34.0) is None


def test_critical_profile_by_hour():
    index = pd.date_range("2026-10-01", periods=24 * 10, freq="h", tz="UTC")
    critical = np.where(index.hour >= 18, 1.2, 0.3)
    audit = pd.DataFrame({"critical_load_kw": critical}, index=index)
    profile = hardware.critical_profile(audit, UTC)
    assert profile[20] == pytest.approx(1.2) and profile[3] == pytest.approx(0.3)


def _learned():
    return {
        "battery": {
            "status": "ok",
            "capacity_kwh": 28.4,
            "eta_one_way": 0.96,
            "round_trip_efficiency": 0.92,
        },
        "limits": {"max_charge_kw": 6.0, "max_discharge_kw": 3.0},
        "pv_geometry": {
            "tilt_deg": 20.0,
            "azimuth_deg": 170.0,
            "kwp": 6.5,
            "mae_kw": 0.1,
            "reliable": True,
        },
        "critical_profile_kw": [0.3] * 18 + [1.2] * 6,
    }


def test_effective_options_off_by_default():
    effective, notes, _ = hardware.effective_options(dict(OPTS), _learned())
    assert effective == OPTS and notes == []


def test_effective_options_apply_learned_values():
    effective, notes, _ = hardware.effective_options(dict(OPTS, autotune=True), _learned())
    assert effective["capacity_kwh"] == 28.4 and effective["_soh_identified"]
    assert effective["charge_efficiency"] == effective["discharge_efficiency"] == 0.96
    assert effective["max_charge_kw"] == 6.0  # observed above the setting: raised
    assert effective["max_discharge_kw"] == 5.0  # observed below: never lowered
    assert effective["panel_kwp"] == 6.5
    assert len(notes) == 3


def test_effective_options_ignore_unreliable_geometry():
    learned = _learned()
    learned["pv_geometry"]["reliable"] = False
    effective, _, _ = hardware.effective_options(dict(OPTS, autotune=True), learned)
    assert "panel_kwp" not in effective


def test_effective_options_ignore_unconfirmed_battery():
    learned = _learned()
    learned["battery"]["status"] = "low_confidence"
    effective, _, _ = hardware.effective_options(dict(OPTS, autotune=True), learned)
    assert effective["capacity_kwh"] == 32.0 and "_soh_identified" not in effective


def test_effective_options_reject_capacity_from_a_mixed_history():
    # After a battery swap the 60-day window holds both batteries' segments;
    # the pooled estimate (40.9 between 32 and 48) must not move the setting.
    learned = _learned()
    learned["battery"]["capacity_kwh"] = 40.9
    effective, notes, _ = hardware.effective_options(dict(OPTS, autotune=True), learned)
    assert effective["capacity_kwh"] == 32.0 and "_soh_identified" not in effective
    assert effective["charge_efficiency"] == 0.95  # eta travels with the capacity
    assert any("расходится" in note for note in notes)
    # A learned value within the tolerance still applies.
    learned["battery"]["capacity_kwh"] = 38.0
    effective, _, _ = hardware.effective_options(dict(OPTS, autotune=True), learned)
    assert effective["capacity_kwh"] == 38.0


def test_reshape_reserve_follows_profile_and_keeps_autonomy():
    starts = pd.date_range("2026-11-20 00:00", periods=24, freq="h", tz="UTC")
    profile = [0.3] * 18 + [1.2] * 6
    reserve = [4.0] * 24
    shaped = hardware.reshape_reserve(reserve, starts, UTC, profile, 3.0, 2.0, 0.95)
    assert shaped[18] > reserve[18] > shaped[3]
    # Autonomy floor: 2 h of the evening profile.
    assert shaped[18] >= 2 * 1.2 / 0.95
    assert hardware.reshape_reserve(reserve, starts, UTC, None, 3.0, 2.0, 0.95) == reserve


def test_identify_survives_an_empty_audit(tmp_path):
    audit = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    result = hardware.identify(audit, OPTS)
    hardware.save(tmp_path, result)
    assert hardware.load(tmp_path)["battery"]["status"] in ("insufficient_data", "error")
    assert not hardware.due(tmp_path)
