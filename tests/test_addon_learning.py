"""Forecast log, online calibration and drift between retrains."""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import learning  # noqa: E402
import retrain  # noqa: E402

TZ = ZoneInfo("UTC")
NOW = datetime(2026, 11, 20, 0, 0, tzinfo=UTC)


def _history(days=14, actual=lambda p50, rng: p50 + 0.5, band=0.2, method="model", seed=1):
    """Daily issues at 00:00 UTC with a flat 1 kW P50 and a +/-band P10/P90."""
    rng = np.random.default_rng(seed)
    log, rows = [], {}
    for day in range(days):
        issue = NOW - timedelta(days=days - day)
        starts = pd.date_range(issue, periods=24, freq="h")
        p50 = np.full(24, 1.0)
        log.append(
            learning.log_entry(
                issue,
                starts,
                (p50 - band, p50, p50 + band),
                method,
                (p50 * 0, p50 * 0, p50 * 0),
                "model",
            )
        )
        for start, value in zip(starts, actual(p50, rng), strict=True):
            rows[start] = value
    index = pd.DatetimeIndex(sorted(rows))
    audit = pd.DataFrame(
        {
            "load_kw": [rows[i] for i in index],
            "load_filter": "keep",
            "pv_measured_kw": 0.0,
            "pv_filter": "keep",
            "solar_elevation_deg": -5.0,
        },
        index=index,
    )
    return log, audit


def test_log_append_dedupes_and_trims(tmp_path):
    path = tmp_path / "log.jsonl"
    starts = pd.date_range(NOW, periods=2, freq="h")
    old = learning.log_entry(
        NOW - timedelta(days=90), starts, ([0], [0], [0]), "model", ([0], [0], [0]), "model"
    )
    entry = learning.log_entry(NOW, starts, ([1], [1], [1]), "model", ([0], [0], [0]), "model")
    learning.append_forecast_log(path, old, now=NOW - timedelta(days=90))
    learning.append_forecast_log(path, entry, now=NOW)
    learning.append_forecast_log(path, entry, now=NOW)
    entries = learning.read_forecast_log(path)
    assert [e["issue_utc"] for e in entries] == [entry["issue_utc"]]


def test_torn_line_is_skipped(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"issue_utc": "2026-11-19T00:00:00+00:00"}\n{"issue', encoding="utf-8")
    assert len(learning.read_forecast_log(path)) == 1


def test_pairs_skip_fallback_and_compute_lead():
    log, audit = _history(days=3)
    log[0]["load"]["method"] = "fallback"
    pairs = learning.forecast_pairs(log, audit, "load", NOW - timedelta(days=10))
    assert len(pairs) == 48  # two model issues x 24 h
    assert pairs["lead"].min() == 1 and pairs["lead"].max() == 24


def test_identity_below_min_pairs():
    log, audit = _history(days=1)
    calibrations, _ = learning.self_correct(log, audit, TZ, NOW)
    assert not calibrations["load"].active
    p10, p50, p90 = calibrations["load"].apply([0.8], [1.0], [1.2], pd.DatetimeIndex([NOW]), TZ)
    assert (p10[0], p50[0], p90[0]) == (0.8, 1.0, 1.2)


def test_load_bias_is_shifted_towards_actuals():
    log, audit = _history(days=14)  # actual is always 0.5 kW above P50
    calibration = learning.self_correct(log, audit, TZ, NOW)[0]["load"]
    assert calibration.active
    # 14 pairs per hour, shrunk with prior 10: 0.5 * 14 / 24.
    assert np.allclose(calibration.hourly, 0.5 * 14 / 24)
    _, p50, _ = calibration.apply(
        [0.8] * 24, [1.0] * 24, [1.2] * 24, pd.date_range(NOW, periods=24, freq="h"), TZ
    )
    assert np.all(p50 > 1.25)


def test_too_narrow_band_is_widened():
    # Band of +/-0.4 kW against N(0, 1) errors: ~31 % raw coverage.
    log, audit = _history(days=14, actual=lambda p50, rng: p50 + rng.normal(0, 1.0, 24), band=0.4)
    calibration = learning.self_correct(log, audit, TZ, NOW)[0]["load"]
    assert calibration.scale_lo > 2 and calibration.scale_hi > 2
    assert calibration.coverage_raw < 0.45
    assert calibration.coverage_calibrated > 0.65


def test_band_scale_is_bounded():
    log, audit = _history(days=14, actual=lambda p50, rng: p50 + rng.normal(0, 1.0, 24), band=0.01)
    calibration = learning.self_correct(log, audit, TZ, NOW)[0]["load"]
    assert calibration.scale_lo <= learning.SCALE_BOUNDS[1]
    assert calibration.scale_hi <= learning.SCALE_BOUNDS[1]


def test_pv_factor_scales_and_keeps_night_zero():
    log, audit = _history(days=14)
    for entry in log:
        entry["pv"] = {"method": "model", "p10": [1.0] * 24, "p50": [2.0] * 24, "p90": [3.0] * 24}
    audit["pv_measured_kw"] = 1.4  # 70 % of the forecast
    audit["solar_elevation_deg"] = 30.0
    calibration = learning.self_correct(log, audit, TZ, NOW)[0]["pv"]
    assert calibration.active
    assert all(f < 1.0 for f in calibration.hourly)
    p10, p50, p90 = calibration.apply(
        [0, 1.0], [0, 2.0], [0, 3.0], pd.date_range(NOW, periods=2, freq="h"), TZ
    )
    assert (p10[0], p50[0], p90[0]) == (0.0, 0.0, 0.0)
    assert p50[1] < 2.0 and p10[1] <= p50[1] <= p90[1]


def test_drift_flags_persistent_bias():
    log, audit = _history(days=10, actual=lambda p50, rng: p50 * 2.0)
    _, drift = learning.self_correct(log, audit, TZ, NOW)
    assert drift["drift"]
    assert any(reason.startswith("load") for reason in drift["reasons"])


def test_no_drift_on_good_forecast():
    log, audit = _history(days=10, actual=lambda p50, rng: p50 + rng.normal(0, 0.1, 24))
    _, drift = learning.self_correct(log, audit, TZ, NOW)
    assert not drift["drift"]


def _retrain_env(monkeypatch, tmp_path, drift):
    monkeypatch.setattr(retrain, "DATA", tmp_path)
    monkeypatch.setattr(retrain, "STATE", tmp_path / "retrain_state.json")
    (tmp_path / "options.json").write_text(
        json.dumps({"retrain_interval_days": 7, "latitude": 1, "longitude": 2}), encoding="utf-8"
    )
    last = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    (tmp_path / "retrain_state.json").write_text(
        json.dumps({"last_retrain_utc": last, "models": {}})
    )
    (tmp_path / "drift_state.json").write_text(json.dumps({"drift": drift}))

    def no_audit():
        raise OSError("no audit in this test")

    monkeypatch.setattr(retrain, "_audit_frame", no_audit)
    return last


def test_drift_triggers_early_retrain(monkeypatch, tmp_path):
    _retrain_env(monkeypatch, tmp_path, drift=True)
    status = retrain.retrain_if_due()
    assert status["models"]["load"]["status"] == "failed"  # it ran (and hit the missing audit)


def test_no_drift_waits_for_interval(monkeypatch, tmp_path):
    last = _retrain_env(monkeypatch, tmp_path, drift=False)
    status = retrain.retrain_if_due()
    assert status["last_retrain_utc"] == last and status["models"] == {}


def test_correction_follows_the_live_model_version():
    """Review case: a retrained model must not inherit the old model's bias fix."""
    log, audit = _history(days=14)  # old model: always 0.5 kW too low
    for entry in log:
        entry["load"]["version"] = "old"
    calibrations, _ = learning.self_correct(log, audit, TZ, NOW, versions={"load": "new"})
    assert not calibrations["load"].active  # nothing issued by the new model yet
    _, p50, _ = calibrations["load"].apply([1.5], [2.0], [2.5], pd.DatetimeIndex([NOW]), TZ)
    assert p50[0] == 2.0  # was 2.0 + inherited shift
    same = learning.self_correct(log, audit, TZ, NOW, versions={"load": "old"})[0]["load"]
    assert same.active and same.hourly[0] > 0.2


def test_log_entry_records_versions_and_bundle_hash(tmp_path):
    starts = pd.date_range(NOW, periods=1, freq="h")
    entry = learning.log_entry(
        NOW, starts, ([0], [0], [0]), "model", ([0], [0], [0]), "model", versions={"load": "abc"}
    )
    assert entry["load"]["version"] == "abc" and entry["pv"]["version"] is None
    (tmp_path / "bundle.json").write_text('{"a": 1}')
    first = learning.bundle_version(tmp_path)
    (tmp_path / "bundle.json").write_text('{"a": 2}')
    assert first and learning.bundle_version(tmp_path) != first
    assert learning.bundle_version(tmp_path / "missing") is None


def _house(ratio, outages=30, hours=240):
    index = pd.date_range("2026-11-10", periods=hours, freq="h", tz="UTC")
    audit = pd.DataFrame(
        {
            "load_kw": 0.6,
            "critical_load_kw": 0.6 * ratio,
            "raw_grid_status": "up",
            "load_filter": "keep",
        },
        index=index,
    )
    hit = audit.index[:outages]
    audit.loc[hit, ["raw_grid_status", "load_filter"]] = ["outage", "grid_quality"]
    return audit


def test_whole_house_backup_keeps_outage_hours_as_load_targets():
    audit = _house(ratio=0.99)
    assert learning.whole_house_on_backup(audit)
    assert learning.load_target_mask(audit).sum() == 240  # was 210


def test_partial_backup_still_excludes_outage_hours():
    audit = _house(ratio=0.6)
    assert not learning.whole_house_on_backup(audit)
    assert learning.load_target_mask(audit).sum() == 210


def test_whole_house_needs_enough_normal_hours_and_columns():
    assert not learning.whole_house_on_backup(_house(ratio=1.0, outages=230))
    assert not learning.whole_house_on_backup(pd.DataFrame({"load_kw": [1.0]}))
    # Hours invalid for other reasons (missing load) never become targets.
    invalid = _house(ratio=0.99)
    invalid.loc[invalid.index[:5], "load_filter"] = "invalid_target"
    assert not learning.load_target_mask(invalid).iloc[:5].any()


def test_retrain_trains_on_outage_hours_of_a_whole_house_site(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain, "MODELS", tmp_path)
    audit = _house(ratio=0.99, hours=24 * 20)
    audit["temp_air_observed_c"] = 5.0
    info = retrain._retrain_load(audit, {"time_zone": "UTC"})
    assert info["train_rows"] == 24 * 20
