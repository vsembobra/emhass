"""Lossless compact storage must preserve training data across migration/failure."""

import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

APP = Path(__file__).resolve().parent.parent / "addon_emhass_shadow" / "app"
sys.path.insert(0, str(APP))
import data_store as store  # noqa: E402


def test_legacy_to_gzip_preserves_every_training_value(tmp_path):
    path = tmp_path / "hourly_audit.jsonl"
    rows = [
        {"start_utc": "2026-10-25T00:00:00Z", "load_kw": 1.23456789012345,
         "pv_filter": "possible_curtailment", "grid_quality": "recorded_outage", "soc_mean": None},
        {"start_utc": "2026-10-25T01:00:00+00:00", "load_kw": 0.0,
         "pv_filter": "keep", "grid_quality": "ok", "soc_mean": 0.51},
    ]
    text = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(text)
    before = store.read_frame(path)
    assert store.compact_existing(path)
    assert not path.exists() and store.exists(path)
    assert gzip.decompress(Path(str(path) + ".gz").read_bytes()).decode() == text
    pd.testing.assert_frame_equal(store.read_frame(path), before)
    assert not store.compact_existing(path)  # retry/restart is idempotent


def test_append_keeps_all_old_rows_and_empty_or_null_values(tmp_path):
    path = tmp_path / "audit.jsonl"
    old = '{"start_utc":"2025-01-01T00:00:00Z","x":null,"quality":""}\n'
    path.write_text(old)
    store.append_jsonl(path, [{"start_utc": "2026-11-26T00:00:00Z", "x": 3.141592653589793}])
    assert store.read_text(path).startswith(old)
    rows = [json.loads(line) for line in store.read_text(path).splitlines()]
    assert rows[0]["x"] is None and rows[0]["quality"] == ""
    assert rows[1]["x"] == 3.141592653589793


def test_failed_atomic_publish_leaves_last_good_plain_data(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"load_kw":1}\n')
    def interrupted(*args):
        raise OSError("simulated power failure")
    monkeypatch.setattr(store.os, "replace", interrupted)
    with pytest.raises(OSError, match="power failure"):
        store.append_jsonl(path, [{"load_kw": 2}])
    assert store.read_text(path) == '{"load_kw":1}\n'
    assert list(tmp_path.iterdir()) == [path]


def test_failed_update_leaves_last_good_compressed_data(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    store.write_json(path, {"version": 1})
    monkeypatch.setattr(store.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("full disk")))
    with pytest.raises(OSError, match="full disk"):
        store.write_json(path, {"version": 2})
    assert store.read_json(path) == {"version": 1}


def test_logical_reports_are_unique_and_keep_date_names(tmp_path):
    path = tmp_path / "2026-11-26.json"
    store.write_json(path, {"issue_utc": "new"})
    path.write_text('{"issue_utc":"old"}')  # interrupted old-file cleanup
    assert store.json_paths(tmp_path) == [path]
    assert store.read_json(path)["issue_utc"] == "new"
    assert not store.compact_existing(path)  # different legacy data is preserved
    assert path.read_text() == '{"issue_utc":"old"}'
    store.write_json(path, {"issue_utc": "newer"})
    assert path.read_text() == '{"issue_utc":"old"}'  # no silent loss on next update


def test_compression_is_substantial_and_does_not_truncate_seasons(tmp_path):
    path = tmp_path / "audit.jsonl"
    rows = [{"start_utc": f"{stamp.isoformat()}", "load_kw": (i % 37) / 7,
             "pv_filter": "keep", "grid_quality": "ok", "soc_mean": (i % 90) / 100}
            for i, stamp in enumerate(pd.date_range("2025-01-01", periods=24 * 365, freq="h", tz="UTC"))]
    store.append_jsonl(path, rows)
    raw = store.read_text(path)
    assert len(raw.splitlines()) == 24 * 365
    assert Path(str(path) + ".gz").stat().st_size < len(raw.encode()) * 0.3


def test_migration_covers_each_weather_source_and_leaves_manual_files(tmp_path):
    paths = [tmp_path / "hourly_audit.jsonl", tmp_path / "forecast_log.jsonl",
             tmp_path / "weather_models/icon/weather_previous_day1.jsonl",
             tmp_path / "shadow_reports/2026-11-25.json", tmp_path / "weather_archive/20261125T000000Z.json"]
    for i, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source": i}) + "\n")
    manual = tmp_path / "manual_notes.json"
    manual.write_text('{"keep":true}')
    custom = tmp_path / "weather_models/custom/weather_previous_day1.jsonl"
    custom.parent.mkdir(parents=True)
    custom.write_text('{"user_backup":true}')
    store.compact_data(tmp_path)
    for i, path in enumerate(paths):
        assert store.read_json(path) == {"source": i}
        assert not path.exists()
    assert manual.exists()
    assert custom.exists()


def test_compaction_does_not_follow_a_linked_weather_models_parent(tmp_path):
    data, outside = tmp_path / "data", tmp_path / "outside"
    data.mkdir()
    path = outside / "icon/weather_previous_day1.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"keep":true}\n')
    (data / "weather_models").symlink_to(outside, target_is_directory=True)
    store.compact_data(data)
    assert path.exists() and not store.compressed(path).exists()


def test_restart_removes_only_verified_redundant_interrupted_writes(tmp_path):
    path = tmp_path / "hourly_audit.jsonl"
    store.write_text(path, '{"load_kw":1}\n')
    duplicate = tmp_path / ".emhass-shadow-hourly_audit.jsonl.gz.duplicate"
    duplicate.write_bytes(gzip.compress(store.read_bytes(path)))
    unique = tmp_path / ".emhass-shadow-hourly_audit.jsonl.gz.unique"
    unique.write_bytes(gzip.compress(b'{"load_kw":2}\n'))
    manual = tmp_path / ".hourly_audit.jsonl.gz.manual"
    manual.write_bytes(duplicate.read_bytes())
    damaged = tmp_path / ".emhass-shadow-hourly_audit.jsonl.gz.damaged"
    raw = bytearray(duplicate.read_bytes())
    raw[10] = 0x07  # valid gzip header, invalid DEFLATE block
    damaged.write_bytes(raw)
    store.compact_data(tmp_path)
    assert not duplicate.exists()
    assert unique.exists() and manual.exists()
    assert damaged.exists()


def test_compressed_history_retrains_both_models_and_keeps_quality_masks(tmp_path, monkeypatch):
    sys.path.insert(0, str(APP.parent / "src"))
    import numpy as np
    import pipeline
    import retrain

    from emhass.load_quantiles import LoadQuantileBundle
    from emhass.pv_quantiles import PvQuantileBundle

    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setattr(retrain, "DATA", tmp_path)
    monkeypatch.setattr(retrain, "MODELS", tmp_path / "models")
    monkeypatch.setattr(retrain, "WEATHER_HIST", tmp_path / "weather_previous_day1.jsonl")
    site = {"latitude": 45.0, "longitude": 30.0, "time_zone": "UTC", "panel_kwp": 7.0,
            "panel_tilt_deg": 30.0, "panel_azimuth_deg": 180.0, "max_charge_kw": 5.0}
    end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=1)
    index = pd.date_range(end=end, periods=24 * 30, freq="h")
    daylight = np.clip(np.sin((index.hour.to_numpy() - 6) / 12 * np.pi), 0, None)
    weather = pd.DataFrame({"shortwave_radiation_previous_day1": daylight * 600,
        "direct_normal_irradiance_previous_day1": daylight * 500,
        "diffuse_radiation_previous_day1": daylight * 100,
        "cloud_cover_previous_day1": 20.0, "temperature_2m_previous_day1": 10.0}, index=index)
    frame = pipeline.pv_weather_frame(retrain.configured_geometry(site), weather, index, "_previous_day1")
    audit = pd.DataFrame({"load_kw": 1.0 + daylight * .3, "load_filter": "keep",
        "temp_air_observed_c": 10.0, "pv_measured_kw": frame["poa_forecast_kw"].to_numpy() * .85,
        "pv_filter": "keep", "soc_mean": .5, "export_kwh": 0.0,
        "battery_power_kw_raw_sign": .3, "grid_power_kw_raw_sign": .2}, index=index)
    audit.iloc[10, audit.columns.get_loc("pv_filter")] = "possible_curtailment"
    audit_path = tmp_path / "hourly_audit.jsonl"
    audit.reset_index(names="start_utc").to_json(audit_path, orient="records", lines=True, date_format="iso")
    weather_path = tmp_path / "weather_previous_day1.jsonl"
    weather.reset_index(names="start_utc").to_json(weather_path, orient="records", lines=True, date_format="iso")
    before = retrain._audit_frame()
    store.compact_data(tmp_path)
    after = retrain._audit_frame()
    pd.testing.assert_frame_equal(before, after)
    monkeypatch.setattr(retrain, "_fetch_weather_chunk", lambda *a: pytest.fail("stored weather was ignored"))
    learned_load = retrain._retrain_load(after, site)
    learned_pv = retrain._retrain_pv(after, site)
    assert learned_load["status"] == learned_pv["status"] == "promoted"
    assert learned_pv["train_rows"] < len(after)  # censored PV is still excluded
    load = LoadQuantileBundle.load(tmp_path / "models" / "load_quantiles_v1")
    pv = PvQuantileBundle.load(tmp_path / "models" / pipeline.PV_MODEL)
    assert pd.Timestamp(load.trained_until) == after.index[-1]
    assert not load.predict(after.iloc[-48:]).empty
    assert not pv.predict(frame.iloc[-24:]).empty

