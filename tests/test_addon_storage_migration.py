"""Consume app-owned share copies only after their data is safely preserved."""

import gzip
import json
import sqlite3
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "addon_emhass_shadow" / "app"
sys.path.insert(0, str(APP))
import data_store as store  # noqa: E402
import storage_migration as migration  # noqa: E402


def test_matching_export_formats_are_removed_without_archiving_duplicates(tmp_path):
    data, share = tmp_path / "data", tmp_path / "share"
    report = {"issue_utc": "2026-11-25T00:00:00Z", "note": "spaces inside text stay", "advisory": {}, "reserve": {}}
    target = data / "shadow_reports/2026-11-25.json"
    store.write_json(target, report)
    source = share / "shadow_reports/2026-11-25.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(report, indent=2))
    result = migration.migrate_shared(data, share)
    assert result["removed"] == 1
    assert not source.exists() and store.read_json(target) == report
    assert not list((data / "legacy_imports").rglob("*.gz"))
    assert migration.migrate_shared(data, share)["removed"] == 0


def test_unique_export_is_archived_without_overwriting_current_state(tmp_path):
    data, share = tmp_path / "data", tmp_path / "share"
    target = data / "weather_models/icon/retrain_state.json"
    store.write_json(target, {"models": {"pv": "new"}})
    source = share / "weather_models/icon/retrain_state.json"
    source.parent.mkdir(parents=True)
    old = b'{"models":{"pv":"old"}}'
    source.write_bytes(old)
    result = migration.migrate_shared(data, share)
    assert result["archived"] == 1 and not source.exists()
    assert store.read_json(target)["models"]["pv"] == "new"
    archives = list((data / "legacy_imports").rglob("*.gz"))
    assert len(archives) == 1 and gzip.decompress(archives[0].read_bytes()) == old


def test_missing_primary_is_restored_before_shared_copy_is_removed(tmp_path):
    data, share = tmp_path / "data", tmp_path / "share"
    source = share / "metrics.jsonl"
    source.parent.mkdir()
    source.write_text('{"issue_utc":"2026-11-25T00:00:00Z","date":"2026-11-25","load":{"mae_kwh":1.25,"coverage_pct":80,"hours":24},"pv":null}\n')
    result = migration.migrate_shared(data, share)
    assert result["restored"] == 1 and not source.exists()
    assert json.loads(store.read_text(data / "metrics.jsonl"))["load"]["mae_kwh"] == 1.25


def test_incomplete_legacy_report_does_not_become_the_live_plan(tmp_path):
    data, share = tmp_path / "data", tmp_path / "share"
    source = share / "shadow_reports/2026-11-25.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"note":"incomplete old report"}')
    migration.migrate_shared(data, share)
    assert not store.exists(data / "shadow_reports" / source.name)
    archives = list((data / "legacy_imports").rglob("*.gz"))
    assert len(archives) == 1
    assert json.loads(gzip.decompress(archives[0].read_bytes()))["note"] == "incomplete old report"


@pytest.mark.parametrize(("name", "value"), [
    ("retrain_state.json", {"last_retrain_utc": "not a date", "models": {}}),
    ("retrain_state.json", {"last_retrain_utc": None, "models": {"pv": "bad shape"}}),
    ("metrics.jsonl", {"date": "2026-11-25", "load": {}}),
    ("seed/outages.jsonl", {"start_utc": "invalid"}),
    ("seed/weather_archive/20261125T000000Z.json", {"hourly": {"time": ["invalid"]}}),
])
def test_invalid_metadata_is_archived_and_never_activated(tmp_path, name, value):
    data, share = tmp_path / "data", tmp_path / "share"
    source = share / name
    source.parent.mkdir(parents=True)
    raw = json.dumps(value).encode()
    source.write_bytes(raw)
    migration.migrate_shared(data, share)
    target = data / (name.removeprefix("seed/"))
    assert not store.exists(target)
    assert not source.exists()
    archives = list((data / "legacy_imports").rglob("*.gz"))
    assert len(archives) == 1 and gzip.decompress(archives[0].read_bytes()) == raw


def test_failed_publish_keeps_the_source_for_retry(tmp_path, monkeypatch):
    data, share = tmp_path / "data", tmp_path / "share"
    source = share / "metrics.jsonl"
    source.parent.mkdir()
    source.write_text('{"date":"2026-11-25"}\n')
    real_replace = store.os.replace
    monkeypatch.setattr(store.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("full disk")))
    assert migration.migrate_shared(data, share)["kept"] == 1
    assert source.exists()
    monkeypatch.setattr(store.os, "replace", real_replace)
    assert migration.migrate_shared(data, share)["removed"] == 1
    assert not source.exists()


def test_seed_hours_are_deduplicated_by_utc_and_missing_hours_are_imported(tmp_path):
    data, share = tmp_path / "data", tmp_path / "share"
    store.append_jsonl(data / "hourly_audit.jsonl", [
        {"start_utc": "2026-11-25T01:00:00Z", "load_kw": 1.25, "pv_filter": "keep"}])
    source = share / "seed/hourly_audit.jsonl"
    store.append_jsonl(source, [
        {"start_utc": "2026-11-25T01:00:00+00:00", "load_kw": 1.25},
        {"start_utc": "2026-11-25T00:00:00Z", "load_kw": 2.123456789012345, "pv_filter": "possible_curtailment"}])
    result = migration.migrate_shared(data, share)
    rows = [json.loads(line) for line in store.read_text(data / "hourly_audit.jsonl").splitlines()]
    assert len(rows) == 2 and result["imported_hours"] == 1
    assert rows[1]["load_kw"] == 2.123456789012345
    assert rows[1]["pv_filter"] == "possible_curtailment"
    assert not store.exists(source)
    assert migration.migrate_shared(data, share)["imported_hours"] == 0


def test_conflicting_seed_values_are_preserved_separately(tmp_path):
    data, share = tmp_path / "data", tmp_path / "share"
    store.append_jsonl(data / "hourly_audit.jsonl", [{"start_utc": "2026-11-25T00:00:00Z", "load_kw": 2.0}])
    source = share / "seed/hourly_audit.jsonl"
    store.append_jsonl(source, [{"start_utc": "2026-11-25T00:00:00Z", "load_kw": 1.0}])
    migration.migrate_shared(data, share)
    assert json.loads(store.read_text(data / "hourly_audit.jsonl"))["load_kw"] == 2.0
    assert not store.exists(source)
    archived = list((data / "legacy_imports").rglob("*.gz"))
    assert len(archived) == 1
    assert json.loads(gzip.decompress(archived[0].read_bytes()))["load_kw"] == 1.0


def test_unknown_files_and_external_symlinks_are_untouched(tmp_path):
    data, share = tmp_path / "data", tmp_path / "share"
    share.mkdir()
    outside = tmp_path / "external.jsonl"
    outside.write_text('{"value":1}\n')
    (share / "metrics.jsonl").symlink_to(outside)
    manual = share / "notes.txt"
    manual.write_text("keep")
    migration.migrate_shared(data, share)
    assert manual.read_text() == "keep" and outside.exists()
    assert (share / "metrics.jsonl").is_symlink()


def test_seed_sqlite_is_removed_only_if_all_rows_are_preserved(tmp_path):
    data, share = tmp_path / "data", tmp_path / "share"
    target, source = data / "outages.sqlite", share / "seed/outages.sqlite"
    for path in (target, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE outage_events (site TEXT, start TEXT, end TEXT)")
            db.execute("INSERT INTO outage_events VALUES ('home','start','end')")
    with sqlite3.connect(target) as db:
        db.execute("INSERT INTO outage_events VALUES ('home','new','event')")
    migration.migrate_shared(data, share)
    assert not source.exists()
    with sqlite3.connect(target) as db:
        assert db.execute("SELECT count(*) FROM outage_events").fetchone()[0] == 2


def test_retrain_state_has_one_authoritative_copy(tmp_path, monkeypatch):
    sys.path.insert(0, str(APP.parent / "src"))
    import retrain
    monkeypatch.setattr(retrain, "DATA", tmp_path)
    monkeypatch.setattr(retrain, "STATE", tmp_path / "retrain_state.json")
    monkeypatch.setattr(retrain, "SHARE", tmp_path / "share")
    state = {"last_retrain_utc": "2026-11-25T00:00:00Z", "models": {}}
    retrain._save_state(state)
    assert retrain._load_state() == state
    assert not (tmp_path / "share").exists()


def test_scoring_metrics_does_not_recreate_a_share_copy(tmp_path, monkeypatch):
    sys.path.insert(0, str(APP.parent / "src"))
    from zoneinfo import ZoneInfo

    import pandas as pd
    import pipeline
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setattr(pipeline, "METRICS", tmp_path / "metrics.jsonl")
    monkeypatch.setattr(pipeline, "SHARE", tmp_path / "share")
    starts = pd.date_range("2026-11-25", periods=24, freq="h", tz="UTC")
    hourly = {"starts_utc": [stamp.isoformat() for stamp in starts]}
    hourly.update({f"{target}_{q}_kw": [1.0] * 24 for target in ("load", "pv") for q in ("p10", "p50", "p90")})
    store.write_json(tmp_path / "shadow_reports/2026-11-25.json", {"issue_utc": starts[0].isoformat(), "hourly": hourly})
    audit = pd.DataFrame({"load_kw": 1.0, "load_filter": "keep", "pv_measured_kw": 1.0,
                          "pv_filter": "keep", "solar_elevation_deg": 30.0}, index=starts)
    assert pipeline.update_metrics(audit, ZoneInfo("UTC")) == 1
    assert store.exists(tmp_path / "metrics.jsonl")
    assert not (tmp_path / "share").exists()
