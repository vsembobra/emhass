"""History backfill: recorder data flows into the audit exactly once."""

import sys
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402

SITE = {"latitude": 45.0, "longitude": 34.0, "time_zone": "UTC"}
ENT = {
    "mean_sensors": {"load_kw": ("sensor.load", 0.001)},
    "voltages": {},
    "soc": "sensor.soc",
    "export_counter": "sensor.export",
    "grid_binary": "binary_sensor.grid",
}


@pytest.fixture
def fake_ha(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setattr(pipeline, "fetch_history", lambda start, end, entities: {})
    monkeypatch.setattr(
        pipeline,
        "build_audit_rows",
        lambda history, hours, lat, lon, ent, **kwargs: [
            {"start_utc": h.isoformat().replace("+00:00", ".000Z"), "load_kw": 1.0}
            for h in hours
        ],
    )
    return tmp_path


def test_backfill_fresh_install(fake_ha):
    added = pipeline.backfill_audit(SITE, ENT, days=2)
    assert added == 48
    lines = pipeline.data_store.read_text(fake_ha / "hourly_audit.jsonl").splitlines()
    assert len(lines) == 48


def test_backfill_is_idempotent(fake_ha):
    pipeline.backfill_audit(SITE, ENT, days=2)
    assert pipeline.backfill_audit(SITE, ENT, days=2) == 0


def test_backfill_extends_only_missing_hours(fake_ha):
    pipeline.backfill_audit(SITE, ENT, days=2)
    added = pipeline.backfill_audit(SITE, ENT, days=4)
    assert added == 48  # only the two older days are new
    audit = pipeline.data_store.read_frame(fake_ha / "hourly_audit.jsonl")
    assert len(audit) == 96
    assert not audit["start_utc"].duplicated().any()


def test_fresh_install_update_audit_uses_backfill_days(fake_ha):
    audit, appended = pipeline.update_audit(SITE, ENT, backfill_days=2)
    # now-48h .. now-1h, but the "last" stub itself is not an audit row
    assert appended == 47
    assert len(audit) == 47


def test_audit_entities_skip_empty_option_values():
    ent = {
        "mean_sensors": {
            "load_kw": ("sensor.load", 0.001),
            "temp_air_observed_c": ("", 1.0),  # cleared option must not reach the query
        },
        "voltages": {},
        "soc": "sensor.soc",
        "export_counter": "sensor.export",
        "grid_binary": "binary_sensor.grid",
    }
    assert pipeline._audit_entities(ent) == [
        "sensor.load",
        "sensor.soc",
        "sensor.export",
        "binary_sensor.grid",
    ]
