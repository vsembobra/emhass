"""The audit file mixes "…Z" and "+00:00" spellings across writers.

update_audit appends build_audit_rows output ("2026-11-20T18:00:00.000Z"),
_prepend_rows rewrites whole files with Timestamp.isoformat()
("2026-11-06T19:00:00+00:00"). Readers must parse both in one file —
strict single-format inference raises and kills retraining.
"""

import sys
from pathlib import Path

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402
import retrain  # noqa: E402

MIXED = (
    '{"start_utc": "2026-11-19T22:00:00+00:00", "load_kw": 1.0}\n'
    '{"start_utc": "2026-11-19T23:00:00.000Z", "load_kw": 1.1}\n'
    '{"start_utc": "2026-11-20T00:00:00.000Z", "load_kw": 1.2}\n'
)


def _write(tmp_path, monkeypatch):
    (tmp_path / "hourly_audit.jsonl").write_text(MIXED, encoding="utf-8")
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setattr(retrain, "DATA", tmp_path)


def test_pipeline_reads_mixed_spelling(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch)
    audit = pipeline._read_audit()
    assert len(audit) == 3
    assert str(audit.index[0].tz) == "UTC"


def test_retrain_reads_mixed_spelling(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch)
    audit = retrain._audit_frame()
    assert audit["load_kw"].notna().sum() == 3


def test_prepend_writes_z_spelling(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch)
    rows = [
        {
            "start_utc": "2026-11-19T21:00:00.000Z",
            "load_kw": 0.9,
            "critical_load_kw": 0.9,
            "grid_quality": "recorded_up",
            "load_filter": "keep",
            "pv_filter": "outside_daylight",
            "solar_elevation_deg": -10.0,
            "possible_pv_curtailment": False,
            "split": "live",
        }
    ]
    audit = pipeline._read_audit()
    pipeline._prepend_rows(audit, rows)
    text = pipeline.data_store.read_text(tmp_path / "hourly_audit.jsonl")
    assert "+00:00" not in text
    assert text.count(".000Z") == 4
