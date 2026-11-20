"""Manual numeric constants where an entity id is expected.

An option like the outdoor temperature accepts "20" as a fixed value
instead of a sensor. Such constants must never reach the HA history
query (a non-entity filter_entity_id answers 400 Bad Request) and must
surface as the value itself in audit rows and entity checks.
"""

import sys
from pathlib import Path

import pandas as pd

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402


def _ent(temp_option):
    return {
        "mean_sensors": {
            "load_kw": ("sensor.load", 0.001),
            "temp_air_observed_c": (temp_option, 1.0),
        },
        "soc": "sensor.soc",
        "export_counter": "sensor.export",
        "grid_binary": "binary_sensor.grid",
        "voltages": {},
    }


def test_plain_number_detection():
    assert pipeline._plain_number("20") == 20.0
    assert pipeline._plain_number(" -5.5 ") == -5.5
    assert pipeline._plain_number("sensor.temp") is None
    assert pipeline._plain_number("") is None
    assert pipeline._plain_number(None) is None
    assert pipeline._plain_number("nan") is None
    assert pipeline._plain_number("inf") is None


def test_audit_entities_skip_manual_constants():
    ids = pipeline._audit_entities(_ent("20"))
    assert "20" not in ids
    assert ids == ["sensor.load", "sensor.soc", "sensor.export", "binary_sensor.grid"]


def test_audit_rows_use_manual_constant():
    hours = list(pd.date_range("2026-06-01 10:00", periods=2, freq="h", tz="UTC"))
    rows = pipeline.build_audit_rows(
        {}, [h.to_pydatetime() for h in hours], 45.0, 35.0, _ent("21.5")
    )
    assert [r["temp_air_observed_c"] for r in rows] == [21.5, 21.5]
    assert all(r["load_kw"] is None for r in rows)
