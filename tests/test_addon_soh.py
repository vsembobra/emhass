"""SoH option accepts an entity id OR a plain number; empty means 100 %."""

import sys
from pathlib import Path

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402


def test_empty_means_fresh_pack():
    assert pipeline._battery_soh({}) == (1.0, "default")
    assert pipeline._battery_soh({"entity_battery_soh": ""}) == (1.0, "default")
    assert pipeline._battery_soh({"entity_battery_soh": "   "}) == (1.0, "default")


def test_plain_number_is_used_directly():
    assert pipeline._battery_soh({"entity_battery_soh": "99"}) == (0.99, "manual")
    assert pipeline._battery_soh({"entity_battery_soh": "0.95"}) == (0.95, "manual")


def test_numbers_are_clamped():
    assert pipeline._battery_soh({"entity_battery_soh": "110"}) == (1.0, "manual")
    assert pipeline._battery_soh({"entity_battery_soh": "30"}) == (0.4, "manual")


def test_entity_id_is_read(monkeypatch):
    monkeypatch.setattr(
        pipeline, "ha_get", lambda path, timeout=180: {"state": "87"}
    )
    assert pipeline._battery_soh({"entity_battery_soh": "sensor.inverter_battery_soh"}) == (
        0.87,
        "entity",
    )


def test_entity_fraction_and_missing(monkeypatch):
    monkeypatch.setattr(pipeline, "ha_get", lambda path, timeout=180: {"state": "0.9"})
    assert pipeline._battery_soh({"entity_battery_soh": "sensor.soh"}) == (0.9, "entity")

    def boom(path, timeout=180):
        raise OSError("404")

    monkeypatch.setattr(pipeline, "ha_get", boom)
    assert pipeline._battery_soh({"entity_battery_soh": "sensor.missing"}) == (
        1.0,
        "unavailable",
    )
