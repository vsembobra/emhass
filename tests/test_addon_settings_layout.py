"""Every HA add-on option must have exactly one editable field on the panel."""

import sys
from collections import Counter
from pathlib import Path

import yaml

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import options_meta  # noqa: E402
import pipeline  # noqa: E402


def test_all_schema_settings_are_editable_on_an_unconfigured_panel(monkeypatch):
    config = yaml.safe_load((ADDON / "config.yaml").read_text())
    monkeypatch.setattr(pipeline, "options", lambda: dict(options_meta.DEFAULTS))
    monkeypatch.setattr(pipeline, "option_sources", lambda: {})
    monkeypatch.setattr(pipeline, "_STATES", None)
    monkeypatch.setattr(pipeline, "ha_get", lambda *a, **kw: [])
    groups = options_meta.options_view(options_meta.DEFAULTS)
    keys = [i["key"] for g in groups for i in g["items"]]
    keys += [c["key"] for c in pipeline.validate_entities() if c["key"]]
    assert set(keys) == set(config["schema"])
    assert all(count == 1 for count in Counter(keys).values())


def test_weather_is_first_and_related_settings_are_grouped():
    groups = options_meta.options_view(options_meta.DEFAULTS)
    assert groups[0]["title"] == "Погода и прогнозы"
    assert groups[0]["items"][0]["key"] == "open_meteo_model"
    by_key = {item["key"]: group["title"] for group in groups for item in group["items"]}
    assert by_key["capacity_kwh"] == by_key["calibration_charge_enabled"] == "Батарея"
    assert by_key["apply_mode"] == by_key["tou_soc_entity_template"] == "Управление инвертором"
    assert by_key["history_backfill_days"] == by_key["retrain_enabled"] == "Обучение и история"
    assert by_key["deferrable_loads"] == "Переносимые нагрузки"


def test_entity_checks_have_translated_logical_groups(monkeypatch):
    monkeypatch.setattr(pipeline, "options", lambda: dict(options_meta.DEFAULTS))
    monkeypatch.setattr(pipeline, "option_sources", lambda: {})
    monkeypatch.setattr(pipeline, "_STATES", None)
    monkeypatch.setattr(pipeline, "ha_get", lambda *a, **kw: [])
    checks = {row["key"]: row for row in pipeline.validate_entities() if row["key"]}
    assert checks["entity_grid_voltage"]["group"] == checks["entity_grid_power"]["group"]
    assert checks["entity_battery_soh"]["group"] == checks["entity_battery_soc"]["group"]
    for row in checks.values():
        assert row["group"] and row["group_en"]
