"""Options left empty are filled from ha-solarman entities and the HA config."""

import json
import sys
from pathlib import Path

import pytest
import yaml

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import discovery  # noqa: E402
import options_meta  # noqa: E402
import pipeline  # noqa: E402

ENTRY = "01JC4ZK8Q2M7T5YB3W9XRDHN6E"  # a config entry id (ULID, as HA makes them)
# Profile item (name, platform) -> entity id after the owner renamed the device "Deye".
ITEMS = {
    ("Load Power", "sensor"): "sensor.deye_load_power",
    ("Load UPS Power", "sensor"): "sensor.deye_load_ups_power",
    ("PV Power", "sensor"): "sensor.deye_pv_power",
    ("Battery Power", "sensor"): "sensor.deye_battery_power",
    ("Grid Power", "sensor"): "sensor.deye_grid_power",
    ("Battery", "sensor"): "sensor.deye_battery",
    ("Grid", "binary_sensor"): "binary_sensor.deye_grid",
    ("Grid L1 Voltage", "sensor"): "sensor.deye_grid_l1_voltage",
    ("Total Energy Export", "sensor"): "sensor.deye_total_energy_export",
    ("Battery SOH", "sensor"): "sensor.deye_battery_soh",
    ("Battery Capacity", "number"): "number.deye_battery_capacity",
    ("Battery Voltage", "sensor"): "sensor.deye_battery_voltage",
    ("Battery Grid Charging Current", "number"): "number.deye_battery_grid_charging_current",
    ("Battery Max Discharging Current", "number"): "number.deye_battery_max_discharging_current",
    ("Battery Control Mode", "select"): "select.deye_battery_control_mode",
    ("Device", "sensor"): "sensor.deye_device",
    **{(f"Program {n} SOC", "number"): f"number.deye_program_{n}_soc" for n in range(1, 7)},
    **{(f"Program {n} Time", "time"): f"time.deye_program_{n}_time" for n in range(1, 7)},
    **{
        (f"Program {n} Charging", "select"): f"select.deye_program_{n}_charging"
        for n in range(1, 7)
    },
}
STATES = {
    "number.deye_battery_capacity": {"state": "620"},
    "sensor.deye_battery_voltage": {"state": "52.36"},
    "number.deye_battery_grid_charging_current": {"state": "120"},
    "number.deye_battery_max_discharging_current": {"state": "350"},
    "select.deye_battery_control_mode": {"state": "Lithium"},
    "sensor.deye_device": {
        "state": "LV 3-Phase Hybrid Inverter",
        "attributes": {"Rated Power": 20000},
    },
}
CORE = {"latitude": 50.45, "longitude": 30.52, "time_zone": "Europe/Kyiv"}


def _registry(items=ITEMS, entry=ENTRY, **extra):
    """Registry entries the way ha-solarman makes them: unique_id = slug(entry_id + item key)."""
    return [
        {
            "entity_id": entity,
            "platform": "solarman",
            "config_entry_id": entry,
            "unique_id": f"{entry}_{name}_{platform}".lower().replace(" ", "_"),
            "original_name": name,
            "disabled_by": None,
            **extra,
        }
        for (name, platform), entity in items.items()
    ]


def test_renamed_deye_is_found_by_its_profile_items():
    found = discovery.discover(_registry(), STATES, CORE)["found"]
    assert found["entity_battery_power"] == "sensor.deye_battery_power"
    assert found["entity_battery_soc"] == "sensor.deye_battery"
    assert found["entity_critical_load_power"] == "sensor.deye_load_ups_power"
    assert found["entity_grid_status"] == "binary_sensor.deye_grid"
    assert found["entity_battery_soh"] == "sensor.deye_battery_soh"
    assert found["tou_zones"] == 6
    assert found["tou_soc_entity_template"] == "number.deye_program_{n}_soc"
    assert found["tou_time_entity_template"] == "time.deye_program_{n}_time"
    assert found["tou_grid_charge_entity_template"] == "select.deye_program_{n}_charging"


def test_inverter_settings_give_capacity_and_limits():
    found = discovery.discover(_registry(), STATES, None)["found"]
    assert found["capacity_kwh"] == pytest.approx(31.7)  # 620 Ah x 51.2 V
    assert found["max_charge_kw"] == pytest.approx(6.1)  # grid charging 120 A x 51.2 V
    assert found["max_discharge_kw"] == pytest.approx(17.9)  # 350 A x 51.2 V, under 20 kW
    small = {**STATES, "sensor.deye_device": {"state": "x", "attributes": {"Rated Power": 8000}}}
    found = discovery.discover(_registry(), small, None)["found"]
    assert found["max_discharge_kw"] == pytest.approx(8.0)  # the inverter's rating caps it


def test_high_voltage_pack_uses_its_own_voltage():
    states = {
        **STATES,
        "sensor.deye_battery_voltage": {"state": "409.6"},
        "number.deye_battery_capacity": {"state": "50"},
    }
    assert discovery.discover(_registry(), states, None)["found"]["capacity_kwh"] == pytest.approx(
        20.5
    )


def test_location_comes_from_the_home_assistant_config():
    result = discovery.discover([], {}, CORE)
    assert result["found"] == CORE
    assert result["source"]["time_zone"] == discovery.CORE


def test_single_phase_hybrid_backup_load_is_its_load_port():
    items = {k: v for k, v in ITEMS.items() if k[0] != "Load UPS Power"}
    found = discovery.discover(_registry(items), STATES, None)["found"]
    assert found["entity_critical_load_power"] == "sensor.deye_load_power"


def test_the_device_with_the_battery_wins_and_disabled_entities_are_skipped():
    meter = _registry({("Grid Power", "sensor"): "sensor.meter_grid_power"}, entry="01METER")
    disabled = _registry({("PV Power", "sensor"): "sensor.old_pv"}, entry=ENTRY, disabled_by="user")
    items = {k: v for k, v in ITEMS.items() if k[0] != "PV Power"}
    found = discovery.discover(meter + _registry(items) + disabled, STATES, None)["found"]
    assert found["entity_grid_power"] == "sensor.deye_grid_power"
    assert "entity_pv_power" not in found


def test_older_unique_ids_fall_back_to_the_item_name():
    registry = _registry()
    for entry in registry:
        entry["unique_id"] = "serial123_" + entry["entity_id"]  # some other format
    found = discovery.discover(registry, STATES, None)["found"]
    assert found["entity_battery_power"] == "sensor.deye_battery_power"


def test_other_integrations_find_nothing():
    registry = [dict(e, platform="sunsynk") for e in _registry()]
    assert discovery.discover(registry, STATES, None)["found"] == {}


def test_owner_value_wins_empty_follows_ha_none_switches_off():
    found = discovery.discover(_registry(), STATES, CORE)
    stored = {
        "capacity_kwh": 30.0,  # typed by the owner
        "entity_pv_power": "",  # left empty
        "entity_battery_soh": "none",  # switched off
        "min_soc_pct": 25.0,
    }
    opts = discovery.effective(stored, found)
    assert opts["capacity_kwh"] == 30.0
    assert opts["entity_pv_power"] == "sensor.deye_pv_power"
    assert opts["entity_battery_soh"] == ""
    assert opts["latitude"] == 50.45 and opts["min_soc_pct"] == 25.0
    assert opts["run_time"] == options_meta.DEFAULTS["run_time"]  # hidden: built-in value
    sources = discovery.sources(stored, found)
    assert sources["capacity_kwh"] == "manual"
    assert sources["entity_pv_power"] == discovery.SOLARMAN
    assert sources["latitude"] == discovery.CORE
    assert sources["run_time"] == "default"


def test_without_discovery_the_defaults_apply():
    opts = discovery.effective({"min_soc_pct": 25.0}, None)
    assert opts == {**options_meta.DEFAULTS, "min_soc_pct": 25.0}


def test_compact_keeps_the_owners_choices_only():
    found = discovery.discover(_registry(), STATES, CORE)
    stored = {
        **options_meta.DEFAULTS,  # an older version saved every option
        "min_soc_pct": 25.0,
        "max_soc_pct": 95.0,  # hidden but chosen: stays
        "entity_pv_power": "sensor.deye_pv_power",  # same as found: automatic now
        "latitude": 50.45,
    }
    kept, dropped = discovery.compact(stored, found)
    assert kept == {
        **{key: stored[key] for key in options_meta.VISIBLE},
        "max_soc_pct": 95.0,
        "tou_grid_charge_entity_template": "none",  # empty meant "leave the flags alone"
    }
    assert "entity_pv_power" in dropped and "latitude" in dropped and "run_time" in dropped
    after = discovery.effective(kept, found)
    # The old version read the empty template as "leave the flags alone": still so.
    assert after["tou_grid_charge_entity_template"] == ""
    assert after["max_soc_pct"] == 95.0 and after["min_soc_pct"] == 25.0


def _supervisor(monkeypatch, tmp_path, stored):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setenv("SUPERVISOR_TOKEN", "x")
    (tmp_path / "options.json").write_text(json.dumps(stored), encoding="utf-8")
    posted = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=0, **kw):
        if request.full_url.endswith("/addons/self/info"):
            return Response({"result": "ok", "data": {"options": dict(stored)}})
        body = json.loads(request.data)
        posted.append(body["options"])
        stored.clear()
        stored.update(body["options"])
        return Response({"result": "ok"})

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)
    return posted


def test_compaction_runs_once_and_shortens_the_settings(monkeypatch, tmp_path):
    stored = dict(options_meta.DEFAULTS, min_soc_pct=25.0)
    posted = _supervisor(monkeypatch, tmp_path, stored)
    discovery.save(tmp_path, discovery.discover(_registry(), STATES, CORE))
    dropped = pipeline.compact_options()
    assert "entity_battery_power" in dropped and len(posted) == 1
    assert set(pipeline.stored_options()) == {
        *options_meta.VISIBLE,
        "tou_grid_charge_entity_template",
    }
    assert pipeline.options()["entity_battery_power"] == "sensor.deye_battery_power"
    assert pipeline.compact_options() == [] and len(posted) == 1  # once


def test_clearing_an_option_in_the_panel_makes_it_automatic(monkeypatch, tmp_path):
    stored = {"min_soc_pct": 25.0, "capacity_kwh": 30.0}
    posted = _supervisor(monkeypatch, tmp_path, stored)
    discovery.save(tmp_path, discovery.discover(_registry(), STATES, CORE))
    result = pipeline.update_options({"capacity_kwh": ""})
    assert result["automatic"] == ["capacity_kwh"]
    assert posted[-1] == {"min_soc_pct": 25.0}
    assert pipeline.options()["capacity_kwh"] == pytest.approx(31.7)


def test_an_option_cleared_in_the_settings_leaves_the_file(monkeypatch, tmp_path):
    stored = {"min_soc_pct": 25.0}
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    (tmp_path / "options.json").write_text(json.dumps({"min_soc_pct": 25.0, "capacity_kwh": 30.0}))
    monkeypatch.setattr(pipeline, "supervisor_options", lambda: dict(stored))
    pipeline.sync_options()
    assert pipeline.stored_options() == {"min_soc_pct": 25.0}


def test_refresh_keeps_the_last_result_when_home_assistant_is_away(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    old = discovery.discover(_registry(), STATES, CORE)
    old["checked_utc"] = "2026-01-01T00:00:00+00:00"  # stale
    discovery.save(tmp_path, old)

    def away(*args, **kwargs):
        raise OSError("Home Assistant restarting")

    monkeypatch.setattr(pipeline, "ha_get", away)
    monkeypatch.setattr(pipeline, "_ws_request", away)
    assert pipeline.refresh_discovery()["found"] == old["found"]


def test_settings_list_only_the_owners_decisions():
    config = yaml.safe_load((ADDON / "config.yaml").read_text(encoding="utf-8"))
    assert list(config["options"]) == list(options_meta.VISIBLE)
    assert all(config["options"][k] == options_meta.DEFAULTS[k] for k in config["options"])
    assert set(config["schema"]) == set(options_meta.DEFAULTS)
    hidden = [k for k in config["schema"] if k not in options_meta.VISIBLE]
    assert all(str(config["schema"][k]).endswith("?") for k in hidden)
    for lang in ("ru", "en"):
        assert set(config["translations"][lang]["configuration"]) >= set(options_meta.DEFAULTS)
