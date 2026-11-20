"""End-to-end pipeline.run() against a fake Home Assistant (no network)."""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402

ZONE_TIMES = ["00:00:00", "06:00:00", "10:00:00", "14:00:00", "18:00:00", "22:00:00"]


def default_options() -> dict:
    config = yaml.safe_load((ADDON / "config.yaml").read_text(encoding="utf-8"))
    return dict(config["options"])


@pytest.fixture
def fake_site(monkeypatch, tmp_path):
    """Run the pipeline in tmp_path with canned HA states and weather."""
    opts = default_options()
    # An empty fake history never saw a full charge, so the BMS calibration
    # would fire on every run; its own tests switch it on explicitly.
    opts["calibration_charge_enabled"] = False
    services = []

    def setup(**overrides):
        opts.update(overrides)
        (tmp_path / "options.json").write_text(json.dumps(opts), encoding="utf-8")
        return opts

    states = {"sensor.inverter_battery": "60", "binary_sensor.inverter_grid": "on"}
    for n, at in enumerate(ZONE_TIMES, start=1):
        states[f"number.inverter_program_{n}_soc"] = "30"
        states[f"time.inverter_program_{n}_time"] = at
    states[opts["floor_input_entity"]] = "20"

    attributes: dict[str, dict] = {}

    def ha_get(path, timeout=180):
        entity = path.rsplit("/", 1)[-1]
        if entity in states:
            return {
                "entity_id": entity,
                "state": states[entity],
                "attributes": attributes.get(entity, {}),
            }
        return {"entity_id": entity, "state": "1000"}

    def ha_service(domain, service, payload):
        services.append((domain, service, payload))
        if service == "set_value":
            states[payload["entity_id"]] = str(payload["value"])
        elif service in ("turn_on", "turn_off"):
            states[payload["entity_id"]] = service.removeprefix("turn_")
        elif service == "select_option":
            states[payload["entity_id"]] = payload["option"]

    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(now - timedelta(hours=48), periods=96, freq="h")
    elevation = np.clip(np.sin((index.hour - 6) / 12 * np.pi), 0, None)
    weather = pd.DataFrame(
        {
            "temperature_2m": 8.0,
            "shortwave_radiation": 700 * elevation,
            "direct_normal_irradiance": 500 * elevation,
            "diffuse_radiation": 120 * elevation,
            "cloud_cover": 20.0,
        },
        index=index,
    )

    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setattr(pipeline, "SHARE", tmp_path / "share")
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(pipeline, "ha_get", ha_get)
    monkeypatch.setattr(pipeline, "ha_service", ha_service)
    monkeypatch.setattr(pipeline, "fetch_history", lambda start, end, entities: {})
    monkeypatch.setattr(pipeline, "archive_weather", lambda site: (weather, "fake.json", False))
    setup()
    return {
        "setup": setup,
        "services": services,
        "states": states,
        "attributes": attributes,
        "data": tmp_path,
    }


def test_run_produces_report_and_logs(fake_site):
    report = pipeline.run()
    data = fake_site["data"]
    assert report["advisory"]["optimizer"] is not None
    assert len(report["hourly"]["load_p50_kw"]) == 24
    assert (data / "forecast_log.jsonl").exists()
    assert (data / "drift_state.json").exists()
    assert "self_learning" in report["data"]
    assert report["notification"].startswith("Теневой план")
    # Shadow mode writes nothing but the notification (plus self-clearing
    # dismissals of the error notifications, which are no-ops here).
    assert {s[:2] for s in fake_site["services"]} == {
        ("persistent_notification", "create"),
        ("persistent_notification", "dismiss"),
    }
    assert not [s for s in fake_site["services"] if s[1] == "create" and "problem" in s[2]["notification_id"]]


def test_apply_mode_writes_zone_targets(fake_site):
    fake_site["setup"](apply_mode=True)
    report = pipeline.run()
    writes = [s for s in fake_site["services"] if s[1] == "set_value"]
    assert writes, report["notification"]
    assert not report["advisory"]["apply_errors"]


LEARNED = {
    "identified_utc": "2026-11-20T00:00:00+00:00",
    "battery": {
        "status": "ok",
        "capacity_kwh": 28.0,
        "round_trip_efficiency": 0.9,
        "eta_one_way": 0.95,
    },
    "limits": {"max_charge_kw": 4.0, "max_discharge_kw": 4.0},
    "pv_geometry": None,
    "critical_profile_kw": [0.3] * 18 + [1.5] * 6,
}


def test_autotune_applies_learned_hardware(fake_site):
    (fake_site["data"] / "hardware.json").write_text(json.dumps(LEARNED), encoding="utf-8")
    fake_site["setup"](autotune=True)
    report = pipeline.run()
    assert report["data"]["battery_soh_source"] == "identified"
    assert any("ёмкость" in note for note in report["data"]["hardware"]["applied"])
    assert "Железо по данным" in report["notification"]


def test_learned_hardware_is_advice_only_without_autotune(fake_site):
    (fake_site["data"] / "hardware.json").write_text(json.dumps(LEARNED), encoding="utf-8")
    report = pipeline.run()
    assert report["data"]["battery_soh_source"] == "default"
    assert report["data"]["hardware"]["applied"] == []
    assert "только совет" in report["notification"]


def test_priority_flexible_loads_and_sensors(fake_site, monkeypatch):
    posted = {}
    monkeypatch.setattr(
        pipeline, "ha_post_state", lambda entity, state, attrs: posted.update({entity: state})
    )
    fake_site["setup"](
        priority_mode="resilience",
        publish_sensors=True,
        deferrable_loads=[{"name": "Boiler", "power_kw": 2.0, "hours": 2}],
    )
    report = pipeline.run()
    assert report["data"]["priority_mode"] == "resilience"
    assert report["data"]["reserve_margin_hours"] == 0.0  # quiet: no margin
    (item,) = report["advisory"]["deferrable"]
    assert item["start_local"] is not None
    assert "Boiler" in report["notification"]
    assert "Автономия сейчас" in report["notification"]
    assert set(posted) == {
        "sensor.emhass_shadow_floor_soc",
        "sensor.emhass_shadow_autonomy_hours",
        "sensor.emhass_shadow_boiler_start",
    }
    assert report["data"]["published_sensors"] == sorted(posted, key=list(posted).index)


def test_apply_mode_drives_grid_charge_flags(fake_site):
    states = fake_site["states"]
    for n in range(1, 7):
        states[f"switch.inverter_program_{n}_charging"] = "on"
    fake_site["setup"](
        apply_mode=True,
        tariff_enabled=True,
        tariff_cheap_windows="23:00-07:00",
        tou_grid_charge_entity_template="switch.inverter_program_{n}_charging",
    )
    report = pipeline.run()
    plan = {item["n"]: item for item in report["advisory"]["grid_charge"]}
    assert plan[1]["desired"] and plan[1]["reason"] == "дешёвый тариф"  # 00:00-06:00
    off = {s[2]["entity_id"] for s in fake_site["services"] if s[:2] == ("switch", "turn_off")}
    # 10:00 and 14:00 zones: expensive hours, the small cold-start reserve is met.
    assert {"switch.inverter_program_3_charging", "switch.inverter_program_4_charging"} <= off
    assert "switch.inverter_program_1_charging" not in off
    assert any(
        line.startswith("заряд из сети зона 3: off") for line in report["advisory"]["applied"]
    )


def test_shadow_mode_reports_grid_charge_advice_without_writing(fake_site):
    fake_site["setup"](tou_grid_charge_entity_template="switch.inverter_program_{n}_charging")
    report = pipeline.run()
    assert len(report["advisory"]["grid_charge"]) == 6
    assert not [s for s in fake_site["services"] if s[0] == "switch"]


def test_ignored_writes_are_not_reported_as_applied(fake_site, monkeypatch):
    """Review case: the device keeps every value; nothing may count as applied."""
    for n in range(1, 7):
        fake_site["states"][f"switch.inverter_program_{n}_charging"] = "on"
    fake_site["setup"](
        apply_mode=True,
        tariff_enabled=True,
        tariff_cheap_windows="23:00-07:00",
        tou_grid_charge_entity_template="switch.inverter_program_{n}_charging",
    )
    monkeypatch.setattr(pipeline, "ha_service", lambda *a: None)  # accepted, not applied
    report = pipeline.run()
    assert report["advisory"]["applied"] == []
    errors = report["advisory"]["apply_errors"]
    assert errors and all("не подтверждена" in e for e in errors)
    assert any(e.startswith("заряд из сети") for e in errors)


def test_readback_tolerance_is_below_half_a_step(fake_site):
    states = fake_site["states"]
    states["number.x"] = "30"
    assert not pipeline.readback_ok("number.x", 35.0, 5)
    states["number.x"] = "34"
    assert pipeline.readback_ok("number.x", 35.0, 5)
    states["switch.y"] = "on"
    assert pipeline.readback_ok("switch.y", "on", 5)
    assert not pipeline.readback_ok("switch.y", "off", 5)


def _write_outages(data, ends_days_ago, hours=3.0):
    now = datetime.now(UTC)
    lines = []
    for days in ends_days_ago:
        end = now - timedelta(days=days)
        start = end - timedelta(hours=hours)
        lines.append(
            json.dumps(
                {
                    "start_utc": start.isoformat(),
                    "end_record_utc": end.isoformat(),
                    "interrupted": False,
                }
            )
        )
    (data / "outages.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_quiet_period_holds_no_outage_reserve(fake_site):
    _write_outages(fake_site["data"], [30, 25, 21])  # a series that ended weeks ago
    report = pipeline.run()
    assert not report["reserve"]["regime"]["active"]
    assert report["reserve"]["reserve_kwh"] == 0.0
    assert report["advisory"]["recommended_floor_soc"] == pytest.approx(0.10)  # min SOC
    assert report["outage_risk"] is None  # old outages leave no risk hours behind
    assert "резерв под отключения не держим" in report["notification"]


def test_live_series_holds_a_reserve_with_margin(fake_site):
    _write_outages(fake_site["data"], [9, 6, 3, 1], hours=4.0)
    fake_site["setup"](priority_mode="resilience")
    report = pipeline.run()
    regime = report["reserve"]["regime"]
    assert regime["active"] and regime["events_in_window"] == 4
    assert report["reserve"]["duration_hours"] == pytest.approx(4.0, abs=0.01)
    assert report["reserve"]["margin_hours"] == 1.0  # resilience default
    assert report["advisory"]["recommended_floor_soc"] > 0.10
    assert "Серия отключений идёт" in report["notification"]


def test_floor_is_never_the_overnight_deficit(fake_site):
    # Winter-like: tiny PV, a whole night ahead; no series -> floor = min SOC.
    report = pipeline.run()
    assert max(step["floor_soc"] for step in report["advisory"]["floor_profile"]) == pytest.approx(
        0.10
    )
    assert not report["advisory"]["optimizer"]["floor_relaxed"]
    assert "не помещается" not in report["notification"]


DEYE_OPTIONS = ["Disabled", "Grid", "Gen", "Grid & Gen"]


def test_apply_mode_drives_deye_select_flags(fake_site):
    for n in range(1, 7):
        entity = f"select.inverter_program_{n}_charging"
        fake_site["states"][entity] = "Grid"
        fake_site["attributes"][entity] = {"options": DEYE_OPTIONS}
    fake_site["states"]["select.inverter_program_6_charging"] = "Grid & Gen"
    fake_site["setup"](
        apply_mode=True,
        tariff_enabled=True,
        tariff_cheap_windows="23:00-07:00",
        tou_grid_charge_entity_template="select.inverter_program_{n}_charging",
    )
    report = pipeline.run()
    plan = {item["n"]: item for item in report["advisory"]["grid_charge"]}
    assert plan[1]["target_state"] == "Grid"  # 00:00-06:00, cheap
    assert plan[3]["target_state"] == "Disabled"  # 10:00, expensive
    assert plan[6]["target_state"] == "Gen"  # 22:00 keeps its generator charging
    selects = [s for s in fake_site["services"] if s[:2] == ("select", "select_option")]
    assert {s[2]["option"] for s in selects} <= {"Disabled", "Gen"}
    assert not report["advisory"]["apply_errors"]
    assert fake_site["states"]["select.inverter_program_3_charging"] == "Disabled"


def test_select_without_a_matching_option_is_reported(fake_site):
    for n in range(1, 7):
        entity = f"select.inverter_program_{n}_charging"
        fake_site["states"][entity] = "Mode A"
        fake_site["attributes"][entity] = {"options": ["Mode A", "Mode B"]}
    fake_site["setup"](
        apply_mode=True, tou_grid_charge_entity_template="select.inverter_program_{n}_charging"
    )
    report = pipeline.run()
    assert any("нет подходящего варианта" in e for e in report["advisory"]["apply_errors"])


def test_routine_replans_stay_quiet_unless_they_write(fake_site):
    report = pipeline.run(notify="changes")  # shadow mode: nothing written
    assert report["advisory"]["applied"] == []
    # No plan summary and no error notifications raised on a healthy quiet run;
    # only self-clearing dismissals are allowed.
    assert not [s for s in fake_site["services"] if s[:2] == ("persistent_notification", "create")]
    fake_site["setup"](apply_mode=True)
    pipeline.run(notify="changes")
    assert [s for s in fake_site["services"] if s[:2] == ("persistent_notification", "create")]


def test_stale_weather_raises_and_clears_error_notification(fake_site, monkeypatch):
    def no_weather(site):
        raise OSError("no route to open-meteo")

    monkeypatch.setattr(pipeline, "archive_weather", no_weather)
    pipeline.run(notify="changes")  # quiet replan still reports the problem
    creates = [s for s in fake_site["services"] if s[:2] == ("persistent_notification", "create")]
    weather = [s for s in creates if s[2]["notification_id"] == "emhass_shadow_problem_weather"]
    assert weather and "Open-Meteo" in weather[0][2]["message"]

    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(now - timedelta(hours=48), periods=96, freq="h")
    elevation = np.clip(np.sin((index.hour - 6) / 12 * np.pi), 0, None)
    fresh = pd.DataFrame(
        {
            "temperature_2m": 8.0,
            "shortwave_radiation": 700 * elevation,
            "direct_normal_irradiance": 500 * elevation,
            "diffuse_radiation": 120 * elevation,
            "cloud_cover": 20.0,
        },
        index=index,
    )
    monkeypatch.setattr(pipeline, "archive_weather", lambda site: (fresh, "fresh.json", False))
    fake_site["services"].clear()
    pipeline.run(notify="changes")
    dismissals = [s for s in fake_site["services"] if s[:2] == ("persistent_notification", "dismiss")]
    assert any(s[2]["notification_id"] == "emhass_shadow_problem_weather" for s in dismissals)
    assert not [
        s for s in fake_site["services"]
        if s[:2] == ("persistent_notification", "create") and "problem" in s[2]["notification_id"]
    ]


def test_bms_calibration_charges_the_night_zone_to_full(fake_site):
    for n in range(1, 7):
        entity = f"select.inverter_program_{n}_charging"
        fake_site["states"][entity] = "Disabled"
        fake_site["attributes"][entity] = {"options": ["Disabled", "Grid", "Generator", "Both"]}
    fake_site["setup"](
        apply_mode=True,
        calibration_charge_enabled=True,
        tou_grid_charge_entity_template="select.inverter_program_{n}_charging",
    )
    report = pipeline.run()
    calib = report["data"]["calibration"]
    assert calib["due"] and calib["days_since_full"] is None  # no full charge in history
    assert 1 in calib["zones"]  # 00:00-06:00 covers the default night window
    zone1 = next(t for t in report["advisory"]["optimizer"]["zone_targets"] if t["n"] == 1)
    assert zone1["optimal_soc_pct"] == 100
    plan = {item["n"]: item for item in report["advisory"]["grid_charge"]}
    assert plan[1]["desired"] and plan[1]["reason"].startswith("калибровка BMS")
    assert fake_site["states"]["select.inverter_program_1_charging"] == "Grid"
    assert fake_site["states"]["number.inverter_program_1_soc"] == "100"
    assert "Калибровка BMS" in report["notification"]


def test_a_full_battery_resets_the_calibration(fake_site):
    fake_site["states"]["sensor.inverter_battery"] = "100"
    fake_site["setup"](calibration_charge_enabled=True)
    report = pipeline.run()
    calib = report["data"]["calibration"]
    assert not calib["due"] and calib["days_since_full"] == 0.0
    assert "Калибровка BMS" not in report["notification"]


def test_calibration_switched_off(fake_site):
    report = pipeline.run()
    assert not report["data"]["calibration"]["due"]
