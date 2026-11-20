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

    def ha_get(path, timeout=180):
        entity = path.rsplit("/", 1)[-1]
        if entity in states:
            return {"entity_id": entity, "state": states[entity]}
        return {"entity_id": entity, "state": "1000"}

    def ha_service(domain, service, payload):
        services.append((domain, service, payload))
        if service == "set_value":
            states[payload["entity_id"]] = str(payload["value"])

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
    return {"setup": setup, "services": services, "states": states, "data": tmp_path}


def test_run_produces_report_and_logs(fake_site):
    report = pipeline.run()
    data = fake_site["data"]
    assert report["advisory"]["optimizer"] is not None
    assert len(report["hourly"]["load_p50_kw"]) == 24
    assert (data / "forecast_log.jsonl").exists()
    assert (data / "drift_state.json").exists()
    assert "self_learning" in report["data"]
    assert report["notification"].startswith("Теневой план")
    # Shadow mode writes nothing but the notification.
    assert {s[:2] for s in fake_site["services"]} == {("persistent_notification", "create")}


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
