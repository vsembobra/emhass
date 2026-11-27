"""A display-name change must remain an update of the existing HA installation."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ADDON = ROOT / "addon_emhass_shadow"
sys.path[:0] = [str(ADDON / "app"), str(ADDON / "src")]
import pipeline  # noqa: E402


def test_solar_reserve_keeps_the_installation_identity():
    config = yaml.safe_load((ADDON / "config.yaml").read_text())
    repository = yaml.safe_load((ROOT / "repository.yaml").read_text())
    assert config["name"].startswith("Solar Reserve")
    assert config["panel_title"] == "Solar Reserve"
    assert config["slug"] == "emhass_shadow"
    assert config["ingress_port"] == 8099
    assert len(config["schema"]) == 57
    assert repository["url"] == "https://github.com/vsembobra/emhass"


def test_display_names_change_but_forecast_sensor_ids_stay_stable():
    sensors = pipeline._forecast_sensors({})
    assert {row[0] for row in sensors} == {
        "sensor.emhass_shadow_pv_forecast", "sensor.emhass_shadow_pv_forecast_p10",
        "sensor.emhass_shadow_load_forecast", "sensor.emhass_shadow_pv_forecast_day_ahead",
        "sensor.emhass_shadow_load_forecast_day_ahead", "sensor.emhass_shadow_weather_irradiance",
        "sensor.emhass_shadow_weather_cloud_cover", "sensor.emhass_shadow_weather_temperature",
    }
    assert all(row[2]["friendly_name"].startswith("Solar Reserve:") for row in sensors)
