"""Discovery is an explicit preview; it never chooses or saves an inverter silently."""

import importlib
import json
import sys
import threading
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import discovery  # noqa: E402
import main  # noqa: E402
import options_meta  # noqa: E402
import pipeline  # noqa: E402


def registry(device, entry="shared-entry"):
    return [
        {
            "platform": "solarman",
            "config_entry_id": entry,
            "device_id": device,
            "unique_id": f"{entry}_{key}",
            "original_name": name,
            "entity_id": f"{domain}.{device}_{suffix}",
        }
        for key, name, domain, suffix in (
            ("battery_sensor", "Battery", "sensor", "soc"),
            ("pv_power_sensor", "PV Power", "sensor", "pv"),
            ("load_power_sensor", "Load Power", "sensor", "load"),
        )
    ]


def test_new_install_never_fills_sensor_bindings_from_defaults_or_cache(tmp_path):
    (tmp_path / "options.json").write_text("{}")
    (tmp_path / "discovery.json").write_text(
        json.dumps({"found": {"entity_battery_soc": "sensor.other_battery", "capacity_kwh": 200}})
    )
    opts = discovery.load_options(tmp_path)
    assert opts["entity_battery_soc"] == ""
    assert opts["entity_load_power"] == ""
    assert opts["tou_soc_entity_template"] == ""
    assert opts["capacity_kwh"] == options_meta.DEFAULTS["capacity_kwh"]


def test_critical_load_must_be_configured_before_training():
    opts = dict.fromkeys(
        ("entity_load_power", "entity_pv_power", "entity_battery_soc", "entity_grid_status"),
        "sensor.configured",
    )
    assert "entity_critical_load_power" in discovery.missing_required(opts)


def test_saving_new_device_clears_old_controls_and_missing_bindings(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    stored = {
        "tou_soc_entity_template": "number.old_{n}",
        "tou_time_entity_template": "time.old_{n}",
        "tou_grid_charge_entity_template": "select.old_{n}",
        "floor_input_entity": "number.old_floor",
        "entity_grid_voltage": "sensor.old_voltage",
        "entity_battery_soc": "sensor.old_soc",
        "tariff_normal_price": 9.0,
    }
    (tmp_path / "options.json").write_text(json.dumps(stored))
    monkeypatch.setattr(pipeline, "supervisor_options", lambda: None)
    monkeypatch.setattr(pipeline, "_post_options", lambda opts: None)
    candidate = discovery.discover(registry("new"), {}, None)["candidates"][0]
    pipeline.apply_discovery(candidate)
    opts = pipeline.options()
    assert opts["entity_battery_soc"] == "sensor.new_soc"
    assert opts["tariff_normal_price"] == 9.0
    for key in (
        "tou_soc_entity_template",
        "tou_time_entity_template",
        "floor_input_entity",
        "tou_grid_charge_entity_template",
        "entity_grid_voltage",
    ):
        assert opts[key] == ""


def test_two_devices_in_same_integration_are_separate_candidates():
    result = discovery.discover(registry("one") + registry("two"), {}, None)
    assert "entity_battery_soc" not in result["found"]
    assert len(result["candidates"]) == 2
    for candidate in result["candidates"]:
        found = candidate["found"]
        device = candidate["device_id"]
        assert found["entity_battery_soc"] == f"sensor.{device}_soc"
        assert found["entity_pv_power"] == f"sensor.{device}_pv"


def test_invalid_registry_rows_do_not_abort_search():
    rows = [None, "invalid", {}, {"platform": "solarman"}] + registry("one")
    result = discovery.discover(rows, {}, None)
    assert result["found"]["entity_battery_soc"] == "sensor.one_soc"


def test_missing_control_templates_produce_no_control_entities():
    ent = pipeline.resolve_entities(options_meta.DEFAULTS)
    assert ent["tou_soc"] == []
    assert ent["tou_time"] == []


def test_search_endpoints_return_while_search_is_running(monkeypatch):
    module = importlib.import_module("discovery_search")
    release = threading.Event()

    def fetch():
        release.wait(2)
        return {"candidates": []}

    search = module.DiscoverySearch(fetch)
    monkeypatch.setattr(main, "SEARCH", search)
    handler = object.__new__(main.Handler)
    handler.path = "/api/discovery"
    responses = []
    handler._json = lambda code, data: responses.append((code, data))
    try:
        handler.do_POST()
        assert responses[-1] == (202, {"started": True})
        handler.do_GET()
        assert responses[-1][0] == 200 and responses[-1][1]["running"]
        handler.do_POST()
        assert responses[-1] == (409, {"started": False})
    finally:
        release.set()
        search.thread.join(2)


def test_empty_configuration_waits_without_bootstrap_or_auto_search(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    (tmp_path / "options.json").write_text("{}")
    monkeypatch.setattr(pipeline, "sync_options", lambda: {})

    def unexpected(*args, **kwargs):
        pytest.fail("unconfigured startup must not search, train, fetch history or plan")

    monkeypatch.setattr(pipeline, "bootstrap", unexpected)
    monkeypatch.setattr(pipeline, "run", unexpected)
    monkeypatch.setattr(main.retrain, "retrain_if_due", unexpected)
    monkeypatch.setattr(pipeline, "ha_get", unexpected)
    monkeypatch.setitem(main.STATE, "running", False)
    monkeypatch.setitem(main.STATE, "last_error", None)
    main.run_pipeline()
    assert not main.STATE["running"]
    assert main.STATE.get("configuration_missing")


def test_validation_fetches_states_once_even_when_every_binding_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    (tmp_path / "options.json").write_text(
        json.dumps(
            {
                "entity_load_power": "sensor.gone_load",
                "entity_battery_soc": "sensor.gone_soc",
                "entity_pv_power": "sensor.gone_pv",
                "entity_grid_status": "binary_sensor.gone_grid",
                "tou_soc_entity_template": "number.gone_{n}",
                "tou_time_entity_template": "time.gone_{n}",
            }
        )
    )
    monkeypatch.setattr(pipeline, "_STATES", None)
    calls = []

    def get(path, timeout=180):
        calls.append(path)
        assert path == "/api/states"
        assert timeout <= 10
        return []

    monkeypatch.setattr(pipeline, "ha_get", get)
    checks = pipeline.validate_entities()
    assert checks and any(c["status"] == "missing" for c in checks)
    assert calls == ["/api/states"]


def test_search_is_background_preview_and_failure_allows_retry():
    module = importlib.import_module("discovery_search")
    entered, release = threading.Event(), threading.Event()

    def fetch():
        entered.set()
        release.wait(2)
        raise TimeoutError("search timed out")

    search = module.DiscoverySearch(fetch)
    assert search.start()
    assert entered.wait(1)
    assert search.status()["running"]
    assert not search.start()
    release.set()
    search.thread.join(2)
    assert not search.status()["running"]
    assert "timed out" in search.status()["error"]
    search.fetch = lambda: {"candidates": []}
    assert search.start()
    search.thread.join(2)
    assert search.status()["result"] == {"candidates": []}
    assert search.status()["error"] is None


def test_registry_search_has_an_overall_timeout(monkeypatch):
    import asyncio

    async def stuck(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(pipeline, "_ws_request", stuck)
    monkeypatch.setattr(pipeline, "DISCOVERY_REGISTRY_TIMEOUT_S", 0.01, raising=False)
    with pytest.raises(TimeoutError):
        pipeline.search_entities()


def test_search_preview_does_not_write_options_or_cache(monkeypatch, tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setenv("SUPERVISOR_TOKEN", "test")
    path = tmp_path / "options.json"
    original = json.dumps({"entity_battery_soc": "sensor.saved_soc"})
    path.write_text(original)

    async def ws(*args, **kwargs):
        return registry("one") + registry("two")

    class HA(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = [] if self.path.endswith("/states") else {"latitude": 45, "longitude": 34}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), HA)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(pipeline, "HA_BASE", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setattr(pipeline, "_ws_request", ws)
    try:
        result = pipeline.search_entities()
    finally:
        server.shutdown()
        server.server_close()
    assert len(result["candidates"]) == 2
    assert "entity_battery_soc" not in result["found"]
    assert path.read_text() == original
    assert not (tmp_path / "discovery.json").exists()


def test_panel_stays_accessible_during_search_and_save_is_explicit(monkeypatch, tmp_path):
    import urllib.request
    from http.server import ThreadingHTTPServer

    module = importlib.import_module("discovery_search")
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    (tmp_path / "options.json").write_text("{}")
    monkeypatch.setattr(pipeline, "supervisor_options", lambda: None)
    saved = []
    monkeypatch.setattr(pipeline, "_post_options", lambda opts: saved.append(opts))
    release = threading.Event()

    def fetch():
        release.wait(3)
        return discovery.discover(registry("one") + registry("two"), {}, None)

    search = module.DiscoverySearch(fetch)
    monkeypatch.setattr(main, "SEARCH", search)
    server = ThreadingHTTPServer(("127.0.0.1", 0), main.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/", timeout=2) as response:
            assert b"searchEntities()" in response.read()
        with urllib.request.urlopen(
            urllib.request.Request(base + "/api/discovery", data=b"{}"), timeout=2
        ) as response:
            assert response.status == 202
        with urllib.request.urlopen(base + "/api/discovery", timeout=2) as response:
            assert json.load(response)["running"]
        with urllib.request.urlopen(base + "/api/options", timeout=2) as response:
            assert json.load(response)["groups"]
        assert saved == []
        release.set()
        search.thread.join(2)
        result = search.status()["result"]
        chosen = result["candidates"][1]
        request = urllib.request.Request(
            base + "/api/discovery/apply",
            data=json.dumps({"candidate_id": chosen["id"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 200
        assert pipeline.options()["entity_battery_soc"] == "sensor.two_soc"
        assert len(saved) == 1
    finally:
        release.set()
        server.shutdown()
        server.server_close()
