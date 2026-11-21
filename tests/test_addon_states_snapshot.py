"""One HA state snapshot per planning run instead of a request per entity."""

import json
import sys
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402

STATES = [
    {"entity_id": "sensor.inverter_battery", "state": "61", "attributes": {}},
    {"entity_id": "number.inverter_program_1_soc", "state": "45", "attributes": {}},
]


@pytest.fixture
def fake_ha(monkeypatch):
    monkeypatch.setenv("SUPERVISOR_TOKEN", "x")
    calls = []

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
        url = request.full_url
        calls.append(url)
        if url.endswith("/api/states"):
            return Response(STATES)
        if "/api/states/" in url:
            entity = url.rsplit("/", 1)[-1]
            return Response({"entity_id": entity, "state": "live"})
        return Response({})

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)
    yield calls
    pipeline.drop_states()


def test_entity_reads_come_from_one_snapshot(fake_ha):
    pipeline.snapshot_states()
    assert pipeline.ha_get("/api/states/sensor.inverter_battery")["state"] == "61"
    assert pipeline.ha_get("/api/states/number.inverter_program_1_soc")["state"] == "45"
    assert len(fake_ha) == 1  # the snapshot itself


def test_an_entity_missing_from_the_snapshot_is_read_live(fake_ha):
    pipeline.snapshot_states()
    assert pipeline.ha_get("/api/states/sensor.new_one")["state"] == "live"


def test_a_write_drops_the_snapshot_a_notification_does_not(fake_ha):
    pipeline.snapshot_states()
    pipeline.ha_service("persistent_notification", "create", {"message": "x"})
    assert pipeline.ha_get("/api/states/sensor.inverter_battery")["state"] == "61"
    pipeline.ha_service("number", "set_value", {"entity_id": "number.inverter_program_1_soc"})
    # The read-back after a write must see the device, not the snapshot.
    assert pipeline.ha_get("/api/states/number.inverter_program_1_soc")["state"] == "live"


def test_a_stale_snapshot_is_ignored(fake_ha, monkeypatch):
    pipeline.snapshot_states()
    monkeypatch.setattr(pipeline, "_STATES_AT", pipeline._STATES_AT - pipeline.STATES_MAX_AGE_S - 1)
    assert pipeline.ha_get("/api/states/sensor.inverter_battery")["state"] == "live"
