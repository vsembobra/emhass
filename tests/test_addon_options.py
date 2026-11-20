"""Panel option edits: only entity-binding keys may be written back."""

import json
import sys
from pathlib import Path

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402


def test_update_options_filters_and_merges(monkeypatch, tmp_path):
    (tmp_path / "options.json").write_text(
        json.dumps({"entity_pv_power": "sensor.old", "capacity_kwh": 32.0})
    )
    monkeypatch.setattr(pipeline, "DATA", tmp_path)

    sent = {}

    class FakeResponse:
        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=0, **kw):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("SUPERVISOR_TOKEN", "x")

    result = pipeline.update_options(
        {
            "entity_pv_power": "sensor.new_pv",
            "capacity_kwh": 999,  # not an entity key — must be dropped
            "evil_key": "rm -rf",
        }
    )

    assert result["saved"] == {"entity_pv_power": "sensor.new_pv"}
    assert sent["url"] == "http://supervisor/addons/self/options"
    written = sent["body"]["options"]
    assert written["entity_pv_power"] == "sensor.new_pv"
    assert written["capacity_kwh"] == 32.0  # untouched
    assert "evil_key" not in written


def test_update_options_empty_changes_is_noop(monkeypatch):
    monkeypatch.setattr(
        pipeline, "options", lambda: (_ for _ in ()).throw(AssertionError("must not read"))
    )
    assert pipeline.update_options({"capacity_kwh": 1}) == {"saved": {}}
