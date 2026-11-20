"""Panel option edits: entity keys plus whitelisted, validated scalar options."""

import json
import sys
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import options_meta  # noqa: E402
import pipeline  # noqa: E402


def _patch_supervisor(monkeypatch, tmp_path, options_json):
    (tmp_path / "options.json").write_text(json.dumps(options_json))
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
    return sent


def test_update_options_filters_and_merges(monkeypatch, tmp_path):
    sent = _patch_supervisor(
        monkeypatch, tmp_path, {"entity_pv_power": "sensor.old", "capacity_kwh": 32.0}
    )

    result = pipeline.update_options(
        {
            "entity_pv_power": "sensor.new_pv",
            "evil_key": "rm -rf",
        }
    )

    assert result["saved"] == {"entity_pv_power": "sensor.new_pv"}
    assert result["rejected"] == ["evil_key"]
    assert sent["url"] == "http://supervisor/addons/self/options"
    written = sent["body"]["options"]
    assert written["entity_pv_power"] == "sensor.new_pv"
    assert written["capacity_kwh"] == 32.0  # untouched
    assert "evil_key" not in written


def test_update_options_coerces_scalar_settings(monkeypatch, tmp_path):
    sent = _patch_supervisor(
        monkeypatch, tmp_path, {"capacity_kwh": 32.0, "apply_mode": False}
    )

    result = pipeline.update_options(
        {
            "capacity_kwh": "40",  # float coerced from panel string
            "apply_mode": True,  # bool stays bool for the schema
            "replan_minutes": "30",  # int coerced
            "day_zone_strategy": "flat",
        }
    )

    assert result["saved"] == {
        "capacity_kwh": 40.0,
        "apply_mode": True,
        "replan_minutes": 30,
        "day_zone_strategy": "flat",
    }
    written = sent["body"]["options"]
    assert written["capacity_kwh"] == 40.0
    assert written["apply_mode"] is True


def test_update_options_rejects_out_of_range(monkeypatch, tmp_path):
    _patch_supervisor(monkeypatch, tmp_path, {"capacity_kwh": 32.0})
    with pytest.raises(ValueError, match="вне диапазона"):
        pipeline.update_options({"capacity_kwh": 999})
    with pytest.raises(ValueError, match="допустимы только"):
        pipeline.update_options({"day_zone_strategy": "spiral"})
    with pytest.raises(ValueError, match="ЧЧ:ММ"):
        pipeline.update_options({"run_time": "6:47"})


def test_update_options_empty_changes_is_noop(monkeypatch):
    monkeypatch.setattr(
        pipeline, "options", lambda: (_ for _ in ()).throw(AssertionError("must not read"))
    )
    assert pipeline.update_options({"evil_key": 1}) == {"saved": {}, "rejected": ["evil_key"]}


def test_coerce_option_bool_strings():
    assert options_meta.coerce_option("apply_mode", "true") is True
    assert options_meta.coerce_option("apply_mode", "0") is False
    with pytest.raises(ValueError):
        options_meta.coerce_option("apply_mode", "maybe")


def test_coerce_option_float_comma_and_int_rounding():
    assert options_meta.coerce_option("charge_efficiency", "0,95") == 0.95
    assert options_meta.coerce_option("tou_zones", "6.0") == 6


def test_options_view_covers_groups_and_values():
    opts = {
        "capacity_kwh": 32.0,
        "apply_mode": False,
        "day_zone_strategy": "ladder",
        "latitude": 44.94892,
    }
    groups = options_meta.options_view(opts)
    keys = [item["key"] for group in groups for item in group["items"]]
    # Every non-entity metadata key is offered to the panel exactly once.
    expected = [k for k, m in options_meta.OPTION_META.items() if not m.get("entity")]
    assert sorted(keys) == sorted(expected)
    assert len(keys) == len(set(keys))
    item = next(i for g in groups for i in g["items"] if i["key"] == "capacity_kwh")
    assert item["value"] == 32.0
    assert item["hint"] and item["label"]
