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


def test_update_options_errors_follow_the_panel_language(monkeypatch, tmp_path):
    _patch_supervisor(monkeypatch, tmp_path, {"capacity_kwh": 32.0})
    with pytest.raises(ValueError, match="999.0 is outside"):
        pipeline.update_options({"capacity_kwh": 999}, lang="en")


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


def test_options_view_carries_english_mirror():
    groups = options_meta.options_view({"capacity_kwh": 32.0})
    for group in groups:
        assert group["title"] and group["title_en"]
        for item in group["items"]:
            assert item["label_en"] and item["hint_en"], item["key"]
    item = next(i for g in groups for i in g["items"] if i["key"] == "capacity_kwh")
    assert item["label_en"] == "Battery capacity"
    titles = [g["title_en"] for g in groups]
    assert "Inverter and battery" in titles and "Application" in titles


def test_entity_metadata_is_bilingual():
    for key, meta in options_meta.OPTION_META.items():
        assert meta.get("label_en") and meta.get("hint_en"), key
    assert options_meta.entity_hint("entity_pv_power")
    assert options_meta.entity_hint("entity_pv_power", "en")
    assert options_meta.entity_hint("entity_pv_power", "en") != options_meta.entity_hint(
        "entity_pv_power"
    )


def test_coerce_option_english_errors():
    with pytest.raises(ValueError, match="is not a number"):
        options_meta.coerce_option("capacity_kwh", "abc", "en")
    with pytest.raises(ValueError, match="999.0 is outside 1–200"):
        options_meta.coerce_option("capacity_kwh", 999, "en")
    with pytest.raises(ValueError, match="HH:MM required"):
        options_meta.coerce_option("run_time", "6:47", "en")


def _supervisor_holding(monkeypatch, tmp_path, local, stored):
    """The Supervisor holds ``stored``; the add-on's file still says ``local``."""
    (tmp_path / "options.json").write_text(json.dumps(local))
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setenv("SUPERVISOR_TOKEN", "x")
    posted = {}

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
            return Response({"result": "ok", "data": {"options": stored}})
        posted["body"] = json.loads(request.data)
        return Response({"result": "ok"})

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)
    return posted


def test_options_saved_in_the_addon_settings_apply_without_a_restart(monkeypatch, tmp_path):
    # Owner case: 25 % saved, the running add-on's file still says 10 %.
    _supervisor_holding(
        monkeypatch, tmp_path, {"min_soc_pct": 10.0, "capacity_kwh": 32.0},
        {"min_soc_pct": 25.0, "capacity_kwh": 32.0},
    )
    assert pipeline.sync_options()["min_soc_pct"] == 25.0
    assert pipeline.options()["min_soc_pct"] == 25.0  # what every run reads


def test_a_panel_save_keeps_edits_made_in_the_addon_settings(monkeypatch, tmp_path):
    posted = _supervisor_holding(
        monkeypatch, tmp_path, {"min_soc_pct": 10.0, "capacity_kwh": 32.0},
        {"min_soc_pct": 25.0, "capacity_kwh": 32.0},
    )
    pipeline.update_options({"capacity_kwh": "30"})
    # Merged into what the Supervisor holds, not into the stale file.
    assert posted["body"]["options"] == {"min_soc_pct": 25.0, "capacity_kwh": 30.0}
    assert pipeline.stored_options() == {"min_soc_pct": 25.0, "capacity_kwh": 30.0}
    assert pipeline.options()["capacity_kwh"] == 30.0  # what a run uses


def test_without_the_supervisor_the_file_stays(monkeypatch, tmp_path):
    (tmp_path / "options.json").write_text(json.dumps({"min_soc_pct": 10.0}))
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    assert pipeline.sync_options() == {"min_soc_pct": 10.0}
