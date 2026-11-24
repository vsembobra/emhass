"""A weather-model switch must not reuse another model's inputs or bundles."""

import json
import sys
import urllib.parse
from pathlib import Path

import pandas as pd
import pytest
import yaml

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import daily  # noqa: E402
import options_meta  # noqa: E402
import pipeline  # noqa: E402
import retrain  # noqa: E402

from emhass.load_quantiles import LoadQuantileBundle  # noqa: E402

MODELS = {"auto": None, "ecmwf": "ecmwf_ifs", "icon": "icon_seamless", "gfs": "gfs_seamless"}
SITE = {"latitude": 44.95, "longitude": 34.11}


def _options(tmp_path, model):
    (tmp_path / "options.json").write_text(json.dumps({"open_meteo_model": model}))


def _weather():
    return {"hourly": {"time": [1795478400, 1795482000], "temperature_2m": [10, 11]}}


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(_weather()).encode()


def test_setting_is_available_in_panel_and_supervisor():
    config = yaml.safe_load((ADDON / "config.yaml").read_text())
    assert options_meta.DEFAULTS.get("open_meteo_model") == "auto"
    assert config["options"]["open_meteo_model"] == "auto"
    assert config["schema"]["open_meteo_model"] == "list(auto|ecmwf|icon|gfs)"
    view = options_meta.options_view(options_meta.DEFAULTS)
    item = next(i for group in view for i in group["items"] if i["key"] == "open_meteo_model")
    assert set(item["choices"]) == set(MODELS)
    assert item["hint"] and item["hint_en"]


@pytest.mark.parametrize("model,api_id", MODELS.items())
def test_live_request_uses_selected_model(tmp_path, monkeypatch, model, api_id):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    queries = []

    def fetch(url, **kwargs):
        queries.append(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query))
        return Response()

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fetch)
    frame, name, stale = pipeline.archive_weather({**SITE, "open_meteo_model": model})
    assert queries[0].get("models") == ([api_id] if api_id else None)
    assert frame["temperature_2m"].tolist() == [10, 11]
    assert not stale
    assert (
        pipeline.weather_models.scoped(tmp_path / "weather_archive", model).joinpath(name).exists()
    )


@pytest.mark.parametrize("model,api_id", MODELS.items())
def test_history_request_matches_live_model(monkeypatch, model, api_id):
    queries = []

    def fetch(request, **kwargs):
        queries.append(urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query))
        return Response()

    monkeypatch.setattr(retrain.urllib.request, "urlopen", fetch)
    retrain._fetch_weather_chunk(
        {**SITE, "open_meteo_model": model}, pd.Timestamp("2026-11-20"), pd.Timestamp("2026-11-21")
    )
    assert queries[0].get("models") == ([api_id] if api_id else None)


def test_offline_switch_never_uses_auto_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    archive = tmp_path / "weather_archive"
    archive.mkdir()
    (archive / "20261124T000000Z.json").write_text(json.dumps(_weather()))

    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", offline)
    with pytest.raises(OSError, match="offline"):
        pipeline.archive_weather({**SITE, "open_meteo_model": "icon"})
    # The legacy Auto archive is still usable when Auto is selected.
    frame, _, stale = pipeline.archive_weather(SITE)
    assert stale and not frame.empty


def test_invalid_manual_selection_fails_before_request(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    with pytest.raises(ValueError, match="open_meteo_model"):
        pipeline.archive_weather({**SITE, "open_meteo_model": "wrong"})


def test_model_paths_preserve_auto_and_separate_explicit_models(tmp_path):
    paths = [pipeline.weather_models.scoped(tmp_path / "models", m) for m in MODELS]
    assert paths[0] == tmp_path / "models"
    assert len(set(paths)) == 4
    assert all(p.is_relative_to(tmp_path) for p in paths)


def test_retrain_state_switch_and_return(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "DATA", tmp_path)
    monkeypatch.setattr(retrain, "STATE", tmp_path / "retrain_state.json")
    monkeypatch.setattr(retrain, "SHARE", tmp_path / "share")
    _options(tmp_path, "auto")
    retrain._save_state({"last_retrain_utc": "2026-11-24T00:00:00+00:00", "models": {"pv": "auto"}})
    _options(tmp_path, "icon")
    assert retrain._load_state()["last_retrain_utc"] is None
    retrain._save_state({"last_retrain_utc": None, "models": {"pv": "icon"}})
    assert retrain._load_state()["models"]["pv"] == "icon"
    _options(tmp_path, "auto")
    assert retrain._load_state()["models"]["pv"] == "auto"


def test_cached_weather_reader_follows_selected_model(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    _options(tmp_path, "auto")
    archive = tmp_path / "weather_archive"
    archive.mkdir()
    (archive / "20261124T000000Z.json").write_text(json.dumps(_weather()))
    assert pipeline.archived_weather() is not None
    _options(tmp_path, "icon")
    assert pipeline.archived_weather() is None


def test_daily_statistics_do_not_read_another_models_history(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "DATA", tmp_path)
    monkeypatch.setattr(daily, "WEATHER_HIST", tmp_path / "weather_previous_day1.jsonl")
    _options(tmp_path, "icon")
    # Invalid legacy contents would fail if opened for the new model.
    (tmp_path / "weather_previous_day1.jsonl").write_text("unrelated auto history")
    result = daily._potential_pv(None, pd.date_range("2026-11-24", periods=2, freq="h", tz="UTC"))
    assert result.isna().all()


def test_training_history_does_not_reuse_auto_file(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "WEATHER_HIST", tmp_path / "weather_previous_day1.jsonl")
    (tmp_path / "weather_previous_day1.jsonl").write_text("unrelated auto history")
    end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=1)
    index = pd.date_range(end=end, periods=2, freq="h")
    fetched = pd.DataFrame({"shortwave_radiation_previous_day1": [50.0, 100.0]}, index=index)
    fetched.index.name = "start_utc"
    sites = []

    def fetch(site, start, finish):
        sites.append(site)
        return fetched

    monkeypatch.setattr(retrain, "_fetch_weather_chunk", fetch)
    result = retrain._ensure_weather_history({**SITE, "open_meteo_model": "icon"}, index[0])
    assert result.iloc[-1, 0] == 100.0
    assert sites[0]["open_meteo_model"] == "icon"
    path = pipeline.weather_models.scoped(tmp_path / "weather_previous_day1.jsonl", "icon")
    assert path.exists()
    assert (tmp_path / "weather_previous_day1.jsonl").read_text() == "unrelated auto history"


def test_training_writes_separate_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "MODELS", tmp_path / "models")
    index = pd.date_range("2026-11-01", periods=72, freq="h", tz="UTC")
    audit = pd.DataFrame(
        {"load_kw": 1.0, "load_filter": "keep", "temp_air_observed_c": 10.0}, index=index
    )
    result = retrain._retrain_load(audit, {**SITE, "open_meteo_model": "icon"})
    assert result["status"] == "promoted"
    path = pipeline.weather_models.scoped(tmp_path / "models", "icon") / "load_quantiles_v1"
    assert LoadQuantileBundle.load(path).trained_until == index[-1].isoformat()
    assert not (tmp_path / "models" / "load_quantiles_v1").exists()


def test_drift_is_separate_for_each_weather_model(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "DATA", tmp_path)
    (tmp_path / "drift_state.json").write_text(json.dumps({"drift": True}))
    _options(tmp_path, "auto")
    assert retrain._drift_detected()
    _options(tmp_path, "icon")
    assert not retrain._drift_detected()


def test_first_switch_trains_despite_recent_auto_state(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    monkeypatch.setattr(retrain, "DATA", tmp_path)
    monkeypatch.setattr(retrain, "STATE", tmp_path / "retrain_state.json")
    monkeypatch.setattr(retrain, "MODELS", tmp_path / "models")
    monkeypatch.setattr(retrain, "SHARE", tmp_path / "share")
    monkeypatch.setattr(retrain.hardware, "outdated", lambda *a: False)
    monkeypatch.setattr(retrain.hardware, "due", lambda *a: False)
    _options(tmp_path, "auto")
    retrain._save_state({"last_retrain_utc": datetime.now(UTC).isoformat(), "models": {}})
    _options(tmp_path, "icon")
    index = pd.date_range("2026-11-01", periods=2, freq="h", tz="UTC")
    audit = pd.DataFrame({"load_kw": 1.0, "temp_air_observed_c": 10.0}, index=index)
    monkeypatch.setattr(retrain, "_audit_frame", lambda: audit)
    monkeypatch.setattr(retrain, "_ensure_weather_history", lambda *a: None)
    models_seen = []

    def train(audit, site, weather):
        models_seen.append(site["open_meteo_model"])
        return {"status": "kept"}

    monkeypatch.setattr(retrain, "_retrain_load", train)
    monkeypatch.setattr(retrain, "_retrain_pv", train)
    retrain.retrain_if_due()
    assert models_seen == ["icon", "icon"]
    assert retrain._load_state()["pv_model_tried"] == pipeline.PV_MODEL
