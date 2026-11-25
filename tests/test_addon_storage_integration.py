"""Existing panel endpoints and bootstrap work with compact and legacy data."""

import json
import subprocess
import sys
import threading
import urllib.request
import zlib
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path[:0] = [str(ADDON / "app"), str(ADDON / "src")]
import data_store  # noqa: E402
import main  # noqa: E402
import pipeline  # noqa: E402


@pytest.mark.parametrize("error", [OSError("disk"), EOFError("truncated gzip"), zlib.error("bad block")])
def test_failed_maintenance_does_not_block_panel_start(monkeypatch, error):
    def failed(*args):
        raise error
    monkeypatch.setattr(data_store, "compact_data", failed)
    # Do not start scheduler/watchers in this startup control-flow check.
    class IdleThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
    monkeypatch.setattr(main.threading, "Thread", IdleThread)
    opened = []
    class Server:
        def __init__(self, *args):
            pass
        def serve_forever(self):
            opened.append(True)
    monkeypatch.setattr(main, "ThreadingHTTPServer", Server)
    main.main()
    assert opened == [True]


@pytest.mark.parametrize("compact", [False, True])
def test_report_history_metrics_and_status_keep_their_api_shape(tmp_path, monkeypatch, compact):
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    (tmp_path / "options.json").write_text("{}")
    report = {"issue_utc": "2026-11-25T00:00:00+00:00", "notification": "full plan",
              "advisory": {"recommended_floor_soc": .3, "current_floor_soc": .2, "delta_pp": 10},
              "reserve": {"reserve_kwh": 4.5, "hold_until_local": "2026-12-09"}}
    directory = tmp_path / "shadow_reports"
    directory.mkdir()
    path = directory / "2026-11-25.json"
    path.write_text(json.dumps(report))
    (tmp_path / "hourly_audit.jsonl").write_text('{"start_utc":"2026-11-25T00:00:00Z"}\n')
    (tmp_path / "metrics.jsonl").write_text('{"date":"2026-11-25","load":{"mae_kwh":1.2}}\n')
    if compact:
        data_store.compact_data(tmp_path)
        path.write_text(json.dumps(report))  # interrupted cleanup must not double UI rows
    server = ThreadingHTTPServer(("127.0.0.1", 0), main.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    def get(endpoint):
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/{endpoint}") as response:
            return json.load(response)
    try:
        assert get("latest") == report
        history = get("history")
        assert len(history) == 1 and history[0]["date"] == "2026-11-25"
        assert get("metrics")["days"][0]["load"]["mae_kwh"] == 1.2
        assert get("status")["audit_rows"] == 1
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("already_present", [False, True])
def test_bootstrap_recognizes_gzip_and_does_not_restore_old_seed_over_it(tmp_path, already_present):
    data, share, models = (tmp_path / name for name in ("data", "share", "models_default"))
    models.mkdir()
    seed = share / "seed"
    seed.mkdir(parents=True)
    data.mkdir()
    data_store.write_text(seed / "hourly_audit.jsonl", '{"value":"seed"}\n')
    if already_present:
        data_store.write_text(data / "hourly_audit.jsonl", '{"value":"current"}\n')
        (seed / "hourly_audit.jsonl").write_text('{"value":"obsolete plain seed"}\n')
    weather_seed = seed / "weather_archive" / "20261125T000000Z.json"
    data_store.write_json(weather_seed, {"value": "new weather"})
    weather_seed.write_text('{"value":"old weather"}')
    script = (ADDON / "run.sh").read_text().replace("/data", str(data))
    script = script.replace("/share/emhass_shadow", str(share)).replace("/opt/models_default", str(models))
    script = script.replace("exec python /opt/app/main.py", "exit 0")
    subprocess.run(["sh", "-c", script], check=True, capture_output=True)
    assert not (data / "hourly_audit.jsonl").exists()
    assert data_store.read_json(data / "hourly_audit.jsonl")["value"] == ("current" if already_present else "seed")
    assert data_store.read_json(data / "weather_archive" / weather_seed.name)["value"] == "new weather"
