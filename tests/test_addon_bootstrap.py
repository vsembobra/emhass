"""Fresh install learns from Home Assistant: recorder, statistics, outages."""

import asyncio
import json
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import main  # noqa: E402
import pipeline  # noqa: E402
import retrain  # noqa: E402

ENT = {
    "mean_sensors": {
        "load_kw": ("sensor.load", 0.001),
        "critical_load_kw": ("sensor.ups", 0.001),
        "pv_measured_kw": ("sensor.pv", 0.001),
        "battery_power_kw_raw_sign": ("sensor.battery", 0.001),
        "grid_power_kw_raw_sign": ("sensor.grid", 0.001),
        "temp_air_observed_c": ("", 1.0),
    },
    "voltages": {"grid_min_v": "sensor.voltage"},
    "soc": "sensor.soc",
    "export_counter": "sensor.export",
    "grid_binary": "binary_sensor.grid",
}
NOON = datetime(2026, 6, 21, 9, tzinfo=UTC)  # daylight at 45N 34E


def _stats(hours, voltage_min=230.0, ms=True):
    def stamp(h):
        return h.timestamp() * 1000 if ms else h.isoformat()

    def series(**values):
        return [{"start": stamp(h), **values} for h in hours]

    return {
        "sensor.load": series(mean=800.0),
        "sensor.ups": series(mean=400.0),
        "sensor.pv": series(mean=2500.0),
        "sensor.battery": series(mean=-1500.0),
        "sensor.grid": series(mean=0.0),
        "sensor.soc": series(mean=60.0, max=65.0),
        "sensor.export": series(change=0.0),
        "sensor.voltage": series(min=voltage_min),
    }


def test_statistics_rows_map_means_soc_export_and_grid():
    hours = [NOON, NOON + timedelta(hours=1)]
    (row, _) = pipeline.statistics_rows(_stats(hours), hours, ENT, 45.0, 34.0, 5.0)
    assert row["load_kw"] == pytest.approx(0.8) and row["critical_load_kw"] == pytest.approx(0.4)
    assert row["battery_power_kw_raw_sign"] == pytest.approx(-1.5)
    assert (row["soc_mean"], row["soc_max"]) == (0.6, 0.65)
    assert row["export_kwh"] == 0.0 and row["grid_quality"] == "recorded_up"
    assert row["load_filter"] == "keep" and row["pv_filter"] == "keep"
    assert row["split"] == "statistics" and row["solar_elevation_deg"] > 10


def test_low_voltage_minimum_marks_an_outage_hour():
    (row,) = pipeline.statistics_rows(_stats([NOON], voltage_min=0.0), [NOON], ENT, 45.0, 34.0)
    assert row["raw_grid_status"] == "outage" and row["load_filter"] == "grid_quality"


def test_without_voltage_statistics_the_grid_is_assumed_up():
    stats = _stats([NOON], ms=False)
    stats.pop("sensor.voltage")
    (row,) = pipeline.statistics_rows(stats, [NOON], ENT, 45.0, 34.0)
    assert row["grid_quality"] == "lts_assumed_up" and row["load_filter"] == "keep"


def test_hours_without_any_statistics_are_skipped():
    later = NOON + timedelta(hours=5)
    rows = pipeline.statistics_rows(_stats([NOON]), [NOON, later], ENT, 45.0, 34.0)
    assert len(rows) == 1


class FakeWs:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    async def receive_json(self, timeout=None):
        return self.replies.pop(0)

    async def send_json(self, payload):
        self.sent.append(payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _fake_aiohttp(monkeypatch, ws):
    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def ws_connect(self, url, **kwargs):
            ws.url = url
            return ws

    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(ClientSession=Session))
    monkeypatch.setenv("SUPERVISOR_TOKEN", "token")


def test_statistics_websocket_protocol(monkeypatch):
    ws = FakeWs(
        [
            {"type": "auth_required"},
            {"type": "auth_ok"},
            {"type": "event", "id": 9},
            {"id": 1, "type": "result", "success": True, "result": {"sensor.load": [{"start": 0}]}},
        ]
    )
    _fake_aiohttp(monkeypatch, ws)
    result = pipeline.fetch_statistics(
        NOON, NOON + timedelta(days=1), ["sensor.load", "binary_sensor.grid", ""]
    )
    assert result == {"sensor.load": [{"start": 0}]}
    auth, query = ws.sent
    assert auth == {"type": "auth", "access_token": "token"}
    assert query["type"] == "recorder/statistics_during_period"
    assert query["statistic_ids"] == ["sensor.load"]  # binary sensors keep no statistics
    assert query["period"] == "hour" and query["units"]["power"] == "W"
    assert ws.url == pipeline.SUPERVISOR_WS


def test_statistics_websocket_auth_failure(monkeypatch):
    _fake_aiohttp(monkeypatch, FakeWs([{"type": "auth_required"}, {"type": "auth_invalid"}]))
    with pytest.raises(PermissionError):
        asyncio.run(pipeline._ws_statistics(NOON, NOON, ["sensor.load"]))


def test_backfill_statistics_prepends_months_once(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    recent = pd.date_range(now - timedelta(days=2), now - timedelta(hours=1), freq="h")
    (tmp_path / "hourly_audit.jsonl").write_text(
        "".join(
            json.dumps({"start_utc": h.isoformat(), "load_kw": 1.0, "split": "live"}) + "\n"
            for h in recent
        )
    )
    calls = []

    def fake_fetch(start, end, ids):
        calls.append((start, end))
        hours = list(pd.date_range(start, end - timedelta(hours=1), freq="h"))
        return _stats([h.to_pydatetime() for h in hours])

    monkeypatch.setattr(pipeline, "fetch_statistics", fake_fetch)
    site = {"latitude": 45.0, "longitude": 34.0}
    added = pipeline.backfill_statistics(site, ENT, 40)
    assert added == 38 * 24  # 40 days minus the 2 already in the audit
    assert len(calls) == 2  # 30-day chunks
    audit = pipeline._read_audit()
    assert audit.index[0] == now - timedelta(days=40)
    assert set(audit["split"]) == {"statistics", "live"}
    assert pipeline.backfill_statistics(site, ENT, 40) == 0


def test_bootstrap_pulls_recorder_outages_and_statistics(monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline, "resolve_entities", lambda opts: ENT)
    monkeypatch.setattr(pipeline, "update_audit", lambda site, ent, backfill_days: (None, 240))
    monkeypatch.setattr(
        pipeline, "update_outage_log", lambda ent, hours: seen.setdefault("hours", hours) and 3
    )

    def broken(*a):
        raise ConnectionError("no websocket")

    monkeypatch.setattr(pipeline, "backfill_statistics", broken)
    opts = {"latitude": 45, "longitude": 34, "time_zone": "UTC", "history_backfill_days": 10}
    result = pipeline.bootstrap(opts)
    assert result["recorder_rows"] == 240 and seen["hours"] == 240
    assert result["statistics_rows"] == 0 and "no websocket" in result["statistics_error"]


def test_fresh_install_bootstraps_then_trains_then_plans(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(main, "DATA", tmp_path)
    monkeypatch.setattr(pipeline, "options", lambda: {})
    monkeypatch.setattr(
        pipeline, "bootstrap", lambda opts: order.append("bootstrap") or {"at": "x"}
    )
    monkeypatch.setattr(
        retrain, "retrain_if_due", lambda force=False: order.append(f"retrain force={force}")
    )
    monkeypatch.setattr(pipeline, "run", lambda **kwargs: order.append("plan"))
    main.STATE["running"] = False
    main.run_pipeline()
    assert order == ["bootstrap", "retrain force=True", "retrain force=False", "plan"]
    order.clear()
    (tmp_path / "hourly_audit.jsonl").write_text("{}\n")
    main.run_pipeline()
    assert order == ["retrain force=False", "plan"]


def test_failed_first_retrain_is_not_stamped(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain, "DATA", tmp_path)
    monkeypatch.setattr(retrain, "STATE", tmp_path / "retrain_state.json")
    monkeypatch.setattr(retrain, "MODELS", tmp_path / "models")
    (tmp_path / "options.json").write_text(json.dumps({"latitude": 1, "longitude": 2}))
    status = retrain.retrain_if_due()
    assert status["last_retrain_utc"] is None  # no audit yet: retry next run
    assert status["models"]["load"]["status"] == "failed"


def test_outage_log_backfill_reads_recorder_depth_in_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    calls = []
    monkeypatch.setattr(
        pipeline,
        "fetch_history",
        lambda start, end, entities: calls.append((start, end, entities)) or {},
    )
    ent = {"grid_binary": "binary_sensor.grid", "mean_sensors": {"load_kw": ("sensor.load", 0.001)}}
    pipeline.update_outage_log(ent, hours=240)
    assert len(calls) == 4  # 10 days in 3-day chunks
    # The load sensor is the heartbeat on sites without sensor.time.
    assert all("sensor.load" in c[2] for c in calls)
    calls.clear()
    pipeline.update_outage_log(ent)
    assert len(calls) == 1 and "sensor.load" in calls[0][2]
