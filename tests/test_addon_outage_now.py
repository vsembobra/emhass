"""The outage in progress and the grid watcher that reacts to it."""

import asyncio
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import main  # noqa: E402
import pipeline  # noqa: E402
import policy  # noqa: E402

from emhass.outage_log import OutageEvent  # noqa: E402

NOW = datetime(2026, 11, 21, 18, 0, tzinfo=UTC)


def _series(open_hours_ago=1.5, extra=()):
    """Finished outages of 1-4 h on past days, plus the one going on now."""
    events = [
        OutageEvent(NOW - timedelta(days=d, hours=h), NOW - timedelta(days=d), "measured", False)
        for d, h in ((1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0))
    ]
    events += list(extra)
    events.append(OutageEvent(NOW - timedelta(hours=open_hours_ago), None, "measured", False))
    return events


def _now(events, soc=0.5, pv=0.0):
    return policy.outage_in_progress(
        events,
        NOW,
        14.0,
        0.9,
        soc=soc,
        min_soc=0.1,
        capacity_kwh=10.0,
        discharge_efficiency=1.0,
        critical_kw=1.0,
        pv_kw=[pv] * 48,
    )


def test_remaining_time_comes_from_outages_that_lasted_longer():
    info = _now(_series())
    # Past outages longer than 1.5 h left 0.5, 1.5 and 2.5 h more.
    assert info["elapsed_h"] == pytest.approx(1.5)
    assert info["remaining_typical_h"] == pytest.approx(1.5)
    assert info["remaining_h"] == pytest.approx(2.3)  # P90
    assert info["basis"] == "series"
    assert info["autonomy_h"] == pytest.approx(4.0) and info["enough"]


def test_short_battery_asks_to_shed_load():
    info = _now(_series(), soc=0.2)  # 1 kWh left
    assert info["autonomy_h"] == pytest.approx(1.0)
    assert not info["enough"]
    assert info["shortfall_kwh"] == pytest.approx(1.3)
    assert info["cut_kw"] == pytest.approx(1.3 / 2.3, abs=0.01)
    assert "снизьте нагрузку" in pipeline.outage_now_line(info)


def test_the_sun_stretches_the_battery():
    assert _now(_series(), soc=0.2, pv=0.5)["autonomy_h"] == pytest.approx(2.0)


def test_longer_than_every_outage_looks_an_hour_ahead():
    info = _now(_series(open_hours_ago=5.0))
    assert info["basis"] == "beyond" and info["remaining_h"] == 1.0
    assert "дольше всех" in pipeline.outage_now_line(info)


def test_a_flapping_grid_counts_from_the_first_drop():
    # The grid came back for 10 minutes 1 h ago: the drain began 2 h ago.
    flap = OutageEvent(
        NOW - timedelta(hours=2), NOW - timedelta(hours=1, minutes=10), "measured", False
    )
    info = _now(_series(open_hours_ago=1.0, extra=[flap]))
    assert info["elapsed_h"] == pytest.approx(2.0)


def test_no_outage_no_estimate():
    events = _series()[:-1]
    assert _now(events) is None


def test_grid_transitions_schedule_a_replan(monkeypatch):
    timers = []

    class FakeTimer:
        def __init__(self, delay, fn):
            timers.append((delay, fn))

        def start(self):
            pass

    monkeypatch.setattr(main.threading, "Timer", FakeTimer)
    assert main.on_grid_change("on", "off")
    assert main.STATE["grid_state"] == "off"
    assert timers and timers[0][0] == main.GRID_EVENT_DELAY_S
    assert not main.on_grid_change("off", "off")
    assert not main.on_grid_change("unavailable", "on")
    assert len(timers) == 1


def test_watcher_subscribes_and_reports_changes(monkeypatch):
    sent, changes = [], []
    incoming = [
        {"type": "auth_required"},
        {"type": "auth_ok"},
        {"id": 1, "type": "result", "success": True},
        {
            "id": 1,
            "type": "event",
            "event": {
                "variables": {
                    "trigger": {"from_state": {"state": "on"}, "to_state": {"state": "off"}}
                }
            },
        },
    ]

    class FakeWs:
        async def receive_json(self, timeout=None):
            if not incoming:
                raise TimeoutError
            return incoming.pop(0)

        async def send_json(self, data):
            sent.append(data)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class FakeSession:
        def ws_connect(self, url, **kw):
            return FakeWs()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(ClientSession=FakeSession))
    monkeypatch.setenv("SUPERVISOR_TOKEN", "x")
    asyncio.run(
        main._watch_grid("binary_sensor.inverter_grid", lambda a, b: changes.append((a, b)))
    )
    assert sent[1]["type"] == "subscribe_trigger"
    assert sent[1]["trigger"] == {"platform": "state", "entity_id": "binary_sensor.inverter_grid"}
    assert changes == [("on", "off")]
