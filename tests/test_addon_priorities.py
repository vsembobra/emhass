"""Priorities, flexible-load windows, autonomy and published sensors."""

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import deferrable  # noqa: E402
import pipeline  # noqa: E402
import policy  # noqa: E402

UTC = ZoneInfo("UTC")
STARTS = pd.date_range("2026-06-21 00:00", periods=24, freq="h", tz="UTC")


def test_priority_maps_reserve_quantile_and_margins():
    assert policy.priority({"priority_mode": "resilience"})["reserve_quantile"] == 0.95
    assert policy.priority({})["reserve_quantile"] == 0.9
    assert policy.priority({"priority_mode": "nonsense"}) == policy.PRIORITIES["balanced"]
    assert policy.priority({"priority_mode": "resilience"})["margins_h"] == (1.0, 2.0)
    assert policy.default_margin({"priority_mode": "resilience"}) == 1.0
    assert policy.default_margin({}) == 0.0
    assert policy.default_margin({"priority_mode": "economy"}) == 0.0


class Event:
    def __init__(self, start, end, source="measured"):
        self.start, self.end, self.source, self.interrupted = start, end, source, False


NOW = pd.Timestamp("2026-11-20 12:00", tz="UTC").to_pydatetime()


def test_outage_regime_series_and_quiet():
    recent = Event(NOW - pd.Timedelta(days=3, hours=3), NOW - pd.Timedelta(days=3))
    old = Event(NOW - pd.Timedelta(days=20, hours=3), NOW - pd.Timedelta(days=20))
    live = policy.outage_regime([old, recent], NOW, 14.0)
    assert live["active"] and live["events_in_window"] == 1
    assert live["days_since_last"] == pytest.approx(3.0)
    quiet = policy.outage_regime([old], NOW, 14.0)
    assert not quiet["active"] and quiet["days_since_last"] == pytest.approx(20.0)
    assert not policy.outage_regime([], NOW, 14.0)["active"]


def test_open_outage_keeps_the_series_live_and_scheduled_ones_do_not_count():
    open_now = Event(NOW - pd.Timedelta(hours=1), None)
    assert policy.outage_regime([open_now], NOW, 14.0)["active"]
    announced = Event(NOW + pd.Timedelta(days=1), NOW + pd.Timedelta(days=1, hours=2), "scheduled")
    assert not policy.outage_regime([announced], NOW, 14.0)["active"]


def test_memory_days_option():
    assert policy.memory_days({}) == 14.0
    assert policy.memory_days({"outage_memory_days": 7}) == 7.0
    assert policy.memory_days({"outage_memory_days": "x"}) == 14.0


def _sun():
    surplus = np.zeros(24)
    surplus[9:16] = 2.0
    curtail = np.zeros(24)
    curtail[12:14] = 3.0  # battery full by noon
    return curtail, surplus


def test_flexible_load_lands_on_curtailed_hours():
    curtail, surplus = _sun()
    (rec,) = deferrable.recommend(
        [{"name": "Бойлер", "power_kw": 2.0, "hours": 2}], STARTS, UTC, curtail, surplus
    )
    assert (rec["start_local"], rec["end_local"]) == ("12:00", "14:00")
    assert rec["free_solar_kwh"] == 4.0 and rec["sunny"]
    assert rec["slug"] == "load"  # cyrillic-only names fall back to a safe slug


def test_flexible_load_respects_its_window_and_partial_hours():
    curtail, surplus = _sun()
    (rec,) = deferrable.recommend(
        [{"name": "Washer", "power_kw": 1.0, "hours": 1.5, "window": "08:00-12:00"}],
        STARTS,
        UTC,
        curtail,
        surplus,
    )
    assert rec["start_local"] == "09:00" and rec["end_local"] == "10:30"
    assert rec["energy_kwh"] == 1.5 and rec["surplus_solar_kwh"] == 1.5


def test_no_sun_and_impossible_window():
    zeros = np.zeros(24)
    recs = deferrable.recommend(
        [
            {"name": "Pump", "power_kw": 1.0, "hours": 1},
            {"name": "Long", "power_kw": 1.0, "hours": 5, "window": "10:00-12:00"},
        ],
        STARTS,
        UTC,
        zeros,
        zeros,
    )
    assert recs[0]["start_local"] == "00:00" and not recs[0]["sunny"]
    assert recs[1]["start_local"] is None


def test_window_wraps_midnight():
    zeros = np.zeros(24)
    (rec,) = deferrable.recommend(
        [{"name": "Night", "power_kw": 1.0, "hours": 2, "window": "23:00-03:00"}],
        STARTS,
        UTC,
        zeros,
        zeros,
    )
    assert rec["start_local"] == "00:00"


def test_autonomy_hours():
    assert pipeline.autonomy(0.6, 0.1, 30.0, 0.95, 1.0) == pytest.approx(14.25)
    assert pipeline.autonomy(0.05, 0.1, 30.0, 0.95, 1.0) == 0.0
    assert pipeline.autonomy(0.6, 0.1, 30.0, 0.95, 0.0) is None
