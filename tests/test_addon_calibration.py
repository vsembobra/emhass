"""BMS calibration: full charge every couple of weeks, in the night window."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import calibration  # noqa: E402

NOW = datetime(2026, 11, 20, 12, tzinfo=UTC)
TZ = ZoneInfo("UTC")


def _audit(full_days_ago=None, days=20):
    index = pd.date_range(end=NOW - timedelta(hours=1), periods=days * 24, freq="h")
    audit = pd.DataFrame({"soc_max": 0.8}, index=index)
    if full_days_ago is not None:
        audit.loc[NOW - timedelta(days=full_days_ago) :, "soc_max"] = 0.8
        stamp = audit.index[audit.index <= NOW - timedelta(days=full_days_ago)][-1]
        audit.loc[stamp, "soc_max"] = 1.0
    return audit


def test_last_full_charge_from_history_or_live_soc():
    assert calibration.last_full_charge(_audit(5), 0.6, NOW) == (NOW - timedelta(days=5)).replace(
        minute=0
    )
    assert calibration.last_full_charge(_audit(), 0.995, NOW) == NOW
    assert calibration.last_full_charge(_audit(), 0.6, NOW) is None


def test_due_after_the_interval_only():
    assert not calibration.status(_audit(5), 0.6, NOW, {})["due"]
    late = calibration.status(_audit(15), 0.6, NOW, {})
    assert late["due"] and late["days_since_full"] == 15.0
    assert calibration.status(_audit(5), 0.6, NOW, {"calibration_interval_days": 3})["due"]
    assert calibration.status(_audit(), 0.6, NOW, {})["due"]  # never full in history


def test_switched_off_is_never_due():
    status = calibration.status(_audit(), 0.6, NOW, {"calibration_charge_enabled": False})
    assert not status["due"] and not status["enabled"]


def test_night_window_uses_cheap_hours_or_default():
    starts = pd.date_range("2026-11-20 18:00", periods=24, freq="h", tz="UTC")
    tariff = {"tariff_enabled": True, "tariff_cheap_windows": "23:00-05:00"}
    assert calibration.night_window(starts, TZ, tariff) == [5, 6, 7, 8, 9, 10]  # 23:00-05:00
    assert calibration.night_window(starts, TZ, {}) == [6, 7, 8, 9, 10, 11, 12]  # 00:00-07:00
    inside = pd.date_range("2026-11-20 03:00", periods=24, freq="h", tz="UTC")
    assert calibration.night_window(inside, TZ, {}) == [0, 1, 2, 3]  # the window running now


def test_zones_covering_the_window():
    starts = pd.date_range("2026-11-20 18:00", periods=24, freq="h", tz="UTC")
    programs = [
        {"n": 1, "from": "00:00", "soc_pct": 50},
        {"n": 2, "from": "07:00", "soc_pct": 50},
        {"n": 3, "from": "20:00", "soc_pct": 50},
    ]
    window = calibration.night_window(starts, TZ, {})
    assert calibration.zones_in_window(programs, starts, TZ, window) == {1}


def test_a_battery_capped_below_99_percent_counts_full_at_its_max():
    capped = {"min_soc_pct": 20, "max_soc_pct": 95}
    # Charged to its 95 % max tonight: done, not due again tomorrow night.
    assert not calibration.status(_audit(), 0.95, NOW, capped)["due"]
    # The usual 99 % still applies at a 100 % max.
    assert calibration.status(_audit(), 0.95, NOW, {"max_soc_pct": 100})["due"]
