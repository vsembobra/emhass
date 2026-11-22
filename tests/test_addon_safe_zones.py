"""No grid charging at full price via zone targets; hourly replans."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import adapter  # noqa: E402
import main  # noqa: E402
import optimize  # noqa: E402
import pipeline  # noqa: E402

UTC_TZ = ZoneInfo("UTC")
STARTS = pd.date_range("2026-11-20 00:00", periods=24, freq="h", tz="UTC")
PROGRAMS = [{"n": 1, "from": "00:00", "soc_pct": 30}, {"n": 2, "from": "10:00", "soc_pct": 30}]
# Night at 40 %, PV lifts the battery 40 -> 90 % between 10:00 and 16:00.
SOC = [0.4] * 10 + [0.5, 0.6, 0.7, 0.8, 0.85, 0.9] + [0.8] * 8


def _targets(cheap, soc=SOC, soc_init=0.4, starts=STARTS, programs=PROGRAMS, reserve=None):
    out = optimize.zone_targets(
        soc,
        soc_init,
        programs,
        starts,
        UTC_TZ,
        0.1,
        1.0,
        cheap_windows=cheap,
        reserve_floors=reserve,
    )
    return {t["n"]: t["optimal_soc_pct"] for t in out}


def test_pv_charging_window_gets_no_charge_target_without_tariff():
    # Was 80 % (end of the window): with grid charge on, a cloudy day would
    # buy up to 80 % at full price. Now the window's discharge floor.
    assert _targets(cheap=[])[2] == 50


def test_cheap_night_window_keeps_its_charge_target():
    soc = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9] + [0.8] * 14
    assert _targets(cheap=[(0, 600)], soc=soc)[1] == 90
    # Same rise outside the cheap hours: the discharge floor instead.
    assert _targets(cheap=[(23 * 60, 7 * 60)], soc=soc)[1] == 40


def test_legacy_callers_without_window_info_keep_targets():
    assert _targets(cheap=None)[2] == 80


ACTIVE = pd.date_range("2026-11-20 11:00", periods=24, freq="h", tz="UTC")


def test_zone_in_effect_is_capped_at_the_measured_soc():
    # 11:00, zone 10:00-24:00 in effect; plan: 0.6 -> 0.9, but the battery
    # actually sits at 47 %: the target may not exceed 45 % (rounded down).
    soc = [0.6, 0.7, 0.8, 0.85, 0.9, 0.9] + [0.8] * 8 + [0.5] * 10
    programs = [{"n": 1, "from": "00:00", "soc_pct": 30}, {"n": 2, "from": "10:00", "soc_pct": 30}]
    out = _targets(cheap=[], soc=soc, soc_init=0.47, starts=ACTIVE, programs=programs)
    assert out[2] == 45


def test_the_reserve_still_wins_over_the_cap():
    soc = [0.6] * 24
    programs = [{"n": 1, "from": "00:00", "soc_pct": 30}, {"n": 2, "from": "10:00", "soc_pct": 30}]
    out = _targets(
        cheap=[], soc=soc, soc_init=0.3, starts=ACTIVE, programs=programs, reserve=[0.5] * 24
    )
    assert out[2] == 50  # below the reserve: restored even at full price


@pytest.mark.parametrize(
    ("desired", "current", "expected"),
    [
        (True, "Disabled", "Grid"),
        (False, "Grid", "Disabled"),
        (True, "Gen", "Grid & Gen"),  # generator charging is kept
        (False, "Grid & Gen", "Gen"),
    ],
)
def test_select_option_recognises_deye_names(desired, current, expected):
    assert (
        adapter.select_option(desired, current, ["Disabled", "Grid", "Gen", "Grid & Gen"])
        == expected
    )


DEYE_SOLARMAN = ["Disabled", "Grid", "Generator", "Both"]  # this site's real options


@pytest.mark.parametrize(
    ("desired", "current", "expected"),
    [
        (True, "Disabled", "Grid"),
        (False, "Grid", "Disabled"),
        (True, "Generator", "Both"),  # generator charging is kept
        (False, "Both", "Generator"),
        (True, "Both", "Both"),
        (False, "Generator", "Generator"),
    ],
)
def test_select_option_on_deye_solarman_names(desired, current, expected):
    assert adapter.select_option(desired, current, DEYE_SOLARMAN) == expected


def test_select_option_overrides_and_unknown_names():
    options = ["Нет", "Сеть", "Ген"]
    assert adapter.select_option(True, "Нет", options) == "Сеть"
    assert adapter.select_option(False, "Сеть", options) == "Нет"
    assert adapter.select_option(True, None, ["A", "B"]) is None
    assert adapter.select_option(True, None, ["A", "B"], on_override="B") == "B"
    assert adapter.select_option(True, None, ["A", "B"], on_override="C") is None


def test_flag_state_and_service_by_domain():
    item = {
        "entity": "select.p1",
        "desired": True,
        "current": "Disabled",
        "options": ["Disabled", "Grid"],
    }
    assert adapter.flag_state(item, {}) == "Grid"
    assert adapter.flag_service("select.p1", "Grid") == (
        "select",
        "select_option",
        {"entity_id": "select.p1", "option": "Grid"},
    )
    assert adapter.flag_state({"entity": "switch.p1", "desired": False}, {}) == "off"
    assert adapter.flag_service("switch.p1", "off") == (
        "switch",
        "turn_off",
        {"entity_id": "switch.p1"},
    )
    assert adapter.flag_state({"entity": "number.p1", "desired": True}, {}) is None


def test_replan_interval_and_trigger(monkeypatch):
    assert main.replan_interval({}) == timedelta(minutes=120)
    assert main.replan_interval({"replan_minutes": 5}) == timedelta(minutes=15)
    monkeypatch.setattr(pipeline, "resolve_entities", lambda opts: {"grid_binary": ""})
    monkeypatch.setattr(main, "latest_report", lambda: None)
    main.STATE.update(
        last_trigger=None, last_run=(datetime.now(UTC) - timedelta(minutes=121)).isoformat()
    )
    assert main._trigger_reason({}) == "плановый пересчёт"
    main.STATE["last_run"] = (datetime.now(UTC) - timedelta(minutes=90)).isoformat()
    assert main._trigger_reason({}) is None  # within the two-hour default


def test_weather_archive_is_pruned(tmp_path):
    now = datetime.now(UTC)
    for days in (10, 8, 1, 0):
        stamp = (now - timedelta(days=days)).strftime("%Y%m%dT%H%M%SZ")
        (tmp_path / f"{stamp}.json").write_text("{}")
    (tmp_path / "seed.json").write_text("{}")
    assert pipeline.prune_weather_archive(tmp_path) == 2
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_routine_outage_log_reads_only_what_is_new(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    last = datetime.now(UTC) - timedelta(hours=1)
    monkeypatch.setattr(pipeline, "_last_observation", lambda: last)
    calls = []
    monkeypatch.setattr(
        pipeline, "fetch_history", lambda start, end, entities: calls.append(start) or {}
    )
    pipeline.update_outage_log({"grid_binary": "binary_sensor.grid", "mean_sensors": {}})
    assert calls and calls[0] >= last - timedelta(hours=2, seconds=1)


def _night_report():
    # Issued 23:00 UTC: one expensive hour, then cheap 00:00-07:00; the plan
    # (later is cheaper) charges 40 -> 60 % only in the last cheap hours.
    starts = [
        (datetime(2026, 11, 23, 23, tzinfo=UTC) + timedelta(hours=h)).isoformat() for h in range(10)
    ]
    soc = [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.45, 0.55, 0.60, 0.58]
    prices = [2.0] + [1.0] * 7 + [2.0] * 2
    return {
        "battery_now_pct": "40",
        "hourly": {"starts_utc": starts},
        "advisory": {"optimizer": {"soc_opt": soc, "prices": prices, "tariff_enabled": True}},
    }


def test_deye_charging_ahead_of_the_plan_at_night_is_not_a_deviation():
    report = _night_report()
    at = datetime(2026, 11, 24, 1, 30, tzinfo=UTC)  # Deye already at 60 %, plan at 40 %
    assert main.soc_deviation_reason(report, 60.0, at) is None
    # A shortfall at night still counts.
    assert main.soc_deviation_reason(report, 25.0, at).startswith("SOC 25%")


def test_the_plan_is_read_at_the_current_minute_by_day():
    report = _night_report()
    at = datetime(2026, 11, 24, 8, 30, tzinfo=UTC)  # 60 -> 58 % this hour: plan ~59 %
    assert main.soc_deviation_reason(report, 59.0, at) is None
    assert main.soc_deviation_reason(report, 75.0, at).startswith("SOC 75% вне плана 59%")
