"""TOU zone targets never fall below the highest outage reserve of their window."""

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import optimize  # noqa: E402
import pipeline  # noqa: E402

TZ = ZoneInfo("UTC")
STARTS = pd.date_range("2026-11-15 00:00", periods=6, freq="h", tz="UTC")
PROGRAMS = [
    {"n": 1, "from": "00:00", "soc_pct": 50.0},
    {"n": 2, "from": "03:00", "soc_pct": 50.0},
]


def _targets(soc_opt, reserve_floors=None, soc_init=0.7):
    out = optimize.zone_targets(
        soc_opt, soc_init, PROGRAMS, STARTS, TZ, 0.1, 1.0, reserve_floors=reserve_floors
    )
    return {t["n"]: t["optimal_soc_pct"] for t in out}


def test_legacy_targets_unchanged_without_reserve():
    # Zone 1 discharges 70 -> 30 %: target is the window minimum, rounded up.
    assert _targets([0.6, 0.4, 0.3, 0.3, 0.3, 0.3])[1] == 30


def test_discharging_zone_raised_to_risk_hour_reserve():
    # A risk hour opens zone 1 (index 0), the plan drains later in the zone.
    # Target 30 % would let a heavy load drain the risky hour below 50 %.
    reserve = [0.5, 0.2, 0.2, 0.2, 0.2, 0.2]
    assert _targets([0.6, 0.4, 0.3, 0.3, 0.3, 0.3], reserve)[1] == 50


def test_v_shaped_zone_protects_end_of_window_reserve():
    # SOC dips mid-zone and the plan counts on PV to refill it; the inverter
    # holds the dip value all zone long unless the reserve bounds it.
    soc_opt = [0.6, 0.3, 0.55, 0.55, 0.55, 0.55]
    reserve = [0.2, 0.25, 0.5, 0.2, 0.2, 0.2]
    assert _targets(soc_opt, reserve, soc_init=0.6)[1] == 50


def test_reserve_never_lowers_a_target():
    # Zone 2 charges 30 -> 80 %: the charge target stays above the reserve.
    soc_opt = [0.6, 0.4, 0.3, 0.5, 0.7, 0.8]
    reserve = [0.2] * 6
    assert _targets(soc_opt, reserve)[2] == 80


def test_reserve_clamped_to_max_soc():
    out = optimize.zone_targets(
        [0.5] * 6, 0.5, PROGRAMS, STARTS, TZ, 0.1, 0.92, reserve_floors=[0.92] * 6
    )
    assert all(t["optimal_soc_pct"] <= 92 for t in out)


def test_heuristic_zone_uses_highest_reserve_in_window():
    starts = pd.date_range("2026-11-15 09:00", periods=24, freq="h", tz="UTC")
    programs = [
        {"n": 1, "from": "09:00", "soc_pct": 20.0},
        {"n": 2, "from": "17:00", "soc_pct": 20.0},
    ]
    floors = [0.2] * 24  # daily low at 09:00 — the old value of zone 1
    reserve = [0.15] * 24
    reserve[7] = 0.43  # 16:00, still inside zone 1 (09:00-17:00)
    pipeline.recommend_zone_socs(programs, floors, reserve, starts, TZ)
    assert programs[0]["recommended_soc_pct"] == 45
    assert programs[1]["recommended_soc_pct"] == 20


def test_heuristic_zone_missing_start_hour():
    starts = pd.date_range("2026-11-15 09:00", periods=3, freq="h", tz="UTC")
    programs = [{"n": 1, "from": "20:00", "soc_pct": 20.0}]
    pipeline.recommend_zone_socs(programs, [0.2] * 3, [0.2] * 3, starts, TZ)
    assert programs[0]["recommended_soc_pct"] is None


def test_zone_windows_wrap_around_horizon_edge():
    starts = pd.date_range("2026-11-15 01:00", periods=24, freq="h", tz="UTC")
    programs = [{"n": 1, "from": "00:00", "soc_pct": 0}, {"n": 2, "from": "07:00", "soc_pct": 0}]
    windows = optimize.zone_windows(programs, starts, TZ)
    match, window, forward_end = windows[1]
    assert match == 23  # tomorrow 00:00
    assert window == [23, 0, 1, 2, 3, 4, 5]  # 00:00 tomorrow, then 01:00-07:00 today
    assert forward_end == 24
