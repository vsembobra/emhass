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


def test_priority_maps_reserve_quantile_and_allowed_modes():
    assert policy.priority({"priority_mode": "resilience"})["reserve_quantile"] == 0.95
    assert policy.priority({})["reserve_quantile"] == 0.9
    assert policy.priority({"priority_mode": "nonsense"}) == policy.PRIORITIES["balanced"]
    assert policy.priority({"priority_mode": "resilience"})["deficit_modes"] == ("conservative",)
    assert policy.default_deficit_mode({"priority_mode": "economy"}) == "moderate"
    assert policy.default_deficit_mode({}) == "conservative"


def test_deficit_inputs_per_mode():
    load = {"p10": [0.5], "p50": [1.0], "p90": [2.0]}
    pv = {"p10": [0.0], "p50": [1.0], "p90": [1.5]}
    assert [float(v[0]) for v in policy.deficit_inputs("conservative", load, pv)] == [2.0, 0.0]
    assert [float(v[0]) for v in policy.deficit_inputs("moderate", load, pv)] == [1.5, 0.5]
    assert [float(v[0]) for v in policy.deficit_inputs("expected", load, pv)] == [1.0, 1.0]
    with pytest.raises(ValueError):
        policy.deficit_inputs("yolo", load, pv)


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
