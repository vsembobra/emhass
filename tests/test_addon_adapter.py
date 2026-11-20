"""Inverter capabilities: zone count, SOC step, per-zone grid-charge policy."""

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import adapter  # noqa: E402
import optimize  # noqa: E402
import pipeline  # noqa: E402

UTC = ZoneInfo("UTC")
STARTS = pd.date_range("2026-11-20 00:00", periods=24, freq="h", tz="UTC")
PROGRAMS = [
    {"n": 1, "from": "00:00", "soc_pct": 30},
    {"n": 2, "from": "07:00", "soc_pct": 30},
    {"n": 3, "from": "17:00", "soc_pct": 30},
]


def test_caps_defaults_match_deye():
    caps = adapter.caps({})
    assert (caps.zones, caps.soc_step_pct, caps.controls_grid_charge) == (6, 5, False)
    assert caps.ceil_soc(0.41) == pytest.approx(0.45)
    assert caps.grid_charge_entities() == []


def test_caps_from_options():
    caps = adapter.caps(
        {
            "tou_zones": 4,
            "tou_soc_step_pct": 10,
            "tou_grid_charge_entity_template": "switch.p{n}_grid",
        }
    )
    assert caps.ceil_soc(0.41) == pytest.approx(0.5)
    assert caps.grid_charge_entities() == [f"switch.p{n}_grid" for n in range(1, 5)]


def test_resolve_entities_follows_zone_count():
    base = {
        "entity_load_power": "a",
        "entity_critical_load_power": "b",
        "entity_pv_power": "c",
        "entity_battery_power": "d",
        "entity_grid_power": "e",
        "entity_outdoor_temperature": "",
        "entity_battery_soc": "f",
        "entity_export_counter": "g",
        "entity_grid_status": "h",
        "tou_soc_entity_template": "number.z{n}_soc",
        "tou_time_entity_template": "time.z{n}",
        "floor_input_entity": "input_number.floor",
        "tou_zones": 3,
        "tou_grid_charge_entity_template": "switch.z{n}_grid",
    }
    ent = pipeline.resolve_entities(base)
    assert ent["tou_soc"] == ["number.z1_soc", "number.z2_soc", "number.z3_soc"]
    assert ent["tou_grid_charge"] == ["switch.z1_grid", "switch.z2_grid", "switch.z3_grid"]


def test_zone_targets_round_to_the_configured_step():
    soc_opt = [0.42] * 24
    out = optimize.zone_targets(soc_opt, 0.42, PROGRAMS, STARTS, UTC, 0.1, 1.0, soc_step_pct=10)
    assert {t["optimal_soc_pct"] for t in out} == {50}


def _plan(opts, reserve=0.2, soc_opt=None, pv=None):
    pv = pv if pv is not None else [0.0] * 7 + [2.0] * 10 + [0.0] * 7
    return {
        item["n"]: (item["desired"], item["reason"])
        for item in adapter.grid_charge_plan(
            PROGRAMS, STARTS, UTC, opts, [reserve] * 24, pv, soc_opt or [0.5] * 24, 0.5
        )
    }


def test_grid_charge_follows_the_cheap_tariff():
    plan = _plan({"tariff_enabled": True, "tariff_cheap_windows": "00:00-07:00"})
    assert plan[1] == (True, "дешёвый тариф")
    assert plan[2] == (False, "дорогой тариф")
    assert plan[3] == (False, "дорогой тариф")


def test_reserve_needs_override_the_price():
    plan = _plan({"tariff_enabled": True, "tariff_cheap_windows": "00:00-07:00"}, reserve=0.7)
    assert all(desired for desired, _ in plan.values())
    assert plan[3][1].startswith("резерв 70%")


def test_one_price_all_day_never_charges_from_grid_without_reserve_need():
    plan = _plan({"tariff_enabled": False})
    assert all(not desired for desired, _ in plan.values())
    assert plan[1][1] == "цена одинаковая — заряд только от солнца"


def test_one_price_still_restores_the_series_reserve():
    plan = _plan({"tariff_enabled": False}, reserve=0.7)
    assert all(desired for desired, _ in plan.values())


def test_active_zone_grid_charge_uses_measured_soc():
    """Review case: 01:00, zone 00:00-07:00 in effect, SOC 50 %, reserve 70 %."""
    starts = pd.date_range("2026-11-20 01:00", periods=24, freq="h", tz="UTC")
    programs = [{"n": 1, "from": "00:00", "soc_pct": 50}, {"n": 2, "from": "07:00", "soc_pct": 50}]
    soc_opt = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8] + [0.7] * 17 + [0.75]
    reserve = [0.7] * 6 + [0.2] * 18
    opts = {"tariff_enabled": True, "tariff_cheap_windows": "02:00-05:00"}
    (zone1, zone2) = adapter.grid_charge_plan(
        programs, starts, UTC, opts, reserve, [0.0] * 24, soc_opt, 0.5
    )
    # Was off as "дорогой тариф": compared against tomorrow's planned 70 %.
    assert zone1["desired"] and zone1["reason"].startswith("резерв 70%")
    assert not zone2["desired"]
