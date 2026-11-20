"""Replay tuner and plan-vs-fact attribution."""

import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import learning  # noqa: E402
import pipeline  # noqa: E402
import tuner  # noqa: E402

TZ = ZoneInfo("UTC")
DAY0 = pd.Timestamp("2026-11-01", tz="UTC")
OPTS = {
    "capacity_kwh": 30.0,
    "min_soc_pct": 10.0,
    "max_soc_pct": 100.0,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "max_charge_kw": 5.0,
    "max_discharge_kw": 5.0,
    "tariff_enabled": False,
    "time_zone": "UTC",
    "priority_mode": "balanced",
}


def _site(days=6, outage=None):
    """Sunny days (3 kW 08-16), 0.5 kW load; optional outage (start, hours, kW)."""
    index = pd.date_range(DAY0, periods=days * 24, freq="h")
    sunny = (index.hour >= 8) & (index.hour < 16)
    audit = pd.DataFrame(
        {
            "pv_measured_kw": [3.0 if s else 0.0 for s in sunny],
            "load_kw": 0.5,
            "critical_load_kw": 0.5,
            "soc_mean": 0.5,
            "raw_grid_status": "up",
        },
        index=index,
    )
    if outage is not None:
        start, hours, kw = outage
        hit = (index >= start) & (index < start + pd.Timedelta(hours=hours))
        audit.loc[hit, ["load_kw", "critical_load_kw"]] = kw
        audit.loc[hit, "raw_grid_status"] = "outage"
    log = []
    for issue in pd.date_range(DAY0, periods=days * 2, freq="12h"):
        starts = pd.date_range(issue, periods=48, freq="h")
        sun = [(s.hour >= 8) & (s.hour < 16) for s in starts]
        pv = (
            [1.0 if s else 0.0 for s in sun],
            [3.0 if s else 0.0 for s in sun],
            [3.5 if s else 0.0 for s in sun],
        )
        load = ([0.3] * 48, [0.5] * 48, [1.5] * 48)
        log.append(learning.log_entry(issue, starts, load, "model", pv, "model", [1.0] * 48))
    return audit, log


NOW = DAY0 + pd.Timedelta(days=6)


def test_replay_windows_run_noon_to_noon_and_need_complete_hours():
    audit, _ = _site()
    audit.loc[DAY0 + pd.Timedelta(days=2, hours=15), "load_kw"] = float("nan")
    windows = tuner.replay_windows(audit, TZ, DAY0, NOW)
    starts = [w[0]["start"] for w in windows]
    assert all(s.hour == 12 for s in starts)
    assert all(len(w) == 24 for w in windows)
    assert datetime(2026, 11, 3, 12, tzinfo=UTC) not in starts  # contains the NaN hour
    assert len(windows) == 4  # 01-12..05-12 minus the broken one


def test_without_outages_the_cheaper_mode_wins():
    audit, log = _site()
    result = tuner.tune(audit, log, OPTS, TZ, NOW)
    candidates = result["candidates"]
    assert set(candidates) == {"conservative", "moderate"}
    assert candidates["moderate"]["unserved_critical_kwh"] == 0.0
    assert candidates["moderate"]["net_cost"] < candidates["conservative"]["net_cost"]
    assert result["best"] == "moderate"


def test_a_long_outage_makes_the_conservative_mode_win():
    outage = (DAY0 + pd.Timedelta(days=3, hours=18), 12, 2.0)
    audit, log = _site(outage=outage)
    result = tuner.tune(audit, log, OPTS, TZ, NOW)
    candidates = result["candidates"]
    assert (
        candidates["conservative"]["unserved_critical_kwh"]
        < candidates["moderate"]["unserved_critical_kwh"]
    )
    assert result["best"] == "conservative"


def test_resilience_only_considers_the_conservative_mode():
    audit, log = _site()
    result = tuner.tune(audit, log, dict(OPTS, priority_mode="resilience"), TZ, NOW)
    assert list(result["candidates"]) == ["conservative"] and result["best"] == "conservative"


def test_too_little_history_gives_no_choice():
    audit, log = _site(days=2)
    result = tuner.tune(audit, log, OPTS, TZ, DAY0 + pd.Timedelta(days=2))
    assert result["best"] is None and "reason" in result


def test_rank_puts_outage_service_before_money():
    safe = {"unserved_critical_kwh": 0.0, "curtailed_kwh": 5.0, "net_cost": 100.0}
    cheap = {"unserved_critical_kwh": 1.0, "curtailed_kwh": 0.0, "net_cost": 10.0}
    assert tuner.rank_key(safe) < tuner.rank_key(cheap)


def test_tuned_mode_needs_autotune_and_an_allowed_choice():
    tuning = {"best": "moderate"}
    assert tuner.tuned_mode(OPTS, tuning) == "conservative"
    assert tuner.tuned_mode(dict(OPTS, autotune=True), tuning) == "moderate"
    resilient = dict(OPTS, autotune=True, priority_mode="resilience")
    assert tuner.tuned_mode(resilient, tuning) == "conservative"


def test_tuning_file_roundtrip(tmp_path):
    assert tuner.due(tmp_path)
    tuner.save(tmp_path, {"tuned_utc": datetime.now(UTC).isoformat(), "best": "moderate"})
    assert not tuner.due(tmp_path) and tuner.load(tmp_path)["best"] == "moderate"


def test_hindsight_import_splits_forecast_and_execution():
    starts = pd.date_range("2026-11-10", periods=24, freq="h", tz="UTC")
    audit = pd.DataFrame({"load_kw": 1.0, "pv_measured_kw": 0.0, "soc_mean": 0.5}, index=starts)
    report = {
        "hourly": {"load_p50_kw": [0.5] * 24, "pv_p50_kw": [0.0] * 24},
        "advisory": {"floor_profile": [{"floor_soc": 0.1}] * 24},
    }
    opts = dict(
        OPTS, latitude=45.0, longitude=34.0, tariff_normal_price=1.0, tariff_cheap_price=1.0
    )
    hindsight = pipeline.hindsight_import(report, audit, starts, opts)
    # 24 kWh of load, 30 kWh x (0.5 - 0.1) x 0.95 = 11.4 kWh from the battery.
    assert hindsight == pytest.approx(24.0 - 11.4, abs=0.5)


def test_plan_vs_fact_note_shows_the_split(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "METRICS", tmp_path / "metrics.jsonl")
    entry = {
        "date": "2026-11-10",
        "plan_vs_fact": {
            "planned_grid_kwh": 10.0,
            "actual_grid_kwh": 14.0,
            "hindsight_grid_kwh": 13.0,
            "forecast_effect_kwh": 3.0,
            "execution_effect_kwh": 1.0,
        },
    }
    (tmp_path / "metrics.jsonl").write_text(
        __import__("json").dumps(entry) + "\n", encoding="utf-8"
    )
    note = pipeline._plan_vs_fact_note()
    assert "из-за прогноза +3.0" in note and "из-за исполнения +1.0" in note
