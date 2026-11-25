"""Data behind the panel's today/tomorrow charts: forecast, actual, SOC."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import charts  # noqa: E402
import learning  # noqa: E402
import pipeline  # noqa: E402

NOW = datetime(2026, 11, 21, 12, 30, tzinfo=UTC)  # 12:30 local (UTC site)


def _entry(issue, pv):
    starts = pd.date_range(issue, periods=48, freq="h")
    return learning.log_entry(
        issue,
        starts,
        ([0.4] * 48, [0.5] * 48, [0.6] * 48),
        "model",
        ([pv * 0.5] * 48, [pv] * 48, [pv * 1.5] * 48),
        "model",
    )


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    (tmp_path / "options.json").write_text(json.dumps({"time_zone": "UTC"}))
    hours = pd.date_range("2026-11-21 00:00", "2026-11-21 11:00", freq="h", tz="UTC")
    rows = [
        {
            "start_utc": h.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "pv_measured_kw": 2.0 if h.hour == 11 else 1.0,
            "soc_mean": 0.5,
            "pv_filter": "possible_curtailment" if h.hour == 11 else "keep",
        }
        for h in hours
    ]
    (tmp_path / "hourly_audit.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    log = [
        _entry(pd.Timestamp("2026-11-21 00:00", tz="UTC"), 1.0),
        _entry(pd.Timestamp("2026-11-21 10:00", tz="UTC"), 3.0),
    ]
    (tmp_path / "forecast_log.jsonl").write_text("\n".join(json.dumps(e) for e in log) + "\n")
    report = {
        "issue_utc": "2026-11-21T12:00:00+00:00",
        "advisory": {"optimizer": {"soc_opt": [0.6, 0.7] + [0.8] * 46}},
    }
    return report


def test_today_shows_the_forecast_in_effect_and_the_measurements(site):
    data = charts.day_charts(site, now=NOW, live=False)
    today, tomorrow = data["days"]
    assert (today["label"], tomorrow["label"]) == ("Сегодня", "Завтра")
    hours = today["hours"]
    assert len(hours) == 24 and hours[12]["current"] and hours[11]["past"]
    # 09:00 used the 00:00 run; 11:00 and later the 10:00 run.
    assert hours[9]["forecast"]["p50"] == 1.0
    assert hours[11]["forecast"] == {"p10": 1.5, "p50": 3.0, "p90": 4.5}
    assert hours[9]["actual_kw"] == 1.0 and not hours[9]["throttled"]
    assert hours[11]["throttled"]  # a full battery throttled the panels
    assert hours[13]["actual_kw"] is None
    # SOC: measured before now; planned wherever a plan covered the hour
    # (no archived plans here, so past hours have none yet).
    assert hours[10]["soc_pct"] == 50.0 and hours[10]["soc_plan_pct"] is None
    assert hours[12]["soc_plan_pct"] == 60.0 and hours[13]["soc_plan_pct"] == 65.0
    assert today["actual_kwh"] == 13.0  # 11 h x 1 kW + 2 kW
    assert today["ahead_kwh"] == pytest.approx(3.0 * 0.5 + 3.0 * 11, abs=0.1)


def test_tomorrow_is_forecast_and_plan_only(site):
    tomorrow = charts.day_charts(site, now=NOW, live=False)["days"][1]
    assert all(h["actual_kw"] is None for h in tomorrow["hours"])
    assert tomorrow["hours"][0]["forecast"]["p50"] == 3.0
    assert tomorrow["hours"][0]["soc_plan_pct"] == 80.0
    assert tomorrow["forecast_kwh"] == pytest.approx(3.0 * 24)


def test_consumption_is_aligned_with_solar_without_fabricating_measurements(site, tmp_path):
    path = tmp_path / "hourly_audit.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[9]["load_kw"] = 1.7
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    days = charts.day_charts(site, now=NOW, live=False)["days"]
    assert days[0]["hours"][9]["load_actual_kw"] == 1.7
    assert days[0]["hours"][8]["load_actual_kw"] is None
    assert days[0]["hours"][9]["load_forecast_kw"] == 0.5
    assert days[1]["hours"][0]["load_forecast_kw"] == 0.5
    assert all(h["load_actual_kw"] is None for h in days[1]["hours"])


def test_without_a_plan_the_chart_says_so(site):
    data = charts.day_charts(None, now=NOW, live=False)
    assert data["plan_note"] and all(
        h["soc_plan_pct"] is None for day in data["days"] for h in day["hours"]
    )


def test_compact_history_gives_identical_chart_data(site, tmp_path):
    before = charts.day_charts(site, now=NOW, live=False)
    charts.data_store.compact_data(tmp_path)
    assert charts.day_charts(site, now=NOW, live=False) == before


def test_past_hours_show_the_plan_that_covered_them(site, tmp_path):
    # Yesterday's evening plan covered today's early hours: the chart must
    # show it next to the measured SOC instead of hiding it in the past.
    archive = tmp_path / "shadow_reports"
    archive.mkdir()
    (archive / "2026-11-20.json").write_text(
        json.dumps(
            {
                "issue_utc": "2026-11-20T20:00:00+00:00",
                "advisory": {"optimizer": {"soc_opt": [0.4] * 48}},
            }
        )
    )
    hours = charts.day_charts(site, now=NOW, live=False)["days"][0]["hours"]
    assert hours[5]["soc_plan_pct"] == 40.0  # from the archived plan
    assert hours[10]["soc_pct"] == 50.0 and hours[10]["soc_plan_pct"] == 40.0
    # Since the live report's issue (12:00) its own plan wins again.
    assert hours[12]["soc_plan_pct"] == 60.0
    # A duplicated archive of the live report changes nothing.
    (archive / "2026-11-21.json").write_text(json.dumps(site))
    assert charts.day_charts(site, now=NOW, live=False)["days"][0]["hours"][12][
        "soc_plan_pct"
    ] == 60.0


@pytest.fixture
def site_tariff(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    (tmp_path / "options.json").write_text(
        json.dumps(
            {
                "time_zone": "UTC",
                "tariff_enabled": True,
                "tariff_cheap_windows": "23:00-07:00",
            }
        )
    )
    hours = pd.date_range("2026-11-21 00:00", "2026-11-21 11:00", freq="h", tz="UTC")
    rows = [
        {
            "start_utc": h.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "pv_measured_kw": 1.0,
            "soc_mean": 0.5,
            "grid_power_kw_raw_sign": 0.8,
            "raw_grid_status": "outage" if h.hour == 3 else "ok",
            "grid_quality": "recorded_outage" if h.hour == 3 else "ok",
        }
        for h in hours
    ]
    (tmp_path / "hourly_audit.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return {
        "issue_utc": "2026-11-21T12:00:00+00:00",
        "advisory": {
            "floor_profile": [
                {"start_local": "12:00", "floor_soc": 0.35}
            ] * 24,
            "optimizer": {"soc_opt": [0.6] * 48, "p_grid_kw": [1.5] * 48},
        },
    }


def test_tariff_outage_and_plan_layers_reach_the_hours(site_tariff):
    data = charts.day_charts(site_tariff, now=NOW, live=False)
    today, tomorrow = data["days"]
    hours = today["hours"]
    # Cheap window 23:00-07:00 wraps midnight: early hours cheap, noon not.
    assert hours[2]["cheap"] and hours[6]["cheap"]
    assert not hours[12]["cheap"]
    assert tomorrow["hours"][2]["cheap"]  # the window repeats every day
    # The outage hour is flagged from the audit's grid status.
    assert hours[3]["outage"] and not hours[4]["outage"]
    # Floor and grid plans stitched from the report cover plan hours.
    assert hours[12]["floor_plan_pct"] == 35.0 and hours[12]["grid_plan_kw"] == 1.5
    assert tomorrow["hours"][0]["floor_plan_pct"] == 35.0
    # Measured grid import reaches the past hours.
    assert hours[3]["grid_actual_kw"] == 0.8
    assert hours[12]["grid_actual_kw"] is None  # future
