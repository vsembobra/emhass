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
    # SOC: measured before now, planned from the current hour on.
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


def test_without_a_plan_the_chart_says_so(site):
    data = charts.day_charts(None, now=NOW, live=False)
    assert data["plan_note"] and all(
        h["soc_plan_pct"] is None for day in data["days"] for h in day["hours"]
    )
