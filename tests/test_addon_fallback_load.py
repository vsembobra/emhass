"""Fallback load forecast: weekday-aware persistence before the models exist."""

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402

UTC = ZoneInfo("UTC")


def _audit(weeks=3):
    """Commercial site: 2.0 kW on weekdays, 0.5 kW on weekends, every hour.

    Starts on Monday 2026-10-05; three full weeks.
    """
    index = pd.date_range("2026-10-05", periods=weeks * 7 * 24, freq="h", tz="UTC")
    assert index[0].day_of_week == 0
    load = [2.0 if stamp.day_of_week < 5 else 0.5 for stamp in index]
    return pd.DataFrame({"load_kw": load, "load_filter": "keep"}, index=index)


def test_fallback_forecast_follows_the_weekday():
    audit = _audit()
    # Saturday then Monday, 10:00 UTC.
    starts = pd.DatetimeIndex(["2026-10-31 10:00", "2026-11-02 10:00"], tz="UTC")
    assert starts[0].day_of_week == 5 and starts[1].day_of_week == 0
    forecast = pipeline._fallback_load_prediction(audit, starts, UTC)
    p50 = forecast["load_p50_kw"]
    assert p50.iloc[0] == pytest.approx(0.5)  # Saturdays only
    assert p50.iloc[1] == pytest.approx(2.0)  # Mondays only
    # Bands keep the deliberate width around the weekday value.
    assert forecast["load_p10_kw"].iloc[0] == pytest.approx(0.3)
    assert forecast["load_p90_kw"].iloc[1] == pytest.approx(3.0)


def test_fallback_forecast_falls_back_to_the_hour_mean_for_a_thin_weekday():
    audit = _audit()
    # Only one clean Monday 10:00 survives the filter: not enough to trust.
    mondays_10 = [
        stamp
        for stamp in audit.index
        if stamp.day_of_week == 0 and stamp.hour == 10
    ]
    audit.loc[mondays_10[:2], "load_filter"] = "grid_quality"
    starts = pd.DatetimeIndex(["2026-11-02 10:00"], tz="UTC")  # a Monday
    forecast = pipeline._fallback_load_prediction(audit, starts, UTC)
    # Two dropped Monday samples leave 19 clean hours of 10:00 with sum 29.
    assert forecast["load_p50_kw"].iloc[0] == pytest.approx(29 / 19)


def test_fallback_forecast_without_history_is_flat():
    audit = pd.DataFrame({"load_kw": [], "load_filter": []}, index=pd.DatetimeIndex([], tz="UTC"))
    starts = pd.date_range("2026-11-02 10:00", periods=3, freq="h", tz="UTC")
    forecast = pipeline._fallback_load_prediction(audit, starts, UTC)
    assert (forecast["load_p50_kw"] == 0.5).all()
