"""Weather history must backfill ranges BEFORE the stored file.

The forward-only cursor never filled days added to the audit later
(recorder backfill), leaving plane-of-array NaN for every old hour and
starving the PV model ("at least 24 usable daylight rows are required").
"""

import sys
from pathlib import Path

import pandas as pd

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import retrain  # noqa: E402

SITE = {"latitude": 45.0, "longitude": 34.0}


def _chunk(start, end):
    hours = pd.date_range(start, end, freq="h", tz="UTC")
    return pd.DataFrame(
        {"shortwave_radiation_previous_day1": 100.0}, index=hours
    )


def test_backfills_earlier_range(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "WEATHER_HIST", tmp_path / "weather.jsonl")
    # Stored file covers only the most recent day.
    recent = _chunk(pd.Timestamp("2026-11-19", tz="UTC"), pd.Timestamp("2026-11-19 23:00", tz="UTC"))
    recent.reset_index(names="start_utc").to_json(
        tmp_path / "weather.jsonl", orient="records", lines=True, date_format="iso"
    )
    fetched = []

    def fake_fetch(site, start, end):
        fetched.append((start, end))
        return _chunk(start, end)

    monkeypatch.setattr(retrain, "_fetch_weather_chunk", fake_fetch)
    needed = pd.Timestamp("2026-11-06", tz="UTC")
    weather = retrain._ensure_weather_history(SITE, needed)
    assert weather.index[0] == needed
    assert fetched, "the earlier range was never fetched"
    assert fetched[0][0] == needed
    # Stored rows survive; the union has no gaps between needed and file end.
    assert weather.index[-1] >= pd.Timestamp("2026-11-19 23:00", tz="UTC")


def test_no_refetch_when_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "WEATHER_HIST", tmp_path / "weather.jsonl")
    # The history is complete when it reaches "now minus a day" (the archive
    # publishes with a delay), so the end must follow the clock.
    fetch_end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=1)
    full = _chunk(pd.Timestamp("2026-11-06", tz="UTC"), fetch_end)
    full.reset_index(names="start_utc").to_json(
        tmp_path / "weather.jsonl", orient="records", lines=True, date_format="iso"
    )
    monkeypatch.setattr(
        retrain,
        "_fetch_weather_chunk",
        lambda site, start, end: (_ for _ in ()).throw(AssertionError("refetch")),
    )
    weather = retrain._ensure_weather_history(SITE, pd.Timestamp("2026-11-06", tz="UTC"))
    assert len(weather) == len(full)
