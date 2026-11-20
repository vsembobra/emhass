"""Open-Meteo temperature backstops audit gaps; observed values win."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import retrain  # noqa: E402


def _frames():
    idx = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    audit = pd.DataFrame(
        {"temp_air_observed_c": [5.0, np.nan, np.nan, np.nan], "load_kw": [1, 1, 1, 1]},
        index=idx,
    )
    weather = pd.DataFrame(
        {"temperature_2m_previous_day1": [10.0, 11.0, 12.0, 13.0]}, index=idx
    )
    return audit, weather


def test_fills_only_missing_values():
    audit, weather = _frames()
    out = retrain._fill_audit_temperature(audit, weather)
    assert list(out["temp_air_observed_c"]) == [5.0, 11.0, 12.0, 13.0]
    # исходный фрейм не тронут
    assert audit["temp_air_observed_c"].isna().sum() == 3


def test_no_weather_keeps_audit_unchanged():
    audit, _ = _frames()
    out = retrain._fill_audit_temperature(audit, None)
    assert out["temp_air_observed_c"].isna().sum() == 3
