"""Per-day energy statistics for the panel, computed from the hourly audit.

The key subtlety is PV curtailment censoring: with no grid feed-in, a full
battery makes the inverter throttle the panels to match the load, so measured
PV under-reports what the array could have produced. Hours flagged
``possible_curtailment`` by the audit are therefore shown both as measured
and as model-estimated potential, and the difference is the energy lost to
curtailment. Sign conventions verified against the site data: battery power
is negative while charging, grid power is positive while importing.
"""

import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pipeline

from emhass.pv_quantiles import try_load

DATA = Path(os.environ.get("EMHASS_SHADOW_DATA", "/data"))
WEATHER_HIST = DATA / "weather_previous_day1.jsonl"
MODELS = DATA / "models"

def _potential_pv(bundle, hours: pd.DatetimeIndex) -> pd.Series:
    """Model P50 for the given hours; NaN where as-of weather is missing."""
    if not WEATHER_HIST.exists():
        return pd.Series(np.nan, index=hours)
    weather = pd.read_json(WEATHER_HIST, lines=True)
    weather["start_utc"] = pd.to_datetime(weather["start_utc"], utc=True)
    weather = weather.set_index("start_utc").sort_index()
    weather = weather[~weather.index.duplicated(keep="last")]
    frame = pipeline.pv_weather_frame(bundle.geometry, weather, hours, "_previous_day1")
    frame = frame.dropna(subset=["poa_forecast_kw"])
    if frame.empty:
        return pd.Series(np.nan, index=hours)
    prediction = bundle.predict(frame)["pv_p50_kw"]
    prediction.index = prediction.index - pipeline.HALF_HOUR  # back to hour starts
    return prediction.reindex(hours)


def daily_stats(days: int = 14) -> list[dict]:
    audit_path = DATA / "hourly_audit.jsonl"
    if not audit_path.exists():
        return []
    opts = json.loads((DATA / "options.json").read_text())
    tz = ZoneInfo(opts.get("time_zone", "UTC"))
    audit = pd.read_json(audit_path, lines=True)
    # Writers disagree on spelling ("…Z" vs "+00:00"); mixed parses both.
    audit["start_utc"] = pd.to_datetime(audit["start_utc"], utc=True, format="mixed")
    audit = audit.set_index("start_utc").sort_index()

    today = pd.Timestamp.now(tz).normalize()
    first_day = today - pd.Timedelta(days=days - 1)
    audit = audit[audit.index >= first_day.tz_convert("UTC")]
    if audit.empty:
        return []

    bundle = try_load(MODELS / pipeline.PV_MODEL)
    curtailed = audit[audit["pv_filter"] == "possible_curtailment"]
    potential = None
    if bundle is not None and not curtailed.empty:
        potential = _potential_pv(bundle, curtailed.index)

    rows = []
    for day in pd.date_range(first_day, today, freq="D", tz=tz):
        chunk = audit[audit.index.tz_convert(tz).normalize() == day]
        if chunk.empty:
            continue
        load = pd.to_numeric(chunk["load_kw"], errors="coerce")
        pv = pd.to_numeric(chunk["pv_measured_kw"], errors="coerce")
        batt = pd.to_numeric(chunk["battery_power_kw_raw_sign"], errors="coerce")
        grid = pd.to_numeric(chunk["grid_power_kw_raw_sign"], errors="coerce")
        cur_hours = chunk.index[chunk["pv_filter"] == "possible_curtailment"]
        lost = None
        if potential is not None and len(cur_hours):
            # Loss is the part of the potential ABOVE what was actually
            # produced; summing the raw potential would count production too.
            values = potential.loc[cur_hours]
            measured = pd.to_numeric(chunk.loc[cur_hours, "pv_measured_kw"], errors="coerce")
            diff = (values - measured).clip(lower=0)
            if diff.notna().any():
                lost = float(diff.sum(min_count=1))
        rows.append(
            {
                "date": day.date().isoformat(),
                "load_kwh": round(float(load.sum(min_count=1)), 2)
                if load.notna().any()
                else None,
                "pv_measured_kwh": round(float(pv.sum(min_count=1)), 2)
                if pv.notna().any()
                else None,
                "pv_curtailed_hours": int(len(cur_hours)),
                "pv_curtailment_loss_kwh": round(lost, 2) if lost is not None else None,
                "battery_charge_kwh": round(abs(float(batt[batt < 0].sum())), 2),
                "battery_discharge_kwh": round(float(batt[batt > 0].sum()), 2),
                "grid_import_kwh": round(float(grid[grid > 0].sum()), 2),
                "min_soc_pct": round(float(pd.to_numeric(chunk["soc_mean"], errors="coerce").min() * 100), 1)
                if chunk["soc_mean"].notna().any()
                else None,
                "grid_down_hours": int((chunk["raw_grid_status"] == "outage").sum()),
            }
        )
    return rows
