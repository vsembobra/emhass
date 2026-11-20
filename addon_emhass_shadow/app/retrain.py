"""Periodic self-retraining of the quantile models inside the add-on.

Runs on the data the add-on collects itself: the hourly audit (load and PV
actuals plus quality filters) and the Open-Meteo previous-runs history
(day-ahead weather as it was predicted, matching the inference contract and
avoiding reanalysis leakage).

Every retrain fits on all history except the last CALIBRATION_DAYS days,
calibrates conformal offsets on that trailing slice, smoke-tests the saved
bundle, and only then atomically replaces the live model (previous version
kept as <name>.prev). Failures never touch the running models: the daily plan
keeps using the last good bundle and the status records what happened.
"""

import json
import os
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pipeline

from emhass.load_quantiles import LoadQuantileBundle
from emhass.pv_quantiles import PvGeometry, PvQuantileBundle, plane_power_kw

DATA = Path(os.environ.get("EMHASS_SHADOW_DATA", "/data"))
SHARE = Path(os.environ.get("EMHASS_SHADOW_SHARE", "/share/emhass_shadow"))
STATE = DATA / "retrain_state.json"
WEATHER_HIST = DATA / "weather_previous_day1.jsonl"
MODELS = DATA / "models"

CALIBRATION_DAYS = 14
FETCH_CHUNK_DAYS = 30

WEATHER_VARIABLES = (
    "shortwave_radiation_previous_day1",
    "direct_normal_irradiance_previous_day1",
    "diffuse_radiation_previous_day1",
    "cloud_cover_previous_day1",
    "temperature_2m_previous_day1",
)


def _load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {"last_retrain_utc": None, "models": {}}


def _save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        SHARE.mkdir(parents=True, exist_ok=True)
        (SHARE / "retrain_state.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def retrain_status() -> dict:
    state = _load_state()
    interval = _interval_days()
    enabled = _enabled()
    last = state.get("last_retrain_utc")
    nxt = None
    if last and enabled:
        nxt = (datetime.fromisoformat(last) + timedelta(days=interval)).isoformat()
    return {
        "last_retrain_utc": last,
        "next_retrain_utc": nxt,
        "interval_days": interval,
        "enabled": enabled,
        "models": state.get("models", {}),
    }


def _interval_days() -> int:
    try:
        opts = json.loads((DATA / "options.json").read_text())
        return max(1, int(opts.get("retrain_interval_days", 7)))
    except (OSError, json.JSONDecodeError, ValueError):
        return 7


def _enabled() -> bool:
    try:
        opts = json.loads((DATA / "options.json").read_text())
        return bool(opts.get("retrain_enabled", True))
    except (OSError, json.JSONDecodeError):
        return True


DRIFT_MIN_RETRAIN_GAP = timedelta(hours=24)


def _drift_detected() -> bool:
    try:
        return bool(json.loads((DATA / "drift_state.json").read_text(encoding="utf-8")).get("drift"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def _audit_frame() -> pd.DataFrame:
    audit = pd.read_json(DATA / "hourly_audit.jsonl", lines=True)
    audit["start_utc"] = pd.to_datetime(audit["start_utc"], utc=True)
    audit = audit.set_index("start_utc").sort_index()
    audit = audit[~audit.index.duplicated(keep="last")]
    full = pd.date_range(audit.index[0], audit.index[-1], freq="h", tz=UTC)
    return audit.reindex(full)


def _swap_in(tmp: Path, final: Path) -> None:
    prev = final.with_name(final.name + ".prev")
    if prev.exists():
        for old in prev.iterdir():
            old.unlink()
        prev.rmdir()
    if final.exists():
        final.rename(prev)
    tmp.rename(final)


def _retrain_load(audit: pd.DataFrame) -> dict:
    final = MODELS / "load_quantiles_v1"
    if (final / "bundle.json").exists():
        current = LoadQuantileBundle.load(final)
        config, seasonal = current.config, current.seasonal
    else:
        # Fresh install: fit from scratch on whatever history has accrued.
        config, seasonal = None, None
    grid = pd.DataFrame(index=audit.index)
    grid["temp_air_observed_c"] = pd.to_numeric(audit["temp_air_observed_c"], errors="coerce")
    load = pd.to_numeric(audit["load_kw"], errors="coerce")
    # Only rows the audit marked clean are training targets; outage and
    # invalid hours would teach the model the wrong consumption pattern.
    grid["load_kw"] = load.where(audit["load_filter"] == "keep")
    split = grid.index[-1] - pd.Timedelta(days=CALIBRATION_DAYS)
    train, calib = grid.loc[:split], grid.loc[split + pd.Timedelta(hours=1) :]
    if train["load_kw"].notna().sum() < 48:
        train, calib = grid, grid.loc[grid.index[-24:]]
    bundle = LoadQuantileBundle.fit(train, config=config, seasonal=seasonal)
    if calib["load_kw"].notna().sum() >= 24:
        bundle.calibrate(calib)
    tmp = final.with_name(final.name + ".new")
    bundle.save(tmp)
    smoke = LoadQuantileBundle.load(tmp)
    smoke.predict(grid.loc[grid.index[-24:]])
    _swap_in(tmp, final)
    return {
        "status": "ok",
        "train_rows": int(train["load_kw"].notna().sum()),
        "trained_until": train.index[-1].isoformat(),
        "calibrated_until": calib.index[-1].isoformat(),
    }


def _fetch_weather_chunk(site: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    query = urllib.parse.urlencode(
        {
            "latitude": site["latitude"],
            "longitude": site["longitude"],
            "hourly": ",".join(WEATHER_VARIABLES),
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "timezone": "UTC",
        }
    )
    url = f"https://previous-runs-api.open-meteo.com/v1/forecast?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "emhass-shadow-addon"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    frame = pd.DataFrame(payload["hourly"])
    frame["start_utc"] = pd.to_datetime(frame.pop("time"), utc=True)
    return frame.set_index("start_utc").sort_index()


def _ensure_weather_history(site: dict, needed_start: pd.Timestamp) -> pd.DataFrame:
    existing = None
    if WEATHER_HIST.exists():
        existing = pd.read_json(WEATHER_HIST, lines=True)
        existing["start_utc"] = pd.to_datetime(existing["start_utc"], utc=True)
        existing = existing.set_index("start_utc").sort_index()
    # The previous-runs archive publishes with a delay; stay a day behind.
    fetch_end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(days=1)
    cursor = needed_start
    if existing is not None and len(existing):
        cursor = max(needed_start, existing.index[-1] + pd.Timedelta(hours=1))
    frames = [] if existing is None else [existing]
    while cursor <= fetch_end:
        chunk_end = min(cursor + pd.Timedelta(days=FETCH_CHUNK_DAYS) - pd.Timedelta(hours=1), fetch_end)
        frames.append(_fetch_weather_chunk(site, cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(hours=1)
        time.sleep(1)
    weather = pd.concat(frames)
    weather = weather[~weather.index.duplicated(keep="last")].sort_index()
    weather.reset_index().to_json(
        WEATHER_HIST, orient="records", lines=True, date_format="iso"
    )
    return weather


def _fill_audit_temperature(audit: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """Backstop missing outdoor-temperature audit values from Open-Meteo.

    Sites without an outdoor sensor have an all-NaN temp column; the archived
    day-ahead temperature is close to the actual and keeps the temp_lag
    feature of the load model alive. Observed values always win.
    """
    # The column can be missing altogether: backfill rewrites the audit
    # without null fields, so a site with no outdoor sensor loses it, and
    # _retrain_load would then fail on the KeyError.
    observed = (
        pd.to_numeric(audit["temp_air_observed_c"], errors="coerce")
        if "temp_air_observed_c" in audit
        else pd.Series(np.nan, index=audit.index)
    )
    if weather is not None and "temperature_2m_previous_day1" in weather:
        observed = observed.fillna(weather["temperature_2m_previous_day1"].reindex(audit.index))
    return audit.assign(temp_air_observed_c=observed)


def _retrain_pv(audit: pd.DataFrame, site: dict, weather: pd.DataFrame | None = None) -> dict:
    final = MODELS / "pv_quantiles_v1"
    if (final / "bundle.json").exists():
        current = PvQuantileBundle.load(final)  # reuse confirmed site geometry
        geometry = current.geometry
        ratio_max = current.ratio_max
        min_clearsky = current.min_clearsky_kw
    else:
        # Fresh install: start from the panel parameters in the add-on
        # options; weekly retraining refines the ratio model on real data.
        geometry = PvGeometry(
            float(site["latitude"]),
            float(site["longitude"]),
            float(site["panel_tilt_deg"]),
            float(site["panel_azimuth_deg"]),
            float(site["panel_kwp"]),
            estimated_from_data=True,
        )
        ratio_max = 1.2
        min_clearsky = 0.1
    weather = _ensure_weather_history(site, audit.index[0]) if weather is None else weather
    poa = plane_power_kw(
        geometry,
        weather.index,
        weather["shortwave_radiation_previous_day1"],
        weather["direct_normal_irradiance_previous_day1"],
        weather["diffuse_radiation_previous_day1"],
    )
    frame = pd.DataFrame(index=audit.index)
    measured = pd.to_numeric(audit["pv_measured_kw"], errors="coerce")
    # Re-check censoring on every stored row: hours recorded before the
    # taper/saturation rules existed still say "keep" in pv_filter.
    censored = pipeline.pv_sink_limited_mask(audit, site.get("max_charge_kw"))
    frame["pv_measured_kw"] = measured.where((audit["pv_filter"] == "keep") & ~censored)
    frame["poa_forecast_kw"] = poa.reindex(audit.index)
    frame["cloud_cover_pct"] = weather["cloud_cover_previous_day1"].reindex(audit.index)
    frame["temp_air_c"] = weather["temperature_2m_previous_day1"].reindex(audit.index)
    # Hours without as-of weather cannot teach the weather->power mapping.
    frame["pv_measured_kw"] = frame["pv_measured_kw"].where(frame["poa_forecast_kw"].notna())
    split = frame.index[-1] - pd.Timedelta(days=CALIBRATION_DAYS)
    train, calib = frame.loc[:split], frame.loc[split + pd.Timedelta(hours=1) :]
    # Fresh install with a short audit: fit on everything instead of failing.
    if train["pv_measured_kw"].notna().sum() < 48:
        train, calib = frame, frame.loc[frame.index[-24:]]
    bundle = PvQuantileBundle.fit(
        train,
        geometry,
        ratio_max=ratio_max,
        min_clearsky_kw=min_clearsky,
    )
    try:
        bundle.calibrate(calib)
    except ValueError:
        pass  # too few usable calibration rows: keep zero offsets
    tmp = final.with_name(final.name + ".new")
    bundle.save(tmp)
    smoke = PvQuantileBundle.load(tmp)
    smoke.predict(frame.loc[frame.index[-24:]])
    _swap_in(tmp, final)
    return {
        "status": "ok",
        "train_rows": int((train["pv_measured_kw"].notna()).sum()),
        "excluded_rows": bundle.excluded_rows,
        "trained_until": train.index[-1].isoformat(),
    }


def retrain_if_due(force: bool = False, site: dict | None = None) -> dict:
    """Retrain both bundles when the configured interval elapsed (or forced).

    The retrain_enabled option gates only the automatic schedule; a forced run
    (panel button) is an explicit action and always goes ahead.
    """
    if not force and not _enabled():
        return retrain_status()
    state = _load_state()
    interval = _interval_days()
    last = state.get("last_retrain_utc")
    if not force and last:
        age = datetime.now(UTC) - datetime.fromisoformat(last)
        due = age >= timedelta(days=interval)
        # Drift (persistent bias or a collapsed P10-P90 band on the last week
        # of issued forecasts) retrains early, at most once a day.
        if not due and _drift_detected() and age >= DRIFT_MIN_RETRAIN_GAP:
            due = True
        if not due:
            return retrain_status()
    if site is None:
        opts = json.loads((DATA / "options.json").read_text())
        site = {
            "latitude": opts["latitude"],
            "longitude": opts["longitude"],
            "panel_kwp": opts.get("panel_kwp", 7.0),
            "panel_tilt_deg": opts.get("panel_tilt_deg", 26.0),
            "panel_azimuth_deg": opts.get("panel_azimuth_deg", 195.0),
            "max_charge_kw": opts.get("max_charge_kw"),
        }
    result = {"started_utc": datetime.now(UTC).isoformat(), "models": {}}
    try:
        audit = _audit_frame()
    except (OSError, ValueError) as error:
        result["models"]["load"] = {"status": "failed", "error": f"audit: {error}"}
        result["models"]["pv"] = {"status": "failed", "error": f"audit: {error}"}
        state["last_retrain_utc"] = result["started_utc"]
        state["models"] = result["models"]
        _save_state(state)
        return retrain_status()
    try:
        weather = _ensure_weather_history(site, audit.index[0])
    except Exception:
        weather = None  # load model can still fit without temperature
    audit = _fill_audit_temperature(audit, weather)
    try:
        result["models"]["load"] = _retrain_load(audit)
    except Exception as error:  # keep the live bundle on any failure
        result["models"]["load"] = {"status": "failed", "error": str(error)[:300]}
    try:
        result["models"]["pv"] = _retrain_pv(audit, site, weather)
    except Exception as error:
        result["models"]["pv"] = {"status": "failed", "error": str(error)[:300]}
    state["last_retrain_utc"] = result["started_utc"]
    state["models"] = result["models"]
    _save_state(state)
    return retrain_status()
