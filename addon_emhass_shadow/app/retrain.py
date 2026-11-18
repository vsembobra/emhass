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

import pandas as pd

from emhass.load_quantiles import LoadQuantileBundle
from emhass.pv_quantiles import PvQuantileBundle, plane_power_kw

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
    current = LoadQuantileBundle.load(final)  # reuse its feature config
    grid = pd.DataFrame(index=audit.index)
    grid["temp_air_observed_c"] = pd.to_numeric(audit["temp_air_observed_c"], errors="coerce")
    load = pd.to_numeric(audit["load_kw"], errors="coerce")
    # Only rows the audit marked clean are training targets; outage and
    # invalid hours would teach the model the wrong consumption pattern.
    grid["load_kw"] = load.where(audit["load_filter"] == "keep")
    split = grid.index[-1] - pd.Timedelta(days=CALIBRATION_DAYS)
    train, calib = grid.loc[:split], grid.loc[split + pd.Timedelta(hours=1) :]
    bundle = LoadQuantileBundle.fit(train, config=current.config, seasonal=current.seasonal)
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


def _retrain_pv(audit: pd.DataFrame, site: dict) -> dict:
    final = MODELS / "pv_quantiles_v1"
    current = PvQuantileBundle.load(final)  # reuse confirmed site geometry
    weather = _ensure_weather_history(site, audit.index[0])
    poa = plane_power_kw(
        current.geometry,
        weather.index,
        weather["shortwave_radiation_previous_day1"],
        weather["direct_normal_irradiance_previous_day1"],
        weather["diffuse_radiation_previous_day1"],
    )
    frame = pd.DataFrame(index=audit.index)
    measured = pd.to_numeric(audit["pv_measured_kw"], errors="coerce")
    frame["pv_measured_kw"] = measured.where(audit["pv_filter"] == "keep")
    frame["poa_forecast_kw"] = poa.reindex(audit.index)
    frame["cloud_cover_pct"] = weather["cloud_cover_previous_day1"].reindex(audit.index)
    frame["temp_air_c"] = weather["temperature_2m_previous_day1"].reindex(audit.index)
    # Hours without as-of weather cannot teach the weather->power mapping.
    frame["pv_measured_kw"] = frame["pv_measured_kw"].where(frame["poa_forecast_kw"].notna())
    split = frame.index[-1] - pd.Timedelta(days=CALIBRATION_DAYS)
    train, calib = frame.loc[:split], frame.loc[split + pd.Timedelta(hours=1) :]
    bundle = PvQuantileBundle.fit(
        train,
        current.geometry,
        ratio_max=current.ratio_max,
        min_clearsky_kw=current.min_clearsky_kw,
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
        due = datetime.now(UTC) - datetime.fromisoformat(last) >= timedelta(days=interval)
        if not due:
            return retrain_status()
    if site is None:
        opts = json.loads((DATA / "options.json").read_text())
        site = {"latitude": opts["latitude"], "longitude": opts["longitude"]}
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
        result["models"]["load"] = _retrain_load(audit)
    except Exception as error:  # keep the live bundle on any failure
        result["models"]["load"] = {"status": "failed", "error": str(error)[:300]}
    try:
        result["models"]["pv"] = _retrain_pv(audit, site)
    except Exception as error:
        result["models"]["pv"] = {"status": "failed", "error": str(error)[:300]}
    state["last_retrain_utc"] = result["started_utc"]
    state["models"] = result["models"]
    _save_state(state)
    return retrain_status()
