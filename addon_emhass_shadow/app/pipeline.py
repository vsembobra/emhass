"""Shadow pipeline for the EMHASS Shadow add-on.

Adapted from data/forecast_training/daily_shadow.py to run inside the add-on:
  * Home Assistant API through the Supervisor proxy (SUPERVISOR_TOKEN),
  * site/battery parameters from /data/options.json,
  * persistent data under /data, mirrors to /share/emhass_shadow,
  * summary delivered as a persistent notification in Home Assistant.

Nothing is written to the inverter: every Home Assistant call is a read,
except the persistent notification service call.
"""

import csv
import json
import os
import ssl
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import certifi
import numpy as np
import pandas as pd

from emhass.battery_planning import ForecastStep, floor_profile
from emhass.load_quantiles import LoadQuantileBundle
from emhass.outage_log import OutageEvent, read_events, record_observation_csv
from emhass.outage_risk import outage_risk_profile
from emhass.pv_quantiles import (
    PvGeometry,
    PvQuantileBundle,
    clear_sky_power_kw,
    plane_power_kw,
)
from emhass.reserve import LoadObservation, ReserveSettings, calculate_reserve, pv_aware_reserve

DATA = Path("/data")
SHARE = Path("/share/emhass_shadow")
HA_BASE = "http://supervisor/core"
SITE_ID = "home"
HORIZON_HOURS = 24
PLAN_HOURS = 48  # MPC planning horizon; only the first day is applied/reported
METRICS = DATA / "metrics.jsonl"

MEAN_SENSOR_FIELDS = {
    "load_kw": ("entity_load_power", 0.001),
    "critical_load_kw": ("entity_critical_load_power", 0.001),
    "pv_measured_kw": ("entity_pv_power", 0.001),
    "battery_power_kw_raw_sign": ("entity_battery_power", 0.001),
    "grid_power_kw_raw_sign": ("entity_grid_power", 0.001),
    "temp_air_observed_c": ("entity_outdoor_temperature", 1.0),
}
TOU_ZONES = 6

TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def options() -> dict:
    return json.loads((DATA / "options.json").read_text())


def resolve_entities(opts: dict) -> dict:
    """Entity map: every HA entity id comes from the add-on options.

    TOU zone entities use templates with {n} (zone number, 1..6) so a
    different naming scheme is one string instead of twelve.
    """
    return {
        "mean_sensors": {
            field: (opts[key], scale)
            for field, (key, scale) in MEAN_SENSOR_FIELDS.items()
        },
        "soc": opts["entity_battery_soc"],
        "export_counter": opts["entity_export_counter"],
        "grid_binary": opts["entity_grid_status"],
        "voltages": (
            {"grid_min_v": opts["entity_grid_voltage"]}
            if str(opts.get("entity_grid_voltage") or "").strip()
            else {}
        ),
        "tou_soc": [
            opts["tou_soc_entity_template"].format(n=n) for n in range(1, TOU_ZONES + 1)
        ],
        "tou_time": [
            opts["tou_time_entity_template"].format(n=n) for n in range(1, TOU_ZONES + 1)
        ],
        "floor_input": opts["floor_input_entity"],
    }


def _token() -> str:
    return os.environ["SUPERVISOR_TOKEN"]


def ha_get(path: str, timeout: float = 180):
    request = urllib.request.Request(
        HA_BASE + path, headers={"Authorization": f"Bearer {_token()}"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def ha_service(domain: str, service: str, payload: dict) -> None:
    request = urllib.request.Request(
        f"{HA_BASE}/api/services/{domain}/{service}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60):
        pass


def fetch_history(start: datetime, end: datetime, entities: list[str]) -> dict[str, list[dict]]:
    entities = [e for e in entities if str(e or "").strip()]
    if not entities:
        return {}
    query = urllib.parse.urlencode(
        {
            "filter_entity_id": ",".join(entities),
            "end_time": end.isoformat(),
            "minimal_response": "",
        }
    )
    path = f"/api/history/period/{urllib.parse.quote(start.isoformat())}?{query}"
    rows = {}
    for entity_states in ha_get(path):
        if entity_states:
            rows[entity_states[0]["entity_id"]] = entity_states
    return rows


def _state_value(state: dict) -> str | None:
    return state.get("state") if "state" in state else state.get("s")


def _state_at(state: dict) -> datetime:
    for key in ("last_updated", "last_changed"):
        if key in state:
            return datetime.fromisoformat(state[key]).astimezone(UTC)
    return datetime.fromtimestamp(state.get("lu", state["lc"]), UTC)


def numeric_samples(states: list[dict], scale: float) -> list[tuple[datetime, float]]:
    samples = []
    for state in states:
        value = _state_value(state)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        samples.append((_state_at(state), number * scale))
    return samples


def _carried(samples: list[tuple[datetime, float]], hour: datetime) -> float | None:
    before = [value for at, value in samples if at < hour]
    return float(before[-1]) if before else None


def hourly_mean(samples, hour):
    values = [v for at, v in samples if hour <= at < hour + timedelta(hours=1)]
    return float(np.mean(values)) if values else _carried(samples, hour)


def hourly_min(samples, hour):
    values = [v for at, v in samples if hour <= at < hour + timedelta(hours=1)]
    return float(min(values)) if values else _carried(samples, hour)


def counter_delta(states, hour):
    samples = numeric_samples(states, 1.0)
    before = [v for at, v in samples if at <= hour]
    after = [v for at, v in samples if at <= hour + timedelta(hours=1)]
    if not before or not after:
        return None
    return max(0.0, float(after[-1] - before[-1]))


def solar_elevation(latitude, longitude, hour):
    import pvlib

    midpoint = pd.DatetimeIndex([hour + timedelta(minutes=30)])
    position = pvlib.location.Location(latitude, longitude).get_solarposition(midpoint)
    return float(position["apparent_elevation"].iloc[0])


def grid_status_series(states):
    series = []
    for state in states:
        value = {"on": "up", "off": "down"}.get(_state_value(state))
        if value is None:
            continue
        series.append((_state_at(state), value))
    return sorted(series)


def build_audit_rows(history, hours, latitude, longitude, ent):
    means = {
        field: numeric_samples(history.get(entity, []), scale)
        for field, (entity, scale) in ent["mean_sensors"].items()
    }
    soc = numeric_samples(history.get(ent["soc"], []), 0.01)
    voltages = {
        field: numeric_samples(history.get(entity, []), 1.0)
        for field, entity in ent["voltages"].items()
    }
    grid_series = grid_status_series(history.get(ent["grid_binary"], []))
    history_start = grid_series[0][0] if grid_series else None

    rows = []
    for hour in hours:
        row = {"start_utc": hour.isoformat().replace("+00:00", ".000Z")}
        for field, samples in means.items():
            row[field] = hourly_mean(samples, hour)
        soc_hour = [v for at, v in soc if hour <= at < hour + timedelta(hours=1)]
        carried_soc = _carried(soc, hour)
        row["soc_mean"] = float(np.mean(soc_hour)) if soc_hour else carried_soc
        row["soc_max"] = float(max(soc_hour)) if soc_hour else carried_soc
        row["export_kwh"] = counter_delta(history.get(ent["export_counter"], []), hour)
        for field, samples in voltages.items():
            row[field] = hourly_min(samples, hour)
        elevation = solar_elevation(latitude, longitude, hour)
        row["solar_elevation_deg"] = elevation

        in_hour = [s for at, s in grid_series if hour <= at < hour + timedelta(hours=1)]
        mins = [row[f] for f in ent["voltages"] if row[f] is not None]
        low_voltage = bool(mins) and min(mins) <= 50.0
        if history_start is not None and hour >= history_start.replace(
            minute=0, second=0, microsecond=0
        ):
            row["raw_grid_status"] = "outage" if "down" in in_hour else "up"
            row["grid_quality"] = (
                "low_grid_voltage"
                if low_voltage
                else ("recorded_outage" if "down" in in_hour else "recorded_up")
            )
        elif not grid_series:
            row["raw_grid_status"] = "unknown"
            row["grid_quality"] = "unknown"
        else:
            row["raw_grid_status"] = "outside_history"
            row["grid_quality"] = "low_grid_voltage" if low_voltage else "voltage_not_flagged"

        bad_grid = row["grid_quality"] in ("recorded_outage", "low_grid_voltage", "unknown")
        if row["load_kw"] is None or row["critical_load_kw"] is None:
            row["load_filter"] = "invalid_target"
        elif bad_grid:
            row["load_filter"] = "grid_quality"
        else:
            row["load_filter"] = "keep"

        if elevation <= 10.0:
            row["pv_filter"] = "outside_daylight"
            row["possible_pv_curtailment"] = False
        else:
            curtailment = (
                row["soc_mean"] is not None
                and row["soc_mean"] >= 0.95
                and row["export_kwh"] is not None
                and row["export_kwh"] <= 0.01
            )
            row["possible_pv_curtailment"] = bool(curtailment)
            if row["pv_measured_kw"] is None:
                row["pv_filter"] = "invalid_target"
            elif row["export_kwh"] is None or row["soc_mean"] is None:
                row["pv_filter"] = "missing_or_invalid_export"
            elif bad_grid:
                row["pv_filter"] = "grid_quality"
            elif curtailment:
                row["pv_filter"] = "possible_curtailment"
            else:
                row["pv_filter"] = "keep"
        row["split"] = "live"
        rows.append(row)
    return rows


BACKFILL_CHUNK_DAYS = 3  # recorder fetches are chunked; one long range can time out


def _audit_entities(ent) -> list[str]:
    # Empty option values ("") must never reach the history query:
    # an empty filter_entity_id entry makes HA answer 400 Bad Request.
    return [
        e
        for e in (
            [e for e, _ in ent["mean_sensors"].values()]
            + list(ent["voltages"].values())
            + [ent["soc"], ent["export_counter"], ent["grid_binary"]]
        )
        if str(e or "").strip()
    ]


def _rows_for_hours(site, ent, hours: list) -> list[dict]:
    """Audit rows for the given hours; HA history fetched in safe chunks."""
    if not hours:
        return []
    entities = _audit_entities(ent)
    rows = []
    cursor = hours[0]
    end = hours[-1] + timedelta(hours=1)
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=BACKFILL_CHUNK_DAYS), end)
        chunk_hours = [h for h in hours if cursor <= h < chunk_end]
        history = fetch_history(cursor, chunk_end, entities)
        rows.extend(
            build_audit_rows(history, chunk_hours, site["latitude"], site["longitude"], ent)
        )
        cursor = chunk_end
    return rows


def _read_audit() -> pd.DataFrame:
    audit = pd.read_json(DATA / "hourly_audit.jsonl", lines=True)
    audit["start_utc"] = pd.to_datetime(audit["start_utc"], utc=True)
    return audit.set_index("start_utc").sort_index()


def _json_default(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return str(v)


def backfill_audit(site, ent, days: float) -> int:
    """Extend the audit backwards up to `days` ago, recorder depth permitting.

    A long-lived Home Assistant already holds weeks of load/PV/SOC history;
    pulling it into the audit lets the quantile models train on day one
    instead of after weeks of passive collection. Idempotent: hours already
    in the audit are never re-fetched. Returns the number of rows added.
    """
    audit_path = DATA / "hourly_audit.jsonl"
    now_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    target = now_hour - timedelta(days=days)
    if audit_path.exists():
        audit = _read_audit()
        first = audit.index[0]
    else:
        audit = pd.DataFrame()
        first = now_hour
    if first <= target:
        return 0
    hours = list(pd.date_range(target, first - timedelta(hours=1), freq="h"))
    rows = _rows_for_hours(site, ent, hours)
    if not rows:
        return 0
    prepended = pd.DataFrame(rows)
    prepended["start_utc"] = pd.to_datetime(prepended["start_utc"], utc=True)
    combined = pd.concat([audit, prepended.set_index("start_utc")]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    audit_path.write_text(
        "".join(
            json.dumps(
                {"start_utc": at.isoformat(), **row.dropna().to_dict()},
                default=_json_default,
            )
            + "\n"
            for at, row in combined.iterrows()
        ),
        encoding="utf-8",
    )
    return len(rows)


def update_audit(site, ent, backfill_days: float = 1.0) -> tuple[pd.DataFrame, int]:
    audit_path = DATA / "hourly_audit.jsonl"
    now_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    if audit_path.exists():
        audit = _read_audit()
        last = audit.index[-1]
    else:
        # Fresh install: backfill the recorder history so the very first run
        # already trains on real data instead of a 24 h stub.
        audit = pd.DataFrame()
        last = now_hour - timedelta(days=max(1.0, backfill_days))
    if last >= now_hour - timedelta(hours=1):
        return audit, 0
    hours = []
    cursor = last + timedelta(hours=1)
    while cursor < now_hour:
        hours.append(cursor)
        cursor += timedelta(hours=1)
    rows = _rows_for_hours(site, ent, hours)
    with audit_path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    appended = pd.DataFrame(rows)
    appended["start_utc"] = pd.to_datetime(appended["start_utc"], utc=True)
    return pd.concat([audit, appended.set_index("start_utc")]).sort_index(), len(rows)


def _last_observation() -> datetime | None:
    """Newest recorded grid observation; makes re-feeding history idempotent."""
    import sqlite3

    database = DATA / "outages.sqlite"
    if not database.exists():
        return None
    try:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT MAX(timestamp) FROM outage_observations").fetchone()
    except sqlite3.Error:
        return None
    return datetime.fromisoformat(row[0]) if row and row[0] else None


def update_outage_log(ent) -> int:
    grid_binary = str(ent.get("grid_binary") or "").strip()
    if not grid_binary:
        return 0
    now = datetime.now(UTC)
    history = fetch_history(now - timedelta(hours=48), now, [grid_binary])
    series = grid_status_series(history.get(grid_binary, []))
    cursor_at = _last_observation()
    if cursor_at is not None:
        series = [sample for sample in series if sample[0] > cursor_at]
    if series:
        csv_path = DATA / "grid_observations_tmp.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "status"])
            for at, status in series:
                writer.writerow([at.isoformat(), status])
        record_observation_csv(DATA / "outages.sqlite", SITE_ID, csv_path, max_gap=timedelta(hours=6))
        csv_path.unlink()
    return len(series)


def archive_weather(site) -> tuple[pd.DataFrame, str, bool]:
    """Fetch a fresh Open-Meteo vintage; fall back to the newest archived one.

    The fallback is explicit and labelled: when the grid (and with it the WAN
    link) is down, the report still ships, marked with the archive vintage.
    """
    variables = (
        "temperature_2m,shortwave_radiation,direct_normal_irradiance,"
        "diffuse_radiation,cloud_cover"
    )
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={site['latitude']}&longitude={site['longitude']}"
        f"&hourly={variables}&forecast_hours=48&timezone=UTC&timeformat=unixtime"
    )
    archive = DATA / "weather_archive"
    archive.mkdir(exist_ok=True)
    stale = False
    try:
        with urllib.request.urlopen(url, timeout=30, context=TLS_CONTEXT) as response:
            body = response.read().decode("utf-8")
        received = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        (archive / f"{received}.json").write_text(body, encoding="utf-8")
        payload = json.loads(body)
    except OSError:
        files = sorted(archive.glob("*.json"))
        if not files:
            raise
        stale = True
        received = files[-1].stem
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["hourly"])
    frame["start_utc"] = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
    return frame.set_index("start_utc").sort_index(), received + ".json", stale


def load_outage_events() -> list[OutageEvent]:
    events = []
    path = DATA / "outages.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            events.append(
                OutageEvent(
                    pd.Timestamp(record["start_utc"]).to_pydatetime(),
                    pd.Timestamp(record["end_record_utc"]).to_pydatetime(),
                    "measured",
                    bool(record["interrupted"]),
                )
            )
    events.extend(read_events(DATA / "outages.sqlite", SITE_ID))
    return events


def _score(actual: pd.Series, p10: list, p50: list, p90: list) -> dict | None:
    """Error, bias and P10–P90 coverage of one 24 h forecast against actuals."""
    mask = actual.notna().to_numpy()
    n = int(mask.sum())
    if n == 0:
        return None
    y = actual[actual.notna()].to_numpy(dtype=float)
    mid = np.array(p50, dtype=float)[mask]
    lo = np.array(p10, dtype=float)[mask]
    hi = np.array(p90, dtype=float)[mask]
    return {
        "hours": n,
        "mae_kwh": round(float(np.abs(y - mid).sum()), 2),
        "bias_kwh": round(float((mid - y).sum()), 2),
        "coverage_pct": round(float(np.mean((y >= lo) & (y <= hi)) * 100), 1),
    }


def update_metrics(audit: pd.DataFrame, tz: ZoneInfo) -> int:
    """Score fully observed past reports against audit actuals, append once."""
    entries = []
    if METRICS.exists():
        entries = [
            json.loads(line)
            for line in METRICS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    scored = {entry["issue_utc"] for entry in entries}
    audit_end = audit.index[-1]
    added = 0
    reports_dir = DATA / "shadow_reports"
    for path in sorted(reports_dir.glob("*.json"))[-10:]:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        hourly = report.get("hourly")
        if not hourly or report["issue_utc"] in scored:
            continue
        starts = pd.DatetimeIndex(pd.to_datetime(hourly["starts_utc"], utc=True))
        if len(starts) != HORIZON_HOURS or starts[-1] > audit_end:
            continue  # horizon not fully observed yet
        actual_load = audit["load_kw"].where(audit["load_filter"] == "keep").reindex(starts)
        daylight = (audit["solar_elevation_deg"].reindex(starts) > 10).fillna(False)
        actual_pv = audit["pv_measured_kw"].where(
            (audit["pv_filter"] == "keep") & daylight
        ).reindex(starts)
        entry = {
            "issue_utc": report["issue_utc"],
            "date": pd.Timestamp(report["issue_utc"]).astimezone(tz).date().isoformat(),
            "load": _score(
                actual_load, hourly["load_p10_kw"], hourly["load_p50_kw"], hourly["load_p90_kw"]
            ),
            "pv": _score(
                actual_pv, hourly["pv_p10_kw"], hourly["pv_p50_kw"], hourly["pv_p90_kw"]
            ),
        }
        # Plan vs fact: the optimizer's promised grid import against the meter.
        plan_grid = ((report.get("advisory") or {}).get("optimizer") or {}).get("p_grid_kw")
        if plan_grid and len(plan_grid) >= HORIZON_HOURS:
            actual_import = audit["grid_power_kw_raw_sign"].reindex(starts).clip(lower=0.0)
            if actual_import.notna().sum() >= 20:
                entry["plan_vs_fact"] = {
                    "planned_grid_kwh": round(
                        sum(max(0.0, float(v)) for v in plan_grid[:HORIZON_HOURS]), 2
                    ),
                    "actual_grid_kwh": round(float(actual_import.sum()), 2),
                }
        entries.append(entry)
        added += 1
    if added:
        entries = entries[-60:]
        METRICS.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
            encoding="utf-8",
        )
        try:
            SHARE.mkdir(parents=True, exist_ok=True)
            (SHARE / "metrics.jsonl").write_text(METRICS.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass
    return added


def _retrain_note() -> str | None:
    """One notification line when models were retrained in the last hours."""
    state_path = DATA / "retrain_state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        last = datetime.fromisoformat(state["last_retrain_utc"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None
    if datetime.now(UTC) - last > timedelta(hours=6):
        return None

    def fmt(info: dict) -> str:
        if info.get("status") == "ok":
            return f"ok ({info.get('train_rows', '?')} строк)"
        return f"ошибка: {info.get('error', '?')[:80]}"

    models = state.get("models", {})
    return (
        f"• Переобучение моделей: нагрузка — {fmt(models.get('load', {}))}, "
        f"солнце — {fmt(models.get('pv', {}))}"
    )


def _plan_vs_fact_note() -> str | None:
    """One notification line comparing yesterday's promised vs metered import."""
    if not METRICS.exists():
        return None
    try:
        entries = [
            json.loads(line)
            for line in METRICS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    for entry in reversed(entries):
        pvf = entry.get("plan_vs_fact")
        if pvf:
            delta = pvf["actual_grid_kwh"] - pvf["planned_grid_kwh"]
            return (
                f"• План vs факт ({entry['date']}): импорт из сети по плану "
                f"{pvf['planned_grid_kwh']} кВт·ч, по счётчику {pvf['actual_grid_kwh']} кВт·ч "
                f"({delta:+.1f})"
            )
    return None


def _fallback_load_prediction(audit: pd.DataFrame, starts: pd.DatetimeIndex, tz) -> pd.DataFrame:
    """Persistence forecast for fresh installs before the models exist.

    Same-local-hour mean of clean audit history; a flat 0.5 kW when there is
    no history at all. Bands are deliberately wide (±40/50 %) so downstream
    planning stays conservative until LightGBM takes over.
    """
    hist = audit["load_kw"].where(audit["load_filter"] == "keep").dropna()
    if len(hist):
        by_hour = hist.groupby(hist.index.tz_convert(tz).hour).mean()
        p50 = np.array(
            [float(by_hour.get(s.tz_convert(tz).hour, np.nan)) for s in starts]
        )
        p50 = np.where(np.isnan(p50), float(hist.mean()), p50)
    else:
        p50 = np.full(len(starts), 0.5)
    return pd.DataFrame(
        {
            "load_p10_kw": p50 * 0.6,
            "load_p50_kw": p50,
            "load_p90_kw": p50 * 1.5,
        },
        index=starts,
    )


def _site_pv_geometry(opts: dict) -> PvGeometry:
    return PvGeometry(
        float(opts["latitude"]),
        float(opts["longitude"]),
        float(opts["panel_tilt_deg"]),
        float(opts["panel_azimuth_deg"]),
        float(opts["panel_kwp"]),
    )


def _fallback_pv_forecast(
    starts: pd.DatetimeIndex, opts: dict, weather: pd.DataFrame | None
) -> tuple[list, list, list]:
    """Clear-sky derated by cloud cover; zeros when even that is impossible."""
    try:
        clearsky = clear_sky_power_kw(_site_pv_geometry(opts), starts).to_numpy()
    except Exception:
        zeros = [0.0] * len(starts)
        return zeros, zeros, zeros
    if weather is not None and "cloud_cover" in weather:
        cloud = weather["cloud_cover"].reindex(starts).to_numpy(dtype=float)
        factor = np.where(np.isnan(cloud), 0.5, 1.0 - 0.75 * cloud / 100.0)
    else:
        factor = np.full(len(starts), 0.5)
    p50 = clearsky * factor
    p10 = p50 * 0.3
    p90 = clearsky * 0.85
    return list(p10), list(p50), list(p90)


def _soc_bounds(opts: dict) -> tuple[float, float]:
    """Percent keys (min_soc_pct) win; pre-0.7 fraction keys still work."""
    if "min_soc_pct" in opts:
        return float(opts["min_soc_pct"]) / 100.0, float(opts["max_soc_pct"]) / 100.0
    return float(opts["min_soc"]), float(opts["max_soc"])


def _clamp_soh(value: float) -> float:
    """Percent or fraction in, fraction in [0.4, 1] out.

    Clamped so a glitchy sensor (or a typo) cannot zero out the reserve.
    """
    return min(1.0, max(0.4, value / 100.0 if value > 1.5 else value))


def _battery_soh(opts: dict) -> tuple[float, str]:
    """State of health: the option accepts an entity id OR a plain number.

    "99"/"0.95" are used as-is (manual), "sensor.inverter_battery_soh" is
    read from Home Assistant, empty means "assume a fresh 100 % pack".
    """
    raw = str(opts.get("entity_battery_soh") or "").strip()
    if not raw:
        return 1.0, "default"
    try:
        return _clamp_soh(float(raw)), "manual"
    except ValueError:
        pass
    try:
        return _clamp_soh(float(ha_get(f"/api/states/{raw}")["state"])), "entity"
    except Exception:
        return 1.0, "unavailable"


ENTITY_CHECK_LABELS = {
    "load_kw": "Мощность нагрузки",
    "critical_load_kw": "Критическая нагрузка",
    "pv_measured_kw": "Мощность солнца",
    "battery_power_kw_raw_sign": "Мощность батареи",
    "grid_power_kw_raw_sign": "Мощность сети",
    "temp_air_observed_c": "Уличная температура",
}


def validate_entities() -> list[dict]:
    """Live-check every configured entity against the HA state machine.

    Returns one row per parameter: label, entity id, current state and a
    status — ok / ok_number (manual number) / skipped (empty optional) /
    missing (404) / not_numeric / unexpected. Never raises.
    """
    opts = options()
    ent = resolve_entities(opts)
    specs: list[tuple[str, str, str]] = [
        (ENTITY_CHECK_LABELS[field], str(entity or ""), "number")
        for field, (entity, _scale) in ent["mean_sensors"].items()
    ]
    specs += [
        ("SOC батареи", str(ent["soc"] or ""), "number"),
        ("Счётчик экспорта", str(ent["export_counter"] or ""), "number"),
        ("Статус сети", str(ent["grid_binary"] or ""), "binary"),
    ]
    specs += [
        ("Напряжение сети", str(entity or ""), "number")
        for entity in ent["voltages"].values()
    ]
    for n in range(TOU_ZONES):
        specs.append((f"Зона {n + 1}: SOC", str(ent["tou_soc"][n] or ""), "number"))
        specs.append((f"Зона {n + 1}: время старта", str(ent["tou_time"][n] or ""), "time"))
    specs.append(("Порог SOC (helper)", str(ent["floor_input"] or ""), "number"))
    specs.append(("SoH батареи", str(opts.get("entity_battery_soh") or ""), "soh"))

    results = []
    for label, entity, kind in specs:
        entity = entity.strip()
        row = {"label": label, "entity": entity, "state": None, "status": "ok"}
        if not entity:
            row["status"] = "skipped"
            results.append(row)
            continue
        if kind == "soh":
            try:
                float(entity)  # manual number, not an entity id
                row.update(state=entity, status="ok_number")
                results.append(row)
                continue
            except ValueError:
                pass
        try:
            state = str(ha_get(f"/api/states/{entity}", timeout=10)["state"])
        except Exception:
            row["status"] = "missing"
            results.append(row)
            continue
        row["state"] = state
        if kind in ("number", "soh"):
            try:
                float(state)
            except ValueError:
                row["status"] = "not_numeric"
        elif kind == "binary" and state not in ("on", "off"):
            row["status"] = "unexpected"
        elif kind == "time" and ":" not in state:
            row["status"] = "unexpected"
        results.append(row)
    return results


def _entity_issue_note(checks: list[dict]) -> str | None:
    """One notification line listing broken entity bindings, if any."""
    problems = [c for c in checks if c["status"] in ("missing", "not_numeric", "unexpected")]
    if not problems:
        return None
    wording = {"missing": "не найдена", "not_numeric": "не число", "unexpected": "странное значение"}
    items = "; ".join(
        f"{c['label']} ({c['entity']}): {wording[c['status']]}" for c in problems
    )
    return f"• ⚠️ Проверь настройки сущностей: {items}"


def run() -> dict:
    opts = options()
    ent = resolve_entities(opts)
    entity_checks = validate_entities()
    site = {
        "latitude": opts["latitude"],
        "longitude": opts["longitude"],
        "time_zone": opts["time_zone"],
    }
    tz = ZoneInfo(site["time_zone"])

    audit, appended = update_audit(
        site, ent, backfill_days=float(opts.get("history_backfill_days", 14))
    )
    new_transitions = update_outage_log(ent)

    issue = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    starts = pd.date_range(issue, periods=HORIZON_HOURS, freq="h")
    starts_plan = pd.date_range(issue, periods=PLAN_HOURS, freq="h")

    models_learning = False
    try:
        load_bundle = LoadQuantileBundle.load(DATA / "models" / "load_quantiles_v1")
        grid = audit[["temp_air_observed_c", "load_kw"]].copy()
        grid = grid.reindex(grid.index.union(starts_plan))
        # Causal 48 h: two 24 h windows. For day 2 the issue time is shifted
        # by 24 h, so lag-24 features read today's actuals — known at issue.
        day1 = load_bundle.predict(grid.loc[: starts[-1]], issue_time=issue).loc[starts]
        day2 = load_bundle.predict(
            grid, issue_time=issue + pd.Timedelta(hours=HORIZON_HOURS)
        ).loc[starts_plan[HORIZON_HOURS:]]
        load_prediction = pd.concat([day1, day2])
    except (OSError, ValueError, KeyError):
        models_learning = True
        load_prediction = _fallback_load_prediction(audit, starts_plan, tz)

    pv_bundle = None
    try:
        pv_bundle = PvQuantileBundle.load(DATA / "models" / "pv_quantiles_v1")
    except (OSError, ValueError):
        models_learning = True
    try:
        weather, weather_name, weather_stale = archive_weather(site)
        if pv_bundle is not None:
            poa = plane_power_kw(
                pv_bundle.geometry,
                weather.index,
                weather["shortwave_radiation"],
                weather["direct_normal_irradiance"],
                weather["diffuse_radiation"],
            )
            pv_frame = pd.DataFrame(
                {
                    "pv_measured_kw": np.nan,
                    "poa_forecast_kw": poa,
                    "cloud_cover_pct": weather["cloud_cover"],
                    "temp_air_c": weather["temperature_2m"],
                }
            ).reindex(starts_plan)
            pv_prediction = pv_bundle.predict(pv_frame)
            pv_p10 = [float(pv_prediction["pv_p10_kw"].iloc[i]) for i in range(PLAN_HOURS)]
            pv_p50 = [float(pv_prediction["pv_p50_kw"].iloc[i]) for i in range(PLAN_HOURS)]
            pv_p90 = [float(pv_prediction["pv_p90_kw"].iloc[i]) for i in range(PLAN_HOURS)]
        else:
            pv_p10, pv_p50, pv_p90 = _fallback_pv_forecast(starts_plan, opts, weather)
        pv_day = {
            "p10": round(float(np.sum(pv_p10[:HORIZON_HOURS])), 2),
            "p50": round(float(np.sum(pv_p50[:HORIZON_HOURS])), 2),
            "p90": round(float(np.sum(pv_p90[:HORIZON_HOURS])), 2),
        }
        pv_day2 = {
            "p10": round(float(np.sum(pv_p10[HORIZON_HOURS:])), 2),
            "p50": round(float(np.sum(pv_p50[HORIZON_HOURS:])), 2),
            "p90": round(float(np.sum(pv_p90[HORIZON_HOURS:])), 2),
        }
    except OSError:
        # No internet and no archived vintage at all: assume zero solar, the
        # conservative direction for battery planning, and say so in the report.
        weather_name, weather_stale = None, True
        pv_p10 = [0.0] * PLAN_HOURS
        pv_p50 = [0.0] * PLAN_HOURS
        pv_p90 = [0.0] * PLAN_HOURS
        pv_day = None
        pv_day2 = None

    # State of health: entity id or a plain number ("99"); missing or
    # unreadable values keep the nominal 100 % and are flagged in the report.
    soh, soh_source = _battery_soh(opts)

    critical = audit["critical_load_kw"].dropna()
    settings = ReserveSettings(
        nominal_capacity_kwh=opts["capacity_kwh"],
        soh=soh,
        floor_soc=_soc_bounds(opts)[0],
        max_soc=_soc_bounds(opts)[1],
        discharge_efficiency=opts["discharge_efficiency"],
        cold_start_kw=float(critical.quantile(0.9)) if len(critical) else 0.5,
        cold_start_hours=min(20.0, max(1.0, float(opts.get("reserve_cold_start_hours", 2.0)))),
        window_days=14.0,
        quantile=0.9,
        margin=0.0,
        decay_days=1.0e6,  # owner rule: hold 14 days, no gradual reduction
        profile_mode="all_hours",
        site_timezone=site["time_zone"],
    )
    events = load_outage_events()
    observations = [
        LoadObservation(at.to_pydatetime(), float(row["critical_load_kw"]))
        for at, row in audit.iterrows()
        if row["grid_quality"] in ("recorded_outage", "low_grid_voltage")
        and pd.notna(row["critical_load_kw"])
    ]
    plan = calculate_reserve(issue, events, observations, settings)
    steps = [
        ForecastStep(
            start=starts_plan[i].to_pydatetime(),
            hours=1.0,
            pv_p10_kw=pv_p10[i],
            load_p90_kw=float(load_prediction["load_p90_kw"].iloc[i]),
        )
        for i in range(PLAN_HOURS)
    ]

    # Probabilistic outage layer: raise the reserve during historically risky
    # local hours instead of relying only on the post-outage reactive hold.
    risk = outage_risk_profile(events, issue, site["time_zone"])
    reserve_by_hour = [plan.reserve_kwh] * PLAN_HOURS
    risk_windows: list[str] = []
    if risk is not None and any(risk.risk_hours):
        risk_energy = (
            plan.critical_load_kw * risk.duration_p90_hours / settings.discharge_efficiency
        )
        for i in range(PLAN_HOURS):
            local_hour = starts_plan[i].astimezone(tz).hour
            if risk.risk_hours[local_hour]:
                reserve_by_hour[i] = max(reserve_by_hour[i], risk_energy)
        start_hour = None
        for hour in range(25):  # group consecutive risky hours into windows
            risky = hour < 24 and risk.risk_hours[hour]
            if risky and start_hour is None:
                start_hour = hour
            elif not risky and start_hour is not None:
                risk_windows.append(f"{start_hour:02d}–{hour:02d}")
                start_hour = None

    # PV-aware layer: where the pessimistic solar forecast covers part of the
    # critical load inside the outage window, the battery reserve for those
    # hours shrinks accordingly (never below the cold-start baseline, never
    # above the outage-statistics value).
    if opts.get("reserve_pv_aware", True):
        baseline_kwh = (
            settings.cold_start_hours * settings.cold_start_kw / settings.discharge_efficiency
        )
        reserve_by_hour = pv_aware_reserve(
            reserve_by_hour,
            pv_p10,
            plan.duration_hours,
            settings.discharge_efficiency,
            baseline_kwh,
        )

    profile = floor_profile(steps, reserve_by_hour, settings)
    floors_plan = [round(step.floor_soc / 0.05) * 0.05 for step in profile.steps]
    floors = floors_plan[:HORIZON_HOURS]  # advisory/report day; plan horizon is longer
    recommended_floor = floors[0]  # binding constraint is the current hour

    floor_entity = ent["floor_input"]
    floor_warning: str | None = None
    try:
        current_floor = float(ha_get(f"/api/states/{floor_entity}")["state"]) / 100.0
    except Exception:
        # A missing/unreadable floor entity must not kill the whole run:
        # assume the configured minimum and say so in the report.
        current_floor = _soc_bounds(opts)[0]
        floor_warning = f"сущность порога SOC ({floor_entity}) недоступна"
    programs = []
    for n, (soc_entity, time_entity) in enumerate(
        zip(ent["tou_soc"], ent["tou_time"], strict=True), start=1
    ):
        soc = float(ha_get(f"/api/states/{soc_entity}")["state"])
        start = ha_get(f"/api/states/{time_entity}")["state"]
        programs.append({"n": n, "from": start, "soc_pct": soc})
    # Each TOU zone gets the floor of its own start hour (next occurrence in
    # the horizon), not a single all-day value.
    for program in programs:
        hour = int(program["from"][:2])
        match = next(
            (
                i
                for i in range(HORIZON_HOURS)
                if starts[i].astimezone(tz).hour == hour and starts[i] >= issue
            ),
            None,
        )
        program["recommended_soc_pct"] = (
            round(floors[match] * 100) if match is not None else None
        )
    would_raise = [
        p
        for p in programs
        if p.get("recommended_soc_pct") is not None
        and p["soc_pct"] < p["recommended_soc_pct"]
    ]

    battery_now = ha_get(f"/api/states/{ent['soc']}")["state"]

    # MILP layer: optimize the 48 h horizon against the tariff and the
    # floating floor; only the first day is applied or reported. Battery
    # energy capacity is scaled by the measured SoH so kWh accounting inside
    # the solver matches the real pack. Advisory only — never let it break
    # the daily plan.
    optimizer_plan = None
    if opts.get("optimizer_enabled", True):
        try:
            import optimize

            load_p50 = [
                float(load_prediction["load_p50_kw"].iloc[i]) for i in range(PLAN_HOURS)
            ]
            opt_opts = dict(opts, capacity_kwh=float(opts["capacity_kwh"]) * soh)
            optimizer_plan = optimize.run_plan(
                starts_plan, pv_p50, load_p50, floors_plan,
                float(battery_now) / 100.0, programs, opt_opts,
            )
        except Exception:
            optimizer_plan = None

    # Apply mode (off by default): write the plan into the inverter through
    # Home Assistant services. Optimizer zone targets win when available,
    # otherwise the heuristic per-zone floors. Writes happen only when the
    # change is at least 5 percentage points — no churn on rounding noise.
    applied: list[str] = []
    apply_errors: list[str] = []
    if opts.get("apply_mode", False):
        zone_pct = {}
        if optimizer_plan is not None:
            for t in optimizer_plan["zone_targets"]:
                if t.get("optimal_soc_pct") is not None:
                    zone_pct[t["n"]] = t["optimal_soc_pct"]
        else:
            for p in programs:
                if p.get("recommended_soc_pct") is not None:
                    zone_pct[p["n"]] = p["recommended_soc_pct"]
        min_pct = int(round(_soc_bounds(opts)[0] * 100))
        max_pct = int(round(_soc_bounds(opts)[1] * 100))
        for p in programs:
            target = zone_pct.get(p["n"])
            if target is None:
                continue
            target = min(max(int(round(target)), min_pct), max_pct)
            if abs(target - p["soc_pct"]) < 5:
                continue
            try:
                ha_service(
                    "number",
                    "set_value",
                    {"entity_id": ent["tou_soc"][p["n"] - 1], "value": target},
                )
                applied.append(f"зона {p['n']} ({p['from'][:5]}) {p['soc_pct']:.0f}→{target}%")
            except Exception:
                apply_errors.append(f"зона {p['n']}")
        floor_target = min(max(int(round(recommended_floor * 100)), min_pct), max_pct)
        if abs(floor_target - round(current_floor * 100)) >= 5:
            try:
                # Domain follows the entity id: number.* and input_number.*
                # share the set_value service name but live in own domains.
                ha_service(
                    ent["floor_input"].split(".", 1)[0],
                    "set_value",
                    {"entity_id": ent["floor_input"], "value": floor_target},
                )
                applied.append(f"порог SOC {round(current_floor * 100)}→{floor_target}%")
            except Exception:
                apply_errors.append("порог SOC")

    last_outage = max(
        (e.end for e in events if e.end is not None and e.source == "measured"), default=None
    )
    hold_until = (
        (last_outage + timedelta(days=14)).astimezone(tz).date().isoformat()
        if last_outage is not None
        else None
    )

    load_day = {
        q: round(float(load_prediction[f"load_{q}_kw"].iloc[:HORIZON_HOURS].sum()), 2)
        for q in ("p10", "p50", "p90")
    }
    load_day2 = {
        q: round(float(load_prediction[f"load_{q}_kw"].iloc[HORIZON_HOURS:].sum()), 2)
        for q in ("p10", "p50", "p90")
    }
    metrics_scored = update_metrics(audit, tz)
    delta_pp = round((recommended_floor - current_floor) * 100)

    pv_line = (
        f"• Солнце 24 ч, кВт·ч: P50 {pv_day['p50']} (пессимистично P10 {pv_day['p10']})"
        if pv_day is not None
        else "• Солнце: прогноз недоступен (нет интернета и архива) — считаем худший случай 0 кВт·ч"
    )
    apply_mode = bool(opts.get("apply_mode", False))
    if apply_mode:
        head = f"План батареи на {issue.astimezone(tz):%d.%m}"
        head += (
            " — применён в инвертор:" if not apply_errors
            else " — применён частично (есть ошибки записи):"
        )
    else:
        head = f"Теневой план на {issue.astimezone(tz):%d.%m} (без записи в инвертор):"
    lines = [
        head,
        f"• Нагрузка 24 ч, кВт·ч: P50 {load_day['p50']} (P10 {load_day['p10']} … P90 {load_day['p90']})",
        pv_line,
        f"• Резерв отключений: {plan.reserve_kwh:.1f} кВт·ч "
        f"(критическая {plan.critical_load_kw:.2f} кВт × {plan.duration_hours:.1f} ч)"
        + (f", повышен держится до {hold_until}" if hold_until else "")
        + ("" if soh >= 0.999 else f", SoH батареи {soh:.0%}"),
        f"• Пол SOC плавающий: сейчас {recommended_floor:.0%}, "
        f"диапазон суток {min(floors):.0%}–{max(floors):.0%} "
        f"(инвертор держит {current_floor:.0%}, {delta_pp:+d} п.п.), батарея сейчас {battery_now}%",
    ]
    if pv_day2 is not None:
        deficit2 = round(load_day2["p50"] - pv_day2["p50"], 2)
        outlook = (
            "солнце перекрывает нагрузку"
            if deficit2 <= 0
            else f"дефицит {deficit2} кВт·ч — учтён в сегодняшнем плане"
        )
        lines.append(
            f"• Завтра: нагрузка P50 {load_day2['p50']} кВт·ч, солнце P50 {pv_day2['p50']} "
            f"кВт·ч — {outlook}"
        )
    if floor_warning:
        lines.append(
            f"• Внимание: {floor_warning} — текущий порог показан как минимум SOC"
        )
    if risk_windows:
        lines.append(
            f"• По журналу отключения чаще всего в {', '.join(risk_windows)} ч — "
            f"резерв в эти часы поднят ({risk.outages_count} отключений за "
            f"{risk.days_observed:.0f} дней наблюдений)"
        )
    if would_raise:
        zones = ", ".join(
            f"зона {p['n']} ({p['from'][:5]}) {p['soc_pct']:.0f}→{p['recommended_soc_pct']:.0f}%"
            for p in would_raise
        )
        lines.append(f"• Подняла бы SOC зон TOU: {zones}")
    else:
        lines.append("• Зоны TOU: без изменений")
    if optimizer_plan is not None:
        horizon_h = optimizer_plan.get("horizon_hours", HORIZON_HOURS)
        day1_cost = optimizer_plan.get("cost_plan_day1")
        if day1_cost is None:
            day1_cost = optimizer_plan["cost_plan"]
        opt_line = (
            f"• Оптимизатор (MILP, горизонт {horizon_h} ч): сегодня обойдётся в {day1_cost:.1f}"
        )
        savings_day1 = optimizer_plan.get("savings_day1")
        if savings_day1 is not None:
            opt_line += (
                f", без батареи было бы {optimizer_plan['cost_baseline_day1']:.1f} — "
                f"экономия {savings_day1:.1f}"
            )
        elif optimizer_plan.get("savings") is not None:
            opt_line += (
                f", без батареи было бы {optimizer_plan['cost_baseline']:.1f} — "
                f"экономия {optimizer_plan['savings']:.1f}"
            )
        if optimizer_plan.get("tariff_enabled"):
            opt_line += " (в тарифных единицах)"
        lines.append(opt_line)
        if optimizer_plan["grid_charge_kwh_cheap"] >= 0.3:
            lines.append(
                f"• Ночная зарядка из сети по дешёвому тарифу: "
                f"{optimizer_plan['grid_charge_kwh_cheap']:.1f} кВт·ч"
            )
        if optimizer_plan["floor_relaxed"]:
            lines.append(
                "• Оптимизатор: плавающий пол недостижим по прогнозу — "
                "план посчитан с плоским минимумом SOC"
            )
        zone_changes = [
            t
            for t in optimizer_plan["zone_targets"]
            if t.get("optimal_soc_pct") is not None
            and abs(t["optimal_soc_pct"] - t["soc_pct"]) >= 5
        ]
        if zone_changes:
            zones = ", ".join(
                f"зона {t['n']} ({t['from'][:5]}) {t['soc_pct']:.0f}→{t['optimal_soc_pct']:.0f}%"
                for t in zone_changes
            )
            lines.append(f"• Целевые SOC зон от оптимизатора: {zones}")
    elif opts.get("optimizer_enabled", True):
        lines.append("• Оптимизатор: расчёт не удался, действует эвристика пола SOC")
    if plan.shortfall_kwh > 0 or profile.max_shortfall_kwh > 0:
        lines.append(
            "• Внимание: резерв не помещается, нехватка "
            f"{max(plan.shortfall_kwh, profile.max_shortfall_kwh):.1f} кВт·ч"
        )
    if apply_mode:
        if applied:
            lines.append(f"• Записано в инвертор: {', '.join(applied)}")
        else:
            lines.append("• Инвертор: все целевые SOC совпадают с текущими — записей не было")
        if apply_errors:
            lines.append(f"• Ошибка записи: {', '.join(apply_errors)}")
    if models_learning:
        lines.append(
            "• Модели прогноза ещё обучаются на данных этого объекта — "
            "пока используется упрощённый прогноз; точность вырастет через несколько дней"
        )
    entity_note = _entity_issue_note(entity_checks)
    if entity_note is not None:
        lines.append(entity_note)
    if weather_stale and weather_name is not None:
        lines.append(
            f"• Погода: интернет недоступен, взят архивный снапшот {weather_name}"
        )
    retrain_note = _retrain_note()
    if retrain_note is not None:
        lines.append(retrain_note)
    pvf_note = _plan_vs_fact_note()
    if pvf_note is not None:
        lines.append(pvf_note)

    report = {
        "issue_utc": issue.isoformat(),
        "site": site,
        "apply_mode": apply_mode,
        "data": {
            "audit_rows_appended": appended,
            "audit_through": audit.index[-1].isoformat(),
            "grid_transitions_recorded": new_transitions,
            "weather_archive": weather_name,
            "weather_stale": weather_stale,
            "models_learning": models_learning,
            "battery_soh": round(soh, 4),
            "battery_soh_source": soh_source,
            "entity_checks": entity_checks,
            "plan_horizon_hours": PLAN_HOURS,
            "reserve_pv_aware": bool(opts.get("reserve_pv_aware", True)),
        },
        "hourly": {
            "starts_utc": [s.isoformat() for s in starts],
            "load_p10_kw": [round(float(load_prediction["load_p10_kw"].iloc[i]), 4) for i in range(HORIZON_HOURS)],
            "load_p50_kw": [round(float(load_prediction["load_p50_kw"].iloc[i]), 4) for i in range(HORIZON_HOURS)],
            "load_p90_kw": [round(float(load_prediction["load_p90_kw"].iloc[i]), 4) for i in range(HORIZON_HOURS)],
            "pv_p10_kw": [round(v, 4) for v in pv_p10[:HORIZON_HOURS]],
            "pv_p50_kw": [round(v, 4) for v in pv_p50[:HORIZON_HOURS]],
            "pv_p90_kw": [round(v, 4) for v in pv_p90[:HORIZON_HOURS]],
        },
        "load_forecast_kwh_24h": load_day,
        "pv_forecast_kwh_24h": pv_day,
        "load_forecast_kwh_day2": load_day2,
        "pv_forecast_kwh_day2": pv_day2,
        "reserve": {
            "reserve_kwh": round(plan.reserve_kwh, 3),
            "floor_soc": round(plan.floor_soc, 4),
            "critical_load_kw": round(plan.critical_load_kw, 3),
            "duration_hours": round(plan.duration_hours, 2),
            "hold_until_local": hold_until,
            "shortfall_kwh": round(plan.shortfall_kwh, 3),
            "soh": round(soh, 4),
        },
        "advisory": {
            "recommended_floor_soc": recommended_floor,
            "exact_floor_soc": round(profile.steps[0].floor_soc, 4),
            "floor_profile": [
                {
                    "start_local": profile.steps[i].start.astimezone(tz).strftime("%H:%M"),
                    "floor_soc": floors[i],
                    "crossover_local": (
                        profile.steps[i].crossover.astimezone(tz).strftime("%H:%M")
                        if profile.steps[i].crossover
                        else None
                    ),
                }
                for i in range(HORIZON_HOURS)
            ],
            "max_floor_soc": round(profile.max_floor_soc / 0.05) * 0.05,
            "energy_to_crossover_kwh": round(profile.steps[0].stored_kwh, 3),
            "crossover_utc": (
                profile.steps[0].crossover.isoformat() if profile.steps[0].crossover else None
            ),
            "current_floor_soc": current_floor,
            "delta_pp": delta_pp,
            "programs": programs,
            "would_raise_zones": would_raise,
            "optimizer": optimizer_plan,
            "applied": applied,
            "apply_errors": apply_errors,
        },
        "outage_risk": (
            {
                "hourly_probability": [round(p, 4) for p in risk.hourly_probability],
                "risk_hours": [h for h in range(24) if risk.risk_hours[h]],
                "risk_windows": risk_windows,
                "mean_duration_hours": round(risk.mean_duration_hours, 2),
                "duration_p90_hours": round(risk.duration_p90_hours, 2),
                "days_observed": round(risk.days_observed, 1),
                "outages_count": risk.outages_count,
            }
            if risk is not None
            else None
        ),
        "battery_now_pct": battery_now,
        "notification": "\n".join(lines),
    }
    out_dir = DATA / "shadow_reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{issue.astimezone(tz):%Y-%m-%d}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        mirror = SHARE / "shadow_reports"
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / out_path.name).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass

    report["data"]["metrics_scored_days"] = metrics_scored

    if opts.get("notifications", True):
        ha_service(
            "persistent_notification",
            "create",
            {
                "title": f"Теневой план батареи — {issue.astimezone(tz):%d.%m}",
                "message": report["notification"],
                "notification_id": "emhass_shadow_plan",
            },
        )
    return report
