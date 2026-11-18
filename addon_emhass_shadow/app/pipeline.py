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
import sys
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
from emhass.pv_quantiles import PvQuantileBundle, plane_power_kw
from emhass.reserve import LoadObservation, ReserveSettings, calculate_reserve

DATA = Path("/data")
SHARE = Path("/share/emhass_shadow")
HA_BASE = "http://supervisor/core"
SITE_ID = "home"
HORIZON_HOURS = 24
METRICS = DATA / "metrics.jsonl"

MEAN_SENSORS = {
    "load_kw": ("sensor.inverter_load_power", 0.001),
    "critical_load_kw": ("sensor.inverter_load_ups_power", 0.001),
    "pv_measured_kw": ("sensor.inverter_pv_power", 0.001),
    "battery_power_kw_raw_sign": ("sensor.inverter_battery_power", 0.001),
    "grid_power_kw_raw_sign": ("sensor.inverter_grid_power", 0.001),
    "temp_air_observed_c": ("sensor.home_temperature", 1.0),
}
SOC_SENSOR = "sensor.inverter_battery"
EXPORT_COUNTER = "sensor.inverter_total_energy_export"
VOLTAGE_SENSORS = {
    "grid_l1_min_v": "sensor.inverter_grid_l1_voltage",
    "grid_l2_min_v": "sensor.inverter_grid_l2_voltage",
    "grid_l3_min_v": "sensor.inverter_grid_l3_voltage",
}
GRID_BINARY = "binary_sensor.inverter_grid"

TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def options() -> dict:
    return json.loads((DATA / "options.json").read_text())


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


def build_audit_rows(history, hours, latitude, longitude):
    means = {
        field: numeric_samples(history.get(entity, []), scale)
        for field, (entity, scale) in MEAN_SENSORS.items()
    }
    soc = numeric_samples(history.get(SOC_SENSOR, []), 0.01)
    voltages = {
        field: numeric_samples(history.get(entity, []), 1.0)
        for field, entity in VOLTAGE_SENSORS.items()
    }
    grid_series = grid_status_series(history.get(GRID_BINARY, []))
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
        row["export_kwh"] = counter_delta(history.get(EXPORT_COUNTER, []), hour)
        for field, samples in voltages.items():
            row[field] = hourly_min(samples, hour)
        elevation = solar_elevation(latitude, longitude, hour)
        row["solar_elevation_deg"] = elevation

        in_hour = [s for at, s in grid_series if hour <= at < hour + timedelta(hours=1)]
        mins = [row[f] for f in VOLTAGE_SENSORS if row[f] is not None]
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


def update_audit(site) -> tuple[pd.DataFrame, int]:
    audit_path = DATA / "hourly_audit.jsonl"
    audit = pd.read_json(audit_path, lines=True)
    audit["start_utc"] = pd.to_datetime(audit["start_utc"], utc=True)
    audit = audit.set_index("start_utc").sort_index()
    last = audit.index[-1]
    now_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    if last >= now_hour - timedelta(hours=1):
        return audit, 0
    hours = []
    cursor = last + timedelta(hours=1)
    while cursor < now_hour:
        hours.append(cursor)
        cursor += timedelta(hours=1)
    entities = (
        [e for e, _ in MEAN_SENSORS.values()]
        + list(VOLTAGE_SENSORS.values())
        + [SOC_SENSOR, EXPORT_COUNTER, GRID_BINARY]
    )
    history = fetch_history(last, now_hour + timedelta(hours=1), entities)
    rows = build_audit_rows(history, hours, site["latitude"], site["longitude"])
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


def update_outage_log() -> int:
    now = datetime.now(UTC)
    history = fetch_history(now - timedelta(hours=48), now, [GRID_BINARY])
    series = grid_status_series(history.get(GRID_BINARY, []))
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


def run() -> dict:
    opts = options()
    site = {
        "latitude": opts["latitude"],
        "longitude": opts["longitude"],
        "time_zone": opts["time_zone"],
    }
    tz = ZoneInfo(site["time_zone"])

    audit, appended = update_audit(site)
    new_transitions = update_outage_log()

    issue = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    starts = pd.date_range(issue, periods=HORIZON_HOURS, freq="h")

    load_bundle = LoadQuantileBundle.load(DATA / "models" / "load_quantiles_v1")
    grid = audit[["temp_air_observed_c", "load_kw"]].copy()
    grid = grid.reindex(grid.index.union(starts))
    load_prediction = load_bundle.predict(grid, issue_time=issue).loc[starts]

    pv_bundle = PvQuantileBundle.load(DATA / "models" / "pv_quantiles_v1")
    try:
        weather, weather_name, weather_stale = archive_weather(site)
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
        ).reindex(starts)
        pv_prediction = pv_bundle.predict(pv_frame)
        pv_p10 = [float(pv_prediction["pv_p10_kw"].iloc[i]) for i in range(HORIZON_HOURS)]
        pv_p50 = [float(pv_prediction["pv_p50_kw"].iloc[i]) for i in range(HORIZON_HOURS)]
        pv_p90 = [float(pv_prediction["pv_p90_kw"].iloc[i]) for i in range(HORIZON_HOURS)]
        pv_day = {
            q: round(float(pv_prediction[f"pv_{q}_kw"].sum()), 2) for q in ("p10", "p50", "p90")
        }
    except OSError:
        # No internet and no archived vintage at all: assume zero solar, the
        # conservative direction for battery planning, and say so in the report.
        weather_name, weather_stale = None, True
        pv_p10 = [0.0] * HORIZON_HOURS
        pv_p50 = [0.0] * HORIZON_HOURS
        pv_p90 = [0.0] * HORIZON_HOURS
        pv_day = None

    critical = audit["critical_load_kw"].dropna()
    settings = ReserveSettings(
        nominal_capacity_kwh=opts["capacity_kwh"],
        soh=1.0,
        floor_soc=opts["min_soc"],
        max_soc=opts["max_soc"],
        discharge_efficiency=opts["discharge_efficiency"],
        cold_start_kw=float(critical.quantile(0.9)),
        cold_start_hours=2.0,
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
            start=starts[i].to_pydatetime(),
            hours=1.0,
            pv_p10_kw=pv_p10[i],
            load_p90_kw=float(load_prediction["load_p90_kw"].iloc[i]),
        )
        for i in range(HORIZON_HOURS)
    ]

    # Probabilistic outage layer: raise the reserve during historically risky
    # local hours instead of relying only on the post-outage reactive hold.
    risk = outage_risk_profile(events, issue, site["time_zone"])
    reserve_by_hour = [plan.reserve_kwh] * HORIZON_HOURS
    risk_windows: list[str] = []
    if risk is not None and any(risk.risk_hours):
        risk_energy = (
            plan.critical_load_kw * risk.duration_p90_hours / settings.discharge_efficiency
        )
        for i in range(HORIZON_HOURS):
            local_hour = starts[i].astimezone(tz).hour
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

    profile = floor_profile(steps, reserve_by_hour, settings)
    floors = [round(step.floor_soc / 0.05) * 0.05 for step in profile.steps]
    recommended_floor = floors[0]  # binding constraint is the current hour

    floor_entity = opts["floor_input_entity"]
    current_floor = float(ha_get(f"/api/states/{floor_entity}")["state"]) / 100.0
    programs = []
    for n in range(1, 7):
        soc = float(ha_get(f"/api/states/number.inverter_program_{n}_soc")["state"])
        start = ha_get(f"/api/states/time.inverter_program_{n}_time")["state"]
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

    battery_now = ha_get(f"/api/states/{SOC_SENSOR}")["state"]
    last_outage = max(
        (e.end for e in events if e.end is not None and e.source == "measured"), default=None
    )
    hold_until = (
        (last_outage + timedelta(days=14)).astimezone(tz).date().isoformat()
        if last_outage is not None
        else None
    )

    load_day = {
        q: round(float(load_prediction[f"load_{q}_kw"].sum()), 2) for q in ("p10", "p50", "p90")
    }
    delta_pp = round((recommended_floor - current_floor) * 100)

    pv_line = (
        f"• Солнце 24 ч, кВт·ч: P50 {pv_day['p50']} (пессимистично P10 {pv_day['p10']})"
        if pv_day is not None
        else "• Солнце: прогноз недоступен (нет интернета и архива) — считаем худший случай 0 кВт·ч"
    )
    lines = [
        f"Теневой план на {issue.astimezone(tz):%d.%m} (без записи в инвертор):",
        f"• Нагрузка 24 ч, кВт·ч: P50 {load_day['p50']} (P10 {load_day['p10']} … P90 {load_day['p90']})",
        pv_line,
        f"• Резерв отключений: {plan.reserve_kwh:.1f} кВт·ч "
        f"(критическая {plan.critical_load_kw:.2f} кВт × {plan.duration_hours:.1f} ч)"
        + (f", повышен держится до {hold_until}" if hold_until else ""),
        f"• Пол SOC плавающий: сейчас {recommended_floor:.0%}, "
        f"диапазон суток {min(floors):.0%}–{max(floors):.0%} "
        f"(инвертор держит {current_floor:.0%}, {delta_pp:+d} п.п.), батарея сейчас {battery_now}%",
    ]
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
    if plan.shortfall_kwh > 0 or profile.max_shortfall_kwh > 0:
        lines.append(
            "• Внимание: резерв не помещается, нехватка "
            f"{max(plan.shortfall_kwh, profile.max_shortfall_kwh):.1f} кВт·ч"
        )
    if weather_stale and weather_name is not None:
        lines.append(
            f"• Погода: интернет недоступен, взят архивный снапшот {weather_name}"
        )
    retrain_note = _retrain_note()
    if retrain_note is not None:
        lines.append(retrain_note)

    report = {
        "issue_utc": issue.isoformat(),
        "site": site,
        "data": {
            "audit_rows_appended": appended,
            "audit_through": audit.index[-1].isoformat(),
            "grid_transitions_recorded": new_transitions,
            "weather_archive": weather_name,
            "weather_stale": weather_stale,
        },
        "hourly": {
            "starts_utc": [s.isoformat() for s in starts],
            "load_p10_kw": [round(float(load_prediction["load_p10_kw"].iloc[i]), 4) for i in range(HORIZON_HOURS)],
            "load_p50_kw": [round(float(load_prediction["load_p50_kw"].iloc[i]), 4) for i in range(HORIZON_HOURS)],
            "load_p90_kw": [round(float(load_prediction["load_p90_kw"].iloc[i]), 4) for i in range(HORIZON_HOURS)],
            "pv_p10_kw": [round(v, 4) for v in pv_p10],
            "pv_p50_kw": [round(v, 4) for v in pv_p50],
            "pv_p90_kw": [round(v, 4) for v in pv_p90],
        },
        "load_forecast_kwh_24h": load_day,
        "pv_forecast_kwh_24h": pv_day,
        "reserve": {
            "reserve_kwh": round(plan.reserve_kwh, 3),
            "floor_soc": round(plan.floor_soc, 4),
            "critical_load_kw": round(plan.critical_load_kw, 3),
            "duration_hours": round(plan.duration_hours, 2),
            "hold_until_local": hold_until,
            "shortfall_kwh": round(plan.shortfall_kwh, 3),
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

    report["data"]["metrics_scored_days"] = update_metrics(audit, tz)

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
