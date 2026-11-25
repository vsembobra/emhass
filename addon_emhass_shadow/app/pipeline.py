"""Shadow pipeline for the EMHASS Shadow add-on.

Adapted from data/forecast_training/daily_shadow.py to run inside the add-on:
  * Home Assistant API through the Supervisor proxy (SUPERVISOR_TOKEN),
  * site/battery parameters from /data/options.json,
  * persistent data under /data, mirrors to /share/emhass_shadow,
  * summary delivered as a persistent notification in Home Assistant.

Nothing is written to the inverter: every Home Assistant call is a read,
except the persistent notification service call.
"""

import bisect
import csv
import json
import math
import os
import ssl
import threading
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import adapter
import calibration
import certifi
import data_store
import deferrable
import discovery
import hardware
import learning
import numpy as np
import optimize
import options_meta
import pandas as pd
import policy
import tuner
import weather_models

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

TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def stored_options() -> dict:
    """The options as saved (only what the owner set, plus older versions' leftovers)."""
    return json.loads((DATA / "options.json").read_text())


def options() -> dict:
    """Explicit owner settings over built-in defaults; no implicit discovery."""
    return discovery.load_options(DATA)


def option_sources() -> dict[str, str]:
    return discovery.sources(stored_options(), None)


def _plain_number(value) -> float | None:
    """A manual constant where an entity id was expected, else None.

    Options like the outdoor temperature accept "20" as a fixed value
    instead of a sensor; entity ids never parse as finite floats.
    """
    try:
        number = float(str(value or "").strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def resolve_entities(opts: dict) -> dict:
    """Entity map: every HA entity id comes from the add-on options.

    TOU zone entities use templates with {n} (zone number, 1..tou_zones)
    so a different naming scheme is one string instead of twelve.
    """
    caps = adapter.caps(opts)
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
            opts["tou_soc_entity_template"].format(n=n) for n in range(1, caps.zones + 1)
        ] if opts["tou_soc_entity_template"] and opts["tou_time_entity_template"] else [],
        "tou_time": [
            opts["tou_time_entity_template"].format(n=n) for n in range(1, caps.zones + 1)
        ] if opts["tou_soc_entity_template"] and opts["tou_time_entity_template"] else [],
        "tou_grid_charge": caps.grid_charge_entities(),
        "floor_input": opts["floor_input_entity"],
    }


def _token() -> str:
    return os.environ["SUPERVISOR_TOKEN"]


# One read of every HA state per planning run instead of one request per
# entity (~50 in sequence: entity checks, zone SOC/time, grid-charge flags,
# SOC, floor). Any service call that can change a state drops it, so the
# read-backs after a write and everything later read live again; a stale
# snapshot (a run that died midway) expires on its own.
STATES_MAX_AGE_S = 300.0
_STATES: dict[str, dict] | None = None
_STATES_AT = 0.0


def snapshot_states() -> None:
    global _STATES, _STATES_AT
    _STATES = None
    try:
        states = ha_get("/api/states", timeout=60)
    except Exception:
        return  # entity reads fall back to one request each
    if isinstance(states, list):
        _STATES = {s["entity_id"]: s for s in states if isinstance(s, dict) and "entity_id" in s}
        _STATES_AT = time.monotonic()


def drop_states() -> None:
    global _STATES
    _STATES = None


def ha_get(path: str, timeout: float = 180):
    states = _STATES
    if (
        states is not None
        and path.startswith("/api/states/")
        and time.monotonic() - _STATES_AT < STATES_MAX_AGE_S
    ):
        cached = states.get(path[len("/api/states/") :])
        if cached is not None:
            return cached
    request = urllib.request.Request(
        HA_BASE + path, headers={"Authorization": f"Bearer {_token()}"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def ha_service(domain: str, service: str, payload: dict) -> None:
    if domain != "persistent_notification":
        drop_states()  # a write: what follows must read the device live
    request = urllib.request.Request(
        f"{HA_BASE}/api/services/{domain}/{service}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60):
        pass


def notify_problem(key: str, title: str, message: str | None) -> None:
    """Raise (or clear, when message is None) a persistent error notification.

    One stable notification_id per problem kind: repeated failures update the
    same notification instead of stacking, and recovery dismisses it. Sent
    regardless of the plan-notification toggle — a broken input is exactly
    what the owner must see. Never raises: alerting must not break the plan.
    """
    try:
        if message is None:
            ha_service(
                "persistent_notification",
                "dismiss",
                {"notification_id": f"emhass_shadow_problem_{key}"},
            )
        else:
            ha_service(
                "persistent_notification",
                "create",
                {
                    "title": title,
                    "message": message,
                    "notification_id": f"emhass_shadow_problem_{key}",
                },
            )
    except Exception:
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
    samples.sort()
    return samples


def _carried(samples: list[tuple], hour: datetime):
    """Last sample value before the hour, as stored (numeric or status str)."""
    before = [value for at, value in samples if at < hour]
    return before[-1] if before else None


def hourly_mean(samples, hour):
    """Time-weighted mean over the hour, carrying the last pre-hour value.

    HA records state changes, so a plain mean of the samples inside the hour
    is wrong whenever the value changes mid-hour: 1 kW held for 59 minutes
    plus 5 kW for 1 minute would average to 3 kW instead of ≈1.07 kW. Each
    value is weighted by the time it was in effect inside the hour; the mean
    is normalized by the covered duration. No samples at all -> carried
    value, or None when nothing precedes the hour either.
    """
    end = hour + timedelta(hours=1)
    points = [(at, value) for at, value in samples if at < end]
    segments: list[tuple[datetime, datetime, float]] = []
    seg_start: datetime | None = None
    seg_value: float | None = None
    for at, value in points:
        if at <= hour:
            # Value in effect at the hour start (carried in).
            seg_start, seg_value = hour, value
        else:
            if seg_value is not None and seg_start is not None:
                segments.append((seg_start, at, seg_value))
            seg_start, seg_value = at, value
    if seg_value is not None and seg_start is not None:
        segments.append((seg_start, end, seg_value))
    if not segments:
        return None
    weighted = sum((b - a).total_seconds() * v for a, b, v in segments)
    covered = sum((b - a).total_seconds() for a, b, v in segments)
    return float(weighted / covered) if covered > 0 else None


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
    """(timestamp, status) transitions from HA history.

    "unknown"/"unavailable" states are kept as "unknown": dropping them would
    silently stretch the previous known state over recorder/sensor outages.
    An unknown blip shorter than 15 minutes that is immediately followed by a
    known state is swallowed — it is a reconnect artifact, not a grid event.
    Consecutive duplicate statuses collapse into one transition.
    """
    raw = []
    for state in states:
        status = {"on": "up", "off": "down"}.get(_state_value(state), "unknown")
        raw.append((_state_at(state), status))
    raw.sort()
    series = []
    for index, (at, status) in enumerate(raw):
        if status == "unknown":
            following = raw[index + 1] if index + 1 < len(raw) else None
            if (
                following is not None
                and following[1] != "unknown"
                and following[0] - at < timedelta(minutes=15)
            ):
                continue
        if series and series[-1][1] == status:
            continue
        series.append((at, status))
    return series


CURTAIL_FULL_SOC = 0.95  # battery full: nothing absorbs a PV surplus
CURTAIL_TAPER_SOC = 0.90  # BMS tapers charge current near full
CURTAIL_SATURATION = 0.9  # share of max charge power counted as "charging flat out"
FLOW_EPS_KW = 0.05  # hourly mean power below this is "no flow"


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def pv_sink_limited(
    soc_mean, export_kwh, battery_kw, grid_kw, max_charge_kw=None
) -> bool:
    """Whether measured PV was likely throttled for lack of a sink.

    With feed-in closed the inverter caps PV at load + battery charge, so
    measured PV is only a lower bound of what the array could produce.
    Exporting means a sink existed, so the hour is never flagged then;
    otherwise three signatures (sign convention: battery < 0 charging,
    grid > 0 importing):

    * battery full (SOC >= 95 %);
    * charge taper: SOC >= 90 %, battery charging, no grid import — the BMS
      limits the current, the surplus above it is throttled;
    * charge saturation: battery charging at >= 90 % of max_charge_kw with
      no grid import — load + max charge is the ceiling.

    On a no-export site export is always 0, so the former rule
    (SOC >= 95 % and no export) missed the taper and saturation hours; they
    reached the PV model as "cloudy" rows and pulled the forecast down.
    Values may be None/NaN (unknown): an unknown export never flags.
    """
    soc = _finite(soc_mean)
    export = _finite(export_kwh)
    if soc is None or export is None or export > 0.01:
        return False
    if soc >= CURTAIL_FULL_SOC:
        return True
    battery = _finite(battery_kw)
    charging_kw = -battery if battery is not None and battery < 0 else 0.0
    grid = _finite(grid_kw)
    if charging_kw <= FLOW_EPS_KW or (grid is not None and grid > FLOW_EPS_KW):
        return False
    if soc >= CURTAIL_TAPER_SOC:
        return True
    limit = _finite(max_charge_kw)
    return bool(limit) and charging_kw >= CURTAIL_SATURATION * limit


def pv_sink_limited_mask(audit: pd.DataFrame, max_charge_kw=None) -> pd.Series:
    """pv_sink_limited over stored audit rows (columns may be absent)."""

    def column(name: str) -> pd.Series:
        if name in audit:
            return audit[name]
        return pd.Series(np.nan, index=audit.index)

    return pd.Series(
        [
            pv_sink_limited(soc, export, battery, grid, max_charge_kw)
            for soc, export, battery, grid in zip(
                column("soc_mean"),
                column("export_kwh"),
                column("battery_power_kw_raw_sign"),
                column("grid_power_kw_raw_sign"),
                strict=True,
            )
        ],
        index=audit.index,
        dtype=bool,
    )


def build_audit_rows(history, hours, latitude, longitude, ent, max_charge_kw=None):
    means = {}
    constants = {}
    for field, (entity, scale) in ent["mean_sensors"].items():
        manual = _plain_number(entity)
        if manual is not None:
            constants[field] = manual * scale
        else:
            means[field] = numeric_samples(history.get(entity, []), scale)
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
        row.update(constants)
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
        carried = _carried(grid_series, hour)
        mins = [row[f] for f in ent["voltages"] if row[f] is not None]
        low_voltage = bool(mins) and min(mins) <= 50.0
        if history_start is not None and hour >= history_start.replace(
            minute=0, second=0, microsecond=0
        ):
            # The carried state matters: a long outage (or a long sensor
            # unavailability) has no new transitions inside the hour, but the
            # grid is still down (or still unobserved) — never mark it "up".
            if "down" in in_hour or carried == "down":
                row["raw_grid_status"] = "outage"
                row["grid_quality"] = (
                    "low_grid_voltage" if low_voltage else "recorded_outage"
                )
            elif "unknown" in in_hour or carried == "unknown":
                row["raw_grid_status"] = "unknown"
                row["grid_quality"] = "low_grid_voltage" if low_voltage else "unknown"
            else:
                row["raw_grid_status"] = "up"
                row["grid_quality"] = (
                    "low_grid_voltage" if low_voltage else "recorded_up"
                )
        elif not grid_series:
            row["raw_grid_status"] = "unknown"
            row["grid_quality"] = "unknown"
        else:
            row["raw_grid_status"] = "outside_history"
            row["grid_quality"] = "low_grid_voltage" if low_voltage else "voltage_not_flagged"

        apply_quality_filters(row, max_charge_kw)
        row["split"] = "live"
        rows.append(row)
    return rows


def apply_quality_filters(row: dict, max_charge_kw=None) -> None:
    """Training filters of one audit row (shared by recorder and statistics rows)."""
    bad_grid = row["grid_quality"] in ("recorded_outage", "low_grid_voltage", "unknown")
    if row.get("load_kw") is None or row.get("critical_load_kw") is None:
        row["load_filter"] = "invalid_target"
    elif bad_grid:
        row["load_filter"] = "grid_quality"
    else:
        row["load_filter"] = "keep"

    if row["solar_elevation_deg"] <= 10.0:
        row["pv_filter"] = "outside_daylight"
        row["possible_pv_curtailment"] = False
        return
    curtailment = pv_sink_limited(
        row.get("soc_mean"),
        row.get("export_kwh"),
        row.get("battery_power_kw_raw_sign"),
        row.get("grid_power_kw_raw_sign"),
        max_charge_kw,
    )
    row["possible_pv_curtailment"] = bool(curtailment)
    if row.get("pv_measured_kw") is None:
        row["pv_filter"] = "invalid_target"
    elif row.get("export_kwh") is None or row.get("soc_mean") is None:
        row["pv_filter"] = "missing_or_invalid_export"
    elif bad_grid:
        row["pv_filter"] = "grid_quality"
    elif curtailment:
        row["pv_filter"] = "possible_curtailment"
    else:
        row["pv_filter"] = "keep"


BACKFILL_CHUNK_DAYS = 3  # recorder fetches are chunked; one long range can time out


def _audit_entities(ent) -> list[str]:
    # Empty option values ("") and manual constants ("20") must never reach
    # the history query: a non-entity filter_entity_id entry makes HA answer
    # 400 Bad Request.
    return [
        e
        for e in (
            [e for e, _ in ent["mean_sensors"].values()]
            + list(ent["voltages"].values())
            + [ent["soc"], ent["export_counter"], ent["grid_binary"]]
        )
        if str(e or "").strip() and _plain_number(e) is None
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
            build_audit_rows(
                history,
                chunk_hours,
                site["latitude"],
                site["longitude"],
                ent,
                max_charge_kw=site.get("max_charge_kw"),
            )
        )
        cursor = chunk_end
    return rows


def _read_audit() -> pd.DataFrame:
    audit = data_store.read_frame(DATA / "hourly_audit.jsonl")
    # Writers disagree on spelling ("…Z" vs "+00:00"); mixed parses both.
    audit["start_utc"] = pd.to_datetime(audit["start_utc"], utc=True, format="mixed")
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
    if data_store.exists(audit_path):
        audit = _read_audit()
        first = audit.index[0]
    else:
        audit = pd.DataFrame()
        first = now_hour
    if first <= target:
        return 0
    hours = list(pd.date_range(target, first - timedelta(hours=1), freq="h"))
    rows = _rows_for_hours(site, ent, hours)
    return _prepend_rows(audit, rows)


def _prepend_rows(audit: pd.DataFrame, rows: list[dict]) -> int:
    """Merge older rows into the audit file (existing hours win)."""
    if not rows:
        return 0
    prepended = pd.DataFrame(rows)
    prepended["start_utc"] = pd.to_datetime(prepended["start_utc"], utc=True)
    combined = pd.concat([prepended.set_index("start_utc"), audit]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    data_store.write_text(DATA / "hourly_audit.jsonl",
        "".join(
            json.dumps(
                # Same spelling as build_audit_rows ("…Z"), so appends and
                # rewrites never mix timestamp formats in one file.
                {"start_utc": at.isoformat().replace("+00:00", ".000Z"), **row.dropna().to_dict()},
                default=_json_default,
            )
            + "\n"
            for at, row in combined.iterrows()
        ),
    )
    return len(rows)


# -- long-term statistics: months of history beyond the recorder purge -------

SUPERVISOR_WS = "ws://supervisor/core/websocket"
STATISTICS_CHUNK_DAYS = 30
STATISTICS_UNITS = {"power": "W", "energy": "kWh", "temperature": "°C", "voltage": "V"}


async def _ws_request(command: dict, timeout: float = 180):
    """One command over the Supervisor WebSocket proxy; its result."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(SUPERVISOR_WS, max_msg_size=0, heartbeat=30) as ws:
            await ws.receive_json(timeout=30)  # auth_required
            await ws.send_json({"type": "auth", "access_token": _token()})
            auth = await ws.receive_json(timeout=30)
            if auth.get("type") != "auth_ok":
                raise PermissionError(f"Home Assistant WebSocket auth failed: {auth.get('type')}")
            await ws.send_json({"id": 1, **command})
            while True:
                message = await ws.receive_json(timeout=timeout)
                if message.get("id") != 1 or message.get("type") != "result":
                    continue
                if not message.get("success"):
                    raise RuntimeError(f"{command['type']} failed: {message.get('error')}")
                return message.get("result")


async def _ws_statistics(start: datetime, end: datetime, ids: list[str]) -> dict:
    """recorder/statistics_during_period over the Supervisor WebSocket proxy."""
    result = await _ws_request(
        {
            "type": "recorder/statistics_during_period",
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "statistic_ids": ids,
            "period": "hour",
            "types": ["mean", "min", "max", "change"],
            "units": STATISTICS_UNITS,
        }
    )
    return result or {}


DISCOVERY_REGISTRY_TIMEOUT_S = 15.0


def search_entities() -> dict:
    """Explicit preview only; at most 15 + 10 + 5 seconds of HA reads."""
    import asyncio

    import aiohttp

    async def preview():
        # Total budget includes authentication, reads and parsing. Socket-idle
        # timeouts alone could wait indefinitely on a slow streaming response.
        async with asyncio.timeout(30):
            registry = await asyncio.wait_for(
                _ws_request({"type": "config/entity_registry/list"}, timeout=10),
                timeout=DISCOVERY_REGISTRY_TIMEOUT_S,
            )
            async with aiohttp.ClientSession() as session:
                async def read(path, seconds):
                    async with session.get(
                        HA_BASE + path, headers={"Authorization": f"Bearer {_token()}"},
                        timeout=aiohttp.ClientTimeout(total=seconds),
                    ) as response:
                        response.raise_for_status()
                        return await response.json()

                rows = await read("/api/states", 10)
                try:
                    core = await read("/api/config", 5)
                except (aiohttp.ClientError, TimeoutError, ValueError):
                    core = None
            if not isinstance(registry, list) or not isinstance(rows, list):
                raise ValueError("Home Assistant returned an invalid registry or state list")
            states = {s["entity_id"]: s for s in rows if isinstance(s, dict) and s.get("entity_id")}
            return discovery.discover(registry, states, core if isinstance(core, dict) else None)

    return asyncio.run(preview())


def fetch_statistics(start: datetime, end: datetime, ids: list[str]) -> dict[str, list[dict]]:
    """Hourly long-term statistics {statistic_id: [rows]}; sensors only."""
    import asyncio

    ids = [i for i in ids if str(i or "").strip() and str(i).startswith("sensor.")]
    if not ids:
        return {}
    return asyncio.run(_ws_statistics(start, end, ids))


def _stat_hour(value) -> datetime:
    """Statistics row start: epoch milliseconds (new HA) or ISO text (old)."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000.0, UTC)
    return datetime.fromisoformat(str(value)).astimezone(UTC)


def statistics_rows(
    stats: dict[str, list[dict]],
    hours: list[datetime],
    ent: dict,
    latitude: float,
    longitude: float,
    max_charge_kw=None,
) -> list[dict]:
    """Audit rows for hours covered by long-term statistics only.

    Power means (converted to W) feed the same kW fields as the recorder
    rows, SOC mean/max become fractions, the export counter's hourly change
    is export_kwh, and the grid-voltage minimum classifies the grid. Binary
    sensors keep no statistics: without a voltage sensor the grid is assumed
    up ("lts_assumed_up") — a few outage hours may then train the load model
    with backup-only consumption, which the next recorder rows outweigh.
    """
    by_entity = {
        entity: {_stat_hour(row["start"]): row for row in rows} for entity, rows in stats.items()
    }

    def stat(entity, key):
        row = by_entity.get(entity, {}).get(current)
        value = None if row is None else row.get(key)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

    import pvlib

    if not hours:
        return []
    midpoints = pd.DatetimeIndex(hours) + pd.Timedelta(minutes=30)
    elevations = (
        pvlib.location.Location(latitude, longitude)
        .get_solarposition(midpoints)["apparent_elevation"]
        .to_numpy()
    )
    voltage_entity = ent["voltages"].get("grid_min_v")
    has_voltage = bool(voltage_entity and by_entity.get(voltage_entity))
    rows = []
    for current, elevation in zip(hours, elevations, strict=True):
        row = {"start_utc": current.isoformat().replace("+00:00", ".000Z")}
        for field, (entity, scale) in ent["mean_sensors"].items():
            manual = _plain_number(entity)
            if manual is not None:
                row[field] = manual * scale
                continue
            value = stat(entity, "mean")
            row[field] = None if value is None else value * scale
        soc_mean, soc_max = stat(ent["soc"], "mean"), stat(ent["soc"], "max")
        row["soc_mean"] = None if soc_mean is None else soc_mean / 100.0
        row["soc_max"] = None if soc_max is None else soc_max / 100.0
        change = stat(ent["export_counter"], "change")
        row["export_kwh"] = None if change is None else max(0.0, change)
        row["solar_elevation_deg"] = float(elevation)
        if has_voltage:
            low = stat(voltage_entity, "min")
            row["grid_min_v"] = low
            if low is None:
                row["raw_grid_status"], row["grid_quality"] = "unknown", "unknown"
            elif low <= 50.0:
                row["raw_grid_status"], row["grid_quality"] = "outage", "low_grid_voltage"
            else:
                row["raw_grid_status"], row["grid_quality"] = "up", "recorded_up"
        else:
            row["raw_grid_status"], row["grid_quality"] = "up", "lts_assumed_up"
        if row.get("load_kw") is None and row.get("pv_measured_kw") is None:
            continue  # no statistics for this hour at all
        apply_quality_filters(row, max_charge_kw)
        row["split"] = "statistics"
        rows.append(row)
    return rows


def backfill_statistics(site, ent, days: float) -> int:
    """Prepend months of long-term statistics before the audit's first hour.

    Idempotent like the recorder backfill: hours already in the audit are
    never replaced. Returns the number of rows added.
    """
    audit_path = DATA / "hourly_audit.jsonl"
    if days <= 0:
        return 0
    now_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    target = now_hour - timedelta(days=days)
    audit = _read_audit() if data_store.exists(audit_path) else pd.DataFrame()
    first = audit.index[0] if len(audit) else now_hour
    if first <= target:
        return 0
    ids = _audit_entities(ent)
    rows: list[dict] = []
    cursor = target
    while cursor < first:
        chunk_end = min(cursor + timedelta(days=STATISTICS_CHUNK_DAYS), first)
        hours = list(pd.date_range(cursor, chunk_end - timedelta(hours=1), freq="h"))
        stats = fetch_statistics(cursor, chunk_end, ids)
        rows.extend(
            statistics_rows(
                stats,
                [h.to_pydatetime() for h in hours],
                ent,
                site["latitude"],
                site["longitude"],
                site.get("max_charge_kw"),
            )
        )
        cursor = chunk_end
    return _prepend_rows(audit, rows)


def bootstrap(opts: dict) -> dict:
    """First start: pull everything Home Assistant already knows.

    Recorder history (full detail, recorder depth) first, then the outage
    journal over the same depth, then long-term statistics for the months
    before. The caller retrains right after, so the very first plan already
    runs on models of this site.
    """
    site = {
        "latitude": opts["latitude"],
        "longitude": opts["longitude"],
        "time_zone": opts["time_zone"],
        "max_charge_kw": opts.get("max_charge_kw"),
    }
    ent = resolve_entities(opts)
    days = float(opts.get("history_backfill_days", 14))
    result: dict = {"at": datetime.now(UTC).isoformat()}
    _, result["recorder_rows"] = update_audit(site, ent, backfill_days=days)
    try:
        result["outage_transitions"] = update_outage_log(ent, hours=max(48.0, days * 24))
    except Exception as error:
        result["outage_error"] = str(error)[:200]
    try:
        result["statistics_rows"] = backfill_statistics(
            site, ent, float(opts.get("history_statistics_days", 365))
        )
    except Exception as error:
        result["statistics_rows"] = 0
        result["statistics_error"] = str(error)[:200]
    return result


def update_audit(site, ent, backfill_days: float = 1.0) -> tuple[pd.DataFrame, int]:
    audit_path = DATA / "hourly_audit.jsonl"
    now_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    if data_store.exists(audit_path):
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
    data_store.append_jsonl(audit_path, rows)
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


def _heartbeat_samples(history, ent) -> list[datetime]:
    """Timestamps proving the recorder was alive.

    sensor.time changes every minute; the load power sensor changes in
    practice every few seconds; sun.sun is the last-resort fallback. The
    first candidate with any history wins.
    """
    candidates = ["sensor.time"]
    load = ent.get("mean_sensors", {}).get("load_kw")
    if load:
        candidates.append(load[0])
    candidates.append("sun.sun")
    for candidate in candidates:
        states = history.get(candidate, [])
        if states:
            return sorted(_state_at(state) for state in states)
    return []


def _grid_carry_rows(series, heartbeats, now) -> list[tuple[datetime, str]]:
    """Hourly carried grid status between transitions.

    Without periodic rows a quiet grid (or one long outage) produces gaps
    larger than max_gap; record_observation_csv then resets to "unknown" and
    the next real outage is recorded interrupted — invisible to the reserve
    statistics. Synthesize one carry row per hour while a known status
    persists, but only where the heartbeat proves the recorder was alive: a
    heartbeat gap > 2 h is real observation loss, and inventing rows there
    would fake knowledge we never had.
    """
    if not series or not heartbeats:
        return []
    beats = sorted(heartbeats)

    def observed(at: datetime) -> bool:
        index = bisect.bisect_right(beats, at)
        return index > 0 and at - beats[index - 1] <= timedelta(hours=2)

    rows = []
    for index, (at, status) in enumerate(series):
        if status == "unknown":
            continue
        end = series[index + 1][0] if index + 1 < len(series) else now
        hour = (at + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        while hour < end and hour <= now:
            if observed(hour):
                rows.append((hour, status))
            hour += timedelta(hours=1)
    return rows


def update_outage_log(ent, hours: float = 48.0) -> int:
    """Record grid transitions of the last ``hours`` into the outage journal.

    The daily run looks back 48 h; a fresh install (or the backfill button)
    passes the recorder depth so past outages count at once. Long ranges
    are fetched in chunks.
    """
    grid_binary = str(ent.get("grid_binary") or "").strip()
    if not grid_binary:
        return 0
    now = datetime.now(UTC)
    fetch_entities = [grid_binary, "sensor.time", "sun.sun"]
    load = ent.get("mean_sensors", {}).get("load_kw")
    # The load sensor is the heartbeat on sites without sensor.time (sun.sun
    # changes only twice a day): chatty, but read in chunks.
    if load and load[0] not in fetch_entities:
        fetch_entities.append(load[0])
    history: dict[str, list[dict]] = {}
    cursor = now - timedelta(hours=hours)
    if hours <= 48:
        # Routine runs (hourly) only need what is new since the journal's
        # last observation, plus a little overlap for the carry rows.
        last = _last_observation()
        if last is not None:
            cursor = max(cursor, last - timedelta(hours=2))
    while cursor < now:
        chunk_end = min(cursor + timedelta(days=BACKFILL_CHUNK_DAYS), now)
        for entity, states in fetch_history(cursor, chunk_end, fetch_entities).items():
            history.setdefault(entity, []).extend(states)
        cursor = chunk_end
    transitions = grid_status_series(history.get(grid_binary, []))
    carry = _grid_carry_rows(transitions, _heartbeat_samples(history, ent), now)
    # A transition wins over a synthesized carry row at an equal timestamp —
    # it is the actual change. Dict keys keep timestamps unique.
    merged = sorted({**dict(carry), **dict(transitions)}.items())
    cursor_at = _last_observation()
    if cursor_at is not None:
        merged = [sample for sample in merged if sample[0] > cursor_at]
    if merged:
        csv_path = DATA / "grid_observations_tmp.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "status"])
            for at, status in merged:
                writer.writerow([at.isoformat(), status])
        # max_gap is small on purpose: carry rows keep observed periods dense,
        # so a gap > 2 h means the recorder was really blind.
        record_observation_csv(
            DATA / "outages.sqlite", SITE_ID, csv_path, max_gap=timedelta(hours=2)
        )
        csv_path.unlink()
    return len(merged)


def archive_weather(site) -> tuple[pd.DataFrame, str, bool]:
    """Fetch a fresh Open-Meteo vintage; fall back to the newest archived one.

    The fallback is explicit and labelled: when the grid (and with it the WAN
    link) is down, the report still ships, marked with the archive vintage.
    """
    variables = (
        "temperature_2m,shortwave_radiation,direct_normal_irradiance,"
        "diffuse_radiation,cloud_cover"
    )
    model = weather_models.selected(site)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={site['latitude']}&longitude={site['longitude']}"
        f"&hourly={variables}&past_hours=48&forecast_hours=49&timezone=UTC&timeformat=unixtime"
    )
    params = weather_models.parameters(site)
    if params:
        url += "&" + urllib.parse.urlencode(params)
    archive = weather_models.scoped(DATA / "weather_archive", model)
    archive.mkdir(parents=True, exist_ok=True)
    stale = False
    try:
        with urllib.request.urlopen(url, timeout=30, context=TLS_CONTEXT) as response:
            body = response.read().decode("utf-8")
        received = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        data_store.write_text(archive / f"{received}.json", body)
        payload = json.loads(body)
        prune_weather_archive(archive)
    except OSError:
        files = data_store.json_paths(archive)
        if not files:
            raise
        stale = True
        received = files[-1].stem
        payload = data_store.read_json(files[-1])
    return _weather_frame(payload), received + ".json", stale


def _weather_frame(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload["hourly"])
    frame["start_utc"] = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
    return frame.set_index("start_utc").sort_index()


def archived_weather() -> pd.DataFrame | None:
    """The newest stored Open-Meteo snapshot, without asking the internet."""
    archive = weather_models.scoped(DATA / "weather_archive", weather_models.from_data(DATA))
    files = data_store.json_paths(archive)
    if not files:
        return None
    try:
        return _weather_frame(data_store.read_json(files[-1]))
    except (OSError, ValueError, KeyError):
        return None


def forecast_load(
    bundle: LoadQuantileBundle,
    audit: pd.DataFrame,
    starts_plan: pd.DatetimeIndex,
    issue: datetime,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Causal 48 h load quantiles: day 1 direct, day 2 recursive.

    The model's 24 h features (load lag/rolling mean, temperature lag) for
    day 2 read day-1 hours, which are still in the future at issue time.
    Left as NaN — a value never seen in training — they degraded day 2 to a
    calendar + lag-168 guess, and day 2 is what drives tonight's charging.
    Day-1 P50 stands in for the missing load actuals, the weather forecast
    for the missing temperatures.

    Temperature gaps (no outdoor sensor, sensor outages) are filled from
    Open-Meteo on the whole grid: the fetch carries past hours, so day 1's
    temp lag reads the recent past too, matching training, which fills the
    same gaps from the previous-runs archive. Observed values always win.
    """
    day1_index = starts_plan[:HORIZON_HOURS]
    day2_index = starts_plan[HORIZON_HOURS:]
    grid = audit.reindex(columns=["temp_air_observed_c", "load_kw"]).apply(
        pd.to_numeric, errors="coerce"
    )
    grid = grid.reindex(grid.index.union(starts_plan))
    if weather is not None and "temperature_2m" in weather:
        forecast_temp = pd.to_numeric(weather["temperature_2m"], errors="coerce").reindex(grid.index)
        grid["temp_air_observed_c"] = grid["temp_air_observed_c"].fillna(forecast_temp)
        # As-of forecast for the target hour (used by lead-aware bundles that
        # opted into forecast_temperature; ignored by older ones).
        grid["temp_forecast_c"] = forecast_temp
    day1 = bundle.predict(grid.loc[: day1_index[-1]], issue_time=issue).loc[day1_index]
    if not len(day2_index):
        return day1
    grid.loc[day1_index, "load_kw"] = day1["load_p50_kw"].to_numpy()
    day2 = bundle.predict(
        grid, issue_time=issue + pd.Timedelta(hours=HORIZON_HOURS)
    ).loc[day2_index]
    return pd.concat([day1, day2])


WEATHER_ARCHIVE_DAYS = 7  # the archive only backs the offline fallback


def prune_weather_archive(archive: Path, keep_days: float = WEATHER_ARCHIVE_DAYS) -> int:
    """Delete snapshots older than keep_days (hourly runs add 24 a day);
    the newest one always stays. Returns the number removed."""
    files = data_store.json_paths(archive)
    cutoff = datetime.now(UTC) - timedelta(days=keep_days)
    removed = 0
    for path in files[:-1]:
        try:
            received = datetime.strptime(path.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue  # seeded or foreign files are left alone
        if received < cutoff:
            data_store.remove(path)
            removed += 1
    return removed


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


def hindsight_import(
    report: dict, audit: pd.DataFrame, starts: pd.DatetimeIndex, opts: dict
) -> float | None:
    """Grid import the same plan would have needed with perfect forecasts.

    Re-solves the day with the measured load and PV, the same starting SOC
    and floors. planned -> hindsight is the cost of forecast error,
    hindsight -> metered is the cost of execution (the inverter or the
    zone translation not following the plan).

    The plan looked 48 h ahead, so the re-solve does too, with the report's
    own forecast standing in for the next day: a 24 h re-solve ends the
    horizon at the day's end, spends the battery differently and showed
    1-3 kWh of "forecast error" even with a perfect forecast.
    """
    load = pd.to_numeric(audit["load_kw"], errors="coerce").reindex(starts)
    pv = pd.to_numeric(audit["pv_measured_kw"], errors="coerce").reindex(starts)
    soc = pd.to_numeric(audit["soc_mean"], errors="coerce").reindex(starts[:1])
    if load.isna().sum() > 4 or pv.isna().sum() > 4 or soc.isna().all():
        return None
    hourly = report["hourly"]
    load = load.fillna(pd.Series(hourly["load_p50_kw"], index=starts))
    pv = pv.fillna(pd.Series(hourly["pv_p50_kw"], index=starts))
    floors = [step["floor_soc"] for step in report["advisory"]["floor_profile"]]
    horizon = starts.append(starts + pd.Timedelta(hours=len(starts)))
    plan = optimize.run_plan(
        horizon,
        list(pv) + list(hourly["pv_p50_kw"]),
        list(load) + list(hourly["load_p50_kw"]),
        floors * 2,
        float(soc.iloc[0]),
        [],
        opts,
    )
    if plan is None or not plan.get("p_grid_kw"):
        return None
    return round(sum(max(0.0, float(v)) for v in plan["p_grid_kw"][: len(starts)]), 2)


def update_metrics(audit: pd.DataFrame, tz: ZoneInfo, opts: dict | None = None) -> int:
    """Score fully observed past reports against audit actuals, append once."""
    entries = []
    if data_store.exists(METRICS):
        entries = [
            json.loads(line)
            for line in data_store.read_text(METRICS).splitlines()
            if line.strip()
        ]
    scored = {entry["issue_utc"] for entry in entries}
    audit_end = audit.index[-1]
    added = 0
    reports_dir = DATA / "shadow_reports"
    for path in data_store.json_paths(reports_dir)[-10:]:
        try:
            report = data_store.read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        hourly = report.get("hourly")
        if not hourly or report["issue_utc"] in scored:
            continue
        starts = pd.DatetimeIndex(pd.to_datetime(hourly["starts_utc"], utc=True))
        if len(starts) != HORIZON_HOURS or starts[-1] > audit_end:
            continue  # horizon not fully observed yet
        actual_load = audit["load_kw"].where(learning.load_target_mask(audit)).reindex(starts)
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
                # The night margin is bought on top of the median plan's import.
                margin = ((report.get("advisory") or {}).get("optimizer") or {}).get(
                    "night_margin_kwh", 0.0
                )
                planned = round(
                    sum(max(0.0, float(v)) for v in plan_grid[:HORIZON_HOURS])
                    + float(margin or 0.0),
                    2,
                )
                actual = round(float(actual_import.sum()), 2)
                entry["plan_vs_fact"] = {"planned_grid_kwh": planned, "actual_grid_kwh": actual}
                hindsight = None
                if opts is not None:
                    try:
                        hindsight = hindsight_import(report, audit, starts, opts)
                    except Exception:
                        hindsight = None
                if hindsight is not None:
                    entry["plan_vs_fact"].update(
                        hindsight_grid_kwh=hindsight,
                        forecast_effect_kwh=round(hindsight - planned, 2),
                        execution_effect_kwh=round(actual - hindsight, 2),
                    )
        entries.append(entry)
        added += 1
    if added:
        entries = entries[-60:]
        data_store.write_text(METRICS,
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
        )
        try:
            SHARE.mkdir(parents=True, exist_ok=True)
            (SHARE / "metrics.jsonl").write_text(data_store.read_text(METRICS), encoding="utf-8")
        except OSError:
            pass
    return added


def _retrain_note() -> str | None:
    """One notification line when models were retrained in the last hours."""
    state_path = weather_models.scoped(DATA / "retrain_state.json", weather_models.from_data(DATA))
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
        status = info.get("status")
        if status in ("ok", "promoted"):
            return f"обновлена ({info.get('train_rows', '?')} строк)"
        if status == "kept":
            return f"оставлена текущая — {info.get('reason', '')}"
        return f"ошибка: {info.get('error', '?')[:80]}"

    models = state.get("models", {})
    return (
        f"• Переобучение моделей: нагрузка — {fmt(models.get('load', {}))}, "
        f"солнце — {fmt(models.get('pv', {}))}"
    )


RISK_MIN_EVENTS = 3  # outages of the live series before its hours count as risky
RISK_MIN_DAYS = 2.0  # ... spread over at least this many days


def _reserve_line(regime: dict, plan, margin_hours: float, hold_until, autonomy_h: float) -> str:
    """Notification line: outage series or quiet, and what is held for it."""
    if regime["active"]:
        line = (
            f"• Серия отключений идёт (последнее {regime['days_since_last']:.1f} дн. назад): "
            f"резерв {plan.reserve_kwh:.1f} кВт·ч — критическая {plan.critical_load_kw:.2f} кВт × "
            f"{plan.duration_hours:.1f} ч"
        )
        if margin_hours:
            line += f" + запас {margin_hours:.0f} ч"
        if hold_until:
            line += f", держится до {hold_until}"
        return line
    since = regime.get("days_since_last")
    quiet = (
        f"отключений не было {since:.0f} дн." if since is not None else "отключений не было"
    )
    if autonomy_h > 0:
        return (
            f"• {quiet.capitalize()} — резерв под отключения не держим; "
            f"минимальная автономия {autonomy_h:.0f} ч ({plan.reserve_kwh:.1f} кВт·ч)"
        )
    return f"• {quiet.capitalize()} — резерв под отключения не держим, только min SOC"


def _tuning_note(tuning: dict | None, margin: float, autotune: bool, series: bool) -> str | None:
    """One notification line: what the replay on the site's history chose."""
    if not tuning or tuning.get("best") is None:
        return None
    # Defense in depth: a foreign/legacy tuning file (string margin names)
    # yields no note instead of crashing the whole run.
    if not isinstance(tuning["best"], (int, float)) or str(tuning["best"]) not in tuning.get(
        "candidates", {}
    ):
        return None
    best = tuning["candidates"][str(tuning["best"])]
    line = (
        f"• Подбор на истории ({tuning['windows']} сут): запас сверх резерва +{float(tuning['best']):.0f} ч "
        f"(недоданная критическая {best['unserved_critical_kwh']:.1f} кВт·ч, "
        f"усечение {best['curtailed_kwh']:.1f} кВт·ч, стоимость {best['net_cost']:.1f})"
    )
    if not autotune:
        return line + " — совет, автонастройка выключена"
    if not series:
        return line + " — сейчас тишина, запас не нужен"
    return line + f" — применён: +{margin:.0f} ч"


def readback_ok(entity: str, expected, step_pct: float) -> bool:
    """Whether the entity now holds the written value.

    Numbers match within half an inverter step (a device may store its own
    rounding of the value, never a whole step off); switch states must be
    exactly "on"/"off" as written.
    """
    try:
        state = ha_get(f"/api/states/{entity}")["state"]
    except Exception:
        return False
    if isinstance(expected, str):
        return str(state) == expected
    try:
        return abs(float(state) - float(expected)) < step_pct / 2.0
    except (TypeError, ValueError):
        return False


def autonomy(
    soc: float, min_soc: float, capacity_kwh: float, discharge_efficiency: float, critical_kw: float
) -> float | None:
    """Hours the battery carries the critical load from now, grid off, no sun."""
    if critical_kw <= 0.05 or capacity_kwh <= 0:
        return None
    return max(0.0, soc - min_soc) * capacity_kwh * discharge_efficiency / critical_kw


def outage_now_line(info: dict) -> str:
    """Notification line for the outage in progress."""
    if info["basis"] == "beyond":
        outlook = "дольше всех отключений этой серии — смотрю на час вперёд и пересчитываю"
    elif info["basis"] == "few":
        outlook = f"по самым длинным отключениям серии — ещё до ~{info['remaining_h']:.1f} ч"
    else:
        outlook = (
            f"по этой серии обычно ещё ~{info['remaining_typical_h']:.1f} ч "
            f"(с запасом {info['remaining_h']:.1f} ч)"
        )
    line = (
        f"• ⚠️ Сеть отключена {info['elapsed_h']:.1f} ч: {outlook}. "
        f"Батареи хватит на ~{info['autonomy_h']:.1f} ч"
    )
    if info["enough"]:
        return line + " — хватает."
    return (
        line + f" — не хватает ~{info['shortfall_kwh']:.1f} кВт·ч: снизьте нагрузку примерно "
        f"на {info['cut_kw']:.2f} кВт до возврата сети."
    )


def _deferrable_line(item: dict) -> str:
    if not item.get("start_local"):
        return f"• {item['name']}: {item.get('reason', 'нет окна')}"
    if not item["sunny"]:
        return (
            f"• {item['name']}: солнечного окна нет, лучшее {item['start_local']}–{item['end_local']} "
            f"(солнце {item['solar_share']:.0%})"
        )
    line = (
        f"• {item['name']}: включить {item['start_local']}–{item['end_local']} "
        f"(на солнце {item['solar_share']:.0%}"
    )
    if item["free_solar_kwh"] >= 0.1:
        line += f", из них {item['free_solar_kwh']:.1f} кВт·ч ушли бы в усечение"
    return line + ")"


def ha_post_state(entity_id: str, state, attributes: dict) -> None:
    """Create/update a Home Assistant state (not a device write)."""
    request = urllib.request.Request(
        f"{HA_BASE}/api/states/{entity_id}",
        data=json.dumps({"state": state, "attributes": attributes}, default=_json_default).encode(),
        headers={"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30):
        pass


def weather_hour(weather: pd.DataFrame | None, hour) -> dict | None:
    """Weather API values for the hour [hour, hour+1h), as the PV model reads
    them (pv_weather_frame): radiation stamped hour+1h, instants averaged."""
    if weather is None:
        return None
    hour = pd.Timestamp(hour)
    after = hour + pd.Timedelta(hours=1)

    def at(name: str, stamp) -> float | None:
        if name not in weather:
            return None
        value = pd.to_numeric(weather[name], errors="coerce").get(stamp)
        return None if value is None or pd.isna(value) else float(value)

    def mean_of_ends(name: str) -> float | None:
        values = [v for v in (at(name, hour), at(name, after)) if v is not None]
        return round(sum(values) / len(values), 1) if values else None

    radiation = at("shortwave_radiation", after)
    return {
        "irradiance_wm2": None if radiation is None else round(radiation, 1),
        "cloud_cover_pct": mean_of_ends("cloud_cover"),
        "temperature_c": mean_of_ends("temperature_2m"),
    }


def _day_ahead_hour(issue) -> dict | None:
    try:
        return learning.day_ahead(learning.read_forecast_log(DATA / "forecast_log.jsonl"), issue)
    except Exception:
        return None


def _forecast_sensors(report: dict) -> list[tuple[str, object, dict]]:
    """This hour's forecast, the day-old forecast for it and the weather API
    values, with state_class so HA keeps their hourly statistics for good:
    compare them in a statistics graph with the inverter's own sensors.
    The state changes at each replan (shortly after the hour starts)."""
    hourly = report.get("hourly") or {}
    hour = (hourly.get("starts_utc") or [None])[0]

    def watts(kw) -> object:
        return "unknown" if kw is None else round(float(kw) * 1000.0)

    def first(key: str):
        values = hourly.get(key) or []
        return values[0] if values else None

    def numeric(value) -> object:
        return "unknown" if value is None else value

    power = {"unit_of_measurement": "W", "device_class": "power", "state_class": "measurement"}
    ahead = report.get("day_ahead_hour") or {}
    weather = report.get("weather_hour") or {}
    name = "EMHASS Shadow: "
    return [
        ("sensor.emhass_shadow_pv_forecast", watts(first("pv_p50_kw")),
         {**power, "hour_start_utc": hour, "friendly_name": name + "солнце, прогноз на этот час"}),
        ("sensor.emhass_shadow_pv_forecast_p10", watts(first("pv_p10_kw")),
         {**power, "hour_start_utc": hour,
          "friendly_name": name + "солнце, пессимистичный прогноз на этот час"}),
        ("sensor.emhass_shadow_load_forecast", watts(first("load_p50_kw")),
         {**power, "hour_start_utc": hour, "friendly_name": name + "нагрузка, прогноз на этот час"}),
        ("sensor.emhass_shadow_pv_forecast_day_ahead", watts(ahead.get("pv_p50_kw")),
         {**power, "issue_utc": ahead.get("issue_utc"),
          "friendly_name": name + "солнце, прогноз на этот час за сутки"}),
        ("sensor.emhass_shadow_load_forecast_day_ahead", watts(ahead.get("load_p50_kw")),
         {**power, "issue_utc": ahead.get("issue_utc"),
          "friendly_name": name + "нагрузка, прогноз на этот час за сутки"}),
        ("sensor.emhass_shadow_weather_irradiance", numeric(weather.get("irradiance_wm2")),
         {"unit_of_measurement": "W/m²", "device_class": "irradiance",
          "state_class": "measurement", "friendly_name": name + "радиация по прогнозу погоды"}),
        ("sensor.emhass_shadow_weather_cloud_cover", numeric(weather.get("cloud_cover_pct")),
         {"unit_of_measurement": "%", "state_class": "measurement",
          "friendly_name": name + "облачность по прогнозу погоды"}),
        ("sensor.emhass_shadow_weather_temperature", numeric(weather.get("temperature_c")),
         {"unit_of_measurement": "°C", "device_class": "temperature",
          "state_class": "measurement", "friendly_name": name + "температура по прогнозу погоды"}),
    ]


def publish_hour_sensors(report: dict, hour) -> list[str]:
    """Move the forecast and weather sensors to ``hour`` between replans.

    Replans run every two hours by default, but the sensors describe the
    current hour: without this every second hour would carry the previous
    hour's forecast into Home Assistant's statistics. Uses the latest plan,
    the forecast log and the stored weather snapshot — nothing is recomputed.
    """
    hourly = report.get("hourly") or {}
    key = pd.Timestamp(hour).isoformat()
    starts = hourly.get("starts_utc") or []
    if key not in starts:
        return []
    index = starts.index(key)
    view = {
        "hourly": {k: v[index:] for k, v in hourly.items() if isinstance(v, list)},
        "day_ahead_hour": _day_ahead_hour(hour),
        "weather_hour": weather_hour(archived_weather(), hour),
    }
    published = []
    for entity_id, state, attributes in _forecast_sensors(view):
        try:
            ha_post_state(entity_id, state, attributes)
            published.append(entity_id)
        except Exception:
            continue
    return published


def publish_sensors(report: dict, flexible: list[dict], tz) -> list[str]:
    """Expose the plan as sensors for the owner's automations; never raises."""
    advisory = report["advisory"]
    outage = advisory.get("outage_now")
    sensors = [
        (
            "sensor.emhass_shadow_floor_soc",
            round(advisory["recommended_floor_soc"] * 100),
            {
                "unit_of_measurement": "%",
                "friendly_name": "EMHASS Shadow: рекомендуемый пол SOC",
                "profile": [step["floor_soc"] for step in advisory["floor_profile"]],
                "issue_utc": report["issue_utc"],
            },
        ),
        (
            "sensor.emhass_shadow_autonomy_hours",
            advisory["autonomy_hours"] if advisory["autonomy_hours"] is not None else "unknown",
            {"unit_of_measurement": "h", "friendly_name": "EMHASS Shadow: автономия батареи"},
        ),
        # For the owner's automations during an outage: "on" = the battery
        # will not last the likely rest of it, shed the loads you can.
        (
            "binary_sensor.emhass_shadow_save_battery",
            "on" if outage and not outage["enough"] else "off",
            {
                "friendly_name": "EMHASS Shadow: экономить батарею",
                "cut_kw": outage["cut_kw"] if outage else 0.0,
                "autonomy_h": outage["autonomy_h"] if outage else None,
            },
        ),
        (
            "sensor.emhass_shadow_outage_remaining_hours",
            outage["remaining_h"] if outage else 0.0,
            {
                "unit_of_measurement": "h",
                "friendly_name": "EMHASS Shadow: отключение, осталось с запасом",
                "elapsed_h": outage["elapsed_h"] if outage else 0.0,
                "typical_h": outage["remaining_typical_h"] if outage else 0.0,
            },
        ),
    ]
    sensors += _forecast_sensors(report)
    for item in flexible:
        sensors.append(
            (
                f"sensor.emhass_shadow_{item['slug']}_start",
                item.get("start_local") or "unknown",
                {"friendly_name": f"EMHASS Shadow: {item['name']} — старт", **item},
            )
        )
    published = []
    for entity_id, state, attributes in sensors:
        try:
            ha_post_state(entity_id, state, attributes)
            published.append(entity_id)
        except Exception:
            continue
    return published


def _hardware_note(configured: dict, learned: dict | None, applied: list[str]) -> str | None:
    """One notification line: what the data says about the hardware."""
    if not learned:
        return None
    facts = []
    battery = learned.get("battery") or {}
    if battery.get("status") == "ok":
        facts.append(
            f"ёмкость {battery['capacity_kwh']:.1f} кВт·ч (в настройках {configured['capacity_kwh']}),"
            f" КПД цикла {battery['round_trip_efficiency']:.0%}"
        )
    geometry = learned.get("pv_geometry")
    if geometry and geometry.get("reliable"):
        facts.append(
            f"панели {geometry['kwp']:.1f} кВтп, наклон {geometry['tilt_deg']:.0f}°, "
            f"азимут {geometry['azimuth_deg']:.0f}°"
        )
    elif geometry:
        facts.append(f"ориентация панелей пока не определена ({geometry.get('reason')})")
    limits = learned.get("limits") or {}
    if limits.get("max_charge_kw"):
        facts.append(f"заряд до {limits['max_charge_kw']:.1f} кВт")
    if not facts:
        return None
    line = "• Железо по данным: " + "; ".join(facts)
    if configured.get("autotune"):
        line += " — применено: " + (", ".join(applied) if applied else "нечего менять")
    else:
        line += " (только совет, автонастройка выключена)"
    return line


def _self_learning_note(calibrations: dict, drift: dict) -> str | None:
    """One notification line on the active forecast self-correction."""
    parts = []
    load = calibrations.get("load")
    if load is not None and load.active:
        day_shift = sum(load.hourly)
        parts.append(
            f"нагрузка {day_shift:+.1f} кВт·ч/сутки, коридор ×{load.scale_lo:.2f}/×{load.scale_hi:.2f}"
        )
    pv = calibrations.get("pv")
    if pv is not None and pv.active:
        daylight = [f for f in pv.hourly if abs(f - 1.0) > 1e-9] or [1.0]
        parts.append(f"солнце ×{float(np.mean(daylight)):.2f}")
    if not parts and not (drift or {}).get("drift"):
        return None
    line = "• Самокоррекция прогноза: " + ("; ".join(parts) if parts else "ещё копит данные")
    if (drift or {}).get("drift"):
        line += " — дрейф (" + ", ".join(drift["reasons"]) + "), переобучение раньше срока"
    return line


def _plan_vs_fact_note() -> str | None:
    """One notification line comparing yesterday's promised vs metered import."""
    if not data_store.exists(METRICS):
        return None
    try:
        entries = [
            json.loads(line)
            for line in data_store.read_text(METRICS).splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    for entry in reversed(entries):
        pvf = entry.get("plan_vs_fact")
        if pvf:
            delta = pvf["actual_grid_kwh"] - pvf["planned_grid_kwh"]
            line = (
                f"• План vs факт ({entry['date']}): импорт из сети по плану "
                f"{pvf['planned_grid_kwh']} кВт·ч, по счётчику {pvf['actual_grid_kwh']} кВт·ч "
                f"({delta:+.1f})"
            )
            if "forecast_effect_kwh" in pvf:
                line += (
                    f": из-за прогноза {pvf['forecast_effect_kwh']:+.1f}, "
                    f"из-за исполнения {pvf['execution_effect_kwh']:+.1f}"
                )
            return line
    return None


def _fallback_load_prediction(audit: pd.DataFrame, starts: pd.DatetimeIndex, tz) -> pd.DataFrame:
    """Persistence forecast for fresh installs before the models exist.

    Same-local-hour mean of clean audit history of the SAME weekday — a
    commercial object's weekend dip must not dilute the weekday average.
    A weekday+hour seen fewer than two times falls back to the all-days
    same-hour mean, then to the overall mean; a flat 0.5 kW when there is
    no history at all. Bands are deliberately wide (±40/50 %) so downstream
    planning stays conservative until LightGBM takes over.
    """
    hist = audit["load_kw"].where(learning.load_target_mask(audit)).dropna()
    if len(hist):
        local = hist.index.tz_convert(tz)
        groups = [local.day_of_week, local.hour]
        by_weekday_hour = hist.groupby(groups).mean()
        weekday_seen = hist.groupby(groups).count()
        by_hour = hist.groupby(local.hour).mean()

        def expected(stamp) -> float:
            here = stamp.tz_convert(tz)
            key = (here.day_of_week, here.hour)
            if weekday_seen.get(key, 0) >= 2:
                return float(by_weekday_hour[key])
            return float(by_hour.get(here.hour, np.nan))

        p50 = np.array([expected(s) for s in starts])
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


# Directory of the PV model. v2: inputs aligned to the measured hour
# (pv_weather_frame); a v1 bundle learned the one-hour lag and is not reused.
PV_MODEL = "pv_quantiles_v2"
HALF_HOUR = pd.Timedelta(minutes=30)


def pv_weather_frame(
    geometry: PvGeometry, weather: pd.DataFrame, hours: pd.DatetimeIndex, suffix: str = ""
) -> pd.DataFrame:
    """PV model inputs for the hourly means [H, H+1), indexed at mid-hour.

    Open-Meteo radiation stamped T is the mean over the PRECEDING hour
    [T-1h, T), so the hour starting at H takes the radiation stamped H+1h;
    cloud cover and temperature are instants, so it takes the mean of the
    values at H and H+1h. The sun position of the transposition — and the
    model's clear-sky denominator, through the index — sit at H+30 min.
    Reading radiation stamped H instead (before v2) fed every hour the sun
    of the hour before: a winter morning hour producing 2.8 kW got an almost
    dark forecast. Callers map predictions back by position.
    """
    hours = pd.DatetimeIndex(hours)
    middle = hours + HALF_HOUR
    after = hours + pd.Timedelta(hours=1)

    def at(name: str, stamps: pd.DatetimeIndex) -> pd.Series:
        values = pd.to_numeric(weather[name + suffix], errors="coerce").reindex(stamps)
        return pd.Series(values.to_numpy(dtype=float), index=middle)

    def mean_of_ends(name: str) -> pd.Series:
        return pd.concat([at(name, hours), at(name, after)], axis=1).mean(axis=1)

    poa = plane_power_kw(
        geometry,
        middle,
        at("shortwave_radiation", after),
        at("direct_normal_irradiance", after),
        at("diffuse_radiation", after),
    )
    return pd.DataFrame(
        {
            "pv_measured_kw": np.nan,
            "poa_forecast_kw": poa,
            "cloud_cover_pct": mean_of_ends("cloud_cover"),
            "temp_air_c": mean_of_ends("temperature_2m"),
        },
        index=middle,
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
        clearsky = clear_sky_power_kw(_site_pv_geometry(opts), starts + HALF_HOUR).to_numpy()
    except Exception:
        zeros = [0.0] * len(starts)
        return zeros, zeros, zeros
    if weather is not None and "cloud_cover" in weather:
        cloud_at = pd.to_numeric(weather["cloud_cover"], errors="coerce")
        cloud = np.nanmean(
            np.vstack(
                [
                    cloud_at.reindex(starts).to_numpy(dtype=float),
                    cloud_at.reindex(starts + pd.Timedelta(hours=1)).to_numpy(dtype=float),
                ]
            ),
            axis=0,
        )
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


ENTITY_OPTION_KEYS = [
    "entity_load_power",
    "entity_critical_load_power",
    "entity_pv_power",
    "entity_battery_power",
    "entity_grid_power",
    "entity_outdoor_temperature",
    "entity_battery_soc",
    "entity_export_counter",
    "entity_grid_status",
    "entity_grid_voltage",
    "floor_input_entity",
    "entity_battery_soh",
]

SUPERVISOR_BASE = "http://supervisor"
_OPTIONS_LOCK = threading.Lock()


def supervisor_options() -> dict | None:
    """The options the Supervisor holds right now; None when it cannot say.

    The Supervisor writes /data/options.json only when the add-on starts
    ("changes to options take effect the next time the app starts"), so a
    value saved in the add-on settings or from the panel stayed invisible
    to the running add-on until a restart: the panel kept showing the old
    min SOC after a save. The Supervisor's stored options are the truth.
    """
    try:
        request = urllib.request.Request(
            SUPERVISOR_BASE + "/addons/self/info",
            headers={"Authorization": f"Bearer {_token()}"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.load(response).get("data") or {}
    except Exception:
        return None
    stored = data.get("options")
    return dict(stored) if isinstance(stored, dict) else None


def _write_options(opts: dict) -> None:
    tmp = DATA / "options.json.tmp"
    tmp.write_text(json.dumps(opts, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DATA / "options.json")


def sync_options() -> dict:
    """Bring /data/options.json up to the Supervisor's stored options.

    Called before every run and whenever the panel reads the options, so an
    edit applies at the next run without restarting the add-on. Without the
    Supervisor (tests, a local run) the file stays as it is.
    """
    with _OPTIONS_LOCK:
        current = stored_options()
        stored = supervisor_options()
        if stored is None:
            return current
        # An empty settings object is a reset, not an unavailable Supervisor.
        if stored != current:
            _write_options(stored)
        return stored


def _post_options(opts: dict) -> None:
    request = urllib.request.Request(
        SUPERVISOR_BASE + "/addons/self/options",
        data=json.dumps({"options": opts}).encode(),
        headers={"Authorization": f"Bearer {_token()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        json.load(response)


def update_options(changes: dict, lang: str = "ru") -> dict:
    """Write option edits from the panel through the Supervisor self-options API.

    Entity-binding keys are stored as stripped strings; every other editable
    option is validated and coerced against options_meta (type, range,
    choices), so a bad edit is rejected with a readable ValueError in the
    panel's language instead of failing the Supervisor schema check. Unknown keys are reported in
    "rejected" and dropped. The edits are merged into the options the
    Supervisor holds now — not into the local file, which lags until a
    restart and would silently undo edits made in the add-on settings —
    and the local file is updated at once, so the next run uses them.
    """
    filtered: dict = {}
    cleared: list[str] = []
    rejected: list[str] = []
    for key, value in changes.items():
        if key in discovery.AUTO_KEYS and not str(value if value is not None else "").strip():
            cleared.append(key)
        elif key in ENTITY_OPTION_KEYS:
            filtered[key] = str(value).strip()
        elif key in options_meta.OPTION_META and not options_meta.OPTION_META[key].get("entity"):
            filtered[key] = options_meta.coerce_option(key, value, lang)
        else:
            rejected.append(key)
    if not filtered and not cleared:
        return {"saved": {}, "rejected": rejected}
    # Serialize saves with refreshes: a late sync must not overwrite a new
    # sensor binding or race another writer's temporary options file.
    with _OPTIONS_LOCK:
        stored = supervisor_options()
        opts = {**(stored_options() if stored is None else stored), **filtered}
        for key in cleared:
            opts.pop(key, None)
        _post_options(opts)
        _write_options(opts)
    return {"saved": filtered, "cleared": cleared, "rejected": rejected}


def apply_discovery(candidate: dict, lang: str = "ru") -> dict:
    """Explicitly save a reviewed device, clearing incompatible old bindings."""
    return update_options(discovery.candidate_options(candidate["found"]), lang=lang)


ENTITY_CHECK_LABELS = {
    "load_kw": "Мощность нагрузки",
    "critical_load_kw": "Критическая нагрузка",
    "pv_measured_kw": "Мощность солнца",
    "battery_power_kw_raw_sign": "Мощность батареи",
    "grid_power_kw_raw_sign": "Мощность сети",
    "temp_air_observed_c": "Уличная температура",
}
ENTITY_CHECK_LABELS_EN = {
    "load_kw": "Load power",
    "critical_load_kw": "Critical load",
    "pv_measured_kw": "PV power",
    "battery_power_kw_raw_sign": "Battery power",
    "grid_power_kw_raw_sign": "Grid power",
    "temp_air_observed_c": "Outdoor temperature",
}


def validate_entities() -> list[dict]:
    """Live-check every configured entity against the HA state machine.

    Returns one row per parameter: label, entity id, current state and a
    status — ok / ok_number (manual number) / skipped (empty optional) /
    missing (404) / not_numeric / unexpected. Never raises.
    """
    opts = options()
    ent = resolve_entities(opts)
    # One bounded read: dozens of missing entities must not each wait ten
    # seconds, especially on an unconfigured or multi-inverter installation.
    states = _STATES if _STATES is not None and time.monotonic() - _STATES_AT < STATES_MAX_AGE_S else None
    if states is None:
        try:
            rows = ha_get("/api/states", timeout=10)
            states = {s["entity_id"]: s for s in rows if isinstance(s, dict) and s.get("entity_id")}
        except Exception:
            states = {}
    # (label ru, label en, entity value, kind, option key or None when not directly editable)
    specs: list[tuple[str, str, str, str, str | None]] = [
        (
            ENTITY_CHECK_LABELS[field],
            ENTITY_CHECK_LABELS_EN[field],
            str(entity or ""),
            "number",
            MEAN_SENSOR_FIELDS[field][0],
        )
        for field, (entity, _scale) in ent["mean_sensors"].items()
    ]
    specs += [
        ("SOC батареи", "Battery SOC", str(ent["soc"] or ""), "number", "entity_battery_soc"),
        ("Счётчик экспорта", "Export counter", str(ent["export_counter"] or ""), "number", "entity_export_counter"),
        ("Статус сети", "Grid status", str(ent["grid_binary"] or ""), "binary", "entity_grid_status"),
    ]
    specs += [
        ("Напряжение сети", "Grid voltage", str(opts.get("entity_grid_voltage") or ""), "number", "entity_grid_voltage")
    ]
    # Zone rows are not edited one by one; their source is the template's.
    zone_source: dict[str, str] = {}
    for n in range(len(ent["tou_soc"])):
        specs.append((f"Зона {n + 1}: SOC", f"Zone {n + 1}: SOC", str(ent["tou_soc"][n] or ""), "number", None))
        zone_source[specs[-1][0]] = "tou_soc_entity_template"
        specs.append((f"Зона {n + 1}: время старта", f"Zone {n + 1}: start time", str(ent["tou_time"][n] or ""), "time", None))
        zone_source[specs[-1][0]] = "tou_time_entity_template"
    for n, entity in enumerate(ent.get("tou_grid_charge", []), start=1):
        kind = "select" if entity.startswith("select.") else "binary"
        specs.append((f"Зона {n}: заряд из сети", f"Zone {n}: grid charge", entity, kind, None))
        zone_source[specs[-1][0]] = "tou_grid_charge_entity_template"
    specs.append(("Порог SOC (helper)", "SOC floor (helper)", str(ent["floor_input"] or ""), "number", "floor_input_entity"))
    specs.append(("SoH батареи", "Battery SoH", str(opts.get("entity_battery_soh") or ""), "soh", "entity_battery_soh"))

    try:
        sources = option_sources()
    except Exception:
        sources = {}
    results = []
    for label, label_en, entity, kind, key in specs:
        entity = entity.strip()
        row = {
            "label": label,
            "label_en": label_en,
            "entity": entity,
            "state": None,
            "status": "ok",
            "key": key,
            "hint": options_meta.entity_hint(key) if key else None,
            "hint_en": options_meta.entity_hint(key, "en") if key else None,
            "source": sources.get(key or zone_source.get(label, "")),
            **options_meta.entity_group(key or zone_source.get(label, "")),
        }
        if not entity:
            row["status"] = "missing" if key in discovery.REQUIRED_KEYS else "skipped"
            results.append(row)
            continue
        if kind in ("number", "soh") and _plain_number(entity) is not None:
            # Manual constant, not an entity id (e.g. outdoor temperature "20").
            row.update(state=entity, status="ok_number")
            results.append(row)
            continue
        try:
            state = str(states[entity]["state"])
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
        elif kind == "select" and state in ("unknown", "unavailable"):
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


def recommend_zone_socs(
    programs: list[dict],
    floors: list[float],
    reserve_floors: list[float],
    starts: pd.DatetimeIndex,
    tz: ZoneInfo,
    step_pct: int = 5,
) -> None:
    """Heuristic Batt% per TOU zone, stored as program["recommended_soc_pct"].

    Each zone gets the floating floor of its own start hour (next occurrence
    in the horizon; the current hour for the zone in effect right now),
    raised to the highest outage reserve inside the zone's window. The start-hour floor alone is the daily low for a zone such as
    09:00-17:00; the inverter holds that one value all afternoon, so on a
    cloudy day the battery could drain below the reserve the evening hours
    of the same zone need.
    """
    windows = optimize.zone_windows(programs, starts, tz)
    for program in programs:
        zone = windows[program["n"]]
        if zone is None:
            program["recommended_soc_pct"] = None
            continue
        start = 0 if zone.current else zone.match
        program["recommended_soc_pct"] = max(
            round(floors[start] * 100),
            optimize.reserve_floor_pct(reserve_floors, zone.window, step_pct),
        )


def run(notify: str = "always") -> dict:
    """One planning run; notify "changes" sends the notification only when
    the run wrote to the inverter or hit a write error."""
    snapshot_states()
    configured = options()
    learned_hardware = hardware.load(DATA)
    # autotune: learned capacity/efficiency/limits/geometry replace the
    # configured ones (in this run's copy only); otherwise opts == configured.
    opts, autotune_notes, autotune_notes_en = hardware.effective_options(
        configured, learned_hardware
    )
    ent = resolve_entities(opts)
    entity_checks = validate_entities()
    site = {
        "latitude": opts["latitude"],
        "longitude": opts["longitude"],
        "time_zone": opts["time_zone"],
        "open_meteo_model": weather_models.selected(opts),
    }
    tz = ZoneInfo(site["time_zone"])

    audit, appended = update_audit(
        dict(site, max_charge_kw=opts.get("max_charge_kw")),
        ent,
        backfill_days=float(opts.get("history_backfill_days", 14)),
    )
    new_transitions = update_outage_log(ent)

    issue = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    starts = pd.date_range(issue, periods=HORIZON_HOURS, freq="h")
    starts_plan = pd.date_range(issue, periods=PLAN_HOURS, freq="h")

    models_learning = False
    model = weather_models.selected(site)
    models_dir = weather_models.scoped(DATA / "models", model)
    # Weather comes first: its temperature backstops the load-model features
    # on sites without an outdoor sensor, then feeds the PV forecast itself.
    # Observed sensor values win where present; Open-Meteo only fills gaps.
    try:
        weather, weather_name, weather_stale = archive_weather(site)
    except OSError:
        weather, weather_name, weather_stale = None, None, True
    load_method = "model"
    try:
        load_bundle = LoadQuantileBundle.load(models_dir / "load_quantiles_v1")
        load_prediction = forecast_load(load_bundle, audit, starts_plan, issue, weather)
    except (OSError, ValueError, KeyError):
        models_learning = True
        load_method = "fallback"
        load_prediction = _fallback_load_prediction(audit, starts_plan, tz)

    pv_bundle = None
    try:
        pv_bundle = PvQuantileBundle.load(models_dir / PV_MODEL)
    except (OSError, ValueError):
        models_learning = True
    if weather is None:
        # No internet and no archived vintage at all: assume zero solar, the
        # conservative direction for battery planning, and say so in the report.
        weather_stale = True
        pv_p10 = [0.0] * PLAN_HOURS
        pv_p50 = [0.0] * PLAN_HOURS
        pv_p90 = [0.0] * PLAN_HOURS
    elif pv_bundle is not None:
        pv_prediction = pv_bundle.predict(
            pv_weather_frame(pv_bundle.geometry, weather, starts_plan)
        )
        pv_p10 = [float(pv_prediction["pv_p10_kw"].iloc[i]) for i in range(PLAN_HOURS)]
        pv_p50 = [float(pv_prediction["pv_p50_kw"].iloc[i]) for i in range(PLAN_HOURS)]
        pv_p90 = [float(pv_prediction["pv_p90_kw"].iloc[i]) for i in range(PLAN_HOURS)]
    else:
        pv_p10, pv_p50, pv_p90 = _fallback_pv_forecast(starts_plan, opts, weather)
    pv_method = "model" if weather is not None and pv_bundle is not None else "fallback"
    model_versions = {
        "load": learning.bundle_version(models_dir / "load_quantiles_v1")
        if load_method == "model"
        else None,
        "pv": learning.bundle_version(models_dir / PV_MODEL)
        if pv_method == "model"
        else None,
    }
    for target, version in model_versions.items():
        if version is None:
            # None means "any version" to calibration. Without a live
            # bundle no previous model's errors should teach this forecast.
            model_versions[target] = f"{model}:unavailable"
        elif model != "auto":
            model_versions[target] = f"{model}:{version}"

    # Self-correction between retrains: the RAW forecasts are logged (below,
    # once the reserve is known) and the last two weeks of issued-vs-actual
    # pairs shift P50 per hour and rescale the P10/P90 band. Never let it
    # break the daily plan.
    raw_load = tuple(
        load_prediction[f"load_{q}_kw"].to_numpy(dtype=float).tolist() for q in ("p10", "p50", "p90")
    )
    raw_pv = (list(pv_p10), list(pv_p50), list(pv_p90))
    calibrations: dict = {}
    drift: dict = {}
    try:
        calibrations, drift = learning.self_correct(
            learning.read_forecast_log(DATA / "forecast_log.jsonl"),
            audit,
            tz,
            issue,
            versions=model_versions,
        )
        drift_path = weather_models.scoped(DATA / "drift_state.json", model)
        drift_path.parent.mkdir(parents=True, exist_ok=True)
        drift_path.write_text(json.dumps(drift, ensure_ascii=False), encoding="utf-8")
        if load_method == "model":
            c10, c50, c90 = calibrations["load"].apply(*raw_load, starts_plan, tz)
            load_prediction = pd.DataFrame(
                {"load_p10_kw": c10, "load_p50_kw": c50, "load_p90_kw": c90}, index=starts_plan
            )
        if pv_method == "model":
            c10, c50, c90 = calibrations["pv"].apply(*raw_pv, starts_plan, tz)
            pv_p10, pv_p50, pv_p90 = c10.tolist(), c50.tolist(), c90.tolist()
    except Exception:
        calibrations, drift = {}, {}

    pv_day = pv_day2 = None
    if weather is not None:
        pv_day = {
            q: round(float(np.sum(values[:HORIZON_HOURS])), 2)
            for q, values in (("p10", pv_p10), ("p50", pv_p50), ("p90", pv_p90))
        }
        pv_day2 = {
            q: round(float(np.sum(values[HORIZON_HOURS:])), 2)
            for q, values in (("p10", pv_p10), ("p50", pv_p50), ("p90", pv_p90))
        }

    # State of health: entity id or a plain number ("99"); missing or
    # unreadable values keep the nominal 100 % and are flagged in the report.
    soh, soh_source = _battery_soh(opts)
    if opts.get("_soh_identified"):
        soh, soh_source = 1.0, "identified"

    critical = audit["critical_load_kw"].dropna()
    memory = policy.memory_days(opts)
    settings = ReserveSettings(
        nominal_capacity_kwh=opts["capacity_kwh"],
        soh=soh,
        floor_soc=_soc_bounds(opts)[0],
        max_soc=_soc_bounds(opts)[1],
        discharge_efficiency=opts["discharge_efficiency"],
        cold_start_kw=float(critical.quantile(0.9)) if len(critical) else 0.5,
        # Minimum autonomy: a reserve kept even with no outage series (0 = none).
        cold_start_hours=min(20.0, max(0.0, float(opts.get("reserve_cold_start_hours", 0.0)))),
        window_days=memory,
        quantile=policy.priority(opts)["reserve_quantile"],
        margin=0.0,
        decay_days=1.0e6,  # owner rule: full reserve while the series lives
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
    # Outage series or quiet: only a live series earns an outage reserve.
    regime = policy.outage_regime(events, issue, memory)
    tuning = tuner.load(DATA)
    if tuner.due(DATA):
        try:
            tuning = tuner.tune(
                audit, learning.read_forecast_log(DATA / "forecast_log.jsonl"), opts, tz, issue, soh
            )
            tuner.save(DATA, tuning)
        except Exception:
            pass  # tuning is advisory; the plan runs on the default mode
    # Extra margin on top of the series reserve, in hours of critical load:
    # the smallest the priority allows, or the tuner's pick under autotune.
    margin_hours = tuner.tuned_margin(opts, tuning) if regime["active"] else 0.0
    margin_kwh = margin_hours * plan.critical_load_kw / settings.discharge_efficiency
    # Informational only: energy to carry the whole house to the next hour
    # where pessimistic PV covers it. It is NOT a floor any more: that
    # deficit is economics, and the optimizer plans it with prices.
    steps = [
        ForecastStep(
            start=starts_plan[i].to_pydatetime(),
            hours=1.0,
            pv_p10_kw=float(pv_p10[i]),
            load_p90_kw=float(load_prediction["load_p90_kw"].iloc[i]),
        )
        for i in range(PLAN_HOURS)
    ]

    # Hours the live series hit get the series' outage energy. A finished
    # series leaves no risk hours behind: no protection against phantoms.
    risk = (
        outage_risk_profile(
            policy.recent_events(events, issue, memory),
            issue,
            site["time_zone"],
            min_events=RISK_MIN_EVENTS,
            min_days=RISK_MIN_DAYS,
        )
        if regime["active"]
        else None
    )
    reserve_by_hour = [plan.reserve_kwh + margin_kwh] * PLAN_HOURS
    if opts.get("autotune") and learned_hardware:
        reserve_by_hour = hardware.reshape_reserve(
            reserve_by_hour,
            starts_plan,
            tz,
            learned_hardware.get("critical_profile_kw"),
            plan.duration_hours,
            settings.cold_start_hours,
            settings.discharge_efficiency,
        )
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

    profile = floor_profile(steps, reserve_by_hour, settings)  # advisory figures only
    # The hard floor is the outage reserve alone, rounded UP to the
    # inverter's step: rounding down would shave the reserve.
    caps = adapter.caps(opts)
    reserve_shortfall_kwh = max(settings.floor_for_energy(kwh)[1] for kwh in reserve_by_hour)
    floors_plan = [caps.ceil_soc(settings.floor_for_energy(kwh)[0]) for kwh in reserve_by_hour]
    floors = floors_plan[:HORIZON_HOURS]  # advisory/report day; plan horizon is longer
    recommended_floor = floors[0]  # binding constraint is the current hour
    # Outage reserve alone, as device SOC per hour: the hard lower bound of
    # every TOU zone (see optimize.zone_targets / recommend_zone_socs).
    reserve_floors_plan = [settings.floor_for_energy(kwh)[0] for kwh in reserve_by_hour]
    try:
        learning.append_forecast_log(
            DATA / "forecast_log.jsonl",
            learning.log_entry(
                issue,
                starts_plan,
                raw_load,
                load_method,
                raw_pv,
                pv_method,
                reserve_by_hour,
                versions=model_versions,
                extras={
                    "outage_regime": bool(regime["active"]),
                    "critical_kw": round(float(plan.critical_load_kw), 4),
                    "margin_hours": margin_hours,
                    # Already inside reserve_kwh_by_hour; the tuner takes it out
                    # before it replays its own margin candidates.
                    "margin_kwh": round(float(margin_kwh), 4),
                    # What the plan used (after self-correction), for the
                    # day-ahead sensors.
                    "pv_p50_used": [round(float(v), 3) for v in pv_p50],
                    "pv_p10_used": [round(float(v), 3) for v in pv_p10],
                    "pv_p90_used": [round(float(v), 3) for v in pv_p90],
                    "load_p50_used": [
                        round(float(v), 3) for v in load_prediction["load_p50_kw"].iloc[:PLAN_HOURS]
                    ],
                },
            ),
        )
    except OSError:
        pass

    floor_entity = ent["floor_input"]
    floor_warning: str | None = None
    current_floor = _soc_bounds(opts)[0]
    if floor_entity:
        try:
            current_floor = float(ha_get(f"/api/states/{floor_entity}")["state"]) / 100.0
        except Exception:
            floor_warning = f"сущность порога SOC ({floor_entity}) недоступна"
    programs = []
    for n, (soc_entity, time_entity) in enumerate(
        zip(ent["tou_soc"], ent["tou_time"], strict=True), start=1
    ):
        soc = float(ha_get(f"/api/states/{soc_entity}")["state"])
        start = ha_get(f"/api/states/{time_entity}")["state"]
        programs.append({"n": n, "from": start, "soc_pct": soc})
    recommend_zone_socs(
        programs, floors, reserve_floors_plan[:HORIZON_HOURS], starts, tz, caps.soc_step_pct
    )
    would_raise = [
        p
        for p in programs
        if p.get("recommended_soc_pct") is not None
        and p["soc_pct"] < p["recommended_soc_pct"]
    ]

    battery_now = ha_get(f"/api/states/{ent['soc']}")["state"]

    # BMS calibration: not full for calibration_interval_days -> charge to
    # 100 % in the next night window (the optimizer plans it as a floor at
    # the window's end; the zones covering it get 100 % and grid charge).
    calib = calibration.status(audit, float(battery_now) / 100.0, issue, opts)
    calib_window = calibration.night_window(starts, tz, opts) if calib["due"] else []
    calib_zones = (
        calibration.zones_in_window(programs, starts, tz, calib_window) if calib_window else set()
    )
    full_soc = _soc_bounds(opts)[1]
    if calib_window:
        calib["window_local"] = (
            f"{starts[calib_window[0]].tz_convert(tz):%H:%M}–"
            f"{(starts[calib_window[-1]] + pd.Timedelta(hours=1)).tz_convert(tz):%H:%M}"
        )
        calib["zones"] = sorted(calib_zones)
        floors_plan = list(floors_plan)
        floors_plan[calib_window[-1]] = max(floors_plan[calib_window[-1]], full_soc)
        for program in programs:
            if program["n"] in calib_zones:
                program["recommended_soc_pct"] = round(full_soc * 100)

    # MILP layer: optimize the 48 h horizon against the tariff and the
    # floating floor; only the first day is applied or reported. Battery
    # energy capacity is scaled by the measured SoH so kWh accounting inside
    # the solver matches the real pack. Advisory only — never let it break
    # the daily plan.
    optimizer_plan = None
    if opts.get("optimizer_enabled", True):
        try:
            load_p50 = [
                float(load_prediction["load_p50_kw"].iloc[i]) for i in range(PLAN_HOURS)
            ]
            opt_opts = dict(opts, capacity_kwh=float(opts["capacity_kwh"]) * soh)
            # The night charge is sized for the owner's sun scenario:
            # optimistic, median or pessimistic (priority_mode); the load
            # scenario is always the median.
            scenario = policy.priority(opts)["forecast"]
            pv_by_q = {"p10": pv_p10, "p50": pv_p50, "p90": pv_p90}
            optimizer_plan = optimize.run_plan(
                starts_plan, pv_p50, load_p50, floors_plan,
                float(battery_now) / 100.0, programs, opt_opts,
                reserve_floors=reserve_floors_plan,
                pv_scenario_kw=list(pv_by_q[scenario["pv"]]),
                load_scenario_kw=[
                    float(load_prediction[f"load_{scenario['load']}_kw"].iloc[i])
                    for i in range(PLAN_HOURS)
                ],
            )
            if optimizer_plan is not None:
                optimizer_plan["night_scenario"] = dict(scenario)
        except Exception:
            optimizer_plan = None
    if optimizer_plan is not None and calib_zones:
        for target in optimizer_plan["zone_targets"]:
            if target["n"] in calib_zones:
                target["optimal_soc_pct"] = round(full_soc * 100)

    # Flexible loads: the best start inside each load's window, on the sun
    # the plan expects to curtail first, then on plain surplus.
    load_p50_day = load_prediction["load_p50_kw"].to_numpy()[:HORIZON_HOURS]
    surplus = np.maximum(np.asarray(pv_p50[:HORIZON_HOURS]) - load_p50_day, 0.0)
    curtail = (
        (optimizer_plan or {}).get("p_curtail_kw") or [0.0] * PLAN_HOURS
    )[:HORIZON_HOURS]
    flexible = deferrable.recommend(
        list(opts.get("deferrable_loads") or []), starts, tz, curtail, surplus
    )
    autonomy_hours = autonomy(
        float(battery_now) / 100.0,
        _soc_bounds(opts)[0],
        float(opts["capacity_kwh"]) * soh,
        settings.discharge_efficiency,
        plan.critical_load_kw,
    )
    outage_now = policy.outage_in_progress(
        events,
        datetime.now(UTC),
        memory,
        policy.priority(opts)["reserve_quantile"],
        soc=float(battery_now) / 100.0,
        min_soc=_soc_bounds(opts)[0],
        capacity_kwh=float(opts["capacity_kwh"]) * soh,
        discharge_efficiency=settings.discharge_efficiency,
        critical_kw=plan.critical_load_kw,
        pv_kw=list(pv_p10),
    )

    # Apply mode (off by default): write the plan into the inverter through
    # Home Assistant services. Optimizer zone targets win when available,
    # otherwise the heuristic per-zone floors. Exception: when the optimizer
    # relaxed the requested floor profile (floor_relaxed), its zone targets
    # were computed against the flat min-SOC band — writing them would
    # silently drop the outage reserve (e.g. requested 80 %, written 20 %),
    # so the heuristic floors are applied instead. Writes happen only when
    # the change is at least one inverter SOC step — no churn on rounding
    # noise. Every write is read back: HTTP 200 only means HA accepted the
    # service call, not that the inverter applied the value.
    grid_charge = []
    if caps.controls_grid_charge:
        grid_charge = adapter.grid_charge_plan(
            programs,
            starts,
            tz,
            opts,
            reserve_floors_plan[:HORIZON_HOURS],
            pv_p10[:HORIZON_HOURS],
            (optimizer_plan or {}).get("soc_opt"),
            float(battery_now) / 100.0,
        )
        for item in grid_charge:
            if item["n"] in calib_zones:
                item["desired"], item["reason"] = True, "калибровка BMS: заряд до 100%"
                item["reason_en"] = "BMS calibration: charge to 100%"
            entity = ent["tou_grid_charge"][item["n"] - 1]
            item["entity"] = entity
            try:
                state = ha_get(f"/api/states/{entity}")
                item["current"] = str(state["state"])
                item["options"] = list((state.get("attributes") or {}).get("options") or [])
            except Exception:
                item["current"] = None
            item["target_state"] = adapter.flag_state(item, opts)
    applied: list[str] = []
    apply_errors: list[str] = []
    # Writes waiting for read-back: (label, entity, expected, ok message);
    # expected is a float for SOC values and "on"/"off" for switches.
    pending: list[tuple[str, str, float | str, str]] = []
    if opts.get("apply_mode", False):
        for item in grid_charge:
            wanted = item["target_state"]
            if wanted is None:
                apply_errors.append(
                    f"заряд из сети, зона {item['n']}: нет подходящего варианта у {item['entity']}"
                )
                continue
            if item["current"] == wanted:
                continue
            try:
                ha_service(*adapter.flag_service(item["entity"], wanted))
            except Exception:
                apply_errors.append(f"заряд из сети, зона {item['n']}")
                continue
            pending.append((
                f"заряд из сети, зона {item['n']}",
                item["entity"],
                wanted,
                f"заряд из сети зона {item['n']}: {wanted} ({item['reason']})",
            ))
        zone_pct = {}
        if optimizer_plan is not None and not optimizer_plan.get("floor_relaxed"):
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
            if abs(target - p["soc_pct"]) < caps.soc_step_pct:
                continue
            try:
                ha_service(
                    "number",
                    "set_value",
                    {"entity_id": ent["tou_soc"][p["n"] - 1], "value": target},
                )
            except Exception:
                apply_errors.append(f"зона {p['n']}")
                continue
            pending.append((
                f"зона {p['n']}",
                ent["tou_soc"][p["n"] - 1],
                float(target),
                f"зона {p['n']} ({p['from'][:5]}) {p['soc_pct']:.0f}→{target}%",
            ))
        floor_target = min(max(int(round(recommended_floor * 100)), min_pct), max_pct)
        if floor_entity and abs(floor_target - round(current_floor * 100)) >= caps.soc_step_pct:
            try:
                # Domain follows the entity id: number.* and input_number.*
                # share the set_value service name but live in own domains.
                ha_service(
                    ent["floor_input"].split(".", 1)[0],
                    "set_value",
                    {"entity_id": ent["floor_input"], "value": floor_target},
                )
                pending.append((
                    "порог SOC",
                    ent["floor_input"],
                    float(floor_target),
                    f"порог SOC {round(current_floor * 100)}→{floor_target}%",
                ))
            except Exception:
                apply_errors.append("порог SOC")
        # Read back after the inverter had a moment to propagate: HTTP 200
        # only means HA accepted the call. A write happens only when the value
        # moves by at least one step, so a device that ignored it is still a
        # full step away — the tolerance must stay below half a step, or an
        # unchanged zone would pass as applied.
        if pending:
            time.sleep(3)
            for label, entity, expected, ok_msg in pending:
                if readback_ok(entity, expected, caps.soc_step_pct):
                    applied.append(ok_msg)
                else:
                    apply_errors.append(f"{label} (запись не подтверждена)")

    hold_until = (
        datetime.fromisoformat(regime["series_until_utc"]).astimezone(tz).date().isoformat()
        if regime["active"] and regime["series_until_utc"]
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
    metrics_scored = update_metrics(audit, tz, dict(opts, capacity_kwh=float(opts["capacity_kwh"]) * soh))
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
        *([outage_now_line(outage_now)] if outage_now else []),
        f"• Нагрузка 24 ч, кВт·ч: P50 {load_day['p50']} (P10 {load_day['p10']} … P90 {load_day['p90']})",
        pv_line,
        _reserve_line(regime, plan, margin_hours, hold_until, settings.cold_start_hours)
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
        scenario = optimizer_plan.get("night_scenario")
        if optimizer_plan.get("tariff_enabled") and scenario:
            mode = str(opts.get("priority_mode", policy.DEFAULT_PRIORITY))
            day = {"economy": "оптимистичный", "balanced": "средний"}.get(mode, "пессимистичный")
            lines.append(
                f"• Ночной заряд рассчитан на {day} день: солнце {scenario['pv'].upper()}, "
                f"нагрузка {scenario['load'].upper()} (приоритет {mode})"
            )
        if optimizer_plan["grid_charge_kwh_cheap"] >= 0.3:
            lines.append(
                f"• Ночная зарядка из сети по дешёвому тарифу: "
                f"{optimizer_plan['grid_charge_kwh_cheap']:.1f} кВт·ч"
                + (
                    f" (из них {optimizer_plan['night_margin_kwh']:.1f} — запас "
                    "сверх среднего прогноза под сценарий приоритета)"
                    if optimizer_plan.get("night_margin_kwh", 0.0) >= 0.3
                    else ""
                )
            )
        if optimizer_plan["floor_relaxed"]:
            relaxed_note = (
                "• Оптимизатор: плавающий пол недостижим по прогнозу — "
                "план посчитан с плоским минимумом SOC"
            )
            if apply_mode:
                relaxed_note += "; цели оптимизатора НЕ применяются, записаны эвристические полы"
            lines.append(relaxed_note)
        zone_changes = [
            t
            for t in optimizer_plan["zone_targets"]
            if t.get("optimal_soc_pct") is not None
            and abs(t["optimal_soc_pct"] - t["soc_pct"]) >= caps.soc_step_pct
        ]
        # When the relaxed targets are not applied (reserve protection), do
        # not present them as the plan's zone targets.
        if zone_changes and not (apply_mode and optimizer_plan["floor_relaxed"]):
            zones = ", ".join(
                f"зона {t['n']} ({t['from'][:5]}) {t['soc_pct']:.0f}→{t['optimal_soc_pct']:.0f}%"
                for t in zone_changes
            )
            lines.append(f"• Целевые SOC зон от оптимизатора: {zones}")
    elif opts.get("optimizer_enabled", True):
        lines.append("• Оптимизатор: расчёт не удался, действует эвристика пола SOC")
    if reserve_shortfall_kwh > 0.05:
        lines.append(
            "• Внимание: резерв на отключение не помещается в батарею, нехватка "
            f"{reserve_shortfall_kwh:.1f} кВт·ч"
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
    tuning_note = _tuning_note(tuning, margin_hours, bool(opts.get("autotune")), regime["active"])
    if tuning_note is not None:
        lines.append(tuning_note)
    hardware_note = _hardware_note(configured, learned_hardware, autotune_notes)
    if hardware_note is not None:
        lines.append(hardware_note)
    if calib.get("window_local"):
        since = calib["days_since_full"]
        lines.append(
            "• Калибровка BMS: "
            + (f"до 100% батарея не заряжалась {since:.0f} дн." if since is not None else "полного заряда в истории нет")
            + f" — заряд до 100% в {calib['window_local']}"
        )
    if autonomy_hours is not None:
        lines.append(
            f"• Автономия сейчас: ~{autonomy_hours:.1f} ч критической нагрузки "
            f"({plan.critical_load_kw:.2f} кВт)"
        )
    for item in flexible:
        lines.append(_deferrable_line(item))
    learning_note = _self_learning_note(calibrations, drift)
    if learning_note is not None:
        lines.append(learning_note)
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
            "weather_model": model,
            "weather_stale": weather_stale,
            "models_learning": models_learning,
            "battery_soh": round(soh, 4),
            "battery_soh_source": soh_source,
            "entity_checks": entity_checks,
            "plan_horizon_hours": PLAN_HOURS,
            "whole_house_backup": learning.whole_house_on_backup(audit),
            "calibration": calib,
            "priority_mode": str(opts.get("priority_mode", policy.DEFAULT_PRIORITY)),
            "reserve_margin_hours": margin_hours,
            "tuning": tuning,
            "reserve_pv_aware": bool(opts.get("reserve_pv_aware", True)),
            "hardware": {
                "learned": learned_hardware,
                "autotune": bool(opts.get("autotune", False)),
                "applied": autotune_notes,
                "applied_en": autotune_notes_en,
            },
            "self_learning": {
                "load": calibrations["load"].to_dict() if "load" in calibrations else None,
                "pv": calibrations["pv"].to_dict() if "pv" in calibrations else None,
                "drift": drift or None,
            },
        },
        "weather_hour": weather_hour(weather, issue),
        "day_ahead_hour": _day_ahead_hour(issue),
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
            "regime": regime,
            "margin_hours": margin_hours,
            "shortfall_reserve_kwh": round(reserve_shortfall_kwh, 3),
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
            "max_floor_soc": caps.ceil_soc(profile.max_floor_soc),
            "energy_to_crossover_kwh": round(profile.steps[0].stored_kwh, 3),
            "crossover_utc": (
                profile.steps[0].crossover.isoformat() if profile.steps[0].crossover else None
            ),
            "current_floor_soc": current_floor,
            "delta_pp": delta_pp,
            "programs": programs,
            "would_raise_zones": would_raise,
            "deferrable": flexible,
            "grid_charge": grid_charge,
            "autonomy_hours": None if autonomy_hours is None else round(autonomy_hours, 2),
            "outage_now": outage_now,
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
    data_store.write_json(out_path, report)

    try:
        mirror = SHARE / "shadow_reports"
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / out_path.name).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass

    report["data"]["metrics_scored_days"] = metrics_scored

    published: list[str] = []
    if opts.get("publish_sensors", False):
        published = publish_sensors(report, flexible, tz)
    report["data"]["published_sensors"] = published

    if opts.get("notifications", True) and (notify == "always" or applied or apply_errors):
        ha_service(
            "persistent_notification",
            "create",
            {
                "title": f"Теневой план батареи — {issue.astimezone(tz):%d.%m}",
                "message": report["notification"],
                "notification_id": "emhass_shadow_plan",
            },
        )

    # Error notifications: stable ids, self-clearing on recovery. Unlike the
    # plan summary these fire on every run, including quiet hourly replans.
    if weather_stale:
        if weather_name is not None:
            notify_problem(
                "weather",
                "EMHASS Shadow: прогноз погоды устарел",
                "Нет связи с Open-Meteo — солнечный прогноз посчитан по архивному "
                f"снапшоту {weather_name} и может расходиться с реальной погодой. "
                "Проверь интернет и маршрут на хосте. Уведомление исчезнет само, "
                "когда свежий прогноз скачается.",
            )
        else:
            notify_problem(
                "weather",
                "EMHASS Shadow: нет прогноза погоды",
                "Нет связи с Open-Meteo и нет ни одного архивного снапшота — "
                "солнечный прогноз принят нулевым, план консервативный. "
                "Проверь интернет и маршрут на хосте.",
            )
    else:
        notify_problem("weather", "", None)
    if entity_note is not None:
        notify_problem(
            "entities",
            "EMHASS Shadow: проблемы с сущностями",
            entity_note.lstrip("• ")
            + " Панель → «Проверка и настройка сущностей» подскажет, что именно не так.",
        )
    else:
        notify_problem("entities", "", None)
    drop_states()
    return report
