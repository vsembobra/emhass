"""Options the owner leaves empty, filled from what Home Assistant already knows.

ha-solarman (github.com/davidrapan/ha-solarman) builds the Deye entities
from fixed profiles, so a role is found by its profile item, not by entity
id: every entity's registry unique_id is "<config entry id>_<item key>",
the key slugified from the item's name and platform ("Battery Power"
sensor -> "battery_power_sensor"). It survives a renamed device or entity.
The same integration exposes the inverter's own settings, which give the
battery capacity (Ah x nominal voltage) and the limits: grid charging and
battery discharging current x voltage, capped by the rated power (the
grid-charge current is what bounds the planned night charge). The Home Assistant
core config gives the site location and time zone.

Per option: a value the owner set > found here > options_meta.DEFAULTS.
"none" switches an optional entity off. What was found is cached in
/data/discovery.json (pipeline.refresh_discovery); reading options never
calls Home Assistant.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import options_meta

DISCOVERY_FILE = "discovery.json"
VERSION = 1
REFRESH_EVERY = timedelta(hours=6)
NONE = "none"
SOLARMAN = "ha-solarman"
CORE = "Home Assistant"

LV_NOMINAL_V = 51.2  # 16-cell LiFePO4: the usual low-voltage Deye pack
LEAD_NOMINAL_V = 48.0
HV_MIN_V = 60.0  # a high-voltage pack: its measured voltage is the nominal one
MAX_ZONES = 12

# option -> profile item keys, first found wins
ENTITY_ITEMS = {
    "entity_load_power": ("load_power_sensor",),
    # Single-phase Deye hybrids have no "Load UPS Power": their load port is the backup.
    "entity_critical_load_power": ("load_ups_power_sensor", "load_power_sensor"),
    "entity_pv_power": ("pv_power_sensor",),
    "entity_battery_power": ("battery_power_sensor",),
    "entity_grid_power": ("grid_power_sensor",),
    "entity_battery_soc": ("battery_sensor",),
    "entity_grid_status": ("grid_binary_sensor",),
    "entity_grid_voltage": ("grid_l1_voltage_sensor", "grid_voltage_sensor"),
    "entity_export_counter": ("total_energy_export_sensor",),
    "entity_battery_soh": ("battery_soh_sensor",),
}
ZONE_ITEMS = {
    "tou_soc_entity_template": "program_{n}_soc_number",
    "tou_time_entity_template": "program_{n}_time_time",
    "tou_grid_charge_entity_template": "program_{n}_charging_select",
}
SETTING_ITEMS = {
    "capacity_ah": "battery_capacity_number",
    "battery_v": "battery_voltage_sensor",
    "charge_a": "battery_grid_charging_current_number",
    "discharge_a": "battery_max_discharging_current_number",
    "control_mode": "battery_control_mode_select",
    "device": "device_sensor",
}
LOCATION_KEYS = ("latitude", "longitude", "time_zone")
AUTO_KEYS = (
    *ENTITY_ITEMS,
    *ZONE_ITEMS,
    "tou_zones",
    "capacity_kwh",
    "max_charge_kw",
    "max_discharge_kw",
    *LOCATION_KEYS,
)


def _slug(text) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _item_key(entry: dict) -> str | None:
    """Profile item key of one registry entry ("battery_power_sensor")."""
    unique = str(entry.get("unique_id") or "").lower()
    prefix = _slug(entry.get("config_entry_id") or "")
    if prefix and unique.startswith(prefix + "_"):
        return unique[len(prefix) + 1 :]
    name = entry.get("original_name")
    if name:  # older unique_id formats: the item name and the platform
        return _slug(f"{name}_{str(entry.get('entity_id', '')).split('.', 1)[0]}")
    return None


def _template(entity_ids: list[str | None]) -> str | None:
    """One {n} template that yields every zone's entity, or None."""
    if not entity_ids or None in entity_ids:
        return None
    first = entity_ids[0]
    for at in (m.start() for m in re.finditer("1", first)):
        candidate = first[:at] + "{n}" + first[at + 1 :]
        if all(candidate.format(n=n) == e for n, e in enumerate(entity_ids, start=1)):
            return candidate
    return None


def _number(states: dict, entity: str | None, attribute: str | None = None) -> float | None:
    state = states.get(entity) if entity else None
    if not state:
        return None
    raw = (state.get("attributes") or {}).get(attribute) if attribute else state.get("state")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _battery_settings(items: dict, states: dict) -> dict:
    def number(name, attribute=None):
        return _number(states, items.get(SETTING_ITEMS[name]), attribute)

    measured = number("battery_v")
    mode = str((states.get(items.get(SETTING_ITEMS["control_mode"], "")) or {}).get("state", ""))
    if measured is not None and measured > HV_MIN_V:
        volts = measured
    elif mode.lower().startswith("lead"):
        volts = LEAD_NOMINAL_V
    else:
        volts = LV_NOMINAL_V
    found = {}
    amp_hours = number("capacity_ah")
    if amp_hours and 1.0 <= amp_hours * volts / 1000.0 <= 200.0:
        found["capacity_kwh"] = round(amp_hours * volts / 1000.0, 1)
    rated = number("device", "Rated Power")  # W, an attribute of the Device sensor
    rated_kw = rated / 1000.0 if rated else math.inf
    for option, name in (("max_charge_kw", "charge_a"), ("max_discharge_kw", "discharge_a")):
        amps = number(name)
        if amps:
            kw = min(amps * volts / 1000.0, rated_kw)
            if 0.5 <= kw <= 50.0:
                found[option] = round(kw, 1)
    return found


def _location(core: dict | None) -> dict:
    if not core:
        return {}
    found = {}
    for key in ("latitude", "longitude"):
        try:
            value = float(core.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value != 0.0:
            found[key] = value
    if str(core.get("time_zone") or "").strip():
        found["time_zone"] = str(core["time_zone"]).strip()
    return found


def discover(
    registry: list[dict], states: dict[str, dict], core: dict | None, now: datetime | None = None
) -> dict:
    """What the options can take from HA: {"found": {option: value}, "source": {option: label}}."""
    by_entry: dict[str, dict[str, str]] = {}
    for entry in registry:
        if entry.get("platform") != "solarman" or entry.get("disabled_by"):
            continue
        key = _item_key(entry)
        if key and entry.get("entity_id"):
            by_entry.setdefault(str(entry.get("config_entry_id")), {})[key] = entry["entity_id"]
    # Several ha-solarman devices (an inverter and a meter, say): the one with the battery.
    items = max(
        by_entry.values(), key=lambda keys: ("battery_sensor" in keys, len(keys)), default={}
    )
    found: dict = {}
    for option, keys in ENTITY_ITEMS.items():
        entity = next((items[k] for k in keys if k in items), None)
        if entity:
            found[option] = entity
    zones = 0
    while zones < MAX_ZONES and all(
        pattern.format(n=zones + 1) in items
        for pattern in (
            ZONE_ITEMS["tou_soc_entity_template"],
            ZONE_ITEMS["tou_time_entity_template"],
        )
    ):
        zones += 1
    if zones:
        found["tou_zones"] = zones
        for option, pattern in ZONE_ITEMS.items():
            template = _template([items.get(pattern.format(n=n)) for n in range(1, zones + 1)])
            if template:
                found[option] = template
    found.update(_battery_settings(items, states))
    location = _location(core)
    source = dict.fromkeys(found, SOLARMAN)
    source.update(dict.fromkeys(location, CORE))
    found.update(location)
    return {
        "version": VERSION,
        "checked_utc": (now or datetime.now(UTC)).isoformat(),
        "solarman_entities": len(items),
        "found": found,
        "source": source,
    }


def save(data_dir: Path, result: dict) -> None:
    path = data_dir / DISCOVERY_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load(data_dir: Path) -> dict | None:
    try:
        result = json.loads((data_dir / DISCOVERY_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return result if isinstance(result, dict) else None


def stale(result: dict | None, now: datetime | None = None) -> bool:
    if not result or result.get("version") != VERSION or not result.get("checked_utc"):
        return True
    checked = datetime.fromisoformat(result["checked_utc"])
    return (now or datetime.now(UTC)) - checked >= REFRESH_EVERY


def _unset(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def effective(stored: dict, discovered: dict | None) -> dict:
    """Options a run uses: DEFAULTS < found in HA < the owner's values."""
    opts = dict(options_meta.DEFAULTS)
    opts.update((discovered or {}).get("found") or {})
    for key, value in stored.items():
        if key in AUTO_KEYS and _unset(value):
            continue  # left empty: found or default
        opts[key] = value
    for key in AUTO_KEYS:
        if isinstance(opts.get(key), str) and opts[key].strip().lower() == NONE:
            opts[key] = ""
    return opts


def sources(stored: dict, discovered: dict | None) -> dict[str, str]:
    """Where each option's value comes from: "manual", a discovery source or "default"."""
    found = (discovered or {}).get("found") or {}
    labels = (discovered or {}).get("source") or {}
    out = {}
    for key in options_meta.DEFAULTS:
        if key in stored and not (key in AUTO_KEYS and _unset(stored[key])):
            out[key] = "manual"
        elif key in found:
            out[key] = labels.get(key, SOLARMAN)
        else:
            out[key] = "default"
    return out


def load_options(data_dir: Path) -> dict:
    """Effective options from data_dir/options.json and the discovery cache."""
    stored = json.loads((data_dir / "options.json").read_text())
    return effective(stored, load(data_dir))


def compact(stored: dict, discovered: dict) -> tuple[dict, list[str]]:
    """Stored options without values that only repeat what would apply anyway.

    Settings saved by an older version hold every option, so none of them
    could be found automatically. A hidden option equal to its default or to
    what was found is dropped (same behaviour, now automatic); a value the
    owner chose stays. An empty grid-charge template meant "leave the flags
    alone" and becomes "none", so a found template never starts steering
    the inverter on its own.
    """
    found = discovered.get("found") or {}
    kept, dropped = {}, []
    for key, value in stored.items():
        if key in options_meta.VISIBLE or key not in options_meta.DEFAULTS:
            kept[key] = value
        elif key == "tou_grid_charge_entity_template" and _unset(value) and found.get(key):
            kept[key] = NONE
        elif value == options_meta.DEFAULTS[key] or (key in found and value == found[key]):
            dropped.append(key)
        else:
            kept[key] = value
    return kept, dropped
