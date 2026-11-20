"""Learn the site's hardware from its own history instead of trusting settings.

Weekly, from the hourly audit:

* battery usable capacity and round-trip efficiency
  (:class:`emhass.battery_identification.BatteryIdentification`);
* sustained charge/discharge power actually observed, overall and by SOC
  band (the BMS taper near full shows up as a lower limit in the top bands);
* PV orientation and effective kWp (:func:`emhass.pv_quantiles.estimate_geometry`
  on the clearest uncensored days), marked reliable only with enough days
  and a close fit — a grid search always returns *some* orientation;
* the critical (backup) load profile by local hour.

Everything is advisory unless the ``autotune`` option is on. Even then the
learned values only move settings in directions the evidence supports:
capacity and efficiencies replace the configured ones only when the
identification passed every guardrail; power limits are only RAISED (an
hourly mean above the configured limit proves the hardware can do more, a
lower one only shows what was needed).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from emhass.battery_identification import BatteryIdentification
from emhass.pv_quantiles import estimate_geometry

HARDWARE_FILE = "hardware.json"
BATTERY_WINDOW_DAYS = 60
PROFILE_WINDOW_DAYS = 30
MIN_POWER_SAMPLES = 20
MIN_PROFILE_SAMPLES = 5
MIN_GEOMETRY_DAYS = 5  # below this nothing is estimated at all
RELIABLE_GEOMETRY_DAYS = 20  # the 10 clearest of fewer days are often cloudy
MAX_GEOMETRY_REL_ERROR = 0.15  # fit MAE relative to the measured peak
MAX_AZIMUTH_CHANGE_DEG = 45.0  # beyond this vs the configured roof: confirm by hand
MAX_TILT_CHANGE_DEG = 20.0
POWER_EPS_KW = 0.05
SOC_BANDS = ((0.0, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01))
RAISE_LIMIT_MARGIN = 1.1  # observed must beat the setting by 10 % to raise it
PROFILE_FACTOR_BOUNDS = (0.5, 1.5)

_LOGGER = logging.getLogger("emhass_shadow.hardware")
_LOGGER.addHandler(logging.NullHandler())


def _column(audit: pd.DataFrame, name: str) -> pd.Series:
    if name in audit:
        return pd.to_numeric(audit[name], errors="coerce")
    return pd.Series(np.nan, index=audit.index)


def _recent(audit: pd.DataFrame, days: int) -> pd.DataFrame:
    if audit.empty:
        return audit
    return audit[audit.index >= audit.index[-1] - pd.Timedelta(days=days)]


def identify_battery(audit: pd.DataFrame, capacity_kwh: float) -> dict:
    frame = _recent(audit, BATTERY_WINDOW_DAYS)
    df = pd.DataFrame(
        {
            "power_w": _column(frame, "battery_power_kw_raw_sign") * 1000.0,
            "soc_pct": _column(frame, "soc_mean") * 100.0,
        },
        index=frame.index,
    )
    try:
        result = BatteryIdentification(_LOGGER, time_step=pd.Timedelta(hours=1)).identify(
            df, "power_w", "soc_pct", configured_capacity_wh=float(capacity_kwh) * 1000.0
        )
    except Exception as error:  # an estimator failure must not break the run
        return {"status": "error", "messages": [str(error)[:200]]}
    data = result.to_dict()
    return {
        "status": result.status,
        "messages": result.messages[:3],
        "capacity_kwh": data["capacity_kwh"]["value"],
        "round_trip_efficiency": data["round_trip_efficiency"]["value"],
        "eta_one_way": result.eta_symmetric,
        "n_segments": [result.n_charge_segments, result.n_discharge_segments],
    }


def observed_limits(audit: pd.DataFrame) -> dict:
    frame = _recent(audit, BATTERY_WINDOW_DAYS)
    battery = _column(frame, "battery_power_kw_raw_sign")
    soc = _column(frame, "soc_mean")
    charging = (-battery).where(battery < -POWER_EPS_KW).dropna()
    discharging = battery.where(battery > POWER_EPS_KW).dropna()
    out: dict = {"charge_samples": int(len(charging)), "discharge_samples": int(len(discharging))}
    out["max_charge_kw"] = (
        round(float(charging.quantile(0.98)), 3) if len(charging) >= MIN_POWER_SAMPLES else None
    )
    out["max_discharge_kw"] = (
        round(float(discharging.quantile(0.98)), 3)
        if len(discharging) >= MIN_POWER_SAMPLES
        else None
    )
    bands = []
    for low, high in SOC_BANDS:
        in_band = charging[
            (soc.reindex(charging.index) >= low) & (soc.reindex(charging.index) < high)
        ]
        bands.append(
            {
                "soc_from": low,
                "soc_to": min(high, 1.0),
                "charge_p95_kw": round(float(in_band.quantile(0.95)), 3)
                if len(in_band) >= MIN_PROFILE_SAMPLES
                else None,
                "samples": int(len(in_band)),
            }
        )
    out["charge_by_soc"] = bands
    return out


def identify_geometry(
    audit: pd.DataFrame,
    latitude: float,
    longitude: float,
    configured_tilt: float | None = None,
    configured_azimuth: float | None = None,
) -> dict | None:
    frame = _recent(audit, BATTERY_WINDOW_DAYS)
    keep = (
        frame["pv_filter"] == "keep"
        if "pv_filter" in frame
        else pd.Series(False, index=frame.index)
    )
    measured = _column(frame, "pv_measured_kw").where(keep)
    measured = measured[measured > 0].dropna()
    if measured.empty or len(set(measured.index.date)) < MIN_GEOMETRY_DAYS:
        return None
    # Hourly means describe the middle of the hour; the clear-sky model is
    # evaluated at the given instants, so shift to avoid an azimuth bias.
    measured.index = measured.index + pd.Timedelta(minutes=30)
    try:
        estimate = estimate_geometry(measured, float(latitude), float(longitude))
    except Exception:
        return None
    geometry = estimate["geometry"]
    best = min(estimate["scores"], key=lambda s: s["mae_kw"])
    days = len(set(measured.index.date))
    peak = float(measured.quantile(0.98))
    relative = float(best["mae_kw"]) / peak if peak > 0 else float("inf")
    # A grid search always returns *some* orientation; only a close fit on
    # enough days is evidence. Autotune applies reliable estimates only.
    reasons = []
    if days < RELIABLE_GEOMETRY_DAYS:
        reasons.append(f"мало дней с солнцем ({days} из {RELIABLE_GEOMETRY_DAYS})")
    if relative > MAX_GEOMETRY_REL_ERROR:
        reasons.append(f"ошибка подгонки {relative:.0%} пика")
    # A clock offset or morning shade fits as a rotated roof with a small
    # error, so a big change against the configured roof is never taken
    # automatically: the owner confirms it by editing the options.
    if configured_azimuth is not None:
        turn = abs(
            (float(geometry.azimuth_deg) - float(configured_azimuth) + 180.0) % 360.0 - 180.0
        )
        if turn > MAX_AZIMUTH_CHANGE_DEG:
            reasons.append(
                f"азимут {geometry.azimuth_deg:.0f}° против {configured_azimuth:.0f}° в настройках — проверьте вручную"
            )
    if (
        configured_tilt is not None
        and abs(float(geometry.tilt_deg) - float(configured_tilt)) > MAX_TILT_CHANGE_DEG
    ):
        reasons.append(
            f"наклон {geometry.tilt_deg:.0f}° против {configured_tilt:.0f}° в настройках — проверьте вручную"
        )
    return {
        "tilt_deg": float(geometry.tilt_deg),
        "azimuth_deg": float(geometry.azimuth_deg),
        "kwp": round(float(geometry.kwp), 3),
        "mae_kw": round(float(best["mae_kw"]), 4),
        "relative_error": round(relative, 3),
        "days": days,
        "reliable": not reasons,
        "reason": "; ".join(reasons) or None,
    }


def critical_profile(audit: pd.DataFrame, tz: ZoneInfo) -> list[float] | None:
    """P90 of the critical load by local hour; global P90 for sparse hours."""
    frame = _recent(audit, PROFILE_WINDOW_DAYS)
    critical = _column(frame, "critical_load_kw").dropna()
    if len(critical) < 24:
        return None
    overall = float(critical.quantile(0.9))
    hours = critical.index.tz_convert(tz).hour
    profile = []
    for hour in range(24):
        values = critical[hours == hour]
        profile.append(
            round(float(values.quantile(0.9)) if len(values) >= MIN_PROFILE_SAMPLES else overall, 4)
        )
    return profile


def identify(audit: pd.DataFrame, opts: dict, now: datetime | None = None) -> dict:
    """Run every estimator; each part fails soft to None/'error'."""
    tz = ZoneInfo(opts.get("time_zone", "UTC"))
    return {
        "identified_utc": (now or datetime.now(UTC)).isoformat(),
        "battery": identify_battery(audit, float(opts["capacity_kwh"])),
        "limits": observed_limits(audit),
        "pv_geometry": identify_geometry(
            audit,
            opts["latitude"],
            opts["longitude"],
            opts.get("panel_tilt_deg"),
            opts.get("panel_azimuth_deg"),
        ),
        "critical_profile_kw": critical_profile(audit, tz),
    }


def save(data_dir: Path, hardware: dict) -> None:
    (data_dir / HARDWARE_FILE).write_text(
        json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load(data_dir: Path) -> dict | None:
    try:
        return json.loads((data_dir / HARDWARE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def due(data_dir: Path, every: timedelta = timedelta(days=7)) -> bool:
    hardware = load(data_dir)
    if not hardware or not hardware.get("identified_utc"):
        return True
    return datetime.now(UTC) - datetime.fromisoformat(hardware["identified_utc"]) >= every


def effective_options(opts: dict, hardware: dict | None) -> tuple[dict, list[str]]:
    """Options with learned hardware applied when ``autotune`` is on.

    Returns the (possibly) adjusted copy and one note per applied change.
    The configured options are never modified in place.
    """
    effective = dict(opts)
    notes: list[str] = []
    if not opts.get("autotune", False) or not hardware:
        return effective, notes
    battery = hardware.get("battery") or {}
    if battery.get("status") == "ok" and battery.get("capacity_kwh"):
        effective["capacity_kwh"] = float(battery["capacity_kwh"])
        # The identified capacity already reflects ageing: no extra SoH.
        effective["_soh_identified"] = True
        notes.append(f"ёмкость {opts['capacity_kwh']}→{effective['capacity_kwh']:.1f} кВт·ч")
        eta = battery.get("eta_one_way")
        if eta:
            effective["charge_efficiency"] = float(eta)
            effective["discharge_efficiency"] = float(eta)
            notes.append(f"КПД в одну сторону {float(eta):.3f}")
    limits = hardware.get("limits") or {}
    for key, label in (("max_charge_kw", "заряд"), ("max_discharge_kw", "разряд")):
        observed = limits.get(key)
        if observed and observed > float(opts[key]) * RAISE_LIMIT_MARGIN:
            effective[key] = round(float(observed), 2)
            notes.append(f"лимит {label}а {opts[key]}→{effective[key]} кВт")
    geometry = hardware.get("pv_geometry")
    if geometry and geometry.get("reliable"):
        effective["panel_tilt_deg"] = geometry["tilt_deg"]
        effective["panel_azimuth_deg"] = geometry["azimuth_deg"]
        effective["panel_kwp"] = geometry["kwp"]
    return effective, notes


def reshape_reserve(
    reserve_by_hour: list[float],
    starts: pd.DatetimeIndex,
    tz: ZoneInfo,
    profile: list[float] | None,
    duration_hours: float,
    autonomy_hours: float,
    discharge_efficiency: float,
) -> list[float]:
    """Redistribute the outage reserve over the day by critical-load shape.

    An outage starting at hour i needs the critical load of hours
    i..i+duration, not the all-day P90. Each hour's reserve is scaled by its
    window mean over the profile's daily mean, so the average reserve stays
    the same while evenings with a heavier backup load get more. The user's
    minimum autonomy (critical load of the next ``autonomy_hours`` by the
    same profile) is never undercut.
    """
    if not profile or len(profile) != 24:
        return list(reserve_by_hour)
    mean = float(np.mean(profile))
    if mean <= 0:
        return list(reserve_by_hour)
    window = max(1, int(np.ceil(duration_hours)))
    autonomy = max(1, int(np.ceil(autonomy_hours)))
    hours = pd.DatetimeIndex(starts).tz_convert(tz).hour
    reshaped = []
    for reserve, hour in zip(reserve_by_hour, hours, strict=True):
        window_mean = float(np.mean([profile[(hour + k) % 24] for k in range(window)]))
        factor = float(np.clip(window_mean / mean, *PROFILE_FACTOR_BOUNDS))
        minimum = sum(profile[(hour + k) % 24] for k in range(autonomy)) / discharge_efficiency
        reshaped.append(max(reserve * factor, minimum))
    return reshaped
