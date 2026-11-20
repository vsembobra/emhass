"""Self-correction between retrains: forecast log, online calibration, drift.

Every run appends the RAW model forecasts it issued (before any correction)
to an append-only log. Once their hours are observed, the issued-vs-actual
pairs of the last two weeks drive a daily correction of the next forecasts:

* load P50 is shifted per local hour by the median residual (additive kW),
  PV P50 is scaled per local hour by actual/forecast energy (multiplicative,
  so night stays exactly zero and dawn cannot fabricate generation);
* the P10/P90 half-widths are scaled so that ~90 % of recent actuals sit
  above the corrected P10 and ~90 % below the corrected P90 — a rolling
  split-conformal recalibration that keeps the P10-P90 band near 80 %
  coverage as the site changes, without waiting for the next retrain.

Every estimate is shrunk towards "no correction" by its sample count, so a
handful of pairs barely moves the forecast. Corrections are always fitted on
raw model outputs, never on already corrected ones, so they cannot compound,
and only on outputs of the model version now in use: a retrained model has
its own errors, and inheriting the old model's correction would compensate
a bias it may no longer have (or double it).

The same pairs give a drift signal: a persistent bias or a collapsed band
over the last week marks the model as stale and lets the retrain run early.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LOG_RETENTION_DAYS = 60
CALIBRATION_WINDOW_DAYS = 14
DRIFT_WINDOW_DAYS = 7
MIN_PAIRS = 48  # below this the calibration is the identity
HOUR_SHRINK = 10.0  # pairs per hour at which a per-hour shift is half-trusted
SCALE_SHRINK = 48.0  # pairs at which the band scale is half-trusted
SCALE_BOUNDS = (0.3, 4.0)
PV_FACTOR_BOUNDS = (0.5, 1.5)
TARGET_ONE_SIDED = 0.9  # P10/P90 one-sided target -> 80 % central band
DRIFT_MIN_PAIRS = 72
DRIFT_MAX_REL_BIAS = 0.25
DRIFT_MIN_COVERAGE = 0.6
NIGHT_KW = 0.01
EPS = 1e-6
TARGETS = ("load", "pv")


# -- forecast log -------------------------------------------------------------


def append_forecast_log(path: Path, entry: dict, now: datetime | None = None) -> None:
    """Append one issued forecast; drop entries older than the retention."""
    now = now or datetime.now(UTC)
    entries = read_forecast_log(path)
    cutoff = now - timedelta(days=LOG_RETENTION_DAYS)
    entries = [e for e in entries if datetime.fromisoformat(e["issue_utc"]) >= cutoff]
    entries = [e for e in entries if e["issue_utc"] != entry["issue_utc"]]
    entries.append(entry)
    path.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8"
    )


def read_forecast_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn last line must not break the whole log
    return entries


def bundle_version(directory: Path) -> str | None:
    """Short content hash of a saved model bundle: changes with every
    promoted retrain, stays put when the retrain kept the live model."""
    try:
        return hashlib.sha1((Path(directory) / "bundle.json").read_bytes()).hexdigest()[:12]
    except OSError:
        return None


def log_entry(
    issue: datetime,
    starts: pd.DatetimeIndex,
    load: tuple[list, list, list],
    load_method: str,
    pv: tuple[list, list, list],
    pv_method: str,
    reserve_kwh_by_hour: list[float] | None = None,
    versions: dict[str, str | None] | None = None,
) -> dict:
    """Serializable record of one run's RAW forecasts, model versions and reserve."""
    versions = versions or {}

    def block(values, method, version):
        p10, p50, p90 = values
        return {
            "method": method,
            "version": version,
            "p10": [round(float(v), 4) for v in p10],
            "p50": [round(float(v), 4) for v in p50],
            "p90": [round(float(v), 4) for v in p90],
        }

    entry = {
        "issue_utc": pd.Timestamp(issue).isoformat(),
        "starts_utc": [s.isoformat() for s in starts],
        "load": block(load, load_method, versions.get("load")),
        "pv": block(pv, pv_method, versions.get("pv")),
    }
    if reserve_kwh_by_hour is not None:
        entry["reserve_kwh_by_hour"] = [round(float(v), 4) for v in reserve_kwh_by_hour]
    return entry


# -- issued-vs-actual pairs ---------------------------------------------------


def actual_series(audit: pd.DataFrame, target: str) -> pd.Series:
    """Clean actuals in kW: the same quality filters the models train on."""
    if target == "load":
        if "load_kw" not in audit or "load_filter" not in audit:
            return pd.Series(dtype=float)
        return pd.to_numeric(audit["load_kw"], errors="coerce").where(
            audit["load_filter"] == "keep"
        )
    if any(c not in audit for c in ("pv_measured_kw", "pv_filter", "solar_elevation_deg")):
        return pd.Series(dtype=float)
    daylight = pd.to_numeric(audit["solar_elevation_deg"], errors="coerce") > 10
    return pd.to_numeric(audit["pv_measured_kw"], errors="coerce").where(
        (audit["pv_filter"] == "keep") & daylight
    )


def forecast_pairs(
    log: list[dict],
    audit: pd.DataFrame,
    target: str,
    since: datetime,
    version: str | None = None,
) -> pd.DataFrame:
    """Issued model forecasts joined with actuals for hours >= since.

    Columns: issue, start, lead (1 = issue hour), p10, p50, p90, actual.
    Fallback (non-model) forecasts are skipped: correcting the model with the
    errors of a different method would teach it the wrong lesson. With
    ``version`` only forecasts of that model version count (entries logged
    before versions were recorded belong to an unknown model and are skipped).
    """
    actual = actual_series(audit, target).dropna()
    if actual.empty:
        return pd.DataFrame(columns=["issue", "start", "lead", "p10", "p50", "p90", "actual"])
    actual.index = pd.DatetimeIndex(actual.index).tz_convert(UTC)
    since = pd.Timestamp(since).tz_convert(UTC)
    frames = []
    for entry in log:
        block = entry.get(target) or {}
        if block.get("method") != "model":
            continue
        if version is not None and block.get("version") != version:
            continue
        issue = pd.Timestamp(entry["issue_utc"]).tz_convert(UTC)
        starts = pd.DatetimeIndex(pd.to_datetime(entry["starts_utc"], utc=True))
        frame = pd.DataFrame(
            {
                "issue": issue,
                "start": starts,
                "lead": ((starts - issue) / pd.Timedelta(hours=1)).astype(int) + 1,
                "p10": block["p10"],
                "p50": block["p50"],
                "p90": block["p90"],
            }
        )
        frame = frame[(frame["start"] >= since) & frame["start"].isin(actual.index)]
        if frame.empty:
            continue
        frame["actual"] = actual.reindex(pd.DatetimeIndex(frame["start"])).to_numpy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["issue", "start", "lead", "p10", "p50", "p90", "actual"])
    pairs = pd.concat(frames, ignore_index=True)
    if target == "pv":
        pairs = pairs[(pairs["p50"] > NIGHT_KW) | (pairs["p90"] > NIGHT_KW)]
    return pairs.reset_index(drop=True)


# -- calibration --------------------------------------------------------------


def _shrink(value: float, neutral: float, count: float, prior: float) -> float:
    weight = count / (count + prior) if count > 0 else 0.0
    return neutral + (value - neutral) * weight


def _coverage(actual, lower, upper) -> float | None:
    actual = np.asarray(actual, dtype=float)
    if not actual.size:
        return None
    return float(np.mean((actual >= lower) & (actual <= upper)))


@dataclass
class Calibration:
    """Per-hour P50 correction plus P10/P90 band scales for one target."""

    target: str
    pairs: int = 0
    hourly: list[float] = field(default_factory=list)  # load: +kW, pv: x factor
    scale_lo: float = 1.0
    scale_hi: float = 1.0
    coverage_raw: float | None = None
    coverage_calibrated: float | None = None

    def __post_init__(self) -> None:
        if not self.hourly:
            self.hourly = [0.0 if self.target == "load" else 1.0] * 24

    @property
    def active(self) -> bool:
        return self.pairs >= MIN_PAIRS

    def apply(
        self, p10, p50, p90, starts: pd.DatetimeIndex, tz: ZoneInfo
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p10 = np.asarray(p10, dtype=float)
        p50 = np.asarray(p50, dtype=float)
        p90 = np.asarray(p90, dtype=float)
        if not self.active:
            return p10, p50, p90
        hours = np.asarray(pd.DatetimeIndex(starts).tz_convert(tz).hour)
        shift = np.asarray(self.hourly, dtype=float)[hours]
        if self.target == "load":
            mid = p50 + shift
            lo = p50 - p10
            hi = p90 - p50
        else:
            mid = p50 * shift
            lo = (p50 - p10) * shift
            hi = (p90 - p50) * shift
        out = np.vstack(
            [
                mid - self.scale_lo * np.maximum(lo, 0.0),
                mid,
                mid + self.scale_hi * np.maximum(hi, 0.0),
            ]
        )
        out = np.sort(np.clip(out, 0.0, None), axis=0)
        if self.target == "pv":
            night = p90 <= NIGHT_KW
            out[:, night] = 0.0
        return out[0], out[1], out[2]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hourly"] = [round(v, 4) for v in self.hourly]
        data["scale_lo"] = round(self.scale_lo, 3)
        data["scale_hi"] = round(self.scale_hi, 3)
        data["active"] = self.active
        return data


def fit_calibration(pairs: pd.DataFrame, target: str, tz: ZoneInfo) -> Calibration:
    """Fit the per-hour correction and band scales from recent pairs."""
    calibration = Calibration(target=target, pairs=len(pairs))
    if len(pairs) < MIN_PAIRS:
        return calibration
    actual = pairs["actual"].to_numpy(dtype=float)
    p10 = pairs["p10"].to_numpy(dtype=float)
    p50 = pairs["p50"].to_numpy(dtype=float)
    p90 = pairs["p90"].to_numpy(dtype=float)
    hours = np.asarray(pd.DatetimeIndex(pairs["start"]).tz_convert(tz).hour)
    calibration.coverage_raw = _coverage(actual, p10, p90)

    hourly = []
    for hour in range(24):
        mask = hours == hour
        count = int(mask.sum())
        if target == "load":
            value = float(np.median(actual[mask] - p50[mask])) if count else 0.0
            hourly.append(_shrink(value, 0.0, count, HOUR_SHRINK))
        else:
            forecast_kwh = float(p50[mask].sum()) if count else 0.0
            if forecast_kwh > 0.5:
                factor = float(np.clip(actual[mask].sum() / forecast_kwh, *PV_FACTOR_BOUNDS))
            else:
                factor = 1.0
            hourly.append(_shrink(factor, 1.0, count, HOUR_SHRINK))
    calibration.hourly = hourly

    shift = np.asarray(hourly)[hours]
    if target == "load":
        mid, lo, hi = p50 + shift, p50 - p10, p90 - p50
    else:
        mid, lo, hi = p50 * shift, (p50 - p10) * shift, (p90 - p50) * shift
    z_lo = (mid - actual) / np.maximum(lo, EPS)
    z_hi = (actual - mid) / np.maximum(hi, EPS)
    count = float(len(pairs))
    for side, z in (("scale_lo", z_lo), ("scale_hi", z_hi)):
        raw = float(np.clip(np.quantile(z, TARGET_ONE_SIDED), *SCALE_BOUNDS))
        setattr(calibration, side, _shrink(raw, 1.0, count, SCALE_SHRINK))

    c10, _, c90 = calibration.apply(p10, p50, p90, pd.DatetimeIndex(pairs["start"]), tz)
    calibration.coverage_calibrated = _coverage(actual, c10, c90)
    return calibration


# -- drift --------------------------------------------------------------------


def drift_status(pairs_by_target: dict[str, pd.DataFrame], now: datetime) -> dict:
    """Bias and band coverage of the RAW forecasts over the last week."""
    since = pd.Timestamp(now).tz_convert(UTC) - pd.Timedelta(days=DRIFT_WINDOW_DAYS)
    status: dict = {"checked_utc": pd.Timestamp(now).isoformat(), "drift": False, "reasons": []}
    for target, pairs in pairs_by_target.items():
        recent = pairs[pd.DatetimeIndex(pairs["start"]) >= since] if len(pairs) else pairs
        info: dict = {"pairs": int(len(recent))}
        if len(recent) >= DRIFT_MIN_PAIRS:
            actual = recent["actual"].to_numpy(dtype=float)
            p50 = recent["p50"].to_numpy(dtype=float)
            rel_bias = float((p50.mean() - actual.mean()) / max(actual.mean(), EPS))
            coverage = _coverage(actual, recent["p10"].to_numpy(), recent["p90"].to_numpy())
            info.update(rel_bias=round(rel_bias, 3), coverage=round(coverage, 3))
            if abs(rel_bias) > DRIFT_MAX_REL_BIAS:
                status["reasons"].append(f"{target}: смещение {rel_bias:+.0%}")
            if coverage < DRIFT_MIN_COVERAGE:
                status["reasons"].append(f"{target}: охват P10–P90 {coverage:.0%}")
        status[target] = info
    status["drift"] = bool(status["reasons"])
    return status


def self_correct(
    log: list[dict],
    audit: pd.DataFrame,
    tz: ZoneInfo,
    now: datetime,
    versions: dict[str, str | None] | None = None,
) -> tuple[dict[str, Calibration], dict]:
    """Calibrations for both targets plus the drift status, from one log read.

    ``versions`` maps each target to the live model version; its pairs alone
    are used, so a promoted retrain starts from no correction. Without it
    every model forecast counts (callers that do not track versions).
    """
    since = pd.Timestamp(now).tz_convert(UTC) - pd.Timedelta(days=CALIBRATION_WINDOW_DAYS)
    versions = versions or {}
    pairs = {
        target: forecast_pairs(log, audit, target, since, versions.get(target))
        for target in TARGETS
    }
    calibrations = {target: fit_calibration(pairs[target], target, tz) for target in TARGETS}
    return calibrations, drift_status(pairs, now)
