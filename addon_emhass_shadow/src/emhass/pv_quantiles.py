"""Opt-in PV P10/P50/P90 quantile model on a pvlib clear-sky denominator.

Standalone from the optimizer, like :mod:`emhass.load_quantiles`. LightGBM is
imported lazily; default installs never require it (extra: ``quantile``).

Approach:

* A clear-sky AC/DC power denominator is built with pvlib from the site
  geometry (coordinates, tilt, azimuth, kWp). Site geometry that is not yet
  confirmed by the owner can be estimated from measurements with
  :func:`estimate_geometry` and is then explicitly marked as data-estimated.
* The LightGBM model fits the *ratio* ``measured / clear-sky`` on daylight
  rows where the denominator is meaningful. Night and near-zero clear-sky
  rows are handled outside the model (prediction is exactly zero when the
  clear-sky denominator is below ``min_clearsky_kw``).
* Predictions reconstruct physically bounded power: ratio clipped to
  ``[0, ratio_max]``, power clipped to ``[0, kWp]``, quantiles ordered.
* As-of weather inputs (e.g. Open-Meteo ``*_previous_day1`` variables) must be
  supplied by the caller for both training and inference; reanalysis or
  observed weather is a leakage source and is never read here.
* SOC/export censoring evidence stays at the data-preparation boundary; this
  module only counts how many rows were excluded before fitting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from emhass.load_quantiles import QUANTILE_LABELS, QUANTILES

MODEL_VERSION = "pv-quantiles-1"

DEFAULT_LGB_PARAMS = {
    "objective": "quantile",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_child_samples": 8,
    "deterministic": True,
    "seed": 20260719,
    "verbose": -1,
}

DEFAULT_FEATURES = (
    "clearsky_kw",
    "poa_forecast_kw",
    "cloud_cover_pct",
    "temp_air_c",
    "hour_of_day",
)


def _lightgbm():
    try:
        return import_module("lightgbm")
    except ImportError as error:
        raise ImportError(
            "PV quantile forecasting requires the optional 'quantile' extra "
            "(pip install 'emhass[quantile]' / uv sync --extra quantile)"
        ) from error


@dataclass(frozen=True)
class PvGeometry:
    """One coplanar array group; combine groups before calling when several."""

    latitude: float
    longitude: float
    tilt_deg: float
    azimuth_deg: float  # pvlib convention: 0=N, 90=E, 180=S, 270=W
    kwp: float
    estimated_from_data: bool = False

    def __post_init__(self) -> None:
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise ValueError("latitude/longitude out of range")
        if not 0 <= self.tilt_deg <= 90:
            raise ValueError("tilt_deg must be in [0, 90]")
        if not 0 <= self.azimuth_deg <= 360:
            raise ValueError("azimuth_deg must be in [0, 360]")
        if self.kwp <= 0:
            raise ValueError("kwp must be positive")


def clear_sky_power_kw(geometry: PvGeometry, times: pd.DatetimeIndex) -> pd.Series:
    """Clear-sky array power (kW) via Ineichen + Hay-Davies + PVWatts at 25 °C."""
    if times.tz is None:
        raise ValueError("times must be timezone-aware")
    location = pvlib.location.Location(geometry.latitude, geometry.longitude)
    position = location.get_solarposition(times)
    clearsky = location.get_clearsky(times, model="ineichen")
    poa = pvlib.irradiance.get_total_irradiance(
        geometry.tilt_deg,
        geometry.azimuth_deg,
        position["apparent_zenith"],
        position["azimuth"],
        clearsky["dni"],
        clearsky["ghi"],
        clearsky["dhi"],
    )
    dc_w_per_kwp = pvlib.pvsystem.pvwatts_dc(
        poa["poa_global"], temp_cell=25.0, pdc0=1000.0, gamma_pdc=-0.004
    )
    return pd.Series(dc_w_per_kwp * geometry.kwp / 1000.0, index=times, name="clearsky_kw")


def plane_power_kw(
    geometry: PvGeometry,
    times: pd.DatetimeIndex,
    ghi_wm2: pd.Series,
    dni_wm2: pd.Series,
    dhi_wm2: pd.Series,
) -> pd.Series:
    """Physical power (kW) from irradiance forecasts (e.g. ``*_previous_day1``)."""
    location = pvlib.location.Location(geometry.latitude, geometry.longitude)
    position = location.get_solarposition(times)
    poa = pvlib.irradiance.get_total_irradiance(
        geometry.tilt_deg,
        geometry.azimuth_deg,
        position["apparent_zenith"],
        position["azimuth"],
        dni_wm2,
        ghi_wm2,
        dhi_wm2,
    )
    dc_w_per_kwp = pvlib.pvsystem.pvwatts_dc(
        poa["poa_global"].clip(lower=0), temp_cell=25.0, pdc0=1000.0, gamma_pdc=-0.004
    )
    return pd.Series(dc_w_per_kwp * geometry.kwp / 1000.0, index=times, name="poa_kw")


def estimate_geometry(
    measured_kw: pd.Series,
    latitude: float,
    longitude: float,
    *,
    nameplate_kwp: float | None = None,
    tilts: range = range(0, 50, 5),
    azimuths: range = range(60, 300, 10),
    clear_days: int = 10,
) -> dict:
    """Grid-search tilt/azimuth against the clearest measured days.

    Provisional helper for unconfirmed geometry. Cloudy rows dominate a plain
    MAE, so candidates are scored only on the ``clear_days`` days with the
    highest measured daily maximum. With ``nameplate_kwp`` the scale is fixed
    at the owner's nameplate capacity and only the orientation is searched;
    otherwise a through-origin scale is fitted on the clear days and stored as
    ``fitted_kwp`` (it absorbs soiling/inverter limits and haze, so it can sit
    well below nameplate). Either way the result is marked
    ``estimated_from_data=True`` and must be confirmed by the owner.
    """
    daily_max = measured_kw.groupby(measured_kw.index.date).max()
    clear = set(daily_max.nlargest(clear_days).index)
    on_clear_days = pd.Series(measured_kw.index.date, index=measured_kw.index).isin(clear)
    best = None
    scores = []
    for tilt in tilts:
        for azimuth in azimuths:
            candidate = PvGeometry(latitude, longitude, tilt, azimuth, 1.0)
            clearsky = clear_sky_power_kw(candidate, measured_kw.index)
            clear_clearsky = clearsky[on_clear_days.to_numpy()]
            clear_measured = measured_kw[on_clear_days.to_numpy()]
            if nameplate_kwp is not None:
                scale = float(nameplate_kwp)
            else:
                scale = float(
                    np.dot(clear_measured, clear_clearsky)
                    / np.dot(clear_clearsky, clear_clearsky)
                )
            mae = float(np.mean(np.abs(clear_measured - scale * clear_clearsky)))
            scores.append({"tilt": tilt, "azimuth": azimuth, "mae_kw": mae, "kwp": scale})
            if best is None or mae < best["mae_kw"]:
                best = scores[-1]
    if best is None:
        raise ValueError("not enough daylight rows to estimate geometry")
    geometry = PvGeometry(
        latitude, longitude, best["tilt"], best["azimuth"], best["kwp"], estimated_from_data=True
    )
    return {"geometry": geometry, "scores": scores}


@dataclass
class PvQuantileBundle:
    """Fitted ratio quantiles plus geometry and calibration metadata."""

    geometry: PvGeometry
    ratio_max: float
    min_clearsky_kw: float
    boosters: dict = field(repr=False)  # LGBMRegressor after fit, Booster after load
    excluded_rows: int = 0
    calibration_offsets: dict = field(default_factory=lambda: {"p10": 0.0, "p90": 0.0})
    trained_from: str | None = None
    trained_until: str | None = None
    calibrated_from: str | None = None
    calibrated_until: str | None = None
    model_version: str = MODEL_VERSION
    lightgbm_version: str | None = None

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column in ("pv_measured_kw", "poa_forecast_kw", "cloud_cover_pct", "temp_air_c"):
            if column not in frame.columns:
                raise ValueError(f"frame must contain {column}")
        clearsky = clear_sky_power_kw(self.geometry, frame.index)
        features = pd.DataFrame(index=frame.index)
        features["clearsky_kw"] = clearsky
        features["poa_forecast_kw"] = pd.to_numeric(frame["poa_forecast_kw"], errors="coerce")
        features["cloud_cover_pct"] = pd.to_numeric(frame["cloud_cover_pct"], errors="coerce")
        features["temp_air_c"] = pd.to_numeric(frame["temp_air_c"], errors="coerce")
        features["hour_of_day"] = frame.index.hour
        return features, clearsky

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        geometry: PvGeometry,
        *,
        ratio_max: float = 1.2,
        min_clearsky_kw: float = 0.1,
        params: dict | None = None,
    ) -> PvQuantileBundle:
        """Fit ratio quantiles on rows with a meaningful clear-sky denominator."""
        lgb = _lightgbm()
        if frame.index.tz is None:
            raise ValueError("frame must be indexed by timezone-aware timestamps")
        bundle = cls(geometry, ratio_max, min_clearsky_kw, boosters={})
        features, clearsky = bundle._features(frame)
        target = pd.to_numeric(frame["pv_measured_kw"], errors="coerce")
        usable = (clearsky >= min_clearsky_kw) & target.notna() & (target >= 0)
        bundle.excluded_rows = int((~usable).sum())
        ratio = (target[usable] / clearsky[usable]).clip(0, ratio_max)
        if len(ratio) < 24:
            raise ValueError("at least 24 usable daylight rows are required")
        merged = {**DEFAULT_LGB_PARAMS, **(params or {})}
        x_train = features.loc[usable, list(DEFAULT_FEATURES)]
        for alpha in QUANTILES:
            booster = lgb.LGBMRegressor(alpha=alpha, **merged)
            booster.fit(x_train, ratio.to_numpy())
            bundle.boosters[alpha] = booster
        bundle.trained_from = frame.index[0].isoformat()
        bundle.trained_until = frame.index[-1].isoformat()
        bundle.lightgbm_version = lgb.__version__
        return bundle

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict bounded, ordered PV P10/P50/P90 in kW; exactly zero at night."""
        features, clearsky = self._features(frame)
        x = features[list(DEFAULT_FEATURES)]
        out = pd.DataFrame(index=frame.index)
        for alpha, label in QUANTILE_LABELS.items():
            booster = self.boosters[alpha]
            ratio = (
                booster.predict(x)
                if hasattr(booster, "fitted_")
                else booster.predict(x.to_numpy(dtype=float))
            )
            power = np.clip(ratio, 0.0, self.ratio_max) * clearsky.to_numpy()
            out[f"pv_{label}_kw"] = np.where(clearsky < self.min_clearsky_kw, 0.0, power)
        out["pv_p10_kw"] += self.calibration_offsets["p10"]
        out["pv_p90_kw"] += self.calibration_offsets["p90"]
        clipped = out.clip(lower=0.0, upper=self.geometry.kwp)
        ordered = np.sort(clipped.to_numpy(dtype=float), axis=1)
        return pd.DataFrame(ordered, index=frame.index, columns=clipped.columns)

    def calibrate(self, frame: pd.DataFrame) -> None:
        """Split-conformal additive P10/P90 adjustment (kW) on a separate slice."""
        features, clearsky = self._features(frame)
        target = pd.to_numeric(frame["pv_measured_kw"], errors="coerce")
        usable = (clearsky >= self.min_clearsky_kw) & target.notna()
        if usable.sum() < 12:
            raise ValueError("at least 12 usable calibration rows are required")
        saved, self.calibration_offsets = self.calibration_offsets, {"p10": 0.0, "p90": 0.0}
        raw = self.predict(frame.loc[usable])
        self.calibration_offsets = saved
        y = target.loc[usable].to_numpy(dtype=float)
        self.calibration_offsets = {
            "p90": _conformal(y - raw["pv_p90_kw"].to_numpy(), 0.9),
            "p10": -_conformal(raw["pv_p10_kw"].to_numpy() - y, 0.9),
        }
        self.calibrated_from = frame.index[0].isoformat()
        self.calibrated_until = frame.index[-1].isoformat()

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for alpha in QUANTILES:
            self.boosters[alpha].booster_.save_model(
                str(directory / f"pv_q{int(alpha * 100):03d}.txt")
            )
        metadata = {
            "model_version": self.model_version,
            "lightgbm_version": self.lightgbm_version,
            "geometry": asdict(self.geometry),
            "ratio_max": self.ratio_max,
            "min_clearsky_kw": self.min_clearsky_kw,
            "excluded_rows": self.excluded_rows,
            "calibration_offsets": self.calibration_offsets,
            "trained_from": self.trained_from,
            "trained_until": self.trained_until,
            "calibrated_from": self.calibrated_from,
            "calibrated_until": self.calibrated_until,
            "saved_at_utc": datetime.now(UTC).isoformat(),
        }
        (directory / "bundle.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str) -> PvQuantileBundle:
        lgb = _lightgbm()
        directory = Path(directory)
        metadata = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
        if metadata["model_version"] != MODEL_VERSION:
            raise ValueError(f"unsupported model version {metadata['model_version']!r}")
        bundle = cls(
            geometry=PvGeometry(**metadata["geometry"]),
            ratio_max=metadata["ratio_max"],
            min_clearsky_kw=metadata["min_clearsky_kw"],
            boosters={
                alpha: lgb.Booster(model_file=str(directory / f"pv_q{int(alpha * 100):03d}.txt"))
                for alpha in QUANTILES
            },
            excluded_rows=metadata["excluded_rows"],
            calibration_offsets=dict(metadata["calibration_offsets"]),
            trained_from=metadata["trained_from"],
            trained_until=metadata["trained_until"],
            calibrated_from=metadata["calibrated_from"],
            calibrated_until=metadata["calibrated_until"],
            lightgbm_version=metadata["lightgbm_version"],
        )
        return bundle


def _conformal(errors: np.ndarray, alpha: float) -> float:
    finite = errors[np.isfinite(errors)]
    if not finite.size:
        return 0.0
    return float(np.quantile(finite, alpha, method="higher"))


def try_load(directory: Path | str) -> PvQuantileBundle | None:
    """Load a bundle, returning None when missing or unusable (fallback path)."""
    try:
        return PvQuantileBundle.load(directory)
    except (OSError, ValueError, ImportError, json.JSONDecodeError, KeyError):
        return None
