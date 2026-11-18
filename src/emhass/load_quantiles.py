"""Opt-in LightGBM P10/P50/P90 load quantile forecaster (CPU, standalone).

This module is deliberately separate from the optimizer. It is only used when a
caller explicitly loads a trained bundle; nothing here is imported by the
default EMHASS runtime and LightGBM is imported lazily, so default installs
never require the ``lightgbm`` package (extra: ``quantile``).

Feature and leakage contract:

* Rows live on a complete hourly UTC grid (rebuild it from the audit before
  computing lags, otherwise ``shift(24)`` counts remaining rows, not hours).
* Lag features (24 h and 168 h) read the actual series at ``t - lag``. For a
  forecast issued at ``T`` with a 24 h horizon every ``t - 24`` is before
  ``T``, so lagged actuals are always available at issue time.
* Trailing rolling means and the lagged observed temperature are computed as
  of the forecast issue time and held constant across the horizon. Training
  rows emulate a one-step-ahead issue, which keeps training and inference on
  the same "known at issue time" semantics without recursive prediction.
* Observed (measured) temperature is only used with a lag of at least the
  horizon, so no weather forecast archive is required. Forecasted temperature
  is intentionally unsupported until an as-of weather archive exists.
* Day-of-year sine/cosine features are only fitted when the training fold
  spans at least ``seasonal_min_days`` (12 months by default).
* Missing lag or temperature values stay missing (NaN); LightGBM handles NaN
  natively. A row with all-missing history still gets a calendar-only
  prediction (documented cold-start behaviour).

Calibration uses a split-conformal additive adjustment of P10/P90 fitted on a
dedicated chronological slice that ends before the test slice. P50 is left as
the raw quantile regression output.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import lightgbm as lgb

MODEL_VERSION = "load-quantiles-1"
QUANTILES = (0.1, 0.5, 0.9)
QUANTILE_LABELS = {0.1: "p10", 0.5: "p50", 0.9: "p90"}

DEFAULT_LGB_PARAMS = {
    "objective": "quantile",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "deterministic": True,
    "seed": 20260719,
    "verbose": -1,
}

DEFAULT_FEATURES = (
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "load_lag_24",
    "load_lag_168",
    "load_roll_mean_24",
    "load_roll_mean_168",
    "temp_lag_24",
)
SEASONAL_FEATURES = ("day_of_year_sin", "day_of_year_cos")


def _lightgbm() -> lgb:
    try:
        return import_module("lightgbm")
    except ImportError as error:
        raise ImportError(
            "load quantile forecasting requires the optional 'quantile' extra "
            "(pip install 'emhass[quantile]' / uv sync --extra quantile)"
        ) from error


@dataclass(frozen=True)
class LoadFeatureConfig:
    """Feature switches shared by training and inference."""

    time_zone: str = "Europe/Istanbul"
    lag_hours: tuple[int, ...] = (24, 168)
    rolling_hours: tuple[int, ...] = (24, 168)
    rolling_min_periods: int = 12
    temperature_lag_hours: int = 24
    # Recurring local holidays as "MM-DD" strings. Empty by default: with a
    # 120-day history the holiday effect is not identifiable, so the flag is
    # opt-in rather than guessed.
    holidays: frozenset[str] = frozenset()
    seasonal_min_days: int = 365

    def __post_init__(self) -> None:
        if not self.lag_hours or any(lag <= 0 for lag in self.lag_hours):
            raise ValueError("lag_hours must be positive")
        if not self.rolling_hours or any(window <= 0 for window in self.rolling_hours):
            raise ValueError("rolling_hours must be positive")
        if not 1 <= self.rolling_min_periods:
            raise ValueError("rolling_min_periods must be positive")
        if self.temperature_lag_hours <= 0:
            raise ValueError("temperature_lag_hours must be positive")
        if self.seasonal_min_days <= 0:
            raise ValueError("seasonal_min_days must be positive")
        for holiday in self.holidays:
            month, day = holiday.split("-")
            if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
                raise ValueError(f"invalid holiday {holiday!r}, expected 'MM-DD'")


def _check_hourly_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("frame must be indexed by timestamps")
    if index.tz is None:
        raise ValueError("grid timestamps must be timezone-aware UTC")
    index = index.tz_convert(UTC)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("grid must be strictly increasing without duplicates")
    if len(index) > 1 and (index.to_series().diff().dropna() != pd.Timedelta(hours=1)).any():
        raise ValueError(
            "grid must be a complete hourly series; rebuild it from the audit "
            "grid before computing lags so shift(24) means 24 hours"
        )
    return index


def _rolling_as_of(series: pd.Series, window: int, min_periods: int, at: pd.Timestamp) -> float:
    """Trailing mean of the ``window`` hourly slots strictly before ``at``."""
    history = series.loc[: at - pd.Timedelta(hours=1)].tail(window)
    if history.count() < min_periods:
        return math.nan
    return float(history.mean())


def build_features(
    grid: pd.DataFrame,
    config: LoadFeatureConfig | None = None,
    *,
    seasonal: bool = False,
    issue_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build the model feature frame on a complete hourly UTC grid.

    ``grid`` must contain ``load_kw`` and may contain ``temp_air_observed_c``;
    both keep missing values as NaN. With ``issue_time`` the trailing rolling
    features are frozen at the issue instant (causal 24 h-ahead inference);
    without it each row emulates a one-step-ahead issue at its own timestamp.
    """
    config = config or LoadFeatureConfig()
    index = _check_hourly_grid(grid.index)
    if "load_kw" not in grid.columns:
        raise ValueError("grid must contain a load_kw column")
    if issue_time is not None:
        issue_time = pd.Timestamp(issue_time)
        if issue_time.tzinfo is None:
            raise ValueError("issue_time must be timezone-aware")
        issue_time = issue_time.tz_convert(UTC)
        if issue_time < index[0] or issue_time > index[-1] + pd.Timedelta(hours=1):
            raise ValueError("issue_time must lie within the grid")

    load = pd.to_numeric(grid["load_kw"], errors="coerce")
    local = index.tz_convert(config.time_zone)
    features = pd.DataFrame(index=index)
    features["hour_of_day"] = local.hour
    features["day_of_week"] = local.day_of_week
    features["is_weekend"] = (local.day_of_week >= 5).astype(float)
    month_day = pd.Index(local.strftime("%m-%d"))
    features["is_holiday"] = month_day.isin(config.holidays).astype(float)
    if seasonal:
        day_angle = 2 * math.pi * (local.day_of_year - 1) / 365.25
        features["day_of_year_sin"] = np.sin(day_angle)
        features["day_of_year_cos"] = np.cos(day_angle)

    for lag in config.lag_hours:
        features[f"load_lag_{lag}"] = load.shift(lag)
    for window in config.rolling_hours:
        column = f"load_roll_mean_{window}"
        if issue_time is None:
            features[column] = load.shift(1).rolling(
                window, min_periods=config.rolling_min_periods
            ).mean()
        else:
            features[column] = _rolling_as_of(
                load, window, config.rolling_min_periods, issue_time
            )
    if "temp_air_observed_c" in grid.columns:
        temp = pd.to_numeric(grid["temp_air_observed_c"], errors="coerce")
        lag = config.temperature_lag_hours
        if issue_time is None:
            features[f"temp_lag_{lag}"] = temp.shift(lag)
        else:
            # Temperature at t - lag is an actual for every step of a horizon
            # up to `lag` hours, so per-step lagged values stay causal.
            features[f"temp_lag_{lag}"] = temp.shift(lag)
    return features


def feature_names(config: LoadFeatureConfig, seasonal: bool) -> list[str]:
    names = [name for name in DEFAULT_FEATURES if not name.startswith("temp_lag_")]
    if seasonal:
        names[4:4] = list(SEASONAL_FEATURES)
    names.append(f"temp_lag_{config.temperature_lag_hours}")
    return names


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"feature frame is missing columns: {missing}")


def pinball_loss(actual: np.ndarray, prediction: np.ndarray, alpha: float) -> float:
    """Mean pinball (quantile) loss; lower is better."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if actual.shape != prediction.shape or not actual.size:
        raise ValueError("actual and prediction must be non-empty and share a shape")
    error = actual - prediction
    return float(np.mean(np.maximum(alpha * error, (alpha - 1) * error)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actuals inside [lower, upper]."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not (actual.shape == lower.shape == upper.shape) or not actual.size:
        raise ValueError("inputs must be non-empty and share a shape")
    return float(np.mean((actual >= lower) & (actual <= upper)))


@dataclass
class LoadQuantileBundle:
    """Fitted P10/P50/P90 boosters plus the metadata needed to reproduce them."""

    config: LoadFeatureConfig
    seasonal: bool
    # LGBMRegressor instances after fit(), plain Booster after load().
    boosters: dict = field(repr=False)
    calibration_offsets: dict[str, float] = field(default_factory=lambda: {"p10": 0.0, "p90": 0.0})
    trained_from: str | None = None
    trained_until: str | None = None
    calibrated_from: str | None = None
    calibrated_until: str | None = None
    model_version: str = MODEL_VERSION
    lightgbm_version: str | None = None

    @property
    def columns(self) -> list[str]:
        return feature_names(self.config, self.seasonal)

    @classmethod
    def fit(
        cls,
        grid: pd.DataFrame,
        config: LoadFeatureConfig | None = None,
        *,
        seasonal: bool | None = None,
        params: dict | None = None,
    ) -> LoadQuantileBundle:
        """Fit one CPU quantile booster per quantile on rows with a valid target."""
        lgb = _lightgbm()
        config = config or LoadFeatureConfig()
        if seasonal is None:
            span_days = (grid.index[-1] - grid.index[0]).total_seconds() / 86400
            seasonal = span_days >= config.seasonal_min_days
        features = build_features(grid, config, seasonal=seasonal)
        names = feature_names(config, seasonal)
        _require_columns(features, names)
        target = pd.to_numeric(grid["load_kw"], errors="coerce")
        valid = target.notna()
        x_train = features.loc[valid, names]
        y_train = target.loc[valid].to_numpy(dtype=float)
        if len(y_train) < 24:
            raise ValueError("at least 24 valid training rows are required")
        merged = {**DEFAULT_LGB_PARAMS, **(params or {})}
        boosters = {}
        for alpha in QUANTILES:
            booster = lgb.LGBMRegressor(alpha=alpha, **merged)
            booster.fit(x_train, y_train)
            boosters[alpha] = booster
        return cls(
            config=config,
            seasonal=seasonal,
            boosters=boosters,
            trained_from=grid.index[0].isoformat(),
            trained_until=grid.index[-1].isoformat(),
            lightgbm_version=lgb.__version__,
        )

    def _raw_predict(self, features: pd.DataFrame) -> pd.DataFrame:
        names = self.columns
        _require_columns(features, names)
        x = features[names]
        out = pd.DataFrame(index=features.index)
        for alpha, label in QUANTILE_LABELS.items():
            booster = self.boosters[alpha]
            # Fitted regressors keep their feature names; boosters loaded from
            # disk are plain Boosters and take raw arrays in the same order.
            values = (
                booster.predict(x)
                if hasattr(booster, "fitted_")
                else booster.predict(x.to_numpy(dtype=float))
            )
            out[f"load_{label}_kw"] = values
        return out

    @staticmethod
    def _conformal_offset(errors: np.ndarray, alpha: float) -> float:
        finite = errors[np.isfinite(errors)]
        if not finite.size:
            return 0.0
        return float(np.quantile(finite, alpha, method="higher"))

    def calibrate(self, grid: pd.DataFrame) -> None:
        """Split-conformal additive P10/P90 adjustment on a chronological slice.

        The slice must end before the test slice; the caller owns the split.
        Offsets target one-sided 90% coverage on the calibration rows.
        """
        features = build_features(grid, self.config, seasonal=self.seasonal)
        target = pd.to_numeric(grid["load_kw"], errors="coerce")
        valid = target.notna()
        if valid.sum() < 24:
            raise ValueError("at least 24 valid calibration rows are required")
        raw = self._raw_predict(features.loc[valid])
        y = target.loc[valid].to_numpy(dtype=float)
        self.calibration_offsets = {
            "p90": self._conformal_offset(y - raw["load_p90_kw"].to_numpy(), 0.9),
            "p10": -self._conformal_offset(raw["load_p10_kw"].to_numpy() - y, 0.9),
        }
        self.calibrated_from = grid.index[0].isoformat()
        self.calibrated_until = grid.index[-1].isoformat()

    def predict(
        self, grid: pd.DataFrame, *, issue_time: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Predict clipped, ordered P10/P50/P90 in kW for every grid row.

        With ``issue_time`` the rolling features are frozen at the issue
        instant; the horizon may not exceed the smallest configured lag,
        because beyond it the lagged actuals would not be known at issue time.
        """
        features = build_features(grid, self.config, seasonal=self.seasonal, issue_time=issue_time)
        if issue_time is not None:
            issue = pd.Timestamp(issue_time).tz_convert(UTC)
            causal_horizon = min(*self.config.lag_hours, self.config.temperature_lag_hours)
            if int((features.index >= issue).sum()) > causal_horizon:
                raise ValueError(
                    f"horizon from issue_time may not exceed {causal_horizon} h "
                    "(smallest configured lag); lagged actuals would not be known"
                )
        raw = self._raw_predict(features)
        raw["load_p10_kw"] += self.calibration_offsets["p10"]
        raw["load_p90_kw"] += self.calibration_offsets["p90"]
        # Physical bound: load is non-negative power.
        clipped = raw.clip(lower=0.0)
        # Quantile crossing is possible after boosting and conformal shifts;
        # enforce p10 <= p50 <= p90 elementwise.
        ordered = np.sort(clipped.to_numpy(dtype=float), axis=1)
        return pd.DataFrame(ordered, index=features.index, columns=clipped.columns)

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for alpha in QUANTILES:
            self.boosters[alpha].booster_.save_model(
                str(directory / f"load_q{int(alpha * 100):03d}.txt")
            )
        metadata = {
            "model_version": self.model_version,
            "lightgbm_version": self.lightgbm_version,
            "config": {
                **asdict(self.config),
                "holidays": sorted(self.config.holidays),
                "lag_hours": list(self.config.lag_hours),
                "rolling_hours": list(self.config.rolling_hours),
            },
            "seasonal": self.seasonal,
            "feature_names": self.columns,
            "calibration_offsets": self.calibration_offsets,
            "trained_from": self.trained_from,
            "trained_until": self.trained_until,
            "calibrated_from": self.calibrated_from,
            "calibrated_until": self.calibrated_until,
            "saved_at_utc": datetime.now(UTC).isoformat(),
        }
        (directory / "bundle.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str) -> LoadQuantileBundle:
        lgb = _lightgbm()
        directory = Path(directory)
        metadata = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
        if metadata["model_version"] != MODEL_VERSION:
            raise ValueError(
                f"unsupported model version {metadata['model_version']!r}, "
                f"expected {MODEL_VERSION!r}"
            )
        config_data = metadata["config"]
        config = LoadFeatureConfig(
            time_zone=config_data["time_zone"],
            lag_hours=tuple(config_data["lag_hours"]),
            rolling_hours=tuple(config_data["rolling_hours"]),
            rolling_min_periods=config_data["rolling_min_periods"],
            temperature_lag_hours=config_data["temperature_lag_hours"],
            holidays=frozenset(config_data["holidays"]),
            seasonal_min_days=config_data["seasonal_min_days"],
        )
        boosters = {}
        for alpha in QUANTILES:
            boosters[alpha] = lgb.Booster(
                model_file=str(directory / f"load_q{int(alpha * 100):03d}.txt")
            )
        return cls(
            config=config,
            seasonal=metadata["seasonal"],
            boosters=boosters,
            calibration_offsets=dict(metadata["calibration_offsets"]),
            trained_from=metadata["trained_from"],
            trained_until=metadata["trained_until"],
            calibrated_from=metadata["calibrated_from"],
            calibrated_until=metadata["calibrated_until"],
            model_version=metadata["model_version"],
            lightgbm_version=metadata["lightgbm_version"],
        )


def try_load(directory: Path | str) -> LoadQuantileBundle | None:
    """Load a bundle, returning None when it is missing or unusable.

    Callers use this for the documented fallback: no usable model means the
    caller stays on the configured (standard) forecast path.
    """
    try:
        return LoadQuantileBundle.load(directory)
    except (OSError, ValueError, ImportError, json.JSONDecodeError, KeyError):
        return None


def _read_jsonl(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True)
    return frame.set_index("start_utc").sort_index()


def _full_grid(audit_path: Path, load_path: Path) -> pd.DataFrame:
    """Rebuild the complete hourly UTC grid and join the clean load target."""
    audit = _read_audit(audit_path)
    load = _read_jsonl(load_path)
    grid = audit[["temp_air_observed_c"]].copy()
    grid["load_kw"] = load["load_kw"]
    return grid


def _read_audit(audit_path: Path) -> pd.DataFrame:
    audit = _read_jsonl(audit_path)
    if (audit.index.to_series().diff().dropna() != pd.Timedelta(hours=1)).any():
        raise ValueError("audit grid is not a complete hourly series")
    return audit


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit_parser = commands.add_parser("fit", help="fit and calibrate a quantile bundle")
    commands.add_parser("predict", help="predict with a saved bundle")
    for command in ("fit", "predict"):
        sub = commands.choices[command]
        sub.add_argument("--audit", type=Path, required=True, help="hourly audit JSONL")
        sub.add_argument("--load", type=Path, help="clean load JSONL (fit only)")
        sub.add_argument("--model", type=Path, required=True, help="bundle directory")
    fit_parser = commands.choices["fit"]
    fit_parser.add_argument("--train-end", required=True, help="ISO instant, exclusive")
    fit_parser.add_argument(
        "--calibration-end", required=True, help="ISO instant, exclusive; test starts here"
    )
    predict_parser = commands.choices["predict"]
    predict_parser.add_argument("--issue-at", required=True, help="forecast issue instant, ISO")
    predict_parser.add_argument("--horizon-hours", type=int, default=24)
    predict_parser.add_argument("--out", type=Path, help="forecast JSONL output")

    args = parser.parse_args(argv)
    if args.command == "fit":
        if args.load is None:
            parser.error("fit requires --load")
        grid = _full_grid(args.audit, args.load)
        train_end = pd.Timestamp(args.train_end)
        calibration_end = pd.Timestamp(args.calibration_end)
        if not train_end < calibration_end:
            parser.error("--train-end must be before --calibration-end")
        bundle = LoadQuantileBundle.fit(grid.loc[: train_end - pd.Timedelta(hours=1)])
        bundle.calibrate(grid.loc[train_end : calibration_end - pd.Timedelta(hours=1)])
        bundle.save(args.model)
        print(json.dumps({"model": str(args.model), "trained_until": bundle.trained_until}))
    else:
        bundle = LoadQuantileBundle.load(args.model)
        audit = _read_audit(args.audit)
        issue_at = pd.Timestamp(args.issue_at)
        if issue_at.tzinfo is None:
            parser.error("--issue-at must include a timezone offset")
        issue_at = issue_at.tz_convert(UTC)
        starts = pd.date_range(
            issue_at, periods=args.horizon_hours, freq="h"
        )
        history_end = starts[-1]
        window = audit.loc[:history_end, ["temp_air_observed_c"]].copy()
        window["load_kw"] = audit["load_kw"]
        # Horizon rows have no actuals yet; keep their target/features NaN.
        grid = window.reindex(window.index.union(starts))
        prediction = bundle.predict(grid, issue_time=issue_at).loc[starts]
        records = [
            {
                "available_at": issue_at.isoformat(),
                "start_utc": start.isoformat(),
                "units": "kW",
                "model": "lightgbm-quantile",
                "model_version": bundle.model_version,
                **{column: float(row[column]) for column in prediction.columns},
            }
            for start, (_, row) in zip(starts, prediction.iterrows(), strict=True)
        ]
        output = "\n".join(json.dumps(record) for record in records) + "\n"
        if args.out:
            args.out.write_text(output, encoding="utf-8")
        else:
            print(output, end="")


if __name__ == "__main__":
    main()
