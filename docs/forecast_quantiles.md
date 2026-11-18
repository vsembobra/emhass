# Opt-in quantile forecasts (load and PV)

Two standalone modules provide CPU LightGBM quantile forecasts for advisory
reserve planning (see `PLAN.md` phases 3–4). They are **not** wired into the
optimizer: nothing changes unless a caller explicitly loads a saved bundle, and
`try_load()` returning `None` means the caller stays on the configured standard
forecast path.

Both modules live outside the default install: LightGBM is imported lazily and
only required via the optional extra:

```bash
uv sync --extra test --extra quantile   # or: pip install 'emhass[quantile]'
```

On macOS LightGBM also needs the OpenMP runtime (`libomp`), e.g.
`brew install libomp`, or point `DYLD_FALLBACK_LIBRARY_PATH` at a directory
containing `libomp.dylib`.

## Load: `emhass.load_quantiles`

`LoadQuantileBundle.fit()` trains one `objective="quantile"` booster per
quantile (P10/P50/P90) on a complete hourly UTC grid with a `load_kw` target
and optional `temp_air_observed_c`. Features: local hour, day of week, weekend
and a configurable holiday flag (`LoadFeatureConfig.holidays`, empty by
default), load lags 24 h/168 h, trailing 24 h/168 h means, and the observed
temperature 24 h back. Day-of-year sine/cosine features are only fitted when
the training fold spans at least 12 months.

Leakage contract:

- Rebuild the complete hourly grid before computing lags; `shift(24)` on a
  filtered frame counts rows, not hours. `build_features` rejects gappy grids.
- Lag features at `t - 24`/`t - 168` are actuals, known at issue time for any
  horizon up to 24 h. `predict(..., issue_time=...)` refuses longer horizons.
- Rolling means are frozen at the issue instant for multi-step forecasts;
  training rows emulate a one-step-ahead issue, keeping training and inference
  on the same "known at issue time" semantics.
- Missing values stay NaN; LightGBM handles them natively. Cold-start rows
  (all history missing) still get a calendar-only prediction.
- Only *observed* temperature is used, always lagged. Forecasted temperature
  requires an as-of weather archive and is not supported yet.

`bundle.calibrate(frame)` applies a split-conformal additive adjustment to
P10/P90 on a dedicated chronological slice that must end before the test
slice. Predictions are clipped at 0 kW and sorted so `P10 <= P50 <= P90` always
holds. `bundle.save(dir)` writes the three boosters plus `bundle.json`
(feature config, train/calibration windows, model version, LightGBM version);
`LoadQuantileBundle.load()` validates the model version.

CLI (see `--help`):

```bash
python -m emhass.load_quantiles fit --audit hourly_audit.jsonl --load load_clean.jsonl \
    --model models/load_v1 --train-end 2026-10-12T21:00:00+00:00 --calibration-end 2026-10-19T21:00:00+00:00
python -m emhass.load_quantiles predict --audit hourly_audit.jsonl --model models/load_v1 \
    --issue-at 2026-11-01T21:00:00+00:00 --horizon-hours 24 --out forecast.jsonl
```

## PV: `emhass.pv_quantiles`

`PvQuantileBundle` fits quantile boosters to the ratio
`measured / clear-sky power`, where the clear-sky denominator comes from pvlib
(Ineichen clear-sky, Hay-Davies transposition, PVWatts DC at 25 °C) and the
site `PvGeometry` (latitude, longitude, tilt, azimuth, kWp). Rows below
`min_clearsky_kw` are excluded from training and predict exactly 0 kW; ratios
are clipped to `[0, ratio_max]` and power to `[0, kWp]`, so outputs are
physically bounded and quantile-ordered. `bundle.excluded_rows` counts the
censored rows so the report can show how much daylight data was dropped.

When the panel orientation is unconfirmed, `estimate_geometry()` grid-searches
tilt/azimuth on the clearest measured days. With `nameplate_kwp` set, only the
orientation is searched; the result is always marked `estimated_from_data` and
must be confirmed by the owner before operational use.

Weather inputs must be *as-of* forecasts supplied by the caller for training
and inference alike. Open-Meteo's Previous Runs API exposes `*_previous_dayN`
variables — the value predicted N×24 h before the valid time — which matches a
day-ahead issue contract without reanalysis leakage; its Historical Forecast
API stitches runs and must not be used for this. Free Open-Meteo endpoints are
CC BY 4.0 and noncommercial; check the terms before commercial deployment.

## Backtest evaluation

`emhass.battery_replay.replay_site` compares the fixed SOC floor against the
advisory `baseline_floor` policy (and a hindsight bound) on identical actuals,
reserve settings and physical limits, ranking by unmet critical load first,
then own-solar use, then cost. Site experiments with private data live outside
the repository history (e.g. `data/forecast_training/`, git-ignored).
