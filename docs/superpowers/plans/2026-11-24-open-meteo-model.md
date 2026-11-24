# Open-Meteo Model Selection Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Select Auto / ECMWF / ICON / GFS consistently for planning and training.

**Architecture:** Add one small Open-Meteo selection/path helper in the add-on;
integrate existing requests, model readers, training writers and settings.

**Tech Stack:** Python, urllib, pandas, existing pytest and YAML settings.

**Spec:** `docs/superpowers/specs/2026-11-24-open-meteo-model-design.md`

## Global Constraints

- Auto preserves existing data paths and behavior.
- No new dependencies or EMHASS core changes.
- No deployment, publication or inverter writes.

## Review Focus

- Unknown choices fail validation, including manually edited options.
- An offline switch cannot reuse another model's archive.
- Training and live requests select the same model.
- Switching bypasses the previous model's retraining schedule.
- Daily statistics and calibration use the selected model's bundles/history.

### Task 1: Selection and settings

Create `app/weather_models.py`; update `app/options_meta.py` and `config.yaml`.

- [x] Add failing tests in `tests/test_addon_weather_models.py` for supported IDs,
  validation, path isolation and settings compatibility; run to confirm failure.
- [x] Implement `selected(options)`, `parameters(options)`, `scoped(path, model)`
  and `from_data(data)`; retain legacy paths for Auto.
- [x] Run the new tests and options tests.

### Task 2: Forecast and training integration

Update `app/pipeline.py`, `app/retrain.py`, `app/daily.py` and add-on documentation.

- [x] Add failing integration tests for request selection and isolated files.
- [x] Wire requests, archives, bundle paths, retrain state, daily statistics and
  calibration IDs to the selection; add the selected model to reports.
- [x] Check first-switch training, switching back, and same-model offline reuse.
- [x] Run add-on regressions, lint and inspect the final diff.

## Execution record

Implementation proceeds inline in the existing checkout, as authorized by the
user's implementation request. No extra worktree or commit is necessary.


## Verification results

- Final regression: `pytest tests/test_addon*.py -q --disable-warnings --maxfail=5`
  — **348 passed**. The first sandboxed run's only failure was the offline
  panel asset HTTP test (localhost socket permission); the complete rerun
  with localhost permitted passed.
- LightGBM used the existing sklearn-bundled libomp through DYLD_LIBRARY_PATH;
  no dependencies or system libraries were installed.
- Ruff check of changed Python files and `git diff --check` passed.
- The panel's real renderOptions JavaScript rendered all four translated choices
  and the Auto default in RU and EN using the real options metadata.
- Live Open-Meteo requests for ECMWF IFS / ICON Seamless / GFS Seamless each
  returned all five required variables for 97 requested hourly timestamps.
  Previous Runs requests each returned all five day-ahead variables for 24 hours.
- Independent review identified a wildcard calibration identity when bundles
  were absent. A failing test reproduced 72 foreign-model calibration pairs;
  explicit unavailable identities now produce zero pairs for both Auto and ICON.
  The reviewer confirmed the fix and reported no remaining findings.
- No deployment, version bump, commit, or publication was performed.
