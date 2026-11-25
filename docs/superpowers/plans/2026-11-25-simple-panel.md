# Simple Panel Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline.

**Goal:** A clear operational panel with Now, History and Settings tabs.
**Architecture:** Reuse existing renderers and Supervisor saves. A standalone
panel_state.js derives honest display states; additive Python status/chart fields
supply observation time and consumption. Keep ECharts bundled.
**Tech Stack:** Python/pytest, vanilla JS, Node's built-in test runner, ECharts.
**Spec:** docs/superpowers/specs/2026-11-25-simple-panel-design.md

## Global Constraints

RU/EN, responsive, offline; all 57 settings retained. No actuation/scheduler/core
changes, dependencies, live writes, deployment, or edits to .tmp_friend.

## Review Focus

- Missing/null values never become zero SOC or a safe-looking autonomy.
- Switching configured mode must not claim an older report applied it.
- Grid and report age remain explicit, including stale/offline states.
- Unsaved changes survive language/tab/search/advanced toggles and minute refresh.
- Hidden chart containers resize when shown; history works with no report.

### Task 1: Operational state and navigation

**Files:** panel_state.js (new), panel.html, main.py, charts.py;
tests/test_addon_panel_state.cjs, test_addon_day_charts.py, test_addon_panel_assets.py.
**Interfaces:** PanelState.summary(report, status, nowMs) returns enum states and
finite-or-null numbers. /api/status adds configured apply_mode/replan_minutes and
grid observation time. /api/day_charts adds load_actual_kw and load_forecast_kw.

- [x] Write failing tests for missing/stale reports, mode changes, partial writes,
  grid expiry, null autonomy, reserve shortfall and forecast/actual load alignment.
- [x] Run tests and observe missing feature failures.
- [x] Implement pure state helper, additive data, tabs, four cards and decision
  summary; preserve existing renderer IDs in collapsed/detail or history sections.
- [x] Add settings search/basic-advanced, expandable hints and sticky save/cancel.
  Exercise dirty edits, filtering and navigation in the real browser.
- [x] Verify targeted tests/lint and the add-on suite; document existing failures.
- [x] Browser-check RU/EN, mobile and desktop, empty/stale/partial-write states and
  chart resizing. Review independently, fix material findings and commit 0.19.0.

## Execution record

Pre-flight: state helper enums feed translated panel rendering; chart load fields
feed existing timestamp slots. No interface conflicts.
Baseline correction: current HEAD contains the user's intermediate 0.18.2 fix
(63269fd7), which updated the hardware tests. The earlier five hardware failures
therefore no longer apply. Current full add-on run: **382 passed, 1 failed**,
`test_addon_pipeline_run.py::test_live_series_holds_a_reserve_with_margin`.
Reproduced the same failure from a clean archive of current HEAD: its
wall-clock-dependent requested floor is 0.10. No pipeline/controller changes.

Verification:

- Targeted Python suite: 38 passed (charts, assets, settings layout and options).
- Node built-in tests: 13 passed, including current/old modes, stale grid/report,
  missing values, HA string SOC, failed setup fetch preserving edits, cleared
  settings acknowledged and historical write failures remaining visible.
- Browser: RU/EN, 390px mobile without horizontal page overflow, all 57 editable
  fields / 45 option fields in advanced mode, combined entity+scalar+load save,
  reload persistence, cancel, entity reset, language/tab/search edit retention,
  first run with no report, stale report and partial-write alerts verified.
- Chart controls and keyboard navigation verified; revealed canvases resized to
  906px width. Browser console: no errors in the final preview.
- Independent review found three issues (failed refresh, cleared-key receipt,
  historical write warning); all reproduced and fixed, reviewer confirmed no
  remaining blocking findings.
- Ruff, JavaScript syntax checks and git diff --check passed.
- Docker smoke unavailable: Docker daemon is not running. The actual Handler and
  settings API were exercised through an isolated native local HTTP preview.
- Screenshot: /private/tmp/emhass-panel-0.19.0.jpg (synthetic data).
- No live Home Assistant changes, inverter writes, push or deployment.

Ruling: retain calendar-day Today / Today and tomorrow chart controls rather than
claim exact elapsed 24/48 hours from the existing calendar-day data. The forecast
cards still show the report's next 24-hour horizon. This avoids misleading DST
and partial-day labels without changing forecast generation.
