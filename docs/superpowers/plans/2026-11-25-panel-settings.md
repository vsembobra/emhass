# Complete Panel Settings Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; follow TDD.

**Goal:** Every add-on setting editable on the shadow panel with clear grouping.
**Architecture:** Extend the existing metadata editor with a flexible-load list;
reuse shared Supervisor saves and attach group metadata to entity checks.
**Tech Stack:** Existing Python/pytest and vanilla JavaScript/CSS.
**Spec:** `docs/superpowers/specs/2026-11-25-panel-settings-design.md`

## Global Constraints

- Preserve the 0.18.1 bottom-of-page settings layout, styles and RU/EN support.
- All 57 schema fields appear once across editable settings and entity bindings.
- No live configuration, inverter writes or deployment.

## Review Focus

- An empty flexible-load list must clear saved loads.
- Invalid power, duration or time windows must not reach the Supervisor.
- Unsaved list edits must survive periodic rendering like scalar edits do.
- Dynamic strings must be escaped and RU/EN labels correctly associated.
- Unknown future schema settings must fail the coverage test instead of disappearing.

### Tasks

- [x] Add failing coverage/grouping and flexible-list save/validation tests.
- [x] Implement full metadata coverage, list validation and meaningful groups.
- [x] Add list UI, grouped entity rendering and associated labels; preserve layout.
- [x] Verify real browser rendering, add/edit/delete/save and translated controls.
- [x] Run add-on regressions and lint, perform independent review, bump 0.18.3.

## Verification results

- Targeted settings/options/discovery/weather tests: 64 passed.
- Full add-on suite: 371 passed, 5 failed. All five failures are unchanged
  `test_addon_hardware.py::test_effective_options_*` expectations of two return
  values, while 0.18.1 added English notes as a third return value. Reproduced
  the same five failures from a clean HEAD archive, without this change.
- Ruff on changed Python files and `git diff --check`: passed.
- Independent code review: no actionable findings.
- Real local browser with synthetic configuration: all 57 editable fields,
  weather model save/reload, flexible-load add/edit/save/remove (including an
  empty list), logical entity groups, Russian and English controls verified.
- Screenshot: `/private/tmp/emhass-panel-settings-0.18.2.jpg`.
- No live Home Assistant settings or inverter writes; no deployment.
