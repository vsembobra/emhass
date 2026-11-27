# Compact Storage Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline.

**Goal:** Store training and reporting data compactly, without redundant mirrors
or loss of learning history.
**Architecture:** New data_store.py owns lossless gzip, atomic writes, legacy
reads, migration and known-file enumeration. Wire existing app readers/writers
through it; preserve JSON content and response interfaces.
**Tech Stack:** Python stdlib gzip/tempfile/os/threading, existing pandas/pytest.
**Spec:** docs/superpowers/specs/2026-11-26-compact-storage-design.md

## Global Constraints

No dependencies, core/controller changes, training-history truncation, live HA
cleanup, deployment or edits to .tmp_friend. User confirmed shared files have no
external consumers; migrate known application-owned exports only.

## Review Focus

- Failed writes must leave the last good data readable and retryable.
- Legacy plus compressed files must not duplicate observations/reports.
- All readers, bootstrap presence checks and weather retention must understand gzip.
- Per-source histories and outage/censoring fields must remain unchanged.
- Only proven redundant, application-owned exports may be removed.

### Task 1: Storage module and integration

**Files:** new app/data_store.py and tests/test_addon_data_store.py; existing app
pipeline.py, learning.py, retrain.py, daily.py, charts.py, main.py, run.sh.
**Interfaces:** exists/read_text/read_json/read_frame/write_text/write_json,
append_jsonl/json_paths/remove/compact_existing. Paths are logical legacy names;
physical compressed files append .gz. json_paths returns logical .json paths.

- [x] Write failing tests for legacy/gzip equivalence, atomic failure, concurrent
  readers, duplicate logical names, append preservation and migration conflicts.
- [x] Implement storage module; watch tests turn green.
- [x] Integrate all training/report/weather readers and writers, presence checks
  and retention; verify existing fixture semantics with updated format reads.
- [x] Resolve share-consumer answer; stop duplicate writes and migrate only safe
  mirrors with verification, preserving unique files and manual seeds.
- [x] Verify training, panel APIs, offline weather, migration/restart and measured
  compression; run add-on regressions and independent review; document version.

## Execution record

Starting from d8f25bd1 in the current checkout; branch codex/compact-storage.
Baseline known: test_live_series_holds_a_reserve_with_margin is wall-clock
dependent and failed on clean HEAD in the previous task; recheck current run.

Compression-only phase (committed as 6cb38ea3): gzip storage and all internal
consumers integrated. At that point share mirror writes remained unchanged
pending the external-consumer answer, and the version was not yet bumped.

Evidence so far:
- Both actual LightGBM load/PV bundles train, save, reload and predict from
  migrated audit/weather history. All timestamps and quality masks retained.
- Synthetic 8760-hour history: 1,079,458 bytes -> 64,603 bytes (94.0% smaller).
  This is a synthetic measurement, not a claim about the live site's ratio.
- Full add-on suite on the current code: **400 passed** (176.17 s). JS tests: 13 passed. Targeted API/seed/chart/
  source/backfill tests: 38 passed; three added startup failure cases also pass.
- Independent review reproduced seed-format precedence and interrupted temporary
  duplication. Fixed gzip-first import and exact-byte duplicate cleanup. Unknown,
  unique or unreadable temp data is preserved. Added malformed-DEFLATE and startup
  error coverage; optional compaction cannot prevent the panel from starting.
- Current live /share/emhass_shadow: about 1.8 MB, including about 1.6 MB of manual
  seed data. Read-only SSH search found no YAML/Python/shell references from /config.
  This does not establish whether scripts outside HA use exported paths.
- No live files changed. .tmp_friend and manual seed data untouched.

Final review follow-up: gzip weather seeds now take precedence over legacy
snapshots. Verified-owned interrupted-write duplicates are removed only on exact
decompressed-byte equality; unique, manual and damaged files remain intact.
Malformed DEFLATE (zlib.error) is covered. Compaction read errors (including
truncated/corrupt gzip) are caught by startup so the panel still opens. Migration
is limited to supported weather model roots and skips symlinked directories.

Compression phase verification: 400 add-on tests and 13 JS tests passed; Ruff,
shell syntax and diff checks passed. Export deduplication followed only after the
user confirmed no external consumers. No live cleanup, push or deployment.

## Completion tasks (2026-11-27)

User confirmed no external shared-file consumers. Continue in the same branch.
- [x] Add storage_migration.py with verified export import/dedup/archive, missing
  seed-hour import and conflict preservation; tests for failure and retry.
- [x] Stop the three mirror writers, migrate before scheduler/watchers start,
  include compressed retrain state and legacy outage JSONL in existing readers.
- [x] Rename visible config/repository/panel/sensor/notification labels to Solar
  Reserve, preserving every technical identifier and shadow-plan detail heading.
- [x] Verify migrations, training, APIs, compatibility IDs, real browser RU/EN,
  add-on regressions, independent review; release 0.20.0 locally.

## Final verification (2026-11-28)

- Final add-on regression suite: **420 passed** (60.48 s), including real
  LightGBM training/reload, local HTTP APIs and full report restoration.
- Panel JavaScript: **13 passed**; JS syntax, Ruff, shell syntax and diff checks
  passed. Configuration remains 57 keys; slug, sensor and notification IDs and
  repository URL are unchanged. Release version is **0.20.0**.
- Independent review found unsafe activation of incomplete legacy metadata and
  a symlinked-parent traversal. Reproducing tests added; both fixed and reviewed
  again, with no remaining migration blockers identified.
- Browser smoke test with synthetic data confirmed Solar Reserve in RU/EN,
  the weather-model selector and the separate full shadow-plan heading in Plan
  details. Screenshot: /private/tmp/solar-reserve-panel.jpg.
- Shared copies are no longer written. Missing seed hours are imported by UTC;
  duplicates are removed only after verification and unique versions archived
  losslessly. Failed transfers retain the source for retry.
- Local code only: no live data cleanup, inverter writes, push or deployment.
  Existing .tmp_friend remains untouched.
