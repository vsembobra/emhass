# Manual Discovery Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Explicit, bounded search and configuration of a single inverter.

**Architecture:** Keep registry parsing in discovery.py, add a small background
search worker, integrate manual preview endpoints and panel controls. Decouple
effective options and startup from discovery.

**Tech Stack:** Existing Python threading/asyncio/urllib, panel JavaScript, pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-manual-discovery-design.md`

## Global Constraints

- No automatic sensor filling or cache-driven changes to saved settings.
- No new dependencies, EMHASS core edits or deployment.
- Two inverters require an explicit candidate selection.

## Review Focus

- Several devices in one integration entry must not merge bindings.
- A failed lookup must clear busy state and permit retry.
- Settings stay usable without configured sensors or reachable Home Assistant.
- Existing manual settings survive search and normal startup.
- Mandatory runtime modules must be included in the Git change.

### Task 1: Manual options and coherent discovery candidates

- [x] Write failing tests for empty defaults, ignored cache and two-device parsing.
- [x] Remove runtime auto filling/compaction; parse per-device candidate previews.
- [x] Verify preserved manual values and explicit-only configuration behavior.

### Task 2: Bounded background search and waiting for configuration

- [x] Add failing tests for startup without bindings, timed-out/malformed search,
  concurrency/retry, and batched entity validation.
- [x] Add discovery_search.py; wire bounded HA reads and GET/POST status endpoints.
- [x] Gate bootstrap/training/planning on required bindings and verify busy cleanup.

### Task 3: Preview UI and regression verification

- [x] Add bilingual search/select/preview/save controls; no automatic save.
- [x] Update configuration hints and README to describe manual behavior.
- [x] Verify endpoint behavior and render/save flow; run add-on regressions and lint.
- [x] Request independent review and fix confirmed findings with reproducers.
- [x] Stage missing runtime modules without committing or publishing.


## Execution record

- Removed implicit discovery/compaction and legacy-cache application. Explicitly
  configured values are retained; fresh bindings are empty.
- Registry parsing now separates devices within a shared integration entry.
- Search uses a background worker with a 30-second total async I/O deadline;
  the panel offers preview, candidate choice and explicit save.
- Independent review found partial candidate merges could keep another inverter's
  controls. A failing reproducer now passes: complete device replacement is
  previewed and applied by an ID-only server endpoint, clearing old bindings.
- Independent review also caught absent critical load invalidating all load
  history. A failing reproducer now passes: critical load is required.
- Backend validation reads one HA state list; blank control templates/floor
  cannot produce writes to empty entity IDs.
- Final regression: 361 add-on tests passed (including real localhost HTTP
  preview/save and two-device tests). Ruff and diff whitespace checks passed.
- The real panel JavaScript was executed in Node: unconfigured panel accessible,
  two-device choice mandatory, translated preview, and save only on explicit action.
- Runtime Python modules exported from the Git index import successfully.
  Missing weather_models.py and new discovery_search.py are staged with tests.
- No commit, push, deployment or inverter actions were performed.
