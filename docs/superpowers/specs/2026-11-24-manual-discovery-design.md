# Explicit sensor discovery

User request: stop automatic sensor filling, provide an automatic-search button,
and keep the add-on responsive with two inverters and failed discovery.

## Approach

Use manual configuration plus an explicit asynchronous search preview. Automatic
background discovery and cache-driven binding changes are removed; an on/off
toggle for background discovery was rejected because it leaves the reported
ambiguity and startup delay in place. No multiple-inverter optimizer is added.

New installations have empty sensor/control bindings. Existing explicitly saved
bindings stay as saved; legacy discovery cache no longer supplies missing values.
Until required load, critical-load, PV, SOC and grid-status bindings are supplied, planning and
training wait for configuration and the panel remains usable.

One search runs independently from planning. HA registry, state and config reads
have a total bounded budget. Failure becomes a status message, clears the busy
flag and allows retry. No search result changes saved options. Registry entries
are grouped by integration entry and device; candidate previews never mix two
inverters. With several candidates, the user selects one, reviews its values and
explicitly saves them. Both languages explain the workflow.
The preview explicitly includes empty bindings; saving replaces all device
bindings as a set, clearing undiscovered old TOU/floor/grid-charge controls.
Non-device settings such as the tariff remain unchanged.

Entity validation uses one bounded HA state-list request instead of a serial
request per missing sensor. Background planning no longer compacts settings.

## Verification

Tests cover no implicit filling/network search, legacy-cache isolation, preserving
manual options, fresh-install waiting, per-device candidates, ambiguous selection,
failed/concurrent/timed-out searches, retry, one-request validation, API handlers,
UI preview/save flow, and add-on regressions. No new dependencies, core EMHASS
changes or deployment. Include the previously untracked weather_models.py in the
Git index so a subsequent commit cannot omit this mandatory runtime dependency.
