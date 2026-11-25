# Simple battery panel

Approved by the user's request to implement the preceding three-tab proposal.

## Intent and approach

Make the daily decision visible: grid state, battery autonomy, protected reserve,
solar forecast and what the controller actually did. Use three tabs (Now, History,
Settings) instead of the existing long page. Keeping everything on one page with
more accordions saves navigation but still mixes operational and diagnostic data;
a separate application would duplicate the existing configuration/persistence.
Keep vanilla JS, bundled ECharts, dark colors and RU/EN. No new dependencies.

## Now

Show current configured mode separately from the last report's confirmed writes.
Show grid state only while its observation is recent (30 minutes); unknown is a
real state. Report timestamp and overdue recalculation remain visible. A plan is
overdue after configured replan_minutes plus 15 minutes of scheduling allowance.
SOC and autonomy are explicitly a calculation-time snapshot, never a live claim.
Available delivered energy is the existing autonomy estimate times critical load.

Four cards: battery, autonomy (with assumed critical kW), reserve (with reason),
solar next 24 h (with model-estimated curtailment when available). A short decision
summary explains reserve, night charging and calibration from existing fields.
Outages, reserve shortfall, stale plans and write failures must not be hidden by
tab choice. No guarantee that an unknown future outage will be covered.

Keep one main chart card, default today; allow today plus tomorrow, labelled as
calendar days rather than claiming exactly 24/48 hours across DST. Add measured
and forecast consumption to the existing aligned power/SOC chart. Technical
floor/optimizer/TOU details stay available in a collapsed Plan details section.

## History and settings

History holds reports, daily totals, model metrics, learning and maintenance
actions. Keep measured and estimated curtailment distinguishable.
All 57 settings remain editable, grouped with basic/advanced display and search.
Hints become expandable. Dirty scalar, flexible-load and entity changes survive
refresh, tab/language changes and filters. A sticky save/cancel bar reflects both
editors. Cancel restores the last fetched values locally, with no write.
Keyboard tab navigation and mobile layouts are required; no external fonts/assets.

## Constraints

No changes to scheduling, optimization, inverter actuation or core EMHASS.
Additive status/chart data only. Offline assets must be packaged and served.
Work in the existing checkout on codex/panel-navigation. Do not touch .tmp_friend.
Bump to 0.19.0 for navigation changes; do not publish or deploy.
