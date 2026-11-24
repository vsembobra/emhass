# Complete grouped shadow-panel settings

User asks for every add-on configuration setting on the shadow panel and sensible
grouping. Audit: 57 schema keys, 44 existing option fields, 12 entity bindings in metadata (only 11 rendered
when unconfigured), and the missing deferrable_loads list. The empty grid-voltage
field must also remain visible. Preserve the user's latest
0.18.1 layout: settings cards remain below the report, with existing styling and
RU/EN support. No changes to optimizer, actuation or discovery behavior.

Use the existing metadata-driven editor and shared Supervisor configuration.
Put weather first in its own group; then site, panels, battery (including BMS
calibration), inverter controls, tariff/planning, reserve/priority, flexible loads,
learning/history, and add-on operation. Group the separate entity editor into
house/solar, battery, grid and inverter controls. Each editable parameter appears
once across these two editors.

Add a structured flexible-load editor (name, kW, hours, optional time window,
add/remove) rather than a raw JSON box or a link to another page. Validate edited
lists at the existing settings boundary and save through the same Supervisor
path. Existing read-only TOU entity checks remain read-only.

Tests must prove complete schema coverage, unique fields, grouping, list editing
and validation, shared settings persistence, translated rendering and unchanged
discovery/battery behavior. Bump the add-on patch version to 0.18.3. Verify the
actual page in a local browser with synthetic configuration, without live writes.
