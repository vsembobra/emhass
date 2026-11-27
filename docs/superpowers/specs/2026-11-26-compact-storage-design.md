# Compact training storage

User asks to remove redundant copies, reduce disk use, preserve learning and
avoid regressions. Keep all training observations and their quality/censoring
fields, precision, UTC timestamps and per-weather-model separation.

Use lossless gzip and atomic replacement in a single authoritative /data store.
Keep reading legacy plain JSON/JSONL; migrate by publishing and syncing a readable
compressed file before removing the original. Do not truncate training history or
delete the last-good model. Preserve existing forecast (21 days), metrics (60
records) and offline-weather (7 days plus newest) policies. Compress reports
without reducing their history or API shape. Stop automatic share mirrors only
after clarifying external consumers; retain unknown/unique user data.

Alternatives: SQLite or columnar storage would require a larger migration and
schema/reader rewrite; deleting old training rows saves space by losing seasons.
Gzip reuses the existing JSON schema with no new dependencies. Atomic complete
writes are preferred over append-only gzip members to survive interrupted writes.

Prove old/new reads, interrupted migration recovery, model training, report APIs,
weather fallback, timestamp and quality equivalence, scope isolation, no duplicate
history in the UI and a measured compression ratio. No core EMHASS changes, live
HA cleanup or deployment. Existing .tmp_friend remains untouched.

## Approved completion (2026-11-27)

User confirmed all shared files are consumed only by the add-on. Remove automatic
mirror writes. Migrate known shared exports before startup compaction: restore
missing primary files, remove verified duplicates, archive differing unique
versions compactly under /data/legacy_imports. Import missing seed audit hours;
existing observations win and conflicting seed rows are retained separately.
Delete a source only after reading back its saved data. For SQLite seed, remove
only if all source tables/rows are covered by the active database; otherwise
retain its unique snapshot compressed. Unknown files and symlinks remain.

Rename the visible product to Solar Reserve (Солнечный резерв), the recommended
name followed by the user's 'continue'. Preserve slug emhass_shadow, sensor and
notification IDs, repository URL, data paths and all 57 schema keys. Preserve the
restored full shadow-plan report title as a functional section, separate from the
product brand. Bump version to 0.20.0. No live migration/push/deployment.
