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
