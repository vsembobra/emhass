# Grid outage log

Use the standalone outage logger to turn grid voltage or status readings into measured outage
events. It also accepts scheduled windows and measured events from another source. Choose a local
SQLite file in your data directory and a stable site identifier. The file's parent directory must
already exist. Several sites can share a database, and each command operates on the named site.

## Record grid observations

Create a UTF-8 CSV with a `timestamp` column and either `status`, `voltage`, or both. Use ISO 8601
timestamps with an explicit UTC offset. Rows in each file must be strictly increasing by UTC
instant, including during daylight saving time changes. Voltage is a finite, non-negative value in
volts. Status accepts `up`, `down`, and `unknown` (case-insensitive). An explicit status takes
precedence over voltage. Leave both readings blank to record an unknown state.

```csv
timestamp,status,voltage
2026-11-16T03:00:00+03:00,,228
2026-11-16T03:01:00+03:00,,0
2026-11-16T03:02:00+03:00,down,
2026-11-16T03:03:00+03:00,,226
```

```bash
python -m emhass.outage_log --database data/outages.sqlite3 --site home observe readings.csv \
  --max-gap-seconds 60 --down-voltage 190 --restore-voltage 205
```

Set `--max-gap-seconds` to the maximum acceptable interval between readings for your sampling
cadence. For voltage-only readings, the state is down at or below `--down-voltage`, and up at or
above `--restore-voltage`. Readings between different thresholds retain the previous known state.
With only `--down-voltage`, readings above that threshold are up. Choose thresholds for the site's
grid voltage and filter or debounce sensor readings upstream when needed. A status-only feed needs
only `--max-gap-seconds`.

Record the next CSV with the same site and database to continue an open outage. An exact reading
replay is harmless. A conflicting reading at an existing timestamp or an unseen reading older
than the latest stored one rejects the whole batch. Use explicit event import for older outage
history.

An outage begins at the first down reading and normally ends at the first up reading. An unknown
reading or a gap longer than the configured maximum interrupts the outage. The logger records the
end at the unknown reading or when the last reading became stale, whichever comes first. A down
reading after an unknown period also begins an event with an uncertain start. These events have
`interrupted: true` and should be excluded from duration statistics. An open record has `end: null`
until another reading resolves it. For current grid availability, check the freshness of the
underlying telemetry.

## Import explicit events

Supply `start,end,source` columns, with the same timestamp format. An end must fall after its start.
A measured event may have an empty end while it remains open. To close an explicitly imported
outage, import its completed interval with the same start. A scheduled event requires both
boundaries. Event rows may arrive in any order, and repeated imports are idempotent.

```csv
start,end,source
2026-11-10T07:00:00Z,2026-11-10T08:45:00Z,measured
2026-11-21T06:00:00Z,2026-11-21T08:00:00Z,scheduled
```

```bash
python -m emhass.outage_log --database data/outages.sqlite3 --site home import-events events.csv
python -m emhass.outage_log --database data/outages.sqlite3 --site home list
```

`list` emits JSON with UTC `start` and `end`, `source` (`measured` or `scheduled`), and
`interrupted`. Completed, uninterrupted measured intervals are coalesced when they overlap.
Scheduled windows and interrupted measurements retain their individual boundaries.

Cancel one scheduled window using both of its original timestamps. The cancellation belongs to
that site and persists if a later CSV contains that same window.

```bash
python -m emhass.outage_log --database data/outages.sqlite3 --site home cancel-schedule \
  2026-11-21T06:00:00Z 2026-11-21T08:00:00Z
```
