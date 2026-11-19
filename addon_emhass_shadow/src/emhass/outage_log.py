"""Store measured and scheduled grid outages in a local SQLite database."""

import argparse
import csv
import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

EventSource = Literal["measured", "scheduled"]


@dataclass(frozen=True)
class OutageEvent:
    """A UTC outage interval with provenance and a flag for uncertain measurement boundaries."""

    start: datetime
    end: datetime | None
    source: EventSource
    interrupted: bool = False


@contextmanager
def _write_transaction(database: Path) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS outage_observations (
                site TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT,
                voltage REAL,
                effective_status TEXT NOT NULL,
                PRIMARY KEY (site, timestamp)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS outage_events (
                site TEXT NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                source TEXT NOT NULL,
                detected INTEGER NOT NULL DEFAULT 0,
                interrupted INTEGER NOT NULL DEFAULT 0,
                cancelled INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (site, start, end, source, detected)
            )
        """)
        connection.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS open_detected_outage
            ON outage_events (site) WHERE detected = 1 AND end = ''
        """)
        yield connection


def _site(site: str) -> str:
    site = site.strip()
    if not site:
        raise ValueError("site must be nonempty")
    return site


def _utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("timestamps must include a UTC offset")
    return value.astimezone(UTC)


def _parse_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid ISO 8601 timestamp {value!r}") from exc
    return _utc(timestamp)


def _timestamp_key(timestamp: datetime) -> str:
    return timestamp.isoformat(timespec="microseconds")


def _csv_rows(
    path: Path, required: set[str], *, one_of: set[str] | None = None
) -> Iterator[tuple[int, dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        fields = reader.fieldnames or []
        columns = set(fields)
        if (
            len(fields) != len(columns)
            or not required <= columns <= required | (one_of or set())
            or (one_of and not columns & one_of)
        ):
            expected = ", ".join(sorted(required))
            if one_of:
                expected += f" and at least one of {' or '.join(sorted(one_of))}"
            raise ValueError(f"{path}: expected CSV columns {expected}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{reader.line_num}: wrong number of CSV fields")
            yield reader.line_num, {key: value.strip() for key, value in row.items()}


def import_event_csv(database: Path, site: str, csv_path: Path) -> None:
    """Import an atomic batch of explicit events with start, end, and source columns."""
    site = _site(site)
    with _write_transaction(database) as connection:
        for line, row in _csv_rows(csv_path, {"start", "end", "source"}):
            try:
                source = row["source"]
                if source not in {"measured", "scheduled"}:
                    raise ValueError("source must be measured or scheduled")
                start = _parse_timestamp(row["start"])
                end = _parse_timestamp(row["end"]) if row["end"] else None
                if end is None and source == "scheduled":
                    raise ValueError("scheduled events require an end")
                if end is not None and end <= start:
                    raise ValueError("end must follow start")
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{csv_path}:{line}: {exc}") from exc

            connection.execute(
                "INSERT INTO outage_events (site, start, end, source) VALUES (?, ?, ?, ?) "
                "ON CONFLICT DO NOTHING",
                (site, _timestamp_key(start), _timestamp_key(end) if end else "", source),
            )


def cancel_scheduled_event(database: Path, site: str, start: datetime, end: datetime) -> None:
    """Cancel one exact scheduled interval and remember its cancellation on later imports."""
    site = _site(site)
    start, end = _utc(start), _utc(end)
    if end <= start:
        raise ValueError("end must follow start")
    with _write_transaction(database) as connection:
        connection.execute(
            "INSERT INTO outage_events (site, start, end, source, cancelled) "
            "VALUES (?, ?, ?, 'scheduled', 1) ON CONFLICT DO UPDATE SET cancelled = 1",
            (site, _timestamp_key(start), _timestamp_key(end)),
        )


def _effective_status(
    status: str | None,
    voltage: float | None,
    previous: str,
    down_voltage: float | None,
    restore_voltage: float | None,
) -> str:
    if status is not None:
        return status
    if voltage is None:
        return "unknown"
    if down_voltage is None or restore_voltage is None:
        raise ValueError("a voltage threshold is required for voltage-only observations")
    if voltage <= down_voltage:
        return "down"
    if voltage >= restore_voltage:
        return "up"
    return previous


def _close_detection(
    connection: sqlite3.Connection, site: str, end: datetime, *, interrupted: bool
) -> None:
    connection.execute(
        "UPDATE outage_events SET end = ?, interrupted = max(interrupted, ?) "
        "WHERE site = ? AND detected = 1 AND end = ''",
        (_timestamp_key(end), interrupted, site),
    )


def record_observation_csv(
    database: Path,
    site: str,
    csv_path: Path,
    *,
    max_gap: timedelta,
    down_voltage: float | None = None,
    restore_voltage: float | None = None,
) -> None:
    """Record chronological voltage or grid-status readings and update measured outages."""
    site = _site(site)
    if max_gap <= timedelta(0):
        raise ValueError("max_gap must be positive")
    if down_voltage is None:
        if restore_voltage is not None:
            raise ValueError("setting restore_voltage requires down_voltage")
    else:
        if not math.isfinite(down_voltage) or down_voltage < 0:
            raise ValueError("down_voltage must be finite and non-negative")
        if restore_voltage is None:
            restore_voltage = down_voltage
        if not math.isfinite(restore_voltage) or restore_voltage < down_voltage:
            raise ValueError("restore_voltage must be finite and at least down_voltage")

    with _write_transaction(database) as connection:
        latest = connection.execute(
            "SELECT timestamp, effective_status FROM outage_observations WHERE site = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (site,),
        ).fetchone()
        last_timestamp, previous = (
            (datetime.fromisoformat(latest[0]), latest[1])
            if latest
            else (
                None,
                "unknown",
            )
        )
        previous_row = None
        for line, row in _csv_rows(csv_path, {"timestamp"}, one_of={"status", "voltage"}):
            try:
                timestamp = _parse_timestamp(row["timestamp"])
                if previous_row is not None and timestamp <= previous_row:
                    raise ValueError("observation timestamps must be strictly increasing")
                previous_row = timestamp
                timestamp_key = _timestamp_key(timestamp)
                status = row.get("status", "").lower() or None
                if status is not None and status not in {"up", "down", "unknown"}:
                    raise ValueError("status must be up, down, or unknown")
                voltage = float(row["voltage"]) if row.get("voltage") else None
                if voltage is not None and (not math.isfinite(voltage) or voltage < 0):
                    raise ValueError("voltage must be finite and non-negative")

                if last_timestamp is not None and timestamp <= last_timestamp:
                    recorded = connection.execute(
                        "SELECT status, voltage FROM outage_observations "
                        "WHERE site = ? AND timestamp = ?",
                        (site, timestamp_key),
                    ).fetchone()
                    if recorded is None:
                        raise ValueError("observation is older than the latest recorded reading")
                    if recorded != (status, voltage):
                        raise ValueError("conflicting observation at an already recorded timestamp")
                    continue

                if last_timestamp is not None and timestamp - last_timestamp > max_gap:
                    if previous == "down":
                        _close_detection(
                            connection, site, last_timestamp + max_gap, interrupted=True
                        )
                    previous = "unknown"
                current = _effective_status(
                    status, voltage, previous, down_voltage, restore_voltage
                )
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{csv_path}:{line}: {exc}") from exc

            if previous == "down" and current != "down":
                _close_detection(connection, site, timestamp, interrupted=current == "unknown")
            elif previous != "down" and current == "down":
                connection.execute(
                    "INSERT INTO outage_events (site, start, end, source, detected, interrupted) "
                    "VALUES (?, ?, '', 'measured', 1, ?)",
                    (site, timestamp_key, previous == "unknown"),
                )
            connection.execute(
                "INSERT INTO outage_observations VALUES (?, ?, ?, ?, ?)",
                (site, timestamp_key, status, voltage, current),
            )
            last_timestamp, previous = timestamp, current


def read_events(database: Path, site: str) -> list[OutageEvent]:
    """Read one site's events, coalescing overlaps among completed measured intervals."""
    site = _site(site)
    if not database.exists():
        return []
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        if (
            connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'outage_events'"
            ).fetchone()
            is None
        ):
            return []
        # Completed intervals can resolve explicitly imported opens. The detector's
        # open event only changes when observations arrive.
        rows = connection.execute(
            """
            SELECT start, end, source, interrupted FROM outage_events AS entry
            WHERE site = ? AND cancelled = 0 AND (
                detected = 1 OR end <> '' OR NOT EXISTS (
                    SELECT 1 FROM outage_events AS completed
                    WHERE completed.site = entry.site AND completed.start = entry.start
                    AND completed.source = 'measured' AND completed.end <> ''
                    AND completed.interrupted = 0
                )
            )
            """,
            (site,),
        )
        records = {
            OutageEvent(
                datetime.fromisoformat(start),
                datetime.fromisoformat(end) if end else None,
                source,
                bool(interrupted),
            )
            for start, end, source, interrupted in rows
        }

    measured = []
    other = []
    for event in records:
        if event.source == "measured" and event.end is not None and not event.interrupted:
            measured.append((event.start, event.end))
        else:
            other.append(event)

    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(measured):
        if merged and start < merged[-1][1]:
            earlier_start, earlier_end = merged[-1]
            merged[-1] = (earlier_start, max(earlier_end, end))
        else:
            merged.append((start, end))
    events = [OutageEvent(start, end, "measured") for start, end in merged] + other
    return sorted(
        events,
        key=lambda event: (event.start, event.source, event.end or event.start, event.interrupted),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--site", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    observe = commands.add_parser("observe", help="record readings from a CSV file")
    observe.add_argument("csv", type=Path)
    observe.add_argument("--max-gap-seconds", type=float, required=True)
    observe.add_argument("--down-voltage", type=float)
    observe.add_argument("--restore-voltage", type=float)
    event_import = commands.add_parser(
        "import-events", help="import explicit events from a CSV file"
    )
    event_import.add_argument("csv", type=Path)
    cancel = commands.add_parser("cancel-schedule", help="cancel an exact scheduled interval")
    cancel.add_argument("start")
    cancel.add_argument("end")
    commands.add_parser("list", help="print the site's events as JSON")
    args = parser.parse_args()

    try:
        match args.command:
            case "observe":
                record_observation_csv(
                    args.database,
                    args.site,
                    args.csv,
                    max_gap=timedelta(seconds=args.max_gap_seconds),
                    down_voltage=args.down_voltage,
                    restore_voltage=args.restore_voltage,
                )
            case "import-events":
                import_event_csv(args.database, args.site, args.csv)
            case "cancel-schedule":
                cancel_scheduled_event(
                    args.database,
                    args.site,
                    _parse_timestamp(args.start),
                    _parse_timestamp(args.end),
                )
            case "list":
                print(
                    json.dumps(
                        [
                            {
                                "start": event.start.isoformat(),
                                "end": event.end.isoformat() if event.end else None,
                                "source": event.source,
                                "interrupted": event.interrupted,
                            }
                            for event in read_events(args.database, args.site)
                        ],
                        indent=2,
                    )
                )
    except (OSError, ValueError, OverflowError, csv.Error, sqlite3.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
