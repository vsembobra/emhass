import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from emhass.outage_log import (
    OutageEvent,
    cancel_scheduled_event,
    import_event_csv,
    read_events,
    record_observation_csv,
)


def at(minute):
    return datetime(2026, 11, 16, 0, minute, tzinfo=UTC)


def write_csv(directory, name, content):
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def test_observations_restore_open_outage_after_restart(tmp_path):
    database = tmp_path / "outages.sqlite3"
    first = write_csv(
        tmp_path,
        "first.csv",
        "timestamp,status,voltage\n"
        "2026-11-16T00:00:00Z,,230\n"
        "2026-11-16T00:02:00Z,,195\n"
        "2026-11-16T00:04:00Z,down,230\n",
    )
    second = write_csv(
        tmp_path,
        "second.csv",
        "timestamp,status,voltage\n"
        "2026-11-16T03:04:00+03:00,DOWN,230.0\n"
        "2026-11-16T00:06:00Z,,195\n"
        "2026-11-16T00:08:00Z,,200\n",
    )
    for path in (first, first):
        record_observation_csv(
            database,
            "site",
            path,
            max_gap=timedelta(minutes=2),
            down_voltage=190,
            restore_voltage=200,
        )
    assert read_events(database, "site") == [OutageEvent(at(4), None, "measured")]

    for path in (second, first, second):
        record_observation_csv(
            database,
            "site",
            path,
            max_gap=timedelta(minutes=2),
            down_voltage=190,
            restore_voltage=200,
        )
    assert read_events(database, "site") == [OutageEvent(at(4), at(8), "measured")]
    assert read_events(database, "other site") == []


def test_unknown_and_stale_observations_censor_durations(tmp_path):
    database = tmp_path / "outages.sqlite3"
    observations = write_csv(
        tmp_path,
        "observations.csv",
        "timestamp,status\n"
        "2026-11-16T00:00:00Z,down\n"
        "2026-11-16T00:02:00Z,up\n"
        "2026-11-16T00:04:00Z,down\n"
        "2026-11-16T00:06:00Z,unknown\n"
        "2026-11-16T00:08:00Z,down\n"
        "2026-11-16T00:16:00Z,up\n"
        "2026-11-16T00:18:00Z,down\n"
        "2026-11-16T00:20:00Z,up\n"
        "2026-11-16T00:25:00Z,down\n",
    )
    record_observation_csv(database, "site", observations, max_gap=timedelta(minutes=3))
    assert read_events(database, "site") == [
        OutageEvent(at(0), at(2), "measured", interrupted=True),
        OutageEvent(at(4), at(6), "measured", interrupted=True),
        OutageEvent(at(8), at(11), "measured", interrupted=True),
        OutageEvent(at(18), at(20), "measured"),
        OutageEvent(at(25), None, "measured", interrupted=True),
    ]


def test_observations_are_ordered_by_instant_during_repeated_hour(tmp_path):
    database = tmp_path / "outages.sqlite3"
    observations = write_csv(
        tmp_path,
        "observations.csv",
        "timestamp,status\n"
        "2026-11-01T01:28:00-04:00,up\n"
        "2026-11-01T01:29:00-04:00,down\n"
        "2026-11-01T01:28:00-05:00,down\n"
        "2026-11-01T01:29:00-05:00,up\n",
    )
    record_observation_csv(database, "site", observations, max_gap=timedelta(hours=1))
    assert read_events(database, "site") == [
        OutageEvent(
            datetime(2026, 11, 1, 5, 29, tzinfo=UTC),
            datetime(2026, 11, 1, 6, 29, tzinfo=UTC),
            "measured",
        ),
    ]


def test_observation_csv_requires_a_reading_column(tmp_path):
    database = tmp_path / "outages.sqlite3"
    observations = write_csv(tmp_path, "readings.csv", "timestamp\n2026-11-16T00:00:00Z\n")
    with pytest.raises(ValueError, match="status or voltage"):
        record_observation_csv(database, "site", observations, max_gap=timedelta(minutes=2))


def test_invalid_observation_batch_does_not_leave_partial_state(tmp_path):
    database = tmp_path / "outages.sqlite3"
    observations = write_csv(
        tmp_path,
        "invalid.csv",
        "timestamp,status,voltage\n"
        "2026-11-16T00:00:00Z,up,\n"
        "2026-11-16T00:01:00Z,down,\n"
        "2026-11-16T00:02:00Z,,180\n",
    )
    with pytest.raises(ValueError, match="voltage threshold"):
        record_observation_csv(database, "site", observations, max_gap=timedelta(minutes=2))
    assert read_events(database, "site") == []

    observations.write_text("timestamp,status\n2026-11-16T00:00:00Z,down\n", encoding="utf-8")
    record_observation_csv(database, "site", observations, max_gap=timedelta(minutes=2))
    assert read_events(database, "site") == [OutageEvent(at(0), None, "measured", True)]


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ("2026-11-16T00:07:00Z,up,\n2026-11-16T00:06:00Z,up,", "strictly increasing"),
        ("2026-11-16T00:05:00Z,down,", "conflicting observation"),
        ("2026-11-16T00:04:00Z,up,", "older than"),
        ("2026-11-16T00:06:00,up,", "UTC offset"),
        ("2026-11-16T00:06:00Z,,nan", "finite and non-negative"),
        ("2026-11-16T00:06:00Z,offline,", "up, down, or unknown"),
    ],
)
def test_rejects_invalid_or_out_of_order_observations(tmp_path, rows, error):
    database = tmp_path / "outages.sqlite3"
    first = write_csv(tmp_path, "first.csv", "timestamp,status\n2026-11-16T00:05:00Z,up\n")
    invalid = write_csv(tmp_path, "invalid.csv", f"timestamp,status,voltage\n{rows}\n")
    record_observation_csv(database, "site", first, max_gap=timedelta(minutes=2))
    with pytest.raises(ValueError, match=error):
        record_observation_csv(database, "site", invalid, max_gap=timedelta(minutes=2))
    assert read_events(database, "site") == []


def test_concurrent_event_imports_coalesce_measured_and_preserve_schedules(tmp_path):
    database = tmp_path / "outages.sqlite3"
    first = write_csv(
        tmp_path,
        "first.csv",
        "start,end,source\n"
        "2026-11-16T00:02:00Z,2026-11-16T00:06:00Z,measured\n"
        "2026-11-16T00:04:00Z,2026-11-16T00:08:00Z,scheduled\n",
    )
    second = write_csv(
        tmp_path,
        "second.csv",
        "start,end,source\n"
        "2026-11-16T03:05:00+03:00,2026-11-16T03:09:00+03:00,measured\n"
        "2026-11-16T00:04:00Z,2026-11-16T00:12:00Z,scheduled\n",
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [
            pool.submit(import_event_csv, database, "site", path) for path in (first, second) * 2
        ]
        for job in jobs:
            job.result()

    assert read_events(database, "site") == [
        OutageEvent(at(2), at(9), "measured"),
        OutageEvent(at(4), at(8), "scheduled"),
        OutageEvent(at(4), at(12), "scheduled"),
    ]

    import_event_csv(database, "other site", first)
    cancel_scheduled_event(database, "site", at(4), at(8))
    import_event_csv(database, "site", first)
    assert read_events(database, "site") == [
        OutageEvent(at(2), at(9), "measured"),
        OutageEvent(at(4), at(12), "scheduled"),
    ]
    assert read_events(database, "other site") == [
        OutageEvent(at(2), at(6), "measured"),
        OutageEvent(at(4), at(8), "scheduled"),
    ]


def test_explicit_change_from_open_to_closed_survives_old_replay(tmp_path):
    database = tmp_path / "outages.sqlite3"
    first = write_csv(tmp_path, "open.csv", "start,end,source\n2026-11-16T00:02:00Z,,measured\n")
    second = write_csv(
        tmp_path,
        "closed.csv",
        "start,end,source\n2026-11-16T00:02:00Z,2026-11-16T00:08:00Z,measured\n",
    )
    assert read_events(database, "site") == []
    assert not database.exists()
    import_event_csv(database, "site", first)
    assert read_events(database, "site") == [OutageEvent(at(2), None, "measured")]
    import_event_csv(database, "site", second)
    import_event_csv(database, "site", first)
    assert read_events(database, "site") == [OutageEvent(at(2), at(8), "measured")]


def test_overlapping_detected_and_imported_events_are_coalesced(tmp_path):
    database = tmp_path / "outages.sqlite3"
    observations = write_csv(
        tmp_path,
        "observations.csv",
        "timestamp,status\n"
        "2026-11-16T00:00:00Z,up\n"
        "2026-11-16T00:01:00Z,down\n"
        "2026-11-16T00:04:00Z,up\n",
    )
    history = write_csv(
        tmp_path,
        "history.csv",
        "start,end,source\n2026-11-16T00:02:00Z,2026-11-16T00:06:00Z,measured\n",
    )
    record_observation_csv(database, "site", observations, max_gap=timedelta(minutes=3))
    import_event_csv(database, "site", history)
    assert read_events(database, "site") == [OutageEvent(at(1), at(6), "measured")]


def test_imported_history_does_not_hide_an_open_detected_outage(tmp_path):
    database = tmp_path / "outages.sqlite3"
    observations = write_csv(
        tmp_path,
        "observations.csv",
        "timestamp,status\n2026-11-16T00:00:00Z,up\n2026-11-16T00:01:00Z,down\n",
    )
    history = write_csv(
        tmp_path,
        "history.csv",
        "start,end,source\n2026-11-16T00:01:00Z,2026-11-16T00:06:00Z,measured\n",
    )
    record_observation_csv(database, "site", observations, max_gap=timedelta(minutes=3))
    import_event_csv(database, "site", history)
    assert read_events(database, "site") == [
        OutageEvent(at(1), None, "measured"),
        OutageEvent(at(1), at(6), "measured"),
    ]


@pytest.mark.parametrize(
    ("row", "error"),
    [
        ("2026-11-16T00:02:00Z,,scheduled", "scheduled events require an end"),
        ("2026-11-16T00:02:00Z,2026-11-16T00:01:00Z,measured", "end must follow start"),
        ("2026-11-16T00:02:00Z,2026-11-16T00:03:00Z,predicted", "measured or scheduled"),
    ],
)
def test_event_csv_rejects_invalid_rows_without_partial_import(tmp_path, row, error):
    database = tmp_path / "outages.sqlite3"
    events = write_csv(
        tmp_path,
        "events.csv",
        f"start,end,source\n2026-11-16T00:00:00Z,2026-11-16T00:01:00Z,measured\n{row}\n",
    )
    with pytest.raises(ValueError, match=error):
        import_event_csv(database, "site", events)
    assert read_events(database, "site") == []


def test_cli_records_observations_and_lists_events(tmp_path):
    database = tmp_path / "outages.sqlite3"
    observations = write_csv(
        tmp_path,
        "voltage.csv",
        "timestamp,voltage\n"
        "2026-11-16T00:00:00Z,230\n"
        "2026-11-16T00:01:00Z,190\n"
        "2026-11-16T00:02:00Z,230\n",
    )
    base = [
        sys.executable,
        "-m",
        "emhass.outage_log",
        "--database",
        str(database),
        "--site",
        "home",
    ]
    subprocess.run(
        [*base, "observe", str(observations), "--max-gap-seconds", "60", "--down-voltage", "190"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([*base, "list"], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [
        {
            "start": "2026-11-16T00:01:00+00:00",
            "end": "2026-11-16T00:02:00+00:00",
            "source": "measured",
            "interrupted": False,
        },
    ]


def test_cli_imports_and_cancels_scheduled_events(tmp_path):
    database = tmp_path / "outages.sqlite3"
    schedule = write_csv(
        tmp_path,
        "schedule.csv",
        "start,end,source\n2026-11-16T00:02:00Z,2026-11-16T00:06:00Z,scheduled\n",
    )
    base = [
        sys.executable,
        "-m",
        "emhass.outage_log",
        "--database",
        str(database),
        "--site",
        "home",
    ]
    subprocess.run([*base, "import-events", str(schedule)], check=True)
    assert read_events(database, "home") == [OutageEvent(at(2), at(6), "scheduled")]
    subprocess.run([*base, "cancel-schedule", at(2).isoformat(), at(6).isoformat()], check=True)
    assert read_events(database, "home") == []
