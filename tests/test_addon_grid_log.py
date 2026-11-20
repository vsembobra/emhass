"""Grid-status series, outage-log carry rows and time-weighted hourly means."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import pipeline  # noqa: E402

T0 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)


def state(minutes: float, value: str) -> dict:
    at = (T0 + timedelta(minutes=minutes)).isoformat()
    return {"state": value, "last_changed": at, "last_updated": at}


def states(*pairs):
    return [state(minutes, value) for minutes, value in pairs]


def test_unknown_blip_swallowed_when_brief_and_recovered():
    series = pipeline.grid_status_series(
        states((0, "on"), (100, "unavailable"), (105, "on"), (200, "off"))
    )
    assert [s for _, s in series] == ["up", "down"]


def test_long_unknown_kept_as_unknown():
    series = pipeline.grid_status_series(
        states((0, "on"), (100, "unavailable"), (130, "on"))
    )
    assert [s for _, s in series] == ["up", "unknown", "up"]


def test_trailing_unknown_kept():
    series = pipeline.grid_status_series(states((0, "on"), (100, "unknown")))
    assert [s for _, s in series] == ["up", "unknown"]


def test_duplicate_states_collapse():
    series = pipeline.grid_status_series(
        states((0, "on"), (60, "on"), (120, "off"), (180, "off"))
    )
    assert [s for _, s in series] == ["up", "down"]


def test_carry_rows_hourly_only_where_heartbeat_covers():
    down_at = T0 + timedelta(hours=10, minutes=30)
    series = [(down_at, "down")]
    # Heartbeat alive for the whole window: carry rows at 11:00, 12:00, 13:00.
    beats = [T0 + timedelta(minutes=m) for m in range(0, 15 * 60)]
    now = T0 + timedelta(hours=13, minutes=30)
    rows = pipeline._grid_carry_rows(series, beats, now)
    assert rows == [
        (T0 + timedelta(hours=h), "down") for h in (11, 12, 13)
    ]


def test_carry_rows_stop_at_heartbeat_gap():
    down_at = T0 + timedelta(hours=10, minutes=30)
    series = [(down_at, "down")]
    # Heartbeat dies at 11:29: hours up to 13:00 are within 2 h of it,
    # 14:00 is not (>2 h stale).
    beats = [T0 + timedelta(minutes=m) for m in range(0, 11 * 60 + 30)]
    now = T0 + timedelta(hours=16)
    rows = pipeline._grid_carry_rows(series, beats, now)
    assert rows == [(T0 + timedelta(hours=h), "down") for h in (11, 12, 13)]


def test_carry_rows_skip_unknown_periods():
    series = [
        (T0, "up"),
        (T0 + timedelta(hours=5), "unknown"),
        (T0 + timedelta(hours=10), "up"),
    ]
    beats = [T0 + timedelta(minutes=m) for m in range(0, 12 * 60)]
    now = T0 + timedelta(hours=12)
    rows = pipeline._grid_carry_rows(series, beats, now)
    statuses = dict(rows)
    assert statuses[T0 + timedelta(hours=4)] == "up"
    assert T0 + timedelta(hours=6) not in statuses  # unknown period: no carry
    assert statuses[T0 + timedelta(hours=11)] == "up"


def test_hourly_mean_is_time_weighted():
    hour = T0
    # 1 kW carried in from before the hour, 5 kW only for the last minute.
    samples = [
        (T0 - timedelta(minutes=30), 1.0),
        (T0 + timedelta(minutes=59), 5.0),
    ]
    mean = pipeline.hourly_mean(samples, hour)
    assert mean == pipeline.hourly_mean(samples, hour)  # deterministic
    assert abs(mean - (59 * 1.0 + 1 * 5.0) / 60) < 1e-9


def test_hourly_mean_carries_value_without_samples():
    samples = [(T0 - timedelta(hours=2), 2.5)]
    assert pipeline.hourly_mean(samples, T0) == 2.5
    assert pipeline.hourly_mean([], T0) is None


def test_hourly_mean_normalizes_by_covered_duration():
    # First sample appears mid-hour: the mean covers only that half hour.
    samples = [(T0 + timedelta(minutes=30), 4.0)]
    assert pipeline.hourly_mean(samples, T0) == 4.0


def test_carried_returns_status_strings_unchanged():
    series = [(T0, "up"), (T0 + timedelta(hours=3), "down")]
    assert pipeline._carried(series, T0 + timedelta(hours=5)) == "down"
    assert pipeline._carried(series, T0) is None


def test_numeric_samples_sorted():
    states_list = [
        {"state": "2.0", "last_updated": (T0 + timedelta(hours=1)).isoformat()},
        {"state": "1.0", "last_updated": T0.isoformat()},
    ]
    samples = pipeline.numeric_samples(states_list, 1.0)
    assert samples == [(T0, 1.0), (T0 + timedelta(hours=1), 2.0)]
