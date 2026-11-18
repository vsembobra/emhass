from datetime import UTC, datetime, timedelta

from emhass.outage_log import OutageEvent
from emhass.outage_risk import outage_risk_profile

NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)
TZ = "Europe/Istanbul"  # UTC+3, no DST


def _outage(day: datetime, hour_local: int, hours: float) -> OutageEvent:
    start = day.replace(hour=0, tzinfo=UTC) + timedelta(hours=hour_local - 3)
    return OutageEvent(start, start + timedelta(hours=hours), "measured", False)


def _sample_events() -> list[OutageEvent]:
    # Evening outages on ten different days spread over ~40 days.
    first = NOW - timedelta(days=40)
    return [_outage(first + timedelta(days=4 * i), 18, 2.0) for i in range(10)]


def test_profile_marks_historically_risky_hours():
    profile = outage_risk_profile(_sample_events(), NOW, TZ)
    assert profile is not None
    assert profile.outages_count == 10
    assert profile.mean_duration_hours == 2.0
    assert profile.hourly_probability[18] > 0
    assert profile.risk_hours[18] and profile.risk_hours[19]
    assert not profile.risk_hours[3]
    assert sum(profile.hourly_probability) < 1


def test_sparse_journals_yield_no_profile():
    assert outage_risk_profile([], NOW, TZ) is None
    assert outage_risk_profile(_sample_events()[:2], NOW, TZ) is None
    # Enough events but too recent a journal to say anything about "typical".
    recent = [_outage(NOW - timedelta(days=i), 18, 1.0) for i in range(5)]
    assert outage_risk_profile(recent, NOW, TZ) is None


def test_interrupted_and_scheduled_events_are_ignored():
    events = _sample_events()
    first = NOW - timedelta(days=40)
    interrupted = OutageEvent(
        first, first + timedelta(hours=3), "measured", interrupted=True
    )
    announced = OutageEvent(
        NOW + timedelta(days=1), NOW + timedelta(days=1, hours=2), "scheduled", False
    )
    profile = outage_risk_profile([*events, interrupted, announced], NOW, TZ)
    assert profile.outages_count == 10


def test_hour_weights_split_across_local_midnight():
    first = NOW - timedelta(days=30)
    events = [_outage(first + timedelta(days=10 * i), 23, 3.0) for i in range(3)]
    profile = outage_risk_profile(events, NOW, TZ)
    assert profile is not None
    assert profile.hourly_probability[23] > 0
    assert profile.hourly_probability[0] > 0
    assert profile.hourly_probability[1] > 0
