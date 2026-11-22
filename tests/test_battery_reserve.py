from datetime import UTC, datetime, timedelta

import pytest

from emhass.outage_log import OutageEvent
from emhass.reserve import LoadObservation, ReserveSettings, calculate_reserve

NOW = datetime(2026, 10, 19, 12, tzinfo=UTC)


def settings(**overrides):
    return ReserveSettings(
        nominal_capacity_kwh=13.0,
        soh=12.5 / 13.0,
        floor_soc=0.1,
        max_soc=0.9,
        discharge_efficiency=0.92,
        cold_start_kw=1.0,
        **overrides,
    )


def measured(end, hours, **kwargs):
    return OutageEvent(end - timedelta(hours=hours), end, "measured", **kwargs)


def test_reserve_quantile_maps_to_full_effective_capacity():
    events = [measured(NOW - timedelta(days=i), hours) for i, hours in enumerate([4, 3.5, 2, 2, 1])]
    loads = [LoadObservation(event.start, 1.5) for event in events]
    plan = calculate_reserve(NOW, events, loads, settings(margin=0.02))
    assert plan.duration_hours == pytest.approx(3.8)
    assert plan.reserve_kwh == pytest.approx(6.19565217)
    assert plan.floor_soc == pytest.approx(0.61565217)
    assert plan.shortfall_kwh == 0


def test_cold_start_excludes_open_old_and_future_events():
    events = [
        measured(NOW - timedelta(days=15), 15),
        measured(NOW + timedelta(hours=1), 3),
        OutageEvent(NOW - timedelta(hours=2), None, "measured"),
    ]
    plan = calculate_reserve(NOW, events, [], settings())
    assert plan.duration_hours == 2
    assert plan.critical_load_kw == 1
    assert plan.reserve_kwh == pytest.approx(2.17391304)
    assert plan.floor_soc == pytest.approx(0.27391304)


def test_an_outage_the_battery_did_not_survive_still_counts():
    # Four 2 h outages the battery rode through; the fifth ran it flat after 3 h,
    # the logger went dark and the event was closed "interrupted". Its recorded
    # span is a lower bound: dropping it kept the reserve at the 2 h it failed with.
    events = [measured(NOW - timedelta(days=d), 2) for d in (1, 2, 3, 4)]
    events.append(measured(NOW - timedelta(days=5), 3, interrupted=True))
    plan = calculate_reserve(NOW, events, [], settings())
    assert plan.duration_hours > 2


def test_one_event_and_overlapping_duplicates_use_the_longest_combined_event():
    events = [measured(NOW, 3), measured(NOW - timedelta(hours=2), 4)]
    loads = [LoadObservation(NOW - timedelta(hours=3), 1.0), LoadObservation(NOW, 900.0)]
    plan = calculate_reserve(NOW, events, loads, settings())
    assert plan.duration_hours == 6
    assert plan.critical_load_kw == 1


def test_reserve_decays_to_cold_start_and_resets_on_a_new_outage():
    config = settings(decay_days=4)
    event = measured(NOW, 6)
    original = calculate_reserve(NOW, [event], [], config)
    halfway = calculate_reserve(NOW + timedelta(days=2), [event], [], config)
    expired = calculate_reserve(NOW + timedelta(days=4), [event], [], config)
    later = NOW + timedelta(days=4)
    reset = calculate_reserve(later, [event, measured(later, 6)], [], config)
    assert original.reserve_kwh == pytest.approx(6.52173913)
    assert halfway.reserve_kwh == pytest.approx(4.34782609)
    assert expired.reserve_kwh == pytest.approx(2.17391304)
    assert reset.reserve_kwh == pytest.approx(original.reserve_kwh)


def test_hourly_profile_keeps_baseline_outside_historical_local_hours():
    event = measured(NOW - timedelta(days=1, hours=1), 2)
    config = settings(profile_mode="historical_hours", site_timezone="Europe/Istanbul")
    inside = calculate_reserve(NOW - timedelta(hours=2), [event], [], config)
    outside = calculate_reserve(NOW, [event], [], config)
    # Use a reserve larger than cold start to distinguish the hourly gate.
    loads = [LoadObservation(event.start, 2)]
    assert inside.reserve_kwh == pytest.approx(outside.reserve_kwh)
    assert calculate_reserve(NOW - timedelta(hours=2), [event], loads, config).reserve_kwh > 3
    assert calculate_reserve(NOW, [event], loads, config).reserve_kwh == pytest.approx(2.17391304)


def test_capacity_and_scheduled_window_shortfalls_remain_visible():
    scheduled_start = NOW + timedelta(hours=3)
    schedule = [
        OutageEvent(scheduled_start, scheduled_start + timedelta(hours=10), "scheduled"),
        OutageEvent(
            scheduled_start + timedelta(hours=8), scheduled_start + timedelta(hours=14), "scheduled"
        ),
    ]
    result = calculate_reserve(NOW, schedule, [], settings(scheduled_precharge_hours=2))
    assert result.floor_soc == pytest.approx(0.27391304)
    assert len(result.scheduled) == 1
    window = result.scheduled[0]
    assert window.precharge_from == NOW + timedelta(hours=1)
    assert window.deadline == scheduled_start
    assert window.floor_soc == 0.9
    assert window.shortfall_kwh == pytest.approx(5.21739130)
    emergency = calculate_reserve(NOW, [measured(NOW, 14)], [], settings())
    assert emergency.floor_soc == 0.9
    assert emergency.shortfall_kwh == pytest.approx(5.21739130)


def test_naive_times_and_bad_physical_parameters_are_rejected():
    with pytest.raises(ValueError, match="timezone"):
        calculate_reserve(NOW.replace(tzinfo=None), [], [], settings())
    with pytest.raises(ValueError, match="efficiency"):
        ReserveSettings(12.5, 1, 0.1, 0.9, 0, 1)


def _flapping(gap_minutes):
    """Three 50-minute outages with the grid back for ``gap_minutes`` between them."""
    events, end = [], NOW
    for _ in range(3):
        events.append(OutageEvent(end - timedelta(minutes=50), end, "measured"))
        end -= timedelta(minutes=50 + gap_minutes)
    return events


def test_a_flapping_grid_counts_as_one_long_outage():
    # Back for 10 minutes: the battery is not refilled, the drain only pauses.
    plan = calculate_reserve(NOW, _flapping(10), [], settings())
    assert plan.duration_hours == pytest.approx(170 / 60)


def test_outages_apart_by_half_an_hour_or_more_stay_separate():
    plan = calculate_reserve(NOW, _flapping(30), [], settings())
    assert plan.duration_hours == pytest.approx(50 / 60)
    glued = calculate_reserve(NOW, _flapping(30), [], settings(flap_gap_hours=1.0))
    assert glued.duration_hours == pytest.approx(210 / 60)
