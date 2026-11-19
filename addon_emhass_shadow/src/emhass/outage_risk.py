"""Empirical per-hour outage risk from the site's own outage journal.

The deterministic reserve in :mod:`emhass.reserve` is reactive: it rises after
an outage and holds while the event stays inside the history window. This
module adds the proactive half — an empirical risk profile over local hours
built from measured outages, so the reserve can rise *before* a historically
typical outage window instead of only after the fact.

Estimators are deliberately simple and honest about sparse data: with a few
dozen outages over a handful of months, hourly frequency and a duration
quantile are identifiable; weather or seasonal conditioning is not, and is
left out rather than invented.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from emhass.reserve import _quantile, utc_time

if TYPE_CHECKING:
    pass

MIN_EVENTS = 3
MIN_DAYS_OBSERVED = 14.0


@dataclass(frozen=True)
class OutageRiskProfile:
    """Empirical outage statistics over local wall-clock hours.

    ``hourly_probability[h]`` is the expected outage-hours per day with local
    hour ``h`` — i.e. how often that hour was inside an outage, normalised by
    the observed span. ``risk_hours`` marks hours whose probability stands out
    against the day's mean.
    """

    hourly_probability: tuple[float, ...]
    mean_duration_hours: float
    duration_p90_hours: float
    days_observed: float
    outages_count: int
    risk_hours: tuple[bool, ...]


def _hour_weights(start: datetime, end: datetime, zone: ZoneInfo) -> Iterable[tuple[int, float]]:
    """Yield (local hour, hours of the interval spent in it)."""
    cursor = start
    while cursor < end:
        local = cursor.astimezone(zone)
        boundary = (local + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        chunk_end = min(end, boundary.astimezone(start.tzinfo))
        yield local.hour, (chunk_end - cursor).total_seconds() / 3600
        cursor = chunk_end


def outage_risk_profile(
    events: Iterable,
    at: datetime,
    site_timezone: str,
    *,
    min_events: int = MIN_EVENTS,
    min_days: float = MIN_DAYS_OBSERVED,
    risk_ratio: float = 2.0,
    min_probability: float = 0.05,
) -> OutageRiskProfile | None:
    """Build the hourly risk profile, or None when the journal is too sparse.

    Only finished, uninterrupted, measured outages count: scheduled
    announcements and still-open events say nothing about typical timing.
    """
    now = utc_time(at)
    zone = ZoneInfo(site_timezone)
    intervals = []
    for event in events:
        if event.source != "measured" or event.end is None or event.interrupted:
            continue
        start, end = utc_time(event.start), utc_time(event.end)
        if end <= now:
            intervals.append((start, end))
    if len(intervals) < min_events:
        return None
    first = min(start for start, _ in intervals)
    days_observed = (now - first).total_seconds() / 86400
    if days_observed < min_days:
        return None

    totals = [0.0] * 24
    durations = []
    for start, end in intervals:
        durations.append((end - start).total_seconds() / 3600)
        for hour, weight in _hour_weights(start, end, zone):
            totals[hour] += weight
    hourly = tuple(total / days_observed for total in totals)
    mean = sum(hourly) / 24
    threshold = max(min_probability, risk_ratio * mean)
    risk = tuple(probability >= threshold for probability in hourly)
    return OutageRiskProfile(
        hourly,
        sum(durations) / len(durations),
        _quantile(durations, 0.9),
        days_observed,
        len(intervals),
        risk,
    )
