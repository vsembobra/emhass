"""Deterministic outage reserve policy for a single battery."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from emhass.outage_log import OutageEvent


def utc_time(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone offset")
    return value.astimezone(UTC)


def finite_number(value: float, name: str, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")


@dataclass(frozen=True)
class ReserveSettings:
    nominal_capacity_kwh: float
    soh: float
    floor_soc: float
    max_soc: float
    discharge_efficiency: float
    cold_start_kw: float
    cold_start_hours: float = 2.0
    window_days: float = 14.0
    quantile: float = 0.9
    margin: float = 0.0
    decay_days: float = 7.0
    profile_mode: Literal["all_hours", "historical_hours"] = "all_hours"
    site_timezone: str = "UTC"
    scheduled_precharge_hours: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "nominal_capacity_kwh",
            "soh",
            "floor_soc",
            "max_soc",
            "discharge_efficiency",
            "cold_start_kw",
            "cold_start_hours",
            "window_days",
            "quantile",
            "margin",
            "decay_days",
            "scheduled_precharge_hours",
        ):
            finite_number(getattr(self, name), name)
        if self.nominal_capacity_kwh == 0 or not 0 < self.soh <= 1:
            raise ValueError("nominal capacity must be positive and soh must be in (0, 1]")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("discharge efficiency must be in (0, 1]")
        if not self.floor_soc < self.max_soc <= 1 or self.margin > 1:
            raise ValueError("SOC bounds must satisfy 0 <= floor < max <= 1 and margin <= 1")
        if not self.window_days or self.quantile > 1:
            raise ValueError("window_days must be positive and quantile must be in [0, 1]")
        if self.profile_mode not in ("all_hours", "historical_hours"):
            raise ValueError("invalid profile_mode")
        ZoneInfo(self.site_timezone)

    @property
    def effective_capacity_kwh(self) -> float:
        return self.nominal_capacity_kwh * self.soh

    def floor_for_energy(self, stored_kwh: float) -> tuple[float, float]:
        """Map energy above the physical floor to device SOC and its shortfall."""
        capacity = self.effective_capacity_kwh
        usable_capacity = capacity * (self.max_soc - self.floor_soc)
        requested_energy = stored_kwh + self.margin * capacity
        floor = min(self.max_soc, self.floor_soc + requested_energy / capacity)
        return floor, max(0.0, requested_energy - usable_capacity)


@dataclass(frozen=True)
class LoadObservation:
    start: datetime
    kw: float
    end: datetime | None = None

    def __post_init__(self) -> None:
        start = utc_time(self.start)
        finite_number(self.kw, "observed critical load")
        if self.end is not None and utc_time(self.end) <= start:
            raise ValueError("load observation end must follow its start")


@dataclass(frozen=True)
class ScheduledReserve:
    precharge_from: datetime
    deadline: datetime
    end: datetime
    reserve_kwh: float
    floor_soc: float
    shortfall_kwh: float


@dataclass(frozen=True)
class ReservePlan:
    reserve_kwh: float
    floor_soc: float
    shortfall_kwh: float
    duration_hours: float
    critical_load_kw: float
    scheduled: tuple[ScheduledReserve, ...]


def _merge_intervals(
    intervals: Iterable[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(intervals):
        if merged and start < merged[-1][1]:
            first, previous_end = merged[-1]
            merged[-1] = (first, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _overlaps_local_hour(start: datetime, end: datetime, hour: int, zone: ZoneInfo) -> bool:
    while start < end:
        local = start.astimezone(zone)
        if local.hour == hour:
            return True
        start += timedelta(
            hours=1, minutes=-local.minute, seconds=-local.second, microseconds=-local.microsecond
        )
    return False


def calculate_reserve(
    at: datetime,
    events: Iterable[OutageEvent],
    observed_load: Iterable[LoadObservation],
    settings: ReserveSettings,
) -> ReservePlan:
    """Return the general reserve and separate pre-charge windows known at `at`."""
    now = utc_time(at)
    cutoff = now - timedelta(days=settings.window_days)
    measured = []
    announced = []
    for event in events:
        if event.end is None or event.interrupted:
            continue
        start, end = utc_time(event.start), utc_time(event.end)
        if event.source == "measured" and cutoff < end <= now:
            measured.append((start, end))
        elif event.source == "scheduled" and end > now:
            announced.append((start, end))
    history = _merge_intervals(measured)
    durations = [(end - start).total_seconds() / 3600 for start, end in history]
    duration = settings.cold_start_hours
    if durations:
        duration = max(durations) if len(durations) < 5 else _quantile(durations, settings.quantile)

    load_values = []
    for observation in observed_load:
        start = utc_time(observation.start)
        end = utc_time(observation.end) if observation.end is not None else start
        if any(first <= start < last and end <= last for first, last in history):
            load_values.append(observation.kw)
    critical_kw = _quantile(load_values, 0.9) if load_values else settings.cold_start_kw
    baseline = settings.cold_start_hours * settings.cold_start_kw / settings.discharge_efficiency
    reserve_kwh = baseline
    if history:
        zone = ZoneInfo(settings.site_timezone)
        hour = now.astimezone(zone).hour
        in_profile = settings.profile_mode == "all_hours" or any(
            _overlaps_local_hour(start, end, hour, zone) for start, end in history
        )
        if in_profile:
            age_days = (now - history[-1][1]) / timedelta(days=1)
            weight = max(0.0, 1.0 - age_days / settings.decay_days) if settings.decay_days else 1.0
            historical = critical_kw * duration / settings.discharge_efficiency
            reserve_kwh += weight * max(0.0, historical - baseline)

    scheduled = []
    for start, end in _merge_intervals(announced):
        if start < now:
            continue
        hours = (end - start).total_seconds() / 3600
        energy = critical_kw * hours / settings.discharge_efficiency
        floor, shortfall = settings.floor_for_energy(energy)
        scheduled.append(
            ScheduledReserve(
                start - timedelta(hours=settings.scheduled_precharge_hours),
                start,
                end,
                energy,
                floor,
                shortfall,
            )
        )

    floor, shortfall = settings.floor_for_energy(reserve_kwh)
    return ReservePlan(reserve_kwh, floor, shortfall, duration, critical_kw, tuple(scheduled))
