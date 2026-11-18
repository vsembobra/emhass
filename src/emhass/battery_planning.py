"""Advisory SOC floor up to the first forecast solar crossover."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from emhass.reserve import ReserveSettings, finite_number, utc_time


@dataclass(frozen=True)
class ForecastStep:
    start: datetime
    hours: float
    pv_p10_kw: float
    load_p90_kw: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", utc_time(self.start))
        finite_number(self.hours, "forecast duration")
        if timedelta(hours=self.hours) <= timedelta(0):
            raise ValueError("forecast duration must be positive")
        finite_number(self.pv_p10_kw, "PV forecast")
        finite_number(self.load_p90_kw, "load forecast")

    @property
    def end(self) -> datetime:
        return self.start + timedelta(hours=self.hours)


@dataclass(frozen=True)
class FloorRecommendation:
    floor_soc: float
    energy_to_crossover_kwh: float
    crossover: datetime | None
    shortfall_kwh: float


def baseline_floor(
    forecast: Sequence[ForecastStep], reserve_kwh: float, settings: ReserveSettings
) -> FloorRecommendation:
    finite_number(reserve_kwh, "reserve energy")
    if not forecast:
        raise ValueError("a forecast must cover at least one interval")
    if any(left.end != right.start for left, right in zip(forecast, forecast[1:], strict=False)):
        raise ValueError("forecast intervals must be contiguous")

    deficit_kwh = 0.0
    crossover = None
    for step in forecast:
        if step.pv_p10_kw > 0 and step.pv_p10_kw >= step.load_p90_kw:
            crossover = step.start
            break
        deficit_kwh += max(0.0, step.load_p90_kw - step.pv_p10_kw) * step.hours

    stored_kwh = deficit_kwh / settings.discharge_efficiency
    floor, shortfall = settings.floor_for_energy(stored_kwh + reserve_kwh)
    return FloorRecommendation(floor, stored_kwh, crossover, shortfall)


@dataclass(frozen=True)
class FloorProfileStep:
    """Floor for one forecast interval: deficit from this interval to its own
    solar crossover plus the per-interval outage reserve."""

    start: datetime
    hours: float
    floor_soc: float
    stored_kwh: float
    shortfall_kwh: float
    crossover: datetime | None


@dataclass(frozen=True)
class FloorProfile:
    steps: tuple[FloorProfileStep, ...]
    max_floor_soc: float
    max_shortfall_kwh: float


def floor_profile(
    forecast: Sequence[ForecastStep],
    reserve_kwh: float | Sequence[float],
    settings: ReserveSettings,
) -> FloorProfile:
    """Time-varying advisory floor: one SOC floor per forecast interval.

    Where :func:`baseline_floor` sizes a single floor from the deficit up to
    the first crossover, this sizes every interval against the deficit from
    that interval to its own next crossover. Nights keep a high floor (long
    deficit ahead), mornings relax as the sun approaches, and evenings rise
    again towards the overnight deficit — as long as the horizon covers the
    next morning, which a 24 h forecast from a morning issue does.

    ``reserve_kwh`` may be a single value (same reserve all day) or one value
    per interval, e.g. to raise the reserve only during historically risky
    outage hours.
    """
    if not forecast:
        raise ValueError("a forecast must cover at least one interval")
    if any(left.end != right.start for left, right in zip(forecast, forecast[1:], strict=False)):
        raise ValueError("forecast intervals must be contiguous")
    count = len(forecast)
    if isinstance(reserve_kwh, (int, float)) and not isinstance(reserve_kwh, bool):
        reserves = [float(reserve_kwh)] * count
    else:
        reserves = [float(value) for value in reserve_kwh]
        if len(reserves) != count:
            raise ValueError("per-interval reserve must match the forecast length")
    for value in reserves:
        finite_number(value, "reserve energy")

    deficit = [max(0.0, step.load_p90_kw - step.pv_p10_kw) * step.hours for step in forecast]
    suffix = [0.0] * (count + 1)
    for index in range(count - 1, -1, -1):
        suffix[index] = suffix[index + 1] + deficit[index]

    crossover_at: list[int | None] = [None] * count
    current: int | None = None
    for index in range(count - 1, -1, -1):
        step = forecast[index]
        if step.pv_p10_kw > 0 and step.pv_p10_kw >= step.load_p90_kw:
            current = index
        crossover_at[index] = current

    steps = []
    for index, step in enumerate(forecast):
        cross = crossover_at[index]
        deficit_kwh = suffix[index] - (suffix[cross] if cross is not None else 0.0)
        stored_kwh = deficit_kwh / settings.discharge_efficiency + reserves[index]
        floor, shortfall = settings.floor_for_energy(stored_kwh)
        steps.append(
            FloorProfileStep(
                step.start,
                step.hours,
                floor,
                stored_kwh,
                shortfall,
                forecast[cross].start if cross is not None else None,
            )
        )
    return FloorProfile(
        tuple(steps),
        max(step.floor_soc for step in steps),
        max(step.shortfall_kwh for step in steps),
    )
