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
