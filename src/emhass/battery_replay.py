"""Offline comparison of fixed, forecast-based and existing EMHASS dispatch."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from math import isclose
from pathlib import Path

from emhass.battery_backtest import Battery, Dispatch, ReplayResult, ReplayStep, simulate
from emhass.battery_planning import ForecastStep, baseline_floor
from emhass.last_run import OK_OPTIM_STATUSES
from emhass.outage_log import OutageEvent, read_events
from emhass.reserve import LoadObservation, ReserveSettings, calculate_reserve, utc_time


@dataclass(frozen=True)
class ForecastRun:
    available_at: datetime
    steps: tuple[ForecastStep, ...]


@dataclass(frozen=True)
class AnnouncedOutage:
    available_at: datetime
    event: OutageEvent
    cancelled_at: datetime | None = None


@dataclass(frozen=True)
class ObjectiveStep:
    available_at: datetime
    start: datetime
    p_batt_w: float
    optim_status: str = "Optimal"


@dataclass(frozen=True)
class ObjectivePlan:
    objective: str
    steps: tuple[ObjectiveStep, ...]


@dataclass(frozen=True)
class ReplayComparison:
    fixed: ReplayResult
    baseline: ReplayResult
    hindsight: ReplayResult | None
    baseline_vs_fixed: str | None
    forecast_fallback_steps: int
    hindsight_steps: int
    missed_precharge_deadlines: int
    objective: str | None
    objective_replay: ReplayResult | None
    objective_steps: int
    objective_vs_fixed: str | None


def _rank(candidate: ReplayResult, fixed: ReplayResult) -> str | None:
    if fixed.outage_hours == 0:
        return None
    pairs = [
        (candidate.unserved_critical_kwh, fixed.unserved_critical_kwh),
        (candidate.unserved_critical_hours, fixed.unserved_critical_hours),
        (-candidate.own_solar_kwh, -fixed.own_solar_kwh),
    ]
    if candidate.net_cost is not None and fixed.net_cost is not None:
        pairs.append((candidate.net_cost, fixed.net_cost))
    for left, right in pairs:
        if not isclose(left, right, abs_tol=1e-9, rel_tol=0.0):
            return "better" if left < right else "worse"
    return "tie"


def _measured_windows(steps: Sequence[ReplayStep]) -> list[OutageEvent]:
    events = []
    outage_start = None
    for step in steps:
        if not step.grid_up and outage_start is None:
            outage_start = utc_time(step.start)
        elif step.grid_up and outage_start is not None:
            events.append(
                OutageEvent(
                    outage_start, utc_time(step.start), "measured", outage_start == steps[0].start
                )
            )
            outage_start = None
    return events


def replay_site(
    steps: Sequence[ReplayStep],
    battery: Battery,
    reserve: ReserveSettings,
    *,
    initial_soc: float,
    fixed_floor_soc: float,
    forecasts: Sequence[ForecastRun] = (),
    measured_events: Sequence[OutageEvent] | None = None,
    outage_load: Sequence[LoadObservation] = (),
    scheduled: Sequence[AnnouncedOutage] = (),
    objective_plan: ObjectivePlan | None = None,
) -> ReplayComparison:
    """Compare policies on identical actuals. Hindsight replaces only matched forecast windows."""
    if (
        battery.min_soc != reserve.floor_soc
        or battery.max_soc != reserve.max_soc
        or battery.discharge_efficiency != reserve.discharge_efficiency
        or not isclose(battery.capacity_kwh, reserve.effective_capacity_kwh)
    ):
        raise ValueError(
            "simulated battery and reserve must share capacity, SOC bounds and efficiency"
        )
    fixed = simulate(steps, battery, initial_soc, lambda _at, _soc: Dispatch(fixed_floor_soc))
    actuals = {utc_time(step.start): step for step in steps}
    measured = _measured_windows(steps) if measured_events is None else list(measured_events)
    if any(event.source != "measured" for event in measured):
        raise ValueError("scheduled events require a separate available_at timestamp")
    past_load = [
        *outage_load,
        *(
            LoadObservation(at, step.critical_load_kw, at + timedelta(hours=step.hours))
            for at, step in actuals.items()
            if not step.grid_up
        ),
    ]

    horizons = {}
    receipt_times = set()
    for run in sorted(forecasts, key=lambda run: utc_time(run.available_at)):
        available_at = utc_time(run.available_at)
        if available_at in receipt_times:
            raise ValueError("forecast runs must have distinct available_at timestamps")
        receipt_times.add(available_at)
        if not run.steps or any(
            left.end != right.start for left, right in zip(run.steps, run.steps[1:], strict=False)
        ):
            raise ValueError("forecast runs must contain contiguous steps on the replay grid")
        for index, step in enumerate(run.steps):
            if step.start in actuals and step.hours != actuals[step.start].hours:
                raise ValueError("forecast and actual interval durations differ")
            if available_at <= step.start and step.start in actuals:
                horizons[step.start] = run.steps[index:]

    schedules = []
    for announcement in scheduled:
        event = announcement.event
        if (
            event.source != "scheduled"
            or event.end is None
            or utc_time(event.end) <= utc_time(event.start)
        ):
            raise ValueError("announcements must contain a finite scheduled outage window")
        available_at = utc_time(announcement.available_at)
        cancelled_at = utc_time(announcement.cancelled_at) if announcement.cancelled_at else None
        if cancelled_at is not None and cancelled_at < available_at:
            raise ValueError("schedule cancellation predates the announcement")
        schedules.append((available_at, cancelled_at, event))

    floors = {}
    hindsight_floors = {}
    precharge = {}
    deadlines = {}
    for at in actuals:
        known = [
            event
            for announced, cancelled, event in schedules
            if announced <= at and (cancelled is None or at < cancelled)
        ]
        plan = calculate_reserve(at, [*measured, *known], past_load, reserve)
        horizon = horizons.get(at)
        floors[at] = (
            baseline_floor(horizon, plan.reserve_kwh, reserve).floor_soc
            if horizon
            else max(fixed_floor_soc, plan.floor_soc)
        )
        if horizon and all(step.start in actuals for step in horizon):
            perfect = [
                ForecastStep(
                    step.start, step.hours, actuals[step.start].pv_kw, actuals[step.start].load_kw
                )
                for step in horizon
            ]
            hindsight_floors[at] = baseline_floor(perfect, plan.reserve_kwh, reserve).floor_soc
        for window in plan.scheduled:
            if window.precharge_from <= at < window.deadline:
                previous = precharge.get(at)
                if previous is None or window.floor_soc > previous:
                    precharge[at] = window.floor_soc
            if at == window.deadline:
                deadlines[at] = window.floor_soc

    missed_deadlines = 0

    def baseline_dispatch(at: datetime, soc: float, *, hindsight: bool = False) -> Dispatch:
        nonlocal missed_deadlines
        if not hindsight and at in deadlines and soc + 1e-9 < deadlines[at]:
            missed_deadlines += 1
        floor = hindsight_floors.get(at, floors[at]) if hindsight else floors[at]
        target = precharge.get(at)
        if target is not None and soc < target:
            stored_needed = (target - soc) * battery.capacity_kwh
            charge_kw = stored_needed / (battery.charge_efficiency * actuals[at].hours)
            return Dispatch(max(floor, target), -charge_kw)
        return Dispatch(max(floor, target) if target is not None else floor)

    baseline = simulate(steps, battery, initial_soc, baseline_dispatch)
    hindsight = None
    if hindsight_floors:
        hindsight = simulate(
            steps, battery, initial_soc, lambda at, soc: baseline_dispatch(at, soc, hindsight=True)
        )

    objective_result = None
    setpoints = {}
    if objective_plan is not None:
        if objective_plan.objective not in {"profit", "cost", "self-consumption"}:
            raise ValueError("objective must be profit, cost or self-consumption")
        seen = set()
        for row in sorted(objective_plan.steps, key=lambda row: utc_time(row.available_at)):
            start, available = utc_time(row.start), utc_time(row.available_at)
            if start in actuals and available <= start and row.optim_status in OK_OPTIM_STATUSES:
                if (start, available) in seen:
                    raise ValueError("duplicate objective setpoint for one step and receipt time")
                seen.add((start, available))
                setpoints[start] = row.p_batt_w / 1000
        if setpoints:
            objective_result = simulate(
                steps,
                battery,
                initial_soc,
                lambda at, _soc: Dispatch(fixed_floor_soc, setpoints.get(at)),
            )

    return ReplayComparison(
        fixed,
        baseline,
        hindsight,
        _rank(baseline, fixed),
        len(actuals) - len(horizons),
        len(hindsight_floors),
        missed_deadlines,
        objective_plan.objective if objective_plan else None,
        objective_result,
        len(setpoints),
        _rank(objective_result, fixed)
        if objective_result and len(setpoints) == len(actuals)
        else None,
    )


def _parse_time(record: dict, *fields: str) -> dict:
    return {
        key: datetime.fromisoformat(value) if key in fields and value is not None else value
        for key, value in record.items()
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="site replay inputs as JSON")
    parser.add_argument(
        "--outage-db", type=Path, help="use measured events from a local outage log"
    )
    parser.add_argument("--site", help="site key in the outage log")
    args = parser.parse_args(argv)
    if bool(args.outage_db) != bool(args.site):
        parser.error("--outage-db and --site must be used together")
    with args.dataset.open(encoding="utf-8") as file:
        data = json.load(file)
    reserve = ReserveSettings(**data["reserve"])
    battery = Battery(
        reserve.effective_capacity_kwh,
        reserve.floor_soc,
        reserve.max_soc,
        discharge_efficiency=reserve.discharge_efficiency,
        **data["battery"],
    )
    events = (
        None
        if args.outage_db is None
        else [
            event for event in read_events(args.outage_db, args.site) if event.source == "measured"
        ]
    )
    objective = data.get("emhass_plan")
    report = replay_site(
        [ReplayStep(**_parse_time(step, "start")) for step in data["actuals"]],
        battery,
        reserve,
        initial_soc=data["initial_soc"],
        fixed_floor_soc=data["fixed_floor_soc"],
        forecasts=[
            ForecastRun(
                datetime.fromisoformat(run["available_at"]),
                tuple(ForecastStep(**_parse_time(step, "start")) for step in run["steps"]),
            )
            for run in data.get("forecasts", [])
        ],
        measured_events=events,
        outage_load=[
            LoadObservation(**_parse_time(row, "start", "end"))
            for row in data.get("outage_load", [])
        ],
        scheduled=[
            AnnouncedOutage(
                datetime.fromisoformat(row["available_at"]),
                OutageEvent(
                    datetime.fromisoformat(row["start"]),
                    datetime.fromisoformat(row["end"]),
                    "scheduled",
                ),
                datetime.fromisoformat(row["cancelled_at"]) if row.get("cancelled_at") else None,
            )
            for row in data.get("scheduled", [])
        ],
        objective_plan=None
        if objective is None
        else ObjectivePlan(
            objective["objective"],
            tuple(
                ObjectiveStep(**_parse_time(row, "available_at", "start"))
                for row in objective["steps"]
            ),
        ),
    )
    print(json.dumps(asdict(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
