"""Replay a single battery on an AC bus using interval-average power in kW."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import inf, isfinite, ulp
from numbers import Real


@dataclass(frozen=True)
class Battery:
    """Effective capacity in kWh, fractional SoC and efficiency, and AC power limits."""

    capacity_kwh: float
    min_soc: float
    max_soc: float
    max_charge_kw: float
    max_discharge_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    export_limit_kw: float | None = None
    import_limit_kw: float | None = None


@dataclass(frozen=True)
class ReplayStep:
    """Actual site readings for the interval beginning at the given aware instant."""

    start: datetime
    hours: float
    pv_kw: float
    load_kw: float
    critical_load_kw: float
    grid_up: bool
    import_price: float | None = None
    export_price: float | None = None


@dataclass(frozen=True)
class Dispatch:
    """Grid-connected SoC floor and optional AC setpoint (positive is discharge)."""

    floor_soc: float
    battery_kw: float | None = None


@dataclass(frozen=True)
class ReplayResult:
    """Energy, outage shortfalls, and energy-only bill totals for a complete replay."""

    imported_kwh: float
    exported_kwh: float
    curtailed_kwh: float
    own_solar_kwh: float
    unserved_critical_kwh: float
    unserved_critical_hours: float
    net_cost: float | None
    final_soc: float
    outage_hours: float


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _nonnegative(value: float, name: str) -> float:
    if _finite(value, name) < 0:
        raise ValueError(f"{name} must be non-negative")
    return float(value)


def _validate_battery(battery: Battery, initial_soc: float) -> None:
    if _finite(battery.capacity_kwh, "capacity_kwh") <= 0:
        raise ValueError("capacity_kwh must be positive")
    minimum = _finite(battery.min_soc, "min_soc")
    maximum = _finite(battery.max_soc, "max_soc")
    if not 0 <= minimum < maximum <= 1:
        raise ValueError("SoC bounds must satisfy 0 <= min_soc < max_soc <= 1")
    if not minimum <= _finite(initial_soc, "initial_soc") <= maximum:
        raise ValueError("initial_soc must be within battery SoC bounds")
    _nonnegative(battery.max_charge_kw, "max_charge_kw")
    _nonnegative(battery.max_discharge_kw, "max_discharge_kw")
    if battery.export_limit_kw is not None:
        _nonnegative(battery.export_limit_kw, "export_limit_kw")
    if battery.import_limit_kw is not None:
        _nonnegative(battery.import_limit_kw, "import_limit_kw")
    if not 0 < _finite(battery.charge_efficiency, "charge_efficiency") <= 1:
        raise ValueError("charge_efficiency must be in (0, 1]")
    if not 0 < _finite(battery.discharge_efficiency, "discharge_efficiency") <= 1:
        raise ValueError("discharge_efficiency must be in (0, 1]")


def _interval(start: datetime, hours: float) -> tuple[datetime, datetime]:
    if not isinstance(start, datetime) or start.utcoffset() is None:
        raise ValueError("interval start must be a timezone-aware datetime")
    start = start.astimezone(UTC)
    try:
        end = start + timedelta(hours=_finite(hours, "hours"))
    except OverflowError as error:
        raise ValueError("interval end is outside the datetime range") from error
    if end <= start:
        raise ValueError("interval duration must be positive")
    return start, end


def _grid_limit(kw: float | None, hours: float, grid_up: bool) -> float:
    if not grid_up:
        return 0.0
    return inf if kw is None else kw * hours


def simulate(
    steps: Iterable[ReplayStep],
    battery: Battery,
    initial_soc: float,
    policy: Callable[[datetime, float], Dispatch],
) -> ReplayResult:
    """Replay contiguous intervals, calling the policy with UTC start time and starting SoC.

    The default grid-connected dispatch charges from surplus PV and serves site load above
    the policy floor. An explicit setpoint can also import for charging or export discharge.
    During an outage PV serves critical and then other load before charging. Discharge serves
    critical load down to the battery's hard minimum. This assumes PV can operate while islanded.

    Own solar counts direct PV to load and PV-origin discharge to load. Stored energy is
    treated as a well-mixed inventory whose initial energy has unknown origin. Net cost is
    available when some prices are supplied and every actual import or export has a price.
    """
    _validate_battery(battery, initial_soc)
    minimum = battery.min_soc * battery.capacity_kwh
    maximum = battery.max_soc * battery.capacity_kwh
    stored = initial_soc * battery.capacity_kwh
    solar_stored = 0.0
    imported = exported = curtailed = own_solar = 0.0
    unserved = unserved_hours = outage_hours = 0.0
    net_cost: float | None = 0.0
    has_prices = False
    previous_end = None

    for step in steps:
        start, end = _interval(step.start, step.hours)
        if previous_end is not None and start != previous_end:
            raise ValueError("replay intervals must be contiguous and ordered by UTC instant")
        previous_end = end
        # Avoid rounding just beyond the declared bounds when converting the inventory to SoC.
        soc = max(battery.min_soc, min(battery.max_soc, stored / battery.capacity_kwh))
        dispatch = policy(start, soc)
        floor_soc = _finite(dispatch.floor_soc, "floor_soc")
        if not battery.min_soc <= floor_soc <= battery.max_soc:
            raise ValueError("floor_soc must be within battery SoC bounds")
        setpoint = dispatch.battery_kw
        if setpoint is not None:
            setpoint = _finite(setpoint, "battery_kw")

        hours = step.hours
        pv = _nonnegative(step.pv_kw, "pv_kw") * hours
        load = _nonnegative(step.load_kw, "load_kw") * hours
        critical = _nonnegative(step.critical_load_kw, "critical_load_kw") * hours
        if not all(map(isfinite, (pv, load, critical))):
            raise ValueError("interval energy must be finite")
        if step.critical_load_kw > step.load_kw:
            raise ValueError("critical_load_kw must not exceed load_kw")
        if not isinstance(step.grid_up, bool):
            raise ValueError("grid_up must be a bool")
        grid_up = step.grid_up
        import_price, export_price = step.import_price, step.export_price
        if import_price is not None:
            import_price = _finite(import_price, "import_price")
        if export_price is not None:
            export_price = _finite(export_price, "export_price")
        has_prices |= import_price is not None or export_price is not None

        direct = min(pv, load)
        surplus = pv - direct
        site_deficit = load - direct
        discharge_deficit = site_deficit if grid_up else max(0.0, critical - pv)
        import_limit = _grid_limit(battery.import_limit_kw, hours, grid_up)
        export_limit = _grid_limit(battery.export_limit_kw, hours, grid_up)
        floor = floor_soc * battery.capacity_kwh if grid_up else minimum
        if grid_up and setpoint is not None:
            requested_charge = max(0.0, -setpoint) * hours
            requested_discharge = max(0.0, setpoint) * hours
        else:
            requested_charge = surplus
            requested_discharge = discharge_deficit

        # Charge consumes available PV first and leaves enough import capacity for site load.
        charge = min(
            requested_charge,
            battery.max_charge_kw * hours,
            (maximum - stored) / battery.charge_efficiency,
            surplus + max(0.0, import_limit - site_deficit),
        )
        # Within the export cap, direct PV gets priority ahead of battery exports.
        discharge = min(
            requested_discharge,
            battery.max_discharge_kw * hours,
            max(0.0, stored - floor) * battery.discharge_efficiency,
            discharge_deficit + max(0.0, export_limit - surplus),
        )

        solar_charge = min(charge, surplus)
        battery_to_load = min(discharge, discharge_deficit)
        grid_to_load = min(import_limit, site_deficit - battery_to_load)
        pv_to_grid = min(export_limit, surplus - solar_charge)
        step_import = grid_to_load + (charge - solar_charge)
        step_export = pv_to_grid + (discharge - battery_to_load)
        curtailed += surplus - solar_charge - pv_to_grid
        imported += step_import
        exported += step_export

        solar_fraction = solar_stored / stored if discharge else 0.0
        own_solar += direct + battery_to_load * solar_fraction
        withdrawal = discharge / battery.discharge_efficiency
        if not grid_up:
            outage_hours += hours
            shortfall = discharge_deficit - battery_to_load
            # A few rounding steps occur when converting stored energy to AC delivery.
            roundoff = 4 * ulp(max(critical, stored * battery.discharge_efficiency))
            if shortfall > roundoff:
                unserved += shortfall
                unserved_hours += hours
        stored = max(
            minimum, min(maximum, stored + charge * battery.charge_efficiency - withdrawal)
        )
        solar_stored += solar_charge * battery.charge_efficiency - withdrawal * solar_fraction
        solar_stored = max(0.0, min(stored, solar_stored))

        if (step_import and import_price is None) or (step_export and export_price is None):
            net_cost = None
        elif net_cost is not None:
            net_cost += step_import * (import_price or 0.0) - step_export * (export_price or 0.0)

    if previous_end is None:
        raise ValueError("replay requires at least one interval")
    return ReplayResult(
        imported_kwh=imported,
        exported_kwh=exported,
        curtailed_kwh=curtailed,
        own_solar_kwh=own_solar,
        unserved_critical_kwh=unserved,
        unserved_critical_hours=unserved_hours,
        net_cost=net_cost if has_prices else None,
        final_soc=max(battery.min_soc, min(battery.max_soc, stored / battery.capacity_kwh)),
        outage_hours=outage_hours,
    )
