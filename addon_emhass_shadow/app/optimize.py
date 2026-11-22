"""MILP day plan for the EMHASS Shadow add-on.

Wraps emhass.optimization.Optimization (CVXPY + HiGHS) with the site's
battery/plant parameters from the add-on options and the shadow pipeline's
P50 forecasts. Produces an optimal 24 h charge/discharge plan, from which
target SOCs for the inverter's TOU zones are derived.

Shadow mode: results are advisory only, nothing is written to the inverter.
"""

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from emhass.optimization import Optimization

DAY_HOURS = 24  # applied/reporting day; the planning horizon may be longer (MPC)


def _cheap_windows(opts: dict) -> list[tuple[int, int]]:
    """Parsed tariff_cheap_windows as (begin, end) minute-of-day pairs.

    A window may wrap midnight ("23:00-07:00"). Unparseable chunks are
    skipped. Used both for prices and for telling the night charge zone
    apart from the expensive period in zone_targets.
    """
    windows = []
    for chunk in str(opts.get("tariff_cheap_windows", "")).split(","):
        chunk = chunk.strip()
        if not chunk or "-" not in chunk:
            continue
        begin, end = chunk.split("-", 1)
        try:
            bh, bm = (int(x) for x in begin.strip().split(":"))
            eh, em = (int(x) for x in end.strip().split(":"))
        except ValueError:
            continue
        windows.append((bh * 60 + bm, eh * 60 + em))
    return windows


def _is_cheap_minute(minute: int, windows: list[tuple[int, int]]) -> bool:
    return any(
        begin <= minute < end if begin < end else minute >= begin or minute < end
        for begin, end in windows
    )


def tariff_prices(starts: pd.DatetimeIndex, tz: ZoneInfo, opts: dict) -> list[float]:
    """Per-hour import price: cheap inside tariff_cheap_windows, normal else.

    tariff_cheap_windows is a comma-separated list of local "HH:MM-HH:MM"
    windows; a window may wrap midnight ("23:00-07:00"). When the tariff is
    disabled the price is flat, which still lets the optimizer arbitrage PV.
    """
    normal = float(opts.get("tariff_normal_price", 1.0))
    cheap = float(opts.get("tariff_cheap_price", 1.0))
    windows = _cheap_windows(opts) if opts.get("tariff_enabled", False) else []
    prices = []
    for stamp in starts:
        local = stamp.tz_convert(tz)
        minute = local.hour * 60 + local.minute
        prices.append(cheap if _is_cheap_minute(minute, windows) else normal)
    return prices


TIE_BREAK = 0.002  # price tilt over the whole horizon: decides ties, never a tariff gap


def later_is_cheaper(prices: list[float]) -> list[float]:
    """Prices the solver sees: each hour a hair dearer than the next.

    Two cheap nights cost the same, so buying tomorrow's energy tonight is a
    tie, and the solver often settles it by buying early. The battery then
    carries energy the day does not need, and every replan treats that
    carry-over as the plan's own and puts the cloudy-day margin on top of it
    again, so the night target creeps up (55 % needed, 90 % charged). With
    later hours marginally cheaper the plan buys as late as it can and spends
    what the battery already holds first. The tilt (0.2 % from the first hour
    to the last) is far below any tariff gap or charging loss, so it decides
    ties only; costs are reported at the real prices.
    """
    n = len(prices)
    if n < 2:
        return list(prices)
    return [p * (1.0 + TIE_BREAK * (n - 1 - i) / (n - 1)) for i, p in enumerate(prices)]


def soc_bounds(opts: dict) -> tuple[float, float]:
    """SOC bounds as fractions; accepts percent keys (min_soc_pct) and the
    pre-0.7 fraction keys (min_soc) so existing configs keep working."""
    if "min_soc_pct" in opts:
        return float(opts["min_soc_pct"]) / 100.0, float(opts["max_soc_pct"]) / 100.0
    return float(opts["min_soc"]), float(opts["max_soc"])


def _build_optimization(
    opts: dict, tz_name: str, battery_off: bool, soc_final_reward_factor: float = 0.0,
    horizon_hours: int = DAY_HOURS,
) -> "Optimization":
    # Lazy: the zone-window helpers below are shared with the heuristic floor
    # path, which must keep working even when the solver stack fails to load.
    from emhass.optimization import Optimization

    logger = logging.getLogger("emhass_shadow.optimize")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())

    min_soc, max_soc = soc_bounds(opts)
    charge_w = 0.0 if battery_off else float(opts["max_charge_kw"]) * 1000.0
    discharge_w = 0.0 if battery_off else float(opts["max_discharge_kw"]) * 1000.0

    retrieve_hass_conf = {
        "optimization_time_step": pd.Timedelta(hours=1),
        "time_zone": tz_name,
        "sensor_power_photovoltaics": "pv",
        "sensor_power_load_no_var_loads": "load",
    }
    optim_conf = {
        "delta_forecast_daily": pd.Timedelta(hours=horizon_hours),
        "num_threads": 0,
        "lp_solver": "HiGHS",
        "set_use_battery": True,
        "set_use_pv": True,
        "set_total_pv_sell": False,  # the site cannot sell to the grid
        "set_nocharge_from_grid": False,
        "set_nodischarge_to_grid": False,
        "set_battery_dynamic": False,
        "battery_dynamic_max": 0.9,
        "battery_dynamic_min": -0.9,
        "weight_battery_discharge": 0.0,
        "weight_battery_charge": 0.0,
        "number_of_deferrable_loads": 0,
        "nominal_power_of_deferrable_loads": [],
        "treat_deferrable_load_as_semi_cont": [],
        "set_deferrable_load_single_constant": [],
        "set_deferrable_startup_penalty": [],
        "operating_hours_of_each_deferrable_load": [],
        "start_timesteps_of_each_deferrable_load": [],
        "end_timesteps_of_each_deferrable_load": [],
        "lp_solver_timeout": 45,
        "lp_solver_mip_rel_gap": 0.01,
        "battery_soc_final_reward_factor": soc_final_reward_factor,
    }
    plant_conf = {
        "inverter_is_hybrid": False,
        # The site cannot export, so surplus PV must be curtailable in the
        # model — without this flag the solve is infeasible on sunny hours.
        "compute_curtailment": True,
        "maximum_power_from_grid": 60000,
        "maximum_power_to_grid": 0,
        "battery_discharge_power_max": discharge_w,
        "battery_charge_power_max": charge_w,
        "battery_minimum_state_of_charge": min_soc,
        "battery_maximum_state_of_charge": max_soc,
        "battery_target_state_of_charge": (min_soc + max_soc) / 2,
        "battery_nominal_energy_capacity": float(opts["capacity_kwh"]) * 1000.0,
        "battery_discharge_efficiency": float(opts["discharge_efficiency"]),
        "battery_charge_efficiency": float(opts["charge_efficiency"]),
        "battery_stress_cost": 0.0,
        "battery_stress_segments": 10,
    }
    import emhass

    emhass_conf = {
        "root_path": Path(emhass.__file__).resolve().parent,
        "data_path": Path("/data"),
    }
    return Optimization(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "unit_load_cost",
        "unit_prod_price",
        "profit",
        emhass_conf,
        logger,
        opt_time_delta=horizon_hours,
    )


def _solve(
    optimizer: "Optimization",
    starts: pd.DatetimeIndex,
    prices: list[float],
    pv_w: np.ndarray,
    load_w: np.ndarray,
    soc_init: float,
    floor_profile: list[float | None],
) -> pd.DataFrame:
    inputs = pd.DataFrame(
        {"unit_load_cost": prices, "unit_prod_price": [0.0] * len(starts)}, index=starts
    )
    # Terminal target: the reserve level of the hour right AFTER the horizon,
    # which in this day-periodic floor profile is floor_profile[0]. Using the
    # last hour's floor — the daily low, "the sun will recharge anyway" —
    # forces the solver to drain into the evening and curtail midday PV
    # (export is closed, so the surplus has nowhere else to go). EMHASS
    # enforces soc_final softly but at 100x the max tariff, i.e. ~hard.
    soc_final = max(floor_profile[-1], floor_profile[0])
    return optimizer.perform_naive_mpc_optim(
        inputs,
        pd.Series(pv_w, index=starts),
        pd.Series(load_w, index=starts),
        len(starts),
        soc_init=soc_init,
        soc_final=soc_final,
        def_total_hours=[],
        def_total_timestep=[],
        def_start_timestep=[],
        def_end_timestep=[],
        soc_floor_profile=floor_profile,
    )


def _charge_target_kept(
    window: list[int],
    strategy: str,
    cheap_windows: list[tuple[int, int]] | None,
    starts: pd.DatetimeIndex,
    tz: ZoneInfo,
) -> bool:
    """Whether a charging window keeps its rounded-up charge target.

    A Deye zone's Batt% is the discharge floor AND the grid-charge target.
    A target above the SOC in a window that is not entirely cheap makes the
    inverter buy the difference at full price (grid charge on) or refuse to
    discharge while it lags the plan (grid charge off) — and PV needs no
    target to charge the battery at all. So, whatever the strategy, only a
    window lying wholly inside the cheap tariff hours keeps its charge
    target; pass an empty list when no tariff is enabled. None means the
    caller gave no window info — keep the target (legacy behaviour).
    """
    if cheap_windows is None:
        return True
    return bool(cheap_windows) and all(
        _is_cheap_minute(
            (lambda local: local.hour * 60 + local.minute)(starts[i].tz_convert(tz)),
            cheap_windows,
        )
        for i in window
    )


class ZoneWindow(NamedTuple):
    """Where one TOU zone sits in the horizon.

    match: index of the zone's next start; window: its indices in cyclic
    horizon order (the next occurrence plus, for the zone in effect right
    now, the remaining hours of the current one); forward_end: end of the
    next occurrence without that wrap; current: remaining hours of the
    current occurrence when the zone is in effect at the horizon start
    (it started before the issue hour), else None.
    """

    match: int
    window: list[int]
    forward_end: int
    current: list[int] | None

    @property
    def now(self) -> tuple[list[int], bool]:
        """Indices the value written now governs first, and whether it is
        the zone in effect right now (entry SOC = measured SOC)."""
        if self.current:
            return self.current, True
        return list(range(self.match, self.forward_end)), self.match == 0


def zone_windows(
    programs: list[dict], starts: pd.DatetimeIndex, tz: ZoneInfo
) -> dict[int, ZoneWindow | None]:
    """Per TOU zone: its :class:`ZoneWindow`, or None when the zone's start
    hour does not occur in the horizon.

    The window runs from the zone's start to the next zone start in cyclic
    horizon order; a zone crossing the horizon edge (e.g. 00:00-07:00 when
    the horizon starts at 01:00) wraps around to the earliest one. That
    wrapping zone is the one in effect at the issue hour: its wrap part is
    happening NOW, not tomorrow, and the decisions it drives (target, grid
    charge) must use the measured SOC and those remaining hours.
    """
    matches = {}
    for program in programs:
        hour = int(program["from"][:2])
        matches[program["n"]] = next(
            (i for i in range(len(starts)) if starts[i].tz_convert(tz).hour == hour),
            None,
        )
    starts_idx = sorted(i for i in matches.values() if i is not None)
    windows: dict[int, ZoneWindow | None] = {}
    for n, match in matches.items():
        if match is None:
            windows[n] = None
            continue
        later = [i for i in starts_idx if i > match]
        end = later[0] if later else (starts_idx[0] if starts_idx else len(starts))
        current = None
        if end > match:
            window = list(range(match, end))
        else:
            window = list(range(match, len(starts))) + list(range(0, end))
            current = list(range(0, end)) or None
        windows[n] = ZoneWindow(
            match, window or [match], later[0] if later else len(starts), current
        )
    return windows


def reserve_floor_pct(reserve_floors: list[float], window: list[int], step_pct: int = 5) -> int:
    """Highest outage-reserve floor inside a zone window, in % rounded UP to the step.

    A TOU zone holds one Batt% for its whole window, so it must protect the
    highest reserve any hour of that window needs — not the reserve at the
    zone start or at the window's SOC minimum.
    """
    return math.ceil(max(reserve_floors[i] for i in window) * 100.0 / step_pct - 1e-9) * step_pct


def zone_targets(
    soc_opt: list[float], soc_init: float, programs: list[dict],
    starts: pd.DatetimeIndex, tz: ZoneInfo, min_soc: float, max_soc: float,
    strategy: str = "ladder", cheap_windows: list[tuple[int, int]] | None = None,
    reserve_floors: list[float] | None = None,
    soc_step_pct: int = 5,
) -> list[dict]:
    """Target SOC per TOU zone, respecting Deye semantics: a zone's Batt% is
    both the discharge floor AND the grid-charge target inside the window.

    Rounding to the inverter's 5 % step is always UP:
    - charging window (SOC rises across it, e.g. cheap night hours) → the
      end-of-window SOC rounded up — this is the charge target;
    - discharging/holding window → the window's minimum SOC rounded up too.
      Rounding down would shave up to 5 pp off the outage reserve (≈1.6 kWh
      on a 32 kWh battery); the small risk of a grid top-up to hold a floor
      that sits 1-4 pp above the plan is the accepted price of never
      under-reserving.

    Charge detection uses only the chronological forward part of the window
    (from the zone start to the next zone start, no cyclic wrap): the wrap
    part belongs to the horizon's beginning, i.e. EARLIER in real time, and
    comparing SOC across it fakes a rising window ("today evening" vs
    "tomorrow morning") and pins the zone to a bogus charge target.

    strategy selects the floor of the discharging windows:
    - "flat" (the add-on default): every zone outside the cheap tariff hours
      gets the lowest allowed SOC — min SOC, lifted by the outage reserve
      below. The inverter never discharges below a zone's Batt%, so any
      higher value would buy grid energy at the day price while the battery
      still holds charge; with one day price there is nothing to save
      energy for inside the day, and carrying it into the night is worse
      still (the night refill is cheaper). A cheap window without a charge
      keeps its own minimum SOC: the battery holds there unless the plan
      empties it for tomorrow's sun.
    - "ladder": the window's own minimum SOC — floors track the optimal
      trajectory hour by hour.
    With either strategy a charging window keeps its rounded-up target only
    when the whole window sits inside tariff_cheap_windows (the night charge
    zone); any other rising window — PV charging, which needs no target —
    gets the discharging floor, so the inverter can never grid-charge at
    full price when PV falls short of the plan. The zone in effect right now
    is also capped at the measured SOC (rounded down) unless it is a cheap
    charge window: hourly replanning keeps that cap current.

    reserve_floors (outage-reserve SOC fraction per horizon step) put a hard
    lower bound on every zone: the highest reserve inside its window. The
    trajectory meets the floor only at step ends, and the inverter holds one
    value for the whole zone, so without this bound a zone whose SOC minimum
    falls on a low-reserve hour lets the battery drain below the reserve of
    a later, riskier hour of the same zone whenever load exceeds forecast.
    """
    windows = zone_windows(programs, starts, tz)

    def clamp_pct(soc: float) -> float:
        return min(max(soc, min_soc), max_soc) * 100.0

    targets = []
    for program in programs:
        zone = windows[program["n"]]
        if zone is None:
            targets.append({**program, "optimal_soc_pct": None})
            continue
        # Decide on the hours the written value governs first. For the zone
        # in effect right now those are its remaining hours, entered at the
        # measured SOC — not tomorrow's occurrence. Otherwise it is the next
        # occurrence without the cyclic wrap: the wrap part is the horizon's
        # beginning, so comparing across it would set "today evening"
        # against "tomorrow morning" and fake a rising window.
        span, in_effect = zone.now
        if in_effect:
            entry_soc = soc_init
        else:
            entry_soc = soc_opt[zone.match - 1]
        end_soc = soc_opt[span[-1]]
        cheap_charge = _charge_target_kept(span, strategy, cheap_windows, starts, tz)
        if end_soc > entry_soc + 0.005 and cheap_charge:
            pct = math.ceil(clamp_pct(end_soc) / soc_step_pct - 1e-9) * soc_step_pct
        elif strategy == "flat" and not (cheap_windows and cheap_charge):
            # Expensive hours: the lowest allowed SOC; the reserve bound
            # below lifts it while an outage series is live.
            pct = math.ceil(min_soc * 100.0 / soc_step_pct - 1e-9) * soc_step_pct
        else:
            floor_soc = min(soc_opt[i] for i in span)
            # Round UP: rounding down would shave up to 5 pp off the reserve.
            pct = math.ceil(clamp_pct(floor_soc) / soc_step_pct - 1e-9) * soc_step_pct
        if in_effect and not cheap_charge:
            # The zone in effect governs NOW: a value above the measured SOC
            # would buy grid energy at full price (or freeze the battery while
            # it lags the plan). Round the measured SOC DOWN for the cap; the
            # outage reserve below still wins and is restored at any price.
            pct = min(pct, math.floor(clamp_pct(soc_init) / soc_step_pct + 1e-9) * soc_step_pct)
        if reserve_floors is not None:
            # The value persists into the zone's next occurrence until a later
            # run rewrites it, so the reserve bound covers the whole window.
            reserve_pct = reserve_floor_pct(reserve_floors, zone.window, soc_step_pct)
            pct = max(pct, min(reserve_pct, math.floor(max_soc * 100.0 + 1e-9)))
        targets.append({**program, "optimal_soc_pct": pct})
    return targets


def cheap_runs(
    starts: pd.DatetimeIndex, tz: ZoneInfo, cheap_windows: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """(first, last) horizon index of every contiguous run of cheap hours."""
    runs: list[tuple[int, int]] = []
    first = None
    for i, stamp in enumerate(starts):
        local = stamp.tz_convert(tz)
        if _is_cheap_minute(local.hour * 60 + local.minute, cheap_windows):
            first = i if first is None else first
        elif first is not None:
            runs.append((first, i - 1))
            first = None
    if first is not None:
        runs.append((first, len(starts) - 1))
    return runs


def night_charge_adjustment(
    soc_p50: list[float],
    floors: list[float],
    pv_p50_kw: list[float],
    pv_p10_kw: list[float],
    runs: list[tuple[int, int]],
    max_soc: float,
    capacity_kwh: float,
    discharge_efficiency: float,
    grid_charging: list[bool] | None = None,
) -> dict[int, float]:
    """SOC change at the end of each cheap run: what the next day needs.

    The night zone should hold what the expensive stretch up to the next
    cheap run needs, not what the median plan happens to reach. Hour by hour
    over that stretch, the sun the pessimistic forecast (PV P10) loses
    against the median one is drawn from the median plan's battery (counted
    at the discharge efficiency, a little on the safe side for surplus
    hours); ``spare`` is the least the battery would then stay above its
    floor.

    * spare < 0, a margin (+): on a dull day the battery would run down to
      the floor and the rest would be bought at the day price, so the night
      charges that much more. Capped by the headroom the median plan leaves
      below max SOC, so it never displaces the median day's sun.
    * spare > 0, a release (-): the battery covers even a dull day with room
      to spare. Holding it at the plan's own end-of-night level would only
      push unusual night load (a heater, a car) onto the grid and keep the
      battery fuller than the day needs, so the zone may go down to the need.
      Only when the plan buys nothing for the battery in that run: energy it
      does buy there may be meant for a later day.

    A run with nothing after it in the horizon gets no change. Earlier
    versions compared the median plan with a plan re-solved on PV P10; the
    gap between two solver trajectories also held tied choices (hold or spend
    a full battery overnight), so a battery that already covered both days
    was still told to hold 95 % all night.
    """
    adjust: dict[int, float] = {}
    for k, (first, last) in enumerate(runs):
        stop = runs[k + 1][0] if k + 1 < len(runs) else len(soc_p50)
        after = range(last + 1, stop)
        if not after:
            continue
        lost = 0.0
        spare = math.inf
        for i in after:
            missing_kw = max(0.0, float(pv_p50_kw[i]) - float(pv_p10_kw[i]))
            lost += missing_kw / (discharge_efficiency * capacity_kwh)  # 1 h steps
            spare = min(spare, soc_p50[i] - floors[i] - lost)
        if spare < 0.0:
            extra = min(-spare, max_soc - max(soc_p50[i] for i in after))
            if extra > 0.005:
                adjust[last] = extra
        elif spare > 0.005 and not (grid_charging and any(grid_charging[first : last + 1])):
            adjust[last] = -spare
    return adjust


def _raise_runs(
    soc: list[float], runs: list[tuple[int, int]], margin: dict[int, float]
) -> list[float]:
    """The trajectory with each adjustment (margin or release) ramped in over its cheap run."""
    out = list(soc)
    for first, last in runs:
        extra = margin.get(last, 0.0)
        for step, i in enumerate(range(first, last + 1), 1):
            out[i] = soc[i] + extra * step / (last - first + 1)
    return out


def run_plan(
    starts: pd.DatetimeIndex,
    pv_p50_kw: list[float],
    load_p50_kw: list[float],
    floors: list[float],
    soc_init: float,
    programs: list[dict],
    opts: dict,
    reserve_floors: list[float] | None = None,
    pv_p10_kw: list[float] | None = None,
) -> dict | None:
    """Optimal plan over len(starts) hours (24 or 48); None when infeasible.

    With a 48 h horizon the plan is MPC-style: tomorrow's hours inform
    today's charge/discharge, but only the first day drives zone targets,
    the cheap-charge total and the day-1 cost figures.
    floors are the shadow pipeline's per-hour minimum SOC fractions (already
    rounded to 5 %); they are clipped into the configured battery band and
    handed to the solver as a hard end-of-step floor profile. The baseline
    re-solve with a disabled battery prices the plan against doing nothing,
    giving the expected daily value of battery actions in tariff units.
    reserve_floors are the per-hour outage-reserve SOC fractions; every zone
    target is held at or above the highest reserve inside its window.
    pv_p10_kw (with a tariff) sets the night zone to what the next day needs
    on the pessimistic sun: a margin or a release (see night_charge_adjustment).
    """
    tz = ZoneInfo(opts["time_zone"])
    tz_name = opts["time_zone"]
    min_soc, max_soc = soc_bounds(opts)
    horizon = len(starts)
    day1 = min(DAY_HOURS, horizon)
    prices = tariff_prices(starts, tz, opts)
    solver_prices = later_is_cheaper(prices)
    pv_w = np.maximum(np.asarray(pv_p50_kw, dtype=float), 0.0) * 1000.0
    load_w = np.maximum(np.asarray(load_p50_kw, dtype=float), 0.0) * 1000.0
    soc_init = min(max(soc_init, min_soc), max_soc)
    floor_profile = [min(max(f, min_soc), max_soc) for f in floors]

    # Value leftover SOC just below the cheapest import: PV surplus is always
    # worth absorbing (reward > 0), but buying grid energy merely to overshoot
    # the terminal target never pays (reward < cheapest price). Without this
    # the soft terminal equality is ~hard, and a sunny horizon is forced to
    # curtail storable PV just to land exactly on the target.
    price_last = prices[-1]
    reward_factor = 0.9 * min(prices) / price_last if price_last > 0 else 0.0

    optimizer = _build_optimization(
        opts, tz_name, battery_off=False, soc_final_reward_factor=reward_factor,
        horizon_hours=horizon,
    )
    result = _solve(optimizer, starts, solver_prices, pv_w, load_w, soc_init, floor_profile)
    status = str(optimizer.optim_status)
    relaxed = False
    if status.lower() != "optimal":
        # A hard floor the forecasts cannot reach must not kill the plan:
        # retry with the flat configured minimum and mark the fallback.
        result = _solve(
            optimizer, starts, solver_prices, pv_w, load_w, soc_init, [min_soc] * horizon
        )
        status = str(optimizer.optim_status)
        relaxed = True
        if status.lower() != "optimal":
            return None

    soc_opt = [float(v) for v in result["SOC_opt"]]
    p_grid = [float(v) for v in result["P_grid_pos"]] if "P_grid_pos" in result else None
    p_batt = [float(v) for v in result["P_batt"]] if "P_batt" in result else None
    p_curtail = (
        [max(0.0, float(v)) for v in result["P_PV_curtailment"]]
        if "P_PV_curtailment" in result
        else None
    )
    # Money spent at the real prices (the solver saw the tie-break tilt); the
    # site cannot sell, so it is the import bill. cost_profit is the fallback.
    spent_plan = (
        float(sum(p_grid[i] / 1000.0 * prices[i] for i in range(horizon)))
        if p_grid is not None else -float(result["cost_profit"].sum())
    )

    # The night zone holds what the next day needs (night_charge_adjustment):
    # a margin where a dull day would end in day-price purchases, a release
    # where the battery covers even a dull day. The reported trajectory and
    # costs stay the median plan's.
    cheap_windows = _cheap_windows(opts) if opts.get("tariff_enabled", False) else []
    runs = cheap_runs(starts, tz, cheap_windows) if cheap_windows else []
    margin: dict[int, float] = {}
    if runs and pv_p10_kw is not None and not relaxed:
        grid_charging = (
            [
                p_batt[i] < -50.0 and p_grid[i] > max(0.0, load_w[i] - pv_w[i]) + 50.0
                for i in range(horizon)
            ]
            if p_grid is not None and p_batt is not None
            else None
        )
        margin = night_charge_adjustment(
            soc_opt,
            floor_profile,
            list(np.maximum(np.asarray(pv_p50_kw, dtype=float), 0.0)),
            list(np.maximum(np.asarray(pv_p10_kw, dtype=float), 0.0)),
            runs,
            max_soc,
            float(opts["capacity_kwh"]),
            float(opts["discharge_efficiency"]),
            grid_charging,
        )
    margin = {last: extra for last, extra in margin.items() if last < day1}
    margin_kwh = sum(max(0.0, extra) for extra in margin.values()) * float(
        opts["capacity_kwh"]
    ) / float(opts["charge_efficiency"])

    baseline = _build_optimization(opts, tz_name, battery_off=True, horizon_hours=horizon)
    base_result = _solve(
        baseline, starts, prices, pv_w, load_w, soc_init, [soc_init] * horizon
    )
    base_ok = str(baseline.optim_status).lower() == "optimal"
    spent_baseline = -float(base_result["cost_profit"].sum()) if base_ok else None

    # Day-1 spend: what the notification means by "today". The full-horizon
    # figures include tomorrow's preparation (e.g. tonight's cheap charge that
    # pays off tomorrow evening), which is why they are reported separately.
    spent_plan_day1 = (
        float(sum(p_grid[i] / 1000.0 * prices[i] for i in range(day1)))
        if p_grid is not None else None
    )
    base_grid = (
        [float(v) for v in base_result["P_grid_pos"]]
        if base_ok and "P_grid_pos" in base_result else None
    )
    spent_baseline_day1 = (
        float(sum(base_grid[i] / 1000.0 * prices[i] for i in range(day1)))
        if base_grid is not None else None
    )

    cheap = float(opts.get("tariff_cheap_price", 1.0))
    grid_charge_kwh = 0.0
    if p_grid is not None and p_batt is not None:
        for i in range(day1):  # only today's charge is actionable now
            charging = p_batt[i] < -50.0
            cheap_hour = opts.get("tariff_enabled", False) and prices[i] <= cheap + 1e-9
            if charging and cheap_hour:
                grid_charge_kwh += max(0.0, p_grid[i] - load_w[i]) / 1000.0

    return {
        "status": status,
        "floor_relaxed": relaxed,
        "horizon_hours": horizon,
        "prices": prices,
        "soc_opt": [round(s, 4) for s in soc_opt],
        "p_grid_kw": [round(v / 1000.0, 3) for v in p_grid] if p_grid else None,
        "p_batt_kw": [round(v / 1000.0, 3) for v in p_batt] if p_batt else None,
        "p_curtail_kw": [round(v / 1000.0, 3) for v in p_curtail] if p_curtail else None,
        "cost_plan": round(spent_plan, 2),
        "cost_baseline": round(spent_baseline, 2) if spent_baseline is not None else None,
        "savings": (
            round(spent_baseline - spent_plan, 2) if spent_baseline is not None else None
        ),
        "cost_plan_day1": round(spent_plan_day1, 2) if spent_plan_day1 is not None else None,
        "cost_baseline_day1": (
            round(spent_baseline_day1, 2) if spent_baseline_day1 is not None else None
        ),
        "savings_day1": (
            round(spent_baseline_day1 - spent_plan_day1, 2)
            if spent_plan_day1 is not None and spent_baseline_day1 is not None else None
        ),
        "grid_charge_kwh_cheap": round(grid_charge_kwh + margin_kwh, 2),
        "night_margin_kwh": round(margin_kwh, 2),
        "tariff_enabled": bool(opts.get("tariff_enabled", False)),
        "day_zone_strategy": str(opts.get("day_zone_strategy", "flat")),
        # TOU zones live in the first day; tomorrow's half of the trajectory
        # is context for today's decisions, not an applied target.
        "zone_targets": zone_targets(
            _raise_runs(soc_opt, runs, margin)[:day1], soc_init, programs, starts[:day1], tz,
            min_soc, max_soc,
            strategy=str(opts.get("day_zone_strategy", "flat")),
            cheap_windows=cheap_windows,
            reserve_floors=reserve_floors[:day1] if reserve_floors is not None else None,
            soc_step_pct=max(1, int(opts.get("tou_soc_step_pct", 5))),
        ),
    }
