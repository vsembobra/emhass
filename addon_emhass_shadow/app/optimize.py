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
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

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


def soc_bounds(opts: dict) -> tuple[float, float]:
    """SOC bounds as fractions; accepts percent keys (min_soc_pct) and the
    pre-0.7 fraction keys (min_soc) so existing configs keep working."""
    if "min_soc_pct" in opts:
        return float(opts["min_soc_pct"]) / 100.0, float(opts["max_soc_pct"]) / 100.0
    return float(opts["min_soc"]), float(opts["max_soc"])


def _build_optimization(
    opts: dict, tz_name: str, battery_off: bool, soc_final_reward_factor: float = 0.0,
    horizon_hours: int = DAY_HOURS,
) -> Optimization:
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
    optimizer: Optimization,
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

    Only "flat" presses charging windows, and only outside the cheap night
    hours: with the target pressed to the flat floor the inverter cannot
    grid-charge at peak prices when PV falls short of the plan. None means
    the caller gave no window info — keep the target (legacy behavior).
    """
    if strategy != "flat" or cheap_windows is None:
        return True
    return all(
        _is_cheap_minute(
            (lambda local: local.hour * 60 + local.minute)(starts[i].tz_convert(tz)),
            cheap_windows,
        )
        for i in window
    )


def zone_targets(
    soc_opt: list[float], soc_init: float, programs: list[dict],
    starts: pd.DatetimeIndex, tz: ZoneInfo, min_soc: float, max_soc: float,
    strategy: str = "ladder", cheap_windows: list[tuple[int, int]] | None = None,
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
    - "ladder": the window's own minimum SOC — floors track the optimal
      trajectory, saving energy for the evening peak and outages;
    - "flat": the whole-horizon minimum SOC — one low floor everywhere, the
      battery is spent whenever loads appear.
    Charging windows are unaffected by "ladder". In "flat" a charging window
    keeps its rounded-up target only when the whole window sits inside
    tariff_cheap_windows (the night charge zone); a charging window in the
    expensive period is pressed to the same flat floor, so the inverter
    cannot grid-charge at peak prices when PV falls short of the plan.
    """
    matches = {}
    for program in programs:
        hour = int(program["from"][:2])
        match = next(
            (i for i in range(len(starts)) if starts[i].tz_convert(tz).hour == hour),
            None,
        )
        matches[program["n"]] = match
    starts_idx = sorted(i for i in matches.values() if i is not None)

    def clamp_pct(soc: float) -> float:
        return min(max(soc, min_soc), max_soc) * 100.0

    horizon_min = min(soc_opt) if soc_opt else soc_init

    targets = []
    for program in programs:
        match = matches[program["n"]]
        if match is None:
            targets.append({**program, "optimal_soc_pct": None})
            continue
        # Window runs from this zone's start to the next zone start in cyclic
        # horizon order; a zone crossing the horizon edge (e.g. 00:00-07:00
        # when the horizon starts at 01:00) wraps around to the earliest one.
        later = [i for i in starts_idx if i > match]
        end = later[0] if later else (starts_idx[0] if starts_idx else len(soc_opt))
        if end > match:
            window = list(range(match, end))
        else:
            window = list(range(match, len(soc_opt))) + list(range(0, end))
        entry_soc = soc_init if match == 0 else soc_opt[match - 1]
        # Charge detection runs on the chronological forward part only:
        # [match, next zone start) without the cyclic wrap. The wrap part is
        # the horizon's beginning — earlier in real time — so its end_soc
        # would compare "today evening" against "tomorrow morning" and could
        # fake a rising window, pinning the zone to a bogus charge target.
        forward_end = later[0] if later else len(soc_opt)
        end_soc = soc_opt[forward_end - 1]
        if not window:
            window = [match]
        if end_soc > entry_soc + 0.005 and _charge_target_kept(window, strategy, cheap_windows, starts, tz):
            pct = math.ceil(clamp_pct(end_soc) / 5.0 - 1e-9) * 5
        else:
            floor_soc = min(soc_opt[i] for i in window)
            if strategy == "flat":
                floor_soc = min(floor_soc, horizon_min)
            # Round UP: rounding down would shave up to 5 pp off the reserve.
            pct = math.ceil(clamp_pct(floor_soc) / 5.0 - 1e-9) * 5
        targets.append({**program, "optimal_soc_pct": pct})
    return targets


def run_plan(
    starts: pd.DatetimeIndex,
    pv_p50_kw: list[float],
    load_p50_kw: list[float],
    floors: list[float],
    soc_init: float,
    programs: list[dict],
    opts: dict,
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
    """
    tz = ZoneInfo(opts["time_zone"])
    tz_name = opts["time_zone"]
    min_soc, max_soc = soc_bounds(opts)
    horizon = len(starts)
    day1 = min(DAY_HOURS, horizon)
    prices = tariff_prices(starts, tz, opts)
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
    result = _solve(optimizer, starts, prices, pv_w, load_w, soc_init, floor_profile)
    status = str(optimizer.optim_status)
    relaxed = False
    if status.lower() != "optimal":
        # A hard floor the forecasts cannot reach must not kill the plan:
        # retry with the flat configured minimum and mark the fallback.
        result = _solve(
            optimizer, starts, prices, pv_w, load_w, soc_init, [min_soc] * horizon
        )
        status = str(optimizer.optim_status)
        relaxed = True
        if status.lower() != "optimal":
            return None

    soc_opt = [float(v) for v in result["SOC_opt"]]
    p_grid = [float(v) for v in result["P_grid_pos"]] if "P_grid_pos" in result else None
    p_batt = [float(v) for v in result["P_batt"]] if "P_batt" in result else None
    # cost_profit is revenue minus spend, so money spent is its negation.
    spent_plan = -float(result["cost_profit"].sum())

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
        "grid_charge_kwh_cheap": round(grid_charge_kwh, 2),
        "tariff_enabled": bool(opts.get("tariff_enabled", False)),
        "day_zone_strategy": str(opts.get("day_zone_strategy", "ladder")),
        # TOU zones live in the first day; tomorrow's half of the trajectory
        # is context for today's decisions, not an applied target.
        "zone_targets": zone_targets(
            soc_opt[:day1], soc_init, programs, starts[:day1], tz, min_soc, max_soc,
            strategy=str(opts.get("day_zone_strategy", "ladder")),
            cheap_windows=_cheap_windows(opts),
        ),
    }
