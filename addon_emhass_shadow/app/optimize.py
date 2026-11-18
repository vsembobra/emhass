"""MILP day plan for the EMHASS Shadow add-on.

Wraps emhass.optimization.Optimization (CVXPY + HiGHS) with the site's
battery/plant parameters from the add-on options and the shadow pipeline's
P50 forecasts. Produces an optimal 24 h charge/discharge plan, from which
target SOCs for the inverter's TOU zones are derived.

Shadow mode: results are advisory only, nothing is written to the inverter.
"""

import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from emhass.optimization import Optimization

HOURS = 24


def tariff_prices(starts: pd.DatetimeIndex, tz: ZoneInfo, opts: dict) -> list[float]:
    """Per-hour import price: cheap inside tariff_cheap_windows, normal else.

    tariff_cheap_windows is a comma-separated list of local "HH:MM-HH:MM"
    windows; a window may wrap midnight ("23:00-07:00"). When the tariff is
    disabled the price is flat, which still lets the optimizer arbitrage PV.
    """
    normal = float(opts.get("tariff_normal_price", 1.0))
    cheap = float(opts.get("tariff_cheap_price", 1.0))
    windows = []
    if opts.get("tariff_enabled", False):
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
    prices = []
    for stamp in starts:
        local = stamp.tz_convert(tz)
        minute = local.hour * 60 + local.minute
        is_cheap = any(
            begin <= minute < end if begin < end else minute >= begin or minute < end
            for begin, end in windows
        )
        prices.append(cheap if is_cheap else normal)
    return prices


def _build_optimization(opts: dict, tz_name: str, battery_off: bool) -> Optimization:
    logger = logging.getLogger("emhass_shadow.optimize")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())

    min_soc = float(opts["min_soc"])
    max_soc = float(opts["max_soc"])
    charge_w = 0.0 if battery_off else float(opts["max_charge_kw"]) * 1000.0
    discharge_w = 0.0 if battery_off else float(opts["max_discharge_kw"]) * 1000.0

    retrieve_hass_conf = {
        "optimization_time_step": pd.Timedelta(hours=1),
        "time_zone": tz_name,
        "sensor_power_photovoltaics": "pv",
        "sensor_power_load_no_var_loads": "load",
    }
    optim_conf = {
        "delta_forecast_daily": pd.Timedelta(hours=HOURS),
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
        opt_time_delta=HOURS,
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
    return optimizer.perform_naive_mpc_optim(
        inputs,
        pd.Series(pv_w, index=starts),
        pd.Series(load_w, index=starts),
        len(starts),
        soc_init=soc_init,
        soc_final=floor_profile[-1],
        def_total_hours=[],
        def_total_timestep=[],
        def_start_timestep=[],
        def_end_timestep=[],
        soc_floor_profile=floor_profile,
    )


def zone_targets(
    soc_opt: list[float], soc_init: float, programs: list[dict],
    starts: pd.DatetimeIndex, tz: ZoneInfo, min_soc: float, max_soc: float,
) -> list[dict]:
    """Target SOC per TOU zone: the optimal SOC at the END of the zone's window.

    Deye semantics: a zone's Batt% is the level the battery is driven toward
    during the window (charged up to with grid charge on, discharged down to
    otherwise), so the meaningful setpoint is where the optimal trajectory
    wants the SOC when the window closes — the start of the next zone, or
    the horizon end for the zone that runs past the last solved hour.
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
    targets = []
    for program in programs:
        match = matches[program["n"]]
        if match is None:
            targets.append({**program, "optimal_soc_pct": None})
            continue
        later = [i for i in starts_idx if i > match]
        soc = soc_opt[later[0] - 1] if later else soc_opt[-1]
        soc = min(max(soc, min_soc), max_soc)
        targets.append({**program, "optimal_soc_pct": round(soc * 20) * 5})
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
    """Optimal 24 h plan; None when the solver cannot produce one.

    floors are the shadow pipeline's per-hour minimum SOC fractions (already
    rounded to 5 %); they are clipped into the configured battery band and
    handed to the solver as a hard end-of-step floor profile. The baseline
    re-solve with a disabled battery prices the plan against doing nothing,
    giving the expected daily value of battery actions in tariff units.
    """
    tz = ZoneInfo(opts["time_zone"])
    tz_name = opts["time_zone"]
    min_soc, max_soc = float(opts["min_soc"]), float(opts["max_soc"])
    prices = tariff_prices(starts, tz, opts)
    pv_w = np.maximum(np.asarray(pv_p50_kw, dtype=float), 0.0) * 1000.0
    load_w = np.maximum(np.asarray(load_p50_kw, dtype=float), 0.0) * 1000.0
    soc_init = min(max(soc_init, min_soc), max_soc)
    floor_profile = [min(max(f, min_soc), max_soc) for f in floors]

    optimizer = _build_optimization(opts, tz_name, battery_off=False)
    result = _solve(optimizer, starts, prices, pv_w, load_w, soc_init, floor_profile)
    status = str(optimizer.optim_status)
    relaxed = False
    if status.lower() != "optimal":
        # A hard floor the forecasts cannot reach must not kill the plan:
        # retry with the flat configured minimum and mark the fallback.
        result = _solve(
            optimizer, starts, prices, pv_w, load_w, soc_init, [min_soc] * len(starts)
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

    baseline = _build_optimization(opts, tz_name, battery_off=True)
    base_result = _solve(
        baseline, starts, prices, pv_w, load_w, soc_init, [soc_init] * len(starts)
    )
    spent_baseline = (
        -float(base_result["cost_profit"].sum())
        if str(baseline.optim_status).lower() == "optimal"
        else None
    )

    cheap = float(opts.get("tariff_cheap_price", 1.0))
    grid_charge_kwh = 0.0
    if p_grid is not None and p_batt is not None:
        for i in range(len(starts)):
            charging = p_batt[i] < -50.0
            cheap_hour = opts.get("tariff_enabled", False) and prices[i] <= cheap + 1e-9
            if charging and cheap_hour:
                grid_charge_kwh += max(0.0, p_grid[i] - load_w[i]) / 1000.0

    return {
        "status": status,
        "floor_relaxed": relaxed,
        "prices": prices,
        "soc_opt": [round(s, 4) for s in soc_opt],
        "p_grid_kw": [round(v / 1000.0, 3) for v in p_grid] if p_grid else None,
        "p_batt_kw": [round(v / 1000.0, 3) for v in p_batt] if p_batt else None,
        "cost_plan": round(spent_plan, 2),
        "cost_baseline": round(spent_baseline, 2) if spent_baseline is not None else None,
        "savings": (
            round(spent_baseline - spent_plan, 2) if spent_baseline is not None else None
        ),
        "grid_charge_kwh_cheap": round(grid_charge_kwh, 2),
        "tariff_enabled": bool(opts.get("tariff_enabled", False)),
        "zone_targets": zone_targets(
            soc_opt, soc_init, programs, starts, tz, min_soc, max_soc
        ),
    }
