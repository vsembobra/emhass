from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from tests.test_soc_recovery_prototype import build_optimization


def solve(optimizer, horizon=10, **kwargs):
    index = pd.date_range("2026-10-19", periods=horizon, freq="30min", tz="UTC")
    inputs = pd.DataFrame({"unit_load_cost": 0.2, "unit_prod_price": 0.05}, index=index)
    return optimizer.perform_naive_mpc_optim(
        inputs,
        pd.Series(0.0, index=index),
        pd.Series(2000.0, index=index),
        horizon,
        soc_init=0.3,
        soc_final=0.3,
        **kwargs,
    )


def test_profile_uses_end_of_step_and_composes_with_point_target():
    optimizer = build_optimization()
    profile = [0.5, 0.6, 0.7, None, 0.5, None, None, None, None, None]
    result = solve(optimizer, soc_floor_profile=profile, soc_target=0.75, soc_target_timestep=1)
    assert optimizer.optim_status == "Optimal"
    expected_floors = np.array([0.5, 0.75, 0.7, 0.5])
    assert (result["SOC_opt"].iloc[[0, 1, 2, 4]].to_numpy() >= expected_floors - 1e-6).all()
    assert result["SOC_opt"].iloc[-1] == pytest.approx(0.3)
    np.testing.assert_allclose(
        optimizer.param_soc_target_floor[0].value, [5000, 7500, 7000, 0, 5000, 0, 0, 0, 0, 0]
    )


def test_omitting_profile_clears_it_on_the_same_cached_problem():
    optimizer = build_optimization()
    baseline = solve(optimizer).copy()
    problem = optimizer.prob
    solve(optimizer, soc_floor_profile=[0.5] * 10)
    result = solve(optimizer)
    assert optimizer.prob is problem
    assert result.columns.equals(baseline.columns)
    assert result["optim_status"].iloc[0] == "Optimal"
    assert result["SOC_opt"].iloc[-1] == pytest.approx(baseline["SOC_opt"].iloc[-1])
    assert result["cost_profit"].sum() == pytest.approx(baseline["cost_profit"].sum())
    np.testing.assert_array_equal(optimizer.param_soc_target_floor[0].value, np.zeros(10))


def test_unreachable_floor_is_a_hard_constraint_even_with_frozen_soc():
    optimizer = build_optimization()
    result = solve(optimizer, soc_floor_profile=[0.4] * 10, battery_availability=0)
    assert list(result.columns) == ["optim_status"]
    assert result["optim_status"].iloc[0].lower() == "infeasible"
    assert solve(optimizer)["optim_status"].iloc[0] == "Optimal"


@pytest.mark.parametrize(
    "profile", [[0.5] * 9, [True] * 10, [float("nan")] * 10, [0.81] * 10, [0.29] * 10]
)
def test_invalid_floors_fail_before_solving(profile):
    with pytest.raises(ValueError, match="soc_floor_profile"):
        solve(build_optimization(), soc_floor_profile=profile)


def test_floor_requires_enabled_battery():
    with pytest.raises(ValueError, match="soc_floor_profile"):
        solve(build_optimization({"set_use_battery": False}), soc_floor_profile=[0.5] * 10)


def test_floor_resizes_with_the_horizon():
    optimizer = build_optimization()
    solve(optimizer, soc_floor_profile=[0.5] * 10)
    result = solve(optimizer, horizon=6, soc_floor_profile=[0.5] * 6)
    assert len(result) == 6
    assert result["SOC_opt"].min() >= 0.5 - 1e-6


def test_concurrent_profiles_keep_their_own_solve_parameters():
    optimizer = build_optimization()
    with ThreadPoolExecutor(max_workers=2) as pool:
        high = pool.submit(solve, optimizer, soc_floor_profile=[0.5] * 10)
        low = pool.submit(solve, optimizer)
    assert high.result()["SOC_opt"].min() >= 0.5 - 1e-6
    assert low.result()["SOC_opt"].iloc[-1] == pytest.approx(0.3)
