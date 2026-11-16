# Optional SOC floor in MPC

For a single enabled battery, Python callers of `Optimization.perform_naive_mpc_optim` can pass `soc_floor_profile` as a keyword argument. Give it one element per solved timestep, in the order of the MPC input rows. A numeric element is the minimum SOC **at the end of that step**, as a fraction within the battery's configured min/max bounds. `None` at a step leaves the existing battery bounds and recovery rules in force. Omitting the entire profile uses the regular MPC path.

```python
result = optimizer.perform_naive_mpc_optim(
    inputs,
    pv_w,
    load_w,
    prediction_horizon=5,
    soc_init=0.5,
    soc_final=0.5,
    soc_floor_profile=[0.5, 0.6, 0.6, None, None],
)
```

If a reserve is needed when an outage starts, apply its floor to the step ending at that boundary. During an outage, use `None` to release the extra reserve for critical loads and let the existing physical SOC bounds apply. When the outage starts at the first step, compare the required reserve with measured `soc_init`. An end-of-step constraint cannot change starting SOC.

The solver combines the profile with `soc_target` by applying the greater floor at a shared step. An unreachable profile can make the solve infeasible, for example when charge power is insufficient or battery availability freezes SOC below the floor. Check `optim_status` before using a result. The profile is currently a Python-only planning input. Publication and Home Assistant integration are separate steps.
