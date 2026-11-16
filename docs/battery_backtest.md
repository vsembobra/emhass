# Single-battery advisory backtest

Run the offline replay from a Python environment with this checkout installed.

```console
python -m emhass.battery_replay site-replay.json
python -m emhass.battery_replay site-replay.json --outage-db path/to/outages.sqlite --site home
```

Use contiguous intervals with explicit UTC offsets (one hour per row for the first site). Put grid transitions at interval boundaries by splitting intervals where the data supports it. Supply estimated **available** AC PV, total load demand and critical load demand. Production suppressed by export limits and consumption suppressed by a power cut cannot be inferred from those meters alone. During an outage, `pv_kw` means AC power available on the islanded bus. Set it to zero for PV that shuts down or is on a different bus. `initial_soc` should be the observed SOC at the first interval start. Set `fixed_floor_soc` to the site's current policy floor. Import/export prices are optional energy rates in the same currency per kWh, and either price can be negative.

The input is a JSON object. This small synthetic dataset illustrates the shape and uses a `profit` dispatch export. Setpoints in a real replay should come from the chosen existing EMHASS objective, with their actual first-available timestamps.

```json
{
  "reserve": {
    "nominal_capacity_kwh": 10,
    "soh": 1.0,
    "floor_soc": 0.1,
    "max_soc": 0.9,
    "discharge_efficiency": 1.0,
    "cold_start_kw": 1.0
  },
  "battery": {
    "max_charge_kw": 5,
    "max_discharge_kw": 5,
    "charge_efficiency": 1.0,
    "export_limit_kw": 0
  },
  "initial_soc": 0.5,
  "fixed_floor_soc": 0.2,
  "actuals": [
    {"start": "2026-10-19T06:00:00Z", "hours": 1, "pv_kw": 0, "load_kw": 3,
     "critical_load_kw": 3, "grid_up": true, "import_price": 0.3},
    {"start": "2026-10-19T07:00:00Z", "hours": 1, "pv_kw": 0, "load_kw": 3,
     "critical_load_kw": 3, "grid_up": false}
  ],
  "forecasts": [
    {"available_at": "2026-10-19T05:45:00Z", "steps": [
      {"start": "2026-10-19T06:00:00Z", "hours": 1, "pv_p10_kw": 3, "load_p90_kw": 3},
      {"start": "2026-10-19T07:00:00Z", "hours": 1, "pv_p10_kw": 0, "load_p90_kw": 3}
    ]}
  ],
  "emhass_plan": {"objective": "profit", "steps": [
    {"available_at": "2026-10-19T05:50:00Z", "start": "2026-10-19T06:00:00Z",
     "p_batt_w": -1000, "optim_status": "Optimal"}
  ]}
}
```

- `reserve` accepts the settings documented in the [reserve guide](battery_reserve.md). `battery` takes AC power limits and charge efficiency, and accepts optional `import_limit_kw` and `export_limit_kw`. Use an explicit zero for a disabled export connection.
- `forecasts` contains archived power outlooks. The rule uses the most recent run received by each decision and integrates positive `load_p90_kw − pv_p10_kw` before the first PV-positive interval when PV meets load. If none qualifies, it integrates to the end of the forecast. When using existing point forecasts as proxies, record that choice with the backtest. The raw [Open-Meteo snapshots](weather_snapshot.md) are weather inputs and need a separate conversion to PV power. When a run is missing, the replay uses the higher of the fixed floor and the calculated reserve and counts that fallback.
- The replay derives measured outage windows from `grid_up`, or uses the measured log with `--outage-db` and `--site`. To supply pre-replay critical-load observations, add `outage_load` entries with `start`, `kw` and optional `end`. Add `scheduled` entries with `available_at`, `start`, `end` and optional `cancelled_at` for schedules known during the replay. Use originals with their announcement times. The replay simulates pre-charge subject to plant limits, counts targets missed at aligned deadlines, and releases the scheduled target at outage onset. Incomplete or interrupted outage durations are excluded from the reserve statistic.
- To evaluate an existing EMHASS objective, use `emhass_plan.objective` and archive successful plan rows with UTC `start` (the EMHASS result `timestamp`), `available_at` (when the row became available), `p_batt_w` (the original `P_batt`), and `optim_status`. Output power is positive for discharge. An archived `/api/v1/plan` snapshot can supply rows and a generation time. Supply the actual receipt time when later. The simulation uses the latest usable setpoint per step and physically clips it by SOC, equipment and grid limits. Missing rows use the fixed-floor dispatch, and partial coverage yields no objective ranking. Setpoints never override islanded critical-load priority. This replay evaluates the site consequences of the selected EMHASS dispatch. It does not reevaluate that solver's objective terms or rerun its optimization.

## Interpreting the result

Every run reports imported, exported, curtailed and own-solar kWh, unserved critical-load kWh and affected interval hours during outages, energy-only net cost, final SOC, and total outage hours. Own solar includes PV used directly on site and the tracked PV share of battery discharge used on site. The model credits the initial inventory with no known PV energy. If a required price is missing, the bill is `null`. The comparison ranks unserved critical energy first, affected duration second within that tier, then own solar, then net cost when available. With zero outage exposure, the ranking is `null`. Compare final SOC as well when evaluating cost and energy over short runs.

`hindsight` reruns the same forecast-based rule with actual PV and load substituted on forecast horizons fully covered by the actuals. It keeps the same reserve, outage information, equipment and forecast timing. `hindsight_steps` shows the number of decisions with a matched horizon. Compare the reported metrics with `baseline` to estimate the effect of forecast quality, alongside the gaps to `fixed` and `objective_replay`. For the example, unserved critical energy is 2 kWh for `fixed`, 1 kWh for `baseline` and 0 kWh for `hindsight`. The replay ranks the rule ahead of the fixed floor because it served more critical load. That synthetic result confirms the comparison works. Deciding whether forecast error is the main loss requires representative site telemetry, sufficient outage exposure, and meaningful forecast and objective coverage.

## Checks

```console
python -m pytest tests/test_battery_reserve.py tests/test_battery_planning.py \
  tests/test_battery_backtest.py tests/test_battery_replay.py \
  tests/test_outage_log.py tests/test_weather_snapshot.py
```
