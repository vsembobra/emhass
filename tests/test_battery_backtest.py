from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest import TestCase
from zoneinfo import ZoneInfo

from emhass.battery_backtest import Battery, Dispatch, ReplayStep, simulate

START = datetime(2025, 1, 1, tzinfo=UTC)
BATTERY = Battery(10, 0.1, 0.9, 3, 3, 1, 1)


def _step(hour, pv_kw=0, load_kw=0, critical_load_kw=0, grid_up=True, **overrides):
    return ReplayStep(
        start=START + timedelta(hours=hour),
        hours=overrides.pop("hours", 1),
        pv_kw=pv_kw,
        load_kw=load_kw,
        critical_load_kw=critical_load_kw,
        grid_up=grid_up,
        **overrides,
    )


class TestBatteryBacktest(TestCase):
    def test_grid_up_respects_floor_and_outage_can_spend_the_reserve(self):
        steps = [_step(0, load_kw=3), _step(1, load_kw=4, critical_load_kw=3, grid_up=False)]

        reserved = simulate(steps, BATTERY, 0.6, lambda start, soc: Dispatch(0.5))
        baseline = simulate(steps, BATTERY, 0.6, lambda start, soc: Dispatch(0.2))

        self.assertEqual(reserved.imported_kwh, 2)
        self.assertEqual(reserved.unserved_critical_kwh, 0)
        self.assertEqual(reserved.unserved_critical_hours, 0)
        self.assertEqual(reserved.outage_hours, 1)
        self.assertAlmostEqual(reserved.final_soc, 0.2)
        self.assertEqual(baseline.imported_kwh, 0)
        self.assertEqual(baseline.unserved_critical_kwh, 1)
        self.assertEqual(baseline.unserved_critical_hours, 1)
        self.assertAlmostEqual(baseline.final_soc, 0.1)

    def test_solar_kept_for_site_load_excludes_losses_export_and_initial_soc(self):
        battery = replace(
            BATTERY,
            max_charge_kw=4,
            charge_efficiency=0.5,
            discharge_efficiency=0.5,
            export_limit_kw=0.5,
        )
        steps = [_step(0, pv_kw=7, load_kw=1), _step(1, load_kw=4)]

        result = simulate(steps, battery, 0.2, lambda start, soc: Dispatch(0.2))

        self.assertEqual(result.imported_kwh, 3)
        self.assertEqual(result.exported_kwh, 0.5)
        self.assertEqual(result.curtailed_kwh, 1.5)
        self.assertEqual(result.own_solar_kwh, 1.5)
        self.assertAlmostEqual(result.final_soc, 0.2)

    def test_plan_can_grid_charge_and_export_within_site_limits(self):
        battery = replace(BATTERY, import_limit_kw=2, export_limit_kw=1)
        steps = [
            _step(0, load_kw=1, import_price=-0.25, export_price=-0.1),
            _step(1, load_kw=0.5, import_price=-0.25, export_price=-0.1),
        ]

        def policy(start, soc):
            return Dispatch(0.1, -3 if start == START else 3)

        result = simulate(steps, battery, 0.2, policy)

        self.assertEqual(result.imported_kwh, 2)
        self.assertEqual(result.exported_kwh, 1)
        self.assertEqual(result.own_solar_kwh, 0)
        self.assertAlmostEqual(result.net_cost, -0.4)
        self.assertAlmostEqual(result.final_soc, 0.15)

    def test_plan_does_not_export_battery_energy_if_present_pv_fills_export_limit(self):
        battery = replace(BATTERY, export_limit_kw=1)
        result = simulate([_step(0, pv_kw=4, load_kw=2)], battery, 0.6, lambda *_: Dispatch(0.1, 3))

        self.assertEqual(result.exported_kwh, 1)
        self.assertEqual(result.curtailed_kwh, 1)
        self.assertEqual(result.final_soc, 0.6)

    def test_plan_keeps_exported_solar_origin_discharge_out_of_own_solar(self):
        steps = [_step(0, pv_kw=3, load_kw=1), _step(1, load_kw=1)]

        def policy(start, soc):
            return Dispatch(0.1, -3 if start == START else 3)

        result = simulate(steps, BATTERY, 0.1, policy)

        self.assertEqual(result.imported_kwh, 1)
        self.assertEqual(result.exported_kwh, 2)
        self.assertEqual(result.own_solar_kwh, 1.5)
        self.assertEqual(result.final_soc, 0.1)

    def test_island_dispatch_can_charge_from_pv_and_reserves_discharge_for_critical_load(self):
        battery = replace(BATTERY, max_discharge_kw=1)
        steps = [
            _step(0, pv_kw=4, load_kw=2, critical_load_kw=1, grid_up=False),
            _step(1, load_kw=3, critical_load_kw=1, grid_up=False),
            _step(2, load_kw=4, critical_load_kw=3, grid_up=False, hours=0.5),
        ]

        result = simulate(steps, battery, 0.2, lambda *_: Dispatch(0.9, -3))

        self.assertEqual(result.imported_kwh, 0)
        self.assertEqual(result.exported_kwh, 0)
        self.assertEqual(result.unserved_critical_kwh, 1)
        self.assertEqual(result.unserved_critical_hours, 0.5)
        self.assertEqual(result.outage_hours, 2.5)
        self.assertEqual(result.own_solar_kwh, 2.75)
        self.assertAlmostEqual(result.final_soc, 0.25)
        self.assertIsNone(result.net_cost)

    def test_grid_connected_import_limit_prioritizes_load_over_charging(self):
        battery = replace(BATTERY, import_limit_kw=1)
        result = simulate(
            [_step(0, load_kw=3, critical_load_kw=2)], battery, 0.1, lambda *_: Dispatch(0.7, -3)
        )

        self.assertEqual(result.imported_kwh, 1)
        self.assertEqual(result.unserved_critical_kwh, 0)
        self.assertEqual(result.unserved_critical_hours, 0)
        self.assertEqual(result.outage_hours, 0)
        self.assertEqual(result.final_soc, 0.1)

    def test_roundoff_at_discharge_boundary_does_not_count_as_unserved(self):
        battery = replace(BATTERY, min_soc=0.2, discharge_efficiency=0.7)
        step = _step(0, load_kw=2.1, critical_load_kw=2.1, grid_up=False)

        result = simulate([step], battery, 0.5, lambda *_: Dispatch(0.2))

        self.assertEqual(result.unserved_critical_kwh, 0)
        self.assertEqual(result.unserved_critical_hours, 0)

    def test_shortfall_larger_than_machine_precision_counts_as_unserved(self):
        battery = replace(BATTERY, min_soc=0.2, discharge_efficiency=0.7)
        step = _step(0, load_kw=2.100000000001, critical_load_kw=2.100000000001, grid_up=False)

        result = simulate([step], battery, 0.5, lambda *_: Dispatch(0.2))

        self.assertGreater(result.unserved_critical_kwh, 0)
        self.assertEqual(result.unserved_critical_hours, 1)

    def test_raised_floor_waits_for_solar_and_charge_clips_to_max_soc(self):
        steps = [_step(0, load_kw=2), _step(1, pv_kw=4), _step(2, load_kw=4)]
        battery = replace(BATTERY, max_soc=0.5, export_limit_kw=0)

        def policy(start, soc):
            return Dispatch(0.5 if start == START else 0.4)

        result = simulate(steps, battery, 0.4, policy)

        self.assertEqual(result.imported_kwh, 5)
        self.assertEqual(result.curtailed_kwh, 3)
        self.assertEqual(result.exported_kwh, 0)
        self.assertAlmostEqual(result.final_soc, 0.4)

    def test_prices_can_be_missing_for_unused_directions_only(self):
        imported = _step(0, load_kw=1, import_price=0.25)
        exported = _step(1, pv_kw=1)

        def idle(start, soc):
            return Dispatch(0.1, 0)

        self.assertAlmostEqual(simulate([imported], BATTERY, 0.1, idle).net_cost, 0.25)
        self.assertIsNone(simulate([imported, exported], BATTERY, 0.1, idle).net_cost)

    def test_policy_receives_distinct_utc_instants_and_starting_soc_on_fall_back(self):
        zone = ZoneInfo("America/New_York")
        first = datetime(2025, 11, 2, 1, tzinfo=zone)
        second = first.replace(fold=1)
        steps = [
            replace(_step(0, pv_kw=1), start=first),
            replace(_step(1, load_kw=2), start=second),
        ]
        decisions = []

        def policy(start, soc):
            decisions.append((start, soc))
            return Dispatch(0.1)

        simulate(iter(steps), BATTERY, 0.2, policy)

        self.assertEqual(decisions, [(first.astimezone(UTC), 0.2), (second.astimezone(UTC), 0.3)])
        self.assertTrue(all(start.tzinfo is UTC for start, soc in decisions))

    def test_invalid_inputs_raise_value_error(self):
        invalid_cases = [
            ([], BATTERY, 0.1, Dispatch(0.1)),
            ([_step(0), _step(2)], BATTERY, 0.1, Dispatch(0.1)),
            ([_step(0), _step(0)], BATTERY, 0.1, Dispatch(0.1)),
            ([replace(_step(0), start=START.replace(tzinfo=None))], BATTERY, 0.1, Dispatch(0.1)),
            ([_step(0, hours=0)], BATTERY, 0.1, Dispatch(0.1)),
            ([_step(0, hours=float("inf"))], BATTERY, 0.1, Dispatch(0.1)),
            ([_step(0, pv_kw=float("nan"))], BATTERY, 0.1, Dispatch(0.1)),
            ([_step(0, load_kw=-1)], BATTERY, 0.1, Dispatch(0.1)),
            ([_step(0, critical_load_kw=1)], BATTERY, 0.1, Dispatch(0.1)),
            ([_step(0, grid_up=1)], BATTERY, 0.1, Dispatch(0.1)),
            ([_step(0, import_price=True)], BATTERY, 0.1, Dispatch(0.1)),
            ([_step(0)], BATTERY, 0.01, Dispatch(0.1)),
            ([_step(0)], BATTERY, 0.1, Dispatch(1)),
            ([_step(0)], BATTERY, 0.1, Dispatch(0.1, float("inf"))),
            ([_step(0)], replace(BATTERY, capacity_kwh=0), 0.1, Dispatch(0.1)),
            ([_step(0)], replace(BATTERY, charge_efficiency=1.1), 0.1, Dispatch(0.1)),
            ([_step(0)], replace(BATTERY, discharge_efficiency=0), 0.1, Dispatch(0.1)),
            ([_step(0)], replace(BATTERY, export_limit_kw=-1), 0.1, Dispatch(0.1)),
            ([_step(0)], replace(BATTERY, min_soc=0.9, max_soc=0.8), 0.1, Dispatch(0.1)),
        ]
        for steps, battery, soc, dispatch in invalid_cases:
            with self.subTest(steps=steps, battery=battery, soc=soc, dispatch=dispatch):
                with self.assertRaises(ValueError):
                    simulate(steps, battery, soc, lambda *_, dispatch=dispatch: dispatch)
