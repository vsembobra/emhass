"""What the inverter can be told: TOU zones, SOC step, per-zone grid charge.

The add-on drives inverters that schedule the battery with time-of-use
programs (Deye/Sunsynk and other TOU-programmable hybrids): N zones, each
with a start time and a Batt% that is both the discharge floor and — when
the zone's grid-charge flag is on — the grid-charge target. Everything
brand-specific is a capability read from the options, so a different TOU
inverter needs only its entity templates, zone count and SOC step.

Inverters steered by power setpoints instead of TOU zones (e.g. Victron
ESS) are out of scope of this adapter.

Grid-charge policy (only when a per-zone grid-charge entity template is
configured, and only in apply mode): a zone may charge from the grid when

* the plan reaches the zone below the outage reserve the zone protects
  (reserve first, at any price), or
* with a tariff: the whole zone lies in cheap hours; without a tariff: the
  zone expects no sun (pessimistic PV) — grid-charging a sunny zone buys
  energy the panels would deliver for free and leads to curtailment later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import optimize
import pandas as pd

NO_SUN_KWH = 0.2


@dataclass(frozen=True)
class InverterCaps:
    zones: int = 6
    soc_step_pct: int = 5
    grid_charge_template: str = ""

    @property
    def soc_step(self) -> float:
        return self.soc_step_pct / 100.0

    @property
    def controls_grid_charge(self) -> bool:
        return bool(self.grid_charge_template.strip())

    def ceil_soc(self, soc: float) -> float:
        """Round a SOC fraction UP to the inverter's step (never shave reserve)."""
        return math.ceil(soc / self.soc_step - 1e-9) * self.soc_step

    def grid_charge_entities(self) -> list[str]:
        if not self.controls_grid_charge:
            return []
        return [self.grid_charge_template.format(n=n) for n in range(1, self.zones + 1)]


def caps(opts: dict) -> InverterCaps:
    return InverterCaps(
        zones=max(1, int(opts.get("tou_zones", 6))),
        soc_step_pct=max(1, int(opts.get("tou_soc_step_pct", 5))),
        grid_charge_template=str(opts.get("tou_grid_charge_entity_template") or ""),
    )


def grid_charge_plan(
    programs: list[dict],
    starts: pd.DatetimeIndex,
    tz,
    opts: dict,
    reserve_floors: list[float],
    pv_p10_kw: list[float],
    soc_opt: list[float] | None,
    soc_init: float,
) -> list[dict]:
    """Desired grid-charge flag per zone with the reason, first day only."""
    windows = optimize.zone_windows(programs, starts, tz)
    cheap = optimize._cheap_windows(opts) if opts.get("tariff_enabled", False) else None
    plan = []
    for program in programs:
        zone = windows[program["n"]]
        if zone is None:
            continue
        # The flag acts first on the hours the zone governs now: for the zone
        # in effect, its remaining hours entered at the MEASURED SOC.
        window, in_effect = zone.now
        entry_soc = soc_init if in_effect or not soc_opt else soc_opt[zone.match - 1]
        reserve_pct = optimize.reserve_floor_pct(reserve_floors, window)
        if reserve_pct > round(entry_soc * 100.0):
            desired, reason = True, f"резерв {reserve_pct}% выше плана на входе"
        elif cheap is not None:
            local = [starts[i].tz_convert(tz) for i in window]
            in_cheap = all(optimize._is_cheap_minute(t.hour * 60 + t.minute, cheap) for t in local)
            desired, reason = in_cheap, "дешёвый тариф" if in_cheap else "дорогой тариф"
        else:
            sun = sum(max(0.0, float(pv_p10_kw[i])) for i in window)
            desired = sun < NO_SUN_KWH
            reason = "солнца нет" if desired else "солнце зарядит"
        plan.append({"n": program["n"], "desired": desired, "reason": reason})
    return plan
