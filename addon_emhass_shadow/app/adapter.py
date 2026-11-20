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
configured, and only in apply mode; switch, input_boolean or select such
as Deye's ``select.inverter_program_{n}_charging``): a zone may charge from
the grid when

* the zone is entered below the outage reserve it protects (reserve first,
  at any price; the reserve is zero outside an outage series), or
* a night tariff is enabled and the whole zone lies in cheap hours — how
  much to buy there is the optimizer's zone target.

With one price all day the grid never charges the battery otherwise: a
charge/discharge round trip loses ~10 % and buys nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import optimize
import pandas as pd


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


OFF_WORDS = ("disabled", "disable", "off", "none", "no", "выкл", "нет", "отключено")


def select_option(
    desired: bool,
    current: str | None,
    options: list[str],
    on_override: str = "",
    off_override: str = "",
) -> str | None:
    """Option of a grid-charge ``select`` that means "grid charge on/off".

    Explicit overrides win when they are valid options. Otherwise the names
    are recognised: "Grid" for on, "Disabled"/"Off" for off. A zone that
    also charges from a generator keeps it — Deye names that "Both" (grid +
    generator), other integrations "Grid & Gen" — so toggling grid charging
    never silently drops generator charging. None when no option fits (the
    caller reports it instead of guessing).
    """
    options = [str(o) for o in options or []]
    override = on_override if desired else off_override
    if override:
        return override if override in options else None
    low = {o: o.lower().strip() for o in options}

    def grid_and_gen(name: str) -> bool:
        return name == "both" or ("grid" in name and "gen" in name)

    def gen_only(name: str) -> bool:
        return "gen" in name and not grid_and_gen(name) and "grid" not in name

    now = str(current or "").lower().strip()
    with_gen = grid_and_gen(now) or gen_only(now)
    if desired:
        if with_gen:
            both = [o for o in options if grid_and_gen(low[o])]
            if both:
                return both[0]
        grid = [
            o
            for o in options
            if ("grid" in low[o] or "сеть" in low[o]) and not grid_and_gen(low[o])
        ]
        exact = [o for o in grid if low[o] in ("grid", "сеть")]
        return (exact or grid or [None])[0]
    if with_gen:
        gen = [o for o in options if gen_only(low[o])]
        if gen:
            return gen[0]
    off = [o for o in options if low[o].strip() in OFF_WORDS]
    return off[0] if off else None


def flag_state(item: dict, opts: dict) -> str | None:
    """State to write for a grid-charge entity, by its domain; None = no fit."""
    domain = item["entity"].split(".", 1)[0]
    if domain in ("switch", "input_boolean"):
        return "on" if item["desired"] else "off"
    if domain == "select":
        return select_option(
            item["desired"],
            item.get("current"),
            item.get("options") or [],
            str(opts.get("tou_grid_charge_on_option") or ""),
            str(opts.get("tou_grid_charge_off_option") or ""),
        )
    return None


def flag_service(entity: str, state: str) -> tuple[str, str, dict]:
    """(domain, service, payload) that sets the grid-charge entity."""
    domain = entity.split(".", 1)[0]
    if domain == "select":
        return "select", "select_option", {"entity_id": entity, "option": state}
    return domain, f"turn_{state}", {"entity_id": entity}


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
    pv_p10_kw: list[float] | None,
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
            desired, reason = False, "цена одинаковая — заряд только от солнца"
        plan.append({"n": program["n"], "desired": desired, "reason": reason})
    return plan
