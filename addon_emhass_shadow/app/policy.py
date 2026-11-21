"""Outage regime and the owner's priority mapped onto the reserve.

Outages come in series: once the grid starts failing it keeps failing, and
two quiet weeks mean the series is over. The add-on therefore protects only
against outages of a live series:

* **series** — a measured outage ended (or is still open) within the last
  ``outage_memory_days`` (14 by default): the reserve covers the critical
  load for the series' outage duration, raised in the hours the series hit;
* **quiet** — no outage for ``outage_memory_days``: no outage reserve at all
  (only the owner's optional minimum autonomy above min SOC). The first
  outage of a new series switches back immediately.

The priority sets how cautious a live series is handled: which quantile of
the series' outage durations the reserve covers, and which extra margin (in
hours of critical load) the replay tuner may add on top of it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from emhass.reserve import _merge_intervals, _quantile

DEFAULT_MEMORY_DAYS = 14.0

PRIORITIES = {
    "resilience": {"reserve_quantile": 0.95, "margins_h": (1.0, 2.0)},
    "balanced": {"reserve_quantile": 0.9, "margins_h": (0.0, 1.0)},
    "economy": {"reserve_quantile": 0.8, "margins_h": (0.0,)},
}
DEFAULT_PRIORITY = "balanced"


def priority(opts: dict) -> dict:
    return PRIORITIES.get(
        str(opts.get("priority_mode", DEFAULT_PRIORITY)), PRIORITIES[DEFAULT_PRIORITY]
    )


def default_margin(opts: dict) -> float:
    """Margin before the tuner has evidence: the smallest one allowed."""
    return min(priority(opts)["margins_h"])


def memory_days(opts: dict) -> float:
    try:
        return max(1.0, float(opts.get("outage_memory_days", DEFAULT_MEMORY_DAYS)))
    except (TypeError, ValueError):
        return DEFAULT_MEMORY_DAYS


def outage_regime(events, now: datetime, memory: float) -> dict:
    """Is an outage series live at ``now``? Measured events only."""
    measured = [e for e in events if getattr(e, "source", None) == "measured"]
    open_now = any(e.end is None for e in measured)
    ends = [e.end for e in measured if e.end is not None and e.end <= now]
    last_end = max(ends) if ends else None
    cutoff = now - timedelta(days=memory)
    in_window = [e for e in measured if e.end is None or e.end >= cutoff]
    days_since = None if last_end is None else (now - last_end).total_seconds() / 86400
    return {
        "active": open_now or (last_end is not None and last_end >= cutoff),
        "open_now": open_now,
        "last_end_utc": last_end.isoformat() if last_end else None,
        "days_since_last": None if days_since is None else round(days_since, 2),
        "events_in_window": len(in_window),
        "memory_days": memory,
        "series_until_utc": (last_end + timedelta(days=memory)).isoformat() if last_end else None,
    }


def recent_events(events, now: datetime, memory: float) -> list:
    """Events of the live series (measured, ended or open within memory)."""
    cutoff = now - timedelta(days=memory)
    return [e for e in events if e.end is None or e.end >= cutoff]


FLAP_GAP_HOURS = 0.5  # back for less than this: the same outage for the battery
BEYOND_SERIES_HOURS = 1.0  # longer than every outage of the series: look an hour ahead


def outage_in_progress(
    events,
    now: datetime,
    memory: float,
    quantile: float,
    *,
    soc: float,
    min_soc: float,
    capacity_kwh: float,
    discharge_efficiency: float,
    critical_kw: float,
    pv_kw: list[float],
) -> dict | None:
    """The outage going on right now: how long it lasts already, how much
    longer it likely lasts (from the series' own outages that ran past this
    point), how long the battery carries the house with the pessimistic sun,
    and how much load to shed when that is not enough. None when the grid
    is up.

    The inverter discharges to its shutdown SOC during an outage whatever
    the TOU zones say, so the only lever left is the load: the answer is a
    figure for the owner and a signal for their automations.
    """
    cutoff = now - timedelta(days=memory)
    measured = [
        (e.start, e.end or now)
        for e in events
        if getattr(e, "source", None) == "measured"
        and e.start <= now
        and (e.end is None or e.end >= cutoff)
    ]
    if not any(e.end is None for e in events if getattr(e, "source", None) == "measured"):
        return None
    glued = _merge_intervals(measured, timedelta(hours=FLAP_GAP_HOURS))
    if not glued or glued[-1][1] < now:
        return None
    start = glued[-1][0]
    elapsed = (now - start).total_seconds() / 3600
    past = [(end - begin).total_seconds() / 3600 for begin, end in glued[:-1]]
    longer = sorted(d - elapsed for d in past if d > elapsed)
    if len(longer) >= 3:
        typical, cautious, basis = _quantile(longer, 0.5), _quantile(longer, quantile), "series"
    elif longer:
        typical = cautious = max(longer)
        basis = "few"
    else:
        typical = cautious = BEYOND_SERIES_HOURS
        basis = "beyond"

    energy = max(0.0, soc - min_soc) * capacity_kwh * discharge_efficiency
    demand = [max(0.0, critical_kw - float(pv)) for pv in pv_kw]

    def drawn(hours: float) -> float:
        whole = int(hours)
        total = sum(demand[:whole])
        if whole < len(demand):
            total += demand[whole] * (hours - whole)
        return total

    autonomy = float(len(demand))
    used = 0.0
    for hour, kw in enumerate(demand):
        if used + kw >= energy:
            autonomy = hour + ((energy - used) / kw if kw > 0 else 0.0)
            break
        used += kw
    shortfall = max(0.0, drawn(cautious) - energy)
    return {
        "since_utc": start.isoformat(),
        "elapsed_h": round(elapsed, 2),
        "remaining_typical_h": round(typical, 2),
        "remaining_h": round(cautious, 2),
        "basis": basis,
        "past_outages_longer": len(longer),
        "autonomy_h": round(autonomy, 2),
        "enough": shortfall <= 0.0,
        "shortfall_kwh": round(shortfall, 2),
        "cut_kw": round(shortfall / cautious, 2) if shortfall > 0 else 0.0,
    }
