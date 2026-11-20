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
