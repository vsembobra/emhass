"""User-facing priorities mapped onto the planner's knobs.

The owner states one thing — how much risk of an empty battery during an
outage is acceptable — and the add-on derives the rest:

* ``resilience``: P95 outage duration, the to-sunrise deficit always sized
  on pessimistic forecasts (load P90, PV P10);
* ``balanced`` (default, the pre-existing behaviour): P90 duration, the
  deficit may be sized on the midpoint quantiles when the replay tuner
  proves it costs no outage energy;
* ``economy``: P80 duration, the deficit may even use the expected (P50)
  forecasts.

The risk appetite bounds the tuner: it may only pick deficit modes the
chosen priority allows, so it can never trade outage safety for money
beyond what the owner accepted.
"""

from __future__ import annotations

import numpy as np

DEFICIT_MODES = ("conservative", "moderate", "expected")

PRIORITIES = {
    "resilience": {"reserve_quantile": 0.95, "deficit_modes": ("conservative",)},
    "balanced": {"reserve_quantile": 0.9, "deficit_modes": ("conservative", "moderate")},
    "economy": {"reserve_quantile": 0.8, "deficit_modes": DEFICIT_MODES},
}
DEFAULT_PRIORITY = "balanced"


def priority(opts: dict) -> dict:
    return PRIORITIES.get(
        str(opts.get("priority_mode", DEFAULT_PRIORITY)), PRIORITIES[DEFAULT_PRIORITY]
    )


def default_deficit_mode(opts: dict) -> str:
    """Mode used before the tuner has evidence: the safest one allowed,
    except economy, which starts from the midpoint."""
    return "moderate" if opts.get("priority_mode") == "economy" else "conservative"


def deficit_inputs(mode: str, load: dict, pv: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load and PV series the to-sunrise deficit is sized on.

    ``load``/``pv`` map "p10"/"p50"/"p90" to equal-length sequences in kW.
    """
    l50, l90 = np.asarray(load["p50"], float), np.asarray(load["p90"], float)
    p10, p50 = np.asarray(pv["p10"], float), np.asarray(pv["p50"], float)
    if mode == "expected":
        return l50, p50
    if mode == "moderate":
        return (l50 + l90) / 2.0, (p10 + p50) / 2.0
    if mode == "conservative":
        return l90, p10
    raise ValueError(f"unknown deficit mode {mode!r}")
