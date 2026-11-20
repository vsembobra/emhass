"""PV-aware reserve with a fractional outage duration."""

import sys
from pathlib import Path

import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))

from emhass.reserve import pv_aware_reserve  # noqa: E402


def test_fractional_outage_keeps_the_real_load_power():
    # Review case: 3.1 h outage, 1 kW critical load, sun after three hours.
    pv = [0, 0, 0, 5, 5, 5] + [0] * 18
    reserve = pv_aware_reserve([3.1] * 24, pv, 3.1, 1.0, 0.0)
    assert reserve[0] == pytest.approx(3.0)  # was 2.325 (3.1 kWh spread over 4 h)


def test_partial_last_hour_counts_only_its_share():
    reserve = pv_aware_reserve([3.1] * 24, [0.0] * 24, 3.1, 1.0, 0.0)
    assert reserve[0] == pytest.approx(3.1)


def test_whole_hour_durations_are_unchanged():
    pv = [0, 0, 2.0, 0] + [0] * 20
    # 1 kW for 3 h, sun covers the third hour: worst deficit is the first two.
    assert pv_aware_reserve([3.0] * 24, pv, 3.0, 1.0, 0.0)[0] == pytest.approx(2.0)


def test_efficiency_and_baseline():
    reserve = pv_aware_reserve([3.1 / 0.9] * 24, [0.0] * 24, 3.1, 0.9, 5.0)
    assert reserve[0] == pytest.approx(5.0)  # the baseline floor wins
    reserve = pv_aware_reserve([3.1 / 0.9] * 24, [0.0] * 24, 3.1, 0.9, 0.0)
    assert reserve[0] == pytest.approx(3.1 / 0.9)


def test_sub_hour_outage():
    assert pv_aware_reserve([0.5] * 24, [0.0] * 24, 0.5, 1.0, 0.0)[0] == pytest.approx(0.5)
