"""Battery capacity learned only from cycles after the latest pack change."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import hardware  # noqa: E402
import retrain  # noqa: E402

ETA = 0.97


def _history(capacities, seed=0):
    """One night charge and one evening discharge a day, hourly means.

    ``capacities`` holds each day's battery size in kWh: a pack removed on
    day N is ``[48] * N + [32] * rest``.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-09-23", periods=len(capacities) * 24, freq="h", tz="UTC")
    discharge_kw = 3.0 * 5 * ETA * ETA / 6  # takes out what the night put in
    power, soc = [], []
    fraction = 0.3
    for n, stamp in enumerate(index):
        kw = -3.0 if 1 <= stamp.hour < 6 else (discharge_kw if 17 <= stamp.hour < 23 else 0.0)
        kw *= 1 + rng.normal(0, 0.01)
        soc.append(fraction)
        power.append(kw)
        fraction += (-kw * ETA if kw < 0 else -kw / ETA) / capacities[n // 24]
    return pd.DataFrame({"battery_power_kw_raw_sign": power, "soc_mean": soc}, index=index)


def test_removed_pack_learns_the_battery_left():
    info = hardware.identify_battery(_history([48.0] * 38 + [32.0] * 22), capacity_kwh=32.0)
    assert info["status"] == "ok", info["messages"]
    # One fit over both batteries would say ~40 kWh.
    assert info["capacity_kwh"] == pytest.approx(32.0, rel=0.05)
    change = info["battery_change"]
    assert pd.Timestamp(change["since_utc"]).date() == pd.Timestamp("2026-10-31").date()
    assert change["ratio"] == pytest.approx(32 / 48, abs=0.03)
    assert "смена батареи" in info["messages"][0]


def test_added_pack_learns_the_bigger_battery():
    info = hardware.identify_battery(_history([32.0] * 40 + [48.0] * 20), capacity_kwh=48.0)
    assert info["status"] == "ok", info["messages"]
    assert info["capacity_kwh"] == pytest.approx(48.0, rel=0.05)
    assert info["battery_change"]["ratio"] == pytest.approx(1.5, abs=0.06)


def test_change_too_fresh_waits_for_cycles_of_the_new_battery():
    info = hardware.identify_battery(_history([48.0] * 58 + [32.0] * 2), capacity_kwh=32.0)
    assert info["battery_change"] is not None
    assert info["status"] != "ok"  # never the blend of both batteries


@pytest.mark.parametrize("seed", range(5))
def test_one_battery_is_not_a_change(seed):
    info = hardware.identify_battery(_history([32.0] * 60, seed=seed), capacity_kwh=32.0)
    assert info["battery_change"] is None
    assert info["capacity_kwh"] == pytest.approx(32.0, rel=0.05)


def test_odd_cycles_are_not_a_change():
    audit = _history([32.0] * 60)
    # Two BMS resyncs a week apart: the SOC jumps 8 points within a charge.
    for day in (50, 57):
        night = audit.index[day * 24 + 3 : day * 24 + 6]
        audit.loc[night, "soc_mean"] += 0.08
        audit.loc[audit.index[day * 24 + 6] :, "soc_mean"] += 0.08
    info = hardware.identify_battery(audit, capacity_kwh=32.0)
    assert info["battery_change"] is None


def test_outdated_hardware_is_relearned_on_the_next_run(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain, "DATA", tmp_path)
    monkeypatch.setattr(retrain, "STATE", tmp_path / "retrain_state.json")
    monkeypatch.setattr(retrain, "MODELS", tmp_path / "models")
    monkeypatch.setattr(retrain, "_enabled", lambda: False)  # the schedule itself is off
    audit = _history([48.0] * 38 + [32.0] * 22)
    monkeypatch.setattr(retrain, "_audit_frame", lambda: audit)
    opts = {"capacity_kwh": 32.0, "latitude": 45.0, "longitude": 34.0, "time_zone": "UTC"}
    (tmp_path / "options.json").write_text(json.dumps(opts), encoding="utf-8")
    stale = {
        "identified_utc": "2026-11-21T00:00:00+00:00",
        "battery": {"status": "ok", "capacity_kwh": 40.1},
    }
    hardware.save(tmp_path, stale)
    (tmp_path / "retrain_state.json").write_text(
        json.dumps({"pv_model_tried": retrain.pipeline.PV_MODEL}), encoding="utf-8"
    )
    retrain.retrain_if_due()
    learned = hardware.load(tmp_path)
    assert learned["version"] == hardware.HARDWARE_VERSION
    assert learned["battery"]["capacity_kwh"] == pytest.approx(32.0, rel=0.05)
    assert not hardware.outdated(tmp_path)
