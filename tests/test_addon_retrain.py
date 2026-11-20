"""Champion/challenger retraining with recency weights."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import retrain  # noqa: E402

from emhass.load_quantiles import LoadFeatureConfig, LoadQuantileBundle  # noqa: E402


def _audit(days: int, end="2026-11-20 00:00") -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=days * 24, freq="h")
    hours = index.hour.to_numpy()
    return pd.DataFrame(
        {
            "load_kw": 1.0 + 0.5 * np.sin(hours * 2 * np.pi / 24),
            "load_filter": "keep",
            "temp_air_observed_c": 5.0,
        },
        index=index,
    )


@pytest.fixture
def models(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain, "MODELS", tmp_path)
    return tmp_path


def test_recency_weights_half_life():
    index = pd.date_range("2026-01-01", periods=61, freq="D", tz="UTC")
    weights = retrain.recency_weights(index)
    assert weights.iloc[-1] == 1.0
    assert weights.iloc[0] == pytest.approx(0.5)


def test_sample_weight_lets_recent_history_dominate():
    index = pd.date_range("2026-01-01", periods=24 * 20, freq="h", tz="UTC")
    grid = pd.DataFrame({"load_kw": 5.0, "temp_air_observed_c": 5.0}, index=index)
    grid.iloc[len(index) // 2 :, 0] = 1.0  # the household changed half-way
    weights = pd.Series(0.0, index=index)
    weights.iloc[len(index) // 2 :] = 1.0
    bundle = LoadQuantileBundle.fit(grid, LoadFeatureConfig(time_zone="UTC"), sample_weight=weights)
    prediction = bundle.predict(grid.iloc[-48:])
    assert prediction["load_p50_kw"].between(0.8, 1.2).all()


def test_sample_weight_must_be_valid():
    index = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    grid = pd.DataFrame({"load_kw": 1.0, "temp_air_observed_c": 5.0}, index=index)
    with pytest.raises(ValueError, match="sample_weight"):
        LoadQuantileBundle.fit(grid, sample_weight=pd.Series(-1.0, index=index))


def test_fresh_install_promotes_and_trains_on_everything(models):
    audit = _audit(30)
    info = retrain._retrain_load(audit)
    assert info["status"] == "promoted"
    bundle = LoadQuantileBundle.load(models / "load_quantiles_v1")
    assert pd.Timestamp(bundle.trained_until) == audit.index[-1]
    # Out-of-sample offsets came from the challenger's holdout.
    assert bundle.calibrated_from == info["holdout_from"]


def test_holdout_starts_after_champion_training(models):
    retrain._retrain_load(_audit(30))
    later = _audit(36, end="2026-11-23 00:00")  # 3 more days
    info = retrain._retrain_load(later)
    # 72 fresh hours since the champion's training end: just enough.
    assert pd.Timestamp(info["holdout_from"]) == pd.Timestamp("2026-11-20 01:00", tz="UTC")
    assert info["status"] in ("promoted", "kept")
    assert info["champion_score"] is not None


def test_too_little_fresh_data_keeps_champion(models):
    retrain._retrain_load(_audit(30))
    before = (models / "load_quantiles_v1" / "bundle.json").read_text()
    info = retrain._retrain_load(_audit(31, end="2026-11-21 00:00"))
    assert info["status"] == "kept"
    assert (models / "load_quantiles_v1" / "bundle.json").read_text() == before


def test_worse_challenger_keeps_champion(models, monkeypatch):
    retrain._retrain_load(_audit(30))
    before = json.loads((models / "load_quantiles_v1" / "bundle.json").read_text())
    scores = iter([(2.0, 100), (1.0, 100)])  # challenger first, then champion
    monkeypatch.setattr(retrain, "load_holdout_score", lambda *a: next(scores))
    info = retrain._retrain_load(_audit(40, end="2026-11-30 00:00"))
    assert info["status"] == "kept"
    assert (info["challenger_score"], info["champion_score"]) == (2.0, 1.0)
    after = json.loads((models / "load_quantiles_v1" / "bundle.json").read_text())
    assert after["trained_until"] == before["trained_until"]


def test_better_challenger_is_refit_on_all_history(models, monkeypatch):
    retrain._retrain_load(_audit(30))
    scores = iter([(1.0, 100), (2.0, 100)])
    monkeypatch.setattr(retrain, "load_holdout_score", lambda *a: next(scores))
    audit = _audit(40, end="2026-11-30 00:00")
    info = retrain._retrain_load(audit)
    assert info["status"] == "promoted"
    bundle = LoadQuantileBundle.load(models / "load_quantiles_v1")
    assert pd.Timestamp(bundle.trained_until) == audit.index[-1]


def test_load_holdout_score_uses_causal_issues(models):
    retrain._retrain_load(_audit(30))
    bundle = LoadQuantileBundle.load(models / "load_quantiles_v1")
    audit = _audit(33, end="2026-11-23 00:00")
    grid = audit[["temp_air_observed_c", "load_kw"]]
    score, pairs = retrain.load_holdout_score(
        bundle, grid, pd.Timestamp("2026-11-20 01:00", tz="UTC")
    )
    assert score is not None and score >= 0
    assert pairs == 24 * 5  # issues every 12 h: 01:00 and 13:00 for 2.5 days


def test_unreadable_champion_is_replaced(models):
    (models / "load_quantiles_v1").mkdir()
    (models / "load_quantiles_v1" / "bundle.json").write_text("{not json")
    assert retrain._retrain_load(_audit(30))["status"] == "promoted"


def test_retrain_note_reports_kept_and_promoted(monkeypatch, tmp_path):
    import pipeline

    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    state = {
        "last_retrain_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "models": {
            "load": {"status": "promoted", "train_rows": 700},
            "pv": {"status": "kept", "reason": "новая модель не лучше текущей на свежих данных"},
        },
    }
    (tmp_path / "retrain_state.json").write_text(json.dumps(state), encoding="utf-8")
    note = pipeline._retrain_note()
    assert "обновлена (700 строк)" in note and "оставлена текущая" in note
