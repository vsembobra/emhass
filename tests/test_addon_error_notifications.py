"""Error notifications: stable ids, self-clearing, never break the pipeline."""

import sys
from pathlib import Path

ADDON = Path(__file__).resolve().parent.parent / "addon_emhass_shadow"
sys.path.insert(0, str(ADDON / "src"))
sys.path.insert(0, str(ADDON / "app"))

import main  # noqa: E402
import pipeline  # noqa: E402


def test_notify_problem_create_then_dismiss(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "ha_service", lambda d, s, p: calls.append((s, p)))

    pipeline.notify_problem("weather", "Заголовок", "текст ошибки")
    pipeline.notify_problem("weather", "", None)

    assert calls[0] == (
        "create",
        {
            "title": "Заголовок",
            "message": "текст ошибки",
            "notification_id": "emhass_shadow_problem_weather",
        },
    )
    assert calls[1] == ("dismiss", {"notification_id": "emhass_shadow_problem_weather"})


def test_notify_problem_never_raises(monkeypatch):
    def boom(*args):
        raise RuntimeError("HA unavailable")

    monkeypatch.setattr(pipeline, "ha_service", boom)
    pipeline.notify_problem("weather", "t", "m")
    pipeline.notify_problem("weather", "", None)


def test_report_retrain_problems_raise_and_clear(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main.pipeline, "notify_problem", lambda k, t, m: calls.append((k, t, m))
    )

    main._report_retrain_problems(
        {
            "models": {
                "load": {"status": "kept"},
                "pv": {"status": "failed", "error": "Network is unreachable"},
            }
        }
    )
    assert calls[0][0] == "retrain"
    assert "солнце" in calls[0][2] and "Network is unreachable" in calls[0][2]
    assert "нагрузка" not in calls[0][2]

    main._report_retrain_problems({"models": {"pv": {"status": "promoted"}}})
    assert calls[1] == ("retrain", "", None)


def test_report_retrain_problems_tolerates_empty(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main.pipeline, "notify_problem", lambda k, t, m: calls.append((k, t, m))
    )
    main._report_retrain_problems({})
    main._report_retrain_problems(None)
    assert calls == [("retrain", "", None), ("retrain", "", None)]
