"""Open-Meteo model selection and model-dependent storage (Auto is legacy)."""

import json
from pathlib import Path

MODEL_IDS = {
    "auto": None,
    "ecmwf": "ecmwf_ifs",
    "icon": "icon_seamless",
    "gfs": "gfs_seamless",
}


def selected(options: dict | None = None) -> str:
    model = (options or {}).get("open_meteo_model", "auto")
    if not isinstance(model, str) or model not in MODEL_IDS:
        raise ValueError("open_meteo_model: expected auto, ecmwf, icon or gfs")
    return model


def parameters(options: dict) -> dict:
    model_id = MODEL_IDS[selected(options)]
    return {"models": model_id} if model_id else {}


def from_data(data: Path) -> str:
    try:
        return selected(json.loads((data / "options.json").read_text(encoding="utf-8")))
    except FileNotFoundError:
        return "auto"


def scoped(path: Path, model: str) -> Path:
    selected({"open_meteo_model": model})
    return path if model == "auto" else path.parent / "weather_models" / model / path.name
