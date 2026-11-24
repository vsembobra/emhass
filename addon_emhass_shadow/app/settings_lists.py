"""Validation for structured add-on settings edited in the panel."""

import math
import re

_WINDOW = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d-(?:(?:[01]\d|2[0-3]):[0-5]\d|24:00)$")


def deferrable_loads(value, lang: str = "ru") -> list[dict]:
    label = "Flexible loads" if lang == "en" else "Переносимые нагрузки"
    if not isinstance(value, list):
        raise ValueError(f"{label}: " + ("expected a list" if lang == "en" else "ожидается список"))
    result = []
    for index, row in enumerate(value, start=1):
        error = f"{label}, {index}: "
        if (
            not isinstance(row, dict)
            or not {"name", "power_kw", "hours"}.issubset(row)
            or set(row) - {"name", "power_kw", "hours", "window"}
            or not isinstance(row["name"], str)
        ):
            raise ValueError(
                error + ("invalid fields" if lang == "en" else "некорректные поля нагрузки")
            )
        item = {"name": row["name"].strip()}
        for key, low, high in (("power_kw", 0.1, 30.0), ("hours", 0.5, 12.0)):
            try:
                if isinstance(row[key], bool):
                    raise ValueError
                number = float(str(row[key]).replace(",", "."))
            except (ValueError, TypeError):
                number = math.nan
            if not math.isfinite(number) or not low <= number <= high:
                raise ValueError(error + f"{key}: {low}–{high}")
            item[key] = number
        window = row.get("window", "")
        if (
            not isinstance(window, str)
            or window.strip()
            and not _WINDOW.fullmatch(window.replace(" ", ""))
        ):
            raise ValueError(
                error
                + (
                    "time window must be HH:MM-HH:MM"
                    if lang == "en"
                    else "окно времени — ЧЧ:ММ-ЧЧ:ММ"
                )
            )
        if window.strip():
            item["window"] = window.replace(" ", "")
        result.append(item)
    return result
