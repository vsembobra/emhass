"""Option metadata for the shadow panel settings card.

Every add-on option that can be edited from the panel is described here:
input type, Russian label and hint (mirrors config.yaml translations),
range/choices for validation. The panel renders inputs from this metadata
and update_options() validates edits against it, so a bad value is rejected
with a readable message instead of breaking the Supervisor schema check.

deferrable_loads stays add-on-config only: it is a list of objects and
does not fit a flat input row.
"""

import re

# key -> metadata. Types: float, int, bool, str, select, time (HH:MM str).
OPTION_META = {
    # --- Site -------------------------------------------------------------
    "latitude": {
        "type": "float", "min": -90, "max": 90, "step": 0.00001,
        "label": "Широта",
        "hint": "Координата объекта, например 44.94892. Нужна для расчёта высоты солнца и погоды.",
    },
    "longitude": {
        "type": "float", "min": -180, "max": 180, "step": 0.00001,
        "label": "Долгота",
        "hint": "Координата объекта, например 34.10887.",
    },
    "time_zone": {
        "type": "str",
        "label": "Часовой пояс",
        "hint": "Имя из базы IANA, например Europe/Istanbul. Все окна и зоны считаются по местному времени.",
    },
    "history_backfill_days": {
        "type": "int", "min": 0, "max": 60, "unit": "дн.",
        "label": "Глубина загрузки истории",
        "hint": "Сколько дней прошлого забирать из истории HA по кнопке «Загрузить историю» (0–60). Реальная глубина ограничена сроком хранения рекордера HA (обычно 10 дней). Данные сразу идут в обучение моделей.",
    },
    "history_statistics_days": {
        "type": "int", "min": 0, "max": 730, "unit": "дн.",
        "label": "Глубина долгосрочной статистики",
        "hint": "Сколько дней почасовой статистики HA подтянуть сверх подробной истории recorder (0–730, по умолчанию 365). С ней модели сразу учатся на сезонах вашего дома. 0 = не использовать.",
    },
    # --- Plant ------------------------------------------------------------
    "capacity_kwh": {
        "type": "float", "min": 1, "max": 200, "step": 0.5, "unit": "кВт·ч",
        "label": "Ёмкость батареи",
        "hint": "Полная номинальная ёмкость АКБ, например 32.",
    },
    "min_soc_pct": {
        "type": "float", "min": 0, "max": 50, "step": 1, "unit": "%",
        "label": "Минимальный SOC",
        "hint": "Абсолютное дно разряда (10 = не разряжать ниже 10%). Резерв отключений может держать пол выше — это лишь жёсткая нижняя граница.",
    },
    "max_soc_pct": {
        "type": "float", "min": 50, "max": 100, "step": 1, "unit": "%",
        "label": "Максимальный SOC",
        "hint": "До скольки процентов разрешено заряжать батарею (100 = полностью).",
    },
    "charge_efficiency": {
        "type": "float", "min": 0.5, "max": 1, "step": 0.01,
        "label": "КПД заряда",
        "hint": "Доля энергии, доходящая до батареи при заряде (0.95 = 95%).",
    },
    "discharge_efficiency": {
        "type": "float", "min": 0.5, "max": 1, "step": 0.01,
        "label": "КПД разряда",
        "hint": "Доля энергии, доходящая до нагрузки при разряде (0.95 = 95%).",
    },
    "max_charge_kw": {
        "type": "float", "min": 0.5, "max": 50, "step": 0.5, "unit": "кВт",
        "label": "Макс. мощность заряда",
        "hint": "Ограничение инвертора на заряд АКБ, например 5.",
    },
    "max_discharge_kw": {
        "type": "float", "min": 0.5, "max": 50, "step": 0.5, "unit": "кВт",
        "label": "Макс. мощность разряда",
        "hint": "Ограничение инвертора на разряд АКБ, например 5.",
    },
    "panel_kwp": {
        "type": "float", "min": 0.5, "max": 100, "step": 0.1, "unit": "кВтп",
        "label": "Мощность панелей",
        "hint": "Суммарная паспортная мощность солнечных панелей. Используется для прогноза солнца, пока модель не переобучилась на ваших данных.",
    },
    "panel_tilt_deg": {
        "type": "float", "min": 0, "max": 90, "step": 1, "unit": "°",
        "label": "Наклон панелей",
        "hint": "0 = лежат горизонтально, 90 = вертикально.",
    },
    "panel_azimuth_deg": {
        "type": "float", "min": 0, "max": 360, "step": 1, "unit": "°",
        "label": "Азимут панелей",
        "hint": "Куда смотрят панели: 180 = строго на юг, 90 = восток, 270 = запад. 15° западнее юга = 195.",
    },
    # --- Plan ---------------------------------------------------------------
    "run_time": {
        "type": "time",
        "label": "Время ежедневного расчёта",
        "hint": "ЧЧ:ММ по местному времени. Полный расчёт и уведомление выполняются раз в день в этот момент.",
    },
    "replan_minutes": {
        "type": "int", "min": 15, "max": 240, "unit": "мин",
        "label": "Пересчёт плана",
        "hint": "Как часто пересчитывать план между ежедневными расчётами (15–240, по умолчанию 120 — раз в два часа). Частый пересчёт держит цели зон на реальном SOC и свежем прогнозе.",
    },
    "apply_mode": {
        "type": "bool",
        "label": "Применять план к инвертору",
        "hint": "ВЫКЛ (рекомендуется на первые 1–2 недели): только советы, в инвертор ничего не пишется. ВКЛ: после расчёта аддон записывает целевые SOC в зоны TOU инвертора и порог SOC в input_number. Записываются только изменения от 5 п.п.",
    },
    "notifications": {
        "type": "bool",
        "label": "Уведомления",
        "hint": "Показывать итог расчёта уведомлением в Home Assistant. Уведомления об ошибках (погода, сбой расчёта или обучения, битые сущности) приходят всегда и снимаются сами.",
    },
    "publish_sensors": {
        "type": "bool",
        "label": "Публиковать сенсоры",
        "hint": "Создавать в HA сенсоры sensor.emhass_shadow_*: рекомендуемый пол SOC, автономия в часах, старт переносимых нагрузок — для ваших автоматизаций. Это не запись в инвертор.",
    },
    "optimizer_enabled": {
        "type": "bool",
        "label": "MILP-оптимизатор",
        "hint": "Суточный оптимальный план заряда/разряда с тарифами и целевыми SOC зон TOU. Выключен = действует только эвристика пола SOC.",
    },
    "day_zone_strategy": {
        "type": "select", "choices": ["ladder", "flat"],
        "label": "Стратегия дневных зон",
        "hint": "flat (по умолчанию): все дорогие зоны — один пол, минимальный SOC или резерв во время серии отключений; батарея тратится днём до этого пола. Ночная зона — цель заряда. ladder (лестница): пол каждой зоны следует почасовому плану — только для тарифа с пиком внутри дня.",
    },
    "priority_mode": {
        "type": "select", "choices": ["resilience", "balanced", "economy"],
        "label": "Приоритет",
        "hint": "resilience — максимум надёжности (резерв под 95% отключений); balanced — как раньше (90%); economy — больше солнца и экономии (80%). Автонастройка выбирает только среди разрешённых приоритетом вариантов.",
    },
    # --- Tariff -------------------------------------------------------------
    "tariff_enabled": {
        "type": "bool",
        "label": "Ночной дешёвый тариф",
        "hint": "Включите, если электричество дешевле в заданные часы — оптимизатор будет заряжать АКБ из сети в дешёвое окно и разряжать в дорогое.",
    },
    "tariff_normal_price": {
        "type": "float", "min": 0, "max": 1000, "step": 0.01,
        "label": "Обычная цена",
        "hint": "Цена кВт·ч вне дешёвых окон, в вашей валюте. Важно соотношение с дешёвой ценой, а не абсолютное значение.",
    },
    "tariff_cheap_price": {
        "type": "float", "min": 0, "max": 1000, "step": 0.01,
        "label": "Дешёвая цена",
        "hint": "Цена кВт·ч внутри дешёвых окон.",
    },
    "tariff_cheap_windows": {
        "type": "str",
        "label": "Окна дешёвого тарифа",
        "hint": "Местное время, ЧЧ:ММ-ЧЧ:ММ; несколько окон через запятую. Окно может переходить через полночь: 23:00-07:00, 13:00-15:00.",
    },
    # --- Reserve ------------------------------------------------------------
    "reserve_cold_start_hours": {
        "type": "float", "min": 0, "max": 20, "step": 0.5, "unit": "ч",
        "label": "Минимальная автономия",
        "hint": "Постоянный запас в часах работы критической нагрузки, который держится даже без отключений (0–20). 0 = в спокойный период держать только минимальный SOC и не защищаться от фантомных отключений.",
    },
    "reserve_pv_aware": {
        "type": "bool",
        "label": "Резерв с учётом солнца",
        "hint": "ВКЛ: в часы, когда пессимистичный прогноз солнца перекрывает критическую нагрузку, резерв на эти часы снижается (ночной не меняется). ВЫКЛ: резерв одинаковый круглые сутки.",
    },
    "outage_memory_days": {
        "type": "int", "min": 3, "max": 60, "unit": "дн.",
        "label": "Память серии отключений",
        "hint": "Отключения идут сериями. Пока последнее отключение было не больше этого числа дней назад, серия считается идущей и резерв держится по её реальной длительности. Дольше тишины — серия закончилась (3–60, по умолчанию 14).",
    },
    "autotune": {
        "type": "bool",
        "label": "Автонастройка по данным",
        "hint": "Применять то, что аддон выучил сам: реальную ёмкость и КПД батареи, ориентацию панелей, профиль критической нагрузки и подобранные на вашей истории настройки. ВЫКЛ = выученное показывается только как совет.",
    },
    # --- TOU ----------------------------------------------------------------
    "tou_zones": {
        "type": "int", "min": 1, "max": 12,
        "label": "Число зон TOU",
        "hint": "Сколько программ (зон) времени использования у инвертора. У Deye — 6. Шаблоны сущностей подставляют номер зоны вместо {n}.",
    },
    "tou_soc_step_pct": {
        "type": "int", "min": 1, "max": 10, "unit": "%",
        "label": "Шаг SOC зон",
        "hint": "С каким шагом инвертор принимает SOC зоны (у Deye — 5). Цели всегда округляются вверх до этого шага, чтобы не срезать резерв.",
    },
    "tou_soc_entity_template": {
        "type": "str",
        "label": "Шаблон сущностей SOC зон",
        "hint": "Шаблон с {n} вместо номера зоны. number.inverter_program_{n}_soc разворачивается в number.inverter_program_1_soc … 6_soc.",
    },
    "tou_time_entity_template": {
        "type": "str",
        "label": "Шаблон сущностей времени зон",
        "hint": "Шаблон с {n} вместо номера зоны. time.inverter_program_{n}_time разворачивается в time.inverter_program_1_time … 6_time.",
    },
    "tou_grid_charge_entity_template": {
        "type": "str",
        "label": "Шаблон флага «заряд из сети»",
        "hint": "Сущность switch или input_boolean, включающая заряд из сети в зоне, с {n} вместо номера. Задано + применение плана = аддон разрешает заряд из сети только там, где он нужен. Пусто = флаги не трогать.",
    },
    "tou_grid_charge_on_option": {
        "type": "str",
        "label": "Вариант select «заряд из сети вкл»",
        "hint": "Только для select-флагов. Пусто = аддон сам выберет вариант со словом Grid (или Grid & Gen, если в зоне включён генератор). Укажите точное название, если варианты называются иначе.",
    },
    "tou_grid_charge_off_option": {
        "type": "str",
        "label": "Вариант select «заряд из сети выкл»",
        "hint": "Только для select-флагов. Пусто = аддон сам выберет Disabled / Off (или Gen, если в зоне включён генератор).",
    },
    # --- Models ---------------------------------------------------------------
    "retrain_enabled": {
        "type": "bool",
        "label": "Автоматическое переобучение",
        "hint": "Модели прогноза дообучаются на данных объекта с заданным интервалом. Кнопка переобучения в панели работает независимо от этого переключателя.",
    },
    "retrain_interval_days": {
        "type": "int", "min": 1, "max": 30, "unit": "дн.",
        "label": "Интервал переобучения",
        "hint": "Как часто модели переобучаются автоматически (1–30).",
    },
    "load_holidays": {
        "type": "str",
        "label": "Праздничные дни",
        "hint": "Ежегодные даты с потреблением как в выходной, через запятую в формате ММ-ДД: 01-01,01-07,03-08. Учитываются со следующего переобучения. Пусто = без праздников.",
    },
    "calibration_charge_enabled": {
        "type": "bool",
        "label": "Калибровочный заряд BMS",
        "hint": "BMS уточняет SOC только при полном заряде; без него ошибка растёт. Если батарея не была на 100% дольше интервала, аддон зарядит её до 100% в ближайшее ночное окно. Полный заряд от солнца сбрасывает отсчёт.",
    },
    "calibration_interval_days": {
        "type": "int", "min": 3, "max": 60, "unit": "дн.",
        "label": "Интервал калибровки",
        "hint": "Через сколько дней без полного заряда устраивать калибровочный заряд до 100% (3–60, по умолчанию 14).",
    },
    # --- Entity bindings (edited in the entities card, hints reused there) ---
    "floor_input_entity": {
        "type": "str", "entity": True,
        "label": "Сущность текущего порога SOC",
        "hint": "input_number или number, где хранится нижний порог SOC инвертора в процентах: input_number.deye_soc_backup или number.inverter_battery_low_soc.",
    },
    "entity_load_power": {
        "type": "str", "entity": True,
        "label": "Мощность нагрузки",
        "hint": "Мощность всех потребителей дома, ватты. Пример: sensor.inverter_load_power. Основной ряд для прогноза нагрузки.",
    },
    "entity_critical_load_power": {
        "type": "str", "entity": True,
        "label": "Критическая нагрузка",
        "hint": "Мощность резервируемых (UPS) потребителей, ватты. По ней считается резерв на случай отключения сети.",
    },
    "entity_pv_power": {
        "type": "str", "entity": True,
        "label": "Мощность солнца",
        "hint": "Мощность солнечных панелей, ватты. Ряд для обучения модели солнечной генерации.",
    },
    "entity_battery_power": {
        "type": "str", "entity": True,
        "label": "Мощность батареи",
        "hint": "Мощность заряда/разряда АКБ, ватты; знак по конвенции вашей интеграции (у Deye: минус = заряд).",
    },
    "entity_grid_power": {
        "type": "str", "entity": True,
        "label": "Мощность сети",
        "hint": "Обмен с сетью, ватты; плюс = потребление из сети.",
    },
    "entity_battery_soc": {
        "type": "str", "entity": True,
        "label": "Заряд батареи",
        "hint": "Текущий уровень заряда АКБ, проценты (0–100). Пример: sensor.inverter_battery.",
    },
    "entity_outdoor_temperature": {
        "type": "str", "entity": True,
        "label": "Уличная температура",
        "hint": "Любой уличный датчик в °C или просто число (20 = 20 °C). Улучшает прогноз нагрузки. Пусто = температура из Open-Meteo.",
    },
    "entity_grid_status": {
        "type": "str", "entity": True,
        "label": "Статус сети",
        "hint": "binary_sensor наличия сети: on = сеть есть, off = отключение. По нему ведётся журнал отключений.",
    },
    "entity_grid_voltage": {
        "type": "str", "entity": True,
        "label": "Напряжение сети",
        "hint": "Необязательно. Напряжение любой фазы в вольтах — помогает распознать отключение в часах, где у binary_sensor нет истории.",
    },
    "entity_export_counter": {
        "type": "str", "entity": True,
        "label": "Счётчик отдачи в сеть",
        "hint": "Суммарная отданная в сеть энергия, кВт·ч (нарастающий счётчик). Нет продажи в сеть — оставьте как есть, покажет ноль.",
    },
    "entity_battery_soh": {
        "type": "str", "entity": True,
        "label": "Износ батареи (SoH)",
        "hint": "Датчик состояния батареи в процентах или просто число (99 = 99%). Учитывается в резерве и оптимизаторе. Пусто = батарея новая (100%).",
    },
}

# Groups for the settings card (non-entity keys, panel order).
OPTION_GROUPS = [
    {"title": "Участок", "keys": [
        "latitude", "longitude", "time_zone",
        "history_backfill_days", "history_statistics_days",
    ]},
    {"title": "Батарея и панели", "keys": [
        "capacity_kwh", "min_soc_pct", "max_soc_pct",
        "charge_efficiency", "discharge_efficiency",
        "max_charge_kw", "max_discharge_kw",
        "panel_kwp", "panel_tilt_deg", "panel_azimuth_deg",
    ]},
    {"title": "План и применение", "keys": [
        "run_time", "replan_minutes", "apply_mode", "notifications",
        "publish_sensors", "optimizer_enabled", "day_zone_strategy", "priority_mode",
    ]},
    {"title": "Тариф", "keys": [
        "tariff_enabled", "tariff_normal_price", "tariff_cheap_price", "tariff_cheap_windows",
    ]},
    {"title": "Резерв отключений", "keys": [
        "reserve_cold_start_hours", "reserve_pv_aware", "outage_memory_days", "autotune",
    ]},
    {"title": "Зоны TOU инвертора", "keys": [
        "tou_zones", "tou_soc_step_pct",
        "tou_soc_entity_template", "tou_time_entity_template",
        "tou_grid_charge_entity_template",
        "tou_grid_charge_on_option", "tou_grid_charge_off_option",
    ]},
    {"title": "Модели и калибровка", "keys": [
        "retrain_enabled", "retrain_interval_days", "load_holidays",
        "calibration_charge_enabled", "calibration_interval_days",
    ]},
]

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def coerce_option(key: str, value):
    """Validate and coerce one panel edit to the option's schema type.

    Returns the coerced value; raises ValueError with a readable Russian
    message when the value cannot be accepted. Unknown keys raise KeyError.
    """
    meta = OPTION_META[key]
    kind = meta["type"]
    label = meta.get("label", key)
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("true", "1", "on", "yes"):
            return True
        if text in ("false", "0", "off", "no"):
            return False
        raise ValueError(f"{label}: ожидается да/нет")
    if kind in ("float", "int"):
        try:
            number = float(str(value).strip().replace(",", "."))
        except (TypeError, ValueError):
            raise ValueError(f"{label}: «{value}» — не число") from None
        if kind == "int":
            number = int(number)
        lo, hi = meta.get("min"), meta.get("max")
        if lo is not None and number < lo or hi is not None and number > hi:
            raise ValueError(f"{label}: {number} вне диапазона {lo}–{hi}")
        return number
    if kind == "select":
        text = str(value).strip()
        if text not in meta["choices"]:
            raise ValueError(
                f"{label}: «{text}» — допустимы только {', '.join(meta['choices'])}"
            )
        return text
    if kind == "time":
        text = str(value).strip()
        if not _TIME_RE.match(text):
            raise ValueError(f"{label}: «{text}» — нужен формат ЧЧ:ММ")
        return text
    return str(value).strip()


def options_view(opts: dict) -> list[dict]:
    """Settings card payload: groups of inputs with current values."""
    groups = []
    for group in OPTION_GROUPS:
        items = []
        for key in group["keys"]:
            meta = OPTION_META[key]
            item = {
                "key": key,
                "type": meta["type"],
                "label": meta["label"],
                "hint": meta["hint"],
                "value": opts.get(key),
            }
            for extra in ("unit", "min", "max", "step", "choices"):
                if extra in meta:
                    item[extra] = meta[extra]
            items.append(item)
        groups.append({"title": group["title"], "items": items})
    return groups


def entity_hint(key: str) -> str | None:
    """Hint text for an entity-binding row in the entities card."""
    meta = OPTION_META.get(key)
    return meta["hint"] if meta and meta.get("entity") else None
