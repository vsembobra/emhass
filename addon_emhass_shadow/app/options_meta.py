"""Option metadata for the shadow panel settings card.

Every add-on option that can be edited from the panel is described here:
input type, Russian and English label and hint (mirrors config.yaml
translations), range/choices for validation. The panel renders inputs from
this metadata and update_options() validates edits against it, so a bad
value is rejected with a readable message instead of breaking the
Supervisor schema check.

deferrable_loads stays add-on-config only: it is a list of objects and
does not fit a flat input row.
"""

import re

# Every option's built-in value. The add-on settings list only VISIBLE (the
# owner's decisions); the rest stay optional in the schema, are applied from
# here, and AUTO_KEYS are first looked up in Home Assistant (discovery.py).
DEFAULTS = {
    "latitude": 44.94892,
    "longitude": 34.10887,
    "time_zone": "Europe/Istanbul",
    "capacity_kwh": 32.0,
    "panel_kwp": 7.0,
    "open_meteo_model": "auto",
    "panel_tilt_deg": 26.0,
    "panel_azimuth_deg": 195.0,
    "min_soc_pct": 10.0,
    "max_soc_pct": 100.0,
    "discharge_efficiency": 0.95,
    "charge_efficiency": 0.95,
    "max_charge_kw": 5.0,
    "max_discharge_kw": 5.0,
    "floor_input_entity": "input_number.deye_soc_backup",
    "entity_load_power": "sensor.inverter_load_power",
    "entity_critical_load_power": "sensor.inverter_load_ups_power",
    "entity_pv_power": "sensor.inverter_pv_power",
    "entity_battery_power": "sensor.inverter_battery_power",
    "entity_grid_power": "sensor.inverter_grid_power",
    "entity_battery_soc": "sensor.inverter_battery",
    "entity_outdoor_temperature": "",
    "entity_grid_status": "binary_sensor.inverter_grid",
    "entity_grid_voltage": "sensor.inverter_grid_l1_voltage",
    "entity_export_counter": "sensor.inverter_total_energy_export",
    "tou_soc_entity_template": "number.inverter_program_{n}_soc",
    "tou_time_entity_template": "time.inverter_program_{n}_time",
    "run_time": "06:47",
    "notifications": True,
    "apply_mode": False,
    "retrain_enabled": True,
    "retrain_interval_days": 7,
    "optimizer_enabled": True,
    "day_zone_strategy": "flat",
    "tariff_enabled": False,
    "tariff_normal_price": 2.0,
    "tariff_cheap_price": 1.0,
    "tariff_cheap_windows": "23:00-07:00",
    "reserve_cold_start_hours": 0.0,
    "reserve_pv_aware": True,
    "history_backfill_days": 14,
    "entity_battery_soh": "",
    "load_holidays": "",
    "autotune": False,
    "priority_mode": "balanced",
    "publish_sensors": False,
    "deferrable_loads": [],
    "tou_zones": 6,
    "tou_soc_step_pct": 5,
    "tou_grid_charge_entity_template": "",
    "outage_memory_days": 14,
    "history_statistics_days": 365,
    "replan_minutes": 120,
    "tou_grid_charge_on_option": "",
    "tou_grid_charge_off_option": "",
    "calibration_charge_enabled": True,
    "calibration_interval_days": 14,
}

# Shown in the add-on settings by default: only what the owner decides.
VISIBLE = (
    "apply_mode",
    "notifications",
    "publish_sensors",
    "min_soc_pct",
    "tariff_enabled",
    "tariff_normal_price",
    "tariff_cheap_price",
    "tariff_cheap_windows",
    "priority_mode",
    "autotune",
    "panel_kwp",
    "open_meteo_model",
    "deferrable_loads",
)

# key -> metadata. Types: float, int, bool, str, select, time (HH:MM str).
# label/hint are Russian, label_en/hint_en their English mirror; the panel
# picks per its language switch.
OPTION_META = {
    "open_meteo_model": {
        "type": "select", "choices": ["auto", "ecmwf", "icon", "gfs"],
        "label": "Погодная модель Open-Meteo", "label_en": "Open-Meteo weather model",
        "hint": "Auto — подбор Open-Meteo; ECMWF — IFS HRES; ICON и GFS — Seamless. При смене модели прогнозы и обучение хранятся отдельно. Первый расчёт может занять дольше из-за обучения.",
        "hint_en": "Auto uses Open-Meteo selection; ECMWF uses IFS HRES; ICON and GFS use Seamless. Each selection has separate forecasts and training. The first run may take longer to train.",
    },
    # --- Site -------------------------------------------------------------
    "latitude": {
        "type": "float", "min": -90, "max": 90, "step": 0.00001,
        "label": "Широта",
        "label_en": "Latitude",
        "hint": "Координата объекта, например 44.94892. Нужна для расчёта высоты солнца и погоды. Пусто = из настроек Home Assistant.",
        "hint_en": "Site coordinate, e.g. 44.94892. Used for solar geometry and weather. Empty = from the Home Assistant settings.",
    },
    "longitude": {
        "type": "float", "min": -180, "max": 180, "step": 0.00001,
        "label": "Долгота",
        "label_en": "Longitude",
        "hint": "Координата объекта, например 34.10887. Пусто = из настроек Home Assistant.",
        "hint_en": "Site coordinate, e.g. 34.10887. Empty = from the Home Assistant settings.",
    },
    "time_zone": {
        "type": "str",
        "label": "Часовой пояс",
        "label_en": "Time zone",
        "hint": "Имя из базы IANA, например Europe/Istanbul. Все окна и зоны считаются по местному времени. Пусто = из настроек Home Assistant.",
        "hint_en": "IANA name, e.g. Europe/Istanbul. All windows and zones use local time. Empty = from the Home Assistant settings.",
    },
    "history_backfill_days": {
        "type": "int", "min": 0, "max": 60, "unit": "дн.",
        "label": "Глубина загрузки истории",
        "label_en": "History backfill depth",
        "hint": "Сколько дней прошлого забирать из истории HA по кнопке «Загрузить историю» (0–60). Реальная глубина ограничена сроком хранения рекордера HA (обычно 10 дней). Данные сразу идут в обучение моделей.",
        "hint_en": "How many past days to pull from Home Assistant history via the 'Backfill history' button (0–60). Actual depth is limited by the HA recorder retention (usually 10 days). This data feeds the model training immediately.",
    },
    "history_statistics_days": {
        "type": "int", "min": 0, "max": 730, "unit": "дн.",
        "label": "Глубина долгосрочной статистики",
        "label_en": "Long-term statistics depth",
        "hint": "Сколько дней почасовой статистики HA подтянуть сверх подробной истории recorder (0–730, по умолчанию 365). С ней модели сразу учатся на сезонах вашего дома. 0 = не использовать.",
        "hint_en": "How many days of HA hourly statistics to pull beyond the detailed recorder history (0–730, default 365). With it the models learn your home's seasons at once. 0 = off.",
    },
    # --- Plant ------------------------------------------------------------
    "capacity_kwh": {
        "type": "float", "min": 1, "max": 200, "step": 0.5, "unit": "кВт·ч",
        "label": "Ёмкость батареи",
        "label_en": "Battery capacity",
        "hint": "Полная номинальная ёмкость АКБ, например 32. Пусто = из настроек инвертора: Battery Capacity (А·ч) × 51,2 В.",
        "hint_en": "Full nominal battery capacity, e.g. 32. Empty = from the inverter settings: Battery Capacity (Ah) × 51.2 V.",
    },
    "min_soc_pct": {
        "type": "float", "min": 0, "max": 50, "step": 1, "unit": "%",
        "label": "Минимальный SOC",
        "label_en": "Minimum SOC",
        "hint": "Абсолютное дно разряда (10 = не разряжать ниже 10%). Резерв отключений может держать пол выше — это лишь жёсткая нижняя граница.",
        "hint_en": "Absolute discharge floor (10 = never discharge below 10%). The outage reserve may hold the floor higher — this is only the hard lower bound.",
    },
    "max_soc_pct": {
        "type": "float", "min": 50, "max": 100, "step": 1, "unit": "%",
        "label": "Максимальный SOC",
        "label_en": "Maximum SOC",
        "hint": "До скольки процентов разрешено заряжать батарею (100 = полностью).",
        "hint_en": "How far the battery may be charged (100 = full).",
    },
    "charge_efficiency": {
        "type": "float", "min": 0.5, "max": 1, "step": 0.01,
        "label": "КПД заряда",
        "label_en": "Charge efficiency",
        "hint": "Доля энергии, доходящая до батареи при заряде (0.95 = 95%).",
        "hint_en": "Share of input energy stored in the battery (0.95 = 95%).",
    },
    "discharge_efficiency": {
        "type": "float", "min": 0.5, "max": 1, "step": 0.01,
        "label": "КПД разряда",
        "label_en": "Discharge efficiency",
        "hint": "Доля энергии, доходящая до нагрузки при разряде (0.95 = 95%).",
        "hint_en": "Share of battery energy reaching the loads (0.95 = 95%).",
    },
    "max_charge_kw": {
        "type": "float", "min": 0.5, "max": 50, "step": 0.5, "unit": "кВт",
        "label": "Макс. мощность заряда",
        "label_en": "Max charge power",
        "hint": "Ограничение инвертора на заряд АКБ, например 5. Пусто = из настроек инвертора: ток заряда АКБ из сети × напряжение, не больше номинала инвертора.",
        "hint_en": "Inverter charge limit for the battery, e.g. 5. Empty = from the inverter settings: battery grid charging current × voltage, capped by the inverter rating.",
    },
    "max_discharge_kw": {
        "type": "float", "min": 0.5, "max": 50, "step": 0.5, "unit": "кВт",
        "label": "Макс. мощность разряда",
        "label_en": "Max discharge power",
        "hint": "Ограничение инвертора на разряд АКБ, например 5. Пусто = из настроек инвертора: максимальный ток разряда АКБ × напряжение, не больше номинала инвертора.",
        "hint_en": "Inverter discharge limit for the battery, e.g. 5. Empty = from the inverter settings: battery max discharging current × voltage, capped by the inverter rating.",
    },
    "panel_kwp": {
        "type": "float", "min": 0.5, "max": 100, "step": 0.1, "unit": "кВтп",
        "label": "Мощность панелей",
        "label_en": "Panel capacity",
        "hint": "Суммарная паспортная мощность солнечных панелей. Используется для прогноза солнца, пока модель не переобучилась на ваших данных.",
        "hint_en": "Total nameplate power of the solar array. Used for the PV forecast until the model has retrained on your data.",
    },
    "panel_tilt_deg": {
        "type": "float", "min": 0, "max": 90, "step": 1, "unit": "°",
        "label": "Наклон панелей",
        "label_en": "Panel tilt",
        "hint": "0 = лежат горизонтально, 90 = вертикально.",
        "hint_en": "0 = flat, 90 = vertical.",
    },
    "panel_azimuth_deg": {
        "type": "float", "min": 0, "max": 360, "step": 1, "unit": "°",
        "label": "Азимут панелей",
        "label_en": "Panel azimuth",
        "hint": "Куда смотрят панели: 180 = строго на юг, 90 = восток, 270 = запад. 15° западнее юга = 195.",
        "hint_en": "Facing direction: 180 = due south, 90 = east, 270 = west. 15 degrees west of south = 195.",
    },
    # --- Plan ---------------------------------------------------------------
    "run_time": {
        "type": "time",
        "label": "Время ежедневного расчёта",
        "label_en": "Daily run time",
        "hint": "ЧЧ:ММ по местному времени. Полный расчёт и уведомление выполняются раз в день в этот момент.",
        "hint_en": "HH:MM local. The full plan and notification are produced once a day at this time.",
    },
    "replan_minutes": {
        "type": "int", "min": 15, "max": 240, "unit": "мин",
        "label": "Пересчёт плана",
        "label_en": "Replan interval",
        "hint": "Как часто пересчитывать план между ежедневными расчётами (15–240, по умолчанию 120 — раз в два часа). Частый пересчёт держит цели зон на реальном SOC и свежем прогнозе.",
        "hint_en": "How often to replan between the daily runs (15–240, default 120 — every two hours). Frequent replans keep the zone targets on the measured SOC and a fresh forecast.",
    },
    "apply_mode": {
        "type": "bool",
        "label": "Применять план к инвертору",
        "label_en": "Apply plan to inverter",
        "hint": "ВЫКЛ (рекомендуется на первые 1–2 недели): только советы, в инвертор ничего не пишется. ВКЛ: после расчёта аддон записывает целевые SOC в зоны TOU инвертора и порог SOC в input_number. Записываются только изменения от 5 п.п.",
        "hint_en": "OFF (recommended for the first 1–2 weeks): advisory only, nothing is written. ON: after each run the add-on writes target SOCs to the inverter's TOU zones and the floor SOC to the input_number. Only changes of 5 pp or more are written.",
    },
    "notifications": {
        "type": "bool",
        "label": "Уведомления",
        "label_en": "Notifications",
        "hint": "Показывать итог расчёта уведомлением в Home Assistant. Уведомления об ошибках (погода, сбой расчёта или обучения, битые сущности) приходят всегда и снимаются сами.",
        "hint_en": "Show the plan summary as a Home Assistant notification. Error alerts (weather, failed run or retrain, broken entities) are always sent and clear themselves on recovery.",
    },
    "publish_sensors": {
        "type": "bool",
        "label": "Публиковать сенсоры",
        "label_en": "Publish sensors",
        "hint": "Создавать в HA сенсоры sensor.emhass_shadow_*: рекомендуемый пол SOC, автономия в часах, старт переносимых нагрузок — для ваших автоматизаций. Это не запись в инвертор.",
        "hint_en": "Create sensor.emhass_shadow_* in Home Assistant: recommended SOC floor, battery autonomy in hours and flexible-load start times — for your automations. This is not an inverter write.",
    },
    "optimizer_enabled": {
        "type": "bool",
        "label": "MILP-оптимизатор",
        "label_en": "MILP optimizer",
        "hint": "Суточный оптимальный план заряда/разряда с тарифами и целевыми SOC зон TOU. Выключен = действует только эвристика пола SOC.",
        "hint_en": "Optimal day-ahead charge/discharge plan with tariffs and per-zone TOU SOC targets. Off = only the SOC floor heuristic runs.",
    },
    "day_zone_strategy": {
        "type": "select", "choices": ["ladder", "flat"],
        "label": "Стратегия дневных зон",
        "label_en": "Daytime zone strategy",
        "hint": "flat (по умолчанию): все дорогие зоны — один пол, минимальный SOC или резерв во время серии отключений; батарея тратится днём до этого пола. Ночная зона — цель заряда. ladder (лестница): пол каждой зоны следует почасовому плану — только для тарифа с пиком внутри дня.",
        "hint_en": "flat (default): every zone outside the cheap hours gets one floor — min SOC, or the outage reserve while an outage series is live; the battery is spent down to that floor during the day. The night zone keeps the charge target. ladder: each zone's floor follows the plan's hourly trajectory — only for a tariff with a peak inside the day.",
    },
    "priority_mode": {
        "type": "select", "choices": ["resilience", "balanced", "economy"],
        "label": "Приоритет",
        "label_en": "Priority",
        "hint": "Насколько осторожно считать план. economy — на солнечный день: ночной заряд рассчитан на солнце P90, резерв под 80% отключений серии; balanced — среднее солнце (P50), резерв под 90%; resilience — на пасмурный день (солнце P10), резерв под 95% и запас 1–2 ч. Нагрузка во всех режимах — медианная (P50). Автонастройка выбирает только среди вариантов, которые разрешает приоритет.",
        "hint_en": "How cautious the plan is. economy: a sunny day — the night charge is sized for sun P90, the reserve for 80% of the series' outages; balanced: median sun (P50), reserve for 90%; resilience: a gloomy day (sun P10), reserve for 95% plus 1–2 h. The load scenario is the median (P50) in every mode. Autotune only picks among the options the priority allows.",
    },
    # --- Tariff -------------------------------------------------------------
    "tariff_enabled": {
        "type": "bool",
        "label": "Ночной дешёвый тариф",
        "label_en": "Cheap night tariff",
        "hint": "Включите, если электричество дешевле в заданные часы — оптимизатор будет заряжать АКБ из сети в дешёвое окно и разряжать в дорогое.",
        "hint_en": "Enable if electricity is cheaper during set hours — the optimizer will charge from the grid in cheap windows and discharge in expensive ones.",
    },
    "tariff_normal_price": {
        "type": "float", "min": 0, "max": 1000, "step": 0.01,
        "label": "Обычная цена",
        "label_en": "Normal price",
        "hint": "Цена кВт·ч вне дешёвых окон, в вашей валюте. Важно соотношение с дешёвой ценой, а не абсолютное значение.",
        "hint_en": "Price per kWh outside cheap windows, in your currency. Relative values are fine — the ratio to the cheap price is what matters.",
    },
    "tariff_cheap_price": {
        "type": "float", "min": 0, "max": 1000, "step": 0.01,
        "label": "Дешёвая цена",
        "label_en": "Cheap price",
        "hint": "Цена кВт·ч внутри дешёвых окон.",
        "hint_en": "Price per kWh inside cheap windows.",
    },
    "tariff_cheap_windows": {
        "type": "str",
        "label": "Окна дешёвого тарифа",
        "label_en": "Cheap tariff windows",
        "hint": "Местное время, ЧЧ:ММ-ЧЧ:ММ; несколько окон через запятую. Окно может переходить через полночь: 23:00-07:00, 13:00-15:00.",
        "hint_en": "Local time, HH:MM-HH:MM; several windows comma-separated. A window may wrap midnight, e.g. 23:00-07:00, 13:00-15:00.",
    },
    # --- Reserve ------------------------------------------------------------
    "reserve_cold_start_hours": {
        "type": "float", "min": 0, "max": 20, "step": 0.5, "unit": "ч",
        "label": "Минимальная автономия",
        "label_en": "Minimum autonomy",
        "hint": "Постоянный запас в часах работы критической нагрузки, который держится даже без отключений (0–20). 0 = в спокойный период держать только минимальный SOC и не защищаться от фантомных отключений.",
        "hint_en": "Standing reserve in hours of critical load, kept even with no outages (0–20). 0 = in quiet periods keep only the minimum SOC, no protection against phantom outages.",
    },
    "reserve_pv_aware": {
        "type": "bool",
        "label": "Резерв с учётом солнца",
        "label_en": "PV-aware reserve",
        "hint": "ВКЛ: в часы, когда пессимистичный прогноз солнца перекрывает критическую нагрузку, резерв на эти часы снижается (ночной не меняется). ВЫКЛ: резерв одинаковый круглые сутки.",
        "hint_en": "ON: during hours where the pessimistic solar forecast covers the critical load, the reserve for those hours is reduced (the night reserve is unchanged). OFF: the reserve is flat around the clock.",
    },
    "outage_memory_days": {
        "type": "int", "min": 3, "max": 60, "unit": "дн.",
        "label": "Память серии отключений",
        "label_en": "Outage series memory",
        "hint": "Отключения идут сериями. Пока последнее отключение было не больше этого числа дней назад, серия считается идущей и резерв держится по её реальной длительности. Дольше тишины — серия закончилась (3–60, по умолчанию 14).",
        "hint_en": "Outages come in series. While the last outage is at most this many days old the series is live and the reserve follows its real durations. Longer quiet means the series is over (3–60, default 14).",
    },
    "autotune": {
        "type": "bool",
        "label": "Автонастройка по данным",
        "label_en": "Autotune from data",
        "hint": "Применять то, что аддон выучил сам: реальную ёмкость и КПД батареи, ориентацию панелей, профиль критической нагрузки и подобранные на вашей истории настройки. ВЫКЛ = выученное показывается только как совет.",
        "hint_en": "Apply what the add-on learned by itself: real battery capacity and efficiency, panel orientation, hourly critical-load profile and plan settings tuned on your history. OFF = learned values are shown as advice only.",
    },
    # --- TOU ----------------------------------------------------------------
    "tou_zones": {
        "type": "int", "min": 1, "max": 12,
        "label": "Число зон TOU",
        "label_en": "TOU zones",
        "hint": "Сколько программ (зон) времени использования у инвертора. У Deye — 6. Шаблоны сущностей подставляют номер зоны вместо {n}. Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "How many time-of-use programs (zones) the inverter has. Deye has 6. The entity templates substitute the zone number for {n}. Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "tou_soc_step_pct": {
        "type": "int", "min": 1, "max": 10, "unit": "%",
        "label": "Шаг SOC зон",
        "label_en": "Zone SOC step",
        "hint": "С каким шагом инвертор принимает SOC зоны (у Deye — 5). Цели всегда округляются вверх до этого шага, чтобы не срезать резерв.",
        "hint_en": "The SOC granularity the inverter accepts for a zone (Deye: 5). Targets are always rounded up to this step so the reserve is never shaved.",
    },
    "tou_soc_entity_template": {
        "type": "str",
        "label": "Шаблон сущностей SOC зон",
        "label_en": "TOU zone SOC entity template",
        "hint": "Шаблон с {n} вместо номера зоны. number.inverter_program_{n}_soc разворачивается в number.inverter_program_1_soc … 6_soc. Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "Template with {n} for the zone number. number.inverter_program_{n}_soc expands to number.inverter_program_1_soc … 6_soc. Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "tou_time_entity_template": {
        "type": "str",
        "label": "Шаблон сущностей времени зон",
        "label_en": "TOU zone time entity template",
        "hint": "Шаблон с {n} вместо номера зоны. time.inverter_program_{n}_time разворачивается в time.inverter_program_1_time … 6_time. Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "Template with {n} for the zone number. time.inverter_program_{n}_time expands to time.inverter_program_1_time … 6_time. Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "tou_grid_charge_entity_template": {
        "type": "str",
        "label": "Шаблон флага «заряд из сети»",
        "label_en": "Zone grid-charge flag template",
        "hint": "Сущность switch или input_boolean, включающая заряд из сети в зоне, с {n} вместо номера. Задано + применение плана = аддон разрешает заряд из сети только там, где он нужен. Пусто = Program N Charging из ha-solarman, если он есть; none = флаги не трогать.",
        "hint_en": "switch or input_boolean entity that enables grid charging in a zone, with {n} for the zone number. Set + apply mode on = the add-on allows grid charging only where it is needed. Empty = Program N Charging from ha-solarman when present; none = leave the flags alone.",
    },
    "tou_grid_charge_on_option": {
        "type": "str",
        "label": "Вариант select «заряд из сети вкл»",
        "label_en": "Select option for grid charge on",
        "hint": "Только для select-флагов. Пусто = аддон сам выберет вариант со словом Grid (или Grid & Gen, если в зоне включён генератор). Укажите точное название, если варианты называются иначе.",
        "hint_en": "Only for select flags. Empty = the add-on picks the option containing Grid (or Grid & Gen when the zone already charges from a generator). Set the exact name if your options are named differently.",
    },
    "tou_grid_charge_off_option": {
        "type": "str",
        "label": "Вариант select «заряд из сети выкл»",
        "label_en": "Select option for grid charge off",
        "hint": "Только для select-флагов. Пусто = аддон сам выберет Disabled / Off (или Gen, если в зоне включён генератор).",
        "hint_en": "Only for select flags. Empty = the add-on picks Disabled / Off (or Gen when the zone charges from a generator).",
    },
    # --- Models ---------------------------------------------------------------
    "retrain_enabled": {
        "type": "bool",
        "label": "Автоматическое переобучение",
        "label_en": "Automatic retraining",
        "hint": "Модели прогноза дообучаются на данных объекта с заданным интервалом. Кнопка переобучения в панели работает независимо от этого переключателя.",
        "hint_en": "Forecast models retrain on site data at the configured interval. The panel retrain button works regardless.",
    },
    "retrain_interval_days": {
        "type": "int", "min": 1, "max": 30, "unit": "дн.",
        "label": "Интервал переобучения",
        "label_en": "Retrain interval",
        "hint": "Как часто модели переобучаются автоматически (1–30).",
        "hint_en": "How often models retrain automatically (1–30).",
    },
    "load_holidays": {
        "type": "str",
        "label": "Праздничные дни",
        "label_en": "Holidays",
        "hint": "Ежегодные даты с потреблением как в выходной, через запятую в формате ММ-ДД: 01-01,01-07,03-08. Учитываются со следующего переобучения. Пусто = без праздников.",
        "hint_en": "Recurring dates with weekend-like consumption, comma-separated MM-DD: 01-01,01-07,03-08. Used from the next retrain. Empty = no holidays.",
    },
    "calibration_charge_enabled": {
        "type": "bool",
        "label": "Калибровочный заряд BMS",
        "label_en": "BMS calibration charge",
        "hint": "BMS уточняет SOC только при полном заряде; без него ошибка растёт. Если батарея не была на 100% дольше интервала, аддон зарядит её до 100% в ближайшее ночное окно. Полный заряд от солнца сбрасывает отсчёт.",
        "hint_en": "A BMS re-anchors SOC only at a full charge; without one the error grows. When the battery has not been at 100% for longer than the interval, the add-on charges it to 100% in the next night window. A full charge from the sun resets the count.",
    },
    "calibration_interval_days": {
        "type": "int", "min": 3, "max": 60, "unit": "дн.",
        "label": "Интервал калибровки",
        "label_en": "Calibration interval",
        "hint": "Через сколько дней без полного заряда устраивать калибровочный заряд до 100% (3–60, по умолчанию 14).",
        "hint_en": "Days without a full charge before a calibration charge to 100% (3–60, default 14).",
    },
    # --- Entity bindings (edited in the entities card, hints reused there) ---
    "floor_input_entity": {
        "type": "str", "entity": True,
        "label": "Сущность текущего порога SOC",
        "label_en": "Current floor SOC entity",
        "hint": "input_number или number, где хранится нижний порог SOC инвертора в процентах: input_number.deye_soc_backup или number.inverter_battery_low_soc.",
        "hint_en": "input_number or number holding the inverter's lower SOC threshold in percent: input_number.deye_soc_backup or number.inverter_battery_low_soc.",
    },
    "entity_load_power": {
        "type": "str", "entity": True,
        "label": "Мощность нагрузки",
        "label_en": "Load power",
        "hint": "Мощность всех потребителей дома, ватты. Пример: sensor.inverter_load_power. Основной ряд для прогноза нагрузки. Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "Total house consumption power, watts. Example: sensor.inverter_load_power. The main series for the load forecast. Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "entity_critical_load_power": {
        "type": "str", "entity": True,
        "label": "Критическая нагрузка",
        "label_en": "Critical load",
        "hint": "Мощность резервируемых (UPS) потребителей, ватты. По ней считается резерв на случай отключения сети. Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "Backup/UPS loads power, watts. Drives the outage reserve sizing. Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "entity_pv_power": {
        "type": "str", "entity": True,
        "label": "Мощность солнца",
        "label_en": "PV power",
        "hint": "Мощность солнечных панелей, ватты. Ряд для обучения модели солнечной генерации. Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "Solar array power, watts. Training series for the solar forecast model. Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "entity_battery_power": {
        "type": "str", "entity": True,
        "label": "Мощность батареи",
        "label_en": "Battery power",
        "hint": "Мощность заряда/разряда АКБ, ватты; знак по конвенции вашей интеграции (у Deye: минус = заряд). Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "Battery charge/discharge power, watts; sign follows your integration's convention (Deye: negative = charging). Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "entity_grid_power": {
        "type": "str", "entity": True,
        "label": "Мощность сети",
        "label_en": "Grid power",
        "hint": "Обмен с сетью, ватты; плюс = потребление из сети. Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "Grid exchange, watts; positive = importing. Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "entity_battery_soc": {
        "type": "str", "entity": True,
        "label": "Заряд батареи",
        "label_en": "Battery SOC",
        "hint": "Текущий уровень заряда АКБ, проценты (0–100). Пример: sensor.inverter_battery. Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "Current battery state of charge, percent (0–100). Example: sensor.inverter_battery. Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "entity_outdoor_temperature": {
        "type": "str", "entity": True,
        "label": "Уличная температура",
        "label_en": "Outdoor temperature",
        "hint": "Любой уличный датчик в °C или просто число (20 = 20 °C). Улучшает прогноз нагрузки. Пусто = температура из Open-Meteo.",
        "hint_en": "Any outdoor sensor in °C or a plain number (20 = 20 °C). Improves the load forecast. Empty = Open-Meteo temperature.",
    },
    "entity_grid_status": {
        "type": "str", "entity": True,
        "label": "Статус сети",
        "label_en": "Grid status",
        "hint": "binary_sensor наличия сети: on = сеть есть, off = отключение. По нему ведётся журнал отключений. Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "binary_sensor for grid availability: on = grid up, off = outage. Feeds the outage log. Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "entity_grid_voltage": {
        "type": "str", "entity": True,
        "label": "Напряжение сети",
        "label_en": "Grid voltage",
        "hint": "Необязательно. Напряжение любой фазы в вольтах — помогает распознать отключение в часах, где у binary_sensor нет истории. Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "Optional. Any phase voltage in volts — helps detect outages in hours where the binary_sensor has no history. Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "entity_export_counter": {
        "type": "str", "entity": True,
        "label": "Счётчик отдачи в сеть",
        "label_en": "Grid export counter",
        "hint": "Суммарная отданная в сеть энергия, кВт·ч (нарастающий счётчик). Нет продажи в сеть — оставьте как есть, покажет ноль. Пусто = найти в ha-solarman (Deye) автоматически; none = не использовать.",
        "hint_en": "Total energy exported to the grid, kWh (cumulative counter). No feed-in? Keep the default, it will read zero. Empty = found in ha-solarman (Deye) automatically; none = not used.",
    },
    "entity_battery_soh": {
        "type": "str", "entity": True,
        "label": "Износ батареи (SoH)",
        "label_en": "Battery SoH",
        "hint": "Датчик состояния батареи в процентах или просто число (99 = 99%). Учитывается в резерве и оптимизаторе. Пусто = Battery SOH из ha-solarman, если он есть, иначе батарея новая (100%); none = не учитывать.",
        "hint_en": "Battery state-of-health sensor in percent or a plain number (99 = 99%). Used for reserve sizing and the optimizer. Empty = Battery SOH from ha-solarman when present, else a new battery (100%); none = ignore.",
    },
}

# Groups for the settings card (non-entity keys, panel order). Inverter and
# battery decisions sit in one group, application behaviour in another.
OPTION_GROUPS = [
    {"title": "Объект", "title_en": "Site", "keys": [
        "latitude", "longitude", "time_zone",
        "history_backfill_days", "history_statistics_days",
    ]},
    {"title": "Инвертор и батарея", "title_en": "Inverter and battery", "keys": [
        "capacity_kwh", "min_soc_pct", "max_soc_pct",
        "charge_efficiency", "discharge_efficiency",
        "max_charge_kw", "max_discharge_kw",
        "apply_mode",
        "tou_zones", "tou_soc_step_pct",
        "tou_soc_entity_template", "tou_time_entity_template",
        "tou_grid_charge_entity_template",
        "tou_grid_charge_on_option", "tou_grid_charge_off_option",
    ]},
    {"title": "Солнечные панели", "title_en": "Solar panels", "keys": [
        "panel_kwp", "panel_tilt_deg", "panel_azimuth_deg", "open_meteo_model",
    ]},
    {"title": "Тариф и планирование", "title_en": "Tariff and planning", "keys": [
        "optimizer_enabled", "day_zone_strategy",
        "tariff_enabled", "tariff_normal_price", "tariff_cheap_price", "tariff_cheap_windows",
    ]},
    {"title": "Резерв и приоритет", "title_en": "Reserve and priority", "keys": [
        "priority_mode",
        "reserve_cold_start_hours", "reserve_pv_aware", "outage_memory_days",
    ]},
    {"title": "Обучение и автонастройка", "title_en": "Learning and autotune", "keys": [
        "autotune",
        "retrain_enabled", "retrain_interval_days", "load_holidays",
        "calibration_charge_enabled", "calibration_interval_days",
    ]},
    {"title": "Приложение", "title_en": "Application", "keys": [
        "run_time", "replan_minutes", "notifications", "publish_sensors",
    ]},
]

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def coerce_option(key: str, value, lang: str = "ru"):
    """Validate and coerce one panel edit to the option's schema type.

    Returns the coerced value; raises ValueError with a readable message in
    the panel's language when the value cannot be accepted. Unknown keys
    raise KeyError.
    """
    meta = OPTION_META[key]
    kind = meta["type"]
    label = meta.get("label_en" if lang == "en" else "label", key)
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("true", "1", "on", "yes"):
            return True
        if text in ("false", "0", "off", "no"):
            return False
        raise ValueError(f"{label}: " + ("expected on/off" if lang == "en" else "ожидается да/нет"))
    if kind in ("float", "int"):
        try:
            number = float(str(value).strip().replace(",", "."))
        except (TypeError, ValueError):
            if lang == "en":
                raise ValueError(f"{label}: «{value}» is not a number") from None
            raise ValueError(f"{label}: «{value}» — не число") from None
        if kind == "int":
            number = int(number)
        lo, hi = meta.get("min"), meta.get("max")
        if lo is not None and number < lo or hi is not None and number > hi:
            if lang == "en":
                raise ValueError(f"{label}: {number} is outside {lo}–{hi}")
            raise ValueError(f"{label}: {number} вне диапазона {lo}–{hi}")
        return number
    if kind == "select":
        text = str(value).strip()
        if text not in meta["choices"]:
            if lang == "en":
                raise ValueError(
                    f"{label}: «{text}» — allowed values: {', '.join(meta['choices'])}"
                )
            raise ValueError(
                f"{label}: «{text}» — допустимы только {', '.join(meta['choices'])}"
            )
        return text
    if kind == "time":
        text = str(value).strip()
        if not _TIME_RE.match(text):
            if lang == "en":
                raise ValueError(f"{label}: «{text}» — HH:MM required")
            raise ValueError(f"{label}: «{text}» — нужен формат ЧЧ:ММ")
        return text
    return str(value).strip()


def options_view(opts: dict, sources: dict | None = None) -> list[dict]:
    """Settings card payload: groups of inputs with current values and their source."""
    groups = []
    for group in OPTION_GROUPS:
        items = []
        for key in group["keys"]:
            meta = OPTION_META[key]
            item = {
                "key": key,
                "type": meta["type"],
                "label": meta["label"],
                "label_en": meta.get("label_en"),
                "hint": meta["hint"],
                "hint_en": meta.get("hint_en"),
                "value": opts.get(key),
                "source": (sources or {}).get(key),
            }
            for extra in ("unit", "min", "max", "step", "choices"):
                if extra in meta:
                    item[extra] = meta[extra]
            items.append(item)
        groups.append({"title": group["title"], "title_en": group.get("title_en"), "items": items})
    return groups


def entity_hint(key: str, lang: str = "ru") -> str | None:
    """Hint text for an entity-binding row in the entities card."""
    meta = OPTION_META.get(key)
    if not meta or not meta.get("entity"):
        return None
    if lang == "en":
        return meta.get("hint_en") or meta["hint"]
    return meta["hint"]
