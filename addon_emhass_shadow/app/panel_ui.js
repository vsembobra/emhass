/* Navigation and presentation only. Settings use the existing Supervisor API. */
const PANEL_I18N = {
  ru: {
    shadow_plan_title:'Теневой план батареи',
    tab_now:'Сейчас', tab_history:'История', tab_settings:'Настройки',
    overview_battery:'Батарея', overview_autonomy:'Автономия', overview_solar:'Солнце, ближайшие 24 ч',
    autonomy_assumption:'Оценка без притока солнца; зависит от нагрузки',
    autonomy_load:'При критической нагрузке {v} кВт', autonomy_unknown:'Недостаточно данных для оценки',
    available_energy:'Доступно до нижнего SOC: ≈{v} кВт·ч', snapshot_at:'Снимок при расчёте: {v}',
    plan_details:'Подробности плана', range_today:'Сегодня', range_both:'Сегодня и завтра',
    settings_search:'Поиск настроек', settings_advanced:'Расширенные',
    settings_shared:'Изменения сохраняются в общую конфигурацию аддона.',
    settings_no_results:'Настройки не найдены. Измените запрос.', settings_unsaved:'Есть несохранённые изменения',
    settings_load_error:'Не удалось обновить настройки. Ваши правки сохранены в форме; попробуйте позже.',
    btn_cancel:'Отменить', setting_help:'Подробнее', open_settings:'Открыть настройки',
    empty_now:'Первого плана ещё нет. Проверьте сенсоры в настройках и запустите расчёт.',
    grid_on:'Сеть есть', grid_off:'Сеть отключена', grid_unknown:'Состояние сети неизвестно',
    mode_unknown:'Режим загружается', report_time:'Расчёт: {v}', report_missing:'Расчёта ещё нет',
    application_verified:'Последние записи подтверждены', application_unchanged:'По последнему расчёту изменений не требовалось',
    application_error:'Есть неподтверждённые изменения', application_pending:'Ожидается расчёт в режиме управления',
    application_unknown:'Подтверждение применения недоступно', application_shadow:'План только для ознакомления',
    plan_stale:'План устарел — требуется пересчёт. Показан последний доступный снимок.',
    connection_lost:'Нет связи с аддоном. Показаны последние полученные данные.',
    reserve_shortfall:'Резерв не помещается в батарею: не хватает {v} кВт·ч.',
    floor_relaxed:'Расчёт не смог обеспечить требуемый резерв. Проверьте подробности плана.',
    outage_alert:'Электросеть отключена. Автономия — оценка по нагрузке на момент расчёта.',
    write_alert:'Не все изменения подтверждены инвертором. Проверьте отчёт и привязки управления.',
    weather_archived:'Погода взята из архива; прогноз может быть менее актуальным.',
    reason_series:'Резерв на серию недавних отключений', reason_baseline:'Постоянный запас автономии', reason_minimum:'Сохраняем минимальный SOC',
    decision_title:'Что запланировано', decision_floor:'Сохраняем нижнюю границу заряда {v}%.',
    decision_night:'Расчётный заряд из сети в дешёвые часы: {v} кВт·ч.',
    decision_calibration:'Калибровочный заряд до 100%: {v}.',
    decision_series:'Запас повышен из-за недавних отключений.',
    solar_clipping:'Ожидаемое ограничение генерации: ≈{v} кВт·ч (модель)',
    ser_load_actual:'Потребление — факт', ser_load_forecast:'Потребление — прогноз',
    th_curtailment_loss:'Ограничение солнца, оценка',
  },
  en: {
    shadow_plan_title:'Battery shadow plan',
    tab_now:'Now', tab_history:'History', tab_settings:'Settings',
    overview_battery:'Battery', overview_autonomy:'Autonomy', overview_solar:'Solar, next 24 h',
    autonomy_assumption:'Estimate without incoming solar; depends on load',
    autonomy_load:'At a critical load of {v} kW', autonomy_unknown:'Not enough data to estimate',
    available_energy:'Available down to minimum SOC: ≈{v} kWh', snapshot_at:'Snapshot at calculation: {v}',
    plan_details:'Plan details', range_today:'Today', range_both:'Today and tomorrow',
    settings_search:'Search settings', settings_advanced:'Advanced',
    settings_shared:'Changes are saved to the shared add-on configuration.',
    settings_no_results:'No matching settings. Try another search.', settings_unsaved:'You have unsaved changes',
    settings_load_error:'Could not refresh settings. Your edits remain in the form; try again later.',
    btn_cancel:'Cancel', setting_help:'Details', open_settings:'Open settings',
    empty_now:'No plan yet. Check the sensors in Settings, then run a calculation.',
    grid_on:'Grid available', grid_off:'Grid outage', grid_unknown:'Grid state unknown',
    mode_unknown:'Loading mode', report_time:'Calculated: {v}', report_missing:'No calculation yet',
    application_verified:'Last writes confirmed', application_unchanged:'No changes needed in the last calculation',
    application_error:'Some changes are unconfirmed', application_pending:'Waiting for a calculation in control mode',
    application_unknown:'Application confirmation unavailable', application_shadow:'Advisory plan only',
    plan_stale:'Plan is overdue for recalculation. Showing the last available snapshot.',
    connection_lost:'Cannot reach the add-on. Showing the last received data.',
    reserve_shortfall:'Reserve exceeds battery capacity by {v} kWh.',
    floor_relaxed:'The calculation could not provide the required reserve. Check Plan details.',
    outage_alert:'The grid is down. Autonomy is estimated using the load at calculation time.',
    write_alert:'Some changes were not confirmed by the inverter. Check the report and control bindings.',
    weather_archived:'Weather comes from an archive; the forecast may be less current.',
    reason_series:'Reserve for a recent series of outages', reason_baseline:'Standing autonomy reserve', reason_minimum:'Keeping minimum SOC',
    decision_title:'The plan', decision_floor:'Keep a lower charge boundary of {v}%.',
    decision_night:'Estimated grid charging in cheap hours: {v} kWh.',
    decision_calibration:'Calibration charge to 100%: {v}.',
    decision_series:'Reserve is raised because of recent outages.',
    solar_clipping:'Expected generation clipping: ≈{v} kWh (model)',
    ser_load_actual:'Consumption — actual', ser_load_forecast:'Consumption — forecast',
    th_curtailment_loss:'Solar clipping, estimate',
  }
};
let lastReport = null, lastStatus = {}, panelOnline = true, chartDays = 1;
let lastSetup = {checks: [], groups: []};
let setupError = false;
const basicSettings = new Set(['open_meteo_model','latitude','longitude','time_zone',
  'panel_kwp','panel_tilt_deg','panel_azimuth_deg','capacity_kwh','min_soc_pct','max_soc_pct',
  'max_charge_kw','max_discharge_kw','apply_mode','tariff_enabled','tariff_normal_price',
  'tariff_cheap_price','tariff_cheap_windows','priority_mode','reserve_cold_start_hours',
  'outage_memory_days','deferrable_loads','notifications']);
const displayNumber = (v, digits=1) => v == null ? '—' : Number(v).toLocaleString(locale(), {maximumFractionDigits:digits});
function reportTime(value) {
  const stamp = Date.parse(value);
  return Number.isFinite(stamp) ? new Date(stamp).toLocaleString(locale(), {
    timeZone: lastReport?.site?.time_zone || undefined, day:'2-digit', month:'2-digit', hour:'2-digit', minute:'2-digit'
  }) : '—';
}
function renderOverview() {
  const s = PanelState.summary(lastReport, lastStatus);
  const badge = document.getElementById('modebadge');
  badge.textContent = t(s.mode === 'apply' ? 'badge_apply' : s.mode === 'shadow' ? 'badge_shadow' : 'mode_unknown');
  badge.style.cssText = s.mode === 'apply' ? 'color:var(--warn);border-color:var(--warn)' : '';
  const grid = panelOnline ? s.grid : 'unknown';
  const items = [
    `<strong class='${grid === 'on' ? 'ok' : grid === 'off' ? 'warning' : ''}'>${esc(t('grid_'+grid))}</strong>`,
    `<span>${esc(t('application_'+s.application))}</span>`,
    `<span>${esc(s.hasReport ? t('report_time',{v:reportTime(lastReport.issue_utc)}) : t('report_missing'))}</span>`
  ];
  if (lastStatus.running) items.push(`<span>${esc(t('st_running'))}</span>`);
  document.getElementById('health').innerHTML = items.join('');
  const alerts = [];
  if (!panelOnline) alerts.push(t('connection_lost'));
  if (setupError) alerts.push(t('settings_load_error'));
  if (grid === 'off') alerts.push(t('outage_alert'));
  if (s.hasWriteErrors) alerts.push(t('write_alert'));
  if (s.reserveShortfallKwh > 0) alerts.push(t('reserve_shortfall',{v:displayNumber(s.reserveShortfallKwh)}));
  if (s.floorRelaxed) alerts.push(t('floor_relaxed'));
  if (s.stale) alerts.push(t('plan_stale'));
  if (lastStatus.last_error) alerts.push(t('last_error')+' '+lastStatus.last_error.trim().split('\n').at(-1));
  if (lastReport?.data?.weather_stale) alerts.push(t('weather_archived'));
  document.getElementById('alerts').innerHTML = alerts.map(text=>`<div class='notice error'>${esc(text)}</div>`).join('');
  document.getElementById('empty-now').hidden = s.hasReport;
  document.getElementById('content').style.display = s.hasReport ? 'block' : 'none';
  document.getElementById('battery-value').textContent = s.batteryPct == null ? '—' : displayNumber(s.batteryPct,0)+'%';
  document.getElementById('battery-energy').textContent = s.energyKwh == null ? t('autonomy_unknown') : t('available_energy',{v:displayNumber(s.energyKwh)});
  document.querySelectorAll('.snapshot-time').forEach(el=>el.textContent = s.hasReport ? t('snapshot_at',{v:reportTime(lastReport.issue_utc)}) : '');
  document.getElementById('autonomy-value').textContent = s.autonomyHours == null ? '—' : '≈'+displayNumber(s.autonomyHours)+' '+unit('ч');
  document.getElementById('autonomy-load').textContent = s.autonomyHours == null ? t('autonomy_unknown') : t('autonomy_load',{v:displayNumber(s.criticalKw,2)});
  document.getElementById('reserve-reason').textContent = t('reason_'+s.reserveReason);
  document.getElementById('solar-excess').textContent = s.curtailedKwh == null ? '' : t('solar_clipping',{v:displayNumber(s.curtailedKwh)});
  const parts = [];
  const floor = lastReport?.advisory?.recommended_floor_soc;
  if (typeof floor === 'number') parts.push(t('decision_floor',{v:Math.round(floor*100)}));
  if (s.reserveReason === 'series') parts.push(t('decision_series'));
  if (s.nightChargeKwh > 0) parts.push(t('decision_night',{v:displayNumber(s.nightChargeKwh)}));
  if (s.calibrationWindow) parts.push(t('decision_calibration',{v:s.calibrationWindow}));
  document.getElementById('decision').innerHTML = s.hasReport
    ? `<h2>${esc(t('decision_title'))}</h2><p>${esc(parts.join(' '))}</p>` : '';
}
function resizeCharts() {
  document.querySelectorAll('.echart').forEach(el => {
    if (el.getClientRects().length) echarts.getInstanceByDom(el)?.resize();
  });
}
function selectTab(name, focus=false) {
  for (const key of ['now','history','settings']) {
    const selected = key === name;
    document.getElementById('view-'+key).hidden = !selected;
    const button = document.getElementById('tab-'+key);
    button.setAttribute('aria-selected',String(selected)); button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  }
  resizeCharts();
}
function setChartDays(days) {
  chartDays = days;
  document.getElementById('tomorrow-chart').hidden = days === 1;
  document.getElementById('range-today').setAttribute('aria-pressed',String(days === 1));
  document.getElementById('range-both').setAttribute('aria-pressed',String(days === 2));
  resizeCharts();
}
function captureEdits(selector) {
  return [...document.querySelectorAll(selector+'.changed')].map(el=>({
    key:el.dataset.key, value:el.classList.contains('entinput') ? el.value : el.dataset.type === 'loads'
      ? [...el.querySelectorAll('.flexload-row')].map(row=>Object.fromEntries([...row.querySelectorAll('.loadinput')].map(i=>[i.dataset.field,i.value])))
      : optionValue(el)
  }));
}
function restoreEdits(edits, selector) {
  for (const edit of edits) {
    const el = [...document.querySelectorAll(selector)].find(e=>e.dataset.key === edit.key);
    if (!el) continue;
    if (el.dataset.type === 'loads') el.querySelector('#flexible-load-rows').innerHTML = edit.value.map(flexibleLoadRow).join('');
    else if (el.type === 'checkbox') el.checked = edit.value;
    else el.value = edit.value;
  }
}
function updateSavebar() {
  document.getElementById('savebar').hidden = !document.querySelector('.optinput.changed,.entinput.changed');
}
function filterSettings() {
  const query = document.getElementById('settings-search').value.trim().toLocaleLowerCase();
  const advanced = document.getElementById('settings-advanced').checked;
  document.querySelectorAll('.optrow').forEach(row=>{
    const key = row.querySelector('.optinput').dataset.key;
    const matches = !query || row.dataset.search.includes(query);
    row.hidden = !row.querySelector('.changed') && !(matches && (query || advanced || basicSettings.has(key)));
  });
  document.querySelectorAll('.optgroup').forEach(group=>{group.hidden = ![...group.querySelectorAll('.optrow')].some(row=>!row.hidden);});
  document.querySelectorAll('#ents tr:not(.entitygroup)').forEach(row=>{
    row.hidden = !row.querySelector('.changed') && !!query && !row.dataset.search.includes(query);
  });
  document.querySelectorAll('#ents .entitygroup').forEach(head=>{
    let row = head.nextElementSibling, shown = false;
    while (row && !row.classList.contains('entitygroup')) { if (!row.hidden) shown = true; row = row.nextElementSibling; }
    head.hidden = !shown;
  });
  if (query) {
    if (document.querySelector('.optrow:not([hidden])')) document.getElementById('optscard').open = true;
    if (document.querySelector('#ents tr:not(.entitygroup):not([hidden])')) document.getElementById('entscard').open = true;
  }
  document.getElementById('settings-no-results').hidden = !!document.querySelector('.optrow:not([hidden]),#ents tr:not(.entitygroup):not([hidden])');
}
function cancelSettings() {
  document.querySelectorAll('.optinput.changed,.entinput.changed').forEach(el=>el.classList.remove('changed'));
  renderEntities(lastSetup.checks); renderOptions(lastSetup.groups); updateSavebar();
}
async function saveSettings() {
  const inputs = [...document.querySelectorAll('.optinput.changed,.entinput.changed')];
  if (!inputs.length) return;
  const options = {}, snapshots = new Map();
  inputs.forEach(el=>{
    options[el.dataset.key] = el.classList.contains('entinput') ? el.value.trim() : optionValue(el);
    snapshots.set(el.dataset.key,el.classList.contains('entinput') ? el.value.trim() : optionSnapshot(el));
  });
  const button = document.getElementById('save-settings'), cancel = document.getElementById('cancel-settings');
  button.disabled = cancel.disabled = true; button.textContent = t('btn_saving');
  try {
    const response = await fetch('api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({options,lang:LANG})});
    const result = await response.json();
    if (!response.ok || result.error) throw new Error(result.error || response.status);
    if (result.rejected?.length) alert(t('saved_rejected')+result.rejected.join(', '));
    inputs.forEach(el=>{
      const key = el.dataset.key;
      const current = el.classList.contains('entinput') ? el.value.trim() : optionSnapshot(el);
      if ((Object.hasOwn(result.saved || {},key) || (result.cleared || []).includes(key)) && current === snapshots.get(key)) {
        el.dataset.original = current; el.classList.remove('changed');
      }
    });
    await loadSetup(); await load();
  } catch (e) { alert(t('save_failed')+e.message); }
  finally { button.disabled = cancel.disabled = false; button.textContent=t('btn_save'); onEntityInput(); onOptionInput(); }
}
function initPanelNavigation() {
  document.querySelector('.tabs').addEventListener('keydown',event=>{
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault();
    const keys=['now','history','settings'];
    const active = keys.findIndex(key=>document.getElementById('tab-'+key).getAttribute('aria-selected') === 'true');
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (active+(event.key === 'ArrowRight' ? 1 : 2))%3;
    selectTab(keys[index],true);
  });
  document.getElementById('plan-details').addEventListener('toggle',resizeCharts);
  window.addEventListener('beforeunload',event=>{
    if (!document.querySelector('.optinput.changed,.entinput.changed')) return;
    event.preventDefault(); event.returnValue='';
  });
  renderOverview();
}
