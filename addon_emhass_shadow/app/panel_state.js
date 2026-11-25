/* Display-only state derived from existing reports. Never controls devices. */
(function(root) {
  function number(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }
  function summary(report, status = {}, now = Date.now()) {
    const r = report || {}, a = r.advisory || {}, reserve = r.reserve || {};
    const opt = a.optimizer || {};
    const issue = Date.parse(r.issue_utc);
    const ageMinutes = Number.isFinite(issue) ? (now - issue) / 60000 : null;
    const interval = number(status.replan_minutes) ?? 120;
    const stale = !!report && (ageMinutes == null || ageMinutes < -15 || ageMinutes > interval + 15);
    const gridAge = (now - Date.parse(status.grid_observed_utc)) / 60000;
    const grid = gridAge >= 0 && gridAge <= 30 && ['on', 'off'].includes(status.grid_state)
      ? status.grid_state : 'unknown';
    const mode = status.apply_mode === true ? 'apply' : status.apply_mode === false ? 'shadow' : 'unknown';
    let application = mode === 'shadow' ? 'shadow' : 'unknown';
    if (mode === 'apply' && report) {
      if (r.apply_mode !== true) application = 'pending';
      else if (a.apply_errors && a.apply_errors.length) application = 'error';
      else if (Array.isArray(a.applied)) application = a.applied.length ? 'verified' : 'unchanged';
    }
    const criticalKw = number(reserve.critical_load_kw);
    const hours = number(a.autonomy_hours);
    const autonomyHours = criticalKw > 0 && hours != null && hours >= 0 ? hours : null;
    // HA state strings are preserved in the report; blank/unavailable is not 0%.
    const rawSoc = r.battery_now_pct;
    const batteryPct = number(typeof rawSoc === 'string' && rawSoc.trim() ? Number(rawSoc) : rawSoc);
    const curtail = opt.p_curtail_kw;
    return {
      hasReport: !!report, mode, application, grid, stale, ageMinutes,
      hasWriteErrors: Array.isArray(a.apply_errors) && a.apply_errors.length > 0,
      batteryPct: batteryPct != null && batteryPct >= 0 && batteryPct <= 100 ? batteryPct : null,
      criticalKw, autonomyHours,
      energyKwh: autonomyHours == null ? null : Math.round(autonomyHours * criticalKw * 100) / 100,
      curtailedKwh: Array.isArray(curtail) && curtail.length && curtail.every(v => number(v) != null)
        ? curtail.slice(0, 24).reduce((sum, v) => sum + Math.max(0, v), 0) : null,
      reserveShortfallKwh: number(reserve.shortfall_reserve_kwh) ?? 0,
      reserveReason: reserve.regime && reserve.regime.active ? 'series' : reserve.reserve_kwh > 0 ? 'baseline' : 'minimum',
      floorRelaxed: opt.floor_relaxed === true,
      nightChargeKwh: opt.tariff_enabled ? number(opt.grid_charge_kwh_cheap) : null,
      calibrationWindow: (r.data && r.data.calibration || {}).window_local || null,
    };
  }
  const api = {summary};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PanelState = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
