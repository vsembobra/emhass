const {test} = require('node:test');
const assert = require('node:assert/strict');
const {summary} = require('../addon_emhass_shadow/app/panel_state.js');
const now = Date.parse('2026-11-25T12:00:00Z');
const report = () => ({
  issue_utc: '2026-11-25T11:30:00Z', apply_mode: true, battery_now_pct: 60,
  reserve: {critical_load_kw: 1.2, reserve_kwh: 4, regime: {active: true}},
  advisory: {autonomy_hours: 8, applied: ['zone 1'], apply_errors: [],
    optimizer: {p_curtail_kw: [0, 2, 1], grid_charge_kwh_cheap: 3}},
});
const status = () => ({apply_mode: true, replan_minutes: 120, grid_state: 'on',
  grid_observed_utc: '2026-11-25T11:55:00Z'});

test('empty and legacy data never become a safe zero or a verified plan', () => {
  const s = summary(null, {}, now);
  assert.equal(s.mode, 'unknown'); assert.equal(s.application, 'unknown');
  assert.equal(s.grid, 'unknown'); assert.equal(s.batteryPct, null);
  assert.equal(s.energyKwh, null); assert.equal(s.autonomyHours, null);
  assert.equal(s.hasReport, false);
});
test('calculation-time autonomy carries its load assumption and delivered energy', () => {
  const s = summary(report(), status(), now);
  assert.equal(s.batteryPct, 60); assert.equal(s.autonomyHours, 8);
  assert.equal(s.criticalKw, 1.2); assert.equal(s.energyKwh, 9.6);
  assert.equal(s.application, 'verified'); assert.equal(s.grid, 'on');
  assert.equal(s.curtailedKwh, 3); assert.equal(s.reserveReason, 'series');
});
test('current configured mode wins over an old report', () => {
  assert.equal(summary(report(), {...status(), apply_mode: false}, now).application, 'shadow');
  const r = {...report(), apply_mode: false};
  assert.equal(summary(r, status(), now).application, 'pending');
});
test('partial writes and no-change results are distinct from confirmed writes', () => {
  const r = report(); r.advisory.apply_errors = ['zone 2'];
  assert.equal(summary(r, status(), now).application, 'error');
  r.advisory.apply_errors = []; r.advisory.applied = [];
  assert.equal(summary(r, status(), now).application, 'unchanged');
  delete r.advisory.applied;
  assert.equal(summary(r, status(), now).application, 'unknown');
});
test('expired grid observations and overdue or future-dated plans are explicit', () => {
  const r = report(); r.issue_utc = '2026-11-25T08:00:00Z';
  assert.equal(summary(r, status(), now).stale, true);
  assert.equal(summary(report(), {...status(), grid_observed_utc:'2026-11-25T10:00:00Z'}, now).grid, 'unknown');
  assert.equal(summary(report(), {...status(), grid_state:'off'}, now).grid, 'off');
  r.issue_utc = '2026-11-26T08:00:00Z';
  assert.equal(summary(r, status(), now).stale, true);
});
test('null or invalid measurements do not masquerade as a calculated autonomy', () => {
  const r = report(); r.advisory.autonomy_hours = null; r.battery_now_pct = null;
  const s = summary(r, status(), now);
  assert.equal(s.autonomyHours, null); assert.equal(s.energyKwh, null);
  assert.equal(s.batteryPct, null);
  r.advisory.autonomy_hours = 5; r.reserve.critical_load_kw = 0;
  assert.equal(summary(r, status(), now).autonomyHours, null);
});
test('reserve shortfall and relaxed floors remain visible', () => {
  const r = report(); r.reserve.shortfall_reserve_kwh = 2.5;
  r.advisory.optimizer.floor_relaxed = true;
  const s = summary(r, status(), now);
  assert.equal(s.reserveShortfallKwh, 2.5); assert.equal(s.floorRelaxed, true);
});

const fs = require('node:fs');
const vm = require('node:vm');
const panelHtml = fs.readFileSync(require('node:path').join(__dirname,'../addon_emhass_shadow/app/panel.html'),'utf8');
const setupFunction = panelHtml.slice(panelHtml.indexOf('async function loadSetup()'),panelHtml.indexOf('let entityIdsLoaded'));
for (const failure of ['http', 'payload']) {
  test('settings fetch '+failure+' failure preserves the existing editors and pending edits', async () => {
    const rendered = [];
    const original = {groups:[{items:[{key:'capacity_kwh',value:32}]}],checks:[{key:'entity_grid_voltage'}]};
    const context = {
      lastSetup: original, console:{error(){}}, renderOverview(){}, setupError:false,
      renderEntities(value){ rendered.push(value); }, renderOptions(value){ rendered.push(value); },
      fetch: async url => ({ok:failure !== 'http', json:async()=> failure === 'payload'
        ? {error:'Supervisor unavailable',checks:[]} : {groups:[],checks:[]}})
    };
    await vm.runInNewContext(setupFunction+'\nloadSetup()',context);
    assert.deepEqual(rendered,[]);
    assert.equal(context.lastSetup, original);
  });
}
const uiSource = fs.readFileSync(require('node:path').join(__dirname,'../addon_emhass_shadow/app/panel_ui.js'),'utf8');
const saveFunction = uiSource.slice(uiSource.indexOf('async function saveSettings()'),uiSource.indexOf('function initPanelNavigation()'));
for (const entity of [true,false]) {
  test('successful reset is acknowledged for '+(entity ? 'entity' : 'scalar')+' settings', async () => {
    let dirty = true;
    const input = {value:'',dataset:{key:entity?'entity_grid_voltage':'capacity_kwh',original:'old'},
      classList:{contains:cls=>cls === 'entinput' && entity,remove:()=>{dirty=false;}}};
    const button = {disabled:false,textContent:''};
    const context = {
      document:{querySelectorAll:()=>[input],getElementById:()=>button}, LANG:'en',t:key=>key,
      optionValue:el=>el.value,optionSnapshot:el=>el.value,
      alert:message=>assert.fail(message), loadSetup:async()=>{}, load:async()=>{}, onEntityInput(){},onOptionInput(){},
      fetch:async()=>({ok:true,json:async()=>({saved:{},cleared:[input.dataset.key]})})
    };
    await vm.runInNewContext(saveFunction+'\nsaveSettings()',context);
    assert.equal(dirty,false); assert.equal(input.dataset.original,'');
  });
}
test('last-report write failures remain visible after switching to advisory mode', () => {
  const r = report(); r.advisory.apply_errors = ['zone 2'];
  const s = summary(r,{...status(),apply_mode:false},now);
  assert.equal(s.mode,'shadow'); assert.equal(s.hasWriteErrors,true);
});
test('Home Assistant SOC strings are displayed while empty or unavailable states stay unknown', () => {
  const r = report();
  for (const value of ['60', '60.5']) {
    r.battery_now_pct = value;
    assert.equal(summary(r,status(),now).batteryPct,Number(value));
  }
  for (const value of ['', ' ', 'unavailable', true, 'NaN', 'Infinity']) {
    r.battery_now_pct = value;
    assert.equal(summary(r,status(),now).batteryPct,null);
  }
});
