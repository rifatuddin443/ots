/* Control room view: you drive the transmitter, the system has to keep up.
 *
 * Press an algorithm to put the radio on air with it; it keeps hopping with that
 * algorithm until you press another one, which switches it mid-engagement (with a
 * fresh secret). Every hop is computed live by the Python fhew.tracker.SwitchTracker
 * through `python -m dashboard.serve` (GET api/control/new, api/control/step): the
 * tracker sees only the sensor's detections and must notice the switch, name the
 * new algorithm and predict the next hop. Channels, hops and the sensor (including
 * a perfect one) are chosen in the setup card. The truth (algorithm on air, its
 * secret, the true channel) is drawn for the audience only.
 *
 * Layout: setup + transmitter cards; an accuracy banner; then two columns -- the hop
 * chart and the whole-session timeline on the left, the system's view and the event
 * log on the right; a per-algorithm table at the bottom.
 */
(function () {
  "use strict";
  const { el, svg, fmt } = OTS;

  const CSS = `
.ctl-page { container: ctl / inline-size; }
.ctl-cols { display: grid; grid-template-columns: minmax(0, 1fr) clamp(320px, 34%, 400px); gap: 18px; align-items: start; }
.ctl-col { display: flex; flex-direction: column; gap: 18px; min-width: 0; }
.ctl-shell { display: grid; grid-template-columns: minmax(0, 1fr) var(--ctl-panel-w, 330px); gap: 18px; align-items: start; }
.ctl-shell.closed { grid-template-columns: minmax(0, 1fr); }
.ctl-panel {
  position: sticky; top: 84px; max-height: calc(100vh - 100px); overflow-y: auto; padding: 16px 16px 20px;
  border: 1px solid var(--line); border-radius: var(--radius); background: var(--side);
}
.ctl-shell.closed .ctl-panel { display: none; }
.ctl-panel-h { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--grid); }
.ctl-panel-h h2 { margin: 0; font: 600 12px/1.2 var(--mono); letter-spacing: 0.14em; text-transform: uppercase; }
.ctl-psec { padding: 12px 0; border-bottom: 1px solid var(--grid); }
.ctl-psec:last-child { border-bottom: 0; }
.ctl-psec > .ctl-group-l { margin-bottom: 8px; }
.ctl-psec .seg .btn { padding: 6px 10px; }
.ctl-tab {
  position: fixed; right: 0; top: 46%; z-index: 40; writing-mode: vertical-rl; padding: 14px 7px; border-radius: 8px 0 0 8px;
  border: 1px solid rgba(34, 211, 238, 0.55); border-right: 0; background: var(--side); color: var(--accent-ink);
  font: 700 11px/1 var(--mono); letter-spacing: 0.14em; text-transform: uppercase; cursor: pointer; box-shadow: -6px 0 20px rgba(0, 0, 0, 0.35);
}
.ctl-tab:hover { background: var(--raise); }
@media (min-width: 861px) { .side-right .ctl-tab { right: var(--side-w); } }   /* clear of a right-docked page menu */
.ctl-tab:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.ctl-backdrop { display: none; }
.ctl-chip { cursor: pointer; text-transform: none; letter-spacing: 0.02em; font-weight: 600; }
.ctl-chip:hover { color: var(--ink); border-color: rgba(34, 211, 238, 0.6); }
@media (max-width: 1100px) {
  .ctl-shell, .ctl-shell.closed { grid-template-columns: minmax(0, 1fr); }
  .ctl-shell:not(.closed) .ctl-panel {
    position: fixed; right: 0; top: 0; bottom: 0; max-height: none; width: min(340px, 92vw); z-index: 60; border-radius: 0;
    box-shadow: -20px 0 50px rgba(0, 0, 0, 0.5);
  }
  .ctl-shell:not(.closed) .ctl-backdrop { display: block; position: fixed; inset: 0; z-index: 55; background: rgba(0, 0, 0, 0.45); }
}
@container ctl (max-width: 820px) { .ctl-cols { grid-template-columns: minmax(0, 1fr); } .ctl-col { display: contents; } }
.ctl-field { display: grid; grid-template-columns: minmax(0, 1fr); gap: 7px; margin-top: 2px; }
.ctl-field > span { font: 600 10.5px/1.2 var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
.ctl-field .seg .btn { padding: 6px 10px; }
.ctl-num { width: 74px; padding: 6px 8px; border-radius: 8px; border: 1px solid var(--line-2); background: var(--surface); font: 600 13px var(--mono); }
.ctl-num:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.ctl-help { font-size: 12.5px; color: var(--ink-2); margin-top: 10px; }
.ctl-help b { color: var(--ink); font-weight: 600; }
.ctl-actions { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 14px; }
.ctl-sliders { display: grid; gap: 8px; margin-top: 8px; font: 500 12px var(--mono); color: var(--ink-2); }
.ctl-sliders label { display: grid; grid-template-columns: 34px 1fr 46px; gap: 8px; align-items: center; }
.ctl-sliders input[type=range] { width: 100%; accent-color: var(--accent); }
.ctl-group { margin-top: 12px; }
.ctl-group-l { font: 600 10.5px/1.2 var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.ctl-algos { display: grid; grid-template-columns: repeat(auto-fill, minmax(138px, 1fr)); gap: 6px; }
.ctl-algo {
  position: relative; text-align: left; border: 1px solid var(--line-2); background: var(--surface); border-radius: 9px;
  padding: 8px 10px 8px 13px; cursor: pointer; display: flex; flex-direction: column; gap: 1px; min-width: 0; color: var(--ink);
}
.ctl-algo i { position: absolute; left: 0; top: 7px; bottom: 7px; width: 4px; border-radius: 0 3px 3px 0; }
.ctl-algo b { font-weight: 600; font-size: 13.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ctl-algo small { font-size: 11px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ctl-algo:hover:not(:disabled) { background: var(--raise); }
.ctl-algo:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.ctl-algo:disabled { opacity: 0.38; cursor: not-allowed; }
.ctl-algo.on { border-color: rgba(34, 211, 238, 0.75); background: var(--accent-soft); box-shadow: 0 0 22px rgba(34, 211, 238, 0.14); }
.ctl-algo.on small { color: var(--accent-ink); font-weight: 600; }
.ctl-algo.on::after { content: ""; position: absolute; right: 9px; top: 10px; width: 8px; height: 8px; border-radius: 50%; background: #ff5d5d; box-shadow: 0 0 8px #ff5d5d; animation: blink 1.1s infinite; }
.ctl-redteam { margin-top: 14px; padding: 11px 12px 13px; border: 1px solid rgba(192, 140, 240, 0.45); border-radius: 10px; background: rgba(192, 140, 240, 0.08); display: flex; flex-direction: column; gap: 4px; }
.ctl-redteam > .ctl-group-l { color: #c9a4f2; margin-bottom: 4px; }
.ctl-redteam .switch { font-size: 13px; }
.ctl-transport { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--grid); }
.ctl-transport .grow { flex: 1; }
.ctl-run { white-space: nowrap; min-width: 116px; justify-content: center; }
.ctl-clock { font: 500 12px/1 var(--mono); color: var(--muted); letter-spacing: 0.08em; white-space: nowrap; }
.ctl-clock b { font: 700 20px/1 var(--mono); color: var(--ink); }

.ctl-acc { display: flex; align-items: center; gap: 10px 18px; flex-wrap: wrap; padding: 12px 16px; }
.ctl-acc-l { font: 600 10.5px/1.2 var(--mono); letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.ctl-acc-v { font: 700 21px/1 var(--sans); color: var(--ink); }
.ctl-acc-v b { font-family: var(--mono); }
.ctl-acc-s { font-size: 12.5px; color: var(--ink-2); }
.ctl-acc-bar { flex: 1 1 160px; min-width: 120px; position: relative; height: 12px; border-radius: 6px; background: #16233a; }
.ctl-acc-bar > i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px; background: var(--accent); transition: width 0.3s ease; }
.ctl-acc-bar > em { position: absolute; top: -3px; bottom: -3px; width: 2px; background: rgba(232, 238, 247, 0.75); }

.ctl-hop svg, .ctl-tl svg { display: block; width: 100%; overflow: visible; }
.ctl-hop .ax, .ctl-tl .ax { stroke: var(--axis); stroke-width: 1; }
.ctl-hop .gl { stroke: var(--grid); stroke-width: 1; opacity: 0.55; }
.ctl-hop .ax-t, .ctl-tl .ax-t { fill: var(--muted); font: 11px var(--mono); }
.ctl-hop .ax-title { fill: var(--ink-2); font: 600 12px var(--sans); }
.ctl-hop .band-l, .ctl-tl .band-l { fill: var(--muted); font: 600 9.5px var(--mono); letter-spacing: 0.08em; }
.ctl-hop .mk-tx { fill: var(--ink); font: 700 10.5px var(--mono); }
.ctl-hop .mk-det { fill: var(--accent-ink); font: 700 10.5px var(--mono); }
.ctl-hop .nx-t { fill: var(--accent-ink); font: 700 10px var(--mono); letter-spacing: 0.12em; }
.ctl-hop .empty-t, .ctl-tl .empty-t { fill: var(--muted); font: 13px var(--sans); }
.ctl-tl .row-l { fill: var(--ink-2); font: 600 11.5px var(--sans); }
.ctl-tl .seg-t { fill: #fff; font: 600 11px var(--sans); pointer-events: none; }

.ctl-vname { display: flex; align-items: center; gap: 10px; font: 700 24px/1.15 var(--sans); letter-spacing: -0.01em; margin: 12px 0 4px; }
.ctl-vname i { width: 12px; height: 12px; border-radius: 3px; flex: none; }
.ctl-vsub { color: var(--ink-2); font-size: 13.5px; }
.ctl-vsub b { color: var(--ink); font-weight: 600; }
.ctl-kv { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 6px 12px; margin: 14px 0 0; font-size: 13px; }
.ctl-kv dt { font: 600 10.5px/1.9 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.ctl-kv dd { margin: 0; color: var(--ink); overflow-wrap: anywhere; }
.ctl-ok { color: var(--good-ink); font-weight: 600; }
.ctl-no { color: var(--critical-ink); font-weight: 600; }
.ctl-soft { color: var(--warning); font-weight: 600; }
.ctl-race { display: flex; flex-direction: column; gap: 4px; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--grid); }
.ctl-rrow { display: grid; grid-template-columns: 10px minmax(84px, 120px) minmax(40px, 1fr) 46px; gap: 9px; align-items: center; font-size: 12.5px; color: var(--ink-2); }
.ctl-rrow .sq { width: 10px; height: 10px; }
.ctl-rrow span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ctl-rtrack { position: relative; height: 8px; border-radius: 4px; background: #16233a; }
.ctl-rtrack > i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; transition: width 0.25s ease; }
.ctl-rtrack > em { position: absolute; top: -3px; bottom: -3px; width: 1.5px; background: rgba(232, 238, 247, 0.55); }
.ctl-rrow b { font: 600 12px var(--mono); text-align: right; font-variant-numeric: tabular-nums; }
.ctl-rrow.lead { color: var(--ink); }
.ctl-flash { animation: ctl-glow 1.6s ease; }
@keyframes ctl-glow { 0%, 55% { box-shadow: 0 0 0 1px var(--accent), 0 0 34px rgba(34, 211, 238, 0.32); } 100% { box-shadow: none; } }

.ctl-log { display: flex; flex-direction: column; gap: 7px; max-height: 330px; overflow-y: auto; padding-right: 4px; }
.ctl-ev { display: grid; grid-template-columns: 52px 10px minmax(0, 1fr); gap: 8px; align-items: baseline; font-size: 13px; color: var(--ink-2); }
.ctl-ev .h { font: 600 11px var(--mono); color: var(--muted); text-align: right; }
.ctl-ev .dot { width: 9px; height: 9px; border-radius: 50%; align-self: center; }
.ctl-ev b { color: var(--ink); font-weight: 600; }
.ctl-table-wrap { overflow-x: auto; }
.ctl-table td .sq { margin-right: 7px; vertical-align: -1px; }
.ctl-jam label { grid-template-columns: 1fr 34px; }
.ctl-tones { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }
.ctl-tone { font: 600 11px/1 var(--mono); padding: 3px 7px; border-radius: 4px; background: var(--raise); color: var(--ink-2); border: 1px solid var(--grid); }
.ctl-tone.lead { background: var(--accent-soft); color: var(--accent-ink); border-color: rgba(34, 211, 238, 0.5); }
.ctl-txpanel { padding-bottom: 12px; margin-bottom: 6px; border-bottom: 1px solid var(--grid); }
.ctl-txpanel .card-h { padding: 0 0 6px; border: 0; }
.ctl-txpanel .card-t { font-size: 13px; }
.ctl-txpanel .ctl-group { margin-top: 10px; }
.ctl-txpanel .ctl-transport { margin-top: 10px; }
.ctl-scrub { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
.ctl-scrub input[type=range] { flex: 1; min-width: 70px; accent-color: var(--accent); }
.ctl-scrub .btn { padding: 4px 9px; }
.ctl-scrub .ctl-help { margin: 0; white-space: nowrap; }

/* Effectiveness scoreboard: the same radio run twice (ignoring the jammer vs fighting
   back), so anyone can see at a glance how much the evasion / deception helped. */
.ctl-score { border-color: rgba(192, 140, 240, 0.5); }
.ctl-score .card-h .card-t::before { content: "🛡 "; }
.ctl-sc-cta { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-top: 4px; }
.ctl-sc-cta .btn { white-space: nowrap; }
.ctl-sc-cta .ctl-help { margin: 0; }
.ctl-sc-prog { height: 10px; border-radius: 5px; background: #16233a; overflow: hidden; margin: 14px 0 4px; }
.ctl-sc-prog > i { display: block; height: 100%; width: 0; background: linear-gradient(90deg, var(--accent), #c08cf0); transition: width 0.25s ease; }
.ctl-sc-prog-l { font: 500 12.5px var(--mono); color: var(--muted); }
.ctl-sc-verdict { font: 600 19px/1.4 var(--sans); letter-spacing: -0.01em; color: var(--ink); margin: 4px 0 2px; }
.ctl-sc-verdict b { color: var(--good-ink); font-weight: 700; }
.ctl-sc-verdict .flat { color: var(--warning); font-weight: 700; }
.ctl-sc-scen { font-size: 13px; color: var(--ink-2); margin-bottom: 14px; }
.ctl-sc-scen b { color: var(--ink); font-weight: 600; }
.ctl-sc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.ctl-sc-row { padding: 13px 15px 15px; border: 1px solid var(--grid); border-radius: 11px; background: var(--surface); }
.ctl-sc-metric { font: 600 14.5px var(--sans); color: var(--ink); }
.ctl-sc-metric .hint { display: block; font: 400 12px var(--sans); color: var(--muted); margin-top: 2px; }
.ctl-sc-bar { display: grid; grid-template-columns: 92px 1fr 48px; gap: 10px; align-items: center; margin: 9px 0 0; font-size: 12.5px; }
.ctl-sc-bar .lab { color: var(--ink-2); white-space: nowrap; }
.ctl-sc-bar .track { position: relative; height: 18px; border-radius: 5px; background: #16233a; overflow: hidden; }
.ctl-sc-bar .track > i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 5px; transition: width 0.6s cubic-bezier(.2,.7,.2,1); min-width: 2px; }
.ctl-sc-bar .track > i.bad { background: var(--serious); }
.ctl-sc-bar .track > i.good { background: var(--good-ink); }
.ctl-sc-bar b { font: 700 14px var(--mono); text-align: right; font-variant-numeric: tabular-nums; }
.ctl-sc-id { display: grid; grid-template-columns: 92px 1fr; gap: 10px; align-items: baseline; margin: 9px 0 0; font-size: 13px; }
.ctl-sc-id .lab { color: var(--ink-2); white-space: nowrap; }
.ctl-sc-id .v { font-weight: 600; }
.ctl-sc-id .v.good { color: var(--good-ink); }
.ctl-sc-id .v.bad { color: var(--critical-ink); }
.ctl-sc-delta { margin-top: 11px; padding-top: 9px; border-top: 1px solid var(--grid); font: 600 13px var(--sans); color: var(--ink-2); }
.ctl-sc-delta.good { color: var(--good-ink); }
.ctl-sc-delta.flat { color: var(--warning); }
.ctl-sc-foot { margin-top: 14px; font-size: 12.5px; color: var(--ink-2); line-height: 1.55; }
.ctl-sc-foot b { color: var(--ink); font-weight: 600; }
.ctl-sc-err { color: var(--critical-ink); font-size: 13px; margin-top: 10px; }
`;
  function injectCss() {
    if (!document.getElementById("css-ctl")) document.head.append(el("style", { id: "css-ctl" }, CSS));
  }

  // ============================================================ constants
  /** The family a correct system names for each transmitter algorithm (fhew.config.GENERATOR_TARGET_FAMILY). */
  const TARGET = {
    lfsr: "lfsr", filtered_lfsr: "filtered_lfsr", lcg: "lcg", lcg_full: "lcg_full", weak_seed: "weak_seed",
    logistic: "chaos", tent: "chaos", henon: "chaos", markov: "markov",
    lagged_fib: "unknown", quadratic_cg: "unknown", sine_map: "unknown", csprng: "unknown",
  };
  const ALGEBRAIC = new Set(["lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed"]);
  const NEEDS_POW2 = new Set(["lfsr", "filtered_lfsr"]);
  const GROUPS = [
    { label: "Known algorithms · the system has an expert for each", ids: ["lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed", "logistic", "tent", "henon", "markov"] },
    { label: "New algorithms · no expert in the bank", ids: ["lagged_fib", "quadratic_cg", "sine_map"] },
    { label: "Control · unbreakable", ids: ["csprng"] },
  ];
  const SENSORS = () => [{ id: "perfect", name: "Perfect · 100%", p_d: 1, p_fa: 0 },
    ...OTS.presets.map((p) => ({ ...p, name: p.id === "clear" ? "Strong" : p.name }))];
  const SPEEDS = [{ v: 1, label: "1/s" }, { v: 3, label: "3/s" }, { v: 10, label: "10/s" }, { v: 0, label: "Max" }];
  const CHANNEL_CHIPS = [8, 16, 32, 64], HOP_CHIPS = [100, 300, 600, 1000];
  const LIMITS = { M: [4, 64], hops: [20, 1000] };
  const ROLL = 10;   // rolling-accuracy window on the timeline

  const pow2 = (n) => n > 0 && (n & (n - 1)) === 0;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const familyColor = (fam) => OTS.hypColor(fam === "unknown" ? "H0" : fam);
  const algoColor = (id) => familyColor(TARGET[id] || "unknown");
  const algoName = (id) => OTS.emitName(id);
  const idName = (label) => (label === "unknown" ? "Unknown" : OTS.hypName(label));
  const sure = (p) => (p >= 0.9995 ? ">99.9%" : p >= 0.995 ? (p * 100).toFixed(1) + "%" : Math.round(p * 100) + "%");
  const pct = (x) => (x == null ? "—" : Math.round(x * 100) + "%");
  const JAM_CAP_MAX = 8;   // never light more than this many transmit frequencies at once
  // How many transmit frequencies to light given prediction confidence `kappa` and a
  // selectable maximum: a confident call (kappa->1) collapses to one spot tone, an
  // unsure one (kappa->0) opens out to `max`. Mirrors fhew.allocator.tones_from_confidence.
  const tonesFor = (kappa, max) => clamp(Math.round(1 + (max - 1) * (1 - clamp(kappa || 0, 0, 1))), 1, max);
  // The system calls ONE channel only when it is sure of it. Below this probability on
  // its top channel it calls a barrage instead: at least two channels together, up to
  // the "max transmit freqs" control. The barrage is both what the jammer lights and
  // what the next-hop accuracy scores (a hit = the true channel is in the barrage).
  const SURE_P = 0.9;

  // ================================================================ state
  // Module level, so a session survives switching to another page and back.
  const S = {
    cfg: { M: 16, hops: 300, sensor: "perfect", custom: { p_d: 0.9, p_fa: 0.01 },
      // red-team transmitter options (session-level: changing them starts a new session)
      evade: false, deceive: false, decoy: "lfsr", nDecoys: 1 },
    sess: null,         // api/control/new response
    recs: [],           // one record per hop (api/control/step)
    algo: null,         // algorithm the transmitter should be on
    playing: false, busy: false, creating: false, speed: 3, timer: 0, error: null,
  };
  // Effectiveness scoreboard: a self-contained A/B that runs the same radio against the
  // same jammer twice -- once passive (ignoring it), once with the chosen countermeasures --
  // so the gain is plain to anyone. It is independent of the interactive session above.
  const AB_CAP = 120;   // measurement horizon cap, so a head-to-head stays snappy
  const AB = { running: false, done: false, error: null, algo: null, hops: 0,
    cfg: null, base: null, rt: null, doneHops: 0, totalHops: 1 };
  // Families a deceiving transmitter can mimic (in-bank, so they can fool the ID).
  const DECOY_FAMILIES = [
    { id: "lfsr", name: "LFSR" }, { id: "lcg", name: "LCG" },
    { id: "weak_seed", name: "Weak-seed PRNG" }, { id: "markov", name: "Markov" },
  ];
  const MAX_DECOYS = 3;
  let ui = null;        // render hooks of the mounted page (null when unmounted)
  const PANEL_KEY = "ots.control.panel";
  let panelOpen = (() => {
    try { const v = localStorage.getItem(PANEL_KEY); if (v != null) return v === "1"; } catch (_) { /* storage unavailable */ }
    return window.innerWidth > 1100;
  })();
  // Max transmit frequencies the jammer may light at once (the "multiple transmit
  // frequency" control). Display-only: changing it re-renders from the recorded
  // forecast ranking, never re-runs the engagement.
  const MAXTONES_KEY = "ots.control.maxtones";
  let maxTones = (() => {
    try { const v = localStorage.getItem(MAXTONES_KEY); if (v != null) return clamp(Math.round(+v) || 3, 1, JAM_CAP_MAX); } catch (_) { /* storage unavailable */ }
    return 3;
  })();
  function setMaxTones(n) {
    maxTones = clamp(Math.round(n) || 1, 1, JAM_CAP_MAX);
    try { localStorage.setItem(MAXTONES_KEY, String(maxTones)); } catch (_) { /* storage unavailable */ }
    refresh();
  }
  // Hop-chart scrubbing: the last hop shown in the chart window. null = follow the
  // latest hop (the default); a number pins the view so you can scroll back through
  // past hops without stopping the run.
  let viewEnd = null;

  const sensorOf = (cfg) => (cfg.sensor === "custom" ? { id: "custom", name: "Custom", ...cfg.custom } : SENSORS().find((s) => s.id === cfg.sensor));
  const done = () => !!(S.sess && S.recs.length >= S.sess.hops);
  /** Do the chosen settings differ from the running session's? */
  function cfgChanged() {
    if (!S.sess) return true;
    const s = sensorOf(S.cfg);
    if (S.sess.channels !== S.cfg.M || S.sess.hops !== S.cfg.hops
      || Math.abs(S.sess.p_d - s.p_d) > 1e-9 || Math.abs(S.sess.p_fa - s.p_fa) > 1e-9) return true;
    // red-team options are baked into the session, so changing any of them needs a new one
    if (!!S.sess.evade !== !!S.cfg.evade || !!S.sess.deceive !== !!S.cfg.deceive) return true;
    if (S.cfg.deceive && (S.sess.decoy !== S.cfg.decoy || S.sess.n_decoys !== S.cfg.nDecoys)) return true;
    // the jammer's barrage budget is part of the loop once the adversary is engaged
    // (use the session's own cap here, not jamCap(), which would call back into us)
    if (S.cfg.evade || S.cfg.deceive) {
      const cap = S.sess.jam_cap || JAM_CAP_MAX;
      if (S.sess.max_tones !== clamp(maxTones, 1, cap)) return true;
    }
    return false;
  }
  /** Is the running session a genuine closed loop (transmitter fights back)? */
  function closedLoop() { return !!(S.sess && !cfgChanged() && (S.sess.evade || S.sess.deceive)); }
  function algoAvailable(id) {
    const M = S.sess && !cfgChanged() ? S.sess.channels : S.cfg.M;
    return pow2(M) || !NEEDS_POW2.has(id);
  }

  // ============================================================ server calls
  async function api(path) {
    const r = await fetch(path, { cache: "no-store" });
    const j = await r.json().catch(() => ({ error: r.statusText }));
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  }
  async function newSession() {
    if (!OTS.live || S.creating) return false;
    pause();
    const s = sensorOf(S.cfg);
    S.creating = true; S.error = null; refresh();
    try {
      const adv = `&evade=${S.cfg.evade ? 1 : 0}&deceive=${S.cfg.deceive ? 1 : 0}`
        + `&decoy=${encodeURIComponent(S.cfg.decoy)}&n_decoys=${S.cfg.nDecoys}&max_tones=${Math.min(maxTones, jamCap())}`;
      S.sess = await api(`api/control/new?channels=${S.cfg.M}&hops=${S.cfg.hops}&p_d=${s.p_d}&p_fa=${s.p_fa}${adv}`);
      S.sess.sensorName = s.id === "perfect" ? "perfect" : s.name.toLowerCase();
      S.recs = []; S.algo = null; viewEnd = null;
      return true;
    } catch (err) {
      S.error = "Could not start a session: " + err.message;
      return false;
    } finally {
      S.creating = false; refresh();
    }
  }
  async function advance(n) {
    if (!S.sess || !S.algo || S.busy || done()) return;
    S.busy = true; S.error = null;
    try {
      const j = await api(`api/control/step?session=${S.sess.session}&algo=${encodeURIComponent(S.algo)}&n=${n}`);
      S.recs.push(...j.records);
      if (j.done) pause();
    } catch (err) {
      S.error = err.message; pause();
    } finally {
      S.busy = false; refresh();
    }
  }

  // ============================================================== transport
  function schedule(ms) { clearTimeout(S.timer); S.timer = setTimeout(tick, ms); }
  async function tick() {
    if (!S.playing || S.busy) return;
    const t0 = performance.now(), n = S.speed ? 1 : 6;
    await advance(n);
    if (S.playing) schedule(S.speed ? Math.max(0, 1000 / S.speed - (performance.now() - t0)) : 0);
  }
  function play() {
    if (!S.algo || done()) return;
    S.playing = true; refresh(); schedule(0);
  }
  function pause() { S.playing = false; clearTimeout(S.timer); }
  /** Run / pause the hops. Every run control (the hop card, the settings panel, Space) uses this. */
  function toggleRun() { if (S.playing) { pause(); refresh(); } else play(); }
  /** Labels and enables a run/pause button from the run state. */
  function syncRun(btn) {
    btn.textContent = S.playing ? "❚❚ Pause" : (S.recs.length && !done() ? "▶ Resume" : "▶ Run");
    btn.disabled = !S.algo || done() || !OTS.live;
    btn.title = (S.playing ? "Pause the hops" : "Run the hops") + " (Space)";
  }
  /** Pressing an algorithm puts the radio on air with it (switching if it was on another).
   *  A run the user paused stays paused: only the first algorithm of a session starts it. */
  async function pressAlgo(id) {
    if (!OTS.live || S.creating || !algoAvailable(id)) return;
    S.error = null;
    if (!S.sess || done() || (cfgChanged() && !S.recs.length) ) { if (!(await newSession())) return; }
    const firstOnAir = !S.algo;
    S.algo = id;
    if (firstOnAir && !S.playing) play(); else refresh();
  }
  function stepOnce() {
    if (!S.algo) { S.error = "Press an algorithm first to put the radio on air."; refresh(); return; }
    pause(); advance(1);
  }

  // ================================================================ derived
  /** Hit tally over records [from, to), where hitOf(i) is true, false or null (no call). */
  function tally(hitOf, from, to) {
    let n = 0, h = 0;
    for (let i = from; i < to; i++) { const x = hitOf(i); if (x == null) continue; n++; if (x) h++; }
    return { n, h, rate: n ? h / n : null };
  }
  /** Next-hop accuracy over records [from, to), scored on the barrage the system called. */
  function hits(from = 0, to = S.recs.length) { return tally(barrageHit, from, to); }
  /** The strict measure, kept alongside: did the single top-1 channel match? */
  function top1(from = 0, to = S.recs.length) { return tally((i) => S.recs[i].hit, from, to); }
  /** Ceiling on transmit freqs: the user's max, capped by the channel count. */
  function jamCap() {
    const base = S.sess && !cfgChanged() ? (S.sess.jam_cap || JAM_CAP_MAX) : Math.min(S.cfg.M, JAM_CAP_MAX);
    return clamp(base, 1, JAM_CAP_MAX);
  }
  /** How many channels the system calls together for the hop after record r. Sure
   *  (top-1 >= SURE_P) -> one channel, spot. Unsure -> a barrage of at least two,
   *  growing with the uncertainty up to `max` (the "max transmit freqs" control). */
  function barrageSize(r, max) {
    const cap = clamp(max, 1, JAM_CAP_MAX);
    if (cap < 2 || r.next_p >= SURE_P) return 1;
    return clamp(tonesFor(r.kappa, cap), 2, cap);
  }
  /** The barrage for the hop after record r: the top-n of its forecast ranking. The
   *  jammer lights exactly these channels, and they are the system's call. In a
   *  closed-loop run (evasion / deception) the backend owns the barrage so the
   *  transmitter can actually dodge it -- use its plan verbatim. */
  function litTones(r) {
    if (!r) return [];
    if (r.next_lit) return r.next_lit;            // backend's plan for the next hop (closed loop)
    if (!r.jam || !r.jam.length) return [];
    return r.jam.slice(0, barrageSize(r, Math.min(maxTones, jamCap())));
  }
  /** The system's call for hop h (1-based): the barrage made from the forecast after
   *  hop h-1. Hop 1 has no forecast yet, so it has no call. */
  function callFor(h) {
    const r = S.recs[h - 1];
    if (r && r.lit) return r.lit;                 // backend's barrage that targeted hop h (closed loop)
    return h >= 2 && S.recs[h - 2] ? litTones(S.recs[h - 2]) : [];
  }
  /** Did hop i+1 (record i) land in the barrage called for it? null = no call. */
  function barrageHit(i) { return i < 1 ? null : callFor(i + 1).includes(S.recs[i].truth); }
  /** Average barrage size over the hops that have a call. */
  function meanBarrage() {
    let n = 0, s = 0;
    for (let h = 2; h <= S.recs.length; h++) { const c = callFor(h); if (c.length) { n++; s += c.length; } }
    return n ? s / n : 0;
  }
  /** Chance of a barrage hit by guessing: the average barrage over the channel count. */
  function barrageChance() { const M = S.sess ? S.sess.channels : S.cfg.M, b = meanBarrage(); return (b || 1) / M; }
  /** Transmitter regimes from the records: [{algo, start (0-based index), end (exclusive), regime}]. */
  function regimes() {
    const out = [];
    S.recs.forEach((r, i) => {
      if (r.switched || !out.length) out.push({ algo: r.algo, start: i, end: i + 1, info: r.regime || null });
      else out[out.length - 1].end = i + 1;
    });
    return out;
  }
  /** Outcome of the system's answer against the truth: ok | soft | pending | wrong. */
  function verdictFor(label, locked, algo) {
    const tgt = TARGET[algo];
    if (label === tgt) return "ok";
    if (tgt === "unknown" && !ALGEBRAIC.has(label)) return "soft";
    return locked ? "wrong" : "pending";
  }
  /** Per-regime summary rows for the table. */
  function regimeRows() {
    return regimes().map((g, k) => {
      const rs = S.recs.slice(g.start, g.end);
      const det = k > 0 ? rs.find((r) => r.switch_detected) : null;
      const from = k === 0 ? rs[0].hop : det ? det.hop : Infinity;
      const lock = rs.find((r) => r.hop >= from && r.id.locked && r.id.label === TARGET[g.algo]);
      const last = rs[rs.length - 1];
      const h = hits(g.start, g.end), tail = hits(Math.max(g.start, g.end - 30), g.end);
      return { ...g, startHop: rs[0].hop, endHop: last.hop, det, lockHops: lock ? lock.id.lock_hop - rs[0].hop + 1 : null,
        final: last.id, verdict: verdictFor(last.id.label, last.id.locked, g.algo), acc: h.rate, accTail: tail.rate };
    });
  }

  // ============================================================== hop chart
  function renderHop(box) {
    const W = box.clientWidth; if (!W) return;
    const sess = S.sess, recs = S.recs, t = recs.length;
    const end = viewEnd != null ? clamp(viewEnd, 1, t) : t;   // last hop shown (scrubbing)
    const M = sess ? sess.channels : S.cfg.M;
    const m = { l: 46, r: 10, t: 50, b: 58 };
    const iw = Math.max(160, W - m.l - m.r);
    const N = clamp(Math.floor(iw / 18) - 1, 10, 48), colW = iw / (N + 1);
    const ih = Math.round(clamp(220 + iw * 0.18, 240, 400)), rowH = ih / M, H = m.t + ih + m.b;
    const first = end - N + 1;                       // hop number shown in column 0
    const cx = (col) => m.l + (col + 0.5) * colW, cy = (ch) => m.t + ih - (ch + 0.5) * rowH;
    const truthOn = !!OTS.prefs.truth, pathOn = !!OTS.prefs.path, hideCaught = !!OTS.prefs.hideCaughtExtras;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img",
      "aria-label": t ? `Hop chart, hops ${Math.max(1, first)} to ${end}: sensor detections, the system's calls and the next-hop call.` : "Hop chart: waiting for the first hop." });
    const yStep = Math.max(1, Math.ceil(13 / rowH));
    for (let c = 0; c < M; c += yStep) {
      root.append(svg("line", { class: "gl", x1: m.l, x2: m.l + (N + 1) * colW, y1: cy(c), y2: cy(c) }));
      root.append(svg("text", { class: "ax-t", x: m.l - 8, y: cy(c) + 4, "text-anchor": "end" }, fmt.ch(c)));
    }
    root.append(svg("line", { class: "ax", x1: m.l, x2: m.l + (N + 1) * colW, y1: m.t + ih, y2: m.t + ih }));
    root.append(svg("text", { class: "ax-title", x: m.l + iw / 2, y: H - 4, "text-anchor": "middle" }, "Hop number (time →)"));
    root.append(svg("text", { class: "ax-title", x: 0, y: 0, "text-anchor": "middle", transform: `translate(13 ${m.t + ih / 2}) rotate(-90)` }, "Channel"));
    // band labels: what the system thinks (top) and what the transmitter runs (bottom)
    const yId = m.t - 15, yTx = m.t + ih + 24;
    root.append(svg("text", { class: "band-l", x: m.l - 6, y: yId + 8, "text-anchor": "end" }, "SYSTEM"));
    root.append(svg("text", { class: "band-l", x: m.l - 6, y: yTx + 8, "text-anchor": "end" }, "TX"));
    // next column: the call for the hop that has not happened yet
    const nx = m.l + N * colW;
    root.append(svg("rect", { x: nx, y: m.t, width: colW, height: ih, fill: "rgba(34, 211, 238, 0.07)" }));
    root.append(svg("line", { x1: nx, x2: nx, y1: m.t, y2: m.t + ih, stroke: "rgba(34, 211, 238, 0.35)", "stroke-dasharray": "3 3" }));
    root.append(svg("text", { class: "nx-t", x: nx + colW / 2, y: m.t - 22, "text-anchor": "middle" }, "NEXT"));
    const ringR = Math.max(4.5, Math.min(colW, rowH) * 0.42), dotR = clamp(Math.min(colW, rowH) * 0.2, 1.8, 3.8);
    const acc = OTS.tok("accent"), bad = OTS.tok("serious"), det = OTS.tok("det"), jamCol = OTS.tok("warning");
    const decoyCol = "#c08cf0";   // planted decoy detections (deception), distinct from ghosts
    // a barrage: the top channel carries the ring, the extra channels an amber bar each
    const barBar = (ch, x0, opacity) => svg("rect", { x: x0 + colW * 0.18, y: cy(ch) - rowH * 0.3, width: colW * 0.64, height: Math.max(3, rowH * 0.6), rx: 2, fill: jamCol, opacity });
    let nextLit = [];
    const showNext = end >= 1 && (end < t || !done());   // a hop after the last shown exists
    if (showNext) {
      const r = recs[end - 1];
      nextLit = litTones(r);
      nextLit.forEach((ch, i) => { if (i > 0) root.append(barBar(ch, nx, 0.55)); });
      root.append(svg("circle", { cx: nx + colW / 2, cy: cy(r.next), r: ringR, fill: "rgba(34, 211, 238, 0.18)", stroke: acc, "stroke-width": 2 }));
      if (nextLit.length > 1) root.append(svg("text", { class: "nx-t", x: nx + colW / 2, y: m.t - 8, "text-anchor": "middle", fill: jamCol }, `jam ×${nextLit.length}`));
    }
    const band = svg("rect", { x: 0, y: m.t, width: colW, height: ih, fill: "rgba(232, 238, 247, 0.06)", visibility: "hidden" });
    root.append(band);
    const clipId = "ctl-clip-" + Math.random().toString(36).slice(2, 8);
    root.append(svg("clipPath", { id: clipId }, svg("rect", { x: m.l, y: 0, width: N * colW, height: H })));
    const g = svg("g", { "clip-path": `url(#${clipId})` });
    root.append(g);
    const xStep = [1, 2, 5, 10, 20].find((s) => s * colW >= 30) || 20;
    const lo = Math.max(1, first);
    if (pathOn && end >= lo) {
      let d = "";
      for (let h = lo; h <= end; h++) d += (h === lo ? "M" : "L") + cx(h - first).toFixed(1) + " " + cy(recs[h - 1].truth).toFixed(1);
      g.append(svg("path", { d, fill: "none", stroke: "rgba(232, 238, 247, 0.32)", "stroke-width": 1.25, "stroke-linejoin": "round" }));
    }
    const seen = { dot: false, hit: false, miss: false, ghost: false, lost: false, jam: nextLit.length > 1 };
    for (let h = lo; h <= end; h++) {
      const col = h - first, x = cx(col), r = recs[h - 1], x0 = m.l + col * colW;
      if (h % xStep === 0) {
        g.append(svg("line", { class: "ax", x1: x, x2: x, y1: m.t + ih, y2: m.t + ih + 4 }));
        g.append(svg("text", { class: "ax-t", x, y: m.t + ih + 16, "text-anchor": "middle" }, String(h)));
      }
      // bands: system's belief (faded until locked) and the transmitter's algorithm
      g.append(svg("rect", { x: x0 + 0.5, y: yId, width: colW - 1, height: 9, rx: 2, fill: familyColor(r.id.label), opacity: r.id.locked ? 1 : 0.35 }));
      g.append(svg("rect", { x: x0 + 0.5, y: yTx, width: colW - 1, height: 9, rx: 2, fill: algoColor(r.algo) }));
      const bar = callFor(h);   // the barrage called for this hop (empty for hop 1)
      if (bar.length) {
        const caught = bar.includes(r.truth);
        // the extra channels, one amber bar each. With "hide on caught hops" on, a caught
        // hop drops them, except the one extra channel that made the catch (if any)
        const extras = hideCaught && caught ? bar.slice(1).filter((ch) => ch === r.truth) : bar.slice(1);
        extras.forEach((ch) => g.append(barBar(ch, x0, 0.55)));
        g.append(svg("circle", { cx: x, cy: cy(bar[0]), r: ringR, fill: "none", stroke: caught ? acc : bad, "stroke-width": 2, opacity: caught ? 1 : 0.85 }));
        if (caught) seen.hit = true; else seen.miss = true;
        if (extras.length) seen.jam = true;
      }
      if (truthOn && !r.dets.includes(r.truth)) {
        g.append(svg("circle", { cx: x, cy: cy(r.truth), r: dotR + 2, fill: "none", stroke: "rgba(200, 215, 235, 0.8)", "stroke-width": 1.3, "stroke-dasharray": "2 2" }));
        seen.lost = true;
      }
      const decoySet = r.decoys && r.decoys.length ? new Set(r.decoys) : null;
      for (const d of r.dets) {
        const y = cy(d), s = dotR + 0.6;
        if (decoySet && decoySet.has(d) && d !== r.truth) {
          // planted decoy: a hollow diamond, so it reads as a fake the system is fed
          g.append(svg("path", { d: `M${x} ${y - s}L${x + s} ${y}L${x} ${y + s}L${x - s} ${y}Z`, fill: "none", stroke: decoyCol, "stroke-width": 1.6 }));
          seen.decoy = true;
        } else if (truthOn && d !== r.truth) {
          g.append(svg("path", { d: `M${x - dotR} ${y - dotR}L${x + dotR} ${y + dotR}M${x + dotR} ${y - dotR}L${x - dotR} ${y + dotR}`, stroke: bad, "stroke-width": 1.8, "stroke-linecap": "round" }));
          seen.ghost = true;
        } else { g.append(svg("circle", { cx: x, cy: y, r: dotR, fill: det })); seen.dot = true; }
      }
    }
    // markers: the real switch (truth) and the moment the system detected one
    for (let h = lo; h <= end; h++) {
      const r = recs[h - 1], col = h - first;
      if (r.switched && h > 1) {
        const x = m.l + col * colW;
        g.append(svg("line", { x1: x, x2: x, y1: yId - 2, y2: yTx + 11, stroke: "rgba(232, 238, 247, 0.8)", "stroke-width": 1.4, "stroke-dasharray": "4 3" }));
        g.append(svg("text", { class: "mk-tx", x: x + 3, y: 12 }, "TX → " + algoName(r.algo)));
      }
      if (r.switch_detected) {
        const x = m.l + (col + 1) * colW;
        g.append(svg("line", { x1: x, x2: x, y1: yId - 2, y2: m.t + ih, stroke: acc, "stroke-width": 1.6 }));
        g.append(svg("text", { class: "mk-det", x: x - 3, y: 25, "text-anchor": "end" }, "switch detected"));
      }
    }
    if (!t) root.append(svg("text", { class: "empty-t", x: m.l + (N * colW) / 2, y: m.t + ih / 2, "text-anchor": "middle" },
      OTS.live ? "Press an algorithm to put the radio on air" : "Start the live server to drive the radio"));
    OTS.clear(box).append(root);
    // hover: one column at a time
    root.addEventListener("pointermove", (ev) => {
      const b = root.getBoundingClientRect(), x = (ev.clientX - b.left) * (W / b.width);
      const col = Math.floor((x - m.l) / colW), h = first + col;
      if (col < 0 || col >= N || h < 1 || h > end) { band.setAttribute("visibility", "hidden"); OTS.tip.hide(); return; }
      band.setAttribute("x", m.l + col * colW); band.setAttribute("visibility", "visible");
      OTS.tip.show(ev, hopTip(recs[h - 1]));
    });
    root.addEventListener("pointerleave", () => { band.setAttribute("visibility", "hidden"); OTS.tip.hide(); });
    return seen;
  }
  function hopTip(r) {
    const rows = [
      { value: algoName(r.algo), label: "transmitter (truth)", color: algoColor(r.algo), kind: "square" },
      { value: r.dets.length ? r.dets.map(fmt.ch).join(", ") : "none", label: "sensor detected" },
    ];
    if (r.decoys && r.decoys.length) rows.push({ value: r.decoys.map(fmt.ch).join(", "),
      label: `planted decoys · ${r.decoys.length} fake${r.decoys.length === 1 ? "" : "s"} fed to the sensor`, color: "#c08cf0", kind: "square" });
    rows.push({ value: fmt.ch(r.truth), label: r.dets.includes(r.truth) ? "true channel" : "true channel · missed" });
    if (r.denied != null) rows.push({ value: r.denied ? "✗ jammed" : "✓ got through",
      label: "this hop vs the barrage", color: r.denied ? OTS.tok("serious") : OTS.tok("good-ink"), kind: "square" });
    const bar = callFor(r.hop);                  // the barrage the system called for this hop
    if (bar.length) {
      const caught = bar.includes(r.truth);
      rows.push({ value: bar.map(fmt.ch).join(", "),
        label: `system's call · ${bar.length} channel${bar.length === 1 ? "" : "s"}${bar.length > 1 ? " barrage" : ""} · top ${fmt.ch(bar[0])} ${sure(r.call_p)} sure · ${caught ? "✓ caught it" : "✗ missed"}`,
        color: caught ? OTS.tok("accent") : OTS.tok("serious"), kind: "square" });
    }
    const lit = litTones(r);
    if (lit.length) {
      const nextR = S.recs[r.hop];                 // the hop this jam plans to deny
      const caught = nextR ? lit.includes(nextR.truth) : null;
      rows.push({ value: lit.map(fmt.ch).join(", "),
        label: `jam → next hop · ${lit.length} freq${lit.length === 1 ? "" : "s"}${lit.length > 1 ? " (unsure → barrage)" : " (sure → spot)"}`
          + (caught == null ? "" : caught ? " · ✓ caught it" : " · ✗ missed"),
        color: OTS.tok("warning"), kind: "square" });
    }
    rows.push({ value: idName(r.id.label), label: `system thinks · ${sure(r.id.p)}${r.id.locked ? " · locked" : ""}`, color: familyColor(r.id.label), kind: "square" });
    if (r.switch_detected) rows.push({ value: "⚑", label: `switch detected · started hop ${r.id.start_hop}` });
    return { title: `Hop ${r.hop}`, rows };
  }
  function hopLegend(box, seen) {
    const items = [];
    if (seen.dot) items.push({ label: OTS.prefs.truth ? "Real detection" : "Sensor detection", color: OTS.tok("det"), kind: "dot" });
    if (seen.hit) items.push({ label: "System's barrage · caught it", kind: "ring" });
    if (seen.miss) items.push({ label: "System's barrage · missed", color: OTS.tok("serious"), kind: "square" });
    if (seen.ghost) items.push({ label: "Ghost (false alarm)", kind: "x" });
    if (seen.decoy) items.push({ label: "Planted decoy (deception)", color: "#c08cf0", kind: "square" });
    if (seen.lost) items.push({ label: "Missed hop (true channel)", kind: "hollow" });
    if (seen.jam) items.push({ label: "Extra barrage channels (unsure)", color: OTS.tok("warning"), kind: "square" });
    OTS.clear(box).append(items.length ? OTS.chart.legend(items) : "");
  }

  // =============================================================== timeline
  function renderTimeline(box) {
    const W = box.clientWidth; if (!W) return;
    const H0 = S.sess ? S.sess.hops : S.cfg.hops, recs = S.recs, t = recs.length;
    const m = { l: 104, r: 14, t: 6, b: 30 }, iw = Math.max(120, W - m.l - m.r);
    const rowH = 20, gap = 8, yTx = m.t, yId = yTx + rowH + gap, yAcc = yId + rowH + gap + 6, accH = 96;
    const H = yAcc + accH + m.b;
    const xh = (h) => m.l + (h - 1) / H0 * iw, wh = iw / H0;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": "Whole-session timeline: the transmitter's algorithm, what the system thinks it is, and the rolling next-hop accuracy." });
    root.append(svg("text", { class: "row-l", x: m.l - 10, y: yTx + 14, "text-anchor": "end" }, "Transmitter"));
    root.append(svg("text", { class: "row-l", x: m.l - 10, y: yId + 14, "text-anchor": "end" }, "System thinks"));
    root.append(svg("text", { class: "row-l", x: m.l - 40, y: yAcc + accH / 2 - 2, "text-anchor": "end" }, "Next-hop"));
    root.append(svg("text", { class: "row-l", x: m.l - 40, y: yAcc + accH / 2 + 12, "text-anchor": "end" }, "hits"));
    root.append(svg("text", { class: "band-l", x: m.l - 40, y: yAcc + accH / 2 + 25, "text-anchor": "end" }, `LAST ${ROLL}`));
    root.append(svg("rect", { x: m.l, y: yTx, width: iw, height: rowH, rx: 3, fill: "#101a29" }));
    root.append(svg("rect", { x: m.l, y: yId, width: iw, height: rowH, rx: 3, fill: "#101a29" }));
    // accuracy axes
    const ya = (v) => yAcc + accH - v * accH;
    for (const v of [0, 0.5, 1]) {
      root.append(svg("line", { class: "ax", x1: m.l, x2: m.l + iw, y1: ya(v), y2: ya(v), opacity: v === 0 ? 1 : 0.35 }));
      root.append(svg("text", { class: "ax-t", x: m.l - 6, y: ya(v) + 4, "text-anchor": "end" }, Math.round(v * 100) + "%"));
    }
    const chance = barrageChance();
    root.append(svg("line", { x1: m.l, x2: m.l + iw, y1: ya(chance), y2: ya(chance), stroke: "rgba(232, 238, 247, 0.5)", "stroke-dasharray": "5 4" }));
    root.append(svg("text", { class: "ax-t", x: m.l + iw - 4, y: ya(chance) - 4, "text-anchor": "end" }, "random barrage"));
    for (const v of niceHops(H0, iw)) {
      root.append(svg("line", { class: "ax", x1: xh(v) + wh / 2, x2: xh(v) + wh / 2, y1: yAcc + accH, y2: yAcc + accH + 4 }));
      root.append(svg("text", { class: "ax-t", x: xh(v) + wh / 2, y: yAcc + accH + 17, "text-anchor": "middle" }, String(v)));
    }
    if (!t) {
      root.append(svg("text", { class: "empty-t", x: m.l + iw / 2, y: yAcc + accH / 2, "text-anchor": "middle" }, "The whole session will fill in here"));
      OTS.clear(box).append(root); return;
    }
    // transmitter regimes (truth)
    for (const g of regimes()) {
      const x = xh(g.start + 1), w = (g.end - g.start) * wh;
      root.append(svg("rect", { x, y: yTx, width: Math.max(1, w), height: rowH, fill: algoColor(g.algo) }));
      const name = algoName(g.algo);
      if (w > name.length * 6.6 + 10) root.append(svg("text", { class: "seg-t", x: x + 5, y: yTx + 14 }, name));
    }
    // what the system thinks: runs of (label, locked)
    let s0 = 0;
    for (let i = 1; i <= t; i++) {
      const a = recs[s0].id, b = i < t ? recs[i].id : null;
      if (b && b.label === a.label && b.locked === a.locked) continue;
      const x = xh(s0 + 1), w = (i - s0) * wh;
      root.append(svg("rect", { x, y: yId, width: Math.max(1, w), height: rowH, fill: familyColor(a.label), opacity: a.locked ? 1 : 0.35 }));
      const name = idName(a.label);
      if (a.locked && w > name.length * 6.6 + 10) root.append(svg("text", { class: "seg-t", x: x + 5, y: yId + 14 }, name));
      s0 = i;
    }
    // switch markers: detection (cyan triangles), truth (dashed white lines)
    recs.forEach((r, i) => {
      if (r.switched && i > 0) root.append(svg("line", { x1: xh(i + 1), x2: xh(i + 1), y1: yTx - 2, y2: yAcc + accH, stroke: "rgba(232, 238, 247, 0.65)", "stroke-dasharray": "3 3" }));
      if (r.switch_detected) {
        const x = xh(i + 2);
        root.append(svg("path", { d: `M${x - 5} ${yId - 7}L${x + 5} ${yId - 7}L${x} ${yId - 1}Z`, fill: OTS.tok("accent") }));
      }
    });
    // rolling accuracy
    let d = "", area = "", started = false;
    const pts = [];
    for (let i = 1; i < t; i++) {
      const lo = Math.max(1, i - ROLL + 1), st = hits(lo, i + 1);
      if (!st.n) continue;
      pts.push([xh(i + 1) + wh / 2, ya(st.rate)]);
    }
    pts.forEach(([x, y], k) => { d += (k ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1); });
    if (pts.length) {
      area = d + `L${pts[pts.length - 1][0].toFixed(1)} ${ya(0)}L${pts[0][0].toFixed(1)} ${ya(0)}Z`;
      root.append(svg("path", { d: area, fill: "rgba(34, 211, 238, 0.12)" }));
      root.append(svg("path", { d, fill: "none", stroke: OTS.tok("accent"), "stroke-width": 1.8, "stroke-linejoin": "round" }));
      started = true;
    }
    const xn = xh(t) + wh;
    root.append(svg("line", { x1: xn, x2: xn, y1: yTx - 2, y2: yAcc + accH, stroke: "rgba(34, 211, 238, 0.5)", "stroke-width": 1 }));
    // hover
    const cross = svg("line", { y1: yTx, y2: yAcc + accH, stroke: "rgba(232, 238, 247, 0.4)", visibility: "hidden" });
    const hit = svg("rect", { x: m.l, y: yTx, width: iw, height: yAcc + accH - yTx, fill: "transparent" });
    root.append(cross, hit);
    hit.addEventListener("pointermove", (ev) => {
      const b = root.getBoundingClientRect(), x = (ev.clientX - b.left) * (W / b.width);
      const h = clamp(Math.floor((x - m.l) / wh) + 1, 1, H0);
      if (h > t) { cross.setAttribute("visibility", "hidden"); OTS.tip.hide(); return; }
      cross.setAttribute("x1", xh(h) + wh / 2); cross.setAttribute("x2", xh(h) + wh / 2); cross.setAttribute("visibility", "visible");
      const r = recs[h - 1], st = hits(Math.max(1, h - ROLL), h);
      OTS.tip.show(ev, { title: `Hop ${h}`, rows: [
        { value: algoName(r.algo), label: "transmitter", color: algoColor(r.algo), kind: "square" },
        { value: idName(r.id.label), label: `system · ${sure(r.id.p)}${r.id.locked ? " · locked" : ""}`, color: familyColor(r.id.label), kind: "square" },
        { value: st.n ? pct(st.rate) : "—", label: `next-hop hits, last ${ROLL}` },
      ] });
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); OTS.tip.hide(); });
    OTS.clear(box).append(root);
    return started;
  }
  function niceHops(n, iw) {
    const step = [10, 20, 25, 50, 100, 200, 250, 500].find((s) => (iw / n) * s >= 60) || 500;
    const out = [1]; for (let v = step; v <= n; v += step) out.push(v);
    return out;
  }

  // ======================================================= system's view
  function renderSystem(card) {
    const recs = S.recs, t = recs.length, r = recs[t - 1];
    OTS.clear(card);
    card.append(el("div", { class: "card-h" }, el("div", {},
      el("p", { class: "card-t" }, "The system's view"),
      el("p", { class: "card-d" }, "From the sensor's detections only. It is never told the algorithm or when you switch."))));
    if (!r) {
      card.append(el("div", { class: "ctl-vname" }, "Waiting for the radio"),
        el("div", { class: "ctl-vsub" }, "Press an algorithm: the radio starts hopping and the system starts listening."));
      return;
    }
    const id = r.id, v = verdictFor(id.label, id.locked, r.algo);
    const recentSwitch = recs.slice(Math.max(0, t - 4)).some((x) => x.switch_detected);
    const st = recentSwitch ? ["warn", "⚑", "Switch detected"] : id.locked ? ["good", "✓", "Locked"]
      : id.label === "unknown" ? ["", "?", "No known pattern"] : ["", "◌", "Identifying"];
    card.append(el("span", { class: "status " + st[0] }, el("span", { class: "ic", "aria-hidden": "true" }, st[1]), st[2]));
    const name = id.label === "unknown" ? (r.kappa > 0.35 ? "Unknown algorithm" : "Unknown · no pattern") : OTS.hypName(id.label);
    card.append(el("div", { class: "ctl-vname" }, el("i", { style: `background:${familyColor(id.label)}` }), name));
    card.append(el("div", { class: "ctl-vsub" }, id.locked
      ? [el("b", {}, sure(id.p)), " sure · locked at hop ", el("b", {}, String(id.lock_hop))]
      : [el("b", {}, sure(id.p)), " so far · it locks at 95%"]));
    const lastTx = [...recs].reverse().find((x) => x.switched);
    const truthLine = v === "ok" ? el("span", { class: "ctl-ok" }, "✓ correct")
      : v === "soft" ? el("span", { class: "ctl-soft" }, "≈ no expert for this one; closest description")
      : v === "pending" ? el("span", { class: "ctl-soft" }, "… not yet")
      : el("span", { class: "ctl-no" }, "✗ wrong");
    const kv = [
      ["Transmitter", [el("span", { class: "sq", style: `display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;background:${algoColor(r.algo)}` }),
        el("b", {}, algoName(r.algo)), " (truth) · ", truthLine]],
      ["Since", [`hop ${id.start_hop} (system's estimate)`, lastTx ? [" · real switch at hop ", String(lastTx.hop)] : null]],
    ];
    if (id.recovered) {
      const ok = id.match === "exact" || id.match === "equivalent" || id.match === "model";
      kv.push(["Recovered", [el("span", { class: "mono" }, id.recovered), " ", el("span", { class: ok ? "ctl-ok" : "ctl-soft" },
        { exact: "✓ exact secret", equivalent: "✓ equivalent key", family: "≈ neighbouring parameter", model: "✓ model fitted", mismatch: "≠ not the exact secret" }[id.match] || "")]]);
    }
    if (lastTx && lastTx.regime && lastTx.regime.secret) kv.push(["True secret", el("span", { class: "mono" }, lastTx.regime.secret)]);
    const carrier = { bank: "the identified expert", context_mixing: "the universal learner (context mixing)", phase_space: "the phase-space predictor", recurrence: "the recurrence learner (fits a modular linear rule)" }[id.carrier] || id.carrier;
    kv.push(["Predicting", done() ? "session finished" : [el("b", {}, `channel ${fmt.ch(r.next)}`), ` top · ${sure(r.next_p)} sure`]]);
    kv.push(["Carried by", carrier]);
    if (!done()) {
      const lit = litTones(r), n = lit.length, cap = Math.min(maxTones, jamCap());
      kv.push(["Barrage next", [
        el("b", {}, `${n} channel${n === 1 ? "" : "s"} together`),
        n > 1 ? ` · unsure, barrage (max ${cap})` : " · sure, one channel",
        el("div", { class: "ctl-tones" }, lit.map((c, i) => el("span", { class: "ctl-tone" + (i === 0 ? " lead" : "") }, fmt.ch(c)))),
      ]]);
    }
    if (closedLoop()) {
      if (S.sess.evade) {
        const pr = r.pressure != null ? r.pressure : 0;
        kv.push(["Evasion", [el("b", {}, "dodging the barrage"), " · pressure ", el("b", {}, pct(pr)),
          pr > 0.45 ? el("span", { class: "ctl-soft" }, " · spreading hops (killing κ)") : ""]]);
      }
      if (S.sess.deceive) {
        const dn = r.decoys ? r.decoys.length : 0;
        kv.push(["Deception", [el("b", {}, `${S.sess.n_decoys}× ${OTS.emitName(S.sess.decoy)}`), " decoys",
          el("span", { class: "muted" }, ` · ${dn} fed this hop`)]]);
      }
      if (r.denied != null) kv.push(["This hop", r.denied
        ? el("span", { class: "ctl-no" }, "✗ jammed")
        : el("span", { class: "ctl-ok" }, "✓ got through the barrage")]);
    }
    card.append(el("dl", { class: "ctl-kv" }, kv.map(([k, val]) => [el("dt", {}, k), el("dd", {}, val)])));
    // race: marginal family posterior
    const labels = S.sess.labels, order = labels.map((l, k) => ({ l, p: r.post[k] })).sort((a, b) => b.p - a.p).slice(0, 5);
    card.append(el("div", { class: "ctl-race", "aria-label": "Which algorithm is it?" },
      el("div", { class: "ctl-group-l" }, "Which algorithm is it?"),
      order.map((o, k) => el("div", { class: "ctl-rrow" + (k === 0 ? " lead" : "") },
        el("i", { class: "sq", style: `background:${familyColor(o.l)}` }), el("span", {}, idName(o.l)),
        el("div", { class: "ctl-rtrack" }, el("i", { style: `width:${(o.p * 100).toFixed(1)}%;background:${familyColor(o.l)}` }), el("em", { style: "left:95%" })),
        el("b", {}, pct(o.p)))),
      el("div", { class: "ctl-rrow", title: "Posterior that the transmitter switched since the current estimate" },
        el("i", { class: "sq", style: "background:transparent;border:1.5px solid var(--accent)" }), el("span", {}, "Switched?"),
        el("div", { class: "ctl-rtrack" }, el("i", { style: `width:${(r.p_switch * 100).toFixed(1)}%;background:var(--accent)` }), el("em", { style: "left:90%" })),
        el("b", {}, pct(r.p_switch)))));
  }

  // ============================================================ event log
  function events() {
    const out = [];
    let prev = null;
    for (const r of S.recs) {
      if (r.switched) out.push({ hop: r.hop, color: algoColor(r.algo), parts: [r.hop === 1 ? "Radio on air: " : "You switched the radio to ",
        el("b", {}, algoName(r.algo)), r.regime && r.regime.secret ? ` (${r.regime.secret})` : ""] });
      if (r.switch_detected) {
        const real = [...S.recs.slice(0, r.hop)].reverse().find((x) => x.switched);
        const lag = real ? r.hop - real.hop : null;
        out.push({ hop: r.hop, color: OTS.tok("accent"), parts: ["System detected a switch",
          lag != null && real.hop > 1 ? [" · ", el("b", {}, `${lag} hop${lag === 1 ? "" : "s"}`), " after it happened, dated to hop ", String(r.id.start_hop)] : ""] });
      }
      if (r.id.locked && (!prev || !prev.id.locked || prev.id.label !== r.id.label)) {
        const v = verdictFor(r.id.label, true, r.algo);
        out.push({ hop: r.hop, color: v === "ok" ? OTS.tok("good-ink") : v === "soft" ? OTS.tok("warning") : OTS.tok("critical-ink"),
          parts: ["System locked: ", el("b", {}, idName(r.id.label)), ` (${sure(r.id.p)})`, v === "ok" ? " ✓" : v === "soft" ? " ≈" : " ✗"] });
      }
      prev = r;
    }
    if (closedLoop() && S.recs.length) {
      const bits = [S.sess.evade ? "evading the jammer" : null,
        S.sess.deceive ? `${S.sess.n_decoys}× ${OTS.emitName(S.sess.decoy)} decoys` : null].filter(Boolean);
      out.push({ hop: 1, color: "#c08cf0", parts: ["Red-team transmitter: ", el("b", {}, bits.join(" + "))] });
    }
    if (done()) out.push({ hop: S.recs.length, color: OTS.tok("muted"), parts: ["Session finished: ", el("b", {}, `${hits().h} of ${hits().n}`), " hops caught by the barrage"] });
    return out.reverse();
  }
  function renderLog(box) {
    const ev = events();
    OTS.clear(box).append(...(ev.length ? ev.map((e) => el("div", { class: "ctl-ev" },
      el("span", { class: "h" }, "hop " + e.hop), el("span", { class: "dot", style: `background:${e.color}` }), el("span", {}, e.parts)))
      : [el("div", { class: "ctl-help" }, "Switches, detections and locks will be listed here.")]));
  }

  // ========================================================= summary table
  function renderTable(box) {
    const rows = regimeRows();
    if (!rows.length) { OTS.clear(box).append(el("div", { class: "ctl-help" }, "One row per algorithm you put on air.")); return; }
    const V = { ok: ["ctl-ok", "✓"], soft: ["ctl-soft", "≈"], pending: ["ctl-soft", "…"], wrong: ["ctl-no", "✗"] };
    OTS.clear(box).append(el("div", { class: "ctl-table-wrap" }, el("table", { class: "table ctl-table" },
      el("thead", {}, el("tr", {}, ["Transmitter", "Hops", "Switch detected", "System's answer", "Locked after", "Next-hop hits", "Last 30"].map((h) => el("th", {}, h)))),
      el("tbody", {}, rows.map((g, k) => el("tr", {},
        el("td", {}, el("i", { class: "sq", style: `background:${algoColor(g.algo)}` }), algoName(g.algo)),
        el("td", {}, `${g.startHop}–${g.endHop}`),
        el("td", {}, k === 0 ? "—" : g.det ? `+${g.det.hop - g.startHop} hops` : "not yet"),
        el("td", {}, el("span", { class: V[g.verdict][0] }, V[g.verdict][1] + " "), idName(g.final.label), g.final.locked ? "" : el("span", { class: "muted" }, ` (${pct(g.final.p)})`)),
        el("td", {}, g.lockHops ? `${g.lockHops} hops` : "—"),
        el("td", {}, pct(g.acc)),
        el("td", {}, pct(g.accTail))))))));
  }

  // ============================================================ the cards
  // The two control cards are built once and updated in place: rebuilding them on
  // every hop would replace a button between mouse-down and mouse-up and lose the click.
  function toggleSel(btns, pred) { for (const [v, b] of btns) b.classList.toggle("sel", pred(v)); }

  function makePanel(panel, onClose) {
    const numInput = (key, [lo, hi]) => {
      const inp = el("input", { class: "ctl-num", type: "number", min: lo, max: hi, step: 1, value: S.cfg[key], "aria-label": key === "M" ? "Number of channels" : "Number of hops" });
      inp.addEventListener("change", () => {
        const v = Math.round(+inp.value);
        if (v >= lo && v <= hi) S.cfg[key] = v; else inp.value = S.cfg[key];
        refresh();
      });
      return inp;
    };
    const chips = (key, vals) => vals.map((v) => [v, el("button", { class: "btn", onclick: () => { S.cfg[key] = v; refresh(); } }, String(v))]);
    const chM = chips("M", CHANNEL_CHIPS), chH = chips("hops", HOP_CHIPS);
    const inM = numInput("M", LIMITS.M), inH = numInput("hops", LIMITS.hops);
    const sensors = [...SENSORS(), { id: "custom", name: "Custom" }].map((p) => [p.id, el("button", {
      class: "btn", onclick: () => { S.cfg.sensor = p.id; refresh(); },
      title: p.id === "custom" ? "Set the detection and false-alarm rates yourself" : `catches ${Math.round(p.p_d * 100)}% of hops, false-alarm rate ${p.p_fa} per channel`,
    }, p.name)]);
    const slider = (label, key, min, max, step) => {
      const out = el("span", { class: "num" }, S.cfg.custom[key].toFixed(3));
      const inp = el("input", { type: "range", min, max, step, value: S.cfg.custom[key], "aria-label": label });
      inp.addEventListener("input", () => { S.cfg.custom[key] = +inp.value; out.textContent = (+inp.value).toFixed(3); refresh(); });
      return el("label", {}, label, inp, out);
    };
    const sliders = el("div", { class: "ctl-sliders" }, slider("Pd", "p_d", 0.5, 1, 0.01), slider("Pfa", "p_fa", 0, 0.1, 0.005));
    const help = el("div", { class: "ctl-help" });
    // "Multiple transmit frequency" control: how many frequencies the jammer may
    // light at once. Display-only (no new session), so it updates live as you drag.
    const jamVal = el("span", { class: "num" });
    const jamInp = el("input", { type: "range", min: 1, max: jamCap(), step: 1, value: Math.min(maxTones, jamCap()), "aria-label": "Maximum transmit frequencies" });
    jamInp.addEventListener("input", () => setMaxTones(+jamInp.value));
    const jamBox = el("div", { class: "ctl-sliders ctl-jam" }, el("label", {}, jamInp, jamVal));
    const jamHelp = el("div", { class: "ctl-help" });
    const startBtn = el("button", { class: "btn", onclick: () => newSession() });
    const status = el("span", { class: "small ink2" });
    const speeds = SPEEDS.map((sp) => [sp.v, el("button", { class: "btn", onclick: () => { S.speed = sp.v; refresh(); },
      title: sp.v ? `${sp.v} hops per second` : "As fast as the tracker runs" }, sp.label)]);
    const prefToggle = (label, key) => {
      const inp = el("input", { type: "checkbox" });
      inp.addEventListener("change", () => OTS.setPref(key, inp.checked));
      return { node: el("label", { class: "switch" }, inp, label), inp, key };
    };
    const toggles = [prefToggle("Reveal sensor mistakes (ghosts, missed hops)", "truth"), prefToggle("Show the true path", "path")];
    // The red-team controls (Evade / Deceive) live in the Transmitter card at the top
    // of the panel (see makeTx), where the operator drives the radio -- not buried here.
    const hideExtras = prefToggle("Hide extra barrage channels on hops the call caught", "hideCaughtExtras");
    const hideHelp = el("div", { class: "ctl-help" }, "Clears the amber boxes where the prediction was right. If an extra channel made the catch, only that one stays. Misses and the NEXT call keep the full barrage.");
    const sec = (label, ...kids) => el("div", { class: "ctl-psec" }, el("div", { class: "ctl-group-l" }, label), kids);
    panel.append(
      el("div", { class: "ctl-panel-h" }, el("h2", {}, "Settings"),
        el("button", { class: "btn", onclick: onClose, title: "Pop the panel out of the page (S)" }, "⇥ Pop in")),
      sec("Channels", el("div", { class: "ctl-field" }, el("div", { class: "row", style: "gap:8px" }, el("div", { class: "seg" }, chM.map((x) => x[1])), inM))),
      sec("Hops", el("div", { class: "ctl-field" }, el("div", { class: "row", style: "gap:8px" }, el("div", { class: "seg" }, chH.map((x) => x[1])), inH))),
      sec("Sensor", el("div", { class: "seg" }, sensors.map((x) => x[1])), sliders, help),
      sec("Session", el("div", { class: "ctl-actions", style: "margin-top:0" }, startBtn), el("div", { class: "ctl-help" }, status)),
      sec("Barrage · max transmit freqs", jamBox, jamHelp, el("div", { style: "margin-top:10px" }, hideExtras.node), hideHelp),
      sec("Speed", el("div", { class: "seg" }, speeds.map((x) => x[1]))),
      sec("Show on the chart", toggles.map((x) => x.node),
        el("div", { class: "ctl-help" }, "The system never sees these: they are drawn for you.")));
    let helpKey = "", btnKey = "";
    return function update() {
      toggleSel(speeds, (v) => v === S.speed);
      for (const x of [...toggles, hideExtras]) x.inp.checked = !!OTS.prefs[x.key];
      toggleSel(chM, (v) => v === S.cfg.M); toggleSel(chH, (v) => v === S.cfg.hops); toggleSel(sensors, (v) => v === S.cfg.sensor);
      if (document.activeElement !== inM) inM.value = S.cfg.M;
      if (document.activeElement !== inH) inH.value = S.cfg.hops;
      sliders.hidden = S.cfg.sensor !== "custom";
      const cap = jamCap(), mt = Math.min(maxTones, cap);
      jamInp.max = cap;
      if (document.activeElement !== jamInp) jamInp.value = mt;
      jamVal.textContent = String(mt);
      OTS.clear(jamHelp).append(...(mt >= 2
        ? ["When the system is unsure of its top channel (under ", el("b", {}, `${Math.round(SURE_P * 100)}%`),
          ") it calls a barrage of ", el("b", {}, `2 to ${mt}`), " channels together; a sure call is one channel."]
        : ["The system calls one channel at a time."]),
        ` Capped at ${cap} for ${S.cfg.M} channels.`);
      const s = sensorOf(S.cfg), ghosts = s.p_fa * (S.cfg.M - 1);
      const hk = [s.p_d, s.p_fa, S.cfg.M].join("|");
      if (hk !== helpKey) {
        helpKey = hk;
        OTS.clear(help).append(...(s.p_d >= 1 && s.p_fa <= 0
          ? [el("b", {}, "Perfect sensor:"), " every hop is caught and there are no false alarms, so the system sees the exact channel sequence."]
          : ["The sensor catches ", el("b", {}, Math.round(s.p_d * 100) + "%"), " of hops and hears about ", el("b", {}, ghosts.toFixed(2)), " ghost signals per hop."]),
        ...(pow2(S.cfg.M) ? [] : [" ", el("b", {}, `${S.cfg.M} channels`), " is not a power of two, so the two shift-register algorithms are unavailable."]));
      }
      const changed = S.sess && cfgChanged();
      startBtn.className = "btn" + (changed || !S.sess ? " primary" : "");
      startBtn.disabled = !OTS.live || S.creating;
      const bk = S.creating ? "busy" : "idle";
      if (bk !== btnKey) { btnKey = bk; OTS.clear(startBtn).append(...(S.creating ? [el("span", { class: "spin" }), " Building the expert bank…"] : ["⟲ Start a new session"])); }
      status.textContent = S.creating ? "the first session for a channel count takes a few seconds"
        : changed ? "new settings apply to the next session"
        : S.sess ? `session: ${S.sess.channels} channels · ${S.sess.hops} hops · ${S.sess.sensorName || "custom"} sensor` : "";
    };
  }

  function makeTx(card, openPanel, inPanel = false) {
    const buttons = new Map();
    card.append(el("div", { class: "card-h" }, el("div", {}, el("p", { class: "card-t" }, "Transmitter"),
      el("p", { class: "card-d" }, "Press an algorithm to put the radio on air. It keeps hopping with it until you press another one, which switches it mid-test with a fresh secret."))));
    for (const grp of GROUPS) {
      card.append(el("div", { class: "ctl-group" }, el("div", { class: "ctl-group-l" }, grp.label),
        el("div", { class: "ctl-algos" }, grp.ids.map((id) => {
          const e = OTS.emitter(id) || { name: id, kind: "" }, small = el("small");
          const b = el("button", { class: "ctl-algo", onclick: () => pressAlgo(id) },
            el("i", { style: `background:${algoColor(id)}` }), el("b", {}, e.name), small);
          buttons.set(id, { b, small, e });
          return b;
        }))));
    }
    // --- Red team: let the transmitter fight back (session-level; starts a new run) ---
    const advToggle = (label, key, title) => {
      const inp = el("input", { type: "checkbox" });
      inp.addEventListener("change", () => { S.cfg[key] = inp.checked; refresh(); });
      return { node: el("label", { class: "switch", title }, inp, label), inp, key };
    };
    const evadeTog = advToggle("Evade the jammer", "evade",
      "The radio senses the jammer's barrage and hops off jammed channels, spreading out to crush the system's confidence.");
    const deceiveTog = advToggle("Deceive the sensor (decoys)", "deceive",
      "Mix coherent fake detections into the sensor stream to mislead the identification and waste jam tones.");
    const decoySel = DECOY_FAMILIES.map((d) => [d.id, el("button", { class: "btn",
      onclick: () => { S.cfg.decoy = d.id; refresh(); } }, d.name)]);
    const nDecVal = el("span", { class: "num" });
    const nDecInp = el("input", { type: "range", min: 1, max: MAX_DECOYS, step: 1, value: S.cfg.nDecoys, "aria-label": "Number of decoy streams" });
    nDecInp.addEventListener("input", () => { S.cfg.nDecoys = +nDecInp.value; refresh(); });
    const decoyBox = el("div", { class: "ctl-group", style: "margin-top:6px" },
      el("div", { class: "ctl-group-l" }, "Decoy family"), el("div", { class: "seg" }, decoySel.map((x) => x[1])),
      el("div", { class: "ctl-sliders ctl-jam", style: "margin-top:8px" }, el("label", {}, "Decoys", nDecInp, nDecVal)));
    const advHelp = el("div", { class: "ctl-help" });
    card.append(el("div", { class: "ctl-redteam" },
      el("div", { class: "ctl-group-l" }, "Red team · make the transmitter fight back"),
      evadeTog.node, deceiveTog.node, decoyBox, advHelp));
    const playBtn = el("button", { class: "btn primary", onclick: toggleRun });
    const stepBtn = el("button", { class: "btn", onclick: stepOnce, title: "Advance one hop (→)" }, "›› 1 hop");
    const chip = el("button", { class: "chip ctl-chip", onclick: openPanel, title: "Open the settings panel (S)" });
    const clockN = el("b", { class: "num" }), clockT = document.createTextNode("");
    const msg = el("div", { class: "ctl-help", role: "status" });
    card.append(el("div", { class: "ctl-transport" }, playBtn, stepBtn, ...(inPanel ? [] : [chip]), el("span", { class: "grow" }),
      el("span", { class: "ctl-clock" }, "HOP ", clockN, clockT)), msg);
    return function update() {
      for (const [id, { b, small, e }] of buttons) {
        const ok = algoAvailable(id), on = S.algo === id && !done();
        b.classList.toggle("on", on);
        b.disabled = !OTS.live || !ok || S.creating;
        b.title = ok ? e.blurb : "Needs a power-of-two channel count (4, 8, 16, 32 or 64)";
        b.setAttribute("aria-pressed", on ? "true" : "false");
        small.textContent = on ? (S.playing ? "on air" : "on air · paused") : ok ? e.kind : "needs 2ⁿ channels";
      }
      syncRun(playBtn);
      stepBtn.disabled = !S.algo || done() || !OTS.live;
      // red-team controls
      evadeTog.inp.checked = !!S.cfg.evade;
      deceiveTog.inp.checked = !!S.cfg.deceive;
      toggleSel(decoySel, (v) => v === S.cfg.decoy);
      for (const [id, b] of decoySel) b.disabled = !(pow2(S.cfg.M) || id !== "lfsr");
      decoyBox.hidden = !S.cfg.deceive;
      if (document.activeElement !== nDecInp) nDecInp.value = S.cfg.nDecoys;
      nDecVal.textContent = String(S.cfg.nDecoys);
      const decoyName = (DECOY_FAMILIES.find((d) => d.id === S.cfg.decoy) || {}).name || S.cfg.decoy;
      const changedAdv = S.sess && cfgChanged() && (S.cfg.evade || S.cfg.deceive
        || (S.sess.evade || S.sess.deceive));
      OTS.clear(advHelp).append(...(
        !S.cfg.evade && !S.cfg.deceive
          ? ["The transmitter ignores the jammer. Turn these on to let it ", el("b", {}, "fight back"), "."]
          : [
            ...(S.cfg.evade ? [el("b", {}, "Evasion: "), "dodges the barrage, spreading its hops. "] : []),
            ...(S.cfg.deceive ? [el("b", {}, "Deception: "), el("b", {}, `${S.cfg.nDecoys}× ${decoyName}`), " decoy", S.cfg.nDecoys === 1 ? "" : "s", " into the sensor stream. "] : []),
            el("span", { class: changedAdv ? "ctl-soft" : "muted" }, changedAdv ? "Press an algorithm to start a session with this." : "Active."),
          ]));
      const s = sensorOf(S.cfg), sp = SPEEDS.find((x) => x.v === S.speed);
      if (!inPanel) chip.textContent = `⚙ ${S.cfg.M} channels · ${S.cfg.hops} hops · ${s.id === "perfect" ? "perfect" : s.name.toLowerCase()} sensor · ${sp ? sp.label : ""}`;
      clockN.textContent = String(S.recs.length).padStart(3, "0");
      clockT.textContent = "/" + (S.sess && !cfgChanged() ? S.sess.hops : S.cfg.hops);
      msg.classList.toggle("ctl-no", !!S.error);
      msg.textContent = S.error || (done() ? "Session finished. Press any algorithm to start a new one with the same settings." : "");
    };
  }
  function renderAcc(box) {
    const all = hits(), top = top1(), lastSw = [...S.recs].reverse().find((r) => r.switched);
    const since = lastSw ? hits(lastSw.hop - 1) : { n: 0, h: 0, rate: null };
    const chance = barrageChance(), avg = meanBarrage();
    const loop = closedLoop();
    // In a closed loop a barrage hit = the true channel was jammed, so this IS the
    // jammer's denial rate; the transmitter's job (evasion / deception) is to push it down.
    const advBits = loop ? [S.sess.evade ? "evasion" : null, S.sess.deceive ? `${S.sess.n_decoys}× ${S.sess.decoy} decoys` : null].filter(Boolean) : [];
    OTS.clear(box).append(
      el("span", { class: "ctl-acc-l" }, loop ? "Jammer denial" : "Next-hop barrage"),
      el("span", { class: "ctl-acc-v" }, all.n ? [el("b", {}, `${all.h} of ${all.n}`), loop ? " jammed · " : " hit · ", pct(all.rate)] : "no calls yet"),
      el("span", { class: "ctl-acc-s" }, all.n ? `exact top-1: ${top.h} of ${top.n} (${pct(top.rate)}) · barrage avg ${avg.toFixed(1)} channels` : ""),
      loop && advBits.length
        ? el("span", { class: "ctl-acc-s", style: "color:var(--accent-ink);font-weight:600" }, `transmitter fighting back · ${advBits.join(" + ")}`)
        : el("span", { class: "ctl-acc-s" }, lastSw && lastSw.hop > 1 && since.n ? `since your last switch: ${since.h} of ${since.n} (${pct(since.rate)})` : ""),
      el("div", { class: "ctl-acc-bar", title: `random guessing of a barrage this size: ${pct(chance)}` },
        el("i", { style: `width:${((all.rate || 0) * 100).toFixed(1)}%` }), el("em", { style: `left:${(chance * 100).toFixed(2)}%` })),
      el("span", { class: "ctl-acc-s" }, `random guessing: ${pct(chance)}`));
  }

  // ===================================================== effectiveness A/B
  /** Reduce one arm's records to the plain outcomes the scoreboard reports. */
  function abAgg(recs, algo) {
    const scored = recs.filter((r) => r.hop > 1);
    const den = scored.filter((r) => r.denied != null);
    const t1 = scored.filter((r) => r.hit != null);
    const last = recs[recs.length - 1];
    const tgt = TARGET[algo];
    const litHops = recs.filter((r) => r.lit && r.lit.length);
    return {
      denied: den.length ? den.filter((r) => r.denied).length / den.length : null,
      top1: t1.length ? t1.filter((r) => r.hit).length / t1.length : null,
      locked: !!(last && last.id.locked),
      lockedCorrect: !!(last && last.id.locked && last.id.label === tgt),
      label: last ? last.id.label : "unknown",
      lockHop: last ? last.id.lock_hop : null,
      decoys: recs.reduce((s, r) => s + (r.decoys ? r.decoys.length : 0), 0),
      meanTones: litHops.length ? litHops.reduce((s, r) => s + r.lit.length, 0) / litHops.length : 0,
    };
  }
  /** Step one fresh session (built from `query`) to `hops`, reporting progress. */
  async function abRunArm(query, hops) {
    const sess = await api(`api/control/new?${query}`);
    const recs = [];
    while (recs.length < hops) {
      const j = await api(`api/control/step?session=${sess.session}&algo=${encodeURIComponent(AB.algo)}&n=25`);
      recs.push(...j.records);
      AB.doneHops += j.records.length; refresh();
      if (j.done || !j.records.length) break;
    }
    return { sess, recs };
  }
  /** Run the head-to-head: the same radio, same jammer, passive vs fighting back. */
  async function measureEffectiveness() {
    if (!OTS.live || AB.running) return;
    pause();
    const M = S.cfg.M, hops = Math.min(S.cfg.hops, AB_CAP), s = sensorOf(S.cfg);
    const algo = S.algo && algoAvailable(S.algo) ? S.algo : (pow2(M) ? "lfsr" : "markov");
    // measure the levers the operator has chosen; if none, preview both so the effect shows
    let evade = !!S.cfg.evade, deceive = !!S.cfg.deceive, decoy = S.cfg.decoy;
    const defaulted = !evade && !deceive;
    if (defaulted) {
      evade = true; deceive = true;
      // preview a decoy family that differs from the radio under test, so deception can
      // actually mislead the identification (a decoy of the same family just reinforces it)
      decoy = ["lcg", "weak_seed", "markov", "lfsr"].find((d) => d !== TARGET[algo] && (d !== "lfsr" || pow2(M))) || "lcg";
    }
    const nDecoys = S.cfg.nDecoys, mt = Math.min(maxTones, Math.min(M, JAM_CAP_MAX));
    Object.assign(AB, { running: true, done: false, error: null, algo, hops,
      cfg: { evade, deceive, decoy, nDecoys, defaulted, M, hops, capped: S.cfg.hops > AB_CAP,
        sensorName: s.id === "perfect" ? "perfect" : s.name.toLowerCase() },
      base: null, rt: null, doneHops: 0, totalHops: 2 * hops });
    refresh();
    try {
      const common = `channels=${M}&hops=${hops}&p_d=${s.p_d}&p_fa=${s.p_fa}&max_tones=${mt}`;
      const base = await abRunArm(`${common}&evade=0&deceive=0&jammer_live=1`, hops);
      AB.base = abAgg(base.recs, algo);
      // the red-team run faces the SAME scenario (same seed), so only the countermeasure differs
      const rt = await abRunArm(`${common}&evade=${evade ? 1 : 0}&deceive=${deceive ? 1 : 0}`
        + `&decoy=${encodeURIComponent(decoy)}&n_decoys=${nDecoys}&seed=${base.sess.seed}`, hops);
      AB.rt = abAgg(rt.recs, algo);
      AB.done = true;
    } catch (err) {
      AB.error = err.message;
    } finally {
      AB.running = false; refresh();
    }
  }
  /** Plain-English headline: what fighting back actually bought you. */
  function scoreVerdict() {
    const b = AB.base, r = AB.rt, wins = [];
    if (b.denied != null && r.denied != null && b.denied - r.denied > 0.08) {
      const rel = Math.round(((b.denied - r.denied) / Math.max(b.denied, 1e-6)) * 100);
      wins.push(["cut the jammer's hits by ", el("b", {}, `${rel}%`)]);
    }
    if (b.lockedCorrect && !r.lockedCorrect) wins.push(["stopped it from identifying your radio"]);
    else if (b.lockedCorrect && r.lockedCorrect === false && r.locked) wins.push(["tricked it into the wrong radio"]);
    if (!wins.length) {
      return el("div", { class: "ctl-sc-verdict" }, "Fighting back ", el("span", { class: "flat" }, "made little difference"),
        " in this run — try a more predictable algorithm (LFSR, LCG, weak-seed) or turn on both levers.");
    }
    const parts = ["Fighting back "];
    wins.forEach((w, i) => { if (i) parts.push(wins.length > 2 || i < wins.length - 1 ? ", " : " and "); parts.push(...w); });
    parts.push(".");
    return el("div", { class: "ctl-sc-verdict" }, parts);
  }
  function denialBar(label, val, cls) {
    const pctv = val == null ? null : Math.round(val * 100);
    return el("div", { class: "ctl-sc-bar" }, el("span", { class: "lab" }, label),
      el("div", { class: "track" }, el("i", { class: cls, style: `width:${val == null ? 0 : (val * 100).toFixed(1)}%` })),
      el("b", {}, pctv == null ? "—" : `${pctv}%`));
  }
  /** One arm's identification outcome as a coloured verdict (bad for you = jammer won). */
  function idOutcome(a, forYouGood) {
    if (a.lockedCorrect) return { txt: `✓ Yes — ${idName(a.label)}${a.lockHop ? `, by hop ${a.lockHop}` : ""}`, good: false };
    if (a.locked) return { txt: `✗ Fooled — locked onto ${idName(a.label)} (a decoy)`, good: true };
    return { txt: "✗ No — never locked on", good: true };
  }
  function renderScore(box) {
    OTS.clear(box);
    box.append(el("div", { class: "card-h" }, el("div", {},
      el("p", { class: "card-t" }, "Did it work?"),
      el("p", { class: "card-d" }, "Runs the same radio against the same jammer twice — once ignoring it, once fighting back — so you can see the difference at a glance."))));
    const body = el("div");
    box.append(body);
    const levers = () => {
      const c = AB.cfg || { evade: !!S.cfg.evade || (!S.cfg.evade && !S.cfg.deceive), deceive: !!S.cfg.deceive || (!S.cfg.evade && !S.cfg.deceive),
        decoy: S.cfg.decoy, nDecoys: S.cfg.nDecoys, defaulted: !S.cfg.evade && !S.cfg.deceive };
      const bits = [c.evade ? "evasion" : null, c.deceive ? `${c.nDecoys}× ${(DECOY_FAMILIES.find((d) => d.id === c.decoy) || {}).name || c.decoy} decoys` : null].filter(Boolean);
      return bits.join(" + ");
    };
    if (!OTS.live) { body.append(el("div", { class: "ctl-help" }, "Start the live server to measure effectiveness.")); return; }
    // call to action + progress
    const algoNow = S.algo && algoAvailable(S.algo) ? S.algo : (pow2(S.cfg.M) ? "lfsr" : "markov");
    const btn = el("button", { class: "btn primary", onclick: measureEffectiveness, disabled: AB.running },
      AB.running ? [el("span", { class: "spin" }), " Measuring…"] : AB.done ? "⟲ Measure again" : "▶ Measure how much it helped");
    body.append(el("div", { class: "ctl-sc-cta" }, btn,
      el("span", { class: "ctl-help" }, AB.running ? "" : ["Tests ", el("b", { style: "color:var(--ink)" }, algoName(algoNow)),
        " with ", el("b", { style: "color:var(--ink)" }, levers()), (!S.cfg.evade && !S.cfg.deceive) ? " (preview — toggle levers in the Transmitter card)" : ""])));
    if (AB.running) {
      const frac = AB.totalHops ? AB.doneHops / AB.totalHops : 0;
      body.append(el("div", { class: "ctl-sc-prog" }, el("i", { style: `width:${(frac * 100).toFixed(0)}%` })),
        el("div", { class: "ctl-sc-prog-l" }, `running ${AB.doneHops < AB.hops ? "the passive radio" : "the radio fighting back"} · ${Math.min(AB.doneHops, AB.totalHops)} / ${AB.totalHops} hops`));
      return;
    }
    if (AB.error) { body.append(el("div", { class: "ctl-sc-err" }, "Could not measure: " + AB.error)); return; }
    if (!AB.done || !AB.base || !AB.rt) {
      body.append(el("div", { class: "ctl-sc-foot" }, "Press ", el("b", {}, "Measure"),
        " to run the head-to-head. The jammer is identical in both runs; the only thing that changes is whether your transmitter fights back."));
      return;
    }
    // --- results ---
    const b = AB.base, r = AB.rt, c = AB.cfg;
    body.append(scoreVerdict(),
      el("div", { class: "ctl-sc-scen" }, "Same radio (", el("b", {}, algoName(AB.algo)), "), same jammer, same ",
        el("b", {}, `${AB.hops} hops`), ", ", el("b", {}, c.sensorName + " sensor"), " — with ", el("b", {}, levers()),
        c.defaulted ? " (both previewed)" : "", "."));
    const rows = el("div", { class: "ctl-sc-grid" });
    // Row 1 -- how often the jammer landed on you (evasion's headline)
    const denDrop = b.denied != null && r.denied != null ? b.denied - r.denied : null;
    const denCls = denDrop != null && denDrop > 0.08 ? "good" : "flat";
    rows.append(el("div", { class: "ctl-sc-row" },
      el("div", { class: "ctl-sc-metric" }, "Jammer jammed your signal",
        el("span", { class: "hint" }, "how often the jammer sat on the channel you used")),
      denialBar("Ignoring it", b.denied, "bad"), denialBar("Fighting back", r.denied, "good"),
      el("div", { class: "ctl-sc-delta " + denCls }, denDrop == null ? "—"
        : denDrop > 0.08 ? `↓ ${Math.round((denDrop / Math.max(b.denied, 1e-6)) * 100)}% fewer hits — the radio slipped the barrage`
        : "about the same — this lever works on identification, not dodging")));
    // Row 2 -- could the jammer identify your radio (deception's headline)
    const bo = idOutcome(b), ro = idOutcome(r);
    const idWin = (b.lockedCorrect && !r.lockedCorrect);
    rows.append(el("div", { class: "ctl-sc-row" },
      el("div", { class: "ctl-sc-metric" }, "Jammer identified your radio",
        el("span", { class: "hint" }, "could it name the algorithm and lock onto it?")),
      el("div", { class: "ctl-sc-id" }, el("span", { class: "lab" }, "Ignoring it"), el("span", { class: "v " + (bo.good ? "good" : "bad") }, bo.txt)),
      el("div", { class: "ctl-sc-id" }, el("span", { class: "lab" }, "Fighting back"), el("span", { class: "v " + (ro.good ? "good" : "bad") }, ro.txt)),
      el("div", { class: "ctl-sc-delta " + (idWin ? "good" : "flat") }, idWin
        ? (r.locked ? "the jammer committed to the wrong radio" : "the jammer could never commit")
        : b.lockedCorrect ? "the jammer still pinned it this run" : "the jammer couldn't pin it either way")));
    body.append(rows);
    // footer: the quieter wins (decoys swallowed, a wrong commit, spread, exact-guess collapse)
    const foot = [];
    if (c.deceive && r.decoys > 0) {
      const bits = ["The jammer swallowed ", el("b", {}, `${r.decoys} fake detections`), "."];
      if (r.locked && !r.lockedCorrect) bits.push(" It committed to ", el("b", {}, idName(r.label)),
        " — one of your decoys — instead of your real ", el("b", {}, algoName(AB.algo)), ".");
      foot.push(bits);
    }
    if (r.meanTones - b.meanTones > 0.3) foot.push(["Forced to spread: the jammer lit ", el("b", {}, `${r.meanTones.toFixed(1)} frequencies`),
      " per hop vs ", el("b", {}, b.meanTones.toFixed(1)), " before — more power spent, less focus."]);
    if (b.top1 != null && r.top1 != null) foot.push(["Its exact next-channel guess fell from ",
      el("b", {}, pct(b.top1)), " to ", el("b", {}, pct(r.top1)), " (a blind guess would be ≈ ", pct(1 / c.M), ")."]);
    if (foot.length) body.append(el("div", { class: "ctl-sc-foot" }, foot.map((f) => el("span", {}, f, " "))));
  }

  function offline() {
    return el("div", { class: "card accent" },
      el("p", { class: "card-t" }, "Live engine needed"),
      el("p", { class: "card-k" }, "The control room drives the real Python tracker hop by hop."),
      el("p", { class: "card-d" }, "From the repository root run ", el("code", {}, "python -m dashboard.serve"),
        " and open ", el("code", {}, "http://127.0.0.1:8050/#/control"), ". The recorded pages work offline; this one computes every hop live."));
  }

  function refresh() { if (ui) ui.render(); }

  // ================================================================= view
  OTS.views.register({
    id: "control", nav: "Control room", title: "Control room", hint: "Drive the transmitter yourself",
    subtitle: "Switch the radio's algorithm mid-test: the system must notice, re-identify and keep predicting",
    icon: "M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6", usesPlayer: false,
    mount(root) {
      injectCss();
      const note = el("div");
      const panel = el("aside", { class: "ctl-panel", "aria-label": "Control room settings" }), tx = el("div", { class: "ctl-txpanel" });
      const acc = el("div", { class: "card ctl-acc", "aria-label": "Next-hop barrage accuracy" });
      const scoreBox = el("div", { class: "card ctl-score", "aria-label": "Countermeasure effectiveness scoreboard" });
      const hopLegendBox = el("div", { style: "min-height:26px" }), hopBox = el("div", { class: "ctl-hop" });
      const scrubBox = el("div", { class: "ctl-scrub", "aria-label": "Scroll back through past hops" });
      // the run/pause control sits on the chart itself, so the hops can be stopped at a glance
      const runBtn = el("button", { class: "btn primary ctl-run", onclick: toggleRun, "aria-label": "Run or pause the hops" });
      const hopCard = el("div", { class: "card" }, el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, "Hop chart"),
        el("p", { class: "card-k" }, "Rings are the system's barrage, made before each hop; amber bars are its extra channels")), runBtn),
        hopLegendBox, hopBox, scrubBox,
        el("div", { class: "card-foot" }, "Top band: what the system thinks (faded until it locks). Bottom band: what the transmitter really runs. The system never sees the truth overlays."));
      const tlBox = el("div", { class: "ctl-tl" });
      const tlCard = el("div", { class: "card" }, el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, "The whole session"),
        el("p", { class: "card-k" }, "Truth vs belief: every switch, how fast it was caught"),
        el("p", { class: "card-d" }, "Dashed lines are your switches; cyan triangles are the moments the system detected one."))), tlBox);
      const sys = el("div", { class: "card" });
      const logBox = el("div", { class: "ctl-log", "aria-live": "polite" });
      const logCard = el("div", { class: "card" }, el("div", { class: "card-h" }, el("div", {}, el("p", { class: "card-t" }, "Event log"))), logBox);
      const tableBox = el("div");
      const tableCard = el("div", { class: "card" }, el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, "Per algorithm"), el("p", { class: "card-k" }, "How each algorithm you put on air was handled"))), tableBox);
      // The panel sits beside the page, outside its CSS container: a query container
      // is the containing block of position:fixed children, which the narrow-screen
      // drawer and the pop-out tab need to escape.
      const tab = el("button", { class: "ctl-tab", onclick: () => setOpen(true), title: "Open settings (S)" }, "⚙ Settings");
      const backdrop = el("div", { class: "ctl-backdrop", onclick: () => setOpen(false) });
      const shell = el("div", { class: "ctl-shell" },
        el("div", { class: "page ctl-page" }, note, acc, scoreBox,
          el("div", { class: "ctl-cols" }, el("div", { class: "ctl-col" }, hopCard, tlCard), el("div", { class: "ctl-col" }, sys, logCard)),
          tableCard),
        panel, backdrop, tab);
      root.append(shell);
      function setOpen(open) {
        panelOpen = open;
        try { localStorage.setItem(PANEL_KEY, open ? "1" : "0"); } catch (_) { /* storage unavailable */ }
        shell.classList.toggle("closed", !open);
        tab.hidden = open;
        renderCharts();
      }

      let lastSwitch = 0;
      function renderCharts() {
        hopLegend(hopLegendBox, renderHop(hopBox) || {});
        renderTimeline(tlBox);
      }
      // Hop-chart scrubber: scroll back through past hops without stopping the run.
      const scrubInput = el("input", { type: "range", min: 1, max: 1, step: 1, value: 1, "aria-label": "Scroll back through past hops" });
      const scrubLatest = el("button", { class: "btn", title: "Jump back to the latest hop", "aria-label": "Jump to the latest hop" }, "⏭");
      const scrubLabel = el("span", { class: "ctl-help" });
      scrubBox.append(scrubLatest, scrubInput, scrubLabel);
      scrubInput.addEventListener("input", () => {
        const v = +scrubInput.value;
        viewEnd = v >= S.recs.length ? null : v;     // at the end = resume following
        renderCharts(); updateScrub();
      });
      scrubLatest.addEventListener("click", () => { viewEnd = null; renderCharts(); updateScrub(); });
      function updateScrub() {
        const t = S.recs.length, en = viewEnd != null ? clamp(viewEnd, 1, t) : t;
        scrubInput.max = Math.max(1, t);
        scrubInput.disabled = t < 2;
        if (document.activeElement !== scrubInput) scrubInput.value = en;
        scrubLatest.disabled = viewEnd == null || t < 2;
        scrubLabel.textContent = !t ? "no hops yet"
          : viewEnd != null ? `viewing up to hop ${en} of ${t} (scrolled back)`
          : `following the latest hop (${t})`;
      }
      const updatePanel = makePanel(panel, () => setOpen(false)), updateTx = makeTx(tx, () => setOpen(true), true);
      panel.prepend(tx);                              // the transmitter (algo select) lives in the side panel
      setOpen(panelOpen);
      ui = {
        render() {
          note.hidden = OTS.live;
          if (!OTS.live && !note.firstChild) note.append(offline());
          updatePanel(); updateTx(); syncRun(runBtn); renderAcc(acc); renderScore(scoreBox); renderCharts(); updateScrub();
          renderSystem(sys); renderLog(logBox); renderTable(tableBox);
          const sw = S.recs.filter((r) => r.switch_detected).length;
          if (sw > lastSwitch) { sys.classList.remove("ctl-flash"); void sys.offsetWidth; sys.classList.add("ctl-flash"); }
          lastSwitch = sw;
        },
      };
      lastSwitch = S.recs.filter((r) => r.switch_detected).length;
      const onKey = (e) => {
        if ((e.target instanceof Element && e.target.matches("input, select, textarea")) || e.metaKey || e.ctrlKey || e.altKey) return;
        if (e.key === " ") { e.preventDefault(); toggleRun(); }
        else if (e.key === "ArrowRight") stepOnce();
        else if (e.key.toLowerCase() === "t") OTS.setPref("truth", !OTS.prefs.truth);
        else if (e.key.toLowerCase() === "s") setOpen(!panelOpen);
        else if (e.key === "Escape" && panelOpen && window.innerWidth <= 1100) setOpen(false);
      };
      document.addEventListener("keydown", onKey);
      const ro = new ResizeObserver(() => renderCharts());
      ro.observe(hopBox); ro.observe(tlBox);
      const subs = [OTS.on("live", refresh), OTS.on("prefs", () => { updatePanel(); renderCharts(); })];
      ui.render();
      return () => {
        pause(); ui = null;
        document.removeEventListener("keydown", onKey);
        ro.disconnect();
        for (const off of subs) off();
        OTS.tip.hide();
      };
    },
  });
})();
