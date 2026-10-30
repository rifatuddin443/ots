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
.ctl-transport { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--grid); }
.ctl-transport .grow { flex: 1; }
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

  // ================================================================ state
  // Module level, so a session survives switching to another page and back.
  const S = {
    cfg: { M: 16, hops: 300, sensor: "perfect", custom: { p_d: 0.9, p_fa: 0.01 } },
    sess: null,         // api/control/new response
    recs: [],           // one record per hop (api/control/step)
    algo: null,         // algorithm the transmitter should be on
    playing: false, busy: false, creating: false, speed: 3, timer: 0, error: null,
  };
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

  const sensorOf = (cfg) => (cfg.sensor === "custom" ? { id: "custom", name: "Custom", ...cfg.custom } : SENSORS().find((s) => s.id === cfg.sensor));
  const done = () => !!(S.sess && S.recs.length >= S.sess.hops);
  /** Do the chosen settings differ from the running session's? */
  function cfgChanged() {
    if (!S.sess) return true;
    const s = sensorOf(S.cfg);
    return S.sess.channels !== S.cfg.M || S.sess.hops !== S.cfg.hops
      || Math.abs(S.sess.p_d - s.p_d) > 1e-9 || Math.abs(S.sess.p_fa - s.p_fa) > 1e-9;
  }
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
      S.sess = await api(`api/control/new?channels=${S.cfg.M}&hops=${S.cfg.hops}&p_d=${s.p_d}&p_fa=${s.p_fa}`);
      S.sess.sensorName = s.id === "perfect" ? "perfect" : s.name.toLowerCase();
      S.recs = []; S.algo = null;
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
  /** Pressing an algorithm puts the radio on air with it (switching if it was on another). */
  async function pressAlgo(id) {
    if (!OTS.live || S.creating || !algoAvailable(id)) return;
    S.error = null;
    if (!S.sess || done() || (cfgChanged() && !S.recs.length) ) { if (!(await newSession())) return; }
    S.algo = id;
    if (!S.playing) play(); else refresh();
  }
  function stepOnce() {
    if (!S.algo) { S.error = "Press an algorithm first to put the radio on air."; refresh(); return; }
    pause(); advance(1);
  }

  // ================================================================ derived
  /** Running hit statistics over records [from, to). */
  function hits(from = 0, to = S.recs.length) {
    let n = 0, h = 0;
    for (let i = from; i < to; i++) { const r = S.recs[i]; if (r.hit == null) continue; n++; if (r.hit) h++; }
    return { n, h, rate: n ? h / n : null };
  }
  /** Ceiling on transmit freqs: the user's max, capped by the channel count. */
  function jamCap() {
    const base = S.sess && !cfgChanged() ? (S.sess.jam_cap || JAM_CAP_MAX) : Math.min(S.cfg.M, JAM_CAP_MAX);
    return clamp(base, 1, JAM_CAP_MAX);
  }
  /** Channels the jammer lights for record r: the top-n of its forecast ranking,
   *  n chosen from confidence (unsure spreads wider), capped by the max control. */
  function litTones(r) {
    if (!r || !r.jam || !r.jam.length) return [];
    return r.jam.slice(0, tonesFor(r.kappa, Math.min(maxTones, jamCap())));
  }
  /** Multi-tone coverage over [from,to): did a lit tone cover the true NEXT channel? */
  function jamStats(from = 0, to = S.recs.length) {
    let n = 0, c = 0, tones = 0;
    for (let i = from; i < to - 1; i++) {          // needs the following hop's truth
      const r = S.recs[i]; if (!r.jam) continue;
      const lit = litTones(r); n++; tones += lit.length;
      if (lit.includes(S.recs[i + 1].truth)) c++;
    }
    return { n, covered: c, rate: n ? c / n : null, meanTones: n ? tones / n : 0 };
  }
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
    const M = sess ? sess.channels : S.cfg.M;
    const m = { l: 46, r: 10, t: 50, b: 58 };
    const iw = Math.max(160, W - m.l - m.r);
    const N = clamp(Math.floor(iw / 18) - 1, 10, 48), colW = iw / (N + 1);
    const ih = Math.round(clamp(220 + iw * 0.18, 240, 400)), rowH = ih / M, H = m.t + ih + m.b;
    const first = t - N + 1;                         // hop number shown in column 0
    const cx = (col) => m.l + (col + 0.5) * colW, cy = (ch) => m.t + ih - (ch + 0.5) * rowH;
    const truthOn = !!OTS.prefs.truth, pathOn = !!OTS.prefs.path;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img",
      "aria-label": t ? `Hop chart, hops ${Math.max(1, first)} to ${t}: sensor detections, the system's calls and the next-hop call.` : "Hop chart: waiting for the first hop." });
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
    const acc = OTS.tok("accent"), bad = OTS.tok("serious"), det = OTS.tok("det");
    let nextLit = [];
    if (t && !done()) {
      const r = recs[t - 1];
      nextLit = litTones(r);
      const jamCol = OTS.tok("warning");
      // the extra transmit frequencies the jammer opens when unsure (the top-1 keeps
      // the cyan call ring below); more bars = lower confidence, up to the max control
      nextLit.forEach((ch, i) => {
        if (i === 0) return;
        root.append(svg("rect", { x: nx + colW * 0.18, y: cy(ch) - rowH * 0.3, width: colW * 0.64, height: Math.max(3, rowH * 0.6), rx: 2, fill: jamCol, opacity: 0.55 }));
      });
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
    if (pathOn && t >= lo) {
      let d = "";
      for (let h = lo; h <= t; h++) d += (h === lo ? "M" : "L") + cx(h - first).toFixed(1) + " " + cy(recs[h - 1].truth).toFixed(1);
      g.append(svg("path", { d, fill: "none", stroke: "rgba(232, 238, 247, 0.32)", "stroke-width": 1.25, "stroke-linejoin": "round" }));
    }
    const seen = { dot: false, hit: false, miss: false, ghost: false, lost: false, jam: nextLit.length > 1 };
    for (let h = lo; h <= t; h++) {
      const col = h - first, x = cx(col), r = recs[h - 1], x0 = m.l + col * colW;
      if (h % xStep === 0) {
        g.append(svg("line", { class: "ax", x1: x, x2: x, y1: m.t + ih, y2: m.t + ih + 4 }));
        g.append(svg("text", { class: "ax-t", x, y: m.t + ih + 16, "text-anchor": "middle" }, String(h)));
      }
      // bands: system's belief (faded until locked) and the transmitter's algorithm
      g.append(svg("rect", { x: x0 + 0.5, y: yId, width: colW - 1, height: 9, rx: 2, fill: familyColor(r.id.label), opacity: r.id.locked ? 1 : 0.35 }));
      g.append(svg("rect", { x: x0 + 0.5, y: yTx, width: colW - 1, height: 9, rx: 2, fill: algoColor(r.algo) }));
      if (r.call != null) {
        g.append(svg("circle", { cx: x, cy: cy(r.call), r: ringR, fill: "none", stroke: r.hit ? acc : bad, "stroke-width": 2, opacity: r.hit ? 1 : 0.85 }));
        if (r.hit) seen.hit = true; else seen.miss = true;
      }
      if (truthOn && !r.dets.includes(r.truth)) {
        g.append(svg("circle", { cx: x, cy: cy(r.truth), r: dotR + 2, fill: "none", stroke: "rgba(200, 215, 235, 0.8)", "stroke-width": 1.3, "stroke-dasharray": "2 2" }));
        seen.lost = true;
      }
      for (const d of r.dets) {
        if (truthOn && d !== r.truth) {
          const s = dotR, y = cy(d);
          g.append(svg("path", { d: `M${x - s} ${y - s}L${x + s} ${y + s}M${x + s} ${y - s}L${x - s} ${y + s}`, stroke: bad, "stroke-width": 1.8, "stroke-linecap": "round" }));
          seen.ghost = true;
        } else { g.append(svg("circle", { cx: x, cy: cy(d), r: dotR, fill: det })); seen.dot = true; }
      }
    }
    // markers: the real switch (truth) and the moment the system detected one
    for (let h = lo; h <= t; h++) {
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
      if (col < 0 || col >= N || h < 1 || h > t) { band.setAttribute("visibility", "hidden"); OTS.tip.hide(); return; }
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
      { value: fmt.ch(r.truth), label: r.dets.includes(r.truth) ? "true channel" : "true channel · missed" },
    ];
    if (r.call != null) rows.push({ value: fmt.ch(r.call), label: `system's call · ${sure(r.call_p)} sure · ${r.hit ? "✓ right" : "✗ wrong"}` });
    const lit = litTones(r);
    if (lit.length) rows.push({ value: lit.map(fmt.ch).join(", "), label: `jam · ${lit.length} freq${lit.length === 1 ? "" : "s"}${lit.length > 1 ? " (unsure → spread)" : " (confident → spot)"}`, color: OTS.tok("warning"), kind: "square" });
    rows.push({ value: idName(r.id.label), label: `system thinks · ${sure(r.id.p)}${r.id.locked ? " · locked" : ""}`, color: familyColor(r.id.label), kind: "square" });
    if (r.switch_detected) rows.push({ value: "⚑", label: `switch detected · started hop ${r.id.start_hop}` });
    return { title: `Hop ${r.hop}`, rows };
  }
  function hopLegend(box, seen) {
    const items = [];
    if (seen.dot) items.push({ label: OTS.prefs.truth ? "Real detection" : "Sensor detection", color: OTS.tok("det"), kind: "dot" });
    if (seen.hit) items.push({ label: "System's call · right", kind: "ring" });
    if (seen.miss) items.push({ label: "System's call · wrong", color: OTS.tok("serious"), kind: "square" });
    if (seen.ghost) items.push({ label: "Ghost (false alarm)", kind: "x" });
    if (seen.lost) items.push({ label: "Missed hop (true channel)", kind: "hollow" });
    if (seen.jam) items.push({ label: "Extra jam freqs (unsure)", color: OTS.tok("warning"), kind: "square" });
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
    const chance = 1 / (S.sess ? S.sess.channels : S.cfg.M);
    root.append(svg("line", { x1: m.l, x2: m.l + iw, y1: ya(chance), y2: ya(chance), stroke: "rgba(232, 238, 247, 0.5)", "stroke-dasharray": "5 4" }));
    root.append(svg("text", { class: "ax-t", x: m.l + iw - 4, y: ya(chance) - 4, "text-anchor": "end" }, "random"));
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
    kv.push(["Predicting", done() ? "session finished" : [el("b", {}, `channel ${fmt.ch(r.next)}`), ` next · ${sure(r.next_p)} sure`]]);
    kv.push(["Carried by", carrier]);
    if (!done()) {
      const lit = litTones(r), n = lit.length, cap = Math.min(maxTones, jamCap());
      kv.push(["Jamming next", [
        el("b", {}, `${n} frequenc${n === 1 ? "y" : "ies"}`),
        n > 1 ? ` · unsure, spread (max ${cap})` : " · confident, spot",
        el("div", { class: "ctl-tones" }, lit.map((c, i) => el("span", { class: "ctl-tone" + (i === 0 ? " lead" : "") }, fmt.ch(c)))),
      ]]);
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
    if (done()) out.push({ hop: S.recs.length, color: OTS.tok("muted"), parts: ["Session finished: ", el("b", {}, `${hits().h} of ${hits().n}`), " next hops called right"] });
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
    const sec = (label, ...kids) => el("div", { class: "ctl-psec" }, el("div", { class: "ctl-group-l" }, label), kids);
    panel.append(
      el("div", { class: "ctl-panel-h" }, el("h2", {}, "Settings"),
        el("button", { class: "btn", onclick: onClose, title: "Pop the panel out of the page (S)" }, "⇥ Pop in")),
      sec("Channels", el("div", { class: "ctl-field" }, el("div", { class: "row", style: "gap:8px" }, el("div", { class: "seg" }, chM.map((x) => x[1])), inM))),
      sec("Hops", el("div", { class: "ctl-field" }, el("div", { class: "row", style: "gap:8px" }, el("div", { class: "seg" }, chH.map((x) => x[1])), inH))),
      sec("Sensor", el("div", { class: "seg" }, sensors.map((x) => x[1])), sliders, help),
      sec("Session", el("div", { class: "ctl-actions", style: "margin-top:0" }, startBtn), el("div", { class: "ctl-help" }, status)),
      sec("Jammer · max transmit freqs", jamBox, jamHelp),
      sec("Speed", el("div", { class: "seg" }, speeds.map((x) => x[1]))),
      sec("Show on the chart", toggles.map((x) => x.node),
        el("div", { class: "ctl-help" }, "The system never sees these: they are drawn for you.")));
    let helpKey = "", btnKey = "";
    return function update() {
      toggleSel(speeds, (v) => v === S.speed);
      for (const x of toggles) x.inp.checked = !!OTS.prefs[x.key];
      toggleSel(chM, (v) => v === S.cfg.M); toggleSel(chH, (v) => v === S.cfg.hops); toggleSel(sensors, (v) => v === S.cfg.sensor);
      if (document.activeElement !== inM) inM.value = S.cfg.M;
      if (document.activeElement !== inH) inH.value = S.cfg.hops;
      sliders.hidden = S.cfg.sensor !== "custom";
      const cap = jamCap(), mt = Math.min(maxTones, cap);
      jamInp.max = cap;
      if (document.activeElement !== jamInp) jamInp.value = mt;
      jamVal.textContent = String(mt);
      OTS.clear(jamHelp).append("When the system is unsure it spreads the jam across up to ",
        el("b", {}, String(mt)), ` frequenc${mt === 1 ? "y" : "ies"}; a confident call uses one. Capped at ${cap} for ${S.cfg.M} channels.`);
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

  function makeTx(card, openPanel) {
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
    const playBtn = el("button", { class: "btn primary", onclick: () => (S.playing ? (pause(), refresh()) : play()) });
    const stepBtn = el("button", { class: "btn", onclick: stepOnce, title: "Advance one hop (→)" }, "›› 1 hop");
    const chip = el("button", { class: "chip ctl-chip", onclick: openPanel, title: "Open the settings panel (S)" });
    const clockN = el("b", { class: "num" }), clockT = document.createTextNode("");
    const msg = el("div", { class: "ctl-help", role: "status" });
    card.append(el("div", { class: "ctl-transport" }, playBtn, stepBtn, chip, el("span", { class: "grow" }),
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
      playBtn.textContent = S.playing ? "❚❚ Pause" : "▶ Run";
      playBtn.disabled = !S.algo || done() || !OTS.live;
      stepBtn.disabled = !S.algo || done() || !OTS.live;
      const s = sensorOf(S.cfg), sp = SPEEDS.find((x) => x.v === S.speed);
      chip.textContent = `⚙ ${S.cfg.M} channels · ${S.cfg.hops} hops · ${s.id === "perfect" ? "perfect" : s.name.toLowerCase()} sensor · ${sp ? sp.label : ""}`;
      clockN.textContent = String(S.recs.length).padStart(3, "0");
      clockT.textContent = "/" + (S.sess && !cfgChanged() ? S.sess.hops : S.cfg.hops);
      msg.classList.toggle("ctl-no", !!S.error);
      msg.textContent = S.error || (done() ? "Session finished. Press any algorithm to start a new one with the same settings." : "");
    };
  }
  function renderAcc(box) {
    const all = hits(), lastSw = [...S.recs].reverse().find((r) => r.switched);
    const since = lastSw ? hits(lastSw.hop - 1) : { n: 0, h: 0, rate: null };
    const M = S.sess ? S.sess.channels : S.cfg.M;
    OTS.clear(box).append(
      el("span", { class: "ctl-acc-l" }, "Next-hop prediction"),
      el("span", { class: "ctl-acc-v" }, all.n ? [el("b", {}, `${all.h} of ${all.n}`), " right · ", pct(all.rate)] : "no calls yet"),
      el("span", { class: "ctl-acc-s" }, lastSw && lastSw.hop > 1 && since.n ? `since your last switch: ${since.h} of ${since.n} (${pct(since.rate)})` : ""),
      el("div", { class: "ctl-acc-bar", title: `random guessing: 1 in ${M}` }, el("i", { style: `width:${((all.rate || 0) * 100).toFixed(1)}%` }), el("em", { style: `left:${(100 / M).toFixed(2)}%` })),
      el("span", { class: "ctl-acc-s" }, `random guessing: 1 in ${M} (${pct(1 / M)})`),
      (() => { const js = jamStats(); const cap = Math.min(maxTones, jamCap());
        return el("span", { class: "ctl-acc-s" }, js.n
          ? `multi-tone jam (≤${cap} freq${cap === 1 ? "" : "s"}) covers ${pct(js.rate)} of next hops · avg ${js.meanTones.toFixed(1)} lit`
          : ""); })());
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
      const panel = el("aside", { class: "ctl-panel", "aria-label": "Control room settings" }), tx = el("div", { class: "card" });
      const acc = el("div", { class: "card ctl-acc", "aria-label": "Next-hop prediction accuracy" });
      const hopLegendBox = el("div", { style: "min-height:26px" }), hopBox = el("div", { class: "ctl-hop" });
      const hopCard = el("div", { class: "card" }, el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, "Hop chart"),
        el("p", { class: "card-k" }, "Rings are the system's calls, made before each hop"))), hopLegendBox, hopBox,
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
        el("div", { class: "page ctl-page" }, note, tx, acc,
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
      const updatePanel = makePanel(panel, () => setOpen(false)), updateTx = makeTx(tx, () => setOpen(true));
      setOpen(panelOpen);
      ui = {
        render() {
          note.hidden = OTS.live;
          if (!OTS.live && !note.firstChild) note.append(offline());
          updatePanel(); updateTx(); renderAcc(acc); renderCharts();
          renderSystem(sys); renderLog(logBox); renderTable(tableBox);
          const sw = S.recs.filter((r) => r.switch_detected).length;
          if (sw > lastSwitch) { sys.classList.remove("ctl-flash"); void sys.offsetWidth; sys.classList.add("ctl-flash"); }
          lastSwitch = sw;
        },
      };
      lastSwitch = S.recs.filter((r) => r.switch_detected).length;
      const onKey = (e) => {
        if ((e.target instanceof Element && e.target.matches("input, select, textarea")) || e.metaKey || e.ctrlKey || e.altKey) return;
        if (e.key === " ") { e.preventDefault(); if (S.playing) { pause(); refresh(); } else play(); }
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
