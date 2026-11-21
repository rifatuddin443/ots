/* Proof view: the paper's latest evaluation, one claim per card (and per slide).
 *
 * A static page (no player). Every number comes from OTS_DATA.paper, which
 * dashboard/paper.py condenses from the paper's saved experiment files in
 * docs/paper/ieee/data (plus three results quoted from the write-up, marked as
 * such), or from OTS_DATA.proof (the dashboard's own ablation reruns).
 *
 *   overview   headline tiles + the five chapters
 *   Name the radio        1 confusion matrix per sensor · 2 vs conventional identifiers
 *   Why it works          3 sensor model ablation + textbook BM · 4 wrong sensor estimate · 5 converse
 *   Call the next channel 6 new algorithms · 7 vs outside predictors (LSTM, PPM-C) · 8 conformal sets
 *   Act on it             9 hops denied · 10 algorithm switches · 11 compute time
 *   Honest limits         12 a radio that fights back · 13 what it does not do yet
 *
 * Each card has its own sensor switch where the experiment ran at several sensors
 * (default: Typical). "Show data tables" (OTS.prefs.tables) adds a table under each
 * card. Each card is a slide for OTS.present (the "Present" button, or P).
 */
(function () {
  "use strict";
  const { el, svg, fmt } = OTS;

  /** Sensor shown by each card, kept across visits. */
  const sel = { conf: "typical", idb: "typical", pred: "typical", pbase: "typical", cp: "typical", deny: "typical", sw: "typical" };

  // ------------------------------------------------------------------ styles
  const CSS = `
.proof { gap: 20px; }
.pf-hero { padding: 26px 30px 24px; }
.pf-hero .lead { font-size: clamp(28px, 2.7vw, 40px); max-width: 30ch; margin-top: 10px; }
.pf-hero .intro { font-size: 16px; max-width: 86ch; }
.pf-tiles { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }
.pf-tiles .tile { display: flex; flex-direction: column; }
.pf-tiles .tile-v { font-size: 32px; color: var(--accent-ink); }
.pf-tiles .tile-s { line-height: 1.4; }
.pf-tiles .tile-x { margin-top: auto; padding-top: 8px; font-size: 12px; color: var(--muted); }
.pf-agenda { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 16px; list-style: none; padding: 0; }
.pf-agenda button { width: 100%; height: 100%; text-align: left; cursor: pointer; border: 1px solid var(--line); border-radius: 10px; background: transparent;
  padding: 10px 12px; display: flex; flex-direction: column; gap: 2px; transition: background 0.15s, border-color 0.15s; }
.pf-agenda button:hover { background: rgba(255, 255, 255, 0.035); border-color: var(--line-2); }
.pf-agenda button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.pf-agenda b { font: 650 14px/1.3 var(--sans); color: var(--ink); }
.pf-agenda small { font: 600 10.5px/1.3 var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); }
.pf-agenda span { font-size: 12.5px; color: var(--ink-2); line-height: 1.4; }
.pf-tools { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--grid); }
.pf-tools .grow { flex: 1; }
.pf-tools .switch { padding: 0; }
.pf-tools-h { font-size: 12.5px; color: var(--muted); }
.pf-ch { margin: 18px 0 -4px; display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; scroll-margin-top: 90px; }
.pf-ch-n { font: 700 13px/1 var(--mono); color: var(--accent); letter-spacing: 0.1em; }
.pf-ch h2 { margin: 0; font: 650 22px/1.3 var(--sans); letter-spacing: -0.01em; }
.pf-ch p { margin: 0; color: var(--ink-2); font-size: 14px; flex-basis: 100%; max-width: 86ch; }
.pc { display: flex; flex-direction: column; scroll-margin-top: 90px; }
.pc-h { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 12px 20px; margin-bottom: 14px; }
.pc-h > div:first-child { flex: 1 1 520px; min-width: 0; }
.pc-k { font: 650 23px/1.28 var(--sans); letter-spacing: -0.01em; margin: 6px 0 0; color: var(--ink); max-width: 46ch; }
.pc-k em { font-style: normal; color: var(--accent-ink); }
.pc .card-d { font-size: 14px; max-width: 92ch; margin-top: 8px; }
.pc-sensor { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }
.pc-sensor-l { font: 600 10.5px/1.2 var(--mono); letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.pc-sensor-h { font-size: 12px; color: var(--muted); }
.pc > .card-foot { margin-top: auto; padding-top: 14px; }
.pc-src { display: inline-block; margin-left: 4px; font-family: var(--mono); font-size: 11.5px; color: var(--ink-2); }
.pc-quoted { color: var(--warning); }
.pc-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px; }
.pc-split { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(0, 1fr); gap: 28px; align-items: start; }
.pc-ch { margin: 0 0 8px; font-size: 14px; font-weight: 600; color: var(--ink); }
.pc-ch small { font-weight: 400; color: var(--muted); font-size: 12.5px; }
.pc-legrow { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 2px 16px; }
.pc-better { font-size: 12.5px; color: var(--muted); margin-bottom: 8px; white-space: nowrap; }
.pc-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.pc-tiles .tile-v { font-size: 30px; }
.pc-tiles .tile-s { line-height: 1.4; }
.pc-tiles .tile-v small { font-size: 15px; }
.pc-tiles.stack { grid-template-columns: minmax(0, 1fr); }
.pc-note { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 8px 12px; margin: 14px 0 0; padding: 12px 14px;
  border: 1px solid var(--line); border-radius: 10px; background: var(--surface); font-size: 13.5px; color: var(--ink-2); }
.pc-note p { margin: 0; flex: 1; min-width: 220px; }
.pc-note b { color: var(--ink); font-weight: 600; }
.pc-checks { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
.pc-check { display: grid; grid-template-columns: 24px minmax(0, 1fr); gap: 10px; align-items: start; padding: 10px 12px; border-radius: 10px;
  border: 1px solid var(--line); border-left-width: 3px; background: var(--surface); font-size: 13.5px; color: var(--ink-2); line-height: 1.45; }
.pc-check .ic { width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; font: 700 12px/1 var(--mono); }
.pc-check.good { border-left-color: var(--good); } .pc-check.good .ic { color: var(--good-ink); background: rgba(12, 163, 12, 0.14); }
.pc-check.bad { border-left-color: var(--critical); } .pc-check.bad .ic { color: var(--critical-ink); background: rgba(208, 59, 59, 0.14); }
.pc-check.warn { border-left-color: var(--warning); } .pc-check.warn .ic { color: var(--warning); background: rgba(250, 178, 25, 0.12); }
.pc-tbl { margin-top: 14px; overflow-x: auto; border-top: 1px solid var(--grid); padding-top: 4px; }
.pc-tbl .table { min-width: 420px; }
.pc-tbl caption { caption-side: top; text-align: left; color: var(--muted); font-size: 12px; padding: 6px 0 4px; }
.pc-tbl tbody th { font: 500 13px var(--sans); letter-spacing: 0; text-transform: none; color: var(--ink); border-top: 1px solid var(--grid); }
.pc-tbl td.us { color: var(--accent-ink); font-weight: 600; }
.pc-limits { margin: 0; padding: 0; list-style: none; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.pc-limits li { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; background: var(--surface); font-size: 13.5px; color: var(--ink-2); line-height: 1.5; }
.pc-limits li b { display: block; color: var(--ink); font-weight: 600; font-size: 14px; margin-bottom: 3px; }
.hm-wrap { overflow-x: auto; }
.hm { display: block; width: 100%; height: auto; }
.hm .hm-c { stroke: var(--panel); stroke-width: 2; }
.hm .hm-c.hov { stroke: var(--ink); stroke-width: 1.5; }
.hm .hm-v { font: 600 11.5px var(--mono); pointer-events: none; }
.hm .hm-r, .hm .hm-col { fill: var(--ink-2); font: 500 12px var(--sans); }
.hm .hm-g { fill: var(--muted); font: 600 10px var(--mono); letter-spacing: 0.12em; }
.hm .hm-gl { stroke: var(--line-2); stroke-width: 1; }
.hm .hm-t { fill: var(--ink-2); font: 600 12px var(--sans); }
.hm .o-right { fill: none; stroke: var(--ink); stroke-width: 2; pointer-events: none; }
.hm .o-safe { fill: none; stroke: var(--ink-2); stroke-width: 1.4; stroke-dasharray: 3 3; pointer-events: none; }
.hm .o-bad { fill: none; stroke: var(--critical-ink); stroke-width: 2.2; pointer-events: none; }
.hm-leg { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--ink-2); margin: 10px 0 0; }
.hm-leg span { display: inline-flex; align-items: center; gap: 7px; }
.hm-leg i { display: inline-block; width: 14px; height: 14px; border-radius: 3px; }
.hm-leg .k-right { border: 2px solid var(--ink); }
.hm-leg .k-safe { border: 1.5px dashed var(--ink-2); }
.hm-leg .k-bad { border: 2px solid var(--critical-ink); }
.hm-leg .k-ramp { width: 64px; height: 10px; border-radius: 2px; }
@media (max-width: 1280px) {
  .pf-tiles { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .pf-agenda { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 1100px) {
  .pc-split { grid-template-columns: minmax(0, 1fr); }
  .pc-sensor { align-items: flex-start; }
}
@media (max-width: 860px) {
  .pf-hero { padding: 20px 18px 18px; }
  .pc-pair { grid-template-columns: minmax(0, 1fr); gap: 22px; }
  .pc-k { font-size: 19px; }
  .pf-tiles, .pf-agenda { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .pc-limits { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 560px) {
  .pf-tiles, .pf-agenda { grid-template-columns: minmax(0, 1fr); }
  .pc-tbl .table { min-width: 0; font-size: 12.5px; }
  .pc-tbl .table th, .pc-tbl .table td { padding: 6px 5px; }
  .pc-tbl thead th { white-space: normal; letter-spacing: 0.04em; vertical-align: bottom; }
}`;
  function injectCss() {
    if (document.getElementById("css-proof")) return;
    document.head.append(el("style", { id: "css-proof" }, CSS));
  }

  // ----------------------------------------------------------------- helpers
  /** One decimal, trailing ".0" dropped: 53.333 -> "53.3". */
  const d1 = (v) => String(Math.round(v * 10) / 10);
  /** Fraction -> "per 100" number: 0.235 -> "23.5". */
  const per = (p) => (p == null ? "—" : d1(p * 100));
  const pct0 = (p) => (p == null || isNaN(p) ? "—" : Math.round(p * 100 + 1e-9) + "%");
  /** Fraction -> "99.5%" (one decimal, ".0" dropped). */
  const pct1 = (p) => (p == null ? "—" : d1(p * 100) + "%");
  /** Ratio as text: "8.5", whole numbers from 10 up. */
  const times = (r) => (r >= 10 ? String(Math.round(r)) : r.toFixed(1));
  const sum = (xs) => xs.reduce((s, v) => s + v, 0);
  const mean = (xs) => (xs.length ? sum(xs) / xs.length : null);
  const niceMax = (v) => [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100].find((m) => m >= v) || Math.ceil(v / 50) * 50;
  const fill = (node, ...kids) => { OTS.clear(node).append(...kids.flat(Infinity).filter((k) => k != null && k !== false && k !== "")); return node; };
  const status = (kind, icon, text) => el("span", { class: "status " + kind },
    el("span", { class: "ic", "aria-hidden": "true" }, icon), el("span", {}, text));
  const sq = (label, color) => ({ label, color, kind: "square" });
  /** A sentence with a status icon (for lists where an uppercase badge would be hard to read). */
  const check = (kind, icon, text) => el("li", { class: "pc-check " + kind }, el("span", { class: "ic", "aria-hidden": "true" }, icon), el("span", {}, text));
  /** Value labels on column caps only where each cap has room (off on phones). */
  const roomy = (box, nCats, nSer) => (box.clientWidth - 70) / nCats >= 23 * nSer + 10;

  /** Colours: OTS is the accent everywhere; rivals keep one colour each across the page. */
  const COL = {
    ots: () => OTS.tok("accent"), base: () => OTS.tok("baseline"),
    idonly: () => "#4f8fe0", orange: () => "#c27f22", purple: () => "#b277ab",
  };
  const METHOD = {
    bank: { label: "OTS (bank + “unknown”)", short: "OTS", color: COL.ots },
    pairwise_sprt: { label: "Standard sequential test", short: "Sequential test", color: COL.base },
    fixed_length: { label: "Fixed-length test", short: "Fixed-length", color: COL.orange },
    learned: { label: "Learned classifier", short: "Classifier", color: COL.purple },
    proposed: { label: "OTS predictor", short: "OTS", color: COL.ots },
    id_only: { label: "Bank alone (name it, then predict)", short: "Bank alone", color: COL.idonly },
    ppm: { label: "PPM-C (text-compression predictor)", short: "PPM-C", color: COL.orange },
    markov1: { label: "Frequency statistics (Markov-1)", short: "Freq. stats", color: COL.base },
    lstm: { label: "LSTM trained on the known families", short: "LSTM", color: COL.purple },
  };

  const P = () => OTS.data.paper;
  const SN = (sid) => P().sensors[sid] || { name: sid, p_d: 1, p_fa: 0 };
  const ghosts = (p_fa) => String(+(p_fa * ((OTS.data.M || 16) - 1)).toFixed(2));
  const sensorHelp = (sid) => { const s = SN(sid); return `Catches ${Math.round(s.p_d * 100)}% of hops · ~${ghosts(s.p_fa)} ghosts per hop`; };
  const sensorOrder = (ids) => Object.keys(P().sensors).filter((s) => ids.includes(s));
  const gname = (g) => OTS.emitName(g);
  /** Short names for crowded chart axes (tooltips and tables keep the full name). */
  const SHORT = { filtered_lfsr: "Filt. LFSR", lcg_full: "Full LCG", weak_seed: "Weak-seed", logistic: "Logistic", tent: "Tent", henon: "Hénon",
    markov: "Markov", lagged_fib: "Lagged gen.", quadratic_cg: "Quadratic", sine_map: "Sine map", csprng: "CSPRNG" };
  const gshort = (g) => SHORT[g] || gname(g);
  const fname = (f) => (f === "unknown" ? "Unknown (H₀)" : OTS.hypName(f));
  const ALGEBRAIC = new Set(["lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed"]);
  /** Deterministic generators the bank has an expert for (the full-state LCG is reported on its own). */
  const DET = ["lfsr", "filtered_lfsr", "lcg", "weak_seed", "logistic", "tent", "henon"];
  const isNew = (g) => !!(OTS.emitter(g) && OTS.emitter(g).new_algo);
  const groupOf = (g) => (g === "csprng" ? "control" : isNew(g) ? "new" : "known");
  const GROUP_SUB = { known: "known", new: "new · no expert", control: "control" };
  const src = (file) => el("span", { class: "pc-src" }, file);

  /** Small data table. cols: [{h, us?}], rows: arrays of cells. */
  function table(caption, cols, rows) {
    return el("div", { class: "pc-tbl" }, el("table", { class: "table" },
      el("caption", {}, caption),
      el("thead", {}, el("tr", {}, cols.map((c) => el("th", { scope: "col" }, c.h)))),
      el("tbody", {}, rows.map((r) => el("tr", {}, r.map((v, i) => el(i ? "td" : "th", { scope: i ? null : "row", class: cols[i].us ? "us" : null }, v)))))));
  }

  /** Card skeleton: kicker, headline (filled by render), description, optional sensor switch, body, table, footnote. */
  function card(n, chapter, title, opts = {}) {
    const k = el("h3", { class: "pc-k" }), d = el("p", { class: "card-d" });
    const right = el("div", { class: "pc-sensor" });
    const body = el("div"), tbl = el("div", { hidden: true }), foot = el("p", { class: "card-foot" });
    const node = el("article", { class: "card pc", id: `claim-${n}`, "data-slide": `Claim ${n} · ${title}`, "aria-label": `Claim ${n}: ${title}` },
      el("div", { class: "pc-h" }, el("div", {}, el("p", { class: "card-t" }, `${chapter} · Claim ${n} · ${title}`), k, d), right),
      body, tbl, foot);
    const c = { node, k, d, right, body, tbl, foot, table: null, ok: false, render: () => {}, ...opts };
    return c;
  }

  /** The per-card sensor switch. Calls onChange after the selection moves. */
  function sensorSeg(c, key, ids, onChange) {
    const order = sensorOrder(ids);
    if (!order.includes(sel[key])) sel[key] = order.includes("typical") ? "typical" : order[0];
    const help = el("span", { class: "pc-sensor-h" });
    const btns = order.map((id) => el("button", {
      class: "btn", type: "button", "data-s": id, "aria-pressed": "false", title: sensorHelp(id),
      onclick: () => { sel[key] = id; sync(); onChange(); },
    }, SN(id).name));
    function sync() {
      for (const b of btns) { const on = b.dataset.s === sel[key]; b.classList.toggle("sel", on); b.setAttribute("aria-pressed", String(on)); }
      help.textContent = sensorHelp(sel[key]);
    }
    sync();
    fill(c.right, el("span", { class: "pc-sensor-l" }, "Sensor"), el("div", { class: "seg", role: "group", "aria-label": "Sensor condition" }, btns), help);
  }

  // ----------------------------------------------------------------- heatmap
  /** Static heatmap SVG (scales with its box). spec = {rows, cols, value(i,j), color(v), dark(v),
   *  text(v), outline(i,j) -> 'right'|'safe'|'bad'|null, tooltip(i,j) -> {title, rows},
   *  groups: [{from, to, label}], xTitle, yTitle, cellW, cellH, labelW, headH, rotate, ariaLabel} */
  function heatmap(spec) {
    const cw = spec.cellW || 52, ch = spec.cellH || 28, gw = spec.groups ? 22 : 0;
    const lw = (spec.labelW || 140) + gw, hh = spec.headH || 40, yt = spec.yTitle ? 20 : 0;
    const W = lw + yt + spec.cols.length * cw + (spec.rotate ? Math.round(hh * 0.85) : 8), H = hh + spec.rows.length * ch + (spec.xTitle ? 28 : 8);
    const x0 = lw + yt, y0 = hh;
    const root = svg("svg", { class: "hm", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.ariaLabel || "heatmap",
      style: `max-width:${Math.round(W * 1.35)}px` });
    spec.cols.forEach((c, j) => {
      const cx = x0 + j * cw + cw / 2;
      root.append(spec.rotate
        ? svg("text", { class: "hm-col", x: cx, y: y0 - 8, "text-anchor": "start", transform: `rotate(-38 ${cx} ${y0 - 8})` }, c)
        : svg("text", { class: "hm-col", x: cx, y: y0 - 10, "text-anchor": "middle" }, c));
    });
    if (spec.yTitle) root.append(svg("text", { class: "hm-t", x: 0, y: 0, "text-anchor": "middle", transform: `translate(${gw + 12} ${y0 + spec.rows.length * ch / 2}) rotate(-90)` }, spec.yTitle));
    if (spec.xTitle) root.append(svg("text", { class: "hm-t", x: x0 + spec.cols.length * cw / 2, y: H - 6, "text-anchor": "middle" }, spec.xTitle));
    for (const g of spec.groups || []) {
      const ya = y0 + g.from * ch + 3, yb = y0 + (g.to + 1) * ch - 3;
      root.append(svg("line", { class: "hm-gl", x1: gw - 6, x2: gw - 6, y1: ya, y2: yb }));
      root.append(svg("text", { class: "hm-g", x: 0, y: 0, "text-anchor": "middle", transform: `translate(${gw - 12} ${(ya + yb) / 2}) rotate(-90)` }, g.label));
      if (g.from > 0) root.append(svg("line", { class: "hm-gl", x1: gw, x2: x0 + spec.cols.length * cw, y1: y0 + g.from * ch, y2: y0 + g.from * ch, opacity: 0.6 }));
    }
    spec.rows.forEach((r, i) => {
      const cy = y0 + i * ch + ch / 2;
      root.append(svg("text", { class: "hm-r", x: x0 - 10, y: cy + 4, "text-anchor": "end" }, r));
      spec.cols.forEach((_, j) => {
        const v = spec.value(i, j), x = x0 + j * cw, y = y0 + i * ch;
        const cell = svg("rect", { class: "hm-c", x, y, width: cw, height: ch, rx: 4, fill: v == null ? "transparent" : spec.color(v) });
        root.append(cell);
        const t = v == null ? "" : spec.text(v);
        if (t) root.append(svg("text", { class: "hm-v", x: x + cw / 2, y: y + ch / 2 + 4, "text-anchor": "middle", fill: spec.dark(v) ? "#031016" : OTS.tok("ink") }, t));
        const o = spec.outline ? spec.outline(i, j) : null;
        if (o) root.append(svg("rect", { class: "o-" + o, x: x + 2, y: y + 2, width: cw - 4, height: ch - 4, rx: 3 }));
        if (spec.tooltip) {
          cell.addEventListener("pointermove", (ev) => { cell.classList.add("hov"); OTS.tip.show(ev, spec.tooltip(i, j)); });
          cell.addEventListener("pointerleave", () => { cell.classList.remove("hov"); OTS.tip.hide(); });
        }
      });
    });
    return el("div", { class: "hm-wrap" }, root);
  }
  /** Sequential single-hue fill: t in [0, 1] -> rgba of the hue. */
  const ramp = (rgb, t) => `rgba(${rgb}, ${(0.1 + 0.85 * Math.max(0, Math.min(1, t))).toFixed(3)})`;
  const CYAN = "34, 211, 238", AMBER = "194, 127, 34";

  // ===================================================================== view
  OTS.views.register({
    id: "proof", nav: "Proof", title: "Proof at scale", hint: "The paper's results, claim by claim",
    subtitle: "Thirteen claims from the paper's latest experiments, each with its evidence and its source",
    icon: "M4 20V10m6 10V4m6 16v-7m4 7H2",
    mount(root) {
      injectCss();
      const DATA = OTS.data, PP = DATA.paper;
      const handles = [], charts = [], offs = [], cards = [];
      const mk = (box, make, spec) => { const h = make(box, spec()); handles.push(h); const c = { h, spec }; charts.push(c); return c; };

      if (!PP) {
        root.append(el("div", { class: "card empty" }, "The paper's results are not in this data.js. From the repository root run ",
          el("code", {}, "python -m dashboard.build --reuse"), " and reload."));
        return () => {};
      }
      const T = "typical";
      const conf = PP.confusion, idb = PP.id_baselines, pr = PP.predict, pb = PP.pred_baselines;
      const cp = PP.conformal, dn = PP.denial, rt = PP.runtime, sw = PP.switch, mm = PP.mismatch, DOC = PP.documented;
      const chanceTxt = () => String(+((pr ? pr.chance : 1 / 16) * 100).toFixed(2)) + "%";

      // ------------------------------------------------------------ hero
      const confRow = (sid, g) => { const s = conf.by_sensor[sid], i = conf.generators.indexOf(g); return { acc: s.acc[i], hard: s.hard[i], hops: s.hops[i] }; };
      const tiles = [];
      if (conf && conf.by_sensor[T]) {
        const rows = DET.map((g) => confRow(T, g));
        const hops = rows.map((r) => r.hops).filter((h) => h != null);
        tiles.push(["Named correctly", pct0(mean(rows.map((r) => r.acc))),
          `Deterministic radios with an expert, Typical sensor: each ≥ ${Math.floor(Math.min(...rows.map((r) => r.acc)) * 100)}%`,
          `in ${Math.round(Math.min(...hops))}–${Math.round(Math.max(...hops))} hops`]);
      }
      if (idb && idb.by_sensor[T]) {
        const m = idb.by_sensor[T].methods;
        tiles.push(["Dangerous false claims", per(m.bank.hard_oob) + " in 100",
          `On radios outside the bank. Standard sequential test: ${per(m.pairwise_sprt.hard_oob)} in 100`, "Typical sensor"]);
      }
      if (pr && pr.by_sensor[T] && pr.by_sensor[T].lagged_fib) {
        const r = pr.by_sensor[T].lagged_fib, l = pb && pb.by_sensor[T] && pb.by_sensor[T].gens.lagged_fib;
        tiles.push(["New algorithm predicted", pct0(r.steady),
          `Next hop of a ${gname("lagged_fib").toLowerCase()} it has no expert for. Bank alone: ${pct0(r.id_only)}${l ? ` · trained LSTM: ${pct0(l.lstm)}` : ""}`,
          "Typical sensor, after it learns"]);
      }
      if (cp && cp.by_sensor[T]) {
        const op = Object.values(cp.by_sensor[T].op);
        tiles.push(["Coverage kept", "≥ " + Math.floor(Math.min(...op.map((o) => o.cov)) * 100) + "%",
          `Promised 90% on every algorithm. A fixed top-3 shortlist: as low as ${pct0(Math.min(...op.map((o) => o.top3_cov)))}`, "Typical sensor"]);
      }
      if (rt && rt.by_M["16"]) {
        const r = rt.by_M["16"], tot = r.bank.med + r.predictor.med + r.conformal_set.med;
        tiles.push(["Time per hop", d1(tot) + " ms", `Name, predict and size the jam on one CPU core. With switch tracking: ${Math.round(r.tracker.med)} ms`, "16 channels, unoptimised Python"]);
      }
      const ep = DOC.episodes;
      const CHAPTERS = [
        { id: "identify", n: "01", name: "Name the radio", blurb: "Which algorithm, and is it safe to say so?", claims: "Claims 1–2" },
        { id: "why", n: "02", name: "Why it works", blurb: "The sensor model, its robustness, the theoretical limit", claims: "Claims 3–5" },
        { id: "predict", n: "03", name: "Call the next channel", blurb: "Even for algorithms it was never built for", claims: "Claims 6–8" },
        { id: "act", n: "04", name: "Act on it", blurb: "Jam, notice a change, keep up in real time", claims: "Claims 9–11" },
        { id: "limits", n: "05", name: "Honest limits", blurb: "Where it can be beaten, and what is not done yet", claims: "Claims 12–13" },
      ];
      const chapterFirst = {};
      const tblInp = el("input", { type: "checkbox", "aria-label": "Show data tables" });
      tblInp.checked = !!OTS.prefs.tables;
      tblInp.addEventListener("change", () => OTS.setPref("tables", tblInp.checked));
      const hero = el("section", { class: "card pf-hero", "data-slide": "Overview" },
        el("p", { class: "card-t" }, "The paper's evaluation, claim by claim"),
        el("h2", { class: "lead" }, "Not one lucky demo: ", el("em", {}, fmt.int(ep.total) + " simulated engagements"), ", every one counted."),
        el("p", { class: "intro" },
          `Each engagement hides one of ${conf ? conf.generators.length : 13} hop algorithms behind a fresh random secret and listens through a sensor that misses hops and hears ghost signals. `,
          "OTS and every rival method see exactly the same detections. The numbers below are read straight from the paper's saved result files; each card names its file."),
        el("div", { class: "pf-tiles" }, tiles.map(([l, v, s, x]) => el("div", { class: "tile" },
          el("div", { class: "tile-l" }, l), el("div", { class: "tile-v num" }, v), el("div", { class: "tile-s" }, s),
          el("div", { class: "tile-x" }, x)))),
        el("ol", { class: "pf-agenda", "aria-label": "Chapters" }, CHAPTERS.map((c) => el("li", {}, el("button", {
          type: "button", onclick: () => OTS.present.goTo(chapterFirst[c.id]),
          title: `Go to “${c.name}”`,
        }, el("small", {}, `${c.n} · ${c.claims}`), el("b", {}, c.name), el("span", {}, c.blurb))))),
        el("div", { class: "pf-tools no-present" },
          OTS.present.button("Present"),
          el("span", { class: "pf-tools-h" }, "One claim per slide · ← → to move · Esc to leave"),
          el("span", { class: "grow" }),
          el("label", { class: "switch" }, tblInp, "Show data tables")));
      const chapter = (c) => el("div", { class: "pf-ch", id: "ch-" + c.id },
        el("span", { class: "pf-ch-n" }, c.n), el("h2", {}, c.name), el("p", {}, /[.?!]$/.test(c.blurb) ? c.blurb : c.blurb + "."));

      // ========================================================= claim 1
      const c1 = card(1, "Name the radio", "Names the radio");
      if (conf) {
        const hmBox = el("div"), side = el("div", { class: "pc-tiles stack" }), note = el("div", { class: "pc-note", hidden: true });
        c1.body.append(el("div", { class: "pc-split" },
          el("div", {}, hmBox, el("div", { class: "hm-leg" },
            el("span", {}, el("i", { class: "k-ramp", style: `background:linear-gradient(90deg, ${ramp(CYAN, 0)}, ${ramp(CYAN, 1)})` }), "share of runs (0 → 100)"),
            el("span", {}, el("i", { class: "k-right" }), "right answer"),
            el("span", {}, el("i", { class: "k-safe" }), "safe answer (no expert exists)"),
            el("span", {}, el("i", { class: "k-bad" }), "dangerous false claim"))),
          el("div", {}, side, note)));
        sensorSeg(c1, "conf", Object.keys(conf.by_sensor), () => c1.render());
        // rows: known generators, then the held-out ones, then the control
        const order = [...conf.generators.filter((g) => groupOf(g) === "known"), ...conf.generators.filter((g) => groupOf(g) === "new"), "csprng"].filter((g) => conf.generators.includes(g));
        const nKnown = order.filter((g) => groupOf(g) === "known").length, nNew = order.filter((g) => groupOf(g) === "new").length;
        c1.ok = true;
        c1.d.textContent = `Rows: the hidden algorithm, ${conf.n_seeds} runs each. Columns: what OTS answered. A clean diagonal is right answers; a red outline is a dangerous false claim, naming a crackable algorithm that is not there.`;
        c1.render = () => {
          const s = conf.by_sensor[sel.conf], gi = (g) => conf.generators.indexOf(g);
          const target = (g) => s.target[gi(g)];
          const bad = (g, f) => ALGEBRAIC.has(f) && f !== target(g);
          fill(hmBox, heatmap({
            rows: order.map(gname), cols: conf.families.map(fname), rotate: true, headH: 96, labelW: 118, cellW: 50, cellH: 27,
            ariaLabel: `Confusion matrix on the ${SN(sel.conf).name} sensor: hidden algorithm against OTS's answer, ${conf.n_seeds} runs per row`,
            groups: [{ from: 0, to: nKnown - 1, label: "KNOWN" }, { from: nKnown, to: nKnown + nNew - 1, label: "NEW" }, { from: nKnown + nNew, to: order.length - 1, label: "CTRL" }],
            value: (i, j) => s.counts[gi(order[i])][j],
            color: (v) => (v === 0 ? "rgba(255,255,255,0.025)" : ramp(CYAN, Math.pow(v / conf.n_seeds, 0.7))),
            dark: (v) => v / conf.n_seeds > 0.55, text: (v) => (v === 0 ? "" : String(v)),
            outline: (i, j) => {
              const g = order[i], f = conf.families[j], v = s.counts[gi(g)][j];
              if (v > 0 && bad(g, f)) return "bad";
              if (groupOf(g) === "new") return f === "markov" || f === "unknown" ? "safe" : null;
              return f === target(g) ? "right" : null;
            },
            tooltip: (i, j) => {
              const g = order[i], f = conf.families[j], v = s.counts[gi(g)][j];
              return { title: `${gname(g)} · ${SN(sel.conf).name} sensor`, rows: [
                { value: `${v} of ${conf.n_seeds}`, label: `answered “${fname(f)}”` },
                { value: bad(g, f) ? "dangerous" : f === target(g) ? "right" : groupOf(g) === "new" && (f === "markov" || f === "unknown") ? "safe" : "soft miss", label: "kind of answer" }] };
            },
          }));
          const det = DET.filter((g) => gi(g) >= 0).map((g) => ({ g, ...confRow(sel.conf, g) }));
          const hops = det.map((r) => r.hops).filter((h) => h != null);
          const nBad = sum(order.map((g) => sum(conf.families.map((f, j) => (bad(g, f) ? s.counts[gi(g)][j] : 0)))));
          const total = order.length * conf.n_seeds;
          const newBad = sum(order.filter((g) => groupOf(g) === "new").map((g) => sum(conf.families.map((f, j) => (bad(g, f) ? s.counts[gi(g)][j] : 0)))));
          const cs = gi("csprng") >= 0 ? confRow(sel.conf, "csprng") : null;
          c1.k.textContent = `${SN(sel.conf).name} sensor: the ${det.length} deterministic radios with an expert are named right ${pct0(mean(det.map((r) => r.acc)))} of the time, each in ${Math.round(Math.min(...hops))}–${Math.round(Math.max(...hops))} hops on average`;
          const tile = (l, v, sub) => el("div", { class: "tile" }, el("div", { class: "tile-l" }, l), el("div", { class: "tile-v num" }, v), el("div", { class: "tile-s" }, sub));
          fill(side,
            tile("Dangerous false claims", [per(nBad / total), el("small", {}, " in 100")], `${nBad} of ${fmt.int(total)} runs named a crackable algorithm that was not there`),
            cs ? tile("Unbreakable radio", pct0(cs.acc), "of keyed-CSPRNG runs answered “unknown”, the only right answer") : null,
            nNew ? tile("New algorithms", `${nNew * conf.n_seeds - newBad} / ${nNew * conf.n_seeds}`, "runs on algorithms with no expert ended in a safe answer (Markov fit or unknown)") : null);
          // the honest weak spot, when the data says it is one
          const lf = gi("lcg_full") >= 0 ? confRow(sel.conf, "lcg_full") : null;
          note.hidden = !(lf && lf.acc < 0.9);
          if (!note.hidden) {
            const asMarkov = s.counts[gi("lcg_full")][conf.families.indexOf("markov")];
            fill(note, status("warn", "!", "Weak spot"), el("p", {}, el("b", {}, `Full-state LCG: ${pct0(lf.acc)} right. `),
              `With ${DATA.M || 16} channels its whole cycle is at most ${DATA.M || 16} hops long, so it also looks like a regular statistical pattern: ${asMarkov} of ${conf.n_seeds} runs were called “Markov hopper”. That is a soft miss, not a false claim of a crack`,
              lf.hard > 0 ? ` (dangerous claims on it: ${per(lf.hard)} in 100).` : "."));
          }
          c1.foot.replaceChildren(`${conf.n_seeds} runs per row × ${conf.generators.length} algorithms × ${Object.keys(conf.by_sensor).length} sensors = ${fmt.int(conf.n_seeds * conf.generators.length * Object.keys(conf.by_sensor).length)} engagements, ${conf.window}-hop window, lock at 95% (δ = ${conf.delta}). Source:`, src(PP.files.confusion));
          c1.table = () => table(`Per algorithm · ${SN(sel.conf).name} sensor`,
            [{ h: "Hidden algorithm" }, { h: "Right", us: true }, { h: "Dangerous / 100", us: true }, { h: "Hops to lock", us: true }],
            order.map((g) => { const r = confRow(sel.conf, g); return [gname(g), groupOf(g) === "new" ? "no expert" : pct0(r.acc), per(r.hard), r.hops == null ? "—" : d1(r.hops)]; }));
        };
      }
      cards.push(c1);

      // ========================================================= claim 2
      const c2 = card(2, "Name the radio", "Safer than conventional identifiers");
      if (idb) {
        const box = el("div");
        const mids = ["bank", "pairwise_sprt", "fixed_length", "learned"];
        c2.body.append(el("div", { class: "pc-legrow" }, OTS.chart.legend(mids.map((m) => sq(METHOD[m].label, METHOD[m].color())))), box);
        sensorSeg(c2, "idb", Object.keys(idb.by_sensor), () => { c2.render(); ch2.h.update(ch2.spec()); });
        c2.ok = true;
        const CATS = [
          { id: "acc", label: "Right on known radios", sub: "▲ higher is better" },
          { id: "unknown", label: "Unbreakable radio → “unknown”", sub: "▲ higher is better" },
          { id: "hard_oob", label: "Dangerous claims, unknown radios", sub: "▼ lower is better" },
        ];
        var ch2 = { spec: () => {
          const m = idb.by_sensor[sel.idb].methods;
          return {
            height: 320, ariaLabel: `Identification on the ${SN(sel.idb).name} sensor: OTS against three conventional identifiers`,
            categories: CATS, series: mids.map((id) => ({ id, label: METHOD[id].short, color: METHOD[id].color() })),
            value: (c, sid) => 100 * m[sid][c], y: { label: "Share of runs (%)", max: 100, ticks: [0, 25, 50, 75, 100], format: String },
            barWidth: 22, valueLabels: roomy(box, CATS.length, mids.length), valueFormat: (v) => ((v > 0 && v < 10) || (v > 99 && v < 100) ? d1(v) : String(Math.round(v))),
            tooltip: (c) => ({ title: `${c.label} · ${SN(sel.idb).name}`, rows: mids.map((id) => ({ value: pct1(m[id][c.id]), label: METHOD[id].label, color: METHOD[id].color(), kind: "square" })) }),
          };
        } };
        c2.render = () => {
          const m = idb.by_sensor[sel.idb].methods, b = m.bank, s = m.pairwise_sprt;
          c2.k.textContent = (b.hard_oob > 0
            ? `Just as accurate on known radios, and ${times(s.hard_oob / b.hard_oob)}× fewer dangerous false claims on unknown ones: ${per(b.hard_oob)} in 100 against ${per(s.hard_oob)}`
            : `Just as accurate on known radios, and no dangerous false claims on unknown ones, against ${per(s.hard_oob)} in 100`) + ` (${SN(sel.idb).name} sensor)`;
          c2.d.textContent = `The standard sequential test and the fixed-length test use OTS's own experts and scoring, minus the “unknown” option, so on known radios they are nearly as accurate (${pct1(s.acc)} vs ${pct1(b.acc)}). `
            + `Faced with a radio outside the bank they must name something: the sequential test called the unbreakable radio crackable in ${pct0(s.per_gen.csprng ? s.per_gen.csprng[1] : null)} of runs. `
            + `The price of caution is small: ${d1(b.hops)} hops on average to a right answer, against ${d1(s.hops)}.`;
          c2.foot.replaceChildren(`${idb.n_seeds} runs × ${idb.generators.length} algorithms per sensor (${fmt.int(idb.n_episodes)} engagements, ${idb.window}-hop window). The fixed-length test decides at hop ${idb.fixed_length_n}; the classifier was trained on ${fmt.int(idb.classifier_train)} simulated runs and has an “unknown” class. Source:`, src(PP.files.id_baselines));
          c2.table = () => table(`Identifiers on identical runs · ${SN(sel.idb).name} sensor`,
            [{ h: "Method" }, { h: "Right, known" }, { h: "CSPRNG → unknown" }, { h: "Dangerous / 100, all" }, { h: "Dangerous / 100, unknown radios" }, { h: "Hops" }],
            mids.map((id) => { const x = m[id]; return [METHOD[id].label, pct1(x.acc), pct0(x.unknown), per(x.hard), per(x.hard_oob), d1(x.hops)]; }));
        };
        c2.mountCharts = () => { Object.assign(ch2, mk(box, OTS.chart.columns, ch2.spec)); };
      }
      cards.push(c2);

      // ========================================================= claim 3
      const c3 = card(3, "Why it works", "Modelling sensor mistakes is the key");
      const ab = DATA.proof && DATA.proof.sensor_ablation, bm = DATA.proof && DATA.proof.bm_vs_filtered;
      if (ab && bm) {
        const aBox = el("div"), bBox = el("div");
        c3.body.append(el("div", { class: "pc-pair" },
          el("div", {}, el("p", { class: "pc-ch" }, "Same experts, two ways of scoring ", el("small", {}, "· right answer")),
            OTS.chart.legend([{ label: "OTS (expects missed hops and ghosts)", color: COL.ots() }, { label: "Sensor-naive scoring", color: COL.base() }]), aBox),
          el("div", {}, el("p", { class: "pc-ch" }, "Cracking a shift register ", el("small", {}, "· secret recovered")),
            OTS.chart.legend([{ label: "OTS (keeps every state that fits)", color: COL.ots() }, { label: "Textbook Berlekamp–Massey", color: COL.base() }]), bBox)));
        c3.ok = true;
        const abRows = [...ab.rows].sort((a, b) => b.p_d - a.p_d), bmRows = [...bm.rows].sort((a, b) => b.p_d - a.p_d);
        const ticks = (rows) => rows.map((r) => Math.round(r.p_d * 100));
        const axis = (rows) => ({ label: "Share of hops the sensor catches (%)", min: 100, max: Math.min(...ticks(rows)), ticks: ticks(rows), format: (v) => v + "%" });
        const lo = Math.min(...abRows.flatMap((r) => [r.sensor_aware.accuracy, r.naive.accuracy])) * 100;
        const specA = () => ({
          height: 250, ariaLabel: "Right answers as the sensor catches fewer hops: sensor-aware versus sensor-naive scoring",
          x: axis(abRows), y: { label: "Right answer (%)", min: Math.min(50, Math.floor((lo - 5) / 10) * 10), max: 100, format: String },
          series: [
            { id: "aware", label: "OTS", color: COL.ots(), marker: "all", emphasis: true, points: abRows.map((r) => [Math.round(r.p_d * 100), 100 * r.sensor_aware.accuracy]) },
            { id: "naive", label: "Naive", color: COL.base(), marker: "all", points: abRows.map((r) => [Math.round(r.p_d * 100), 100 * r.naive.accuracy]) }],
          tooltip: (x) => { const r = abRows.find((q) => Math.round(q.p_d * 100) === x); return { title: `Sensor catches ${x}% of hops`, rows: [
            { value: pct0(r.sensor_aware.accuracy), label: "OTS right", color: COL.ots() }, { value: pct0(r.naive.accuracy), label: "Naive right", color: COL.base() },
            r.sensor_aware.mean_hops != null ? { value: d1(r.sensor_aware.mean_hops), label: "OTS hops to lock (mean)" } : null,
            r.naive.mean_hops != null ? { value: d1(r.naive.mean_hops), label: "Naive hops to lock (mean)" } : null].filter(Boolean) }; },
        });
        const specB = () => ({
          height: 250, ariaLabel: "Shift-register secrets recovered as the sensor catches fewer hops: textbook Berlekamp-Massey versus OTS",
          x: axis(bmRows), y: { label: "Secret recovered (%)", min: 0, max: 100, ticks: [0, 25, 50, 75, 100], format: String },
          series: [
            { id: "ots", label: "OTS", color: COL.ots(), marker: "all", emphasis: true, points: bmRows.map((r) => [Math.round(r.p_d * 100), 100 * r.state_filtered]) },
            { id: "bm", label: "Textbook", color: COL.base(), marker: "all", points: bmRows.map((r) => [Math.round(r.p_d * 100), 100 * r.bm]) }],
          tooltip: (x) => { const r = bmRows.find((q) => Math.round(q.p_d * 100) === x); return { title: `Sensor catches ${x}% of hops`, rows: [
            { value: pct0(r.state_filtered), label: "OTS recovered", color: COL.ots() }, { value: pct0(r.bm), label: "Textbook recovered", color: COL.base() }] }; },
        });
        c3.render = () => {
          const w = abRows[abRows.length - 1], firstGap = bmRows.find((r) => r.p_d < 1) || bmRows[0], worstBm = bmRows[bmRows.length - 1];
          const missed = (p) => (p === 0.5 ? "half the hops" : Math.round((1 - p) * 100) + "% of hops");
          c3.k.textContent = `With ${missed(w.p_d)} missed, OTS is right ${pct0(w.sensor_aware.accuracy)} of the time and the same experts scored naively ${pct0(w.naive.accuracy)}; the textbook shift-register attack recovers ${pct0(firstGap.bm)} of secrets once ${Math.round((1 - firstGap.p_d) * 100)}% of hops are missed`;
          c3.d.textContent = "Every expert scores the detections through one likelihood that expects missed hops and ghost signals. Take that away and a missed hop reads as evidence against the truth. "
            + `Berlekamp–Massey, the classic attack on shift-register radios, assumes every hop is heard; OTS keeps every register state that could explain the gaps and still recovers ${pct0(worstBm.state_filtered)} with ${missed(worstBm.p_d)} missed.`;
          const gens = ab.generators.map(gname).join(", ").replace(/, ([^,]*)$/, " and $1");
          c3.foot.replaceChildren(`The dashboard's own reruns of the paper's two ablations: ${gens}, ${ab.n_seeds} seeds each at ~${ghosts(ab.p_fa)} ghosts per hop (left); a ${bm.length}-bit LFSR, ${bm.n_seeds} seeds per point, no ghosts (right). Source:`, src("data.js · proof"), ` (dashboard build of ${DATA.generated}).`);
          c3.table = () => table("Sensor-aware vs naive scoring, and OTS vs Berlekamp–Massey",
            [{ h: "Sensor catches" }, { h: "OTS right", us: true }, { h: "Naive right" }, { h: "OTS hops", us: true }, { h: "Naive hops" }, { h: "OTS LFSR recovered", us: true }, { h: "Textbook recovered" }],
            abRows.map((r) => { const b = bmRows.find((q) => q.p_d === r.p_d); return [Math.round(r.p_d * 100) + "%", pct0(r.sensor_aware.accuracy), pct0(r.naive.accuracy),
              r.sensor_aware.mean_hops == null ? "—" : d1(r.sensor_aware.mean_hops), r.naive.mean_hops == null ? "—" : d1(r.naive.mean_hops), b ? pct0(b.state_filtered) : "—", b ? pct0(b.bm) : "—"]; }));
        };
        c3.mountCharts = () => { mk(aBox, OTS.chart.line, specA); mk(bBox, OTS.chart.line, specB); };
      }
      cards.push(c3);

      // ========================================================= claim 4
      const c4 = card(4, "Why it works", "Survives a wrong sensor estimate");
      if (mm) {
        const pct = (v) => Math.round(v * 100) + "%";
        const accs = mm.pd.flat().map((c) => c[0]), hards = [...mm.pd.flat(), ...mm.pfa.flat()].map((c) => c[1]);
        const hmA = heatmap({
          rows: mm.p_d_grid.map(pct), cols: mm.p_d_grid.map(pct), cellW: 56, cellH: 30, labelW: 44, headH: 30,
          yTitle: "True detection rate", xTitle: "Detection rate OTS assumes",
          ariaLabel: "Identification accuracy as the assumed detection rate departs from the true one",
          value: (i, j) => mm.pd[i][j][0], color: (v) => ramp(CYAN, (v - 0.8) / 0.2), dark: (v) => v > 0.93, text: (v) => String(Math.round(v * 100)),
          outline: (i, j) => (i === j ? "right" : null),
          tooltip: (i, j) => ({ title: `True ${pct(mm.p_d_grid[i])}, assumed ${pct(mm.p_d_grid[j])}`, rows: [
            { value: pct1(mm.pd[i][j][0]), label: "right answer" }, { value: per(mm.pd[i][j][1]) + " / 100", label: "dangerous false claims" }] }),
        });
        const hmB = heatmap({
          rows: mm.p_fa_grid.map(String), cols: mm.p_fa_grid.map(String), cellW: 66, cellH: 30, labelW: 44, headH: 30,
          yTitle: "True false-alarm rate", xTitle: "False-alarm rate OTS assumes",
          ariaLabel: "Dangerous false claims as the assumed false-alarm rate departs from the true one",
          value: (i, j) => mm.pfa[i][j][1], color: (v) => ramp(AMBER, v / 0.05), dark: () => false, text: (v) => d1(v * 100),
          outline: (i, j) => (i === j ? "right" : null),
          tooltip: (i, j) => ({ title: `True ${mm.p_fa_grid[i]}, assumed ${mm.p_fa_grid[j]}`, rows: [
            { value: per(mm.pfa[i][j][1]) + " / 100", label: "dangerous false claims" }, { value: pct1(mm.pfa[i][j][0]), label: "right answer" }] }),
        });
        c4.ok = true;
        c4.body.append(el("div", { class: "pc-pair" },
          el("div", {}, el("p", { class: "pc-ch" }, "Right answers (%) ", el("small", {}, `· false-alarm rate ${mm.p_fa_fixed}`)), hmA),
          el("div", {}, el("p", { class: "pc-ch" }, "Dangerous false claims per 100 ", el("small", {}, `· detection rate ${Math.round(mm.p_d_fixed * 100)}%`)), hmB)),
          el("div", { class: "hm-leg" }, el("span", {}, el("i", { class: "k-right" }), "OTS assumes the true value")));
        c4.render = () => {
          c4.k.textContent = `Guess the sensor wrong and OTS still gets ${Math.round(Math.min(...accs) * 100)}–${Math.round(Math.max(...accs) * 100)}% right, with never more than ${per(Math.max(...hards))} dangerous false claims in 100`;
          c4.d.textContent = "OTS scores with the detection and false-alarm rates it believes the sensor has. Here the belief is deliberately wrong. A missed hop only removes a piece of evidence, it does not invent any, so an over-optimistic detection rate costs little; the diagonal is the correct belief.";
          c4.foot.replaceChildren(`${fmt.int(mm.n_per_cell)} runs per cell (${mm.n_seeds} seeds × ${mm.generators.map(gname).join(", ")}), ${mm.window}-hop window. Source:`, src(PP.files.mismatch));
          c4.table = () => table("Right answers (%) · rows: true detection rate, columns: assumed",
            [{ h: "True \\ assumed" }, ...mm.p_d_grid.map((p) => ({ h: pct(p) }))],
            mm.pd.map((row, i) => [pct(mm.p_d_grid[i]), ...row.map((c) => String(Math.round(c[0] * 100)))]));
        };
      }
      cards.push(c4);

      // ========================================================= claim 5
      const c5 = card(5, "Why it works", "Close to the theoretical limit");
      const cv = DOC.converse;
      if (cv) {
        const box = el("div");
        c5.body.append(OTS.chart.legend([{ label: "OTS, measured", color: COL.ots() }, { label: "Floor no identifier can beat", kind: "dash" }]), box);
        c5.ok = true;
        const ratio = cv.measured.map((m, i) => m / cv.bound[i]);
        const xs = cv.p_d.map((p) => Math.round(p * 100));
        const spec5 = () => ({
          height: 270, ariaLabel: "Hops to identify against the information-theoretic lower bound, as the sensor catches fewer hops",
          x: { label: "Share of hops the sensor catches (%)", min: 100, max: Math.min(...xs), ticks: xs, format: (v) => v + "%" },
          y: { label: "Hops to identify (mean)", min: 0, max: niceMax(Math.max(...cv.measured) + 5), format: String },
          series: [
            { id: "m", label: "OTS", color: COL.ots(), marker: "all", emphasis: true, points: xs.map((x, i) => [x, cv.measured[i]]) },
            { id: "b", label: "Floor", color: OTS.tok("ink-2"), dash: "6 4", marker: "all", points: xs.map((x, i) => [x, cv.bound[i]]) }],
          tooltip: (x) => { const i = xs.indexOf(x); return { title: `Sensor catches ${x}% of hops`, rows: [
            { value: d1(cv.measured[i]), label: "OTS hops (measured)", color: COL.ots() }, { value: d1(cv.bound[i]), label: "lower bound" },
            { value: ratio[i].toFixed(2) + "×", label: "OTS ÷ floor" }] }; },
        });
        c5.render = () => {
          c5.k.textContent = `OTS needs ${Math.min(...ratio).toFixed(1)}–${Math.max(...ratio).toFixed(1)}× the hops of an information-theoretic floor that no identifier can beat, and the floor rises exactly as the sensor worsens`;
          c5.d.textContent = "To be 95% sure, the evidence for the true algorithm must climb to log(19) ≈ 2.9, and on average it can climb at most a fixed amount per hop (less when hops are missed). Dividing gives the floor. "
            + "It is measured on a Markov hopper, the one source in the bank slow enough to make the floor visible; deterministic radios lock in a handful of hops.";
          c5.foot.replaceChildren(el("span", { class: "pc-quoted" }, "Quoted"), ` from ${cv.source}; its result file is not in the repository. The paper's abstract quotes 1.3–2.7×.`);
          c5.table = () => table("Hops to identify vs the lower bound",
            [{ h: "Sensor catches" }, { h: "OTS measured", us: true }, { h: "Lower bound" }, { h: "Ratio", us: true }, { h: "Drift per hop" }],
            xs.map((x, i) => [x + "%", d1(cv.measured[i]), d1(cv.bound[i]), ratio[i].toFixed(2) + "×", String(cv.drift[i])]));
        };
        c5.mountCharts = () => mk(box, OTS.chart.line, spec5);
      }
      cards.push(c5);

      // ========================================================= claim 6
      const c6 = card(6, "Call the next channel", "Predicts algorithms it was never built for");
      if (pr) {
        const box = el("div");
        c6.body.append(el("div", { class: "pc-legrow" }, OTS.chart.legend([sq(METHOD.proposed.label, COL.ots()), sq(METHOD.id_only.label, COL.idonly()),
          { label: `Random guess (${chanceTxt()})`, kind: "dash" }]), el("span", { class: "pc-better" }, "▲ higher is better")), box);
        c6.ok = true;
        sensorSeg(c6, "pred", Object.keys(pr.by_sensor), () => { c6.render(); ch6.h.update(ch6.spec()); });
        var ch6 = { spec: () => {
          const s = pr.by_sensor[sel.pred], gens = Object.keys(s);
          return {
            height: 320, ariaLabel: `Next channel called right on the ${SN(sel.pred).name} sensor: OTS predictor against the bank alone, per algorithm`,
            categories: gens.map((g) => ({ id: g, label: gname(g), sub: GROUP_SUB[groupOf(g)] })),
            series: [{ id: "steady", label: "OTS predictor", color: COL.ots() }, { id: "id_only", label: "Bank alone", color: COL.idonly() }],
            value: (g, k) => 100 * s[g][k], y: { label: "Next channel called right (%)", max: 100, ticks: [0, 25, 50, 75, 100], format: String },
            barWidth: 20, valueLabels: roomy(box, gens.length, 2), valueFormat: (v) => String(Math.round(v)), refLines: [{ y: 100 * pr.chance }],
            tooltip: (c) => { const r = s[c.id]; return { title: `${c.label} · ${SN(sel.pred).name}`, rows: [
              { value: pct0(r.steady), label: "OTS predictor, after it learns", color: COL.ots(), kind: "square" },
              { value: pct0(r.id_only), label: "Bank alone", color: COL.idonly(), kind: "square" },
              { value: pct0(r.top1), label: "OTS, whole run" }, { value: pct0(r.top3), label: "OTS, true channel in its top 3" }] }; },
          };
        } };
        c6.render = () => {
          const s = pr.by_sensor[sel.pred], gens = Object.keys(s), news = gens.filter((g) => groupOf(g) === "new");
          const best = news.reduce((a, b) => (s[b].steady - s[b].id_only > s[a].steady - s[a].id_only ? b : a), news[0]);
          c6.k.textContent = `${SN(sel.pred).name} sensor: on the ${news.length} new algorithms it has no expert for, OTS calls ${pct0(mean(news.map((g) => s[g].steady)))} of next channels right once it has learned them, against ${pct0(mean(news.map((g) => s[g].id_only)))} for the bank alone`
            + (best ? ` (${gname(best)}: ${pct0(s[best].steady)} vs ${pct0(s[best].id_only)})` : "");
          c6.d.textContent = "Naming the algorithm is not required to predict it. Alongside the experts, the predictor carries universal learners (context mixing and a rule search) that pick up a new rule from the hops themselves, and a running vote that trusts whoever has been predicting best. "
            + `Statistical sources stay hard (Markov hopper ${pct0(s.markov ? s.markov.steady : null)}, sine map ${pct0(s.sine_map ? s.sine_map.steady : null)}), and on the unbreakable CSPRNG everyone falls to chance, as they must.`;
          c6.foot.replaceChildren(`${pr.n_seeds} runs per algorithm and sensor, ${pr.window}-hop runs; “after it learns” is the steady state of each run. Chance is 1 in ${OTS.data.M || 16}. Source:`, src(PP.files.predict));
          c6.table = () => table(`Next channel called right · ${SN(sel.pred).name} sensor`,
            [{ h: "Algorithm" }, { h: "Group" }, { h: "OTS after it learns", us: true }, { h: "OTS whole run", us: true }, { h: "OTS top 3", us: true }, { h: "Bank alone" }, { h: "Freq. stats" }],
            gens.map((g) => { const r = s[g]; return [gname(g), GROUP_SUB[groupOf(g)], pct0(r.steady), pct0(r.top1), pct0(r.top3), pct0(r.id_only), pct0(r.markov)]; }));
        };
        c6.mountCharts = () => { Object.assign(ch6, mk(box, OTS.chart.columns, ch6.spec)); };
      }
      cards.push(c6);

      // ========================================================= claim 7
      const c7 = card(7, "Call the next channel", "Beats outside predictors, including a neural network");
      if (pb) {
        const box = el("div"), ms = ["proposed", "ppm", "markov1", "lstm"];
        c7.body.append(el("div", { class: "pc-legrow" }, OTS.chart.legend([...ms.map((m) => sq(METHOD[m].label, METHOD[m].color())), { label: `Random guess (${chanceTxt()})`, kind: "dash" }]),
          el("span", { class: "pc-better" }, "▲ higher is better")), box);
        c7.ok = true;
        sensorSeg(c7, "pbase", Object.keys(pb.by_sensor), () => { c7.render(); ch7.h.update(ch7.spec()); });
        var ch7 = { spec: () => {
          const s = pb.by_sensor[sel.pbase].gens, gens = Object.keys(s);
          return {
            height: 320, ariaLabel: `Next channel called right on the ${SN(sel.pbase).name} sensor: OTS against PPM-C, frequency statistics and a trained LSTM`,
            categories: gens.map((g) => ({ id: g, label: gname(g), sub: GROUP_SUB[groupOf(g)] })),
            series: ms.map((m) => ({ id: m, label: METHOD[m].short, color: METHOD[m].color() })),
            value: (g, m) => 100 * s[g][m], y: { label: "Next channel called right (%)", max: 100, ticks: [0, 25, 50, 75, 100], format: String },
            barWidth: 16, valueLabels: roomy(box, gens.length, ms.length), valueFormat: (v) => String(Math.round(v)), refLines: [{ y: 100 * pb.chance }],
            tooltip: (c) => ({ title: `${c.label} · ${SN(sel.pbase).name}`, rows: ms.map((m) => ({ value: pct0(s[c.id][m]), label: METHOD[m].label, color: METHOD[m].color(), kind: "square" })) }),
          };
        } };
        c7.render = () => {
          const s = pb.by_sensor[sel.pbase].gens, lf = s.lagged_fib, lfsr = s.lfsr;
          c7.k.textContent = lf ? `${SN(sel.pbase).name} sensor: a neural network trained on the known families calls ${pct0(lf.lstm)} of the new lagged generator's hops; untrained OTS calls ${pct0(lf.proposed)}` : "OTS against outside predictors";
          c7.d.textContent = (lfsr ? `The LSTM is excellent on what it was trained on (${pct0(lfsr.lstm)} on the LFSR) but learned its families, not how to learn a new rule. ` : "")
            + (lf ? `PPM-C, a training-free text-compression predictor, partly finds the lagged rule (${pct0(lf.ppm)}) but has no sensor model, so every missed hop breaks its context and it fades as the sensor worsens.` : "");
          c7.foot.replaceChildren(`${pb.n_seeds} runs per algorithm and sensor on the same runs as claim 6 (${fmt.int(pb.n_episodes)} in total), steady-state accuracy. The LSTM (2 layers, 128 units) was trained on ${fmt.int(pb.by_sensor[sel.pbase].lstm_train)} sequences of ${pb.train_families.length} known families at this sensor. Source:`, src(PP.files.pred_baselines));
          c7.table = () => table(`Outside predictors · ${SN(sel.pbase).name} sensor`,
            [{ h: "Algorithm" }, ...ms.map((m) => ({ h: METHOD[m].short, us: m === "proposed" }))],
            Object.keys(s).map((g) => [gname(g), ...ms.map((m) => pct0(s[g][m]))]));
        };
        c7.mountCharts = () => { Object.assign(ch7, mk(box, OTS.chart.columns, ch7.spec)); };
      }
      cards.push(c7);

      // ========================================================= claim 8
      const c8 = card(8, "Call the next channel", "Says how sure it is, and keeps the promise");
      if (cp) {
        const aBox = el("div"), bBox = el("div");
        c8.body.append(el("div", { class: "pc-pair" },
          el("div", {}, el("p", { class: "pc-ch" }, "Promise vs delivery ", el("small", {}, "· all algorithms pooled")),
            OTS.chart.legend([{ label: "Conformal set", color: COL.ots() }, { label: "Promised = delivered", kind: "dash" }]), aBox),
          el("div", {}, el("p", { class: "pc-ch" }, "True next channel covered, per algorithm ", el("small", {}, "· promise 90%")),
            OTS.chart.legend([sq("Conformal set", COL.ots()), sq("Fixed top-3 shortlist", COL.base())]), bBox)));
        c8.ok = true;
        sensorSeg(c8, "cp", Object.keys(cp.by_sensor), () => { c8.render(); chA.h.update(chA.spec()); chB.h.update(chB.spec()); });
        var chA = { spec: () => {
          const c = cp.by_sensor[sel.cp].calib, xs = c.target.map((t) => Math.round(t * 100));
          const lo = Math.min(c.top1, c.top3, ...c.target) * 100;
          return {
            height: 260, ariaLabel: "Coverage the conformal set delivers against the coverage it promises",
            x: { label: "Promised coverage (%)", min: Math.min(...xs), max: Math.max(...xs), ticks: xs, format: String },
            y: { label: "Delivered coverage (%)", min: Math.floor((lo - 5) / 10) * 10, max: 100, format: String },
            series: [
              { id: "conf", label: "Conformal", color: COL.ots(), marker: "all", emphasis: true, points: xs.map((x, i) => [x, 100 * c.conformal[i]]) },
              { id: "diag", label: "Promise", color: OTS.tok("ink-2"), dash: "6 4", endLabel: false, points: xs.map((x) => [x, x]) }],
            refLines: [{ y: 100 * c.top3, label: `Fixed top-3: ${pct0(c.top3)}` }, { y: 100 * c.top1, label: `Single best guess: ${pct0(c.top1)}`, below: true }],
            tooltip: (x) => { const i = xs.indexOf(x); return { title: `Promised ${x}%`, rows: [{ value: pct1(c.conformal[i]), label: "conformal set delivered", color: COL.ots() }] }; },
          };
        } };
        var chB = { spec: () => {
          const op = cp.by_sensor[sel.cp].op, gens = Object.keys(op);
          return {
            height: 260, ariaLabel: "Share of next channels covered per algorithm: conformal set against a fixed top-3 shortlist",
            categories: gens.map((g) => ({ id: g, label: gshort(g), full: gname(g) })),
            series: [{ id: "cov", label: "Conformal", color: COL.ots() }, { id: "top3_cov", label: "Top-3", color: COL.base() }],
            value: (g, k) => 100 * op[g][k], y: { label: "Covered (%)", max: 100, ticks: [0, 25, 50, 75, 100], format: String },
            barWidth: 14, valueLabels: false, refLines: [{ y: 100 * cp.target, label: "promise" }],
            tooltip: (c) => { const r = op[c.id]; return { title: `${c.full} · ${SN(sel.cp).name}`, rows: [
              { value: pct1(r.cov), label: "conformal set covered", color: COL.ots(), kind: "square" },
              { value: d1(r.size), label: "channels in the set (mean)" },
              { value: pct1(r.top3_cov), label: "fixed top-3 covered", color: COL.base(), kind: "square" }] }; },
          };
        } };
        c8.render = () => {
          const op = cp.by_sensor[sel.cp].op, v = Object.values(op);
          const minCov = Math.min(...v.map((o) => o.cov)), minTop3 = Math.min(...v.map((o) => o.top3_cov));
          const sz = (g) => (op[g] ? d1(op[g].size) : "—");
          c8.k.textContent = `${SN(sel.cp).name} sensor: promised 90%, OTS's channel set covers the true next channel at least ${Math.floor(minCov * 100)}% of the time on every algorithm; a fixed shortlist of three covers as little as ${pct0(minTop3)}`;
          c8.d.textContent = `Instead of one guess, OTS names a set of channels sized to its uncertainty, with a coverage guarantee that needs no assumptions about the radio (adaptive conformal inference). `
            + `The set is ${sz("lfsr")} channels on a pinned-down shift register and ${sz("csprng")} of ${OTS.data.M || 16} on the CSPRNG: the data sets the size, not an operator.`;
          c8.foot.replaceChildren(`${cp.n_seeds} runs per algorithm, ${cp.window}-hop runs, target ${Math.round(cp.target * 100)}%. The calibration bit fires only on cleanly heard hops, which is all a passive sensor can check. Source:`, src(PP.files.conformal));
          c8.table = () => table(`Conformal sets · ${SN(sel.cp).name} sensor`,
            [{ h: "Algorithm" }, { h: "Covered", us: true }, { h: "Channels in set", us: true }, { h: "Predictability κ" }, { h: "Fixed top-3 covered" }],
            Object.keys(op).map((g) => [gname(g), pct1(op[g].cov), d1(op[g].size), op[g].kappa.toFixed(2), pct1(op[g].top3_cov)]));
        };
        c8.mountCharts = () => { Object.assign(chA, mk(aBox, OTS.chart.line, chA.spec)); Object.assign(chB, mk(bBox, OTS.chart.columns, chB.spec)); };
      }
      cards.push(c8);

      // ========================================================= claim 9
      const c9 = card(9, "Act on it", "Jams the promised share of hops");
      if (dn) {
        const aBox = el("div"), bBox = el("div");
        const M = dn.budgets[dn.budgets.length - 1];
        c9.body.append(el("div", { class: "pc-pair" },
          el("div", {}, el("p", { class: "pc-ch" }, "Hops denied ", el("small", {}, "· tones the set asks for, no budget cap")),
            OTS.chart.legend([sq("OTS conformal jam plan", COL.ots()), sq("Jam a fixed top 3", COL.base())]), aBox),
          el("div", {}, el("p", { class: "pc-ch" }, "The price: channels jammed per hop ", el("small", {}, `· of ${M}`)),
            OTS.chart.legend([sq("Tones the set asks for (mean)", COL.ots())]), bBox)));
        c9.ok = true;
        sensorSeg(c9, "deny", Object.keys(dn.by_sensor), () => { c9.render(); chA9.h.update(chA9.spec()); chB9.h.update(chB9.spec()); });
        var chA9 = { spec: () => {
          const s = dn.by_sensor[sel.deny], gens = Object.keys(s);
          return {
            height: 260, ariaLabel: "Share of hops denied per algorithm: conformal jam plan against jamming a fixed top 3",
            categories: gens.map((g) => ({ id: g, label: gshort(g), full: gname(g) })),
            series: [{ id: "u", label: "OTS", color: COL.ots() }, { id: "t", label: "Top 3", color: COL.base() }],
            value: (g, k) => 100 * (k === "u" ? s[g].unlimited : s[g].topk[2]), y: { label: "Hops denied (%)", max: 100, ticks: [0, 25, 50, 75, 100], format: String },
            barWidth: 16, valueLabels: false, refLines: [{ y: 100 * dn.target, label: "promise" }],
            tooltip: (c) => ({ title: `${c.full} · ${SN(sel.deny).name}`, rows: [
              { value: pct1(s[c.id].unlimited), label: "denied, OTS jam plan", color: COL.ots(), kind: "square" },
              { value: d1(s[c.id].mean_tones), label: "tones it used (mean)" },
              { value: pct1(s[c.id].topk[2]), label: "denied, fixed top 3", color: COL.base(), kind: "square" }] }),
          };
        } };
        var chB9 = { spec: () => {
          const s = dn.by_sensor[sel.deny], gens = Object.keys(s);
          return {
            height: 260, ariaLabel: "Mean number of channels the conformal jam plan lights per hop, per algorithm",
            categories: gens.map((g) => ({ id: g, label: gshort(g), full: gname(g) })), series: [{ id: "n", label: "Tones", color: COL.ots() }],
            value: (g) => s[g].mean_tones, y: { label: "Channels jammed per hop", max: M, ticks: [0, 4, 8, 12, 16].filter((t) => t <= M), format: String },
            barWidth: 24, valueLabels: true, valueFormat: d1, refLines: [{ y: M, label: "whole band" }],
            tooltip: (c) => ({ title: `${c.full} · ${SN(sel.deny).name}`, rows: [{ value: d1(s[c.id].mean_tones), label: `channels jammed per hop, of ${M}`, color: COL.ots(), kind: "square" }] }),
          };
        } };
        c9.render = () => {
          const s = dn.by_sensor[sel.deny], v = Object.entries(s);
          const minU = Math.min(...v.map(([, x]) => x.unlimited));
          const cheap = v.reduce((a, b) => (b[1].mean_tones < a[1].mean_tones ? b : a)), dear = v.reduce((a, b) => (b[1].mean_tones > a[1].mean_tones ? b : a));
          c9.k.textContent = `${SN(sel.deny).name} sensor: the jam plan denies at least ${Math.floor(minU * 100)}% of hops on every algorithm, from ${d1(cheap[1].mean_tones)} channels per hop on the ${gname(cheap[0])} to ${d1(dear[1].mean_tones)} of ${M} on the ${gname(dear[0])}`;
          c9.d.textContent = "The channel set from claim 8 is the jam plan: light every channel in it and the true hop is hit at the promised rate. What changes is the price, and the price is information: needing most of the band is OTS reporting that only a barrage works. "
            + `A fixed top-3 plan is cheap but blind: it denies ${pct0(s.markov ? s.markov.topk[2] : null)} of a Markov hopper's hops and ${pct0(s.csprng ? s.csprng.topk[2] : null)} of the CSPRNG's.`;
          c9.foot.replaceChildren(`${dn.n_seeds} runs per algorithm, ${dn.window}-hop runs, conformal target ${Math.round(dn.target * 100)}%; a hop counts as denied when its true channel is jammed, scored on every hop. Source:`, src(PP.files.denial));
          c9.table = () => table(`Hops denied against the tone budget · ${SN(sel.deny).name} sensor`,
            [{ h: "Algorithm" }, { h: "No cap", us: true }, { h: "Tones used", us: true }, ...[1, 2, 4, 8].map((b) => ({ h: `Budget ${b}` }))],
            v.map(([g, x]) => [gname(g), pct0(x.unlimited), d1(x.mean_tones), ...[1, 2, 4, 8].map((b) => pct0(x.denied[dn.budgets.indexOf(b)]))]));
        };
        c9.mountCharts = () => { Object.assign(chA9, mk(aBox, OTS.chart.columns, chA9.spec)); Object.assign(chB9, mk(bBox, OTS.chart.columns, chB9.spec)); };
      }
      cards.push(c9);

      // ========================================================= claim 10
      const c10 = card(10, "Act on it", "Notices when the radio changes algorithm");
      if (sw) {
        const tl = el("div", { class: "pc-tiles" }), box = el("div");
        c10.body.append(tl, el("p", { class: "pc-ch", style: "margin-top:18px" }, "Next channel called right after a switch ", el("small", {}, "· by hops since the switch")),
          OTS.chart.legend([sq("OTS switch tracker", COL.ots()), { label: `Random guess (${chanceTxt()})`, kind: "dash" }]), box);
        c10.ok = true;
        sensorSeg(c10, "sw", Object.keys(sw.by_sensor), () => { c10.render(); ch10.h.update(ch10.spec()); });
        var ch10 = { spec: () => {
          const t = sw.by_sensor[sel.sw].top1_since_switch, ks = Object.keys(t);
          return {
            height: 230, ariaLabel: "Next channel called right in the hops after an algorithm switch",
            categories: ks.map((k) => ({ id: k, label: k.replace("-", "–") + " hops" })), series: [{ id: "v", label: "OTS", color: COL.ots() }],
            value: (k) => 100 * t[k], y: { label: "Called right (%)", max: 100, ticks: [0, 25, 50, 75, 100], format: String },
            x: { label: "Hops since the radio switched" }, barWidth: 34, valueLabels: true, valueFormat: (v) => String(Math.round(v)), refLines: [{ y: 100 / (OTS.data.M || 16) }],
          };
        } };
        c10.render = () => {
          const s = sw.by_sensor[sel.sw];
          const tile = (l, v, sub) => el("div", { class: "tile" }, el("div", { class: "tile-l" }, l), el("div", { class: "tile-v num" }, v), el("div", { class: "tile-s" }, sub));
          fill(tl,
            tile("Switches detected", pct0(s.detect_rate), `of ${s.n_switches} switches`),
            tile("Detected after", [d1(s.delay_median), el("small", {}, " hops")], `median · 9 in 10 within ${Math.round(s.delay_p90)}`),
            tile("Dangerous locks", d1(s.dangerous_locks_per_regime * 100), `per 100 algorithm stretches: ${s.n_dangerous === 0 ? "none" : s.n_dangerous} in ${s.n_regimes}`),
            tile("False alarms", d1(s.false_alarms_per_100_hops), "switches reported per 100 hops when none happened"));
          c10.k.textContent = `${SN(sel.sw).name} sensor: ${pct0(s.detect_rate)} of algorithm switches caught, a median ${d1(s.delay_median)} hops after they happen, with ${s.n_dangerous === 0 ? "no dangerous locks" : `${s.n_dangerous} dangerous lock${s.n_dangerous === 1 ? "" : "s"} in ${s.n_regimes} stretches`}`;
          c10.d.textContent = "A radio can change its algorithm mid-engagement. The switch tracker spots the change, dates it, forgets the old evidence and starts naming and predicting again. The chart shows how fast prediction comes back.";
          c10.foot.replaceChildren(`${sw.n_plans} random switch plans over all ${conf ? conf.generators.length : 13} algorithms (${s.n_switches} switches, stretches of ${sw.regime_hops ? sw.regime_hops.join("–") : "50–90"} hops), ${sw.window}-hop runs. Source:`, src(PP.files.switch));
          c10.table = () => table("Switch tracking by sensor",
            [{ h: "Sensor" }, { h: "Detected", us: true }, { h: "Delay median / 90th" }, { h: "Dangerous / 100 stretches" }, ...Object.keys(s.top1_since_switch).map((k) => ({ h: `Right, ${k.replace("-", "–")}` }))],
            sensorOrder(Object.keys(sw.by_sensor)).map((sid) => { const x = sw.by_sensor[sid]; return [SN(sid).name, pct0(x.detect_rate), `${d1(x.delay_median)} / ${Math.round(x.delay_p90)}`, d1(x.dangerous_locks_per_regime * 100), ...Object.values(x.top1_since_switch).map(pct0)]; }));
        };
        c10.mountCharts = () => { Object.assign(ch10, mk(box, OTS.chart.columns, ch10.spec)); };
      }
      cards.push(c10);

      // ========================================================= claim 11
      const c11 = card(11, "Act on it", "Fast enough to run live");
      if (rt) {
        const tl = el("div", { class: "pc-tiles" });
        c11.ok = true;
        c11.body.append(tl);
        c11.render = () => {
          const a = rt.by_M["16"], b = rt.by_M["64"];
          const tot = (r) => r.bank.med + r.predictor.med + r.conformal_set.med;
          const ms = (v) => (v < 0.1 ? v.toFixed(2) : v < 10 ? v.toFixed(1) : String(Math.round(v)));
          const tile = (l, v, sub, hot) => el("div", { class: "tile" + (hot ? " hot" : "") }, el("div", { class: "tile-l" }, l), el("div", { class: "tile-v num" }, v, el("small", {}, " ms")), el("div", { class: "tile-s" }, sub));
          fill(tl,
            tile("Name the radio", ms(a.bank.med), `${OTS.hypotheses.length - 1} experts and “unknown”, per hop (median)`),
            tile("Predict the next channel", ms(a.predictor.med), `all experts plus the universal learners · 95th percentile ${ms(a.predictor.p95)} ms`),
            tile("Size the jam", ms(a.conformal_set.med), "conformal set and tone choice"),
            tile("All three", ms(tot(a)), b ? `${ms(tot(b))} ms with 64 channels` : "", true),
            tile("Switch tracker", ms(a.tracker.med), "several banks at once; the slow part"));
          c11.k.textContent = `About ${ms(tot(a))} ms per hop on one CPU core: inside the time a radio hopping a few hundred times a second spends on each channel`;
          c11.d.textContent = `The cost follows the number of candidate settings, not the number of channels, so 64 channels cost little more than 16. Two parts are slower: the predictor's search experts fire in bursts, and the switch tracker, which runs several banks side by side, takes ${ms(a.tracker.med)} ms.`;
          c11.foot.replaceChildren(`Unoptimised NumPy on one core of an ${rt.cpu}, ${rt.threads} thread; a one-off ${d1(a.build_s.bank)} s builds the experts' tables. Source:`, src(PP.files.runtime));
          c11.table = () => table("Median time per hop (95th percentile)",
            [{ h: "Component" }, { h: "16 channels", us: true }, { h: "64 channels" }],
            [["bank", "Name the radio"], ["predictor", "Predict"], ["conformal_set", "Conformal set"], ["tracker", "Switch tracker"]].map(([k, n]) =>
              [n, `${ms(a[k].med)} ms (${ms(a[k].p95)})`, b ? `${ms(b[k].med)} ms (${ms(b[k].p95)})` : "—"]));
        };
      }
      cards.push(c11);

      // ========================================================= claim 12
      const c12 = card(12, "Honest limits", "A radio that fights back");
      const ev = DOC.evasion, dc = DOC.deception;
      if (ev && dc) {
        const box = el("div"), verd = el("ul", { class: "pc-checks" });
        c12.body.append(el("div", { class: "pc-split" },
          el("div", {}, el("p", { class: "pc-ch" }, "Hops the jammer denies ", el("small", {}, "· strong sensor")),
            OTS.chart.legend([sq("Radio ignores the jammer", COL.ots()), sq("Radio evades: dodges, spreads, re-keys", COL.orange())]), box),
          el("div", {}, el("p", { class: "pc-ch" }, "Decoys against identification"), verd)));
        c12.ok = true;
        const spec12 = () => ({
          height: 270, ariaLabel: "Share of hops the OTS jammer denies against a passive radio and against a radio that evades",
          categories: ev.rows.map((r) => ({ id: r.id, label: gshort(r.id), sub: GROUP_SUB[groupOf(r.id)] })),
          series: [{ id: "p", label: "Ignores", color: COL.ots() }, { id: "e", label: "Evades", color: COL.orange() }],
          value: (id, k) => { const r = ev.rows.find((x) => x.id === id); return 100 * (k === "p" ? r.denied_passive : r.denied_evading); },
          y: { label: "Hops denied (%)", max: 100, ticks: [0, 25, 50, 75, 100], format: String },
          barWidth: 22, valueLabels: roomy(box, ev.rows.length, 2), valueFormat: (v) => "≈" + Math.round(v),
          tooltip: (c) => { const r = ev.rows.find((x) => x.id === c.id); return { title: gname(c.id), rows: [
            { value: "≈" + pct0(r.denied_passive), label: "denied, radio ignores the jammer", color: COL.ots(), kind: "square" },
            { value: "≈" + pct0(r.denied_evading), label: "denied, radio evades", color: COL.orange(), kind: "square" },
            { value: `${r.kappa_passive} → ${r.kappa_evading}`, label: "predictability κ" }] }; },
        });
        c12.render = () => {
          const l = ev.rows.find((r) => r.id === "lfsr") || ev.rows[0];
          c12.k.textContent = `A radio that fights back can escape: reacting to being jammed cuts denial from ≈${pct0(l.denied_passive)} to ≈${pct0(l.denied_evading)}, and persistent decoys can fool identification`;
          c12.d.textContent = "So far the radio ignored the jammer. Here it fights back from what a real radio knows (its own failed hops): it dodges jammed channels, spreads its hops out and re-keys, so prediction and denial collapse together. The CSPRNG has nothing left to lose.";
          fill(verd,
            check("good", "✓", `Intermittent decoy (on ≤ ${Math.round(dc.low_persistence * 100)}% of the time) over an unbreakable radio: OTS still answers “unknown” about ${pct0(dc.bank_unknown_low)} of the time, fooled in ${pct0(dc.bank_false_id_low)}`),
            check("bad", "✕", `The standard sequential test, with no “unknown” option, names the decoy every time (${pct0(dc.sprt_false_id)})`),
            check("warn", "!", `A coherent decoy on ${Math.round(dc.defeats_h0_from * 100)}% of the time or more embeds a real LFSR and fools OTS too: separating emitters is future work`),
            check("warn", "!", `Decoys also pull the jammer off the real hop: denial ≈${pct0(dc.counter_jam_denied_before)} → ≈${pct0(dc.counter_jam_denied_after)}`));
          c12.foot.replaceChildren(el("span", { class: "pc-quoted" }, "Quoted"), ` from ${ev.source} and ${dc.source}; their result files are not in the repository. Playable in the Control room: tick “Evade” or “Deceive”.`);
          c12.table = () => table("Jammer denial, passive vs evading radio (strong sensor, approximate)",
            [{ h: "Radio" }, { h: "Denied, ignores", us: true }, { h: "Denied, evades" }, { h: "κ ignores → evades" }],
            ev.rows.map((r) => [gname(r.id), "≈" + pct0(r.denied_passive), "≈" + pct0(r.denied_evading), `${r.kappa_passive} → ${r.kappa_evading}`]));
        };
        c12.mountCharts = () => mk(box, OTS.chart.columns, spec12);
      }
      cards.push(c12);

      // ========================================================= claim 13
      const c13 = card(13, "Honest limits", "What it does not do yet");
      {
        const lf = conf && conf.generators.includes("lcg_full") ? confRow(T, "lcg_full") : null;
        const st = pr && pr.by_sensor[T];
        const items = [
          ["Short-cycle radios look statistical", lf ? `The full-state LCG is named right ${pct0(lf.acc)} of the time on the Typical sensor: at 16 channels its cycle is at most 16 hops and reads as a Markov cycle. A soft miss, never a crack claimed.` : "At 16 channels a full-state LCG reads as a Markov cycle."],
          ["Statistical sources stay partly unpredictable", st ? `Only rules can be learned. A Markov hopper (${pct0(st.markov && st.markov.steady)}) and the sine map (${pct0(st.sine_map && st.sine_map.steady)}) are called above chance but far from always; the conformal set then grows to say so.` : "Markov and sine-map sources are only partly predictable."],
          ["One emitter at a time", "The bank assumes the detections come from one radio. A persistent, coherent decoy defeats it (claim 12); sorting interleaved emitters is the next step."],
          ["Simulated, not yet on real captures", "Slot-synchronous sensor with independent misses and ghosts. Hop-timing jitter, bursty fades and real recordings are future work."],
          ["Same authors on both sides", "The generators and the experts were written by the same people. The held-out algorithms reduce this bias but do not remove it."],
          ["Guarantees have edges", "The anytime error guarantee is proven for radios inside the bank; outside it, protection comes from the “unknown” option and is measured, not proven."],
        ];
        c13.ok = true;
        c13.body.append(el("ul", { class: "pc-limits" }, items.map(([t, d]) => el("li", {}, el("b", {}, t), d))));
        c13.render = () => {
          c13.k.textContent = "Knowing where it stops is part of the result";
          c13.d.textContent = "The limits the paper states, with the numbers behind them.";
          c13.foot.replaceChildren("From the paper's Discussion and Limitations; numbers from ", src(PP.files.confusion || "—"), " and ", src(PP.files.predict || "—"), ".");
        };
      }
      cards.push(c13);

      // ---------------------------------------------------------- layout
      const byCh = { identify: [c1, c2], why: [c3, c4, c5], predict: [c6, c7, c8], act: [c9, c10, c11], limits: [c12, c13] };
      const page = el("div", { class: "page proof" }, hero);
      for (const c of CHAPTERS) {
        const list = byCh[c.id].filter((x) => x.ok);
        page.append(chapter(c));
        for (const x of list) page.append(x.node);
        chapterFirst[c.id] = list.length ? list[0].node : null;
      }
      root.append(page);

      // charts mount after the page is in the DOM so they know their width
      for (const c of cards) if (c.mountCharts) c.mountCharts();

      function renderTables() {
        for (const c of cards) {
          c.tbl.hidden = !OTS.prefs.tables || !c.table;
          if (!c.tbl.hidden) fill(c.tbl, c.table()); else OTS.clear(c.tbl);
        }
        tblInp.checked = !!OTS.prefs.tables;
      }
      // a re-render (sensor switch) also refreshes that card's table
      for (const c of cards) {
        const r = c.render;
        c.render = () => { r(); if (!c.tbl.hidden || OTS.prefs.tables) { c.tbl.hidden = !OTS.prefs.tables || !c.table; if (!c.tbl.hidden) fill(c.tbl, c.table()); } };
        c.render();
      }
      renderTables();
      offs.push(OTS.on("prefs", renderTables));
      OTS.present.attach(page);

      // value labels switch on/off with the available width
      let raf = 0;
      const onResize = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { for (const c of charts) c.h.update(c.spec()); }); };
      window.addEventListener("resize", onResize);

      return () => {
        offs.forEach((off) => off());
        handles.forEach((h) => h.destroy());
        window.removeEventListener("resize", onResize);
        cancelAnimationFrame(raf);
        OTS.tip.hide();
      };
    },
  });
})();
