/* OTS dashboard runtime: shared data helpers, chart kit, player, settings, router.
 *
 * Every view lives in views/<id>.js and calls OTS.views.register({...}). Views
 * use only this API (documented inline) so the pages look and behave alike.
 * Plain classic scripts (no ES modules) so the page also works from file://.
 */
window.OTS = (function () {
  "use strict";
  const OTS = {};
  const DATA = (OTS.data = window.OTS_DATA || null);

  // ================================================================== events
  const handlers = new Map();
  /** Subscribe to an event; returns an unsubscribe function.
   *  Events: 'load' (trace), 'tick' ({t, prev, animate}), 'lock' (trace), 'end' (trace),
   *  'state' ({playing}), 'prefs' (prefs), 'live' (bool), 'route' (viewId), 'resize'. */
  OTS.on = (evt, fn) => {
    if (!handlers.has(evt)) handlers.set(evt, new Set());
    handlers.get(evt).add(fn);
    return () => handlers.get(evt).delete(fn);
  };
  OTS.emit = (evt, payload) => {
    for (const fn of [...(handlers.get(evt) || [])]) {
      try { fn(payload); } catch (err) { console.error(`[OTS] ${evt} handler failed`, err); }
    }
  };

  // ===================================================================== DOM
  /** el('div', {class, style:'css text', onclick: fn, 'data-x': 1, title}, ...children)
   *  Children may be strings, numbers, nodes or (nested) arrays; null/false are skipped.
   *  Text is always inserted as text (never parsed as HTML). */
  OTS.el = function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    setAttrs(n, attrs);
    appendKids(n, kids);
    return n;
  };
  /** svg('line', {x1, y1, ...}, ...children) -- SVG namespace; strings become text. */
  OTS.svg = function svg(tag, attrs, ...kids) {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    setAttrs(n, attrs);
    appendKids(n, kids);
    return n;
  };
  function setAttrs(n, attrs) {
    if (!attrs) return;
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.setAttribute("class", v);
      else if (k === "style") n.style.cssText = v;
      else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
  }
  function appendKids(n, kids) {
    for (const c of kids.flat(Infinity)) {
      if (c == null || c === false) continue;
      n.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
  }
  OTS.clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); return n; };
  OTS.$ = (sel, root) => (root || document).querySelector(sel);

  // ================================================================== format
  OTS.fmt = {
    /** probability -> "97.3%", ">99.99%", "<0.01%", "0.42%" */
    pct(p) {
      if (p == null || isNaN(p)) return "—";
      const v = p * 100;
      if (v >= 99.995) return ">99.99%";
      if (v >= 99) return v.toFixed(2) + "%";
      if (v === 0) return "0%";
      if (v < 0.01) return "<0.01%";
      if (v < 1) return v.toFixed(2) + "%";
      return v.toFixed(1) + "%";
    },
    /** probability -> whole percent "97%" */
    pct0: (p) => (p == null || isNaN(p)) ? "—" : Math.round(p * 100) + "%",
    int: (n) => (n == null || isNaN(n)) ? "—" : Math.round(n).toLocaleString("en-US"),
    /** channel index -> "07" */
    ch: (c) => String(c).padStart(2, "0"),
    /** rate per 100 engagements, one decimal when small: 0.0167 -> "1.7" */
    per100: (p) => (p == null || isNaN(p)) ? "—" : (p * 100 < 10 && p > 0 ? (p * 100).toFixed(1) : String(Math.round(p * 100))),
  };

  // ======================================================= tokens and names
  const tokCache = new Map();
  /** Resolved CSS custom property, e.g. OTS.tok('accent'). Cached. */
  OTS.tok = (name) => {
    if (!tokCache.has(name)) tokCache.set(name, getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim());
    return tokCache.get(name);
  };
  const HYP = new Map(), EMIT = new Map();
  if (DATA) {
    for (const h of DATA.hypotheses) HYP.set(h.id, h);
    for (const e of DATA.emitters) EMIT.set(e.id, e);
    // Regime-switch scenarios are selectable in the live demo but kept out of
    // OTS.emitters, so the scoreboard and other pages are unaffected; register
    // their metadata so emitName/emitter resolve a scenario id.
    for (const s of (DATA.scenarios || [])) {
      if (!EMIT.has(s.id)) EMIT.set(s.id, { ...s, scenario: true });
    }
  }
  OTS.hypotheses = DATA ? DATA.hypotheses : [];
  OTS.emitters = DATA ? DATA.emitters : [];
  OTS.scenarios = DATA && DATA.scenarios ? DATA.scenarios : [];
  OTS.presets = DATA ? DATA.presets : [];
  OTS.searchSizes = DATA ? DATA.search_sizes || {} : {};
  const hid = (id) => (id === "unknown" ? "H0" : id);
  /** Series colour for a hypothesis id ('H0' or 'unknown' -> neutral grey). */
  OTS.hypColor = (id) => OTS.tok("h-" + hid(id)) || "#8b95a7";
  /** Display names. */
  OTS.hypName = (id) => (hid(id) === "H0" ? "Unknown (H₀)" : (HYP.get(id)?.name || id));
  OTS.hypDesc = (id) => HYP.get(hid(id))?.desc || "";
  OTS.emitName = (id) => EMIT.get(id)?.name || id;
  OTS.emitter = (id) => EMIT.get(id) || null;
  OTS.preset = (id) => OTS.presets.find((p) => p.id === id) || null;
  /** "Typical sensor (catches 80% of hops)" style label for any trace. */
  OTS.sensorLabel = (tr) => {
    const p = OTS.preset(tr.preset);
    const pd = Math.round(tr.p_d * 100);
    return (p ? p.name + " sensor" : "Live run") + ` · catches ${pd}% of hops`;
  };
  /** Expected ghost (false-alarm) detections per hop for a trace. */
  OTS.ghostsPerHop = (tr) => tr.p_fa * (tr.M - 1);

  // ================================================================ traces
  const TRACES = new Map();
  if (DATA) for (const tr of DATA.traces) {
    const k = tr.emitter + "|" + tr.preset;
    if (!TRACES.has(k)) TRACES.set(k, []);
    TRACES.get(k).push(tr);
  }
  /** Recorded engagements for an emitter + sensor preset (array, may be empty). */
  OTS.tracesFor = (emitter, preset) => TRACES.get(emitter + "|" + preset) || [];

  /* Trace layout (all arrays indexed by 0-based slot j = hop number - 1):
   *   tr.truth[j]  true channel          tr.dets[j]  detected channels (what OTS sees)
   *   tr.post[j][k] posterior after hop j+1, k aligned with tr.labels ('H0' first)
   *   tr.fc[j]     bank next-hop forecast for slot j+1, in permille per channel
   *   tr.fc_pred[j] Fixed-Share predictor's forecast for slot j+1 (permille per channel)
   *   tr.pred_top1 / tr.pred_top1_steady  predictor next-hop top-1 (whole run / 2nd half)
   *   tr.new_algo  true when the bank has no expert for this emitter
   *   tr.alive[j]  candidate generators still plausible   tr.kappa[j] predictability
   *   tr.declared / tr.tau (hop of the lock, 1-based) / tr.decision / tr.correct
   *   tr.secret_true / tr.secret_recovered / tr.secret_match / tr.baseline {decision, declared, tau, correct}
   * "t" below is the number of hops shown so far (0..window). */
  const T = (OTS.T = {});
  T.argmax = (a) => { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; };
  T.idx = (tr, id) => tr.labels.indexOf(hid(id));
  /** Leading hypothesis after t hops: {k, id, p}. */
  T.leader = (tr, t) => {
    if (t <= 0) return { k: 0, id: tr.labels[0], p: 1 / tr.labels.length };
    const post = tr.post[t - 1], k = T.argmax(post);
    return { k, id: tr.labels[k], p: post[k] };
  };
  /** Posterior of a hypothesis after t hops (null before the first hop). */
  T.post = (tr, t, id) => (t <= 0 ? null : tr.post[t - 1][T.idx(tr, id)]);
  /** Has the lock happened by hop t? */
  T.lockedAt = (tr, t) => !!(tr.declared && tr.tau != null && t >= tr.tau);
  /** Forecast distribution (permille array) made before slot j, or null for j = 0. */
  T.forecastFor = (tr, j) => (j <= 0 || j - 1 >= tr.fc.length ? null : tr.fc[j - 1]);
  /** Top forecast channel for slot j: {ch, p (0..1)} or null. */
  T.forecastTop = (tr, j) => { const f = T.forecastFor(tr, j); if (!f) return null; const ch = T.argmax(f); return { ch, p: f[ch] / 1000 }; };
  /** The Fixed-Share predictor's forecast (permille) before slot j, or null. */
  T.predForecastFor = (tr, j) => (j <= 0 || !tr.fc_pred || j - 1 >= tr.fc_pred.length ? null : tr.fc_pred[j - 1]);
  /** The predictor's top forecast channel for slot j: {ch, p} or null. */
  T.predForecastTop = (tr, j) => { const f = T.predForecastFor(tr, j); if (!f) return null; const ch = T.argmax(f); return { ch, p: f[ch] / 1000 }; };
  /** The predictor's next-hop hit rate over the whole run so far (hops 2..t). */
  T.predHitStats = (tr, t) => {
    let n = 0, hit = 0;
    for (let j = 1; j < t; j++) { const f = T.predForecastTop(tr, j); if (!f) continue; n++; if (f.ch === tr.truth[j]) hit++; }
    return { n, hit, rate: n ? hit / n : null };
  };
  /** Is this emitter a "new algorithm" the bank has no expert for? */
  T.isNewAlgo = (tr) => !!(tr && tr.new_algo);
  /** Next-hop hit rate up to hop t: since the lock if locked, else overall. */
  T.hitStats = (tr, t) => {
    const sinceLock = T.lockedAt(tr, t), from = sinceLock ? tr.tau : 1;
    let n = 0, hit = 0;
    for (let j = from; j < t; j++) { const f = T.forecastTop(tr, j); if (!f) continue; n++; if (f.ch === tr.truth[j]) hit++; }
    return { n, hit, rate: n ? hit / n : null, sinceLock };
  };
  /** Sensor errors in the first t hops: missed true hops and ghost detections. */
  T.sensorStats = (tr, t) => {
    let miss = 0, ghost = 0;
    for (let j = 0; j < t; j++) { const d = tr.dets[j], c = tr.truth[j]; if (!d.includes(c)) miss++; ghost += d.filter((x) => x !== c).length; }
    return { miss, ghost };
  };
  /** True when the declared family keeps >= 90% for the rest of the window. */
  T.holdsLock = (tr) => {
    if (!tr.declared) return false;
    const k = T.idx(tr, tr.decision);
    return k >= 0 && tr.post.slice(tr.tau - 1).every((p) => p[k] >= 0.9);
  };
  const ALGEBRAIC = new Set(["lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed"]);
  /** Is a decision label a claimed algebraic break (the dangerous error when wrong)? */
  T.isAlgebraic = (id) => ALGEBRAIC.has(id);

  // ============================================================= scoreboard
  const median = (OTS.median = (xs) => {
    if (!xs || !xs.length) return null;
    const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  });
  const S = (OTS.S = {});
  /** Raw scoreboard rows, optionally for one preset id ('all' = every row). */
  S.rows = (preset = "all") => (DATA ? DATA.scoreboard.rows.filter((r) => preset === "all" || r.preset === preset) : []);
  function agg(rows) {
    const n = rows.reduce((s, r) => s + r.n, 0) || 1;
    const w = (f) => rows.reduce((s, r) => s + (r[f] == null ? 0 : r[f] * r.n), 0) / n;
    const has = (f) => rows.some((r) => r[f] != null);
    const wn = (f) => (has(f) ? w(f) : null);   // null (not 0) when a field is absent in older data
    const lockedRows = rows.filter((r) => r.ots_top1_locked != null);
    const nl = lockedRows.reduce((s, r) => s + r.n, 0);
    const hops = rows.flatMap((r) => r.ots_hops);
    return {
      n: rows.reduce((s, r) => s + r.n, 0),
      ots_acc: w("ots_acc"), base_acc: w("base_acc"),
      ots_fb: w("ots_false_break"), base_fb: w("base_false_break"),
      ots_unknown: w("ots_unknown"), ots_locked: w("ots_locked"),
      top1: w("ots_top1"), markov_top1: w("markov_top1"),
      // the Fixed-Share predictor (the system's actual next-channel output)
      pred_top1: wn("pred_top1"), pred_top1_steady: wn("pred_top1_steady"),
      top1_locked: nl ? lockedRows.reduce((s, r) => s + r.ots_top1_locked * r.n, 0) / nl : null,
      hops, hops_med: median(hops),
    };
  }
  /** Per-emitter aggregate (weighted by runs) in emitter order:
   *  [{id, name, kind, n, ots_acc, base_acc, ots_fb, base_fb, ots_unknown, ots_locked,
   *    top1, top1_locked, markov_top1, hops[], hops_med}] */
  S.byEmitter = (preset = "all") => OTS.emitters.map((e) => {
    const rows = S.rows(preset).filter((r) => r.emitter === e.id);
    return rows.length ? { id: e.id, name: e.name, kind: e.kind, ...agg(rows) } : null;
  }).filter(Boolean);
  /** Per-preset aggregate across emitters: [{id, name, p_d, p_fa, ...agg}] */
  S.byPreset = () => OTS.presets.map((p) => ({ ...p, ...agg(S.rows(p.id)) }));
  /** One aggregate over everything matching the preset filter. */
  S.overall = (preset = "all") => agg(S.rows(preset));
  /** Aggregate over rows matching the preset filter AND a row predicate. */
  S.overallWhere = (preset = "all", pred = () => true) => agg(S.rows(preset).filter(pred));
  /** Is an emitter id a "new algorithm" the bank has no expert for? */
  S.isNewAlgo = (id) => !!(EMIT.get(id) && EMIT.get(id).new_algo);
  /** Chance of guessing a hop at random (1 / channels). */
  S.chance = () => (DATA ? DATA.scoreboard.chance_top1 : 1 / 16);

  // ================================================================ tooltip
  const tip = { node: null };
  /** Show the shared tooltip near a pointer event. content: node(s) or {title, rows:[{value, label, color, kind}]} */
  OTS.tip = {
    show(ev, content) {
      if (!tip.node) { tip.node = OTS.el("div", { class: "tip", role: "tooltip" }); document.body.append(tip.node); }
      const n = OTS.clear(tip.node);
      if (content && content.rows) {
        if (content.title) n.append(OTS.el("div", { class: "tt" }, content.title));
        for (const r of content.rows) {
          n.append(OTS.el("div", { class: "tr" },
            r.color ? OTS.el("i", { class: r.kind === "square" ? "sq" : "lk", style: `background:${r.color}` }) : null,
            OTS.el("b", {}, r.value), OTS.el("span", {}, r.label)));
        }
      } else n.append(content);
      n.classList.add("show");
      const w = n.offsetWidth, h = n.offsetHeight;
      let x = ev.clientX + 16, y = ev.clientY + 14;
      if (x + w > window.innerWidth - 8) x = ev.clientX - w - 16;
      if (y + h > window.innerHeight - 8) y = ev.clientY - h - 14;
      n.style.left = Math.max(8, x) + "px"; n.style.top = Math.max(8, y) + "px";
    },
    hide() { if (tip.node) tip.node.classList.remove("show"); },
  };

  // ============================================================== chart kit
  const chart = (OTS.chart = {});
  /** Legend row. items: [{label, color, kind: 'line'|'dash'|'square'|'dot'|'ring'|'hollow'|'x'}] */
  chart.legend = (items) => OTS.el("div", { class: "legend" }, items.map((it) => OTS.el("span", {},
    it.kind === "square" ? OTS.el("i", { class: "sq", style: `background:${it.color}` })
      : it.kind === "dot" ? OTS.el("i", { class: "dotk", style: `background:${it.color}` })
      : it.kind === "ring" ? OTS.el("i", { class: "ringk" })
      : it.kind === "hollow" ? OTS.el("i", { class: "hollowk" })
      : it.kind === "x" ? OTS.el("i", { class: "xk" }, "×")
      : it.kind === "dash" ? OTS.el("i", { class: "lk dash" })
      : OTS.el("i", { class: "lk", style: `background:${it.color}` }),
    it.label)));

  function niceTicks(min, max, count = 5) {
    const span = max - min || 1, step0 = span / count, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
    const out = []; for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function measureText(txt, font) {
    const c = measureText.c || (measureText.c = document.createElement("canvas").getContext("2d"));
    c.font = font; return c.measureText(txt).width;
  }
  function mountChart(container, spec, render) {
    container.classList.add("chart");
    let cur = spec;
    const draw = () => { const w = container.clientWidth; if (w > 0) render(container, cur, w); };
    const ro = new ResizeObserver(() => draw());
    ro.observe(container);
    draw();
    return {
      update(next) { cur = next; draw(); },
      destroy() { ro.disconnect(); OTS.clear(container); },
    };
  }

  /** Line chart with labelled axes, direct end labels, reference lines and a crosshair tooltip.
   *  refLines labels sit inside the plot just above the line (set below: true to put them under it).
   *  spec = {
   *    height=280, ariaLabel,
   *    x: {label, min, max, ticks?: number[], format?: v => string},
   *    y: {label, min, max, ticks?, format?, log?: bool},
   *    series: [{id, label, color, points: [[x, y], ...], width?, dash?: '6 4', opacity?, endLabel?: bool (default true), marker?: 'end'|'all'}],
   *    refLines: [{y, label}], vLines: [{x, label}],
   *    endLabels: max number of direct end labels (default 4, highest last values first),
   *    tooltip: true (default: every series at the hovered x) | false | (x) => {title, rows},
   *    tooltipFormat: v => string (default y.format), emptyText
   *  }  -> {update(spec), destroy()} */
  chart.line = (container, spec) => mountChart(container, spec, renderLine);
  function renderLine(container, spec, W) {
    const H = spec.height || 280, svg = OTS.svg;
    const sx = spec.x, sy = spec.y, log = !!sy.log;
    const yfmt = sy.format || ((v) => String(v)), xfmt = sx.format || ((v) => String(v));
    const yTicks = sy.ticks || (log ? logTicks(sy.min, sy.max) : niceTicks(sy.min, sy.max));
    const xTicks = sx.ticks || niceTicks(sx.min, sx.max, Math.max(3, Math.floor(W / 110)));
    const tickW = Math.max(...yTicks.map((v) => measureText(yfmt(v), "11px " + OTS.tok("mono"))), 10);
    const series = (spec.series || []).filter((s) => s.points && s.points.length);
    const nEnd = spec.endLabels == null ? 4 : spec.endLabels;
    const endCands = series.filter((s) => s.endLabel !== false)
      .sort((a, b) => b.points[b.points.length - 1][1] - a.points[a.points.length - 1][1]).slice(0, nEnd);
    const endW = endCands.length ? Math.min(160, Math.max(...endCands.map((s) => measureText(s.label, "600 12px " + OTS.tok("sans")))) + 18) : 0;
    const m = { l: tickW + 34, r: Math.max(16, endW), t: 14, b: 50 };
    const iw = Math.max(40, W - m.l - m.r), ih = H - m.t - m.b;
    const ymap = log
      ? (v) => m.t + ih - (Math.log10(Math.max(v, sy.min)) - Math.log10(sy.min)) / (Math.log10(sy.max) - Math.log10(sy.min)) * ih
      : (v) => m.t + ih - (v - sy.min) / (sy.max - sy.min || 1) * ih;
    const xmap = (v) => m.l + (v - sx.min) / (sx.max - sx.min || 1) * iw;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": spec.ariaLabel || spec.y.label || "chart" });
    for (const v of yTicks) {
      root.append(svg("line", { class: "grid-l", x1: m.l, x2: m.l + iw, y1: ymap(v), y2: ymap(v) }));
      root.append(svg("text", { class: "ax-t", x: m.l - 8, y: ymap(v) + 4, "text-anchor": "end" }, yfmt(v)));
    }
    root.append(svg("line", { class: "ax", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));
    for (const v of xTicks) {
      root.append(svg("line", { class: "ax", x1: xmap(v), x2: xmap(v), y1: m.t + ih, y2: m.t + ih + 5 }));
      root.append(svg("text", { class: "ax-t", x: xmap(v), y: m.t + ih + 18, "text-anchor": "middle" }, xfmt(v)));
    }
    root.append(svg("text", { class: "ax-title", x: m.l + iw / 2, y: H - 6, "text-anchor": "middle" }, sx.label || ""));
    root.append(svg("text", { class: "ax-title", x: 0, y: 0, "text-anchor": "middle",
      transform: `translate(14 ${m.t + ih / 2}) rotate(-90)` }, sy.label || ""));
    for (const r of spec.refLines || []) {
      root.append(svg("line", { class: "ref", x1: m.l, x2: m.l + iw, y1: ymap(r.y), y2: ymap(r.y) }));
      if (r.label) root.append(svg("text", { class: "ref-t", x: m.l + 8, y: ymap(r.y) + (r.below ? 15 : -6) }, r.label));
    }
    for (const v of spec.vLines || []) {
      if (v.x < sx.min || v.x > sx.max) continue;
      root.append(svg("line", { class: "v-line", x1: xmap(v.x), x2: xmap(v.x), y1: m.t, y2: m.t + ih, style: v.color ? `stroke:${v.color}` : null }));
      if (v.label) root.append(svg("text", { class: "v-t", x: xmap(v.x) + 6, y: m.t + 12 }, v.label));
    }
    const clipId = "clip" + Math.random().toString(36).slice(2, 8);
    root.append(svg("clipPath", { id: clipId }, svg("rect", { x: m.l - 6, y: m.t - 6, width: iw + 12, height: ih + 12 })));
    const g = svg("g", { "clip-path": `url(#${clipId})` }); root.append(g);
    const order = [...series].sort((a, b) => (a.emphasis ? 1 : 0) - (b.emphasis ? 1 : 0));
    for (const s of order) {
      let d = "";
      s.points.forEach(([x, y], i) => { d += (i ? "L" : "M") + xmap(x).toFixed(1) + " " + ymap(y).toFixed(1); });
      g.append(svg("path", { class: "s-line", d, stroke: s.color, "stroke-width": s.width || 2,
        "stroke-dasharray": s.dash || null, "stroke-opacity": s.opacity == null ? 1 : s.opacity }));
      const marks = s.marker === "all" ? s.points : s.marker === "end" ? [s.points[s.points.length - 1]] : [];
      for (const [x, y] of marks) g.append(svg("circle", { cx: xmap(x), cy: ymap(y), r: 4.5, fill: s.color, stroke: OTS.tok("panel"), "stroke-width": 2 }));
    }
    // direct end labels with simple collision avoidance + leader lines
    const labs = endCands.map((s) => { const [x, y] = s.points[s.points.length - 1]; return { s, x: xmap(x), y0: ymap(y), y: ymap(y) }; })
      .sort((a, b) => a.y0 - b.y0);
    for (let i = 1; i < labs.length; i++) if (labs[i].y - labs[i - 1].y < 15) labs[i].y = labs[i - 1].y + 15;
    if (labs.length && labs[labs.length - 1].y > m.t + ih + 4) {
      labs[labs.length - 1].y = m.t + ih + 4;   // pull an overflowing stack back up, moving only what must move
      for (let i = labs.length - 2; i >= 0; i--) labs[i].y = Math.min(labs[i].y, labs[i + 1].y - 15);
    }
    for (const l of labs) {
      const lx = Math.min(l.x + 8, m.l + iw + 8);
      if (Math.abs(l.y - l.y0) > 3) root.append(svg("line", { class: "s-lead", x1: l.x + 2, y1: l.y0, x2: lx - 2, y2: l.y }));
      root.append(svg("text", { class: "s-label", x: lx, y: l.y + 4 }, l.s.label));
    }
    if (!series.length && spec.emptyText) root.append(svg("text", { class: "empty-t", x: m.l + iw / 2, y: m.t + ih / 2, "text-anchor": "middle" }, spec.emptyText));
    // crosshair tooltip
    if (spec.tooltip !== false && series.length) {
      const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
      const cross = svg("line", { class: "cross", y1: m.t, y2: m.t + ih, visibility: "hidden" });
      const hit = svg("rect", { class: "hit", x: m.l, y: m.t, width: iw, height: ih });
      root.append(cross, hit);
      const tfmt = spec.tooltipFormat || yfmt;
      hit.addEventListener("pointermove", (ev) => {
        const box = root.getBoundingClientRect(), px = (ev.clientX - box.left) * (W / box.width);
        const xv = sx.min + (px - m.l) / iw * (sx.max - sx.min);
        let best = xs[0]; for (const v of xs) if (Math.abs(v - xv) < Math.abs(best - xv)) best = v;
        cross.setAttribute("x1", xmap(best)); cross.setAttribute("x2", xmap(best)); cross.setAttribute("visibility", "visible");
        const content = typeof spec.tooltip === "function" ? spec.tooltip(best) : {
          title: (sx.label ? sx.label + ": " : "") + xfmt(best),
          rows: series.map((s) => { const p = s.points.find((q) => q[0] === best); return p ? { value: tfmt(p[1]), label: s.label, color: s.color, y: p[1] } : null; })
            .filter(Boolean).sort((a, b) => b.y - a.y),
        };
        OTS.tip.show(ev, content);
      });
      hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); OTS.tip.hide(); });
    }
    OTS.clear(container).append(root);
  }
  function logTicks(min, max) {
    const out = []; for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) { const v = Math.pow(10, e); if (v >= min && v <= max) out.push(v); }
    return out;
  }

  /** Grouped column chart with labelled axes and value labels on the caps.
   *  spec = {
   *    height=300, ariaLabel,
   *    categories: [{id, label, sub?}],
   *    series: [{id, label, color}],
   *    value: (catId, seriesId) => number | null,
   *    y: {label, min=0, max, ticks?, format?}, x: {label},
   *    barWidth=16 (capped at 24), valueLabels=true, valueFormat?: v => string,
   *    refLines: [{y, label}], tooltip: true | (cat) => {title, rows}
   *  }  -> {update(spec), destroy()} */
  chart.columns = (container, spec) => mountChart(container, spec, renderColumns);
  function renderColumns(container, spec, W) {
    const H = spec.height || 300, svg = OTS.svg, sy = spec.y, ymin = sy.min || 0;
    const yfmt = sy.format || ((v) => String(v)), vfmt = spec.valueFormat || yfmt;
    const yTicks = sy.ticks || niceTicks(ymin, sy.max);
    const tickW = Math.max(...yTicks.map((v) => measureText(yfmt(v), "11px " + OTS.tok("mono"))), 10);
    const cats = spec.categories, ser = spec.series;
    const m = { l: tickW + 34, r: 12, t: 22, b: 0 };
    const iw = Math.max(60, W - m.l - m.r), band = iw / cats.length;
    const bw = Math.min(24, spec.barWidth || 16, Math.max(4, (band - 10) / ser.length - 2));
    const rotate = band < 74;
    const catFont = "500 12px " + OTS.tok("sans");
    const longest = Math.max(...cats.map((c) => measureText(c.label, catFont)));
    m.b = rotate ? Math.min(140, longest * 0.72 + 44) : (cats.some((c) => c.sub) ? 66 : 50);
    const ih = H - m.t - m.b;
    const ymap = (v) => m.t + ih - (Math.min(Math.max(v, ymin), sy.max) - ymin) / (sy.max - ymin || 1) * ih;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": spec.ariaLabel || sy.label || "chart" });
    for (const v of yTicks) {
      root.append(svg("line", { class: "grid-l", x1: m.l, x2: m.l + iw, y1: ymap(v), y2: ymap(v) }));
      root.append(svg("text", { class: "ax-t", x: m.l - 8, y: ymap(v) + 4, "text-anchor": "end" }, yfmt(v)));
    }
    root.append(svg("line", { class: "ax", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));
    root.append(svg("text", { class: "ax-title", x: 0, y: 0, "text-anchor": "middle", transform: `translate(14 ${m.t + ih / 2}) rotate(-90)` }, sy.label || ""));
    if (spec.x && spec.x.label) root.append(svg("text", { class: "ax-title", x: m.l + iw / 2, y: H - 6, "text-anchor": "middle" }, spec.x.label));
    const groupW = ser.length * bw + (ser.length - 1) * 3;
    cats.forEach((c, i) => {
      const cx = m.l + band * (i + 0.5), x0 = cx - groupW / 2;
      const g = svg("g", { class: "colgrp" });
      ser.forEach((s, k) => {
        const v = spec.value(c.id, s.id);
        if (v == null || isNaN(v)) return;
        const x = x0 + k * (bw + 3), y = ymap(v), h = Math.max(1.5, m.t + ih - y), r = Math.min(4, bw / 2, h);
        const d = `M${x} ${m.t + ih} V${m.t + ih - h + r} Q${x} ${m.t + ih - h} ${x + r} ${m.t + ih - h} H${x + bw - r} Q${x + bw} ${m.t + ih - h} ${x + bw} ${m.t + ih - h + r} V${m.t + ih} Z`;
        g.append(svg("path", { class: "col", d, fill: s.color }));
        if (spec.valueLabels !== false) g.append(svg("text", { class: "col-v", x: x + bw / 2, y: m.t + ih - h - 5, "text-anchor": "middle" }, vfmt(v)));
      });
      if (rotate) {
        g.append(svg("text", { class: "cat-t", x: cx, y: m.t + ih + 14, "text-anchor": "end", transform: `rotate(-35 ${cx} ${m.t + ih + 14})` }, c.label));
      } else {
        g.append(svg("text", { class: "cat-t", x: cx, y: m.t + ih + 18, "text-anchor": "middle" }, c.label));
        if (c.sub) g.append(svg("text", { class: "cat-s", x: cx, y: m.t + ih + 33, "text-anchor": "middle" }, c.sub));
      }
      const hit = svg("rect", { class: "hit", x: m.l + band * i, y: m.t, width: band, height: ih + 6 });
      g.append(hit);
      if (spec.tooltip !== false) {
        hit.addEventListener("pointermove", (ev) => {
          g.querySelectorAll(".col").forEach((n) => n.classList.add("hov"));
          OTS.tip.show(ev, typeof spec.tooltip === "function" ? spec.tooltip(c) : {
            title: c.label + (c.sub ? " · " + c.sub : ""),
            rows: ser.map((s) => { const v = spec.value(c.id, s.id); return v == null ? null : { value: vfmt(v), label: s.label, color: s.color, kind: "square" }; }).filter(Boolean),
          });
        });
        hit.addEventListener("pointerleave", () => { g.querySelectorAll(".col").forEach((n) => n.classList.remove("hov")); OTS.tip.hide(); });
      }
      root.append(g);
    });
    for (const r of spec.refLines || []) {
      root.append(svg("line", { class: "ref", x1: m.l, x2: m.l + iw, y1: ymap(r.y), y2: ymap(r.y) }));
      if (r.label) root.append(svg("text", { class: "ref-t", x: m.l + iw - 4, y: ymap(r.y) - 6, "text-anchor": "end" }, r.label));
    }
    OTS.clear(container).append(root);
  }

  // ================================================================ prefs
  const PREF_KEY = "ots.prefs.v1";
  OTS.prefs = { truth: false, path: false, tables: false, speed: 1 };
  try { Object.assign(OTS.prefs, JSON.parse(localStorage.getItem(PREF_KEY) || "{}")); } catch (_) { /* storage unavailable */ }
  /** Change a display preference ('truth' sensor mistakes, 'path' hidden path, 'tables', 'speed'). */
  OTS.setPref = (k, v) => {
    OTS.prefs[k] = v;
    try { localStorage.setItem(PREF_KEY, JSON.stringify(OTS.prefs)); } catch (_) { /* ignore */ }
    OTS.emit("prefs", OTS.prefs);
  };

  // ================================================================ player
  /** Shared playback of one engagement for the Live and Analysis views.
   *  player.trace, player.t (hops shown, 0..window), player.playing; methods below. */
  const P = (OTS.player = { trace: null, t: 0, playing: false, holdUntil: 0, acc: 0, last: 0 });
  P.load = (tr) => {
    P.trace = tr; P.t = 0; P.acc = 0; P.holdUntil = 0;
    OTS.emit("load", tr); OTS.emit("tick", { t: 0, prev: 0, animate: false });
  };
  P.play = () => {
    if (!P.trace) return;
    if (P.t >= P.trace.window) P.seek(0);
    P.playing = true; P.last = performance.now(); OTS.emit("state", { playing: true });
  };
  P.pause = () => { if (P.playing) { P.playing = false; OTS.emit("state", { playing: false }); } };
  P.toggle = () => (P.playing ? P.pause() : P.play());
  /** Jump to hop t (0..window). Crossing the lock forward fires 'lock'. */
  P.seek = (t, animate = false) => {
    const tr = P.trace; if (!tr) return;
    const prev = P.t; P.t = Math.max(0, Math.min(tr.window, t));
    if (P.t === prev) return;
    OTS.emit("tick", { t: P.t, prev, animate: animate && P.t === prev + 1 });
    if (tr.declared && prev < tr.tau && P.t >= tr.tau) OTS.emit("lock", tr);
    if (P.t >= tr.window && prev < tr.window) OTS.emit("end", tr);
  };
  P.step = (d) => P.seek(P.t + d, d > 0);
  P.restart = () => { P.pause(); P.seek(0); };
  /** Auto pace: slow while searching and around the lock, fast while tracking. */
  P.intervalMs = () => {
    const tr = P.trace; if (!tr) return 400;
    const searching = tr.declared ? P.t < tr.tau : P.t < 16;
    const nearLock = tr.declared && P.t >= tr.tau - 1 && P.t < tr.tau + 3;
    return (searching || nearLock ? 560 : 170) / (OTS.prefs.speed || 1);
  };
  function loop(now) {
    const dt = Math.min(250, now - (P.last || now)); P.last = now;
    if (P.playing && P.trace && now >= P.holdUntil) {
      P.acc += dt;
      if (P.acc >= P.intervalMs()) {
        P.acc = 0;
        P.seek(P.t + 1, true);
        if (P.trace.declared && P.t === P.trace.tau) P.holdUntil = now + 1800 / (OTS.prefs.speed || 1);
        if (P.t >= P.trace.window) P.pause();
      }
    }
    if (tour.active && tour.nextAt && now >= tour.nextAt) { tour.nextAt = 0; tour.next(); }
    OTS.emit("frame", now);
    requestAnimationFrame(loop);
  }

  // ======================================================= scenario + live
  const SC = (OTS.scenario = { emitter: "lfsr", preset: "typical", run: 0, custom: { p_d: 0.8, p_fa: 0.02 } });
  OTS.live = false;
  /** Load a recorded engagement (emitter id, preset id, run index; wraps around). */
  OTS.select = (emitter, preset, run = 0) => {
    SC.emitter = emitter; SC.preset = preset;
    const list = OTS.tracesFor(emitter, preset);
    if (!list.length) { if (OTS.live) return OTS.runLive(); return; }
    SC.run = ((run % list.length) + list.length) % list.length;
    P.pause(); P.load(list[SC.run]); renderDrawer();
  };
  /** Next/previous recorded run for the current emitter + preset (live mode: a fresh run). */
  OTS.nextRun = (d = 1) => {
    if (OTS.live && (SC.preset === "custom" || !OTS.tracesFor(SC.emitter, SC.preset).length)) return OTS.runLive();
    OTS.select(SC.emitter, SC.preset, SC.run + d);
  };
  OTS.loadingLive = false;
  /** Run a brand-new engagement through the Python bank (live server only). */
  OTS.runLive = async () => {
    if (!OTS.live || OTS.loadingLive) return;
    const pr = SC.preset === "custom" ? SC.custom : OTS.preset(SC.preset) || SC.custom;
    const seed = Math.floor(Math.random() * 2147483647);
    OTS.loadingLive = true; P.pause(); OTS.emit("liveloading", true); renderDrawer();
    try {
      const r = await fetch(`api/run?emitter=${encodeURIComponent(SC.emitter)}&p_d=${pr.p_d}&p_fa=${pr.p_fa}&seed=${seed}`, { cache: "no-store" });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.statusText);
      OTS.loadingLive = false; OTS.emit("liveloading", false);
      P.load(j); renderDrawer(); P.play();
    } catch (err) {
      OTS.loadingLive = false; OTS.emit("liveloading", false); renderDrawer();
      console.error("[OTS] live run failed", err);
      alert("Live run failed: " + err.message);
    }
  };
  /** Index of the clearest recorded run for a tour stop. */
  OTS.bestRun = (emitter, preset) => {
    const list = OTS.tracesFor(emitter, preset);
    const tests = emitter === "csprng"
      ? [(t) => t.correct && !t.baseline.correct && t.baseline.declared, (t) => t.correct]
      : [(t) => t.correct && t.declared && T.holdsLock(t), (t) => t.correct && t.declared, (t) => t.correct];
    for (const ok of tests) { const i = list.findIndex(ok); if (i >= 0) return i; }
    return 0;
  };

  // ================================================================== tour
  const tour = (OTS.tour = { active: false, i: -1, nextAt: 0 });
  /** Guided demo: plays DATA.tour scenarios in the Live view, one after another. */
  tour.start = () => { tour.active = true; tour.i = -1; if (route.current !== "live") location.hash = "#/live"; tour.next(); renderDrawer(); };
  tour.stop = () => { if (!tour.active) return; tour.active = false; tour.nextAt = 0; renderDrawer(); OTS.emit("tour", false); };
  tour.next = () => {
    const stops = (DATA && DATA.tour) || []; if (!stops.length) return;
    tour.i = (tour.i + 1) % stops.length;
    const s = stops[tour.i];
    OTS.select(s.emitter, s.preset, OTS.bestRun(s.emitter, s.preset));
    OTS.emit("tour", true);
    setTimeout(() => { if (tour.active) P.play(); }, 1500);
  };
  OTS.on("end", () => { if (tour.active) tour.nextAt = performance.now() + 5500; });

  // ================================================================ views
  const VIEWS = [];
  const route = { current: null, cleanup: null };
  /** Register a page. view = {
   *    id, title, subtitle, nav (short label), hint (one-line nav hint), icon (SVG path 'd' string, 24x24),
   *    usesPlayer: bool (shows transport + settings drawer), mount(root) -> cleanup function }  */
  OTS.views = {
    register(v) { VIEWS.push(v); },
    list: () => VIEWS,
    go: (id) => { location.hash = "#/" + id; },
  };
  function navTo() {
    const id = (location.hash.match(/^#\/([\w-]+)/) || [])[1];
    const v = VIEWS.find((x) => x.id === id) || VIEWS[0];
    if (!v) return;
    if (route.current === v.id && route.cleanup !== null) return;
    if (route.cleanup) { try { route.cleanup(); } catch (e) { console.error(e); } }
    OTS.tip.hide();
    route.current = v.id;
    for (const a of document.querySelectorAll(".nav a")) a.classList.toggle("on", a.dataset.id === v.id);
    OTS.$("#vTitle").textContent = v.title;
    OTS.$("#vSub").textContent = v.subtitle || "";
    document.title = "OTS · " + v.nav;
    const shell = OTS.$(".shell");
    shell.classList.toggle("has-drawer", !!v.usesPlayer);
    OTS.$("#transport").hidden = !v.usesPlayer;
    OTS.$("#settingsBtn").hidden = !v.usesPlayer;
    if (v.usesPlayer && !P.trace) OTS.select(SC.emitter, SC.preset, OTS.bestRun(SC.emitter, SC.preset));
    if (!v.usesPlayer) P.pause();
    if (v.id !== "live" && v.id !== "analysis") tour.stop();
    const root = OTS.clear(OTS.$("#view"));
    window.scrollTo(0, 0);
    if (!DATA) {
      root.append(OTS.el("div", { class: "card empty" }, "No recorded engagements found. From the repository root run ",
        OTS.el("code", {}, "python -m dashboard.build"), " and reload, or start ", OTS.el("code", {}, "python -m dashboard.serve"), "."));
      route.cleanup = () => {};
      return;
    }
    try { route.cleanup = v.mount(root) || (() => {}); }
    catch (err) { console.error(err); root.append(OTS.el("div", { class: "card empty" }, "This page failed to load: " + err.message)); route.cleanup = () => {}; }
    OTS.emit("route", v.id);
    renderTransport();
  }

  // ============================================================== chrome
  const ICON = (d) => OTS.svg("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 1.8, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" }, OTS.svg("path", { d }));
  function buildNav() {
    const nav = OTS.clear(OTS.$("#nav"));
    VIEWS.forEach((v, i) => nav.append(OTS.el("a", { href: "#/" + v.id, "data-id": v.id, title: `${v.title} (${i + 1})` },
      ICON(v.icon || "M4 12h16"), OTS.el("b", {}, v.nav), OTS.el("small", {}, v.hint || ""))));
  }
  function renderTransport() {
    const tr = P.trace; if (!tr) return;
    const play = OTS.$("#playBtn");
    play.textContent = P.playing ? "❚❚ Pause" : (P.t >= tr.window ? "▶ Replay" : "▶ Play");
    OTS.clear(OTS.$("#hopClock")).append("HOP ", OTS.el("b", { class: "num" }, String(P.t).padStart(3, "0")), "/" + tr.window);
    OTS.$("#scenChip").textContent = `${OTS.emitName(tr.emitter)} · ${OTS.preset(tr.preset)?.name || "live"} sensor`;
  }
  function section(label, ...kids) { return OTS.el("div", { class: "dsec" }, OTS.el("span", { class: "dlabel" }, label), kids); }
  function renderDrawer() {
    const d = OTS.$("#drawerBody"); if (!d || !DATA) return;
    OTS.clear(d);
    const tr = P.trace;
    d.append(section("Hidden radio pattern",
      OTS.el("div", { class: "opt-grid" }, OTS.emitters.map((e) => OTS.el("button", {
        class: "opt" + (SC.emitter === e.id ? " sel" : ""), title: e.blurb,
        onclick: () => {
          tour.stop();
          if (SC.preset === "custom") { SC.emitter = e.id; renderDrawer(); OTS.runLive(); }
          else OTS.select(e.id, SC.preset, OTS.bestRun(e.id, SC.preset));
        },
      }, OTS.el("b", {}, e.name), OTS.el("small", {}, e.id === "csprng" ? "control: unbreakable"
        : e.new_algo ? "new · no expert in bank" : e.kind))))));
    if (OTS.scenarios && OTS.scenarios.length) {
      d.append(section("The radio changes recipe (adaptive)",
        OTS.el("div", { class: "opt-grid" }, OTS.scenarios.map((s) => OTS.el("button", {
          class: "opt" + (SC.emitter === s.id ? " sel" : ""), title: s.blurb,
          onclick: () => {
            tour.stop();
            const pr = OTS.tracesFor(s.id, SC.preset).length ? SC.preset
              : (OTS.tracesFor(s.id, "clear").length ? "clear" : "typical");
            OTS.select(s.id, pr, OTS.bestRun(s.id, pr));
          },
        }, OTS.el("b", {}, s.name), OTS.el("small", {}, "switches mid-engagement"))))));
    }
    const presetBtns = OTS.presets.map((p) => OTS.el("button", {
      class: "btn" + (SC.preset === p.id ? " sel" : ""),
      onclick: () => { tour.stop(); OTS.select(SC.emitter, p.id, OTS.bestRun(SC.emitter, p.id)); },
      title: `catches ${Math.round(p.p_d * 100)}% of hops, false-alarm rate ${p.p_fa} per channel`,
    }, p.name));
    if (OTS.live) presetBtns.push(OTS.el("button", { class: "btn" + (SC.preset === "custom" ? " sel" : ""),
      onclick: () => { tour.stop(); SC.preset = "custom"; renderDrawer(); OTS.runLive(); } }, "Custom"));
    const cur = SC.preset === "custom" ? SC.custom : OTS.preset(SC.preset);
    d.append(section("Sensor quality", OTS.el("div", { class: "seg" }, presetBtns),
      cur ? OTS.el("div", { class: "dhelp" }, `Catches ${Math.round(cur.p_d * 100)}% of hops and hears about ${(cur.p_fa * 15).toFixed(2)} ghost signals per hop.`) : null,
      SC.preset === "custom" ? OTS.el("div", { class: "sliders" },
        slider("Pd", "p_d", 0.5, 1, 0.05), slider("Pfa", "p_fa", 0, 0.1, 0.01)) : null));
    const list = OTS.tracesFor(SC.emitter, SC.preset);
    const isLive = tr && tr.preset === "live";
    d.append(section("Engagement",
      OTS.el("div", { class: "row", style: "gap:6px" },
        OTS.el("button", { class: "btn icon", "aria-label": "Previous run", disabled: !list.length && !OTS.live, onclick: () => { tour.stop(); OTS.nextRun(-1); } }, "‹"),
        OTS.el("span", { class: "mono small ink2", style: "min-width:128px;text-align:center" },
          isLive ? `live · seed ${tr.seed}` : list.length ? `run ${SC.run + 1} of ${list.length} · seed ${list[SC.run]?.seed}` : "—"),
        OTS.el("button", { class: "btn icon", "aria-label": "Next run", disabled: !list.length && !OTS.live, onclick: () => { tour.stop(); OTS.nextRun(1); } }, "›")),
      OTS.live ? OTS.el("button", { class: "btn go", style: "margin-top:10px", disabled: OTS.loadingLive, onclick: () => { tour.stop(); OTS.runLive(); } },
        OTS.loadingLive ? [OTS.el("span", { class: "spin" }), " Running…"] : "⚡ Run a new live engagement") : null,
      OTS.el("div", { class: "dhelp" }, OTS.live ? "Live: new engagements are computed by the Python bank on demand."
        : ["Replay: pre-recorded real runs. Run ", OTS.el("code", { style: "white-space:nowrap" }, "python -m dashboard.serve"), " for live runs."])));
    d.append(section("Show on the chart",
      toggle("Reveal sensor mistakes (ghosts and missed hops)", "truth"),
      toggle("Show the hidden true path", "path"),
      OTS.el("div", { class: "dhelp" }, "OTS never sees these: they are drawn for the audience.")));
    d.append(section("Playback speed", OTS.el("div", { class: "seg" }, [0.5, 1, 2, 4].map((s) => OTS.el("button", {
      class: "btn" + (OTS.prefs.speed === s ? " sel" : ""), onclick: () => { OTS.setPref("speed", s); renderDrawer(); } }, s + "×")))));
    d.append(section("Guided demo",
      OTS.el("button", { class: "btn" + (tour.active ? " sel" : ""), onclick: () => (tour.active ? tour.stop() : tour.start()) }, tour.active ? "■ Stop guided demo" : "★ Start guided demo"),
      OTS.el("div", { class: "dhelp" }, "Plays five scenarios back to back with narration. Key: G")));
  }
  function slider(label, key, min, max, step) {
    const out = OTS.el("span", { class: "num" }, SC.custom[key].toFixed(2));
    const inp = OTS.el("input", { type: "range", min, max, step, value: SC.custom[key] });
    inp.addEventListener("input", () => { SC.custom[key] = +inp.value; out.textContent = (+inp.value).toFixed(2); });
    inp.addEventListener("change", () => OTS.runLive());
    return OTS.el("label", {}, label, inp, out);
  }
  function toggle(label, key) {
    const inp = OTS.el("input", { type: "checkbox" }); inp.checked = !!OTS.prefs[key];
    inp.addEventListener("change", () => OTS.setPref(key, inp.checked));
    return OTS.el("label", { class: "switch" }, inp, label);
  }
  OTS.renderDrawer = renderDrawer;
  OTS.openDrawer = (open) => {
    const s = OTS.$(".shell"); s.classList.toggle("drawer-open", open == null ? !s.classList.contains("drawer-open") : open);
    OTS.emit("resize");
  };

  // ============================================================ keyboard
  function onKey(e) {
    if ((e.target instanceof Element && e.target.matches("input, select, textarea")) || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase(), v = VIEWS.find((x) => x.id === route.current);
    const n = parseInt(k, 10);
    if (n >= 1 && n <= VIEWS.length) { OTS.views.go(VIEWS[n - 1].id); return; }
    if (k === "f") { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.(); return; }
    if (k === "g") { tour.active ? tour.stop() : tour.start(); return; }
    if (!v || !v.usesPlayer) return;
    if (k === " ") { e.preventDefault(); P.toggle(); }
    else if (k === "arrowright") { tour.stop(); P.pause(); P.step(1); }
    else if (k === "arrowleft") { tour.stop(); P.pause(); P.step(-1); }
    else if (k === "r") { tour.stop(); P.restart(); }
    else if (k === "n") { tour.stop(); OTS.nextRun(1); }
    else if (k === "t") { OTS.setPref("truth", !OTS.prefs.truth); renderDrawer(); }
    else if (k === "s") OTS.openDrawer();
  }

  // ================================================================ boot
  OTS.boot = () => {
    buildNav();
    OTS.$("#playBtn").addEventListener("click", () => P.toggle());
    OTS.$("#restartBtn").addEventListener("click", () => { tour.stop(); P.restart(); });
    OTS.$("#backBtn").addEventListener("click", () => { tour.stop(); P.pause(); P.step(-1); });
    OTS.$("#fwdBtn").addEventListener("click", () => { tour.stop(); P.pause(); P.step(1); });
    OTS.$("#settingsBtn").addEventListener("click", () => OTS.openDrawer());
    OTS.$("#drawerClose").addEventListener("click", () => OTS.openDrawer(false));
    OTS.$(".drawer-backdrop").addEventListener("click", () => OTS.openDrawer(false));
    for (const e of ["tick", "state", "load"]) OTS.on(e, renderTransport);
    OTS.on("load", renderDrawer);
    document.addEventListener("keydown", onKey);
    window.addEventListener("hashchange", navTo);
    if (window.innerWidth > 1280) OTS.$(".shell").classList.add("drawer-open");
    if (DATA) {
      const first = DATA.tour && DATA.tour[0] ? DATA.tour[0] : { emitter: "lfsr", preset: "typical" };
      SC.emitter = first.emitter; SC.preset = first.preset;
      OTS.$("#dataStamp").textContent = `Data built ${DATA.generated}`;
    }
    navTo();
    if (location.protocol.startsWith("http")) {
      fetch("api/ping", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((j) => {
        if (j && j.live) {
          OTS.live = true;
          const chip = OTS.$("#modeChip"); chip.classList.add("live"); OTS.$("#modeText").textContent = "Live engine";
          renderDrawer(); OTS.emit("live", true);
        }
      }).catch(() => {});
    }
    requestAnimationFrame(loop);
  };
  return OTS;
})();
