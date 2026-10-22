/* Story view: the 60-second landing page for a non-technical audience.
 *
 * A static page (no player), one idea per block:
 *   hero        the claim in one line, two plain sentences, three calls to action
 *   3 steps     the problem as three sketches drawn from one real recorded run
 *               (the radio's hops, what the sensor hears, what OTS calls)
 *   achieves    five headline tiles computed from the 600-run scoreboard (OTS.S),
 *               then one real run told start to finish and a "false claims out of
 *               100 runs" dot picture
 *   vs          OTS against the standard approach, every row backed by data
 *               (scoreboard, DATA.proof, recorded traces) plus the weak spot
 *   rigor       where every number comes from, with links to Live and Proof
 * Nothing here is typed in by hand: results come from OTS_DATA; the only fixed
 * numbers are facts about the method (16 channels, 95% lock, 8 hypotheses).
 */
(function () {
  "use strict";
  const { el, svg, fmt, S, T } = OTS;

  // ================================================================== styles
  const CSS = `
.story-page { gap: 20px; }
.story-hero { padding: 34px 36px 30px; overflow: hidden;
  background: radial-gradient(700px 300px at 100% 0%, rgba(34, 211, 238, 0.12), transparent 70%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.022), transparent 140px), var(--panel); }
.story-hero .lead { font-size: clamp(28px, 3vw, 44px); max-width: 30ch; margin-top: 12px; }
.story-hero .intro { font-size: 18px; max-width: 66ch; margin-top: 16px; }
.story-cta { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; }
.story-hero-foot { margin: 18px 0 0; font-size: 13px; color: var(--muted); }
.story-sec { margin: 22px 0 -4px; }
.story-sec h2 { margin: 0; font: 650 21px/1.3 var(--sans); letter-spacing: -0.005em; }
.story-sec p { margin: 4px 0 0; color: var(--ink-2); font-size: 14px; max-width: 82ch; }
.story-step { display: flex; flex-direction: column; }
.story-step-h { display: flex; align-items: center; gap: 12px; }
.story-num { flex: none; width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center;
  font: 700 14px/1 var(--mono); color: var(--accent-ink); border: 1px solid rgba(34, 211, 238, 0.55); background: var(--accent-soft); }
.story-step .card-k { margin: 0; font-size: 17px; }
.story-step .card-d { margin-top: 10px; }
.story-sketch { display: block; width: 100%; height: auto; margin: 14px 0 8px; }
.story-sketch .story-lane { stroke: var(--grid); stroke-width: 1; }
.story-sketch .story-ax { fill: var(--ink-2); font: 600 10.5px var(--sans); }
.story-sketch .story-tick { fill: var(--muted); font: 9.5px var(--mono); }
.story-sketch .story-note { fill: var(--ink-2); font: 600 10px var(--sans); }
.story-sketch .story-lock { stroke: var(--accent); stroke-width: 1.2; stroke-dasharray: 3 3; }
.story-sketch .story-lock-t { fill: var(--accent-ink); font: 700 9.5px var(--mono); letter-spacing: 0.04em; }
.story-step .legend { margin: 0; font-size: 12.5px; gap: 4px 14px; }
.story-step .ringk.story-ring-sm { width: 12px; height: 12px; }
.story-src { margin: 0; font-size: 12.5px; color: var(--muted); }
.story-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(196px, 1fr)); gap: 14px; }
.story-tiles .tile { display: flex; flex-direction: column; padding: 16px 16px 15px; }
.story-tiles .tile-v { color: var(--ink); }
.story-tiles .tile-s { line-height: 1.45; }
.story-tiles .tile-s b { color: var(--ink); font-weight: 600; }
.story-case ol { list-style: none; margin: 4px 0 0; padding: 0; display: flex; flex-direction: column; gap: 0; }
.story-case li { display: grid; grid-template-columns: 34px minmax(0, 1fr); gap: 0 12px; position: relative; padding-bottom: 14px; }
.story-case li:not(:last-child)::before { content: ""; position: absolute; left: 16px; top: 30px; bottom: 2px; width: 1px; background: var(--line-2); }
.story-case .story-ic svg { width: 17px; height: 17px; }
.story-case .story-ic { width: 32px; height: 32px; border-radius: 50%; display: grid; place-items: center; font: 700 13px/1 var(--mono);
  color: var(--ink-2); border: 1px solid var(--line-2); background: var(--surface); }
.story-case li.story-key .story-ic { color: var(--accent-ink); border-color: rgba(34, 211, 238, 0.55); background: var(--accent-soft); }
.story-case .story-when { font: 600 10.5px/1.3 var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-top: 1px; }
.story-case .story-what { font-size: 15px; color: var(--ink); margin-top: 2px; }
.story-case .story-what .mono { color: var(--accent-ink); }
.story-case .story-sub { font-size: 13px; color: var(--ink-2); margin-top: 2px; }
.story-case .status { margin-top: 6px; }
.story-card-row { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px 14px; margin-top: 4px; }
.story-card-row .card-foot { margin: 0; flex: 1; min-width: 200px; }
.story-dots-wrap { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px 26px; margin-top: 6px; }
.story-dots-col h3 { margin: 0 0 8px; font: 600 14px/1.3 var(--sans); color: var(--ink); }
.story-dots-col h3 small { display: block; font: 600 22px/1.25 var(--sans); margin-top: 2px; }
.story-dots-col h3 small span { font-size: 13px; color: var(--ink-2); font-weight: 500; }
.story-dots-col.story-us h3 small { color: var(--accent-ink); }
.story-dots { display: block; width: 100%; max-width: 230px; height: auto; }
.story-dots .story-fx { fill: #fff; font: 700 9px/1 var(--sans); }
.story-dot-key { display: inline-block; width: 12px; height: 12px; border-radius: 50%; }
.story-dot-key.story-bad { background: var(--critical); color: #fff; font: 700 9px/12px var(--sans); text-align: center; }
.story-vs { margin-top: 2px; }
.story-vs th, .story-vs td { text-align: left; vertical-align: top; padding: 12px 12px; }
.story-vs th { padding-top: 4px; }
.story-vs th.story-col-us { color: var(--accent-ink); }
.story-vs td:first-child { font-weight: 600; color: var(--ink); width: 26%; font-size: 14px; }
.story-vs td:first-child small { display: block; font-weight: 400; color: var(--muted); font-size: 12px; margin-top: 2px; }
.story-cell { display: grid; grid-template-columns: 22px minmax(0, 1fr); gap: 0 6px; }
.story-cell b { font-size: 15px; color: var(--ink); font-weight: 650; }
.story-cell span.story-det { grid-column: 2; font-size: 12.5px; color: var(--ink-2); line-height: 1.4; }
.story-mark { grid-row: span 2; font: 700 15px/1.25 var(--sans); }
.story-mark.story-yes { color: var(--good-ink); }
.story-mark.story-no { color: var(--critical-ink); }
.story-mark.story-part { color: var(--warning); }
.story-weak { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 8px 12px; margin-top: 16px; padding: 12px 14px;
  border: 1px solid var(--line); border-radius: 10px; background: var(--surface); font-size: 13.5px; color: var(--ink-2); }
.story-weak p { margin: 0; flex: 1; min-width: 220px; max-width: 100ch; }
.story-weak b { color: var(--ink); font-weight: 600; }
.story-rigor { display: flex; flex-wrap: wrap; align-items: center; gap: 14px 22px; padding: 20px 22px; margin-top: 6px; }
.story-rigor-ic { flex: none; width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center; color: var(--accent);
  border: 1px solid rgba(34, 211, 238, 0.45); background: var(--accent-soft); }
.story-rigor-ic svg { width: 24px; height: 24px; }
.story-rigor-t { flex: 1; min-width: 240px; }
.story-rigor-t p { margin: 0; font-size: 16px; color: var(--ink); max-width: 80ch; }
.story-rigor-t small { display: block; margin-top: 4px; color: var(--muted); font-size: 12.5px; }
.story-rigor-b { display: flex; flex-wrap: wrap; gap: 8px; }
@media (max-width: 860px) {
  .story-hero { padding: 22px 18px 20px; }
  .story-hero .intro { font-size: 16px; }
  .story-cta .btn { flex: 1 1 auto; justify-content: center; }
}
@media (max-width: 640px) {
  .story-rigor-ic { display: none; }
  .story-vs thead { display: none; }
  .story-vs tr { display: block; border-top: 1px solid var(--grid); padding: 10px 0 4px; }
  .story-vs td { display: block; border-top: 0; padding: 4px 0; width: auto !important; }
  .story-vs td[data-label]::before { content: attr(data-label); display: block; font: 600 10px/1.4 var(--mono);
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 2px; }
  .story-dots-wrap { gap: 16px; }
  .story-dots-col h3 small { font-size: 19px; }
}
`;
  function injectCss() {
    if (document.getElementById("css-story")) return;
    document.head.append(el("style", { id: "css-story" }, CSS));
  }

  // ============================================================ data helpers
  /** The engagement the sketches and the case file are drawn from: the clearest
   *  recorded LFSR run with the typical sensor (else the first guided-demo stop). */
  function exampleRun() {
    const cands = [{ emitter: "lfsr", preset: "typical" }, ...((OTS.data && OTS.data.tour) || [])];
    for (const c of cands) {
      const list = OTS.tracesFor(c.emitter, c.preset);
      if (!list.length) continue;
      const idx = OTS.bestRun(c.emitter, c.preset), tr = list[idx];
      if (tr && tr.declared && tr.correct) return { tr, idx, emitter: c.emitter, preset: c.preset };
    }
    return null;
  }
  /** Weighted mean of a scoreboard field over rows that have it (weights = runs). */
  function wmean(rows, f) {
    const ok = rows.filter((r) => r[f] != null), n = ok.reduce((s, r) => s + r.n, 0);
    return n ? ok.reduce((s, r) => s + r[f] * r.n, 0) / n : null;
  }
  /** "8.5×" style ratio b / a, or null when it is not meaningful. */
  function times(a, b) {
    if (!(a > 0) || !(b > 0)) return null;
    const r = b / a;
    return (r >= 10 ? String(Math.round(r)) : r.toFixed(1).replace(/\.0$/, "")) + "×";
  }
  const pctWord = (p) => fmt.pct0(p);
  /** Tiny 24x24 stroke icon from one or more path strings. */
  const icon = (...ds) => svg("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 1.9,
    "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" }, ds.map((d) => svg("path", { d })));
  const ICONS = {
    listen: ["M12 20v-8", "M8.5 8.5a5 5 0 0 1 7 0", "M5.6 5.6a9 9 0 0 1 12.8 0", "M12 12h.01"],
    lock: ["M6 11h12v9H6z", "M8.5 11V8a3.5 3.5 0 0 1 7 0v3"],
    key: ["M14.5 9.5m-4 0a4 4 0 1 0 8 0a4 4 0 1 0-8 0", "M11.6 12.4L4 20", "M6.5 17.5l2 2", "M8.5 15.5l1.5 1.5"],
    next: ["M4 12h14", "M13 6l6 6-6 6"],
  };
  /** "x of n" counts from a rate over n runs. */
  const ofRuns = (p, n) => `${Math.round(p * n)} of ${n}`;

  /** Everything the page shows, computed once per mount. */
  function facts() {
    const all = S.overall("all");
    const rows = S.rows("all");
    const emit = S.byEmitter("all");
    const cs = emit.find((r) => r.id === "csprng") || null;
    const worstPreset = [...OTS.presets].sort((a, b) => a.p_d - b.p_d)[0] || null;
    const harsh = worstPreset ? S.overall(worstPreset.id) : null;
    // next-hop calls after the lock, over the key-based (deterministic) patterns only
    const detRows = rows.filter((r) => T.isAlgebraic(r.emitter));
    const detTop1 = wmean(detRows, "ots_top1_locked");
    // secrets recovered in the recorded engagements of key-based patterns
    const keyTraces = (OTS.data.traces || []).filter((tr) => T.isAlgebraic(tr.emitter));
    const keyOk = keyTraces.filter((tr) => tr.secret_match === "exact" || tr.secret_match === "equivalent").length;
    // textbook Berlekamp-Massey vs OTS on a shift register with missed hops
    const bmRows = ((OTS.data.proof && OTS.data.proof.bm_vs_filtered && OTS.data.proof.bm_vs_filtered.rows) || []).filter((r) => r.p_d < 1);
    const bm = bmRows.length ? {
      bmMax: Math.max(...bmRows.map((r) => r.bm)),
      sfMin: Math.min(...bmRows.map((r) => r.state_filtered)),
      firstMiss: 1 - Math.max(...bmRows.map((r) => r.p_d)),
      maxMiss: 1 - Math.min(...bmRows.map((r) => r.p_d)),
      seeds: OTS.data.proof.bm_vs_filtered.n_seeds,
      noGhosts: !OTS.data.proof.bm_vs_filtered.p_fa,
    } : null;
    // the weakest pattern x sensor combination, shown rather than hidden
    const worst = rows.length ? rows.reduce((w, r) => (r.ots_acc < w.ots_acc ? r : w), rows[0]) : null;
    return { all, cs, worstPreset, harsh, detTop1, keyTraces, keyOk, bm, worst,
      chance: S.chance(), M: (OTS.data && OTS.data.M) || 16, nTraces: (OTS.data.traces || []).length };
  }

  // ================================================================ sketches
  /** One of the three "problem" sketches, drawn from the first hops of a real run.
   *  mode 'truth': the radio's real path · 'sensor': what the sensor reports, with
   *  misses and ghosts marked · 'ots': all detections look alike to OTS; rings are its
   *  calls for each hop after the lock, plus the next one. */
  function sketch(tr, mode, n) {
    const M = tr.M, VW = 320, VH = 176, L = 40, R = 30, TOP = 16, B = 34;   // R leaves room for the "next" label
    const iw = VW - L - R, ih = VH - TOP - B;
    const x = (j) => L + 10 + j * ((iw - 20) / n);          // columns 0..n (n = the "next" hop)
    const y = (c) => TOP + (M - 1 - c) * (ih / (M - 1));
    const label = mode === "truth" ? "Sketch of the radio's real channel for each of the first " + n + " hops"
      : mode === "sensor" ? "Sketch of what the sensor reported for the first " + n + " hops, with missed hops and ghost signals marked"
        : "Sketch of OTS's calls for each hop after it locked, including the next hop";
    const root = svg("svg", { viewBox: `0 0 ${VW} ${VH}`, class: "story-sketch", role: "img", "aria-label": label });
    for (let c = 0; c < M; c++) root.append(svg("line", { class: "story-lane", x1: L, x2: VW - R, y1: y(c), y2: y(c) }));
    // axes: channel ticks at both ends, titles on both axes
    root.append(svg("text", { class: "story-tick", x: L - 5, y: y(M - 1) + 3, "text-anchor": "end" }, fmt.ch(M - 1)));
    root.append(svg("text", { class: "story-tick", x: L - 5, y: y(0) + 3, "text-anchor": "end" }, fmt.ch(0)));
    root.append(svg("text", { class: "story-ax", x: 0, y: 0, "text-anchor": "middle", transform: `translate(12 ${TOP + ih / 2}) rotate(-90)` }, M + " channels"));
    root.append(svg("text", { class: "story-tick", x: x(0), y: TOP + ih + 14, "text-anchor": "middle" }, "1"));
    root.append(svg("text", { class: "story-tick", x: x(n - 1), y: TOP + ih + 14, "text-anchor": "middle" }, String(n)));
    root.append(svg("text", { class: "story-ax", x: L + iw / 2, y: VH - 4, "text-anchor": "middle" }, "Hop number (time) →"));
    const det = OTS.tok("det"), accent = OTS.tok("accent");
    const dot = (j, c, op) => svg("circle", { cx: x(j), cy: y(c), r: 3.4, fill: det, "fill-opacity": op == null ? 1 : op });

    if (mode === "truth") {
      let d = "";
      for (let j = 0; j < n; j++) d += (j ? "L" : "M") + x(j).toFixed(1) + " " + y(tr.truth[j]).toFixed(1);
      root.append(svg("path", { d, fill: "none", stroke: OTS.tok("muted"), "stroke-width": 1.5, "stroke-linejoin": "round" }));
      for (let j = 0; j < n; j++) root.append(dot(j, tr.truth[j]));
    } else if (mode === "sensor") {
      const marks = [], misses = [], ghosts = [];
      for (let j = 0; j < n; j++) {
        const c = tr.truth[j], ds = tr.dets[j];
        if (ds.includes(c)) root.append(dot(j, c));
        else {
          root.append(svg("circle", { cx: x(j), cy: y(c), r: 4.2, fill: "none", stroke: "rgba(200, 215, 235, 0.8)", "stroke-width": 1.3, "stroke-dasharray": "2 2" }));
          misses.push([x(j), y(c)]);
        }
        marks.push([x(j), y(c)]);
        for (const g of ds.filter((v) => v !== c)) {
          root.append(svg("path", { d: `M${x(j) - 3.4} ${y(g) - 3.4}L${x(j) + 3.4} ${y(g) + 3.4}M${x(j) + 3.4} ${y(g) - 3.4}L${x(j) - 3.4} ${y(g) + 3.4}`,
            stroke: OTS.tok("serious"), "stroke-width": 2, "stroke-linecap": "round" }));
          ghosts.push([x(j), y(g)]); marks.push([x(j), y(g)]);
        }
      }
      // one direct label per kind, placed beside the first mark whose label spot is free
      const taken = [];
      for (const [list, text] of [[misses, "missed"], [ghosts, "ghost"]]) {
        const w = text.length * 5.6 + 4;
        outer: for (const [mx, my] of list) for (const side of [1, -1]) {
          const x0 = side > 0 ? mx + 7 : mx - 7 - w, box = [x0 - 2, my - 7, x0 + w + 2, my + 7];
          if (box[0] < L || box[2] > VW - R) continue;
          const hit = marks.some(([px, py]) => px + 4 > box[0] && px - 4 < box[2] && py + 4 > box[1] && py - 4 < box[3])
            || taken.some((b) => b[0] < box[2] && b[2] > box[0] && b[1] < box[3] && b[3] > box[1]);
          if (hit) continue;
          taken.push(box);
          root.append(svg("text", { class: "story-note", x: side > 0 ? mx + 7 : mx - 7, y: my + 3.5, "text-anchor": side > 0 ? "start" : "end" }, text));
          break outer;
        }
      }
    } else {
      for (let j = 0; j < n; j++) for (const c of tr.dets[j]) root.append(dot(j, c, 0.32));
      const lockX = x(tr.tau - 1);
      root.append(svg("line", { class: "story-lock", x1: lockX, x2: lockX, y1: TOP - 6, y2: TOP + ih }));
      root.append(svg("text", { class: "story-lock-t", x: lockX - 4, y: TOP - 8 + 1, "text-anchor": "end" }, "LOCKED"));
      for (let j = tr.tau; j <= n; j++) {
        const f = T.forecastTop(tr, j); if (!f) continue;
        const next = j === n;
        root.append(svg("circle", { cx: x(j), cy: y(f.ch), r: next ? 6 : 5, fill: next ? "rgba(34, 211, 238, 0.18)" : "none", stroke: accent, "stroke-width": 2 }));
        if (next) root.append(svg("text", { class: "story-note", x: x(j) + 9, y: y(f.ch) + 3.5 }, "next"));
      }
    }
    return root;
  }
  /** 10 x 10 dot picture of 100 runs; `bad` of them marked as false claims. */
  function dots100(bad, faint, aria) {
    const root = svg("svg", { viewBox: "0 0 200 200", class: "story-dots", role: "img", "aria-label": aria });
    const crit = OTS.tok("critical");
    for (let i = 0; i < 100; i++) {
      const cx = 10 + (i % 10) * 20, cy = 10 + Math.floor(i / 10) * 20, isBad = i < bad;
      root.append(svg("circle", { cx, cy, r: 7.4, fill: isBad ? crit : faint, "fill-opacity": isBad ? 1 : 0.32 }));
      if (isBad) root.append(svg("text", { class: "story-fx", x: cx, y: cy + 3.2, "text-anchor": "middle" }, "✕"));
    }
    return root;
  }

  // ================================================================== blocks
  function hero(F) {
    return el("section", { class: "card accent story-hero" },
      el("p", { class: "card-t" }, "OTS · frequency-hop pattern identification"),
      el("h2", { class: "lead" }, "Cracking a hopping radio's secret pattern from noisy signals, ", el("em", {}, "in a handful of hops"), "."),
      el("p", { class: "intro" },
        `Some radios keep jumping between ${F.M} channels to avoid being followed, using a secret recipe. `,
        "OTS listens through an imperfect sensor, works out which recipe and which secret key are in use, then calls where the radio will jump next, ",
        "and it honestly says “unknown” when there is nothing to crack."),
      el("div", { class: "story-cta" },
        el("button", { class: "btn primary big", onclick: () => OTS.tour.start(), title: "Plays five scenarios with narration (G)" }, "▶ Start the guided demo"),
        el("button", { class: "btn big", onclick: () => OTS.views.go("challenge") }, "Try to beat it"),
        el("button", { class: "btn big", onclick: () => OTS.views.go("proof") }, "See the proof")),
      el("p", { class: "story-hero-foot" },
        `Every result below is computed from real simulated runs (${fmt.int(F.all.n)} scored engagements and more), not typed in by hand.`));
  }

  function sectionHead(title, text) {
    return el("div", { class: "story-sec" }, el("h2", {}, title), text ? el("p", {}, text) : null);
  }

  function steps(F, ex) {
    const tr = ex && ex.tr;
    // enough hops to show the lock and a few calls after it
    const n = tr ? Math.min(24, Math.max(14, tr.tau + 7)) : 0;
    const ss = tr ? T.sensorStats(tr, n) : null;
    const leg = (items) => OTS.chart.legend(items);
    const card = (num, title, art, legend, text, extra) => el("article", { class: "card story-step" },
      el("div", { class: "story-step-h" }, el("span", { class: "story-num", "aria-hidden": "true" }, num), el("h3", { class: "card-k" }, title)),
      art, legend, el("p", { class: "card-d" }, text), extra);
    const det = OTS.tok("det");
    const grid = el("div", { class: "grid-3" },
      card("1", "A radio hops to a secret recipe",
        tr ? sketch(tr, "truth", n) : null,
        tr ? leg([{ label: "where the radio really was", kind: "dot", color: det }]) : null,
        `The radio jumps between ${F.M} channels. A hidden recipe (a small algorithm with a secret key) decides every jump, so whoever knows the recipe and the key knows every future channel.`),
      card("2", "Our sensor is imperfect",
        tr ? sketch(tr, "sensor", n) : null,
        tr ? leg([{ label: "heard", kind: "dot", color: det }, { label: "missed hop", kind: "hollow" }, { label: "ghost signal", kind: "x" }]) : null,
        ["The listening sensor misses some hops and hears “ghosts”: signals that were never the radio.",
          ss ? ` Here it missed ${ss.miss} of the first ${n} hops and heard ${ss.ghost} ghosts.` : ""]),
      card("3", "OTS cracks it, or says “unknown”",
        tr ? sketch(tr, "ots", n) : null,
        tr ? leg([{ label: "what OTS hears (it cannot tell ghosts apart)", kind: "dot", color: "rgba(234, 243, 255, 0.4)" }, { label: "OTS's call", kind: "ring" }]) : null,
        [tr ? `By hop ${tr.tau} OTS had worked out the recipe and its key. Each ring is its call for a hop, made before the hop happened, even where the sensor missed it. ` : "",
          "When no recipe fits, for example real encryption, it says “unknown” instead of guessing."]));
    return [grid, tr ? el("p", { class: "story-src" },
      `Sketches drawn from a real recorded run: ${OTS.emitName(tr.emitter)} radio, ${OTS.sensorLabel(tr).toLowerCase()}. OTS never sees the real path or which signals are ghosts.`) : null];
  }

  function tiles(F) {
    const out = [];
    const tile = (lab, val, unit, sub, hot) => el("div", { class: "tile" + (hot ? " hot" : "") },
      el("div", { class: "tile-l" }, lab), el("div", { class: "tile-v num" }, val, unit ? el("small", {}, " " + unit) : null),
      el("div", { class: "tile-s" }, sub));
    if (F.all.hops_med != null) out.push(tile("Spots the pattern in", fmt.int(F.all.hops_med), "hops",
      ["Half of all correct identifications needed ", el("b", {}, fmt.int(F.all.hops_med) + " hops or fewer"), ` (each recording is ${OTS.data.window} hops long).`], true));
    if (F.detTop1 != null) out.push(tile("Next hop called right", fmt.pct0(F.detTop1), null,
      ["Once locked onto a key-based radio. A random guess gets ", el("b", {}, `1 in ${Math.round(1 / F.chance)}`), ` (${fmt.pct(F.chance)}).`]));
    const tx = times(F.all.ots_fb, F.all.base_fb);
    if (tx) out.push(tile("Fewer false claims", tx, null,
      ["Fewer dangerous “I cracked it” claims that were not true: ", el("b", {}, fmt.per100(F.all.ots_fb)), " vs ",
        el("b", {}, fmt.per100(F.all.base_fb)), " per 100 runs for the standard test."]));
    else out.push(tile("Dangerous false claims", fmt.per100(F.all.ots_fb), "per 100",
      ["Standard test: ", el("b", {}, fmt.per100(F.all.base_fb)), " per 100 runs."]));
    if (F.cs) out.push(tile("Honest “unknown”", fmt.pct0(F.cs.ots_unknown), null,
      ["of runs against unbreakable encryption end in “unknown”, the right answer. The standard test: right ", el("b", {}, fmt.pct0(F.cs.base_acc)), " of the time."]));
    if (F.harsh) out.push(tile("Right with a bad sensor", fmt.pct0(F.harsh.ots_acc), null,
      ["Right answers when the sensor misses ", el("b", {}, pctWord(1 - F.worstPreset.p_d) + " of hops"),
        " and hears frequent ghosts (standard test: ", fmt.pct0(F.harsh.base_acc), ")."]));
    return el("div", { class: "story-tiles" }, out);
  }

  function caseFile(F, ex) {
    if (!ex) return null;
    const tr = ex.tr, all = T.sensorStats(tr, tr.window), hits = T.hitStats(tr, tr.window);
    const conf = T.post(tr, tr.tau, tr.decision);
    const match = tr.secret_match;
    const okSecret = match === "exact" || match === "equivalent";
    const item = (ic, when, what, sub, key, extra) => el("li", { class: key ? "story-key" : null },
      el("span", { class: "story-ic", "aria-hidden": "true" }, ic),
      el("div", {}, el("div", { class: "story-when" }, when), el("div", { class: "story-what" }, what), sub ? el("div", { class: "story-sub" }, sub) : null, extra));
    const em = OTS.emitter(tr.emitter);
    return el("article", { class: "card story-case" },
      el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, "One real run, start to finish"),
        el("p", { class: "card-k" }, `Locked at hop ${tr.tau}, then called ${hits.hit} of the next ${hits.n} hops right`))),
      el("ol", {},
        item(icon(...ICONS.listen), "Hop 1 · listening", `A ${em ? em.kind.toLowerCase() : "hidden"} radio, heard through a ${OTS.preset(tr.preset)?.name.toLowerCase() || ""} sensor`,
          `Over the ${tr.window}-hop recording the sensor missed ${all.miss} hops and heard ${all.ghost} ghosts.`),
        item(icon(...ICONS.lock), `Hop ${tr.tau} · locked`, [`${fmt.pct(conf)} sure it is the `, el("b", {}, OTS.hypName(tr.decision)), " pattern"],
          `It passed the 95% bar after only ${tr.tau} hops, so OTS named the pattern.`, true),
        item(icon(...ICONS.key), "Secret key recovered", el("span", { class: "mono" }, tr.secret_recovered || "—"), null, true,
          okSecret ? el("span", { class: "status good" }, el("span", { class: "ic" }, "✓"), match === "exact" ? "matches the real key" : "equivalent key, same hops")
            : el("span", { class: "status bad" }, el("span", { class: "ic" }, "✗"), "does not match")),
        item(icon(...ICONS.next), `Hops ${tr.tau + 1}–${tr.window} · predicting`, `Called ${hits.hit} of ${hits.n} next hops right`,
          `A random guess would get about ${Math.round(hits.n * F.chance)}.`, true)),
      el("div", { class: "story-card-row" },
        el("p", { class: "card-foot" }, `Picked as a clear example from ${F.nTraces} recorded runs. The tiles above count all ${fmt.int(F.all.n)} runs, good and bad.`),
        el("button", { class: "btn", onclick: () => { OTS.select(ex.emitter, ex.preset, ex.idx); OTS.views.go("live"); } }, "▶ Watch this run")));
  }

  function falseClaims(F) {
    const a = F.all, us = Math.round(a.ots_fb * 100), them = Math.round(a.base_fb * 100);
    const col = (cls, name, p, bad, color) => el("div", { class: "story-dots-col " + cls },
      el("h3", {}, name, el("small", { class: "num" }, fmt.per100(p), el("span", {}, " per 100 runs"))),
      dots100(bad, color, `${name}: about ${bad} of 100 runs ended in a dangerous false claim`));
    return el("article", { class: "card" },
      el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, "Out of 100 runs"),
        el("p", { class: "card-k" }, "How often each method claimed a crack that was not real"),
        el("p", { class: "card-d" }, "A false “I cracked it” is the dangerous mistake: people would act on a prediction that is wrong. Each dot is one run."))),
      el("div", { class: "legend" },
        el("span", {}, el("i", { class: "story-dot-key story-bad", "aria-hidden": "true" }, "✕"), "false claim"),
        el("span", {}, el("i", { class: "story-dot-key", style: "background:rgba(34, 211, 238, 0.35)" }), "other OTS run"),
        el("span", {}, el("i", { class: "story-dot-key", style: "background:rgba(123, 134, 152, 0.45)" }), "other standard-test run")),
      el("div", { class: "story-dots-wrap" },
        col("story-us", "OTS", a.ots_fb, us, OTS.tok("accent")),
        col("", "Standard test", a.base_fb, them, OTS.tok("baseline"))),
      el("p", { class: "card-foot" }, `Measured over all ${fmt.int(a.n)} runs; dots rounded to whole runs.`));
  }

  function comparison(F) {
    const mark = (kind) => el("span", { class: "story-mark story-" + kind, role: "img",
      "aria-label": kind === "yes" ? "Yes" : kind === "no" ? "No" : "Partly" }, kind === "yes" ? "✓" : kind === "no" ? "✗" : "△");
    const cell = (label, kind, value, detail) => el("td", { "data-label": label },
      el("div", { class: "story-cell" }, mark(kind), el("b", {}, value), el("span", { class: "story-det" }, detail)));
    const row = (what, sub, us, them) => el("tr", {}, el("td", {}, what, sub ? el("small", {}, sub) : null),
      cell("OTS", ...us), cell("Standard approach", ...them));
    const rows = [];
    if (F.bm) rows.push(row("Copes with missed hops", "Cracking a shift-register radio",
      ["yes", `${fmt.pct0(F.bm.sfMin)} cracked`, `even with up to ${fmt.pct0(F.bm.maxMiss)} of hops missed (${F.bm.seeds} secrets per sensor level${F.bm.noGhosts ? ", no ghosts" : ""}).`],
      ["no", `${fmt.pct0(F.bm.bmMax)} cracked`, `The textbook decoder (Berlekamp–Massey) fails once ${fmt.pct0(F.bm.firstMiss)} or more of hops are missed.`]));
    if (F.cs) rows.push(row("Says “I don't know”", "When there is nothing to find",
      ["yes", `${fmt.pct0(F.cs.ots_unknown)} of the time`, `against unbreakable encryption (${F.cs.n} runs).`],
      ["no", F.cs.base_acc === 0 ? "Never right" : `Right ${fmt.pct0(F.cs.base_acc)}`, `The standard test has no “unknown” option and claimed a crack in ${fmt.pct0(F.cs.base_fb)} of those runs.`]));
    rows.push(row("Predicts the next hop", "Over whole runs, every pattern",
      ["yes", `${fmt.pct0(F.all.top1)} called right`, `Random guessing: ${fmt.pct(F.chance)}.`],
      ["part", `${fmt.pct0(F.all.markov_top1)} called right`, "Frequency statistics (counting which jumps were common so far)."]));
    if (F.keyTraces.length) rows.push(row("Recovers the secret key", "Key-based radios, recorded runs",
      ["yes", `${F.keyOk} of ${F.keyTraces.length} runs`, "the exact key, or a different key that gives the identical hops."],
      ["no", "No", "The standard test only names the type of pattern; it reports no key."]));
    rows.push(row("Dangerous false claims", "“I cracked it” when it had not",
      ["yes", `${fmt.per100(F.all.ots_fb)} per 100 runs`, `over all ${fmt.int(F.all.n)} runs.`],
      ["no", `${fmt.per100(F.all.base_fb)} per 100 runs`, "Same runs, same detections."]));
    const table = el("table", { class: "table story-vs" },
      el("thead", {}, el("tr", {}, el("th", {}, "What matters"), el("th", { class: "story-col-us" }, "OTS"), el("th", {}, "Standard approach"))),
      el("tbody", {}, rows));
    // the weakest pattern x sensor combination, explained in plain words
    let weak = null;
    const w = F.worst;
    if (w && w.ots_acc < 0.8) {
      const pr = OTS.preset(w.preset), wrong = Math.round((1 - w.ots_acc) * w.n), dangerous = Math.round(w.ots_false_break * w.n);
      const why = w.emitter === "lcg_full"
        ? "This radio's secret can only be solved from an unbroken run of cleanly heard hops, and a sensor that misses this many hops and hears ghosts rarely delivers one. "
        : "It is the hardest combination in the test set. ";
      weak = el("div", { class: "story-weak" },
        el("span", { class: "status warn" }, el("span", { class: "ic" }, "!"), "Weak spot"),
        el("p", {}, el("b", {}, `${OTS.emitName(w.emitter)} with a ${pr ? pr.name.toLowerCase() : w.preset} sensor: right in ${ofRuns(w.ots_acc, w.n)} runs `),
          `(standard test: ${ofRuns(w.base_acc, w.n)}). `, why,
          wrong ? `${wrong - dangerous} of its ${wrong} misses were safe answers, not false “I cracked it” claims.` : ""));
    }
    return el("article", { class: "card" },
      el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, "Side by side"),
        el("p", { class: "card-k" }, "OTS does five things the standard tools cannot, or do far worse"),
        el("p", { class: "card-d" }, "Both sides get exactly the same noisy detections. “Standard approach” is the usual tool for each job: the standard identification test, the textbook decoder, or simple frequency statistics."))),
      el("div", { style: "overflow-x:auto" }, table), weak);
  }

  function rigor(F) {
    const nE = OTS.emitters.length, nP = OTS.presets.length, seeds = OTS.data.scoreboard.n_seeds;
    return el("section", { class: "card story-rigor" },
      el("div", { class: "story-rigor-ic" }, icon("M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6l7-3z", "M8.5 12l2.5 2.5 4.5-5")),
      el("div", { class: "story-rigor-t" },
        el("p", {}, `Every number here comes from ${fmt.int(F.all.n)} real simulated engagements across ${nE} hidden patterns and ${nP} sensor conditions.`),
        el("small", {}, `${nE} patterns × ${nP} sensors × ${seeds} random secrets each, plus ${F.nTraces} recorded engagements you can replay hop by hop${F.bm ? ` and a separate missed-hop test of the textbook decoder (${F.bm.seeds} secrets per sensor level)` : ""}. Data built ${OTS.data.generated}.`)),
      el("div", { class: "story-rigor-b" },
        el("button", { class: "btn", onclick: () => OTS.views.go("live") }, "▶ Watch the live demo"),
        el("button", { class: "btn", onclick: () => OTS.views.go("proof") }, "See all the evidence")));
  }

  // ================================================================== view
  OTS.views.register({
    id: "story", nav: "Story", title: "Why it matters", hint: "The 60-second version for anyone",
    subtitle: "What OTS does, in plain English, and the numbers that show it works",
    icon: "M3 12h4l3 8 4-16 3 8h4", usesPlayer: false,
    mount(root) {
      injectCss();
      const F = facts(), ex = exampleRun();
      root.append(el("div", { class: "page story-page" },
        hero(F),
        sectionHead("The problem in three steps", null),
        steps(F, ex),
        sectionHead("What it achieves", `Measured over all ${fmt.int(F.all.n)} runs: every hidden pattern, every sensor condition, the easy runs and the hard ones.`),
        tiles(F),
        el("div", { class: "grid-2" }, caseFile(F, ex), falseClaims(F)),
        sectionHead("OTS vs the standard approach", null),
        comparison(F),
        rigor(F)));
      // static page: no subscriptions, timers or charts to release
      return () => {};
    },
  });
})();
