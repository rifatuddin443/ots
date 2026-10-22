/* Challenge view: "Beat the algorithm", a 20-round guessing game.
 *
 * The player sees the last few hops exactly as the sensor heard them (white dots: real hops
 * can be missing and ghosts look the same as real detections) and guesses the next channel
 * on a 16-key pad. The reveal shows the real channel next to OTS's call for that hop, which
 * was recorded before the hop happened (OTS.T.forecastTop). Scores: You, OTS and the
 * random-guessing expectation. After 20 rounds a summary explains, from the data, what OTS did.
 *
 * Uses recorded 'typical' sensor runs only and keeps its own state (never touches OTS.player).
 * The game state lives at module level, so leaving the page and coming back resumes the game.
 */
(function () {
  "use strict";
  const { el, svg, fmt, T } = OTS;
  const ROUNDS = 20;          // guesses per game (round r: hops 1..r seen, guess hop r + 1)
  const PRESET = "typical";   // recorded runs with the typical sensor
  /** Patterns offered in the setup row, with a plain-English line each (facts about the radio, not results). */
  const PATTERNS = [
    { id: "lfsr", tag: "Shift register", plain: "A small circuit (a shift register) picks each channel from a secret starting state." },
    { id: "weak_seed", tag: "Small secret key", plain: "A strong scrambler picks each channel, but it was started from one of only 256 secret seeds." },
    { id: "henon", tag: "Chaotic", plain: "A chaotic formula picks each channel: tiny differences grow fast, so it can only be predicted a short way ahead." },
    { id: "markov", tag: "Habits, not rules", plain: "Each hop is a weighted dice roll that depends on the current channel: the radio has habits, not fixed rules." },
    { id: "csprng", tag: "Unbreakable control", plain: "Cryptographic code (SHA-256 with a 128-bit secret key) picks each channel. Nobody can predict it: this is the fairness check." },
  ];

  // ================================================================ styles
  const CSS = `
.challenge-page { container: challenge / inline-size; }
.challenge-setup { padding: 14px 18px 15px; }
.challenge-setup-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; }
.challenge-setup .card-t { flex: none; }
.challenge-pats { gap: 6px; }
.challenge-pat { flex-direction: column; align-items: flex-start; gap: 2px; padding: 7px 12px; text-align: left; }
.challenge-pat b { font-weight: 600; font-size: 13.5px; color: var(--ink); }
.challenge-pat small { font-size: 11px; color: var(--muted); }
.challenge-pat.sel small { color: var(--ink-2); }
.challenge-grow { flex: 1 1 0; min-width: 0; }
.challenge-rec { font-size: 13px; color: var(--ink-2); white-space: nowrap; }
.challenge-desc { margin: 10px 0 0; font-size: 13.5px; color: var(--ink-2); }
.challenge-desc .challenge-sens { color: var(--muted); }

.challenge-main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 18px; align-items: start; }
.challenge-left { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
.challenge-game { container: chgame / inline-size; }
.challenge-game .card-h { margin-bottom: 8px; }
.challenge-head:focus { outline: none; }
.challenge-game.is-done .challenge-mini { visibility: hidden; height: 0; }
.challenge-mini { display: none; font: 600 13px/1.4 var(--mono); color: var(--ink-2); white-space: nowrap; }
.challenge-mini b { color: var(--ink); }
.challenge-mini .challenge-mini-o b { color: var(--accent-ink); }

.challenge-play { display: grid; grid-template-columns: minmax(0, 1fr) 272px; gap: 22px; align-items: start; }
.challenge-chartcol { min-width: 0; }
.challenge-legend { margin: 10px 0 0; font-size: 12.5px; gap: 6px 16px; }
.challenge-kdia { display: inline-block; width: 9px; height: 9px; margin: 0 2px; border: 2px solid var(--ink); background: var(--panel); transform: rotate(45deg); }
.challenge-kband { display: inline-block; width: 18px; height: 10px; border-radius: 3px; background: var(--det); }
.challenge-hop { touch-action: pan-y; }
.challenge-hop .gl { stroke: var(--grid); stroke-width: 1; opacity: 0.7; }
.challenge-hop .ax-t.is-true { fill: var(--ink); font-weight: 700; }
.challenge-hop .nx-t { fill: var(--accent-ink); font: 700 10px var(--mono); letter-spacing: 0.12em; }
.challenge-hop .nx-tick { fill: var(--accent-ink); font: 700 11px var(--mono); }
.challenge-hop .q-t { fill: var(--muted); font: 700 17px var(--mono); }
.challenge-hop .guide { stroke: rgba(232, 238, 247, 0.4); stroke-width: 1; stroke-dasharray: 3 4; }
.challenge-pop { transform-box: fill-box; transform-origin: center; animation: challenge-pop 0.42s cubic-bezier(0.2, 0.9, 0.3, 1.25) both; }
.challenge-fade { animation: challenge-fade 0.5s ease both; }
@keyframes challenge-pop { from { opacity: 0; transform: scale(0.2); } to { opacity: 1; transform: scale(1); } }
@keyframes challenge-fade { from { opacity: 0; } to { opacity: 1; } }

.challenge-padcol { min-width: 0; }
.challenge-padh { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 8px; }
.challenge-padh b { font: 600 11px/1.3 var(--mono); letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-2); }
.challenge-padh span { font-size: 11.5px; color: var(--muted); text-align: right; }
.challenge-pad { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
.challenge-key {
  min-height: 52px; min-width: 0; padding: 4px 2px; border-radius: 10px; border: 1px solid var(--line-2);
  background: var(--surface); color: var(--ink); font: 700 18px/1 var(--mono); cursor: pointer;
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 3px;
  transition: background 0.15s, border-color 0.15s, transform 0.1s;
}
.challenge-key small { font: 700 8.5px/1 var(--mono); letter-spacing: 0.1em; text-transform: uppercase; }
.challenge-key:hover:not(:disabled) { background: var(--raise); border-color: rgba(34, 211, 238, 0.6); }
.challenge-key:active:not(:disabled) { transform: scale(0.96); }
.challenge-key:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.challenge-key:disabled { cursor: default; opacity: 0.35; }
.challenge-key.is-pick { opacity: 1; box-shadow: inset 0 0 0 2px var(--ink); }
.challenge-key.is-real { opacity: 1; background: var(--det); border-color: var(--det); color: #05080e; }

.challenge-result { margin-top: 14px; min-height: 190px; display: flex; flex-direction: column; gap: 8px; }
.challenge-res-h { font: 600 10.5px/1.3 var(--mono); letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.challenge-res-big { font: 700 20px/1.25 var(--sans); color: var(--ink); letter-spacing: -0.01em; }
.challenge-res-big .num { font-family: var(--mono); }
.challenge-res-row { display: grid; grid-template-columns: 128px minmax(0, 1fr); align-items: center; gap: 4px 10px; font-size: 13px; color: var(--ink-2); }
.challenge-res-row .status { justify-self: start; }
.challenge-res-note { font-size: 12.5px; line-height: 1.45; color: var(--ink-2); }
.challenge-res-note.muted { color: var(--muted); }
.challenge-result .btn.big { justify-content: center; margin-top: 4px; }
.challenge-in { animation: challenge-in 0.4s ease both; }
@keyframes challenge-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

.challenge-note { font-size: 14.5px; }

.challenge-score .card-h { margin-bottom: 12px; }
.challenge-tiles { display: flex; flex-direction: column; gap: 8px; }
.challenge-st { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 11px 14px; }
.challenge-st .tile-v { margin-top: 0; font-size: 30px; white-space: nowrap; }
.challenge-st .tile-s { margin-top: 2px; font-size: 12.5px; }
.challenge-st.is-rnd .tile-v { color: var(--ink-2); font-size: 24px; }
.challenge-bump { display: inline-block; animation: challenge-bump 0.5s ease; }
@keyframes challenge-bump { 0% { transform: scale(1); } 40% { transform: scale(1.22); } 100% { transform: scale(1); } }
.challenge-tracks { margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--grid); }
.challenge-tracks.is-done { margin-top: 0; padding-top: 0; border-top: 0; }
.challenge-track-l { display: flex; justify-content: space-between; font-size: 12.5px; color: var(--ink-2); margin: 8px 0 4px; }
.challenge-track-l b { font-weight: 600; color: var(--ink); }
.challenge-track { display: grid; grid-template-columns: repeat(20, minmax(0, 1fr)); gap: 2px; }
.challenge-cell {
  height: 19px; border-radius: 3px; background: #142034; display: flex; align-items: center; justify-content: center;
  font: 700 10px/1 var(--sans); min-width: 0; overflow: hidden;
}
.challenge-cell.ok { background: rgba(12, 163, 12, 0.28); color: var(--good-ink); }
.challenge-cell.no { background: rgba(208, 59, 59, 0.24); color: var(--critical-ink); }
.challenge-cell.now { box-shadow: inset 0 0 0 1.5px var(--accent); }
.challenge-track-s { display: flex; justify-content: space-between; font: 10.5px var(--mono); color: var(--muted); margin-top: 4px; }
.challenge-track-k { font-size: 12px; color: var(--muted); margin-top: 10px; }

.challenge-sum { display: flex; flex-direction: column; gap: 14px; }
.challenge-sumtiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.challenge-sumtiles .tile-v { font-size: 38px; }
.challenge-sumtiles .is-rnd .tile-v { color: var(--ink-2); }
.challenge-explain { margin: 0; font-size: 15px; line-height: 1.55; color: var(--ink); max-width: 78ch; }
.challenge-explain.sub { font-size: 13.5px; color: var(--ink-2); }
.challenge-secret { font-family: var(--mono); font-size: 0.92em; color: var(--accent-ink); white-space: nowrap; }
.challenge-sumgrid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px 26px; align-items: start; }
.challenge-sumtext { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.challenge-sumchart { min-width: 0; }
.challenge-sumchart .legend { margin-bottom: 4px; }
.challenge-sumbtns { display: flex; flex-wrap: wrap; gap: 10px; }

@container challenge (max-width: 1040px) {
  .challenge-main { grid-template-columns: minmax(0, 1fr); }
  .challenge-mini { display: block; }
}
@container challenge (max-width: 560px) {
  .challenge-setup-row .challenge-grow { display: none; }
  .challenge-pats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); width: 100%; }
  .challenge-pat { width: 100%; }
  .challenge-pat b, .challenge-pat small { max-width: 100%; overflow: hidden; text-overflow: ellipsis; }
}
@container chgame (max-width: 660px) {
  .challenge-play, .challenge-sumgrid { grid-template-columns: minmax(0, 1fr); gap: 16px; }
  .challenge-result { min-height: 0; }
}
@container chgame (max-width: 480px) {
  .challenge-sumtiles { grid-template-columns: minmax(0, 1fr); }
  .challenge-sumtiles .tile { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
  .challenge-sumtiles .tile-v { margin-top: 0; font-size: 30px; }
  .challenge-key { min-height: 50px; }
}`;
  function injectCss() {
    if (document.getElementById("css-challenge")) return;
    document.head.append(el("style", { id: "css-challenge" }, CSS));
  }

  // =============================================================== helpers
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const pl = (n, word, many) => `${n} ${n === 1 ? word : many || word + "s"}`;
  /** Expected count for random guessing: "0", "0.3", "1.3", "12". */
  const approx = (v) => (v === 0 ? "0" : v < 10 ? v.toFixed(1) : String(Math.round(v)));
  /** Forecast confidence that never rounds up to a false 100%. */
  const sure = (p) => (p >= 0.995 ? ">99%" : p > 0 && p < 0.01 ? "<1%" : fmt.pct0(p));
  const quoteName = (id) => "“" + (id === "unknown" || id === "H0" ? "unknown pattern" : OTS.hypName(id)) + "”";
  const runsFor = (emitter) => OTS.tracesFor(emitter, PRESET);
  const chance = () => OTS.S.chance();
  /** "catches 80% of hops and hears a ghost signal about every 3 hops" */
  function sensorText(tr) {
    const g = OTS.ghostsPerHop(tr);
    const ghosts = g <= 0 ? "hears no ghost signals" : g < 1 ? `hears a ghost signal about every ${Math.max(2, Math.round(1 / g))} hops` : `hears about ${g.toFixed(1)} ghost signals per hop`;
    return `catches ${Math.round(tr.p_d * 100)}% of hops and ${ghosts}`;
  }
  /** One plain sentence about the sensor's mistakes on one revealed hop (slot j). */
  function sensorNote(tr, j) {
    const d = tr.dets[j], c = tr.truth[j], heard = d.includes(c), g = d.filter((x) => x !== c).length;
    if (heard && !g) return "The sensor heard this hop cleanly.";
    if (heard) return `The sensor heard it, plus ${pl(g, "ghost")}.`;
    if (!g) return "The sensor missed this hop: no dot will appear.";
    return `The sensor missed this hop and heard ${pl(g, "ghost")} instead.`;
  }

  // ============================================================ game state
  const G = { emitter: "lfsr", run: 0, tr: null, round: 1, phase: "ask", log: [], hover: null, fresh: false };
  function newGame(emitter, run) {
    const list = runsFor(emitter);
    if (!list.length) return false;
    G.emitter = emitter; G.run = ((run % list.length) + list.length) % list.length;
    G.tr = list[G.run]; G.round = 1; G.phase = "ask"; G.log = []; G.hover = null; G.fresh = false;
    return true;
  }
  /** Record a guess for the current round. OTS's call is the forecast it made after hop r, before hop r + 1. */
  function guess(ch) {
    if (G.phase !== "ask") return false;
    const tr = G.tr, r = G.round, truth = tr.truth[r], f = T.forecastTop(tr, r);
    G.log.push({ round: r, you: ch, ots: f ? f.ch : null, p: f ? f.p : 0, truth, youRight: ch === truth, otsRight: !!f && f.ch === truth });
    G.phase = "reveal"; G.hover = null; G.fresh = true;
    return true;
  }
  function advance() {
    if (G.phase !== "reveal") return;
    if (G.round >= ROUNDS) G.phase = "done";
    else { G.round++; G.phase = "ask"; }
    G.fresh = false;
  }
  const tally = () => ({ n: G.log.length, you: G.log.filter((e) => e.youRight).length, ots: G.log.filter((e) => e.otsRight).length });
  /** OTS's right calls over rounds 1..ROUNDS of any trace (for the "all recordings" footnote). */
  function otsHits(tr) {
    let h = 0;
    for (let r = 1; r <= ROUNDS; r++) { const f = T.forecastTop(tr, r); if (f && f.ch === tr.truth[r]) h++; }
    return h;
  }

  // ============================================================= hop chart
  /** Last few hops as the sensor heard them, plus the "next hop" column with You / OTS sub-columns. */
  function makeHopChart(box) {
    let W = 0;
    const ro = new ResizeObserver(() => { if (box.clientWidth && box.clientWidth !== W) draw(false); });
    ro.observe(box);

    function draw(animate) {
      const tr = G.tr, w = box.clientWidth;
      if (!tr || !w || G.phase === "done") return;
      W = w;
      const M = tr.M, r = G.round, rec = G.phase === "reveal" ? G.log[G.log.length - 1] : null;
      const small = w < 440;
      const m = { l: small ? 42 : 48, r: 4, t: 28, b: 44 };
      const iw = w - m.l - m.r;
      const nextW = clamp(Math.round(iw * 0.2), 76, 104), gap = 8;
      const N = clamp(Math.floor((iw - nextW - gap) / 28), 4, 10);      // history columns ("last ~10 hops")
      const colW = (iw - nextW - gap) / N;
      const ih = small ? 256 : 288, rowH = ih / M, H = m.t + ih + m.b;
      const cx = (i) => m.l + (i + 0.5) * colW;
      const cy = (c) => m.t + ih - (c + 0.5) * rowH;
      const nx = m.l + N * colW + gap, youX = nx + nextW * 0.3, otsX = nx + nextW * 0.7, right = nx + nextW;
      const first = r - N + 1;                                            // hop number in column 0
      const det = OTS.tok("det"), acc = OTS.tok("accent"), ink = OTS.tok("ink"), panel = OTS.tok("panel");
      const delay = (ms) => (animate ? `animation-delay:${ms}ms` : null);
      const pop = animate ? "challenge-pop" : null;                        // reveal animation only on the guess itself
      const lo = Math.max(1, first);
      const label = `Hop chart: hops ${lo} to ${r} as the sensor heard them (white dots). The next hop, ${r + 1}, is hidden.` +
        (rec ? ` It was channel ${fmt.ch(rec.truth)}; you guessed ${fmt.ch(rec.you)}; OTS called ${rec.ots == null ? "nothing" : fmt.ch(rec.ots)}.` : "");
      const root = svg("svg", { viewBox: `0 0 ${w} ${H}`, height: H, role: "img", "aria-label": label });

      // channel grid + y ticks (every channel while the rows are tall enough)
      const yStep = rowH >= 14 ? 1 : 3;
      for (let c = 0; c < M; c++) {
        root.append(svg("line", { class: "gl", x1: m.l, x2: right, y1: cy(c), y2: cy(c) }));
        if (c % yStep === 0 || c === M - 1)
          root.append(svg("text", { class: "ax-t" + (rec && rec.truth === c ? " is-true" : ""), x: m.l - 8, y: cy(c) + 3.5, "text-anchor": "end" }, fmt.ch(c)));
      }
      root.append(svg("line", { class: "ax", x1: m.l, x2: right, y1: m.t + ih, y2: m.t + ih }));
      root.append(svg("text", { class: "ax-title", x: m.l + iw / 2, y: H - 6, "text-anchor": "middle" }, "Hop number (time →)"));
      root.append(svg("text", { class: "ax-title", x: 0, y: 0, "text-anchor": "middle", transform: `translate(13 ${m.t + ih / 2}) rotate(-90)` }, `Channel (0–${M - 1})`));

      // history: one column per hop, white dots = everything the sensor heard (ghosts included)
      const dotR = clamp(Math.min(colW, rowH) * 0.3, 3.5, 5.5);
      const xEvery = colW >= 22 ? 1 : 2;
      for (let i = 0; i < N; i++) {
        const h = first + i;
        if (h < 1) continue;
        const x = cx(i), d = tr.dets[h - 1];
        if ((r - h) % xEvery === 0) {
          root.append(svg("line", { class: "ax", x1: x, x2: x, y1: m.t + ih, y2: m.t + ih + 4 }));
          root.append(svg("text", { class: "ax-t", x, y: m.t + ih + 17, "text-anchor": "middle" }, String(h)));
        }
        for (const c of d) root.append(svg("circle", { cx: x, cy: cy(c), r: dotR, fill: det }));
        const hit = svg("rect", { class: "hit", x: m.l + i * colW, y: m.t, width: colW, height: ih });
        hit.addEventListener("pointermove", (ev) => OTS.tip.show(ev, {
          title: `Hop ${h}`,
          rows: d.length ? d.map((c) => ({ value: fmt.ch(c), label: "the sensor heard a signal here", color: det, kind: "square" }))
            : [{ value: "—", label: "the sensor heard nothing" }],
        }));
        hit.addEventListener("pointerleave", () => OTS.tip.hide());
        root.append(hit);
      }

      // next-hop column: shaded band with "YOU" and "OTS" sub-columns
      root.append(svg("rect", { x: nx, y: m.t, width: nextW, height: ih, rx: 4, fill: "rgba(34, 211, 238, 0.07)" }));
      root.append(svg("line", { x1: nx, x2: nx, y1: m.t, y2: m.t + ih, stroke: "rgba(34, 211, 238, 0.35)", "stroke-dasharray": "3 3" }));
      root.append(svg("text", { class: "nx-tick", x: nx + nextW / 2, y: m.t + ih + 17, "text-anchor": "middle" }, `${r + 1}${rec ? "" : " ?"}`));
      const head = (x, txt) => svg("text", { class: "nx-t", x, y: m.t - 10, "text-anchor": "middle" }, txt);
      if (rec) {
        root.append(head(youX, "YOU"), head(otsX, "OTS"));
        // the real channel: a lit band across the next column, plus a guide line back to its axis label
        const y = cy(rec.truth);
        root.append(svg("line", { class: "guide" + (animate ? " challenge-fade" : ""), x1: m.l, x2: nx, y1: y, y2: y, style: delay(320) }));
        root.append(svg("rect", { class: pop, x: nx + 4, y: y - rowH / 2 + 1, width: nextW - 8, height: rowH - 2, rx: 4, fill: det, style: delay(320) }));
        const s = clamp(rowH * 0.44, 6, 8);
        const yy = cy(rec.you);
        root.append(svg("path", { class: pop, d: `M${youX} ${yy - s}L${youX + s} ${yy}L${youX} ${yy + s}L${youX - s} ${yy}Z`, fill: panel, stroke: ink, "stroke-width": 2, style: delay(0) }));
        if (rec.ots != null) root.append(svg("circle", { class: pop, cx: otsX, cy: cy(rec.ots), r: clamp(rowH * 0.42, 6, 8), fill: panel, stroke: acc, "stroke-width": 2.5, style: delay(160) }));
        const hit = svg("rect", { class: "hit", x: nx, y: m.t, width: nextW, height: ih });
        hit.addEventListener("pointermove", (ev) => OTS.tip.show(ev, {
          title: `Hop ${r + 1}`,
          rows: [
            { value: fmt.ch(rec.truth), label: "actual channel", color: det, kind: "square" },
            { value: fmt.ch(rec.you), label: rec.youRight ? "your guess: right" : "your guess: wrong" },
            rec.ots == null ? null : { value: fmt.ch(rec.ots), label: `OTS's call (${sure(rec.p)} sure): ${rec.otsRight ? "right" : "wrong"}`, color: acc, kind: "square" },
          ].filter(Boolean),
        }));
        hit.addEventListener("pointerleave", () => OTS.tip.hide());
        root.append(hit);
      } else {
        root.append(head(youX, "YOU"), head(otsX, "OTS"));
        // OTS has already made its call: it stays sealed until the player guesses
        root.append(svg("text", { class: "q-t", x: otsX, y: m.t + ih / 2 + 6, "text-anchor": "middle" }, "?"));
        if (G.hover != null) {                     // preview of the key under the pointer / keyboard focus
          const y = cy(G.hover), s = clamp(rowH * 0.44, 6, 8);
          root.append(svg("line", { class: "guide", x1: m.l, x2: nx, y1: y, y2: y }));
          root.append(svg("path", { d: `M${youX} ${y - s}L${youX + s} ${y}L${youX} ${y + s}L${youX - s} ${y}Z`, fill: panel, stroke: ink, "stroke-width": 2, opacity: 0.75 }));
        } else {
          root.append(svg("text", { class: "q-t", x: youX, y: m.t + ih / 2 + 6, "text-anchor": "middle" }, "?"));
        }
      }
      OTS.clear(box).append(root);
    }
    return { draw, destroy() { ro.disconnect(); OTS.clear(box); } };
  }
  function hopLegend() {
    const item = (icon, label) => el("span", {}, icon, label);
    return el("div", { class: "legend challenge-legend" },
      item(el("i", { class: "dotk", style: `background:${OTS.tok("det")}` }), "Sensor detection (real hop or ghost)"),
      item(el("i", { class: "challenge-kdia" }), "Your guess"),
      item(el("i", { class: "ringk" }), "OTS's call, made before the hop"),
      item(el("i", { class: "challenge-kband" }), "Actual channel (after you guess)"));
  }

  // ======================================================= summary wording
  /** Plain-English explanation of this game, computed from the trace and the game log: [main, ...extra]. */
  function explain(tr, log) {
    const kind = (OTS.emitter(tr.emitter) || {}).kind || "";
    const n = log.length, ots = log.filter((e) => e.otsRight).length, rnd = n * chance();
    let s = n; while (s > 0 && log[s - 1].otsRight) s--;
    const streak = n - s, startRound = s + 1;          // OTS's final unbroken run of right calls began in this round
    const tau = tr.tau, lockedIn = tr.declared && tau != null && tau <= ROUNDS;
    const why = kind === "Chaotic" ? "Chaotic patterns can only be predicted a short way ahead, and every missed hop or ghost blurs the picture."
      : kind === "Statistical" ? "A radio with habits instead of rules can only ever be predicted with an educated guess, even by a perfect model."
      : "";
    const secret = () => {
      if (tr.secret_match === "exact" && tr.secret_recovered) return [", and it recovered the radio's secret setting exactly (", el("span", { class: "challenge-secret" }, tr.secret_recovered), ")"];
      if (tr.secret_match === "family" && tr.secret_recovered) return [", and it found a close match for the radio's secret setting (", el("span", { class: "challenge-secret" }, tr.secret_recovered), ")"];
      if (tr.secret_match === "model") return ", and it learned the radio's habits from the hops";
      return "";
    };
    const out = [];
    if (tr.emitter === "csprng") {
      const row = OTS.S.byEmitter(PRESET).find((e) => e.id === tr.emitter);
      out.push(["Nobody can predict this radio: it is the unbreakable control. ",
        ots ? `OTS's ${pl(ots, "right call")} ${ots === 1 ? "was" : "were"} luck` : "OTS got none right",
        row && row.top1 != null ? `: over ${row.n} full test runs it called ${fmt.pct(row.top1)} of hops, about the same as random guessing (${+(chance() * 100).toFixed(2)}%).` : "."]);
      if (!tr.declared) out.push("What matters is that OTS never claimed to have cracked it. It kept answering “unknown pattern”, which is exactly the right answer.");
      else out.push(`In this recording OTS wrongly settled on ${quoteName(tr.decision)} at hop ${tau}: a false claim.`);
      if (tr.baseline && !tr.baseline.correct && tr.baseline.decision)
        out.push(`The standard test has no “unknown” answer: its best answer for this recording was ${quoteName(tr.baseline.decision)}, which is wrong.`);
      return out;
    }
    if (lockedIn && tr.correct && streak >= 5) {
      if (startRound <= tau) out.push([`OTS worked out the hidden pattern after seeing just ${pl(startRound, "hop")} and called every hop after that: ${streak} in a row. `,
        `It was 95% sure the pattern was ${quoteName(tr.decision)} by hop ${tau}`, secret(), "."]);
      else {
        const after = log.filter((e) => e.round >= tau), hit = after.filter((e) => e.otsRight).length;
        out.push([`OTS was 95% sure the pattern was ${quoteName(tr.decision)} by hop ${tau}`, secret(),
          `. After that it called ${hit} of the ${after.length} hops, finishing with ${streak} in a row. `, why]);
      }
    } else if (lockedIn && tr.correct) {
      const after = log.filter((e) => e.round >= tau), hit = after.filter((e) => e.otsRight).length;
      out.push([`OTS identified the pattern (${quoteName(tr.decision)}) at hop ${tau}`, secret(), `. After that it called ${hit} of the ${after.length} hops, against about ${approx(after.length * chance())} for random guessing. `, why]);
    } else if (lockedIn) {
      out.push(`In this recording OTS settled on the wrong pattern (${quoteName(tr.decision)} at hop ${tau}): a miss. ${why}`);
    } else if (tr.declared && tr.correct) {
      out.push(`Twenty hops were not enough to pin this radio down: OTS named it at hop ${tau} of the full recording. ${why} Even so it called ${ots} of ${n}, against about ${approx(rnd)} for random guessing.`);
    } else if (tr.declared) {
      out.push(`OTS had not settled on a pattern within these ${n} hops; later in the recording it settled on the wrong one (${quoteName(tr.decision)} at hop ${tau}): a miss. ${why}`);
    } else {
      out.push(`In this recording OTS never became 95% sure, so it kept saying “unknown pattern” instead of making a false claim: a cautious miss. ${why}`);
    }
    return out;
  }

  // ================================================================= view
  OTS.views.register({
    id: "challenge", nav: "Challenge", title: "Beat the algorithm", hint: "Can you guess the next hop?",
    subtitle: "Guess the radio's next channel from the same noisy signals OTS gets, then compare scores",
    icon: "M12 2a10 10 0 100 20 10 10 0 000-20zm0 6v4l3 2", usesPlayer: false,
    mount(root) {
      injectCss();
      const pats = PATTERNS.filter((p) => runsFor(p.id).length);
      if (!pats.length) {
        root.append(el("div", { class: "card empty" }, "No recorded runs with the typical sensor were found for this game."));
        return () => {};
      }
      if (!G.tr || !runsFor(G.emitter).length) newGame(pats.some((p) => p.id === G.emitter) ? G.emitter : pats[0].id, 0);

      // ---- layout
      const patSeg = el("div", { class: "seg challenge-pats", role: "group", "aria-label": "Hidden radio pattern" });
      const recLabel = el("span", { class: "challenge-rec" });
      const newBtn = el("button", { class: "btn", type: "button", title: "Same pattern, another recorded run", onclick: () => restart(G.emitter, G.run + 1) }, "↻ New game");
      const desc = el("p", { class: "challenge-desc" });
      const setup = el("div", { class: "card challenge-setup" },
        el("div", { class: "challenge-setup-row" }, el("p", { class: "card-t" }, "Hidden radio pattern"), patSeg,
          el("div", { class: "challenge-grow" }), recLabel, newBtn),
        desc);

      const roundT = el("p", { class: "card-t" }), headK = el("p", { class: "card-k" }), headD = el("p", { class: "card-d" });
      const headBox = el("div", { class: "challenge-head", tabindex: "-1" }, roundT, headK, headD);
      const mini = el("div", { class: "challenge-mini num" });
      const chartBox = el("div", { class: "chart challenge-hop" });
      const pad = el("div", { class: "challenge-pad", role: "group", "aria-label": "Your guess: pick a channel" });
      const result = el("div", { class: "challenge-result", "aria-live": "polite" });
      const play = el("div", { class: "challenge-play" },
        el("div", { class: "challenge-chartcol" }, chartBox, hopLegend()),
        el("div", { class: "challenge-padcol" },
          el("div", { class: "challenge-padh" }, el("b", {}, "Your guess"), el("span", {}, "high channels on top")),
          pad, result));
      const summary = el("div", { class: "challenge-sum", hidden: true });
      const game = el("div", { class: "card challenge-game" }, el("div", { class: "card-h" }, headBox, mini), play, summary);
      const note = el("div", { class: "callout challenge-note" }, el("span", { class: "tag" }, "FAIR PLAY"),
        el("strong", {}, "OTS sees exactly what you see: the noisy detections."),
        " It is never told the real channels; you are, after every guess.");
      const score = el("div", { class: "card challenge-score", "aria-label": "Scoreboard" });
      root.append(el("div", { class: "page challenge-page" }, setup,
        el("div", { class: "challenge-main" }, el("div", { class: "challenge-left" }, game, note), score)));

      const chart = makeHopChart(chartBox);
      let sumChart = null, nextBtn = null, last = { you: 0, ots: 0, n: 0 };

      // ---- channel keys: 4 per row, highest channels on the top row (mirrors the chart's y axis)
      const keys = [];
      const M = G.tr.M, rows = Math.ceil(M / 4);
      for (let row = rows - 1; row >= 0; row--) for (let ch = row * 4; ch < Math.min(M, row * 4 + 4); ch++) {
        const b = el("button", { class: "challenge-key num", type: "button", "aria-label": `Guess channel ${fmt.ch(ch)}`,
          onclick: () => onGuess(ch),
          onpointerenter: (ev) => { if (ev.pointerType === "mouse") setHover(ch); },
          onpointerleave: () => setHover(null),
          onfocus: () => { if (b.matches(":focus-visible")) setHover(ch); },
          onblur: () => setHover(null) }, el("span", {}, fmt.ch(ch)), el("small", {}));
        keys[ch] = b; pad.append(b);
      }

      // ---- actions
      function setHover(ch) {
        if (G.phase !== "ask" || G.hover === ch) return;
        G.hover = ch; chart.draw(false);
      }
      function onGuess(ch) {
        if (!guess(ch)) return;
        renderAll(true);
        if (nextBtn) nextBtn.focus({ preventScroll: true });
      }
      function onNext() {
        advance();
        renderAll(false);
        headBox.focus({ preventScroll: true });
        if (G.phase === "done") game.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
      function restart(emitter, run) {
        newGame(emitter, run);
        renderAll(false);
        headBox.focus({ preventScroll: true });
      }
      function otherPattern() {
        const i = pats.findIndex((p) => p.id === G.emitter);
        return pats[(i + 1) % pats.length];
      }

      // ---- renderers
      function renderSetup() {
        const tr = G.tr, list = runsFor(G.emitter), p = pats.find((x) => x.id === G.emitter);
        OTS.clear(patSeg).append(...pats.map((x) => {
          const on = x.id === G.emitter;
          return el("button", { class: "btn challenge-pat" + (on ? " sel" : ""), type: "button", "aria-pressed": String(on), title: x.plain,
            onclick: () => restart(x.id, 0) }, el("b", {}, OTS.emitName(x.id)), el("small", {}, x.tag));
        }));
        recLabel.textContent = `Recording ${G.run + 1} of ${list.length}`;
        OTS.clear(desc).append(p ? p.plain + " " : "",
          el("span", { class: "challenge-sens" }, `${(OTS.preset(tr.preset) || { name: "Typical" }).name} sensor: ${sensorText(tr)}.`));
      }
      function renderHead() {
        const t = tally(), rnd = t.n * chance();
        if (G.phase === "done") {
          roundT.textContent = `Game over · ${ROUNDS} hops`;
          headK.textContent = t.ots > t.you ? `OTS called ${t.ots} of ${t.n} hops. You called ${t.you}.`
            : t.you > t.ots ? `You beat OTS this time: ${t.you} to ${t.ots}.` : `A draw: ${t.you} each.`;
          headD.textContent = `${OTS.emitName(G.emitter)} · recording ${G.run + 1} of ${runsFor(G.emitter).length} · ${(OTS.preset(G.tr.preset) || {}).name || ""} sensor`;
        } else {
          roundT.textContent = `Round ${G.round} of ${ROUNDS}`;
          headK.textContent = "Where will the radio hop next?";
          headD.textContent = "Each column is one hop, newest on the right. White dots are what the sensor heard: some real hops are missing and some dots are ghosts. You can't tell which, and neither can OTS.";
        }
        OTS.clear(mini).append("You ", el("b", {}, `${t.you}/${t.n}`), " · ",
          el("span", { class: "challenge-mini-o" }, "OTS ", el("b", {}, `${t.ots}/${t.n}`)), " · Random ≈ ", approx(rnd));
      }
      function renderPad() {
        const rec = G.phase === "reveal" ? G.log[G.log.length - 1] : null;
        keys.forEach((b, ch) => {
          if (!b) return;
          b.disabled = G.phase !== "ask";
          const pick = !!rec && rec.you === ch, real = !!rec && rec.truth === ch;
          b.classList.toggle("is-pick", pick);
          b.classList.toggle("is-real", real);
          b.lastChild.textContent = pick && real ? "you ✓" : real ? "actual" : pick ? "you" : "";
          b.setAttribute("aria-label", `Guess channel ${fmt.ch(ch)}` + (real ? " (the actual channel)" : "") + (pick ? " (your guess)" : ""));
        });
      }
      function status(ok, who) {
        return el("span", { class: "status " + (ok ? "good" : "bad") }, el("span", { class: "ic" }, ok ? "✓" : "✗"), `${who}: ${ok ? "right" : "wrong"}`);
      }
      function renderResult(animate) {
        const tr = G.tr, r = G.round;
        OTS.clear(result);
        nextBtn = null;
        if (G.phase === "ask") {
          result.append(
            el("div", { class: "challenge-res-h" }, "Your turn"),
            el("div", { class: "challenge-res-note" }, `OTS has already made its call for hop ${r + 1}. It stays sealed until you choose.`),
            el("div", { class: "challenge-res-note muted" }, "Tap the channel where you think the radio will land."));
          return;
        }
        const rec = G.log[G.log.length - 1];
        const a = (ms) => ({ class: "challenge-in", style: animate ? `animation-delay:${ms}ms` : "animation:none" });
        const lockNote = T.lockedAt(tr, r)
          ? `OTS had settled on ${quoteName(tr.decision)} at hop ${tr.tau} (95% sure).`
          : "OTS was still working out the pattern.";
        nextBtn = el("button", { class: "btn primary big", type: "button", onclick: onNext }, r >= ROUNDS ? "See the result →" : "Next hop →");
        result.append(
          el("div", a(0), el("div", { class: "challenge-res-h" }, `Hop ${r + 1}`),
            el("div", { class: "challenge-res-big" }, "It hopped to channel ", el("span", { class: "num" }, fmt.ch(rec.truth)))),
          el("div", Object.assign(a(120), { class: "challenge-in challenge-res-row" }), status(rec.youRight, "You"), `picked ${fmt.ch(rec.you)}`),
          el("div", Object.assign(a(260), { class: "challenge-in challenge-res-row" }), status(rec.otsRight, "OTS"),
            rec.ots == null ? "no call" : `called ${fmt.ch(rec.ots)} · ${sure(rec.p)} sure`),
          el("div", Object.assign(a(380), { class: "challenge-in challenge-res-note" }), lockNote, " ",
            el("span", { class: "muted" }, sensorNote(tr, r))),
          el("div", a(420), nextBtn));
        nextBtn.style.width = "100%";
      }
      function renderScore() {
        const t = tally(), rnd = t.n * chance();
        const k = !t.n ? "Can you beat OTS?" : t.ots > t.you ? `OTS leads by ${t.ots - t.you}` : t.you > t.ots ? `You lead by ${t.you - t.ots}` : "Level so far";
        const bump = (who) => G.fresh && t[who] > last[who];
        const tile = (cls, lab, val, sub, b) => el("div", { class: "tile challenge-st" + cls },
          el("div", {}, el("div", { class: "tile-l" }, lab), el("div", { class: "tile-s" }, sub)),
          el("div", { class: "tile-v num" }, el("span", { class: b ? "challenge-bump" : null }, val), el("small", {}, ` / ${t.n}`)));
        const track = (who, key) => el("div", {},
          el("div", { class: "challenge-track-l" }, el("b", {}, who), el("span", {}, `${t[key === "youRight" ? "you" : "ots"]} right`)),
          el("div", { class: "challenge-track", role: "img",
            "aria-label": `${who}, round by round: ` + (G.log.map((e) => (e[key] ? "right" : "wrong")).join(", ") || "no rounds yet") },
          Array.from({ length: ROUNDS }, (_, i) => {
            const e = G.log[i], now = !e && G.phase === "ask" && i + 1 === G.round;
            return el("span", { class: "challenge-cell" + (e ? (e[key] ? " ok" : " no") : "") + (now ? " now" : ""), title: `Round ${i + 1}` + (e ? (e[key] ? ": right" : ": wrong") : "") },
              e ? (e[key] ? "✓" : "✗") : "");
          })));
        // after the game the summary carries the big numbers, so the side card keeps only the round-by-round view
        const done = G.phase === "done";
        OTS.clear(score).append(...[
          el("div", { class: "card-h" }, el("div", {}, el("p", { class: "card-t" }, done ? "Round by round" : "Scoreboard"),
            el("p", { class: "card-k" }, done ? "Where each side scored" : k))),
          done ? null : el("div", { class: "challenge-tiles" },
            tile("", "You", String(t.you), "right guesses", bump("you")),
            tile(" hot", "OTS", String(t.ots), "right calls, made in advance", bump("ots")),
            tile(" is-rnd", "Random guessing", "≈ " + approx(rnd), `1 chance in ${Math.round(1 / chance())} per hop`, false)),
          el("div", { class: "challenge-tracks" + (done ? " is-done" : "") },
            done ? null : el("div", { class: "tile-l" }, "Round by round"),
            track("You", "youRight"), track("OTS", "otsRight"),
            el("div", { class: "challenge-track-s" }, el("span", {}, "Round 1"), el("span", {}, `Round ${ROUNDS}`)),
            el("div", { class: "challenge-track-k" }, done ? "✓ right · ✗ wrong" : "✓ right · ✗ wrong · outlined = this round"))].filter(Boolean));
        last = t;
      }
      function renderSummary() {
        if (sumChart) { sumChart.destroy(); sumChart = null; }
        OTS.clear(summary);
        if (G.phase !== "done") return;
        const tr = G.tr, t = tally(), rnd = t.n * chance();
        const tile = (cls, lab, val, sub) => el("div", { class: "tile" + cls },
          el("div", {}, el("div", { class: "tile-l" }, lab), el("div", { class: "tile-s" }, sub)),
          el("div", { class: "tile-v num" }, val, el("small", {}, ` / ${t.n}`)));
        const [main, ...extra] = explain(tr, G.log);
        const ss = T.sensorStats(tr, ROUNDS + 1);
        const noise = `The sensor was noisy too: over these ${ROUNDS + 1} hops it missed ${pl(ss.miss, "real hop")} and heard ${pl(ss.ghost, "ghost signal")}.`;
        // honest context: the same 20 rounds on every recording of this pattern, and the full test result
        const list = runsFor(G.emitter), all = list.reduce((s, x) => s + otsHits(x), 0);
        const row = OTS.S.byEmitter(PRESET).find((e) => e.id === G.emitter);
        const chartBox2 = el("div");
        const ink = OTS.tok("ink"), acc = OTS.tok("accent"), ink2 = OTS.tok("ink-2");
        const testLen = (OTS.data.scoreboard && OTS.data.scoreboard.window) || G.tr.window;
        const accText = !row ? "" : G.emitter === "csprng"
          ? ` In the full test (${row.n} runs of ${testLen} hops) it correctly answered “unknown” ${fmt.pct0(row.ots_acc)} of the time.`
          : ` In the full test (${row.n} runs of ${testLen} hops) it named this pattern correctly ${fmt.pct0(row.ots_acc)} of the time.`;
        summary.append(
          el("div", { class: "challenge-sumtiles" },
            tile("", "You", String(t.you), "right guesses"),
            tile(" hot", "OTS", String(t.ots), "right calls, made in advance"),
            tile(" is-rnd", "Random guessing", "≈ " + approx(rnd), "average for pure luck")),
          el("div", { class: "challenge-sumgrid" },
            el("div", { class: "challenge-sumtext" },
              el("p", { class: "challenge-explain" }, main),
              el("p", { class: "challenge-explain sub" }, extra.map((x) => [x, " "]), noise),
              el("div", { class: "challenge-sumbtns" },
                el("button", { class: "btn primary big", type: "button", title: "Same pattern, another recorded run", onclick: () => restart(G.emitter, G.run + 1) }, "↻ Play again"),
                el("button", { class: "btn big", type: "button", title: `Next: ${OTS.emitName(otherPattern().id)}`, onclick: () => restart(otherPattern().id, 0) }, "Try another pattern →"))),
            el("div", { class: "challenge-sumchart" },
              OTS.chart.legend([{ label: "OTS", color: acc }, { label: "You", color: ink }, { label: "Random guessing (average)", kind: "dash" }]),
              chartBox2)),
          el("div", { class: "card-foot" },
            `Same ${ROUNDS} rounds on all ${pl(list.length, "recording")} of this pattern with this sensor: OTS called ${all} of ${list.length * ROUNDS} hops (random guessing: about ${approx(list.length * ROUNDS * chance())}).`,
            accText));
        const cum = (key) => { let s = 0; return G.log.map((e) => [e.round, (s += e[key] ? 1 : 0)]); };
        sumChart = OTS.chart.line(chartBox2, {
          height: 210, ariaLabel: `Running score over ${ROUNDS} rounds: OTS ${t.ots}, you ${t.you}, random guessing about ${approx(rnd)}.`,
          x: { label: "Round", min: 1, max: ROUNDS, ticks: [1, 5, 10, 15, 20] },
          y: { label: "Right calls so far", min: 0, max: ROUNDS, ticks: [0, 5, 10, 15, 20] },
          series: [
            { id: "ots", label: "OTS", color: acc, points: cum("otsRight") },
            { id: "you", label: "You", color: ink, points: cum("youRight") },
            { id: "rnd", label: "Random guessing", color: ink2, dash: "6 4", points: G.log.map((e) => [e.round, e.round * chance()]) },
          ],
          tooltipFormat: (v) => (Number.isInteger(v) ? String(v) : "≈ " + v.toFixed(1)),
        });
      }
      function renderAll(animate) {
        const done = G.phase === "done";
        play.hidden = done; summary.hidden = !done; game.classList.toggle("is-done", done);
        renderSetup(); renderHead();
        if (!done) { renderPad(); renderResult(animate); chart.draw(animate); }
        renderSummary(); renderScore();
        G.fresh = false;
      }
      renderAll(false);

      return () => {
        chart.destroy();
        if (sumChart) sumChart.destroy();
        OTS.tip.hide();
      };
    },
  });
})();
