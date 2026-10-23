/* Live demo view: one engagement replayed hop by hop.
 *
 * Layout: a one-sentence narration on top; below it two columns. Left: the hop chart
 * (a sliding window of recent hops plus a shaded "next" column holding OTS's forecast
 * before the hop happens) and a slim next-hop strip. Right: a compact verdict card and
 * the "which pattern is it?" race. On narrow containers everything stacks.
 * Everything renders from OTS.player.trace / OTS.player.t on 'load' and 'tick'.
 */
(function () {
  "use strict";
  const { el, svg, fmt, T } = OTS;
  const P = OTS.player;
  const CALL_P = 0.5;             // a forecast is drawn as a "call" ring from 50% sure

  // ================================================================ styles
  const CSS = `
.live-page { container: live / inline-size; }
.live-narr { display: flex; align-items: baseline; gap: 4px 12px; flex-wrap: wrap; min-height: 50px; }
.live-narr .tag { flex: none; }
.live-narr-x { flex: 1 1 320px; min-width: 0; }
.live-cols { display: grid; grid-template-columns: minmax(0, 1fr) clamp(330px, 36%, 400px); gap: 18px; align-items: start; }
.live-col { display: flex; flex-direction: column; gap: 18px; min-width: 0; }
.live-page .card-h { margin-bottom: 10px; }
.live-legend { min-height: 26px; }
.live-legend .legend { margin: 0; }
.live-hop { position: relative; margin-top: 4px; touch-action: pan-y; }
.live-hop svg { display: block; width: 100%; overflow: visible; }
.live-hop .ax { stroke: var(--axis); stroke-width: 1; }
.live-hop .gl { stroke: var(--grid); stroke-width: 1; opacity: 0.55; }
.live-hop .ax-t { fill: var(--muted); font: 11px var(--mono); }
.live-hop .ax-title { fill: var(--ink-2); font: 600 12px var(--sans); }
.live-hop .nx-t { fill: var(--accent-ink); font: 700 10px var(--mono); letter-spacing: 0.12em; }
.live-hop .lk-t { fill: var(--accent-ink); font: 700 11px var(--mono); letter-spacing: 0.03em; }
.live-hop .q-t { fill: var(--muted); font: 600 12px var(--mono); }
.live-hop .empty-t { fill: var(--muted); font: 13px var(--sans); }
.live-lockfx { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; pointer-events: none; }
.live-lockfx-box {
  text-align: center; background: rgba(5, 8, 14, 0.88); border: 1px solid var(--accent); border-radius: 12px;
  padding: 14px 22px; box-shadow: 0 0 46px rgba(34, 211, 238, 0.35); animation: live-pop 1.8s ease both; max-width: 92%;
}
.live-lockfx-t { font: 700 12px/1.2 var(--mono); letter-spacing: 0.18em; text-transform: uppercase; color: var(--accent-ink); }
.live-lockfx-n { font: 700 26px/1.2 var(--sans); color: var(--ink); margin-top: 4px; letter-spacing: -0.01em; }
.live-lockfx-d { font-size: 13px; color: var(--ink-2); margin-top: 2px; }
.live-lockfx-s { font-size: 13px; color: var(--critical-ink); margin-top: 6px; }
@keyframes live-pop { 0% { opacity: 0; transform: scale(0.86); } 12% { opacity: 1; transform: scale(1.04); } 20% { transform: scale(1); } 82% { opacity: 1; } 100% { opacity: 0; } }

.live-strip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); padding: 13px 4px 14px; }
.live-strip > div { padding: 0 12px; border-left: 1px solid var(--grid); min-width: 0; }
.live-strip > div:first-child { border-left: 0; }
.live-sl { font-size: 12.5px; line-height: 1.3; color: var(--ink-2); }
.live-sv { font-size: 14.5px; color: var(--ink); margin-top: 4px; line-height: 1.3; }
.live-sv b { font: 700 19px/1.1 var(--mono); letter-spacing: 0.01em; }
.live-ss { font-size: 12px; color: var(--muted); margin-top: 3px; line-height: 1.35; }
.live-ok { color: var(--good-ink); font-weight: 600; }
.live-no { color: var(--critical-ink); font-weight: 600; }
.live-soft { color: var(--warning); font-weight: 600; }

.live-vtop { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.live-vname { display: flex; align-items: center; gap: 10px; font: 700 25px/1.15 var(--sans); letter-spacing: -0.01em; margin: 14px 0 4px; }
.live-vname i { width: 12px; height: 12px; border-radius: 3px; flex: none; }
.live-vsub { color: var(--ink-2); font-size: 13.5px; }
.live-vsub b { color: var(--ink); font-weight: 600; }
.live-prog { margin-top: 16px; }
.live-prog-l { display: flex; justify-content: space-between; gap: 8px; font-size: 12.5px; color: var(--ink-2); margin-bottom: 6px; }
.live-prog-tr { position: relative; height: 10px; border-radius: 5px; background: #16233a; }
.live-prog-tr > i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 5px; transition: width 0.3s ease; }
.live-tick { position: absolute; top: -4px; bottom: -4px; width: 2px; background: rgba(232, 238, 247, 0.75); border-radius: 1px; }
.live-sec { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 5px 12px; margin: 16px 0 0; font-size: 13px; }
.live-sec dt { font: 600 10.5px/1.9 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.live-sec dd { margin: 0; font-family: var(--mono); color: var(--ink); overflow-wrap: anywhere; }
.live-match { margin-top: 10px; font-size: 13.5px; }
.live-why { margin-top: 10px; font-size: 12.5px; color: var(--ink-2); }
.live-base { margin-top: 14px; padding-top: 11px; border-top: 1px solid var(--grid); font-size: 13px; color: var(--ink-2); }
.live-base b { color: var(--ink); font-weight: 600; }
.live-verdict.live-flash { animation: live-glow 1.8s ease; }
@keyframes live-glow { 0%, 60% { box-shadow: 0 0 0 1px var(--accent), 0 0 38px rgba(34, 211, 238, 0.35); } 100% { box-shadow: none; } }

.live-rows { position: relative; display: flex; flex-direction: column; gap: 3px; margin-top: 4px; }
.live-rrow {
  position: relative; display: grid; grid-template-columns: 10px 112px minmax(36px, 180px) 40px; gap: 9px;
  align-items: center; padding: 4px 7px; border-radius: 7px; border: 1px solid transparent; transition: opacity 0.25s;
}
.live-rrow .sq { width: 10px; height: 10px; }
.live-rname { font-size: 13px; color: var(--ink-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.live-rtrack { position: relative; height: 8px; border-radius: 4px; background: #16233a; }
.live-rtrack > i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; transition: width 0.25s ease; }
.live-rtrack .live-tick { top: -3px; bottom: -3px; width: 1.5px; background: rgba(232, 238, 247, 0.55); }
.live-rpct { font: 600 12px var(--mono); color: var(--ink-2); text-align: right; font-variant-numeric: tabular-nums; }
.live-rrow.lead { background: rgba(255, 255, 255, 0.04); border-color: var(--line-2); }
.live-rrow.lead .live-rname, .live-rrow.lead .live-rpct { color: var(--ink); font-weight: 600; }
.live-rrow.won { background: var(--accent-soft); border-color: rgba(34, 211, 238, 0.6); }
.live-rrow.won .live-rname, .live-rrow.won .live-rpct { color: var(--ink); font-weight: 700; }
.live-rrow.dim { opacity: 0.42; }
.live-rrow.flip { transition: transform 0.3s ease, opacity 0.25s; }
.live-badge {
  position: absolute; right: 8px; top: -9px; font: 700 9.5px/1 var(--mono); letter-spacing: 0.12em; padding: 3px 6px;
  border-radius: 4px; background: var(--accent); color: #031016;
}
.live-rows.has-badge { margin-top: 12px; }
.live-rscale { display: grid; grid-template-columns: 10px 112px minmax(36px, 180px) 40px; gap: 9px; padding: 0 8px; font: 10px var(--mono); color: var(--muted); }
.live-rscale span { position: relative; height: 12px; }
.live-rscale em { position: absolute; font-style: normal; transform: translateX(-50%); white-space: nowrap; }

@container live (max-width: 780px) {
  .live-cols { grid-template-columns: minmax(0, 1fr); }
  .live-col { display: contents; }
  .live-strip { grid-template-columns: minmax(0, 1fr); padding: 6px 14px; }
  .live-strip > div { border-left: 0; border-top: 1px solid var(--grid); padding: 9px 0; }
  .live-strip > div:first-child { border-top: 0; }
  .live-narr { font-size: 15px; }
}`;
  function injectCss() {
    if (document.getElementById("css-live")) return;
    document.head.append(el("style", { id: "css-live" }, CSS));
  }

  // =============================================================== helpers
  const hid = (id) => (id === "unknown" ? "H0" : id);
  const shortName = (id) => (hid(id) === "H0" ? "Unknown" : OTS.hypName(id));
  const reducedMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  /** "98%" style confidence that never rounds up to a false 100%. */
  const sure = (p) => (p >= 0.995 ? ">99%" : p > 0 && p < 0.005 ? "<1%" : fmt.pct0(p));
  /** One-decimal confidence for headline numbers ("99.6%"). */
  const conf = (p) => (p >= 0.9995 ? ">99.9%" : (p * 100).toFixed(1) + "%");
  const lockLine = (tr) => 1 - (tr.delta || 0.05);
  const article = (name) => (/^([AEIOU]|[FHLMNRSX][A-Z])/.test(name) ? "an " : "a ") + name;   // "an LFSR", "a Tent map"
  const about = (n) => (n < 1 ? "less than 1" : "about " + Math.round(n));

  /** Engagement phase at hop t: brief | search | lock | track | end. */
  function phaseOf(tr, t) {
    if (t <= 0) return "brief";
    if (t >= tr.window) return "end";
    if (T.lockedAt(tr, t)) return t <= tr.tau + 2 ? "lock" : "track";
    return "search";
  }
  /** Calls since the lock, plus how many of the right ones landed on hops the sensor missed. */
  function lockStats(tr, t) {
    const h = T.hitStats(tr, t);
    let unheard = 0;
    if (h.sinceLock) for (let j = tr.tau; j < t; j++) {
      const f = T.forecastTop(tr, j);
      if (f && f.ch === tr.truth[j] && !tr.dets[j].includes(tr.truth[j])) unheard++;
    }
    return { ...h, unheard };
  }
  /** The ghost rate in words: "a fake "ghost" signal about every 3 hops". */
  function ghostWords(tr) {
    const g = OTS.ghostsPerHop(tr);
    if (g <= 0) return "no fake signals";
    const every = Math.round(1 / g);
    return every >= 2 ? `a fake “ghost” signal about every ${every} hops` : `about ${g.toFixed(1)} fake “ghost” signals per hop`;
  }
  const decisionP = (tr, t) => (t > 0 ? tr.post[t - 1][T.idx(tr, tr.decision)] : null);
  const strong = (x) => el("strong", {}, x);

  // ============================================================= narration
  function narration(tr, t) {
    const ph = phaseOf(tr, t), em = OTS.emitter(tr.emitter) || { name: tr.emitter };
    const nExp = tr.labels.length - 1;
    if (ph === "brief") {
      if (tr.emitter === "csprng") return ["BRIEFING", ["Control test: this radio uses an ", strong("unbreakable keyed generator"),
        ". There is no pattern to find, so the honest answer is “unknown”. Watch whether OTS says so."]];
      return ["BRIEFING", ["The radio is secretly driven by ", strong(article(em.name)), ". OTS is not told: it sees only a sensor that catches ",
        strong(Math.round(tr.p_d * 100) + "% of hops"), " and hears ", ghostWords(tr), "."]];
    }
    if (ph === "search") {
      const L = T.leader(tr, t);
      if (L.p < 0.25) return ["SCORING", ["Hop ", String(t), ": OTS checks every detection against all ", String(tr.labels.length),
        " explanations. They are still roughly level; it needs ", strong("95%"), " to lock."]];
      if (L.id === "H0") return ["SCORING", ["Hop ", String(t), ": none of the ", String(nExp), " known patterns fits these hops, so ",
        strong("“unknown”"), " leads (", sure(L.p), "). OTS claims nothing below 95%."]];
      return ["SCORING", ["Hop ", String(t), ": ", strong(OTS.hypName(L.id)), " leads at ", strong(sure(L.p)),
        ". OTS keeps scoring every detection; it locks only at ", strong("95%"), "."]];
    }
    if (ph === "lock") {
      const p = tr.post[tr.tau - 1][T.idx(tr, tr.decision)];
      if (hid(tr.decision) === "H0") return ["LOCKED", ["Locked at hop ", strong(String(tr.tau)), ": OTS is ", strong(conf(p)),
        " sure that no known pattern fits, and says “unknown”."]];
      if (!tr.correct) return ["LOCKED", ["Locked at hop ", strong(String(tr.tau)), " on ", strong(OTS.hypName(tr.decision)),
        ", but that is wrong: the truth is ", strong(em.name), ". One of the rare misidentifications."]];
      // how much of the secret came back, in words that do not overclaim
      const sec = tr.secret_recovered, what = {
        exact: ["and has recovered its secret setting (", strong(sec), ")"],
        equivalent: ["and has found a key that gives the identical hop sequence (", strong(sec), ")"],
        family: ["and has estimated its setting (", strong(sec), ", close to the truth)"],
        model: ["and has fitted its hidden hop table"],
      }[tr.secret_match] || [];
      return ["LOCKED", ["Locked at hop ", strong(String(tr.tau)), ": OTS is ", strong(conf(p)), " sure it is ",
        strong(article(OTS.hypName(tr.decision))), what.length ? [" ", ...what] : null, "."]];
    }
    if (ph === "track") {
      const h = lockStats(tr, t), now = decisionP(tr, t);
      const score = strong(`${h.hit} of ${h.n}`);
      if (now < 0.9 && h.rate != null && h.rate >= 0.6) return ["TRACKING", ["A burst of sensor noise briefly boosts a rival, but OTS keeps calling hops: ",
        score, " right since the lock."]];
      if (h.rate != null && h.rate >= 0.6) return ["TRACKING", ["OTS now calls each hop ", strong("before it happens"), ": ", score,
        " right since the lock", h.unheard ? [", including ", strong(String(h.unheard)), " the sensor never heard"] : null, "."]];
      return ["TRACKING", ["This pattern is only partly predictable, yet OTS calls ", score, " hops right since the lock; random guessing would get ",
        about(h.n / tr.M), "."]];
    }
    // end of the recording
    const b = tr.baseline;
    if (T.lockedAt(tr, t)) {
      const h = lockStats(tr, t);
      if (hid(tr.decision) === "H0") return ["VERDICT", ["Done: OTS concluded “unknown” at hop ", strong(String(tr.tau)),
        tr.correct ? " and made no false claim." : [", but the truth was ", strong(em.name), "."]]];
      if (!tr.correct) return ["VERDICT", ["Done: OTS locked on the wrong pattern (", strong(OTS.hypName(tr.decision)), "; the truth is ",
        strong(em.name), "). Its later calls hit ", strong(`${h.hit} of ${h.n}`), "."]];
      return ["VERDICT", ["Done: OTS named the ", strong(OTS.hypName(tr.decision)), " after ", strong(tr.tau + " hops"), " and called ",
        strong(`${h.hit} of ${h.n}`), " later hops right (random guessing: ", about(h.n / tr.M), ")."]];
    }
    if (hid(tr.decision) === "H0" && tr.correct) return ["VERDICT", ["Verdict: ", strong("“unknown”"),
      ". There is no exploitable pattern, so OTS refuses to guess", b.correct ? "." : [", while the standard test ",
        b.declared ? "wrongly names it " : "leans wrongly towards ", strong(OTS.hypName(b.decision)), "."]]];
    if (hid(tr.decision) === "H0") return ["VERDICT", ["No lock in ", String(tr.window), " hops: OTS says “unknown” rather than risk a false claim (the truth was ",
      strong(em.name), ")."]];
    return ["VERDICT", ["No lock in ", String(tr.window), " hops, so nothing is declared. OTS's best guess, ", strong(OTS.hypName(tr.decision)), ", is ",
      tr.correct ? "right." : "wrong, but it was never claimed."]];
  }

  // ============================================================= hop chart
  /** Sliding-window hop chart. Columns = hops (newest on the right) + one "next" column. */
  function makeHopChart(box, legendBox, onHover) {
    const plot = el("div");
    const fx = el("div", { class: "live-lockfx", hidden: true });
    box.append(plot, fx);
    let W = 0, geo = null, anim = null, slideG = null, nextG = null, hoverCol = null, lastPt = null, legendKey = "", fxTimer = 0;

    function layout(w, M) {
      const small = w < 400;
      const m = { l: small ? 44 : 50, r: 10, t: 34, b: 46 };   // top margin: lock label row + "NEXT" row
      const iw = Math.max(120, w - m.l - m.r);
      const N = clamp(Math.floor(iw / 16) - 1, 8, 24);          // hops in the window
      const colW = iw / (N + 1);
      const ih = Math.round(clamp(220 + iw * 0.2, 250, 390));
      return { m, iw, N, colW, ih, rowH: ih / M, H: m.t + ih + m.b, small };
    }

    function render() {
      const tr = P.trace, w = box.clientWidth;
      if (!tr || !w) return;
      W = w;
      const t = P.t, g = (geo = layout(w, tr.M)), { m, N, colW, ih, rowH, H } = g;
      const first = t - N + 1;                                   // hop number shown in column 0
      const cx = (col) => m.l + (col + 0.5) * colW;
      const cy = (ch) => m.t + ih - (ch + 0.5) * rowH;
      const truthOn = !!OTS.prefs.truth, pathOn = !!OTS.prefs.path;
      const seen = { dot: false, ring: false, ghost: false, miss: false, path: false };
      const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img",
        "aria-label": t ? `Hop chart: hops ${Math.max(1, first)} to ${t}. White dots are sensor detections, cyan rings are OTS's calls, the shaded column is the next hop.`
          : "Hop chart: waiting for the first hop." });

      // y axis: channel gridlines and ticks
      const yStep = g.small || rowH < 15 ? 3 : 1;
      for (let c = 0; c < tr.M; c += yStep) {
        root.append(svg("line", { class: "gl", x1: m.l, x2: m.l + (N + 1) * colW, y1: cy(c), y2: cy(c) }));
        root.append(svg("text", { class: "ax-t", x: m.l - 8, y: cy(c) + 4, "text-anchor": "end" }, fmt.ch(c)));
      }
      root.append(svg("line", { class: "ax", x1: m.l, x2: m.l + (N + 1) * colW, y1: m.t + ih, y2: m.t + ih }));
      root.append(svg("text", { class: "ax-title", x: m.l + g.iw / 2, y: H - 6, "text-anchor": "middle" }, "Hop number (time →)"));
      root.append(svg("text", { class: "ax-title", x: 0, y: 0, "text-anchor": "middle", transform: `translate(13 ${m.t + ih / 2}) rotate(-90)` }, "Channel"));

      // the "next" column: OTS's forecast before the hop happens
      const nx = m.l + N * colW;
      root.append(svg("rect", { x: nx, y: m.t, width: colW, height: ih, fill: "rgba(34, 211, 238, 0.07)" }));
      root.append(svg("line", { x1: nx, x2: nx, y1: m.t, y2: m.t + ih, stroke: "rgba(34, 211, 238, 0.35)", "stroke-dasharray": "3 3" }));
      root.append(svg("text", { class: "nx-t", x: nx + colW / 2, y: m.t - 7, "text-anchor": "middle" }, "NEXT"));
      // hover band (under the marks)
      const band = svg("rect", { x: 0, y: m.t, width: colW, height: ih, fill: "rgba(232, 238, 247, 0.06)", visibility: "hidden" });
      root.append(band);

      nextG = svg("g");
      const ended = t >= tr.window;
      const fNext = !ended ? T.forecastTop(tr, t) : null;
      if (fNext && fNext.p >= CALL_P) {
        seen.ring = true;
        nextG.append(svg("circle", { cx: nx + colW / 2, cy: cy(fNext.ch), r: ringR(g), fill: "rgba(34, 211, 238, 0.16)", stroke: OTS.tok("accent"), "stroke-width": 2 }));
      } else if (!ended) {
        nextG.append(svg("text", { class: "q-t", x: nx + colW / 2, y: m.t + ih / 2 + 4, "text-anchor": "middle" }, "?"));
      }
      root.append(nextG);

      // sliding hops (clipped to the hop area; the extra column at -1 slides out on the left)
      const clipId = "live-clip-" + Math.random().toString(36).slice(2, 8);
      root.append(svg("clipPath", { id: clipId },
        svg("rect", { x: m.l, y: 0, width: N * colW, height: m.t + ih }),                 // marks stop at the "next" column
        svg("rect", { x: m.l - 2, y: m.t + ih, width: (N + 1) * colW + 2, height: m.b })));  // tick labels may use its width
      const clipG = svg("g", { "clip-path": `url(#${clipId})` });
      slideG = svg("g");
      clipG.append(slideG);
      root.append(clipG);
      const xStep = [1, 2, 5, 10].find((s) => s * colW >= 28) || 10;
      const lo = Math.max(1, first - 1), hi = t;
      if (pathOn && hi >= lo) {
        let d = "";
        for (let h = lo; h <= hi; h++) d += (h === lo ? "M" : "L") + cx(h - first).toFixed(1) + " " + cy(tr.truth[h - 1]).toFixed(1);
        slideG.append(svg("path", { d, fill: "none", stroke: "rgba(232, 238, 247, 0.3)", "stroke-width": 1.25, "stroke-linejoin": "round" }));
        seen.path = true;
      }
      const dotR = clamp(Math.min(colW, rowH) * 0.18, 2.6, 3.8);
      const det = OTS.tok("det"), ghostC = OTS.tok("serious"), acc = OTS.tok("accent");
      for (let h = lo; h <= hi; h++) {
        const col = h - first, x = cx(col), j = h - 1, c = tr.truth[j], dets = tr.dets[j];
        const vis = col >= 0;
        if (h % xStep === 0) {
          slideG.append(svg("line", { class: "ax", x1: x, x2: x, y1: m.t + ih, y2: m.t + ih + 4 }));
          slideG.append(svg("text", { class: "ax-t", x, y: m.t + ih + 17, "text-anchor": "middle" }, String(h)));
        }
        const f = T.forecastTop(tr, j);
        if (f && f.p >= CALL_P) {
          slideG.append(svg("circle", { cx: x, cy: cy(f.ch), r: ringR(g), fill: "none", stroke: acc, "stroke-width": 2 }));
          if (vis) seen.ring = true;
        }
        if (truthOn && !dets.includes(c)) {
          slideG.append(svg("circle", { cx: x, cy: cy(c), r: dotR + 2, fill: "none", stroke: "rgba(200, 215, 235, 0.8)", "stroke-width": 1.3, "stroke-dasharray": "2 2" }));
          if (vis) seen.miss = true;
        }
        for (const d of dets) {
          if (truthOn && d !== c) {
            const s = dotR * 0.95, y = cy(d);
            slideG.append(svg("path", { d: `M${x - s} ${y - s}L${x + s} ${y + s}M${x + s} ${y - s}L${x - s} ${y + s}`, stroke: ghostC, "stroke-width": 1.9, "stroke-linecap": "round" }));
            if (vis) seen.ghost = true;
          } else {
            slideG.append(svg("circle", { cx: x, cy: cy(d), r: dotR, fill: det }));
            if (vis) seen.dot = true;
          }
        }
      }
      // the lock: a vertical line right after the lock hop, labelled directly
      if (tr.declared && tr.tau != null && t >= tr.tau) {
        const name = shortName(tr.decision);
        if (tr.tau >= first - 1) {
          const lx = m.l + (tr.tau - first + 1) * colW;
          slideG.append(svg("line", { x1: lx, x2: lx, y1: m.t, y2: m.t + ih, stroke: acc, "stroke-width": 1.6 }));
          const label = "Locked: " + name, lw = label.length * 6.9;
          const left = lx - m.l >= lw + 8;   // label left of the line until it nears the left edge
          slideG.append(svg("text", { class: "lk-t", x: left ? lx - 2 : lx + 2, y: 13, "text-anchor": left ? "end" : "start" }, label));
        } else {
          root.append(svg("text", { class: "lk-t", x: m.l + 2, y: 13 }, `◂ Locked at hop ${tr.tau}: ${name}`));
        }
      }
      if (t === 0) root.append(svg("text", { class: "empty-t", x: m.l + (N * colW) / 2, y: m.t + ih / 2, "text-anchor": "middle" }, "Press Play to start"));

      // slide-in animation state for this frame
      if (anim) { slideG.setAttribute("transform", `translate(${colW} 0)`); nextG.setAttribute("opacity", "0"); }
      OTS.clear(plot).append(root);
      geo.band = band;
      if (hoverCol != null && lastPt) showTip(lastPt);
      renderLegend(seen, truthOn);
    }
    function ringR(g) { return Math.max(5, Math.min(g.colW, g.rowH) * 0.42); }

    function renderLegend(seen, truthOn) {
      const items = [];
      if (seen.dot) items.push({ label: truthOn ? "Real detection" : "Sensor detection", color: OTS.tok("det"), kind: "dot" });
      if (seen.ring) items.push({ label: "OTS's call (at least 50% sure)", kind: "ring" });
      if (seen.ghost) items.push({ label: "Ghost (false alarm)", kind: "x" });
      if (seen.miss) items.push({ label: "Missed hop (true channel)", kind: "hollow" });
      if (seen.path) items.push({ label: "True path", color: "rgba(232, 238, 247, 0.5)", kind: "line" });
      const key = items.map((i) => i.label).join("|");
      if (key === legendKey) return;
      legendKey = key;
      OTS.clear(legendBox).append(items.length ? OTS.chart.legend(items) : "");
    }

    /** Start the one-column slide for a new hop (skipped for fast playback or reduced motion). */
    function slide() {
      const dur = Math.min(260, P.intervalMs() * 0.7);
      anim = reducedMotion() || dur < 90 ? null : { start: performance.now(), dur };
    }
    function frame(now) {
      if (!anim || !slideG || !geo) return;
      const k = clamp((now - anim.start) / anim.dur, 0, 1), e = 1 - Math.pow(1 - k, 3);
      slideG.setAttribute("transform", `translate(${(geo.colW * (1 - e)).toFixed(2)} 0)`);
      nextG.setAttribute("opacity", e.toFixed(3));
      if (k >= 1) { anim = null; slideG.removeAttribute("transform"); nextG.removeAttribute("opacity"); }
    }

    // ---- hover tooltip: one column at a time
    function colAt(ev) {
      const s = plot.firstChild; if (!s || !geo) return null;
      const r = s.getBoundingClientRect(), x = (ev.clientX - r.left) * (W / r.width), y = (ev.clientY - r.top) * (geo.H / r.height);
      if (y < geo.m.t - 14 || y > geo.m.t + geo.ih + 24) return null;
      const col = Math.floor((x - geo.m.l) / geo.colW);
      return col >= 0 && col <= geo.N ? col : null;
    }
    function showTip(pt) {
      const col = colAt(pt);
      hoverCol = col;
      const content = col == null ? null : onHover(col, geo);
      if (!content) { hideTip(); return; }
      geo.band.setAttribute("x", geo.m.l + col * geo.colW);
      geo.band.setAttribute("visibility", "visible");
      OTS.tip.show(pt, content);
    }
    function hideTip() { hoverCol = null; if (geo && geo.band) geo.band.setAttribute("visibility", "hidden"); OTS.tip.hide(); }
    const onMove = (ev) => { lastPt = { clientX: ev.clientX, clientY: ev.clientY }; showTip(lastPt); };
    const onLeave = () => { lastPt = null; hideTip(); };
    box.addEventListener("pointermove", onMove);
    box.addEventListener("pointerdown", onMove);
    box.addEventListener("pointerleave", onLeave);

    const ro = new ResizeObserver(() => { if (Math.abs(box.clientWidth - W) > 0.5) render(); });
    ro.observe(box);

    /** The short celebratory "LOCKED ON" overlay. */
    function lockFx(tr) {
      clearTimeout(fxTimer);
      const em = OTS.emitter(tr.emitter);
      const p = tr.post[tr.tau - 1][T.idx(tr, tr.decision)];
      OTS.clear(fx).append(el("div", { class: "live-lockfx-box", role: "status" },
        el("div", { class: "live-lockfx-t" }, "Locked on"),
        el("div", { class: "live-lockfx-n" }, shortName(tr.decision)),
        el("div", { class: "live-lockfx-d" }, `at hop ${tr.tau} · ${conf(p)} sure`),
        !tr.correct ? el("div", { class: "live-lockfx-s" }, "✗ wrong pattern: the truth is " + (em ? em.name : tr.emitter)) : null));
      fx.hidden = false;
      fxTimer = setTimeout(hideFx, 1800);
    }
    function hideFx() { clearTimeout(fxTimer); fx.hidden = true; OTS.clear(fx); }

    return {
      render, frame, slide, lockFx, hideFx,
      reset() { anim = null; hideFx(); },
      destroy() {
        ro.disconnect(); clearTimeout(fxTimer); hideTip();
        box.removeEventListener("pointermove", onMove); box.removeEventListener("pointerdown", onMove); box.removeEventListener("pointerleave", onLeave);
      },
    };
  }

  /** Tooltip content for one hop column (or the "next" column). */
  function hopTip(col, geo) {
    const tr = P.trace, t = P.t; if (!tr) return null;
    const truthOn = !!OTS.prefs.truth;
    if (col === geo.N) {
      if (t >= tr.window) return { title: "Next hop", rows: [{ value: "—", label: `the recording ended at hop ${tr.window}` }] };
      const f = T.forecastTop(tr, t);
      if (!f) return { title: `Next · hop ${t + 1}`, rows: [{ value: "—", label: "no call yet: OTS needs to hear a hop first" }] };
      return { title: `Next · hop ${t + 1} (not happened yet)`, rows: [
        { value: fmt.ch(f.ch), label: (f.p >= CALL_P ? "OTS's call · " : "best guess · only ") + sure(f.p) + " sure" }] };
    }
    const h = t - geo.N + 1 + col;
    if (h < 1 || h > t) return null;
    const j = h - 1, d = tr.dets[j], c = tr.truth[j], f = T.forecastTop(tr, j);
    const rows = [{ value: d.length ? d.map(fmt.ch).join(", ") : "none", label: d.length > 1 ? `sensor detected (${d.length} signals)` : d.length ? "sensor detected" : "sensor heard nothing" }];
    if (!f) rows.push({ value: "—", label: "no call before the first hop" });
    else rows.push({ value: fmt.ch(f.ch), label: (f.p >= CALL_P ? "OTS's call · " : "best guess · ") + sure(f.p) + " sure" });
    if (truthOn) {
      rows.push({ value: fmt.ch(c), label: d.includes(c) ? "true channel" : "true channel · sensor missed it" });
      const gh = d.filter((x) => x !== c);
      if (gh.length) rows.push({ value: gh.map(fmt.ch).join(", "), label: gh.length > 1 ? "ghosts (false alarms)" : "ghost (false alarm)" });
      if (f) rows.push({ value: f.ch === c ? "✓" : "✗", label: f.ch === c ? "the call was right" : "the call was wrong" });
    }
    const after = tr.declared && tr.tau != null && j >= tr.tau;
    return { title: `Hop ${h}` + (after ? " · after the lock" : ""), rows };
  }

  // ======================================================== next-hop strip
  function cell(label, value, sub) {
    return el("div", {}, el("div", { class: "live-sl" }, label), el("div", { class: "live-sv" }, value), sub ? el("div", { class: "live-ss" }, sub) : null);
  }
  function renderStrip(box, tr, t) {
    const locked = T.lockedAt(tr, t), ended = t >= tr.window, M = tr.M;
    const odds = `random guessing: 1\u00a0in\u00a0${M}`;
    // 1. the call for the hop that has not happened yet
    let c1;
    if (t === 0) c1 = cell("Next hop", "waiting for the first hop", "OTS needs to hear a hop first");
    else if (ended) c1 = cell("Next hop", "recording ended", `after ${tr.window} hops`);
    else {
      const f = T.forecastTop(tr, t);
      c1 = f && (locked || f.p >= CALL_P)
        ? cell(`OTS's call: hop ${t + 1}`, [el("b", {}, "channel " + fmt.ch(f.ch))], `${sure(f.p)} sure`)
        : cell(`Hop ${t + 1}`, "still learning the pattern", f ? `best guess channel ${fmt.ch(f.ch)}, only ${sure(f.p)} sure` : null);
    }
    // 2. how the call for the latest hop turned out
    let c2;
    const f0 = t >= 1 ? T.forecastTop(tr, t - 1) : null;
    if (!f0) c2 = cell(t ? `Hop ${t}` : "Last hop", "no call yet", t ? "there is no call before the first hop" : null);
    else {
      const counted = tr.declared && tr.tau != null && t - 1 >= tr.tau;
      const ok = f0.ch === tr.truth[t - 1];
      if (counted || f0.p >= CALL_P) {
        c2 = cell(`Hop ${t} result`, `called ${fmt.ch(f0.ch)}, it was ${fmt.ch(tr.truth[t - 1])}`,
          ok ? el("span", { class: "live-ok" }, "✓ right")
            : f0.p >= CALL_P ? el("span", { class: "live-no" }, "✗ wrong")
            : [el("span", { class: "live-soft" }, "✗ missed"), ` (only ${sure(f0.p)} sure)`]);
      } else c2 = cell(`Hop ${t}`, ended ? "no confident call" : "no confident call yet", ended ? "OTS never locked" : "calls count once OTS locks");
    }
    // 3. the running tally
    let c3;
    if (locked) {
      const h = lockStats(tr, t);
      c3 = h.n === 0 ? cell("Since the lock", "scoring starts now", odds)
        : cell("Right since the lock", [el("b", {}, `${h.hit} of ${h.n}`)], odds);
    } else if (!ended) c3 = cell("Score", "starts at the lock", odds);
    else {
      const h = T.hitStats(tr, t);
      c3 = cell("No lock: best guesses", [el("b", {}, `${h.hit} of ${h.n}`), " right"], `random guessing would get ${about(h.n / M)}`);
    }
    OTS.clear(box).append(c1, c2, c3);
  }

  // ============================================================== verdict
  const MATCH = {
    exact: ["live-ok", "✓ Exact match: the secret is recovered"],
    equivalent: ["live-ok", "✓ Equivalent key: identical hop sequence"],
    family: ["live-soft", "≈ Right map, neighbouring parameter"],
    model: ["live-ok", "✓ Model fitted (no single secret to recover)"],
    mismatch: ["live-soft", "≠ Mismatch: not the exact true secret"],
  };
  function status(kind, icon, text) {
    return el("span", { class: "status " + kind }, el("span", { class: "ic", "aria-hidden": "true" }, icon), text);
  }
  function baselineLine(tr) {
    const b = tr.baseline;
    return el("div", { class: "live-base" }, "Standard test (no “unknown” option): ",
      b.declared ? "named it " : "leaned towards ", el("b", {}, OTS.hypName(b.decision)),
      b.declared && b.tau ? ` at hop ${b.tau}` : "", " · ",
      b.correct ? el("span", { class: "live-ok" }, "✓ right") : el("span", { class: "live-no" }, "✗ wrong"));
  }
  function renderVerdict(card, tr, t) {
    const ph = phaseOf(tr, t), locked = T.lockedAt(tr, t), em = OTS.emitter(tr.emitter) || { name: tr.emitter };
    const unk = hid(tr.decision) === "H0";
    let st, name, sub, body = [];
    const nameRow = (txt, id) => el("div", { class: "live-vname" }, id ? el("i", { style: `background:${OTS.hypColor(id)}` }) : null, txt);
    if (ph === "brief") {
      st = status("", "◌", "Standby");
      name = nameRow("Ready");
      sub = [`${tr.labels.length} possible explanations are in the running. Press Play (Space) to start.`];
    } else if (locked) {
      const k = T.idx(tr, tr.decision), p = tr.post[tr.tau - 1][k];
      st = unk ? (tr.correct ? status("good", "✓", "Unknown · correct") : status("warn", "◐", "Said unknown"))
        : tr.correct ? status("good", "✓", "Identified") : status("bad", "✗", "Misidentified");
      name = nameRow(unk ? "Unknown: no pattern fits" : OTS.hypName(tr.decision), tr.decision);
      sub = ["locked after ", el("b", {}, tr.tau + " hops"), " · ", el("b", {}, conf(p)), " sure",
        !tr.correct ? [" · the truth is ", el("b", {}, em.name)] : null];
      if (!unk && tr.secret_recovered) {
        body.push(el("dl", { class: "live-sec" },
          el("dt", {}, "Secret recovered"), el("dd", {}, tr.secret_recovered),
          el("dt", {}, "True secret"), el("dd", {}, tr.secret_true)));
        const mt = !tr.correct && tr.secret_match !== "exact" ? ["live-no", "✗ Wrong pattern, so this is not the true secret"] : MATCH[tr.secret_match];
        if (mt) body.push(el("div", { class: "live-match " + mt[0] }, mt[1]));
      }
    } else if (ph === "search") {
      const L = T.leader(tr, t), need = lockLine(tr);
      st = status("live", "◉", "Analysing");
      if (L.p < 0.25) { name = nameRow("Too early to tell"); sub = [`hop ${t}: all explanations are still close`]; }
      else if (L.id === "H0") { name = nameRow("No known pattern fits yet", "H0"); sub = ["“Unknown” leads at ", el("b", {}, sure(L.p)), " · nothing is claimed below 95%"]; }
      else { name = nameRow(OTS.hypName(L.id), L.id); sub = ["leading at ", el("b", {}, sure(L.p)), " · needs 95% to lock"]; }
      body.push(el("div", { class: "live-prog" },
        el("div", { class: "live-prog-l" }, el("span", {}, `Leader's confidence: ${sure(L.p)}`), el("span", {}, `lock line ${Math.round(need * 100)}%`)),
        el("div", { class: "live-prog-tr", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": Math.round(L.p * 100), "aria-label": "Leader's confidence toward the 95% lock line" },
          el("i", { style: `width:${(clamp(L.p, 0, 1) * 100).toFixed(1)}%;background:${OTS.hypColor(L.id)}` }),
          el("span", { class: "live-tick", style: `left:calc(${need * 100}% - 1px)` }))));
    } else {
      // the recording ended without a lock
      if (unk && tr.correct) {
        st = status("good", "✓", "Unknown · correct");
        name = nameRow("No exploitable structure", "H0");
        sub = ["OTS refuses to guess: the right call for ", tr.emitter === "csprng" ? "an unbreakable source." : "this source."];
      } else if (unk) {
        st = status("warn", "◐", "No lock");
        name = nameRow("Reported unknown", "H0");
        sub = ["Not enough evidence in ", el("b", {}, tr.window + " hops"), ", so no claim is made · the truth is ", el("b", {}, em.name)];
      } else {
        st = status("warn", "◐", "No lock");
        name = nameRow(OTS.hypName(tr.decision), tr.decision);
        sub = ["best guess, never declared (it stayed below 95%)"];
        body.push(el("div", { class: "live-match " + (tr.correct ? "live-ok" : "live-soft") },
          tr.correct ? "✓ The best guess is right" : [el("span", { class: "mono" }, "◐"), ` Wrong (the truth is ${em.name}), but never declared: a soft miss`]));
      }
      if (!unk || !tr.correct) body.push(el("dl", { class: "live-sec" }, el("dt", {}, "True secret"), el("dd", {}, tr.secret_true)));
    }
    // the known weak spot, explained in plain words rather than hidden
    if (tr.emitter === "lcg_full" && (ph === "end" || locked) && !tr.correct) body.push(el("div", { class: "live-why" },
      "Why: this pattern is solved from an unbroken run of clean hops (no misses, no ghosts). A noisy sensor rarely delivers one, so this is a known weak spot."));
    // the next-channel predictor keeps calling the next hop even when the bank
    // cannot name the pattern — the whole point for a new algorithm.
    if (t > 0 && tr.fc_pred) {
      const pstat = T.predHitStats(tr, t), chance = 1 / tr.M, newAlgo = T.isNewAlgo(tr);
      if (pstat.n >= 3) {
        const good = pstat.rate >= 1.6 * chance;
        body.push(el("div", { class: "live-match " + (good ? "live-ok" : "live-soft") },
          `Next-hop predictor: ${pstat.hit} of ${pstat.n} calls right (${Math.round(pstat.rate * 100)}%` +
          `, random ${Math.round(chance * 100)}%)` +
          (newAlgo ? " — no expert for this scheme, yet the next hop is still predicted." : "")));
      } else if (newAlgo) {
        body.push(el("div", { class: "live-match live-soft" },
          "New scheme: no expert in the bank — watch the predictor still call the next hop."));
      }
    }
    if (ph === "end") body.push(baselineLine(tr));
    card.classList.toggle("accent", locked);
    OTS.clear(card).append(
      el("div", { class: "live-vtop" }, el("p", { class: "card-t" }, "Verdict"), st),
      name, el("div", { class: "live-vsub" }, sub), ...body,
      el("div", { class: "card-foot" }, OTS.sensorLabel(tr)));
  }

  // ================================================================= race
  /** The hypothesis race: persistent rows, sorted by whole-percent buckets (stable, no jitter). */
  function makeRace(card) {
    const head = el("div", { class: "card-h" }, el("div", {}, el("p", { class: "card-t" }, "Which pattern is it?"), el("p", { class: "card-k" })));
    const rowsBox = el("div", { class: "live-rows" });
    const scale = el("div", { class: "live-rscale", "aria-hidden": "true" });
    const foot = el("div", { class: "card-foot" }, "Bar length = how sure OTS is. First past the 95% line wins.");
    card.append(head, rowsBox, scale, foot);
    let rows = new Map(), order = [], trRef = null;

    function build(tr) {
      trRef = tr; rows = new Map(); OTS.clear(rowsBox);
      const need = lockLine(tr);
      // experts first, "Unknown" last: the natural reading order before any evidence
      order = [...tr.labels.filter((id) => id !== "H0"), ...tr.labels.filter((id) => id === "H0")];
      for (const id of order) {
        const fill = el("i", { style: `background:${OTS.hypColor(id)};width:0` });
        const pct = el("span", { class: "live-rpct" }, "—");
        const row = el("div", { class: "live-rrow", title: `${OTS.hypName(id)}: ${OTS.hypDesc(id)}` },
          el("i", { class: "sq", style: `background:${OTS.hypColor(id)}` }),
          el("span", { class: "live-rname" }, shortName(id)),
          el("span", { class: "live-rtrack" }, fill, el("span", { class: "live-tick", style: `left:calc(${need * 100}% - 0.75px)` })),
          pct);
        rows.set(id, { row, fill, pct, badge: null });
        rowsBox.append(row);
      }
      OTS.clear(scale).append(el("span"), el("span"),
        el("span", {}, el("em", { style: "left:0;transform:none" }, "0%"), el("em", { style: `left:${need * 100}%` }, "95%")), el("span"));
    }

    /** glide: animate a reorder (only for single-hop steps; jumps and reloads snap). */
    function render(tr, t, glide) {
      const fresh = tr !== trRef;
      if (fresh) build(tr);
      const locked = T.lockedAt(tr, t), L = T.leader(tr, t);
      const won = locked ? hid(tr.decision) : null;
      const p = (id) => (t > 0 ? T.post(tr, t, id) : 1 / tr.labels.length);
      // new order: winner pinned first; otherwise by whole percent, ties keep the previous order
      const prev = new Map(order.map((id, i) => [id, i]));
      const next = [...order].sort((a, b) => {
        if (won) { if (a === won) return -1; if (b === won) return 1; }
        return Math.round(p(b) * 100) - Math.round(p(a) * 100) || prev.get(a) - prev.get(b);
      });
      const moved = next.some((id, i) => id !== order[i]);
      const dur = Math.min(300, P.intervalMs() * 0.6);
      // FLIP step 1: where each row is on screen now (including any slide still running)
      const before = moved && glide && !fresh && dur >= 80 && !reducedMotion()
        ? new Map(next.map((id) => [id, rows.get(id).row.getBoundingClientRect().top])) : null;
      if (moved) {
        order = next;
        for (const id of order) { const row = rows.get(id).row; row.classList.remove("flip"); row.style.transform = ""; rowsBox.append(row); }
      }
      for (const id of order) {
        const r = rows.get(id), v = p(id);
        r.fill.style.width = (clamp(v, 0, 1) * 100).toFixed(1) + "%";
        r.pct.textContent = sure(v);
        r.row.classList.toggle("lead", t > 0 && !locked && id === L.id);
        r.row.classList.toggle("won", id === won);
        r.row.classList.toggle("dim", !!won && id !== won);
        if (id === won && !r.badge) { r.badge = el("span", { class: "live-badge" }, "LOCKED"); r.row.append(r.badge); }
        if (id !== won && r.badge) { r.badge.remove(); r.badge = null; }
      }
      rowsBox.classList.toggle("has-badge", !!won);
      // FLIP step 2: start each moved row at its old place, then let it glide to the new one
      if (before) {
        const shifts = order.map((id) => [rows.get(id).row, before.get(id) - rows.get(id).row.getBoundingClientRect().top]).filter(([, dy]) => Math.abs(dy) >= 1);
        for (const [row, dy] of shifts) row.style.transform = `translateY(${dy.toFixed(1)}px)`;
        void rowsBox.offsetWidth;
        for (const [row] of shifts) { row.style.transitionDuration = dur + "ms"; row.classList.add("flip"); row.style.transform = ""; }
      }
      // takeaway headline
      const ph = phaseOf(tr, t);
      let k;
      if (ph === "brief") k = `${tr.labels.length} explanations start level, ${fmt.pct(1 / tr.labels.length)} each.`;
      else if (won) k = won === "H0" ? "“Unknown” passed 95%: no known pattern fits." : `${OTS.hypName(won)} passed 95% and locked.`;
      else if (ph === "end") k = "Nothing reached 95%, so OTS claims nothing.";
      else if (L.p < 0.25) k = "Still level: the evidence is just arriving.";
      else if (L.id === "H0") k = "“Unknown” leads: no known pattern fits yet.";
      else k = `${OTS.hypName(L.id)} leads; 95% locks it in.`;
      head.querySelector(".card-k").textContent = k;
    }
    return { render, reset() { trRef = null; } };
  }

  // ================================================================= view
  OTS.views.register({
    id: "live", nav: "Live demo", title: "Watch it work", hint: "One engagement, hop by hop",
    subtitle: "Name it when it can — predict the next hop always", icon: "M5 3l14 9-14 9V3z", usesPlayer: true,
    mount(root) {
      injectCss();
      const narr = el("div", { class: "callout live-narr" });
      const legendBox = el("div", { class: "live-legend" });
      const hopBox = el("div", { class: "live-hop" });
      const hopFoot = el("div", { class: "card-foot" });
      const hopCard = el("div", { class: "card live-hopcard" },
        el("div", { class: "card-h" }, el("div", {},
          el("p", { class: "card-t" }, "Hop chart"),
          el("p", { class: "card-k" }, "Cyan rings are OTS's calls, made in advance."))),
        legendBox, hopBox, hopFoot);
      const strip = el("div", { class: "card live-strip", "aria-label": "Next hop" });
      const verdict = el("div", { class: "card live-verdict" });
      const raceCard = el("div", { class: "card live-race" });
      root.append(el("div", { class: "page live-page" }, narr,
        el("div", { class: "live-cols" },
          el("div", { class: "live-col" }, hopCard, strip),
          el("div", { class: "live-col" }, verdict, raceCard))));

      const chart = makeHopChart(hopBox, legendBox, hopTip);
      const race = makeRace(raceCard);
      let narrKey = "", footKey = "";

      function renderNarr(tr, t) {
        let tag, parts;
        if (OTS.loadingLive) { tag = "LIVE"; parts = ["Running a brand-new engagement through the Python bank…"]; }
        else [tag, parts] = narration(tr, t);
        if (OTS.tour.active && DATA_TOUR()) tag += ` · DEMO ${OTS.tour.i + 1}/${DATA_TOUR()}`;
        const text = el("span", { class: "live-narr-x" }, parts);
        const key = tag + "|" + text.textContent;
        if (key === narrKey) return;
        narrKey = key;
        OTS.clear(narr).append(el("span", { class: "tag" }, tag), text);
      }
      function renderFoot() {
        const key = OTS.prefs.truth ? "on" : "off";
        if (key === footKey) return;
        footKey = key;
        hopFoot.textContent = OTS.prefs.truth
          ? "Ghosts and missed hops are drawn for you only: OTS never sees them."
          : "Newest hop on the right. Press T to reveal the sensor's mistakes.";
      }
      function renderAll(glide) {
        const tr = P.trace; if (!tr) return;
        const t = P.t;
        renderNarr(tr, t);
        chart.render();
        renderStrip(strip, tr, t);
        renderVerdict(verdict, tr, t);
        race.render(tr, t, !!glide);
        renderFoot();
      }

      const subs = [
        OTS.on("load", () => { chart.reset(); race.reset(); verdict.classList.remove("live-flash"); renderAll(); }),
        OTS.on("tick", (e) => {
          const tr = P.trace;
          if (e && e.animate && e.t === e.prev + 1) chart.slide();
          if (tr && (!tr.declared || P.t < tr.tau)) chart.hideFx();
          renderAll(e && Math.abs(e.t - e.prev) === 1);
        }),
        OTS.on("lock", (tr) => {
          chart.lockFx(tr);
          verdict.classList.remove("live-flash"); void verdict.offsetWidth; verdict.classList.add("live-flash");
        }),
        OTS.on("prefs", () => { chart.render(); renderFoot(); }),
        OTS.on("frame", (now) => chart.frame(now)),
        OTS.on("resize", () => chart.render()),
        OTS.on("tour", () => { const tr = P.trace; if (tr) renderNarr(tr, P.t); }),
        OTS.on("liveloading", () => { const tr = P.trace; if (tr) renderNarr(tr, P.t); }),
      ];
      renderAll();
      return () => {
        for (const off of subs) off();
        chart.destroy();
        OTS.tip.hide();
      };
    },
  });
  const DATA_TOUR = () => ((OTS.data && OTS.data.tour) || []).length;
})();
