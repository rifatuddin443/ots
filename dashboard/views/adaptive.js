/* Adapts & jams view: the Revision-3 layer on top of the identification bank.
 *
 * Three static sections, all computed from OTS_DATA.adaptive (real runs of the
 * Fixed-Share mixture, the regime monitor and the threshold-aware allocator):
 *   1. Regime switch   -- recovery time after the radio changes recipe mid-stream;
 *                         flushing the predictor's memory re-points it fastest.
 *   2. Jam the hops     -- multi-tone power-on-target: the knapsack tracks the best
 *                         allocation, and reactive follow jams even an unbreakable
 *                         radio once it lingers on a channel.
 *   3. What's possible  -- the feasibility map, including the reactive region.
 */
(function () {
  "use strict";
  const { el, svg, fmt } = OTS;

  // Okabe-Ito palette, matching the thesis figures (fhew/experiments/style.py).
  const C = {
    blue: "#4ea3e0", orange: "#e8a33d", green: "#34c79a", vermillion: "#e4714a",
    purple: "#c98bbf", sky: "#7cc7f0", grey: "#8b95a7",
  };
  const REGION = {
    exact_recovery: { c: C.blue, label: "exact recovery" },
    short_horizon_prediction: { c: C.sky, label: "short-horizon prediction" },
    reactive_follower: { c: C.green, label: "reactive / follow" },
    barrage: { c: C.grey, label: "barrage floor" },
  };

  const CSS = `
.adapt-page .how-sec { margin: 26px 2px 6px; }
.adapt-page .how-sec h2 { margin: 0; font-size: 19px; }
.adapt-page .how-sec p { margin: 6px 0 0; color: var(--ink-2); max-width: 70ch; }
.adapt-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 4px 0 2px; }
.adapt-two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 920px) { .adapt-two { grid-template-columns: 1fr; } }
.heat { width: 100%; height: auto; display: block; }
.heat .cell { stroke: rgba(6,18,27,0.65); stroke-width: 1; }
.heat .ax-t { fill: var(--muted); font: 11px var(--mono); }
.heat .ax-title { fill: var(--ink-2); font: 600 12px var(--sans); }
.region-legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-top: 10px; }
.region-legend span { display: inline-flex; align-items: center; gap: 7px; font-size: 12.5px; color: var(--ink-2); }
.region-legend i { width: 13px; height: 13px; border-radius: 3px; display: inline-block; }
`;

  function injectCss() {
    if (!document.getElementById("css-adapt")) {
      document.head.append(el("style", { id: "css-adapt" }, CSS));
    }
  }

  function tile(label, value, sub, status) {
    return el("div", { class: "tile" },
      el("div", { class: "tile-l" }, label),
      el("div", { class: "tile-v num" }, value),
      sub ? el("div", { class: "tile-s" }, sub) : null,
      status ? el("span", { class: "status " + status.kind },
        el("span", { class: "ic" }, status.kind === "good" ? "✓" : "•"), status.text) : null);
  }

  function sectionHead(title, text) {
    return el("div", { class: "how-sec" }, el("h2", {}, title), text ? el("p", {}, text) : null);
  }

  function card(title, kicker, desc, body, pending) {
    const box = el("div");
    if (pending && body) pending.push({ box, spec: body });
    return el("article", { class: "card" },
      el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, title),
        kicker ? el("p", { class: "card-k" }, kicker) : null,
        desc ? el("p", { class: "card-d" }, desc) : null)),
      box);
  }

  // ============================================================ 1. regime switch
  function regimeSection(A, pending) {
    const R = A.regime;
    const alphas = R.alpha_grid;
    const det = R.detection || {};
    const meth = [
      { id: "mix_flush", label: "Mixture + memory flush", color: C.green },
      { id: "mix_weights_only", label: "Mixture, weights-only reset", color: C.orange },
      { id: "bank_alone", label: "Identification bank alone", color: C.vermillion },
    ];
    const series = meth.map((m) => ({
      id: m.id, label: m.label, color: m.color, marker: "all",
      points: R.series[m.id].map((row) => [row.alpha, row.mean_recovery]),
    }));
    // headline speedup: mean over alpha of weights-only / flush
    const flush = R.series.mix_flush.map((r) => r.mean_recovery);
    const wonly = R.series.mix_weights_only.map((r) => r.mean_recovery);
    const ratios = flush.map((v, i) => (v > 0 ? wonly[i] / v : NaN)).filter((x) => !isNaN(x));
    const speedup = ratios.length ? ratios.reduce((s, x) => s + x, 0) / ratios.length : NaN;
    const maxY = Math.max(...meth.flatMap((m) => R.series[m.id].map((r) => r.mean_recovery))) * 1.12;

    const legend = OTS.chart.legend(meth.map((m) => ({ label: m.label, color: m.color })));
    const chartSpec = {
      height: 300,
      ariaLabel: "Hops to recover the next-hop prediction after the radio changes recipe, vs source predictability",
      x: { label: "source determinism (smaller = more predictable)", min: Math.min(...alphas), max: Math.max(...alphas),
           ticks: alphas, format: (v) => v.toFixed(2) },
      y: { label: "hops to recover (lower is better)", min: 0, max: maxY, format: (v) => Math.round(v) },
      tooltipFormat: (v) => Math.round(v) + " hops",
      series,
    };
    return [
      sectionHead("The radio changes recipe mid-contact",
        "A frequency-hopping radio can switch its hop generator part-way through an engagement. Nobody re-points the receiver by hand: the Fixed-Share mixture weights every expert by its recent skill, so the expert that fits the new recipe climbs on its own. But the universal predictor's memory is still full of the old recipe — so when the change monitor fires it must flush that memory, not just reset the weights. This chart measures how many hops each strategy needs to recover its next-hop call after the switch."),
      el("div", { class: "adapt-tiles" },
        tile("Memory flush", "×" + speedup.toFixed(1), "faster recovery than resetting weights alone",
          { kind: "good", text: "re-points the predictor" }),
        tile("Change monitor", fmt.pct0(det.fired_frac), "of switches detected",
          { kind: "good", text: "mean delay " + Math.round(det.mean_post_switch_delay) + " hops" }),
        tile("False alarms", (det.pre_switch_false_alarms_per_episode || 0).toFixed(2),
          "per engagement, before the real switch")),
      card("Recovery after a mid-engagement switch",
        "Flushing the predictor's memory re-points the mixture fastest",
        "Resetting only the mixture weights leaves the universal predictor mispredicting from stale context; the cumulative identification bank, which never forgets, is slowest of all.",
        chartSpec, pending),
      legend,
    ];
  }

  // ============================================================ 2. allocation
  function allocationSection(A, pending) {
    const AL = A.allocation;
    const pa = AL.panelA, pb = AL.panelB;
    const latency = (AL.meta && AL.meta.reaction_latency) || 1;

    // Panel A: reward vs predictability
    const allocMeta = [
      { id: "greedy_knapsack", label: "Greedy knapsack (ours)", color: C.blue, width: 2.6 },
      { id: "weighted_water_filling", label: "Weighted water-filling", color: C.orange, dash: "6 4" },
      { id: "water_filling", label: "Water-filling", color: C.green, dash: "6 4" },
      { id: "equal", label: "Equal power", color: C.grey, dash: "2 4" },
    ];
    const seriesA = allocMeta.map((m) => ({
      id: m.id, label: m.label, color: m.color, width: m.width, dash: m.dash, marker: "all",
      points: pa.allocators[m.id].map((row, i) => [pa.predictability[i], row.mean]),
    }));
    const specA = {
      height: 290,
      ariaLabel: "Power-on-target reward vs adversary predictability, per allocator",
      x: { label: "adversary predictability", min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => v.toFixed(2) },
      y: { label: "power-on-target reward", min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => v.toFixed(2) },
      tooltipFormat: (v) => v.toFixed(2),
      series: seriesA,
    };

    // Panel B: reward vs dwell, reactive vs predict-only, on two unpredictable hoppers
    const dwell = pb.dwell_grid;
    const genMeta = [
      { id: "csprng", label: "Keyed CSPRNG", color: C.vermillion },
      { id: "entropy_max", label: "Entropy-max hopper", color: C.purple },
    ];
    const seriesB = [];
    for (const g of genMeta) {
      const s = pb.series[g.id];
      seriesB.push({ id: g.id + "_react", label: g.label + " · +follow", color: g.color, width: 2.6, marker: "all",
        points: s.knapsack_reactive.map((r) => [r.dwell, r.mean]) });
      seriesB.push({ id: g.id + "_pred", label: g.label + " · predict only", color: g.color, dash: "5 4", opacity: 0.75,
        points: s.knapsack_predict_only.map((r) => [r.dwell, r.mean]) });
    }
    const maxDwell = Math.max(...dwell);
    const specB = {
      height: 290,
      ariaLabel: "Power-on-target reward vs dwell, reactive follow vs predict-only, on unpredictable hoppers",
      x: { label: "dwell (slots per channel)", min: 1, max: maxDwell, ticks: dwell, format: (v) => String(v) },
      y: { label: "power-on-target reward", min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => v.toFixed(2) },
      tooltipFormat: (v) => v.toFixed(2),
      vLines: [{ x: latency, label: "reaction latency" }],
      series: seriesB,
    };

    const legendA = OTS.chart.legend(allocMeta.map((m) => ({ label: m.label, color: m.color, kind: m.id === "greedy_knapsack" ? "line" : "dash" })));
    const legendB = OTS.chart.legend([
      ...genMeta.map((g) => ({ label: g.label, color: g.color })),
      { label: "+follow (solid) vs predict-only (dashed)", kind: "dash" },
    ]);
    return [
      sectionHead("Jamming the hops: a power vector, not a guess",
        "One predicted channel is never enough to act on, so the action is a multi-tone power vector. Because a channel is denied only above a power threshold, the allocation is a knapsack, not a smooth spread: it concentrates on a few channels when the prediction is sharp (spot) and spreads to the barrage floor when it is flat — from one calibrated knob, no mode switch. And a long dwell is a second, separate weapon: detect the hop and jam it where it sits, no prediction needed."),
      el("div", { class: "adapt-two" },
        el("div", {},
          card("Spot ↔ barrage from the predictive",
            "The knapsack tracks the best allocation across the range",
            "Water-filling spreads below the denial threshold and lands nothing when the prediction is flat; the threshold-aware knapsack still denies a few channels (a barrage floor) and concentrates when it can.",
            specA, pending),
          legendA),
        el("div", {},
          card("Reactive follow beats an unbreakable radio",
            "A slow hopper is jammable even when it is unpredictable",
            "Against a keyed CSPRNG prediction is hopeless, but once its dwell exceeds the reaction latency, detect-and-follow jams it anyway. The entropy-max hopper at dwell 1 defeats both — that corner is the true predictability boundary.",
            specB, pending),
          legendB)),
    ];
  }

  // ============================================================ 3. feasibility
  function heatmap(panel, xlabel, ylabel) {
    const grid = panel.grid, ys = panel.rows, xs = panel.cols;
    const W = 440, H = 300, m = { l: 58, r: 14, t: 12, b: 46 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const cw = iw / xs.length, chh = ih / ys.length;
    const root = svg("svg", { class: "heat", viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `${ylabel} vs ${xlabel} feasibility map` });
    grid.forEach((row, i) => {
      row.forEach((regionIdx, j) => {
        const reg = panel.regions[regionIdx];
        root.append(svg("rect", { class: "cell", x: m.l + j * cw, y: m.t + (ys.length - 1 - i) * chh,
          width: cw, height: chh, fill: (REGION[reg] || { c: C.grey }).c }));
      });
    });
    // y ticks (rows) — show a subset to avoid crowding
    const yStep = Math.ceil(ys.length / 7);
    ys.forEach((v, i) => {
      if (i % yStep !== 0 && i !== ys.length - 1) return;
      const y = m.t + (ys.length - 1 - i) * chh + chh / 2 + 3.5;
      root.append(svg("text", { class: "ax-t", x: m.l - 7, y, "text-anchor": "end" }, String(v)));
    });
    const xStep = Math.ceil(xs.length / 7);
    xs.forEach((v, j) => {
      if (j % xStep !== 0 && j !== xs.length - 1) return;
      root.append(svg("text", { class: "ax-t", x: m.l + j * cw + cw / 2, y: H - m.b + 16, "text-anchor": "middle" }, String(v)));
    });
    root.append(svg("text", { class: "ax-title", x: m.l + iw / 2, y: H - 6, "text-anchor": "middle" }, xlabel));
    root.append(svg("text", { class: "ax-title", x: 0, y: 0, "text-anchor": "middle",
      transform: `translate(14 ${m.t + ih / 2}) rotate(-90)` }, ylabel));
    return root;
  }

  function regionLegend(regionIds) {
    return el("div", { class: "region-legend" }, regionIds.map((r) =>
      el("span", {}, el("i", { style: `background:${(REGION[r] || { c: C.grey }).c}` }), (REGION[r] || { label: r }).label)));
  }

  function feasibilitySection(A) {
    const F = A.feasibility;
    const regions = F.meta.regions;
    const pA = { grid: F.panelA.grid, rows: F.panelA.dwells, cols: F.panelA.latencies, regions };
    const pB = { grid: F.panelB.grid, rows: F.panelB.windows, cols: F.panelB.hop_rates, regions };
    const present = (panel) => [...new Set(panel.grid.flat())].sort((a, b) => a - b).map((i) => regions[i]);
    return [
      sectionHead("What's possible, per regime",
        "The feasibility map names the strongest strategy available in each regime, so “what if the radio is fast or unpredictable?” has a principled answer rather than a shrug."),
      el("div", { class: "adapt-two" },
        el("article", { class: "card" },
          el("div", { class: "card-h" }, el("div", {},
            el("p", { class: "card-t" }, "Unpredictable radio (prediction useless)"),
            el("p", { class: "card-d" }, "Even when nothing can be predicted, every regime where the dwell outlasts the reaction latency is still jammable by detect-and-follow. Only the top-left (fast hops, slow reaction) falls to the barrage floor."))),
          heatmap(pA, "reaction latency (slots)", "dwell (slots / channel)"),
          regionLegend(present(pA))),
        el("article", { class: "card" },
          el("div", { class: "card-h" }, el("div", {},
            el("p", { class: "card-t" }, "Breakable radio"),
            el("p", { class: "card-d" }, "With a breakable generator the plane splits the classic three ways: too few hops to work with → barrage; enough to forecast → short-horizon prediction; enough to solve the generator → exact recovery."))),
          heatmap(pB, "hop rate (hops / slot)", "engagement window (slots)"),
          regionLegend(present(pB)))),
    ];
  }

  // ============================================================ register
  OTS.views.register({
    id: "adaptive", nav: "Adapts & jams", title: "Adapts & jams",
    hint: "Keeps up when the radio changes, and jams what it can't predict",
    subtitle: "Re-point on a switch · a power vector, not a guess · what's possible per regime",
    icon: "M3 12h4l3-8 4 16 3-8h4", usesPlayer: false,
    mount(root) {
      injectCss();
      const A = OTS.data.adaptive;
      if (!A) {
        root.append(el("div", { class: "card empty" },
          "No adaptive data found. Rebuild with ", el("code", {}, "python -m dashboard.build"), " and reload."));
        return () => {};
      }
      const pending = [];
      root.append(el("div", { class: "page adapt-page" },
        regimeSection(A, pending),
        allocationSection(A, pending),
        feasibilitySection(A)));
      const charts = pending.map((p) => OTS.chart.line(p.box, p.spec));
      return () => { for (const c of charts) c.destroy(); };
    },
  });
})();
