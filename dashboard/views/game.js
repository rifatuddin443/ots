/* Game page: the RL game, visualised.
 *
 *  Two cards, both working offline from data.js and live from the server:
 *   1. The engagement, hop by hop  -- replay a hopper vs the OTS jammer: the true
 *      hop, the jammer's lit tones, denied vs escaped, and the running denial rate.
 *      Switch hopper to watch an oblivious LFSR get crushed and a learned radio escape.
 *   2. The exploitability certificate -- the PSRO curve (exploitability + meta denial)
 *      and the final payoff game with its meta-Nash support. Live: step PSRO and watch
 *      the curve fall.
 */
(function () {
  const G = (window.OTS_DATA && window.OTS_DATA.game) || null;
  const el = OTS.el, svg = OTS.svg, pct = OTS.fmt.pct;

  // --- a small denial colour scale (dark -> warm) -------------------------------
  function heat(v) {
    v = Math.max(0, Math.min(1, v));
    const a = [12, 26, 38], b = [255, 107, 74];              // #0c1a26 -> #ff6b4a
    const c = a.map((x, i) => Math.round(x + (b[i] - x) * v));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  /** Run fn now if the live server already answered, else once it does.
   *  Returns an unsubscribe function for the page cleanup. */
  function whenLive(fn) {
    if (OTS.live) { fn(); return () => {}; }
    const off = OTS.on("live", (on) => { if (on) fn(); });
    return off;
  }

  // ===================================================================== replay
  function replayCard(root) {
    const card = el("div", { class: "card" });
    card.append(el("h2", {}, "The game, hop by hop"),
      el("p", { class: "muted" },
        "The jammer predicts where the radio will hop and lights those tones (shaded). ",
        "A hop on a lit tone is ", el("b", { style: "color:#ff6b4a" }, "denied"),
        "; a clear hop ", el("b", { style: "color:#34d399" }, "escapes"),
        ". Switch the radio to watch an oblivious LFSR get jammed and a learned radio slip away."));

    const duels = (G && G.duels) || [];
    const state = { duel: duels[0] || null, t: 0, playing: false, timer: null, live: false };

    // hopper selector
    const sel = el("div", { class: "seg", role: "group", "aria-label": "Radio" });
    duels.forEach((d, i) => sel.append(el("button", {
      class: "btn" + (i === 0 ? " sel" : ""), type: "button",
      onclick: () => { pick(d, sel.children[i]); },
    }, d.name)));

    const blurb = el("div", { class: "muted", style: "margin:6px 0 10px" });
    const chartWrap = el("div", { class: "gchart" });
    const chartSvg = svg("svg", { class: "hopgrid", preserveAspectRatio: "none" });
    chartWrap.append(chartSvg);

    // transport
    const playBtn = el("button", { class: "btn primary", type: "button",
      onclick: () => (state.playing ? pause() : play()) }, "▶ Play");
    const stepB = el("button", { class: "btn icon", type: "button", title: "Step back",
      onclick: () => { pause(); seek(state.t - 1); } }, "‹‹");
    const stepF = el("button", { class: "btn icon", type: "button", title: "Step forward",
      onclick: () => { pause(); seek(state.t + 1); } }, "››");
    const slider = el("input", { type: "range", min: "0", value: "0", class: "gslider",
      oninput: (e) => { pause(); seek(+e.target.value); } });
    const clock = el("span", { class: "hopclock" }, "hop 0");
    const transport = el("div", { class: "gtransport" }, playBtn, stepB, stepF, slider, clock);

    // live controls
    const liveBar = el("div", { class: "gstat", style: "gap:8px;flex-wrap:wrap" });
    function addLive() {
      if (liveBar.firstChild) return;
      const runBtn = el("button", { class: "btn go", type: "button" }, "⟲ New live run");
      runBtn.onclick = async () => {
        if (!state.duel) return;
        runBtn.disabled = true; runBtn.textContent = "running…";
        try {
          const seed = Math.floor(Math.random() * 1e6);
          const r = await fetch(`api/game/duel?hopper=${state.duel.hopper}&seed=${seed}&window=${state.duel.window}`,
            { cache: "no-store" });
          const tr = await r.json();
          const keep = state.duel.name;
          state.duel = tr; state.duel.name = keep; state.live = true;
          draw(); seek(0);
        } catch (e) { console.error(e); }
        runBtn.disabled = false; runBtn.textContent = "⟲ New live run";
      };
      liveBar.append(runBtn, el("span", { class: "muted" }, "live: a fresh engagement from the Python engine"));
    }
    const offLive = whenLive(addLive);

    // running stats
    const stat = el("div", { class: "gstat" });

    card.append(sel, blurb, chartWrap, transport, stat, liveBar);
    root.append(card);

    function pick(d, btn) {
      [...sel.children].forEach((b) => b.classList.remove("sel"));
      if (btn) btn.classList.add("sel");
      pause(); state.duel = d; state.live = false; draw(); seek(0);
    }

    function draw() {
      const d = state.duel; if (!d) return;
      blurb.textContent = d.blurb || "";
      slider.max = String(d.window - 1);
      drawGrid(chartSvg, d, state.t);
      updateStat();
    }
    function seek(t) {
      const d = state.duel; if (!d) return;
      state.t = Math.max(0, Math.min(d.window - 1, t));
      slider.value = String(state.t);
      clock.textContent = `hop ${state.t} / ${d.window - 1}`;
      drawGrid(chartSvg, d, state.t);
      updateStat();
    }
    function play() {
      if (!state.duel) return;
      if (state.t >= state.duel.window - 1) state.t = 0;
      state.playing = true; playBtn.textContent = "❚❚ Pause";
      state.timer = setInterval(() => {
        if (state.t >= state.duel.window - 1) { pause(); return; }
        seek(state.t + 1);
      }, 180);
    }
    function pause() {
      state.playing = false; playBtn.textContent = "▶ Play";
      if (state.timer) { clearInterval(state.timer); state.timer = null; }
    }
    function updateStat() {
      const d = state.duel; if (!d) return;
      let den = 0, n = 0;
      for (let i = 1; i <= state.t; i++) { n++; if (d.hops[i].den) den++; }
      const frac = n ? den / n : 0;
      const finalFrac = d.summary.denied_frac;
      OTS.clear(stat);
      stat.append(
        statPill("denied so far", `${den}/${n}`, pct(frac), heat(frac)),
        statPill("final denial", "", pct(finalFrac), heat(finalFrac)),
        statPill("jammer next-hop top-1", "", pct(d.summary.nexthop_top1), "#3b82f6"),
        statPill("confidence κ (mean)", "", d.summary.kappa_mean.toFixed(2), "#8b5cf6"));
      // denial bar
      const bar = el("div", { class: "gbar" }, el("div", {
        class: "gbar-fill", style: `width:${(finalFrac * 100).toFixed(1)}%;background:${heat(finalFrac)}` }));
      stat.append(el("div", { class: "gbar-wrap", title: "final fraction of hops denied" }, bar));
    }

    pick(duels[0], sel.children[0]);
    return () => { pause(); offLive(); };        // cleanup
  }

  function statPill(label, sub, big, color) {
    return el("div", { class: "gpill" },
      el("div", { class: "gpill-big", style: `color:${color}` }, big),
      el("div", { class: "gpill-lab" }, label + (sub ? ` (${sub})` : "")));
  }

  // draw the hop grid up to hop t
  function drawGrid(node, d, t) {
    OTS.clear(node);
    const M = d.M, W = d.window;
    const vw = 1000, vh = 300, mL = 26, mT = 8, mB = 14;
    node.setAttribute("viewBox", `0 0 ${vw} ${vh}`);
    const plotW = vw - mL, plotH = vh - mT - mB;
    const cw = plotW / W, rh = plotH / M;
    const X = (i) => mL + i * cw, Y = (c) => mT + (M - 1 - c) * rh;

    // channel axis labels (every 4)
    for (let c = 0; c < M; c += 4) {
      node.append(svg("text", { x: mL - 4, y: Y(c) + rh * 0.72, "text-anchor": "end",
        class: "glab" }, String(c)));
    }
    // faint row separators
    for (let c = 0; c <= M; c += 4) {
      node.append(svg("line", { x1: mL, x2: vw, y1: mT + (M - c) * rh, y2: mT + (M - c) * rh,
        class: "ggrid" }));
    }
    // reveal hops 0..t
    for (let i = 0; i <= t; i++) {
      const h = d.hops[i];
      const x = X(i);
      // lit tones
      for (const lc of h.lit) {
        node.append(svg("rect", { x: x, y: Y(lc), width: Math.max(cw - 0.6, 0.6), height: rh,
          class: "glit" }));
      }
      // true hop marker
      const cx = x + cw / 2, cy = Y(h.ch) + rh / 2;
      node.append(svg("circle", { cx, cy, r: Math.min(cw, rh) * 0.34,
        class: h.den ? "ghop den" : "ghop esc" }));
    }
    // cursor
    const cx = X(t) + cw / 2;
    node.append(svg("line", { x1: cx, x2: cx, y1: mT, y2: mT + plotH, class: "gcursor" }));
  }

  // ================================================================= certificate
  function certCard(root) {
    const card = el("div", { class: "card" });
    card.append(el("h2", {}, "The exploitability certificate (PSRO)"),
      el("p", { class: "muted" },
        "Both sides grow a population of strategies, each a best response to the other. ",
        el("b", {}, "Exploitability"), " — how much a fresh best-response adversary still ",
        "beats the current meta — falls from an exploitable start toward the equilibrium: ",
        "evidence the system is not just winning against adversaries we wrote."));

    const psro = (G && G.psro) || null;
    const curveWrap = el("div", { class: "gcurve" });
    const heatWrap = el("div", { class: "gheat" });
    const grid = el("div", { class: "gcert" }, curveWrap, heatWrap);
    const live = el("div", { class: "gstat", style: "gap:8px;margin-top:10px;flex-wrap:wrap" });
    card.append(grid, live);
    root.append(card);

    let cur = psro ? deepcopy(psro) : null;
    function render() {
      if (!cur) { curveWrap.append(el("div", { class: "empty" }, "No PSRO data.")); return; }
      drawCurve(curveWrap, cur);
      drawHeat(heatWrap, cur);
    }
    render();

    function addLive() {
      if (live.firstChild) return;
      let session = null, busy = false;
      const stepBtn = el("button", { class: "btn go", type: "button" }, "▶ Step PSRO");
      const resetBtn = el("button", { class: "btn", type: "button" }, "⟲ Reset");
      const note = el("span", { class: "muted" }, "live: each step trains fresh best responses (~7 s)");
      stepBtn.onclick = async () => {
        if (busy) return; busy = true; stepBtn.disabled = true;
        const was = stepBtn.textContent; stepBtn.textContent = "training…";
        try {
          if (!session) {
            const r = await fetch("api/game/psro/new", { cache: "no-store" });
            const s = await r.json(); session = s.session;
          }
          const r = await fetch(`api/game/psro/step?session=${session}`, { cache: "no-store" });
          const s = await r.json();
          cur = {
            iterations: s.iterations, exploitability: s.exploitability,
            meta_value: s.meta_value, payoff_matrix: s.payoff_matrix,
            hopper_labels: s.hopper_labels, jammer_labels: s.jammer_labels,
            final_sigma_hopper: s.sigma_hopper, final_sigma_jammer: s.sigma_jammer,
            csprng_floor: cur ? cur.csprng_floor : null,
          };
          render();
          if (s.done) { stepBtn.disabled = true; stepBtn.textContent = "converged"; busy = false; return; }
        } catch (e) { console.error(e); }
        stepBtn.textContent = was; stepBtn.disabled = false; busy = false;
      };
      resetBtn.onclick = () => { session = null; cur = psro ? deepcopy(psro) : null; render();
        stepBtn.disabled = false; stepBtn.textContent = "▶ Step PSRO"; };
      live.append(stepBtn, resetBtn, note);
    }
    return whenLive(addLive);
  }

  function drawCurve(wrap, c) {
    OTS.clear(wrap);
    const it = c.iterations || [];
    const series = [
      { id: "expl", label: "exploitability", color: "#ff6b4a",
        points: it.map((x, i) => [x, c.exploitability[i]]), marker: "all" },
      { id: "meta", label: "meta-Nash denial", color: "#9aa7b4", dash: "5 4",
        points: it.map((x, i) => [x, c.meta_value[i]]), marker: "all" },
    ];
    const maxY = Math.max(0.1, ...c.exploitability, ...c.meta_value) * 1.15 + 0.03;
    const refs = (c.csprng_floor != null && !isNaN(c.csprng_floor))
      ? [{ y: c.csprng_floor, label: "CSPRNG floor" }] : [];
    OTS.chart.line(wrap, {
      height: 260, ariaLabel: "Exploitability and meta-Nash denial vs PSRO iteration",
      x: { label: "PSRO iteration", min: 0, max: Math.max(1, it[it.length - 1] || 1),
        ticks: it, format: (v) => String(Math.round(v)) },
      y: { label: "fraction denied / residual gain", min: 0, max: maxY, format: (v) => v.toFixed(2) },
      series, refLines: refs, endLabels: 2,
    });
  }

  function drawHeat(wrap, c) {
    OTS.clear(wrap);
    const U = c.payoff_matrix || [];
    const hl = c.hopper_labels || [], jl = c.jammer_labels || [];
    if (!U.length || !U[0].length) { wrap.append(el("div", { class: "empty" }, "payoff matrix pending")); return; }
    const sh = c.final_sigma_hopper || [], sj = c.final_sigma_jammer || [];
    const title = el("div", { class: "gheat-title muted" }, "Denial payoff game — ",
      el("span", { style: "color:#3b82f6" }, "blue"), " / ",
      el("span", { style: "color:#34d399" }, "green"), " = meta-Nash support");
    const R = U.length, Cn = U[0].length;
    const vw = 520, vh = 300, mL = 92, mT = 14, mB = 70;
    const node = svg("svg", { class: "heatgrid", viewBox: `0 0 ${vw} ${vh}`, preserveAspectRatio: "xMidYMid meet" });
    const cw = (vw - mL - 8) / Cn, rh = (vh - mT - mB) / R;
    for (let i = 0; i < R; i++) {
      node.append(svg("text", { x: mL - 5, y: mT + i * rh + rh * 0.62, "text-anchor": "end", class: "glab" },
        shorten(hl[i])));
      if (sh[i] > 0.02) node.append(svg("rect", { x: 1, y: mT + i * rh, width: vw - 2, height: rh,
        class: "supp-h" }));
      for (let j = 0; j < Cn; j++) {
        const v = U[i][j];
        node.append(svg("rect", { x: mL + j * cw, y: mT + i * rh, width: cw - 1, height: rh - 1,
          fill: heat(v) }));
        node.append(svg("text", { x: mL + j * cw + cw / 2, y: mT + i * rh + rh * 0.62,
          "text-anchor": "middle", class: "gcell" + (v < 0.55 ? " lite" : " dark") }, v.toFixed(2)));
      }
    }
    for (let j = 0; j < Cn; j++) {
      const tx = mL + j * cw + cw / 2, ty = mT + R * rh + 12;
      node.append(svg("text", { x: tx, y: ty, "text-anchor": "end", class: "glab rot",
        transform: `rotate(-40 ${tx} ${ty})` }, shorten(jl[j])));
      if (sj[j] > 0.02) node.append(svg("rect", { x: mL + j * cw, y: mT, width: cw, height: R * rh,
        class: "supp-j" }));
    }
    wrap.append(title, node);
  }

  function shorten(s) { return (s || "").replace("_knapsack", "").replace("rl_", "rl·").replace("_", " "); }
  function deepcopy(o) { return JSON.parse(JSON.stringify(o)); }

  // ======================================================================== mount
  OTS.views.register({
    id: "game", nav: "The game", title: "The game",
    hint: "Learned radios vs the jammer",
    subtitle: "Watch a learned radio escape the jammer, and the exploitability certificate fall",
    icon: "M12 2a10 10 0 100 20 10 10 0 000-20zM12 8a4 4 0 100 8 4 4 0 000-8zM12 2v3M12 19v3M2 12h3M19 12h3",
    usesPlayer: false,
    mount(root) {
      if (!G) {
        root.append(el("div", { class: "card empty" },
          "No game data — run ", el("code", {}, "python -m dashboard.build"),
          " to record the RL game, or start the live server."));
        return () => {};
      }
      const stopReplay = replayCard(root);
      const stopCert = certCard(root);
      return () => { try { stopReplay && stopReplay(); } catch (e) {} try { stopCert && stopCert(); } catch (e) {} };
    },
  });
})();
