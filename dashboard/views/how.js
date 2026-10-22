/* How it works view: the method in plain words.
 *
 * A static page (no player):
 *   pipeline     Sense -> Score -> Decide -> Predict, each step with one plain sentence
 *                and what it looked like in one real recorded run
 *   expert bank  one card per hypothesis (OTS.hypotheses): what it looks for and how
 *                many candidate settings it searches (OTS.searchSizes)
 *   safety       the 95% lock rule (a real run's confidence curves) and the "Unknown"
 *                option (the unbreakable-encryption control, from the scoreboard)
 *   fair play    what OTS is given and what only the audience sees
 *   glossary     plain definitions, plus the keyboard shortcuts
 * Results come from OTS_DATA; the fixed numbers are facts about the method.
 */
(function () {
  "use strict";
  const { el, svg, fmt, S, T } = OTS;

  // ================================================================== styles
  const CSS = `
.how-page { gap: 20px; }
.how-hero { padding: 26px 30px 24px; }
.how-hero .lead { font-size: clamp(26px, 2.6vw, 36px); max-width: 32ch; margin-top: 10px; }
.how-hero .intro { font-size: 16.5px; max-width: 74ch; }
.how-sec { margin: 22px 0 -4px; }
.how-sec h2 { margin: 0; font: 650 21px/1.3 var(--sans); letter-spacing: -0.005em; }
.how-sec p { margin: 4px 0 0; color: var(--ink-2); font-size: 14px; max-width: 84ch; }
.how-pipe { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 34px; list-style: none; margin: 0; padding: 0; }
.how-stage { position: relative; display: flex; flex-direction: column; border: 1px solid var(--line); border-radius: var(--radius);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.022), transparent 140px), var(--panel); padding: 18px 18px 16px; min-width: 0; }
.how-stage:not(:last-child)::after { content: "→"; position: absolute; right: -27px; top: 30px; font: 700 20px/1 var(--sans); color: var(--accent); }
.how-stage-h { display: flex; align-items: center; gap: 12px; }
.how-ic { flex: none; width: 40px; height: 40px; border-radius: 11px; display: grid; place-items: center; color: var(--accent);
  border: 1px solid rgba(34, 211, 238, 0.45); background: var(--accent-soft); }
.how-ic svg { width: 21px; height: 21px; }
.how-stage .card-t { margin: 0; }
.how-stage h3 { margin: 2px 0 0; font: 650 18px/1.25 var(--sans); }
.how-stage p { margin: 12px 0 0; font-size: 14px; color: var(--ink-2); line-height: 1.5; }
.how-ex { margin-top: auto !important; padding-top: 12px; }
.how-ex > span { display: block; border-top: 1px solid var(--grid); padding-top: 10px; font-size: 12.5px; color: var(--muted); }
.how-ex b { color: var(--ink); font-weight: 600; }
.how-ex .mono { color: var(--accent-ink); }
.how-nw { white-space: nowrap; }
.how-bank { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.how-exp { display: flex; flex-direction: column; padding: 16px 16px 14px; }
.how-exp-h { display: flex; align-items: center; gap: 10px; }
.how-sw { flex: none; width: 14px; height: 14px; border-radius: 4px; }
.how-exp h3 { margin: 0; font: 650 16px/1.25 var(--sans); }
.how-tag { font: 600 10.5px/1.3 var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-top: 6px; }
.how-exp p { margin: 8px 0 0; font-size: 13.5px; color: var(--ink-2); line-height: 1.5; }
.how-exp-f { margin-top: auto; padding-top: 12px; }
.how-search { display: block; border-top: 1px solid var(--grid); padding-top: 10px; font-size: 13px; color: var(--ink); }
.how-search b { font-weight: 650; }
.how-tech { display: block; font-size: 12px; color: var(--muted); margin-top: 3px; }
.how-rule { display: flex; flex-direction: column; }
.how-rule .card-foot { margin-top: auto; padding-top: 12px; }
.how-out { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 4px; }
.how-out .tile-v { color: var(--ink); }
.how-out .status { margin-top: 10px; white-space: normal; line-height: 1.3; }
.how-out .tile-s b { color: var(--ink); }
.how-fair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px 26px; }
.how-fair h3 { margin: 0 0 8px; display: flex; align-items: center; gap: 10px; font: 650 15.5px/1.3 var(--sans); }
.how-fair h3 svg { width: 20px; height: 20px; flex: none; }
.how-fair .how-given h3 svg { color: var(--accent); }
.how-fair .how-hidden h3 svg { color: var(--muted); }
.how-fair ul { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 8px; }
.how-fair li { position: relative; padding-left: 18px; font-size: 14px; color: var(--ink-2); line-height: 1.5; }
.how-fair li::before { content: ""; position: absolute; left: 2px; top: 9px; width: 6px; height: 6px; border-radius: 50%; background: var(--line-2); }
.how-fair .how-given li::before { background: var(--accent); }
.how-fair li b { color: var(--ink); font-weight: 600; }
.how-fair-col { border: 1px solid var(--line); border-radius: 11px; padding: 14px 16px; background: var(--surface); }
.how-fair-col.how-hidden { border-style: dashed; }
.how-last { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 18px; align-items: start; }
.how-gloss { display: grid; grid-template-columns: minmax(120px, max-content) minmax(0, 1fr); gap: 0 18px; margin: 0; }
.how-gloss dt, .how-gloss dd { padding: 9px 0; border-top: 1px solid var(--grid); margin: 0; }
.how-gloss dt { font-weight: 600; color: var(--ink); font-size: 14px; }
.how-gloss dt small { display: block; font-weight: 400; color: var(--muted); font-size: 12px; }
.how-gloss dd { font-size: 13.5px; color: var(--ink-2); line-height: 1.5; }
.how-keys { width: 100%; border-collapse: collapse; }
.how-keys td { padding: 8px 0; border-top: 1px solid var(--grid); font-size: 13.5px; color: var(--ink-2); vertical-align: top; }
.how-keys td:first-child { width: 96px; white-space: nowrap; padding-right: 12px; }
.how-keys kbd { font-size: 11.5px; padding: 2px 6px; }
.how-keys .how-grp td { border-top: 0; padding: 14px 0 2px; font: 600 10.5px/1.3 var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
@media (max-width: 1280px) {
  .how-bank { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 1100px) {
  .how-pipe { grid-template-columns: minmax(0, 1fr); gap: 30px; }
  .how-stage:not(:last-child)::after { content: "↓"; right: auto; left: 30px; top: auto; bottom: -27px; }
  .how-last { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 860px) {
  .how-hero { padding: 20px 18px 18px; }
  .how-hero .intro { font-size: 15.5px; }
}
@media (max-width: 640px) {
  .how-bank, .how-fair { grid-template-columns: minmax(0, 1fr); }
  .how-out { grid-template-columns: minmax(0, 1fr); }
  .how-gloss { grid-template-columns: minmax(0, 1fr); }
  .how-gloss dd { border-top: 0; padding-top: 0; }
}
`;
  function injectCss() {
    if (document.getElementById("css-how")) return;
    document.head.append(el("style", { id: "css-how" }, CSS));
  }

  // ================================================================ content
  /** Tiny 24x24 stroke icon from one or more path strings. */
  const icon = (...ds) => svg("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 1.9,
    "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" }, ds.map((d) => svg("path", { d })));
  const ICONS = {
    sense: ["M12 20v-8", "M8.5 8.5a5 5 0 0 1 7 0", "M5.6 5.6a9 9 0 0 1 12.8 0", "M12 12h.01"],
    score: ["M4 20V11", "M10 20V5", "M16 20v-6", "M21 20H3"],
    decide: ["M6 11h12v9H6z", "M8.5 11V8a3.5 3.5 0 0 1 7 0v3"],
    predict: ["M12 12m-7 0a7 7 0 1 0 14 0a7 7 0 1 0-14 0", "M12 12m-2.5 0a2.5 2.5 0 1 0 5 0a2.5 2.5 0 1 0-5 0", "M12 2v3", "M12 19v3", "M2 12h3", "M19 12h3"],
    given: ["M5 12.5l4.5 4.5L19 7.5"],
    eye: ["M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z", "M12 12m-2.8 0a2.8 2.8 0 1 0 5.6 0a2.8 2.8 0 1 0-5.6 0"],
  };

  /** What each expert looks for, in plain words (facts about the method, not results). */
  const PLAIN = {
    H0: { tag: "Safety valve", text: "The “none of the above” option. It assumes the hops are pure chance, so it wins whenever no recipe explains them better, for example against real encryption." },
    lfsr: { tag: "Shift register", text: "A tiny circuit that shifts bits along and feeds some back picks each channel. This expert keeps every starting state that still fits what was heard, so missed hops do not throw it off." },
    filtered_lfsr: { tag: "Scrambled shift register", text: "The same kind of circuit, but its output passes through a scrambling rule. This expert spots the faint link between circuit and channel that survives the scrambling." },
    lcg: { tag: "Number recipe", text: "A classic “multiply, add, wrap around” number generator, with only part of each number used as the channel. This expert tries every possible secret starting seed." },
    lcg_full: { tag: "Exposed number recipe", text: "The same “multiply, add” recipe, but the channel is the whole internal number. From an unbroken run of cleanly heard hops it solves for the two secret constants directly." },
    weak_seed: { tag: "Small secret key", text: "A strong scrambler that was started from one of too few possible keys. This expert simply tries every key and keeps the ones that match what was heard." },
    chaos: { tag: "Chaotic formula", text: "Fixed formulas (logistic, tent and Hénon maps) that blow tiny differences up fast. This expert finds which formula and setting fit; such radios are predictable only a few hops ahead." },
    markov: { tag: "Habits, not rules", text: "No fixed rule, just habits: from each channel some next channels are more likely than others. This expert learns those habits while it listens." },
  };
  /** Experts that search nothing: why, in plain words. */
  const NO_SEARCH = {
    H0: "Nothing to search: it is the yardstick the other experts must beat",
    lcg_full: "No search needed: it solves for the secret with algebra",
    markov: "No search needed: it learns the habits by counting as it listens",
  };

  /** The run used for the worked examples: the clearest recorded LFSR run with the typical sensor. */
  function exampleRun() {
    const cands = [{ emitter: "lfsr", preset: "typical" }, ...((OTS.data && OTS.data.tour) || [])];
    for (const c of cands) {
      const list = OTS.tracesFor(c.emitter, c.preset);
      if (!list.length) continue;
      const tr = list[OTS.bestRun(c.emitter, c.preset)];
      if (tr && tr.declared && tr.correct) return tr;
    }
    return null;
  }
  function wmean(rows, f) {
    const ok = rows.filter((r) => r[f] != null), n = ok.reduce((s, r) => s + r.n, 0);
    return n ? ok.reduce((s, r) => s + r[f] * r.n, 0) / n : null;
  }

  // ================================================================= blocks
  function sectionHead(title, text) {
    return el("div", { class: "how-sec" }, el("h2", {}, title), text ? el("p", {}, text) : null);
  }

  function hero() {
    const nHyp = OTS.hypotheses.length;
    return el("section", { class: "card how-hero" },
      el("p", { class: "card-t" }, "The method in plain words"),
      el("h2", { class: "lead" }, `${nHyp} experts listen to the same noisy radio. OTS waits until one of them is `, el("em", {}, "almost certainly right"), "."),
      el("p", { class: "intro" },
        `Each expert stands for one possible recipe behind the radio's hops (or, for one of them, “no recipe at all”). `,
        "After every hop OTS checks how well each expert explains what the sensor heard, allowing for missed hops and ghosts, and only speaks up when one stands out."));
  }

  function pipeline(tr) {
    const k = tr ? T.idx(tr, tr.decision) : -1;
    const ex = (parts) => (tr ? el("p", { class: "how-ex" }, el("span", {}, el("b", {}, "In a real run: "), parts)) : null);
    let senseEx = null, scoreEx = null, decideEx = null, predictEx = null;
    if (tr) {
      const ss = T.sensorStats(tr, tr.tau);
      const heard = tr.dets.slice(0, tr.tau).reduce((s, d) => s + d.length, 0);
      senseEx = ex(`in the first ${tr.tau} hops the sensor reported ${heard} signals: ${ss.miss} real hop${ss.miss === 1 ? " was" : "s were"} missed and ${ss.ghost} signal${ss.ghost === 1 ? " was a ghost" : "s were ghosts"}.`);
      const seq = tr.post.slice(0, tr.tau).map((p, j) => (j === tr.tau - 1 ? fmt.pct(p[k]) : fmt.pct0(p[k])));
      scoreEx = ex([`confidence in ${OTS.hypName(tr.decision)} after hops 1 to ${tr.tau}: `, el("b", { class: "num" }, seq.join(" → ")), "."]);
      decideEx = ex([`at hop ${tr.tau} ${OTS.hypName(tr.decision)} passed 95%, so OTS locked and read off the key `,
        el("span", { class: "how-nw" }, el("span", { class: "mono" }, tr.secret_recovered || "—"), ".")]);
      const h = T.hitStats(tr, tr.window);
      predictEx = ex(`it then called ${h.hit} of the next ${h.n} hops right.`);
    }
    const stage = (n, key, name, title, text, exNode) => el("li", { class: "how-stage" },
      el("div", { class: "how-stage-h" }, el("span", { class: "how-ic" }, icon(...ICONS[key])),
        el("div", {}, el("p", { class: "card-t" }, `${n} · ${name}`), el("h3", {}, title))),
      el("p", {}, text), exNode);
    const nHyp = OTS.hypotheses.length;
    return [
      el("ol", { class: "how-pipe", "aria-label": "The four steps: sense, score, decide, predict" },
        stage(1, "sense", "Sense", "Listen",
          `In every time slot the sensor reports which of the ${(OTS.data && OTS.data.M) || 16} channels seemed busy. Some real hops are missed and some reports are ghosts.`, senseEx),
        stage(2, "score", "Score", "Weigh every explanation",
          `Each of the ${nHyp} experts asks “how well does my recipe explain what was just heard?”, allowing for misses and ghosts. Its confidence rises or falls after every hop.`, scoreEx),
        stage(3, "decide", "Decide", "Lock at 95%",
          "When one expert passes 95% confidence, OTS locks: it names the recipe and reads off the secret settings. Until then it keeps listening.", decideEx),
        stage(4, "predict", "Predict", "Call the next hop",
          "With the recipe and its secret, OTS calls the next channel before the radio gets there. If “Unknown” wins, it makes no claim.", predictEx)),
      tr ? el("p", { class: "card-foot", style: "margin:0" },
        `Real run: ${OTS.emitName(tr.emitter)} radio, ${OTS.sensorLabel(tr).toLowerCase()}, recorded seed ${tr.seed}.`) : null,
    ];
  }

  function bank() {
    const sizes = OTS.searchSizes, total = Object.values(sizes).reduce((s, v) => s + (v || 0), 0);
    const cards = OTS.hypotheses.map((h) => {
      const p = PLAIN[h.id] || { tag: "", text: OTS.hypDesc(h.id) };
      const n = sizes[h.id] || 0;
      const search = n > 0 ? ["Searches ", el("b", { class: "num" }, fmt.int(n)), ` candidate setting${n === 1 ? "" : "s"}`]
        : (NO_SEARCH[h.id] || "No search needed: it works the answer out algebraically or statistically");
      return el("article", { class: "card how-exp" },
        el("div", { class: "how-exp-h" }, el("span", { class: "how-sw", style: `background:${OTS.hypColor(h.id)}`, "aria-hidden": "true" }), el("h3", {}, OTS.hypName(h.id))),
        p.tag ? el("div", { class: "how-tag" }, p.tag) : null,
        el("p", {}, p.text),
        el("div", { class: "how-exp-f" }, el("span", { class: "how-search" }, search),
          h.desc ? el("span", { class: "how-tech" }, "Technical name: " + h.desc) : null));
    });
    return [
      sectionHead("The expert bank",
        `${OTS.hypotheses.length} explanations compete on every hop: ${OTS.hypotheses.length - 1} kinds of radio recipe plus “Unknown”. Together they weigh ${fmt.int(total)} candidate settings at once. The colour of each expert is the same on every page.`),
      el("div", { class: "how-bank" }, cards),
    ];
  }

  function rules(tr, handles) {
    // ---- rule 1: the 95% lock, shown on a real run's confidence curves
    let chartBox = null, legend = null;
    if (tr) {
      const k = T.idx(tr, tr.decision), k0 = 0;
      const xmax = Math.min(tr.window, Math.max(20, tr.tau + 10));
      // the strongest rival: highest confidence any other expert reached in the window
      let rival = -1, best = -1;
      tr.labels.forEach((id, i) => {
        if (i === k || i === k0) return;
        const peak = Math.max(...tr.post.slice(0, xmax).map((p) => p[i]));
        if (peak > best) { best = peak; rival = i; }
      });
      const series = [k, k0, rival].filter((i) => i >= 0).map((i) => ({
        id: tr.labels[i], label: OTS.hypName(tr.labels[i]), color: OTS.hypColor(tr.labels[i]),
        points: tr.post.slice(0, xmax).map((p, j) => [j + 1, p[i]]),
      }));
      legend = OTS.chart.legend([...series.map((s) => ({ label: s.label + (s.id === tr.decision ? " (the right answer)" : s.id === "H0" ? "" : " (closest rival)"), color: s.color })),
        { label: "95% lock line", kind: "dash" }]);
      chartBox = el("div");
      handles.push({ box: chartBox, spec: {
        height: 250, ariaLabel: `Confidence in each explanation over the first ${xmax} hops of a real run; ${OTS.hypName(tr.decision)} passes the 95% line at hop ${tr.tau}`,
        x: { label: "Hop number", min: 1, max: xmax },
        // headroom above 100% keeps the lock label clear of curves that reach the top
        y: { label: "Confidence", min: 0, max: 1.12, ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => Math.round(v * 100) + "%" },
        tooltipFormat: (v) => fmt.pct(v),
        series, refLines: [{ y: 0.95 }], vLines: [{ x: tr.tau, label: `Locked · hop ${tr.tau}` }],
      } });
    }
    const all = S.overall("all");
    const rule1 = el("article", { class: "card how-rule" },
      el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, "Safety rule 1 · the 95% lock"),
        el("p", { class: "card-k" }, "No claim until one expert is at least 95% sure"),
        el("p", { class: "card-d" }, `While experts are still competing, OTS stays quiet and keeps listening. It names a pattern only when one passes 95%; if none ever does, it says so. By design this keeps the chance of a wrong lock to about ${Math.round(((OTS.data && OTS.data.delta) || 0.05) * 100)} in 100 or less.`))),
      legend, chartBox,
      el("p", { class: "card-foot" }, tr ? `A real recorded run (${OTS.emitName(tr.emitter)} radio, ${OTS.preset(tr.preset)?.name.toLowerCase() || ""} sensor). Across all ${fmt.int(all.n)} test runs, dangerous false claims happened ${fmt.per100(all.ots_fb)} times per 100.`
        : `Across all ${fmt.int(all.n)} test runs, dangerous false claims happened ${fmt.per100(all.ots_fb)} times per 100.`));

    // ---- rule 2: "Unknown" is always on the table (the unbreakable-encryption control)
    const cs = S.byEmitter("all").find((r) => r.id === "csprng");
    const other = wmean(S.rows("all").filter((r) => r.emitter !== "csprng"), "ots_unknown");
    const em = OTS.emitter("csprng");
    const rule2 = el("article", { class: "card how-rule" },
      el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, "Safety rule 2 · “Unknown” is always an option"),
        el("p", { class: "card-k" }, "OTS can say “I don't know”. The standard test cannot."),
        el("p", { class: "card-d" }, "The Unknown expert competes like any other. Against a radio with no pattern to find it wins, so OTS makes no claim instead of forcing a guess. The standard test must always name one of the recipes."))),
      cs ? el("div", { class: "how-out" },
        el("div", { class: "tile" }, el("div", { class: "tile-l" }, "OTS"),
          el("div", { class: "tile-v num" }, fmt.pct0(cs.ots_unknown)),
          el("div", { class: "tile-s" }, "of the runs ended in “unknown”"),
          el("span", { class: "status good" }, el("span", { class: "ic" }, "✓"), "the right answer")),
        el("div", { class: "tile" }, el("div", { class: "tile-l" }, "Standard test"),
          el("div", { class: "tile-v num" }, fmt.pct0(cs.base_fb)),
          el("div", { class: "tile-s" }, "of the runs ended in a claimed crack"),
          el("span", { class: "status bad" }, el("span", { class: "ic" }, "✗"), "a false claim"))) : null,
      el("p", { class: "card-foot" }, cs ? [`${cs.n} runs against unbreakable encryption (${em ? em.blurb.replace(/\.$/, "") : "a keyed CSPRNG"}). `,
        other != null ? `The option costs little elsewhere: on the other ${OTS.emitters.length - 1} patterns OTS said “unknown” in only ${fmt.pct0(other)} of runs.` : ""] : ""));
    return [sectionHead("Two safety rules", "Both rules exist to stop the dangerous mistake: confidently claiming to have cracked a radio when it has not."),
      el("div", { class: "grid-2" }, rule1, rule2)];
  }

  function fairPlay() {
    const nPat = OTS.hypotheses.length - 1;
    return [sectionHead("Fair play", null),
      el("article", { class: "card" },
        el("div", { class: "card-h" }, el("div", {},
          el("p", { class: "card-t" }, "What OTS gets, and what it never sees"),
          el("p", { class: "card-k" }, "The algorithm only ever sees the sensor's detections"),
          el("p", { class: "card-d" }, "The true path and the sensor's mistakes are drawn on some charts for the audience only, so you can check the answer. They are never given to OTS."))),
        el("div", { class: "how-fair" },
          el("div", { class: "how-fair-col how-given" },
            el("h3", {}, icon(...ICONS.given), "Given to OTS"),
            el("ul", {},
              el("li", {}, el("b", {}, "The sensor's detections: "), "which channels seemed busy in each time slot, and nothing else about the radio."),
              el("li", {}, el("b", {}, "How good the sensor is: "), "how often it misses a hop and how many ghosts to expect."),
              el("li", {}, el("b", {}, "The list of suspects: "), `${nPat} kinds of radio recipe, plus the “unknown” option.`))),
          el("div", { class: "how-fair-col how-hidden" },
            el("h3", {}, icon(...ICONS.eye), "Shown only to you, the audience"),
            el("ul", {},
              el("li", {}, el("b", {}, "The radio's true path, "), "hop by hop."),
              el("li", {}, el("b", {}, "Which signals were ghosts "), "and which hops the sensor missed."),
              el("li", {}, el("b", {}, "The hidden recipe and its real secret key, "), "used only after the run to mark OTS's answer.")))),
        el("p", { class: "card-foot" }, "The standard test is given exactly the same detections. Every run uses a fresh random secret. In the Live demo the settings switches “Reveal sensor mistakes” and “Show the hidden true path” change only the picture, never what OTS is given."))];
  }

  function glossary() {
    const nHyp = OTS.hypotheses.length, M = (OTS.data && OTS.data.M) || 16;
    const terms = [
      ["Hop", null, "One jump of the radio to a channel. The radio hops at a steady pace, one hop per time slot."],
      ["Channel", null, `One of the ${M} radio frequencies the radio can jump between, numbered ${fmt.ch(0)} to ${fmt.ch(M - 1)}.`],
      ["Detection rate", null, "How many of the radio's real hops the sensor catches. 80% means 1 hop in 5 goes unheard (a “missed hop”)."],
      ["Ghost", "false alarm", "A signal the sensor reports on a channel the radio was not using, caused by noise or other transmitters."],
      ["Expert", "hypothesis", `One possible explanation of the radio's recipe. OTS keeps ${nHyp} of them running side by side.`],
      ["Confidence", "posterior", `How believable each expert is, given everything heard so far. The ${nHyp} confidences always add up to 100%.`],
      ["Lock", null, "The moment one expert passes 95% confidence. OTS then names the pattern and reads off its secret."],
      ["Secret", "seed, key or register state", "The hidden setting that makes one radio's recipe unique. Knowing it lets OTS predict every future hop."],
      ["Unknown", "H₀", "The “none of the above” expert. If it wins, OTS reports that no known recipe fits."],
      ["Standard test", "pairwise SPRT", "The conventional comparison: a sequential test over the same experts and detections, but with no “unknown” option."],
    ];
    const gl = el("article", { class: "card" },
      el("div", { class: "card-h" }, el("div", {}, el("p", { class: "card-t" }, "Glossary"), el("p", { class: "card-k" }, "Words used on these pages"))),
      el("dl", { class: "how-gloss" }, terms.map(([t, aka, d]) => [el("dt", {}, t, aka ? el("small", {}, aka) : null), el("dd", {}, d)])));
    const nViews = OTS.views.list().length;
    const playerPages = OTS.views.list().filter((v) => v.usesPlayer).map((v) => v.nav);
    const key = (...ks) => el("td", {}, ks.map((k, i) => [i ? " " : null, el("kbd", {}, k)]));
    const row = (ks, text) => el("tr", {}, key(...ks), el("td", {}, text));
    const grp = (text) => el("tr", { class: "how-grp" }, el("td", { colspan: 2 }, text));
    const kb = el("article", { class: "card" },
      el("div", { class: "card-h" }, el("div", {}, el("p", { class: "card-t" }, "Keyboard shortcuts"), el("p", { class: "card-k" }, "Drive the demo without a mouse"))),
      el("table", { class: "how-keys" }, el("tbody", {},
        grp("Anywhere"),
        row([`1–${nViews}`], "Switch page (in the order of the side menu)"),
        row(["G"], "Start or stop the guided demo"),
        row(["F"], "Full screen"),
        grp(playerPages.length ? `On the ${playerPages.join(" and ")} pages` : "On player pages"),
        row(["Space"], "Play or pause"),
        row(["←", "→"], "Step one hop back or forward"),
        row(["R"], "Restart the run"),
        row(["N"], "Next recorded run"),
        row(["T"], "Reveal or hide the sensor's mistakes"),
        row(["S"], "Open or close the settings panel"))));
    return [sectionHead("Glossary and shortcuts", null), el("div", { class: "how-last" }, gl, kb)];
  }

  // ================================================================== view
  OTS.views.register({
    id: "how", nav: "How it works", title: "How it works", hint: "The method in plain words",
    subtitle: "Listen, weigh every explanation, lock only when sure, then predict",
    icon: "M12 3l9 5-9 5-9-5 9-5zm-9 9l9 5 9-5M3 16l9 5 9-5", usesPlayer: false,
    mount(root) {
      injectCss();
      const tr = exampleRun(), pending = [];
      root.append(el("div", { class: "page how-page" },
        hero(),
        sectionHead("Four steps, repeated on every hop", null),
        pipeline(tr),
        bank(),
        rules(tr, pending),
        fairPlay(),
        glossary()));
      // charts mount after their containers are in the document (they measure width)
      const charts = pending.map((p) => OTS.chart.line(p.box, p.spec));
      return () => { for (const c of charts) c.destroy(); };
    },
  });
})();
