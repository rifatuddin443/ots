/* How it works view: the whole method in plain words, one slide per section.
 *
 * A static page (no player). Each section is a slide for OTS.present:
 *   idea        one likelihood, four jobs
 *   system      a diagram of the pipeline: sense -> score -> identify / predict -> respond, watched for a change
 *   six steps   Sense, Score, Identify, Predict, Respond, Watch, each with what it did in one real recorded run
 *   bank        one card per identification expert (OTS.hypotheses) and how much it searches
 *   predict     the universal learners that keep prediction going when no expert fits, and the running vote
 *   respond     the predictor's real forecast at three moments of a run, turned into a jam plan
 *   safety      the 95% lock (a real run's confidence curves) and the "Unknown" option
 *   fair play   what OTS is given and what only the audience sees
 *   glossary    plain definitions, plus the keyboard shortcuts
 * Real-run examples come from the recorded traces; the paper numbers from OTS_DATA.paper.
 */
(function () {
  "use strict";
  const { el, svg, fmt, T } = OTS;

  // ================================================================== styles
  const CSS = `
.how-page { gap: 26px; }
.how-hero { padding: 28px 30px 26px; }
.how-hero .lead { font-size: clamp(28px, 2.7vw, 40px); max-width: 30ch; margin-top: 10px; }
.how-hero .intro { font-size: 16.5px; max-width: 80ch; }
.how-jobs { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 20px 0 0; padding: 0; list-style: none; }
.how-jobs li { border: 1px solid var(--line); border-radius: 11px; background: var(--surface); padding: 12px 14px; }
.how-jobs b { display: block; font: 650 15px/1.3 var(--sans); color: var(--ink); }
.how-jobs span { display: block; margin-top: 3px; font-size: 13px; color: var(--ink-2); line-height: 1.45; }
.how-jobs small { display: block; font: 600 10.5px/1.3 var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); margin-bottom: 4px; }
.how-tools { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--grid); }
.how-tools span { font-size: 12.5px; color: var(--muted); }
.how-block { display: flex; flex-direction: column; gap: 14px; }
.how-sec h2 { margin: 0; font: 650 22px/1.3 var(--sans); letter-spacing: -0.01em; }
.how-sec h2 .how-n { font: 700 12px/1 var(--mono); color: var(--accent); letter-spacing: 0.12em; text-transform: uppercase; margin-right: 12px; vertical-align: 3px; }
.how-sec p { margin: 5px 0 0; color: var(--ink-2); font-size: 14.5px; max-width: 90ch; }
.how-dg { padding: 18px 18px 14px; }
.how-dg-wrap { overflow-x: auto; }
.how-dg svg { display: block; width: 100%; min-width: 760px; height: auto; }
.dg-box { fill: var(--surface); stroke: var(--line-2); stroke-width: 1.2; }
.dg-box.out { stroke: rgba(34, 211, 238, 0.6); fill: rgba(34, 211, 238, 0.06); }
.dg-k { fill: var(--accent); font: 600 10.5px var(--mono); letter-spacing: 0.12em; }
.dg-t { fill: var(--ink); font: 650 15px var(--sans); }
.dg-s { fill: var(--ink-2); font: 12.5px var(--sans); }
.dg-e { fill: none; stroke: var(--ink-2); stroke-width: 1.5; }
.dg-e.soft { stroke-dasharray: 3 4; stroke: var(--muted); }
.dg-el { fill: var(--muted); font: 11.5px var(--sans); }
.dg-frame { fill: none; stroke: var(--line-2); stroke-width: 1.3; stroke-dasharray: 7 5; }
.dg-ft { fill: var(--ink-2); font: 12.5px var(--sans); }
.dg-ft tspan.k { fill: var(--accent); font: 600 10.5px var(--mono); letter-spacing: 0.12em; }
.dg-arrow { fill: var(--ink-2); }
.how-pipe { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; list-style: none; margin: 0; padding: 0; }
.how-stage { position: relative; display: flex; flex-direction: column; border: 1px solid var(--line); border-radius: var(--radius);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.022), transparent 140px), var(--panel); padding: 16px 18px 14px; min-width: 0; }
.how-stage-h { display: flex; align-items: center; gap: 12px; }
.how-ic { flex: none; width: 40px; height: 40px; border-radius: 11px; display: grid; place-items: center; color: var(--accent);
  border: 1px solid rgba(34, 211, 238, 0.45); background: var(--accent-soft); }
.how-ic svg { width: 21px; height: 21px; }
.how-stage .card-t { margin: 0; }
.how-stage h3 { margin: 2px 0 0; font: 650 17px/1.25 var(--sans); }
.how-stage p { margin: 10px 0 0; font-size: 14px; color: var(--ink-2); line-height: 1.5; }
.how-ex { margin-top: auto !important; padding-top: 10px; }
.how-ex > span { display: block; border-top: 1px solid var(--grid); padding-top: 9px; font-size: 12.5px; color: var(--muted); }
.how-ex b { color: var(--ink); font-weight: 600; }
.how-ex .mono { color: var(--accent-ink); }
.how-nw { white-space: nowrap; }
.how-foot { margin: 0; color: var(--muted); font-size: 12px; }
.how-bank { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.how-exp { display: flex; flex-direction: column; padding: 15px 16px 13px; }
.how-exp-h { display: flex; align-items: center; gap: 10px; }
.how-sw { flex: none; width: 14px; height: 14px; border-radius: 4px; }
.how-exp h3 { margin: 0; font: 650 16px/1.25 var(--sans); }
.how-tag { font: 600 10.5px/1.3 var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-top: 6px; }
.how-exp p { margin: 7px 0 0; font-size: 13.5px; color: var(--ink-2); line-height: 1.5; }
.how-exp-f { margin-top: auto; padding-top: 11px; }
.how-search { display: block; border-top: 1px solid var(--grid); padding-top: 9px; font-size: 13px; color: var(--ink); }
.how-search b { font-weight: 650; }
.how-tech { display: block; font-size: 12px; color: var(--muted); margin-top: 3px; }
.how-uni { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 16px; align-items: stretch; }
.how-uni-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.how-uni .how-exp .how-sw { background: var(--accent); }
.how-vote { display: flex; flex-direction: column; gap: 12px; }
.how-vote .tile-v { color: var(--accent-ink); }
.how-vote .tile-s b { color: var(--ink); }
.how-fc { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.how-fc-c { display: flex; flex-direction: column; }
.how-fc-c h3 { margin: 0; font: 650 16px/1.3 var(--sans); }
.how-fc-c .card-t { margin: 0 0 4px; }
.how-fc-c p { margin: 4px 0 0; font-size: 13px; color: var(--ink-2); }
.how-fc svg { display: block; width: 100%; height: auto; margin-top: 10px; }
.fc-bar { fill: rgba(168, 181, 199, 0.28); }
.fc-bar.in { fill: var(--accent); }
.fc-ax { stroke: var(--axis); stroke-width: 1; }
.fc-t { fill: var(--muted); font: 10px var(--mono); }
.fc-true { fill: var(--ink); }
.fc-tl { fill: var(--ink); font: 600 10.5px var(--sans); }
.how-fc-leg { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--ink-2); }
.how-fc-leg span { display: inline-flex; align-items: center; gap: 7px; }
.how-fc-leg i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; }
.how-price { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.how-price .tile-v small { font-size: 15px; }
.how-rule { display: flex; flex-direction: column; }
.how-rule .card-foot { margin-top: auto; padding-top: 12px; }
.how-out { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 4px; }
.how-out .tile-v { color: var(--ink); }
.how-out .status { margin-top: 10px; white-space: normal; line-height: 1.3; }
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
.how-gloss dt, .how-gloss dd { padding: 8px 0; border-top: 1px solid var(--grid); margin: 0; }
.how-gloss dt { font-weight: 600; color: var(--ink); font-size: 14px; }
.how-gloss dt small { display: block; font-weight: 400; color: var(--muted); font-size: 12px; }
.how-gloss dd { font-size: 13.5px; color: var(--ink-2); line-height: 1.5; }
.how-keys { width: 100%; border-collapse: collapse; }
.how-keys td { padding: 7px 0; border-top: 1px solid var(--grid); font-size: 13.5px; color: var(--ink-2); vertical-align: top; }
.how-keys td:first-child { width: 96px; white-space: nowrap; padding-right: 12px; }
.how-keys kbd { font-size: 11.5px; padding: 2px 6px; }
.how-keys .how-grp td { border-top: 0; padding: 14px 0 2px; font: 600 10.5px/1.3 var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
@media (max-width: 1280px) {
  .how-bank { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .how-uni { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 1100px) {
  .how-pipe { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .how-last { grid-template-columns: minmax(0, 1fr); }
  .how-jobs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 860px) {
  .how-hero { padding: 20px 18px 18px; }
  .how-hero .intro { font-size: 15.5px; }
  .how-uni-grid, .how-fc { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 640px) {
  .how-pipe, .how-bank, .how-fair, .how-out, .how-jobs { grid-template-columns: minmax(0, 1fr); }
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
    respond: ["M4 18h2V9H4z", "M9 18h2V5H9z", "M14 18h2v-6h-2z", "M19 18h2v-3h-2z", "M3 21h18"],
    watch: ["M3 12h4l3-7 4 14 3-7h4"],
    given: ["M5 12.5l4.5 4.5L19 7.5"],
    eye: ["M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z", "M12 12m-2.8 0a2.8 2.8 0 1 0 5.6 0a2.8 2.8 0 1 0-5.6 0"],
  };

  /** What each identification expert looks for, in plain words (facts about the method, not results). */
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
  /** The predictor's extra experts: they name nothing, they only forecast. */
  const UNIVERSAL = [
    { name: "Context mixing", tag: "Habits of any length", text: "Learns which channel tends to follow the last one, two, three or four, all at once. Catches longer rules that a one-step habit counter misses." },
    { name: "Rule search", tag: "Thousands of small formulas", text: "Tries short formulas side by side (add, multiply, XOR, rotate, polynomials, lookup tables) and keeps those that still fit. Starts over if the rule changes." },
    { name: "Seed beam", tag: "Big key search", text: "Searches the whole key space of weak-seed and congruential generators, dropping each key the moment it contradicts a cleanly heard hop." },
    { name: "Wide Berlekamp–Massey", tag: "Long shift registers", text: "Fits shift registers up to 32 bits long from runs of cleanly heard hops, and drops a fit after two contradictions." },
    { name: "Phase space", tag: "Unseen chaos", text: "Forecasts chaotic maps outside the chaos expert's list from the shape of the recent hops; useful a few hops ahead." },
    { name: "Barrage floor", tag: "The honest floor", text: "The flat guess: every channel equally likely. Against real encryption nothing beats it, and the vote then says so." },
  ];

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
  /** Smallest set of top-ranked channels whose forecast weight reaches level (permille forecast). */
  function coverSet(f, level = 0.9) {
    const order = f.map((v, c) => c).sort((a, b) => f[b] - f[a]);
    const out = []; let cum = 0;
    for (const c of order) { out.push(c); cum += f[c] / 1000; if (cum >= level - 1e-9) break; }
    return out;
  }
  const P = () => OTS.data && OTS.data.paper;
  const M = () => (OTS.data && OTS.data.M) || 16;
  const d1 = (v) => String(Math.round(v * 10) / 10);
  const pct0 = (p) => (p == null || isNaN(p) ? "—" : Math.round(p * 100 + 1e-9) + "%");

  // ================================================================= blocks
  function sectionHead(n, title, text) {
    return el("div", { class: "how-sec" }, el("h2", {}, n ? el("span", { class: "how-n" }, n) : null, title), text ? el("p", {}, text) : null);
  }
  const block = (slide, ...kids) => el("section", { class: "how-block", "data-slide": slide }, kids);

  function hero() {
    const nExp = OTS.hypotheses.length - 1;
    return el("section", { class: "card how-hero", "data-slide": "The idea" },
      el("p", { class: "card-t" }, "The method in plain words"),
      el("h2", { class: "lead" }, "One noisy sensor, one way to score it, ", el("em", {}, "four jobs"), "."),
      el("p", { class: "intro" },
        "A passive receiver hears a frequency-hopping radio through a sensor that misses hops and hears ghosts. OTS turns each slot of detections into one score that expects those mistakes, ",
        "then reuses that score everywhere: to name the radio's recipe, to call its next channel, to decide how many channels to jam, and to notice when the radio changes recipe."),
      el("ol", { class: "how-jobs", "aria-label": "The four jobs" },
        el("li", {}, el("small", {}, "Identify"), el("b", {}, "Name the recipe"), el("span", {}, `${nExp} experts and “Unknown” compete; OTS names one only when it is 95% sure.`)),
        el("li", {}, el("small", {}, "Predict"), el("b", {}, "Call the next channel"), el("span", {}, "Even when no expert fits: universal learners pick up a new rule from the hops.")),
        el("li", {}, el("small", {}, "Respond"), el("b", {}, "Size the jam"), el("span", {}, "A set of channels that covers the next hop with a promised probability.")),
        el("li", {}, el("small", {}, "Watch"), el("b", {}, "Notice a change"), el("span", {}, "If the radio switches recipe, OTS dates the switch and re-learns."))),
      el("div", { class: "how-tools no-present" }, OTS.present.button("Present"), el("span", {}, "One section per slide · ← → to move · Esc to leave")));
  }

  /** The pipeline as a diagram: sense -> score -> identify / predict -> respond, inside the box the tracker watches. */
  function diagram() {
    const nExp = OTS.hypotheses.length - 1;
    const box = (x, y, w, h, k, t, s, out) => [
      svg("rect", { class: "dg-box" + (out ? " out" : ""), x, y, width: w, height: h, rx: 10 }),
      svg("text", { class: "dg-k", x: x + 14, y: y + 22 }, k),
      svg("text", { class: "dg-t", x: x + 14, y: y + 44 }, t),
      ...[].concat(s).map((line, i) => svg("text", { class: "dg-s", x: x + 14, y: y + 63 + i * 16 }, line)),
    ];
    const edge = (d, soft) => svg("path", { class: "dg-e" + (soft ? " soft" : ""), d, "marker-end": "url(#how-arrow)" });
    const root = svg("svg", { viewBox: "0 0 1160 340", role: "img",
      "aria-label": "The OTS pipeline: sensor detections are scored by one sensor-aware likelihood, which feeds the identification bank and the next-channel predictor; the predictor's channel set becomes the jam plan; a switch tracker watches the whole decision box and resets it when the radio changes recipe." },
      svg("defs", {}, svg("marker", { id: "how-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
        svg("path", { class: "dg-arrow", d: "M0 0L10 5L0 10z" }))),
      svg("rect", { class: "dg-frame", x: 446, y: 8, width: 706, height: 318, rx: 14 }),
      ...box(10, 128, 176, 98, "1 · SENSE", "Detections", ["each slot, 16 channels", "misses hops, hears ghosts"]),
      ...box(220, 128, 196, 98, "2 · SCORE", "One likelihood", ["expects misses and ghosts", "shared by every expert"]),
      ...box(470, 26, 236, 98, "3 · IDENTIFY", `${nExp} experts + “Unknown”`, ["confidence after every hop", "lock only at 95%"]),
      ...box(470, 164, 236, 98, "4 · PREDICT", "Experts + universal learners", ["a running vote trusts", "whoever predicts best"]),
      ...box(746, 26, 184, 98, "OUTPUT", "Name + secret", ["or “unknown”: no", "recipe fits"], true),
      ...box(746, 164, 184, 98, "5 · RESPOND", "Channel set", ["covers the next hop", "with a promised 90%"], true),
      ...box(964, 164, 176, 98, "OUTPUT", "Jam plan", ["one tone per channel", "size set by the data"], true),
      edge("M186 177 H214"),
      edge("M416 177 H436 V75 H464"), edge("M416 177 H436 V213 H464"),
      edge("M706 75 H740"), edge("M706 213 H740"), edge("M930 213 H958"),
      edge("M588 124 V158", true),
      svg("text", { class: "dg-el", x: 596, y: 146 }, "experts join the vote"),
      svg("text", { class: "dg-ft", x: 470, y: 300 },
        svg("tspan", { class: "k" }, "6 · WATCH  "),
        svg("tspan", {}, "A switch tracker watches this box and resets it when the radio changes recipe.")));
    return block("The system at a glance",
      sectionHead(null, "The system at a glance", "Everything downstream of the sensor reads the same score, so a missed hop or a ghost is handled once, in one place, and every part benefits."),
      el("div", { class: "card how-dg" }, el("div", { class: "how-dg-wrap" }, root)));
  }

  function pipeline(tr) {
    const k = tr ? T.idx(tr, tr.decision) : -1;
    const ex = (lead, parts) => (parts ? el("p", { class: "how-ex" }, el("span", {}, el("b", {}, lead), parts)) : null);
    const pp = P();
    let senseEx = null, scoreEx = null, decideEx = null, predictEx = null, respondEx = null, watchEx = null;
    if (tr) {
      const ss = T.sensorStats(tr, tr.tau);
      const heard = tr.dets.slice(0, tr.tau).reduce((s, d) => s + d.length, 0);
      senseEx = ex("In a real run: ", `in the first ${tr.tau} hops the sensor reported ${heard} signals: ${ss.miss} real hop${ss.miss === 1 ? " was" : "s were"} missed and ${ss.ghost} signal${ss.ghost === 1 ? " was a ghost" : "s were ghosts"}.`);
      const seq = tr.post.slice(0, tr.tau).map((p, j) => (j === tr.tau - 1 ? fmt.pct(p[k]) : fmt.pct0(p[k])));
      scoreEx = ex("In a real run: ", [`confidence in ${OTS.hypName(tr.decision)} after hops 1 to ${tr.tau}: `, el("b", { class: "num" }, seq.join(" → ")), "."]);
      decideEx = ex("In a real run: ", [`at hop ${tr.tau} ${OTS.hypName(tr.decision)} passed 95%, so OTS locked and read off the key `,
        el("span", { class: "how-nw" }, el("span", { class: "mono" }, tr.secret_recovered || "—"), ".")]);
      const h = T.hitStats(tr, tr.window);
      predictEx = ex("In a real run: ", `it then called ${h.hit} of the next ${h.n} hops right.`);
      if (tr.fc_pred) {
        const first = coverSet(tr.fc_pred[0]).length, later = coverSet(tr.fc_pred[Math.min(tr.fc_pred.length - 1, tr.tau + 3)]).length;
        respondEx = ex("In a real run: ", `after hop 1 it needed ${first} of ${M()} channels to cover 90% of its forecast; after hop ${tr.tau + 4}, ${later === 1 ? "one" : later}.`);
      }
    }
    if (pp && pp.switch && pp.switch.by_sensor.typical) {
      const s = pp.switch.by_sensor.typical;
      watchEx = ex("In the paper's tests: ", `${pct0(s.detect_rate)} of switches caught on the Typical sensor, a median ${d1(s.delay_median)} hops after they happened.`);
    }
    const stage = (n, key, name, title, text, exNode) => el("li", { class: "how-stage" },
      el("div", { class: "how-stage-h" }, el("span", { class: "how-ic" }, icon(...ICONS[key])),
        el("div", {}, el("p", { class: "card-t" }, `${n} · ${name}`), el("h3", {}, title))),
      el("p", {}, text), exNode);
    const nHyp = OTS.hypotheses.length;
    return block("Six steps, every hop",
      sectionHead(null, "Six steps, repeated on every hop", tr ? `Each step with what it did in one real recorded run: ${OTS.emitName(tr.emitter)} radio, ${OTS.sensorLabel(tr).toLowerCase()}, seed ${tr.seed}.` : null),
      el("ol", { class: "how-pipe", "aria-label": "The six steps" },
        stage(1, "sense", "Sense", "Listen",
          `In every time slot the sensor reports which of the ${M()} channels seemed busy. Some real hops are missed and some reports are ghosts.`, senseEx),
        stage(2, "score", "Score", "Weigh every explanation",
          `Each of the ${nHyp} explanations asks “how well does my recipe explain what was just heard?” through one score that expects misses and ghosts. A missed hop removes evidence; it never counts against the truth.`, scoreEx),
        stage(3, "decide", "Identify", "Lock at 95%",
          "When one expert passes 95% confidence, OTS names the recipe and reads off its secret. If no recipe fits, “Unknown” wins and OTS says so instead of guessing.", decideEx),
        stage(4, "predict", "Predict", "Call the next channel",
          "Every expert forecasts the next channel and a running vote weights them by how well they have been predicting. Universal learners keep it working on recipes no expert knows.", predictEx),
        stage(5, "respond", "Respond", "Size the jam",
          "Instead of one guess, OTS names the smallest set of channels that covers the next hop with a promised probability, and jams those: one channel when sure, many when not.", respondEx),
        stage(6, "watch", "Watch", "Notice a change",
          "A switch tracker watches for the moment the hops stop fitting. It dates the change, clears the old evidence and lets every part re-learn.", watchEx)));
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
    return block("Identify: the expert bank",
      sectionHead("Step 3", "Identify: the expert bank",
        `${OTS.hypotheses.length} explanations compete on every hop: ${OTS.hypotheses.length - 1} kinds of radio recipe plus “Unknown”. Together they weigh ${fmt.int(total)} candidate settings at once. The colour of each expert is the same on every page.`),
      el("div", { class: "how-bank" }, cards));
  }

  function universal() {
    const pp = P(), st = pp && pp.predict && pp.predict.by_sensor.typical, lf = st && st.lagged_fib;
    const lstm = pp && pp.pred_baselines && pp.pred_baselines.by_sensor.typical && pp.pred_baselines.by_sensor.typical.gens.lagged_fib;
    const cards = UNIVERSAL.map((u) => el("article", { class: "card how-exp" },
      el("div", { class: "how-exp-h" }, el("span", { class: "how-sw", "aria-hidden": "true" }), el("h3", {}, u.name)),
      el("div", { class: "how-tag" }, u.tag), el("p", {}, u.text)));
    const tile = (l, v, s) => el("div", { class: "tile" }, el("div", { class: "tile-l" }, l), el("div", { class: "tile-v num" }, v), el("div", { class: "tile-s" }, s));
    return block("Predict: when no expert fits",
      sectionHead("Step 4", "Predict: when no expert fits",
        "Naming the recipe is not needed to call the next channel. The predictor carries the bank's experts plus universal learners that name nothing and only forecast, and lets them vote."),
      el("div", { class: "how-uni" },
        el("div", { class: "how-uni-grid" }, cards),
        el("div", { class: "how-vote" },
          el("article", { class: "card" },
            el("p", { class: "card-t" }, "The running vote"),
            el("p", { class: "card-k" }, "Trust whoever has been predicting best"),
            el("p", { class: "card-d" }, "After each hop every expert's weight is scaled by how well it predicted that hop. A small share is always passed around, so a newcomer that starts predicting well takes over within a few hops, and a radio that switches recipe is followed.")),
          lf ? tile("On a recipe no expert knows", pct0(lf.steady), [`of the ${OTS.emitName("lagged_fib").toLowerCase()}'s next hops called right on the Typical sensor. `,
            el("b", {}, `Bank alone: ${pct0(lf.id_only)}`), lstm ? `; an LSTM trained on the known families: ${pct0(lstm.lstm)}.` : "."]) : null)));
  }

  function respond(tr, handles) {
    const pp = P();
    let charts = null;
    if (tr && tr.fc_pred) {
      // three moments of the example run: before the lock, as it narrows, after the lock
      const n = tr.fc_pred.length;
      const narrow = (() => { for (let j = 1; j < Math.min(n, tr.tau + 2); j++) { const s = coverSet(tr.fc_pred[j]).length; if (s > 1 && s <= 4) return j; } return Math.min(n - 1, Math.max(1, tr.tau - 1)); })();
      const moments = [0, narrow, Math.min(n - 1, tr.tau + 3)];
      const mini = (j) => {
        const f = tr.fc_pred[j], set = new Set(coverSet(f)), truth = tr.truth[j + 1];
        const W = 320, H = 132, m = { l: 6, r: 6, t: 12, b: 26 }, bw = (W - m.l - m.r) / M();
        const top = Math.max(...f, 1);
        const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Forecast after hop ${j + 1}: ${set.size} channels needed to cover 90%; the true next channel was ${fmt.ch(truth)}` });
        root.append(svg("line", { class: "fc-ax", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
        f.forEach((v, c) => {
          const h = Math.max(1.5, (v / top) * (H - m.t - m.b)), x = m.l + c * bw + 2;
          root.append(svg("rect", { class: "fc-bar" + (set.has(c) ? " in" : ""), x, y: H - m.b - h, width: bw - 4, height: h, rx: 2 }));
          if (c % 3 === 0) root.append(svg("text", { class: "fc-t", x: x + (bw - 4) / 2, y: H - m.b + 12, "text-anchor": "middle" }, fmt.ch(c)));
        });
        const tx = m.l + truth * bw + bw / 2;
        root.append(svg("path", { class: "fc-true", d: `M${tx} ${H - m.b + 15} l-5 8 h10 z` }));
        return { root, size: set.size, hit: set.has(truth), pTop: Math.max(...f) / 10 };
      };
      const label = (i, j) => (i === 0 ? "Just started" : j + 1 >= tr.tau ? "Locked on" : "Narrowing down");
      charts = el("div", { class: "how-fc" }, moments.map((j, i) => {
        const r = mini(j);
        return el("article", { class: "card how-fc-c" },
          el("p", { class: "card-t" }, `After hop ${j + 1} · ${label(i, j)}`),
          el("h3", {}, r.size === 1 ? "Jam 1 channel" : `Jam ${r.size} channels`),
          el("p", {}, `Top guess ${d1(r.pTop)}% sure. The true next channel ${r.hit ? "was inside the set" : "fell outside the set"}.`),
          r.root);
      }));
    }
    const dn = pp && pp.denial && pp.denial.by_sensor.typical;
    const price = dn ? el("div", { class: "how-price" }, Object.entries(dn).map(([g, x]) => el("div", { class: "tile" },
      el("div", { class: "tile-l" }, OTS.emitName(g)), el("div", { class: "tile-v num" }, d1(x.mean_tones), el("small", {}, ` of ${M()}`)),
      el("div", { class: "tile-s" }, `channels per hop · ${pct0(x.unlimited)} of hops denied`)))) : null;
    return block("Respond: from a forecast to a jam plan",
      sectionHead("Step 5", "Respond: from a forecast to a jam plan",
        "The predictor gives every channel a probability. OTS jams the smallest set of top channels that covers the next hop with a promised probability; the promise is kept by adjusting the set's size online from the hops it can check."),
      charts, charts ? el("div", { class: "how-fc-leg" },
        el("span", {}, el("i", { style: "background:var(--accent)" }), "in the set: jammed"),
        el("span", {}, el("i", { style: "background:rgba(168,181,199,0.28)" }), "left alone"),
        el("span", {}, "▲ the channel the radio actually used next"),
        el("span", { class: "muted" }, `${OTS.emitName(tr.emitter)} radio, seed ${tr.seed}; sets drawn at a fixed 90% of the forecast's weight for illustration.`)) : null,
      price ? sectionHead(null, "The size of the set is the report", "Typical sensor, from the paper's response experiment: a pinned-down radio costs one channel, a statistical one most of the band, and real encryption the whole barrage.") : null,
      price);
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
    // dangerous false claims across the paper's identification runs (every sensor)
    const pp = P(), conf = pp && pp.confusion;
    let falseTxt = "";
    if (conf) {
      const ALG = new Set(["lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed"]);
      let bad = 0, tot = 0;
      for (const s of Object.values(conf.by_sensor)) {
        s.counts.forEach((row, i) => { tot += conf.n_seeds; row.forEach((v, j) => { const f = conf.families[j]; if (ALG.has(f) && f !== s.target[i]) bad += v; }); });
      }
      falseTxt = `Across the paper's ${fmt.int(tot)} identification runs on all ${Object.keys(conf.by_sensor).length} sensors, a crackable algorithm was falsely named ${d1((100 * bad) / tot)} times per 100.`;
    }
    const delta = Math.round(((OTS.data && OTS.data.delta) || 0.05) * 100);
    const rule1 = el("article", { class: "card how-rule" },
      el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, "Safety rule 1 · the 95% lock"),
        el("p", { class: "card-k" }, "No claim until one expert is at least 95% sure"),
        el("p", { class: "card-d" }, `While experts are still competing, OTS stays quiet and keeps listening. It names a pattern only when one passes 95%; if none ever does, it says so. By design this keeps the chance of a wrong lock to about ${delta} in 100 or less.`))),
      legend, chartBox,
      el("p", { class: "card-foot" }, (tr ? `A real recorded run (${OTS.emitName(tr.emitter)} radio, ${OTS.preset(tr.preset)?.name.toLowerCase() || ""} sensor). ` : "") + falseTxt));

    // ---- rule 2: "Unknown" is always on the table (the unbreakable-encryption control)
    const idb = pp && pp.id_baselines && pp.id_baselines.by_sensor.typical;
    const em = OTS.emitter("csprng");
    let outs = null, foot2 = "";
    if (idb) {
      const b = idb.methods.bank, s = idb.methods.pairwise_sprt;
      const crack = s.per_gen.csprng ? s.per_gen.csprng[1] : null;
      outs = el("div", { class: "how-out" },
        el("div", { class: "tile" }, el("div", { class: "tile-l" }, "OTS"),
          el("div", { class: "tile-v num" }, pct0(b.unknown)),
          el("div", { class: "tile-s" }, "of the runs ended in “unknown”"),
          el("span", { class: "status good" }, el("span", { class: "ic" }, "✓"), "the right answer")),
        el("div", { class: "tile" }, el("div", { class: "tile-l" }, "Standard sequential test"),
          el("div", { class: "tile-v num" }, pct0(crack)),
          el("div", { class: "tile-s" }, "of the runs ended in a claimed crack"),
          el("span", { class: "status bad" }, el("span", { class: "ic" }, "✗"), "a false claim")));
      foot2 = `${pp.id_baselines.n_seeds} runs against unbreakable encryption on the Typical sensor (${em ? em.blurb.replace(/\.$/, "") : "a keyed CSPRNG"}). The option costs little: on the radios it does know, OTS is still right ${d1(100 * b.acc)}% of the time, against ${d1(100 * s.acc)}% for the standard test.`;
    }
    const rule2 = el("article", { class: "card how-rule" },
      el("div", { class: "card-h" }, el("div", {},
        el("p", { class: "card-t" }, "Safety rule 2 · “Unknown” is always an option"),
        el("p", { class: "card-k" }, "OTS can say “I don't know”. The standard test cannot."),
        el("p", { class: "card-d" }, "The Unknown expert competes like any other. Against a radio with no pattern to find it wins, so OTS makes no claim instead of forcing a guess. The standard sequential test uses the same experts but must always name one of them."))),
      outs, el("p", { class: "card-foot" }, foot2));
    return block("Two safety rules",
      sectionHead(null, "Two safety rules", "Both rules exist to stop the dangerous mistake: confidently claiming to have cracked a radio when it has not."),
      el("div", { class: "grid-2" }, rule1, rule2));
  }

  function fairPlay() {
    const nPat = OTS.hypotheses.length - 1;
    return block("Fair play",
      sectionHead(null, "Fair play", null),
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
              el("li", {}, el("b", {}, "How good the sensor is: "), "how often it misses a hop and how many ghosts to expect (and it copes when that estimate is wrong)."),
              el("li", {}, el("b", {}, "The list of suspects: "), `${nPat} kinds of radio recipe, plus the “unknown” option. The new algorithms on the Proof page are not on it.`))),
          el("div", { class: "how-fair-col how-hidden" },
            el("h3", {}, icon(...ICONS.eye), "Shown only to you, the audience"),
            el("ul", {},
              el("li", {}, el("b", {}, "The radio's true path, "), "hop by hop."),
              el("li", {}, el("b", {}, "Which signals were ghosts "), "and which hops the sensor missed."),
              el("li", {}, el("b", {}, "The hidden recipe and its real secret key, "), "used only after the run to mark OTS's answer.")))),
        el("p", { class: "card-foot" }, "Every rival method is given exactly the same detections, and every run uses a fresh random secret. When the radio fights back (Control room, red team), it too decides only from what a real radio could know. In the Live demo, “Reveal sensor mistakes” and “Show the hidden true path” change only the picture.")));
  }

  function glossary() {
    const nHyp = OTS.hypotheses.length, m = M();
    const terms = [
      ["Hop", null, "One jump of the radio to a channel. The radio hops at a steady pace, one hop per time slot."],
      ["Channel", null, `One of the ${m} radio frequencies the radio can jump between, numbered ${fmt.ch(0)} to ${fmt.ch(m - 1)}.`],
      ["Detection rate", null, "How many of the radio's real hops the sensor catches. 80% means 1 hop in 5 goes unheard (a “missed hop”)."],
      ["Ghost", "false alarm", "A signal the sensor reports on a channel the radio was not using, caused by noise or other transmitters."],
      ["Expert", "hypothesis", `One possible explanation of the radio's recipe. OTS keeps ${nHyp} of them running side by side.`],
      ["Confidence", "posterior", `How believable each expert is, given everything heard so far. The ${nHyp} confidences always add up to 100%.`],
      ["Lock", null, "The moment one expert passes 95% confidence. OTS then names the pattern and reads off its secret."],
      ["Secret", "seed, key or register state", "The hidden setting that makes one radio's recipe unique. Knowing it lets OTS predict every future hop."],
      ["Unknown", "H₀", "The “none of the above” expert. If it wins, OTS reports that no known recipe fits."],
      ["Universal learner", null, "A forecaster that names no recipe but learns rules from the hops themselves, so prediction works on recipes no expert knows."],
      ["Running vote", "Fixed-Share mixture", "How the forecasts are combined: weight follows recent accuracy, with a share always passed around so a newcomer can take over."],
      ["Channel set", "conformal prediction set", "The smallest group of channels that covers the next hop with a promised probability (90% here). Its size is the jam plan."],
      ["Switch", "regime change", "The radio changing its recipe mid-engagement. The switch tracker dates it and starts the experts over."],
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
        row(["M"], "Minimize or expand the side menu"),
        grp("On the Proof and How it works pages"),
        row(["P"], "Present: one slide at a time, full screen"),
        row(["→", "Space"], "Next slide (← back, Home / End first / last)"),
        row(["Esc"], "Stop presenting"),
        grp(playerPages.length ? `On the ${playerPages.join(" and ")} pages` : "On player pages"),
        row(["Space"], "Play or pause"),
        row(["←", "→"], "Step one hop back or forward"),
        row(["R"], "Restart the run"),
        row(["N"], "Next recorded run"),
        row(["T"], "Reveal or hide the sensor's mistakes"),
        row(["S"], "Open or close the settings panel"))));
    return block("Glossary and shortcuts", sectionHead(null, "Glossary and shortcuts", null), el("div", { class: "how-last" }, gl, kb));
  }

  // ================================================================== view
  OTS.views.register({
    id: "how", nav: "How it works", title: "How it works", hint: "The method in plain words",
    subtitle: "Listen, weigh every explanation, lock only when sure, predict, size the jam, watch for a change",
    icon: "M12 3l9 5-9 5-9-5 9-5zm-9 9l9 5 9-5M3 16l9 5 9-5", usesPlayer: false,
    mount(root) {
      injectCss();
      const tr = exampleRun(), pending = [];
      const page = el("div", { class: "page how-page" },
        hero(),
        diagram(),
        pipeline(tr),
        bank(),
        universal(),
        respond(tr, pending),
        rules(tr, pending),
        fairPlay(),
        glossary());
      root.append(page);
      OTS.present.attach(page);
      // charts mount after their containers are in the document (they measure width)
      const charts = pending.map((p) => OTS.chart.line(p.box, p.spec));
      return () => { for (const c of charts) c.destroy(); };
    },
  });
})();
