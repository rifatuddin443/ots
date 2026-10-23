/* Proof view: evidence across hundreds of recorded runs, one claim per card.
 *
 * A static page (no player). The filter row picks the sensor condition for the
 * scoreboard-based parts (headline tiles and claims 1, 2, 3 and 7); the
 * "Show data tables" switch (OTS.prefs.tables) adds a small table under every chart.
 *   tiles    right answer · dangerous false claims · hops to lock · next hop
 *   claim 1  wrong answers per 100 runs for each hidden pattern (+ the Full-state LCG weak spot)
 *   claim 2  dangerous false claims per 100 runs for each sensor condition
 *   claim 3  the keyed-CSPRNG control: who answers "unknown"
 *   claim 4  right answers and hops to lock as the sensor gets worse (two charts, one axis each)
 *   claim 5  textbook Berlekamp-Massey vs OTS on a shift register with missed hops (DATA.proof)
 *   claim 6  the same bank with and without its sensor model (DATA.proof)
 *   claim 7  next-hop calls vs a frequency-statistics forecaster and a random guess
 * Every number is computed from OTS_DATA (scoreboard rows, proof rows, recorded traces).
 */
(function () {
  "use strict";
  const { el, fmt, S } = OTS;

  /** Sensor condition shown by the scoreboard cards: 'all' or a preset id. Kept across visits. */
  let filter = "all";

  // ------------------------------------------------------------------ styles
  const CSS = `
.proof-hero .intro { margin-top: 10px; }
.proof-card { display: flex; flex-direction: column; }
.proof-card > .card-foot { margin-top: auto; padding-top: 12px; }
.proof-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; padding: 10px 14px;
  border: 1px solid var(--line-2); border-radius: 11px; background: rgba(11, 19, 31, 0.94); backdrop-filter: blur(8px); }
.proof-bar-l { font: 600 10.5px/1.2 var(--mono); letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.proof-bar-h { font-size: 12.5px; color: var(--ink-2); min-width: 0; }
.proof-bar .proof-grow { flex: 1; min-width: 8px; }
.proof-bar .switch { padding: 0; }
.proof-tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.proof-tiles .tile-s { line-height: 1.4; }
.proof-sec { margin: 12px 0 -6px; }
.proof-sec h2 { margin: 0; font: 650 18px/1.3 var(--sans); letter-spacing: -0.005em; }
.proof-sec p { margin: 3px 0 0; color: var(--ink-2); font-size: 13.5px; max-width: 80ch; }
.proof-legrow { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 2px 16px; }
.proof-better { font-size: 12.5px; color: var(--muted); margin-bottom: 8px; white-space: nowrap; }
.proof-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 26px; }
.proof-ch { margin: 0 0 8px; font-size: 13.5px; font-weight: 600; color: var(--ink); }
.proof-ch small { font-weight: 400; color: var(--muted); font-size: 12.5px; }
.proof-note { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 8px 12px; margin: 14px 0 0; padding: 12px 14px;
  border: 1px solid var(--line); border-radius: 10px; background: var(--surface); font-size: 13.5px; color: var(--ink-2); }
.proof-note p { margin: 0; flex: 1; min-width: 220px; max-width: 96ch; }
.proof-note b { color: var(--ink); font-weight: 600; }
.proof-verdicts { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.proof-verdicts .status { white-space: normal; line-height: 1.3; text-align: left; }
.proof-tbl { margin-top: 14px; overflow-x: auto; border-top: 1px solid var(--grid); padding-top: 4px; }
.proof-tbl .table { min-width: 360px; }
.proof-tbl caption { caption-side: top; text-align: left; color: var(--muted); font-size: 12px; padding: 6px 0 4px; }
.proof-tbl tbody th { font: 500 13px var(--sans); letter-spacing: 0; text-transform: none; color: var(--ink); border-top: 1px solid var(--grid); }
.proof-tbl td.proof-us { color: var(--accent-ink); font-weight: 600; }
.proof-dim .colgrp { opacity: 0.35; transition: opacity 0.2s; }
.proof-dim.proof-hi-1 .colgrp:nth-of-type(1), .proof-dim.proof-hi-2 .colgrp:nth-of-type(2),
.proof-dim.proof-hi-3 .colgrp:nth-of-type(3), .proof-dim.proof-hi-4 .colgrp:nth-of-type(4) { opacity: 1; }
@media (min-width: 861px) {
  .proof-bar { position: sticky; top: var(--proof-top, 64px); z-index: 15; box-shadow: 0 8px 22px rgba(0, 0, 0, 0.35); }
}
@media (max-width: 1100px) {
  .proof-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 760px) {
  .proof-pair { grid-template-columns: minmax(0, 1fr); gap: 22px; }
  .proof-bar .proof-grow { display: none; }
  .proof-bar-h { flex-basis: 100%; }
}
@media (max-width: 560px) {
  .proof-tiles { gap: 10px; }
  .proof-tiles .tile { padding: 12px 12px; }
  .proof-tiles .tile-v { font-size: 24px; }
  .proof-sec h2 { font-size: 17px; }
  .proof-tbl .table { min-width: 0; font-size: 12.5px; }
  .proof-tbl .table th, .proof-tbl .table td { padding: 6px 5px; }
  .proof-tbl thead th { white-space: normal; letter-spacing: 0.04em; vertical-align: bottom; }
  .proof-tbl tbody th { font-size: 12.5px; }
}`;
  function injectCss() {
    if (document.getElementById("css-proof")) return;
    document.head.append(el("style", { id: "css-proof" }, CSS));
  }

  // ----------------------------------------------------------------- helpers
  /** One decimal, trailing ".0" dropped: 53.333 -> "53.3", 5.0000001 -> "5". */
  const d1 = (v) => String(Math.round(v * 10) / 10);
  /** Fraction -> "per 100 runs" number for sentences: 0.1133 -> "11.3". */
  const per = (p) => (p == null ? "—" : d1(p * 100));
  /** Fraction -> "91.3%" (one decimal, ".0" dropped). */
  const pct1 = (p) => (p == null ? "—" : d1(p * 100) + "%");
  const pct0 = fmt.pct0;
  /** Chance of a random guess, exact to two decimals: 0.0625 -> "6.25%". */
  const chanceTxt = () => String(+(S.chance() * 100).toFixed(2)) + "%";
  /** Column cap labels: one decimal below 10, whole numbers above (keeps pairs from colliding). */
  const capLabel = (v) => (v > 0 && v < 10 ? d1(v) : String(Math.round(v)));
  /** "8.5" for a ratio; whole numbers from 10 up. */
  const times = (r) => (r >= 10 ? String(Math.round(r)) : r.toFixed(1));
  /** Smallest round axis maximum that fits v. */
  const niceMax = (v) => [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100].find((m) => m >= v) || Math.ceil(v / 50) * 50;
  const plural = (n, one, many) => fmt.int(n) + " " + (n === 1 ? one : many || one + "s");
  const count = (p, n) => Math.round(p * n);
  /** "4 hops in 10" for a detection probability of 0.6. */
  const missedIn10 = (p_d) => plural(Math.round((1 - p_d) * 10), "hop") + " in 10";
  /** Presets from the best sensor to the worst. */
  const presetsByQuality = () => [...OTS.presets].sort((a, b) => b.p_d - a.p_d);
  const condName = (id) => (id === "all" ? "All conditions" : OTS.preset(id)?.name || id);
  /** Short phrase for the filter: "the Typical sensor" or "all three sensor conditions". */
  const condPhrase = (id) => (id === "all" ? `all ${OTS.presets.length} sensor conditions` : `the ${condName(id)} sensor`);
  /** Expected ghost detections per hop for a false-alarm rate, as text: 0.05 -> "0.75". */
  const ghosts = (p_fa) => String(+(p_fa * ((OTS.data.M || 16) - 1)).toFixed(2));
  const status = (kind, icon, text) => el("span", { class: "status " + kind },
    el("span", { class: "ic", "aria-hidden": "true" }, icon), el("span", {}, text));
  const fill = (node, ...kids) => { OTS.clear(node).append(...kids.flat(Infinity).filter((k) => k != null && k !== false && k !== "")); return node; };
  const C = () => ({ ots: OTS.tok("accent"), base: OTS.tok("baseline") });
  const sq = (label, color) => ({ label, color, kind: "square" });
  /** Value labels on the column caps only where a pair of caps has room (off on phones). */
  const roomy = (box, nCats) => (box.clientWidth - 70) / nCats >= 44;
  /** Small data table. cols: [{h, us?: bool}], rows: arrays of cells. */
  function table(caption, cols, rows) {
    return el("div", { class: "proof-tbl" }, el("table", { class: "table" },
      el("caption", {}, caption),
      el("thead", {}, el("tr", {}, cols.map((c) => el("th", { scope: "col" }, c.h)))),
      el("tbody", {}, rows.map((r) => el("tr", {}, r.map((v, i) => el(i ? "td" : "th", { scope: i ? null : "row", class: cols[i].us ? "proof-us" : null }, v)))))));
  }
  /** Card skeleton: claim title, takeaway headline (filled later), description, body, table slot, footnote. */
  function shell(n, title) {
    const k = el("p", { class: "card-k" }), d = el("p", { class: "card-d" });
    const body = el("div"), tbl = el("div", { hidden: true }), foot = el("div", { class: "card-foot" });
    const node = el("article", { class: "card proof-card", "aria-label": `Claim ${n}: ${title}` },
      el("div", { class: "card-h" }, el("div", {}, el("p", { class: "card-t" }, `Claim ${n} · ${title}`), k, d)),
      body, tbl, foot);
    return { node, k, d, body, tbl, foot };
  }
  /** Scoreboard row of one emitter for a filter value (null when absent). */
  const emitRow = (id, f) => S.byEmitter(f).find((r) => r.id === id) || null;
  // "New algorithm" emitters (no expert in the bank) are kept out of the
  // identification claims and get their own prediction claim.
  const isNewId = (id) => S.isNewAlgo(id);
  const knownRow = (r) => !isNewId(r.emitter);
  /** Aggregate over the known patterns only (excludes new-algorithm emitters). */
  const overK = (f = "all") => S.overallWhere(f, knownRow);
  /** Aggregate over the new-algorithm emitters only. */
  const overN = (f = "all") => S.overallWhere(f, (r) => isNewId(r.emitter));
  /** Per-emitter scoreboard rows for the known patterns only. */
  const byEmitterK = (f) => S.byEmitter(f).filter((r) => !isNewId(r.id));
  /** Per-emitter scoreboard rows for the new-algorithm emitters only. */
  const byEmitterNew = (f) => S.byEmitter(f).filter((r) => isNewId(r.id));
  /** Column categories for the per-condition charts: each preset, then "All conditions". */
  const condCats = () => [
    ...presetsByQuality().map((p) => ({ id: p.id, label: p.name, sub: "catches " + Math.round(p.p_d * 100) + "%" })),
    { id: "all", label: "All conditions", sub: fmt.int(overK("all").n) + " runs" },
  ];
  /** Highlight the filtered condition's column pair in a per-condition chart (survives redraws: pure CSS). */
  function highlight(box, cats) {
    for (let i = 1; i <= 4; i++) box.classList.remove("proof-hi-" + i);
    const i = cats.findIndex((c) => c.id === filter);
    box.classList.toggle("proof-dim", filter !== "all" && i >= 0);
    if (filter !== "all" && i >= 0 && i < 4) box.classList.add("proof-hi-" + (i + 1));
  }
  /** Line-chart x axis: share of hops caught, best sensor on the left. */
  const caughtAxis = (min, ticks) => ({ label: "Share of hops the sensor catches (%)", min: 100, max: min, ticks, format: (v) => v + "%" });

  // ===================================================================== view
  OTS.views.register({
    id: "proof", nav: "Proof", title: "Proof at scale", hint: "Hundreds of runs, side by side",
    subtitle: "Seven claims, each checked on hundreds of real runs, with every run counted",
    icon: "M4 20V10m6 10V4m6 16v-7m4 7H2", usesPlayer: false,
    mount(root) {
      injectCss();
      const handles = [], offs = [];
      const DATA = OTS.data, sb = DATA.scoreboard;
      // "New algorithm" emitters have no expert in the bank: they are kept out of
      // the identification claims (1-7, about the known patterns) and get their
      // own next-channel-prediction claim (8).
      const knownEmitters = OTS.emitters.filter((e) => !e.new_algo);
      const newEmitters = OTS.emitters.filter((e) => e.new_algo);
      const all = overK("all");
      const nPat = knownEmitters.length, nCond = OTS.presets.length, nSeeds = sb.n_seeds;

      // ------------------------------------------------------------ hero
      const hero = el("div", { class: "proof-hero" },
        el("h2", { class: "lead" }, "Not one lucky demo: ", el("em", {}, fmt.int(all.n) + " runs"), ", every one counted."),
        el("p", { class: "intro" },
          `Each run hides one of ${nPat} radio patterns behind a fresh random secret, heard through a sensor that misses hops and hears ghost signals. `,
          "OTS and the standard test see exactly the same detections. Each card below makes one claim and shows the evidence.",
          newEmitters.length ? ` A final claim adds ${newEmitters.length} algorithms the bank has no expert for at all, to show the next channel is still predicted.` : ""));

      // ------------------------------------------------------ filter row
      const segBtns = [{ id: "all", name: "All conditions" }, ...presetsByQuality()].map((p) => el("button", {
        class: "btn", type: "button", "data-f": p.id, "aria-pressed": "false",
        title: p.id === "all" ? "Every sensor condition together" : `Catches ${Math.round(p.p_d * 100)}% of hops`,
        onclick: () => { filter = p.id; renderFiltered(); },
      }, p.name));
      const barHelp = el("span", { class: "proof-bar-h" });
      const tblInp = el("input", { type: "checkbox", "aria-label": "Show data tables" });
      tblInp.checked = !!OTS.prefs.tables;
      tblInp.addEventListener("change", () => OTS.setPref("tables", tblInp.checked));
      const bar = el("div", { class: "proof-bar", role: "group", "aria-label": "Sensor condition filter" },
        el("span", { class: "proof-bar-l" }, "Sensor condition"),
        el("div", { class: "seg" }, segBtns), barHelp, el("span", { class: "proof-grow" }),
        el("label", { class: "switch" }, tblInp, "Show data tables"));

      // ----------------------------------------------------------- tiles
      const tiles = el("div", { class: "proof-tiles", "aria-live": "polite" });
      function renderTiles() {
        const o = overK(filter);
        const tile = (label, value, unit, sub) => el("div", { class: "tile" },
          el("div", { class: "tile-l" }, label),
          el("div", { class: "tile-v" }, value, unit ? el("small", {}, " " + unit) : null),
          el("div", { class: "tile-s" }, sub));
        fill(tiles,
          tile("Right answer", pct1(o.ots_acc), null, `Standard test: ${pct1(o.base_acc)}`),
          tile("Dangerous false claims", per(o.ots_fb), "per 100", `Standard test: ${per(o.base_fb)} per 100`),
          tile("Locks on in", o.hops_med == null ? "—" : d1(o.hops_med), "hops", o.hops_med == null ? "No correct locks here" : `Half of its correct locks need ${d1(o.hops_med)} hops or fewer (a run is ${DATA.window})`),
          tile("Next hop called", pct0(o.top1), null, `Frequency statistics: ${pct0(o.markov_top1)} · random guess: ${chanceTxt()}`));
      }

      const section = (title, text) => el("div", { class: "proof-sec" }, el("h2", {}, title), el("p", {}, text));
      const legend = (items, note) => el("div", { class: "proof-legrow" }, OTS.chart.legend(items), note ? el("span", { class: "proof-better" }, note) : null);

      // ========================================================= claim 1
      const A = shell(1, "Fewer wrong answers");
      const aBox = el("div"), aNote = el("div", { class: "proof-note", hidden: true });
      A.d.textContent = "Each pair of columns is one kind of hidden radio. The columns count how often each method named the wrong pattern in 100 runs: shorter is better.";
      A.body.append(legend([sq("OTS", C().ots), sq("Standard test", C().base)], "▼ lower is better"), aBox, aNote);
      function specA() {
        const rows = byEmitterK(filter), by = new Map(rows.map((r) => [r.id, r]));
        const wrong = (r, s) => 100 * (1 - (s === "ots" ? r.ots_acc : r.base_acc));
        return {
          height: 330, ariaLabel: "Wrong answers per 100 runs for each hidden pattern: OTS versus the standard test",
          categories: rows.map((r) => ({ id: r.id, label: r.name })),
          series: [{ id: "ots", label: "OTS", color: C().ots }, { id: "base", label: "Standard test", color: C().base }],
          value: (c, s) => wrong(by.get(c), s),
          y: { label: "Wrong answers per 100 runs", max: niceMax(Math.max(...rows.flatMap((r) => [wrong(r, "ots"), wrong(r, "base")]))), format: String },
          x: { label: "Hidden hop pattern" }, barWidth: 20, valueLabels: roomy(aBox, rows.length), valueFormat: capLabel,
          tooltip: (c) => { const r = by.get(c.id); return { title: `${r.name} · ${condName(filter)}`, rows: [
            { value: d1(wrong(r, "ots")), label: "OTS: wrong per 100 runs", color: C().ots, kind: "square" },
            { value: d1(wrong(r, "base")), label: "Standard test: wrong per 100", color: C().base, kind: "square" },
            { value: fmt.int(r.n), label: "runs" }] }; },
        };
      }
      function renderA() {
        const o = overK(filter), wo = 1 - o.ots_acc, wb = 1 - o.base_acc;
        const pre = filter === "all" ? "" : condName(filter) + " sensor: ";
        A.k.textContent = pre + (wo === 0 && wb > 0 ? `no wrong answers at all, against ${per(wb)} per 100 runs for the standard test`
          : wo < wb ? `${times(wb / wo)}× fewer wrong answers than the standard test: ${per(wo)} vs ${per(wb)} per 100 runs`
          : `${per(wo)} wrong answers per 100 runs for OTS, ${per(wb)} for the standard test`);
        if (!pre) A.k.textContent = A.k.textContent[0].toUpperCase() + A.k.textContent.slice(1);
        // honest weak spot: the Full-state LCG, shown only when the data says it is one
        const lf = emitRow("lcg_full", filter);
        const weak = lf && (1 - lf.ots_acc) >= 0.1 && (1 - lf.ots_acc) >= 2 * wo;
        aNote.hidden = !weak;
        if (weak) {
          const traces = (filter === "all" ? OTS.presets.map((p) => p.id) : [filter]).flatMap((p) => OTS.tracesFor("lcg_full", p));
          const misses = traces.filter((t) => !t.correct), asMarkov = misses.filter((t) => t.decision === "markov").length;
          const often = misses.length && asMarkov * 2 >= misses.length;
          fill(aNote, status("warn", "!", "Weak spot"), el("p", {},
            el("b", {}, `Full-state LCG: ${per(1 - lf.ots_acc)} wrong per 100. `),
            `With only ${DATA.M} channels, this radio's whole cycle also looks like a perfectly regular statistical pattern, so OTS `,
            often ? `often names it “Markov hopper” instead (${misses.length === 1 ? "as in the one miss" : `${asMarkov} of ${misses.length} misses`} in the recorded replays). ` : "often cannot tell it apart from one. ",
            "That is a soft miss, not a claim that the code is cracked ",
            lf.ots_fb === 0 ? "(no false claims on this pattern)." : `(false claims on this pattern: ${per(lf.ots_fb)} per 100).`,
            lf.base_acc >= lf.ots_acc ? ` The standard test does no better here (${per(1 - lf.base_acc)} wrong per 100).` : ""));
        }
        const rows = byEmitterK(filter);
        A.foot.textContent = `${fmt.int(o.n)} runs on ${condPhrase(filter)}: ${nPat} patterns × ${filter === "all" ? nCond + " conditions × " : ""}${nSeeds} seeds each. Every run counted, none hand-picked.`;
        A.table = () => table(`Wrong answers per 100 runs · ${condName(filter)}`,
          [{ h: "Hidden pattern" }, { h: "Runs" }, { h: "OTS", us: true }, { h: "Standard test" }],
          rows.map((r) => [r.name, fmt.int(r.n), capLabel(100 * (1 - r.ots_acc)), capLabel(100 * (1 - r.base_acc))]));
      }

      // ========================================================= claim 2
      const B = shell(2, "Far fewer dangerous false claims");
      const bBox = el("div");
      B.d.textContent = "A dangerous false claim is announcing “I have cracked this code” when the radio actually uses something else. Anyone acting on it would follow the wrong channels.";
      B.body.append(legend([sq("OTS", C().ots), sq("Standard test", C().base)], "▼ lower is better"), bBox);
      function specB() {
        const cats = condCats(), agg = new Map(cats.map((c) => [c.id, overK(c.id)]));
        const v = (id, s) => 100 * (s === "ots" ? agg.get(id).ots_fb : agg.get(id).base_fb);
        return {
          height: 290, ariaLabel: "Dangerous false claims per 100 runs for each sensor condition: OTS versus the standard test",
          categories: cats, series: [{ id: "ots", label: "OTS", color: C().ots }, { id: "base", label: "Standard test", color: C().base }],
          value: v, y: { label: "False claims per 100 runs", max: niceMax(Math.max(...cats.flatMap((c) => [v(c.id, "ots"), v(c.id, "base")]))), format: String },
          x: { label: "Sensor condition" }, barWidth: 20, valueLabels: roomy(bBox, cats.length), valueFormat: capLabel,
          tooltip: (c) => ({ title: c.label + (c.sub ? " · " + c.sub : ""), rows: [
            { value: d1(v(c.id, "ots")), label: "OTS: false claims per 100", color: C().ots, kind: "square" },
            { value: d1(v(c.id, "base")), label: "Standard test: per 100", color: C().base, kind: "square" },
            { value: fmt.int(agg.get(c.id).n), label: "runs" }] }),
        };
      }
      function renderB() {
        const o = overK(filter), on = filter === "all" ? "" : ` with the ${condName(filter)} sensor`;
        B.k.textContent = o.ots_fb === 0 && o.base_fb > 0 ? `No dangerous false claims at all${on}, against ${per(o.base_fb)} per 100 runs for the standard test`
          : o.ots_fb < o.base_fb ? `${times(o.base_fb / o.ots_fb)}× fewer${on}: ${per(o.ots_fb)} per 100 runs, against ${per(o.base_fb)} for the standard test`
          : `${per(o.ots_fb)} per 100 runs for OTS${on}, ${per(o.base_fb)} for the standard test`;
        highlight(bBox, condCats());
        B.foot.textContent = `${fmt.int(all.n)} runs: ${fmt.int(all.n / nCond)} per sensor condition (${nPat} patterns × ${nSeeds} seeds). `
          + (filter === "all" ? "Use the filter above to focus on one condition." : `Headline: the ${condName(filter)} sensor, ${fmt.int(o.n)} runs.`);
        B.table = () => table("Dangerous false claims per 100 runs",
          [{ h: "Sensor condition" }, { h: "Runs" }, { h: "OTS", us: true }, { h: "Standard test" }],
          condCats().map((c) => { const a = overK(c.id); return [c.label, fmt.int(a.n), capLabel(100 * a.ots_fb), capLabel(100 * a.base_fb)]; }));
      }

      // ========================================================= claim 3
      const Cc = shell(3, "Honest when there is nothing to find");
      const cBox = el("div"), cVerdict = el("div", { class: "proof-verdicts" });
      const csName = OTS.emitName("csprng");
      Cc.body.append(legend([sq("OTS", C().ots), sq("Standard test", C().base)], "▲ higher is better"), cBox, cVerdict);
      function specC() {
        const cats = condCats(), rows = new Map(cats.map((c) => [c.id, emitRow("csprng", c.id)]));
        // on the CSPRNG the only correct answer is "unknown", so "right" = answered "unknown"
        const v = (id, s) => { const r = rows.get(id); return r ? 100 * (s === "ots" ? r.ots_acc : r.base_acc) : null; };
        return {
          height: 290, ariaLabel: `Share of runs on the ${csName} answered unknown, for each sensor condition: OTS versus the standard test`,
          categories: cats, series: [{ id: "ots", label: "OTS", color: C().ots }, { id: "base", label: "Standard test", color: C().base }],
          value: v, y: { label: "Runs answered “unknown” (%)", max: 100, ticks: [0, 25, 50, 75, 100], format: String },
          x: { label: "Sensor condition" }, barWidth: 20, valueLabels: roomy(cBox, cats.length), valueFormat: (x) => String(Math.round(x)),
          tooltip: (c) => { const r = rows.get(c.id); return { title: `${csName} · ${c.label}`, rows: [
            { value: pct0(r.ots_acc), label: "OTS answered “unknown”", color: C().ots, kind: "square" },
            { value: pct0(r.base_acc), label: "Standard test answered “unknown”", color: C().base, kind: "square" },
            { value: fmt.int(r.n), label: "runs" }] }; },
        };
      }
      function renderC() {
        const r = emitRow("csprng", filter);
        if (!r) { Cc.k.textContent = "No runs on the unbreakable control radio in this data."; return; }
        const pre = filter === "all" ? "" : condName(filter) + " sensor: ";
        Cc.k.textContent = pre + (pre ? "on" : "On") + ` the unbreakable radio, OTS says “unknown” in ${pct0(r.ots_acc)} of runs; the standard test `
          + (r.base_acc === 0 ? "never does" : `does in ${pct0(r.base_acc)}`);
        Cc.d.textContent = `The ${csName} is the control: it is cryptographically secure, so there is no pattern to find and the only right answer is “unknown”. `
          + `The standard test has no such option and must name a pattern; in ${count(r.base_fb, r.n)} of ${r.n} runs it even claimed a crackable code.`;
        const verdict = (who, acc, extra) => status(acc >= 0.5 ? "good" : "bad", acc >= 0.5 ? "✓" : "✕",
          `${who}: right in ${count(acc, r.n)} of ${r.n}`);
        fill(cVerdict, verdict("OTS", r.ots_acc), verdict("Standard test", r.base_acc));
        highlight(cBox, condCats());
        Cc.foot.textContent = `${fmt.int(emitRow("csprng", "all").n)} runs on the ${csName}: ${nSeeds} seeds per sensor condition. `
          + (filter === "all" ? "" : `Headline: the ${condName(filter)} sensor, ${r.n} runs.`);
        Cc.table = () => table(`${csName}: runs answered “unknown”`,
          [{ h: "Sensor condition" }, { h: "Runs" }, { h: "OTS", us: true }, { h: "Standard test" }, { h: "Standard test claims a crack" }],
          condCats().map((c) => { const x = emitRow("csprng", c.id); return [c.label, fmt.int(x.n), pct0(x.ots_acc), pct0(x.base_acc), pct0(x.base_fb)]; }));
      }

      // ========================================================= claim 4
      const Dd = shell(4, "Still works as the sensor gets worse");
      const dAcc = el("div"), dHops = el("div");
      const byP = () => presetsByQuality().map((p) => ({ ...p, ...overK(p.id) }));
      Dd.body.append(el("div", { class: "proof-pair" },
        el("div", {}, el("p", { class: "proof-ch" }, "Right answer ", el("small", {}, "· share of runs")),
          OTS.chart.legend([{ label: "OTS", color: C().ots }, { label: "Standard test", color: C().base }]), dAcc),
        el("div", {}, el("p", { class: "proof-ch" }, "Hops OTS needs to lock on ", el("small", {}, "· median of correct locks")),
          OTS.chart.legend([{ label: "OTS", color: C().ots }]), dHops)));
      const xTicksP = () => presetsByQuality().map((p) => Math.round(p.p_d * 100));
      const ptip = (rowsFor) => (x) => { const p = byP().find((q) => Math.round(q.p_d * 100) === x); return { title: `${p.name} · catches ${x}% of hops`, rows: rowsFor(p) }; };
      function specDAcc() {
        const ps = byP(), lo = Math.min(...ps.flatMap((p) => [p.ots_acc, p.base_acc])) * 100;
        const xmin = Math.min(...xTicksP()) - 5;
        return {
          height: 260, ariaLabel: "Share of runs with the right answer as the sensor catches fewer hops: OTS versus the standard test",
          x: caughtAxis(xmin, xTicksP()),
          y: { label: "Right answer (%)", min: Math.min(50, Math.floor((lo - 5) / 10) * 10), max: 100, format: String },
          series: [
            { id: "ots", label: "OTS", color: C().ots, marker: "all", emphasis: true, points: ps.map((p) => [Math.round(p.p_d * 100), 100 * p.ots_acc]) },
            { id: "base", label: "Standard test", color: C().base, marker: "all", points: ps.map((p) => [Math.round(p.p_d * 100), 100 * p.base_acc]) }],
          tooltip: ptip((p) => [{ value: pct1(p.ots_acc), label: "OTS right", color: C().ots }, { value: pct1(p.base_acc), label: "Standard test right", color: C().base }]),
        };
      }
      function specDHops() {
        const ps = byP().filter((p) => p.hops_med != null);
        return {
          height: 260, ariaLabel: "Median hops OTS needs to lock on as the sensor catches fewer hops",
          x: caughtAxis(Math.min(...xTicksP()) - 5, xTicksP()),
          y: { label: "Hops to lock (median)", min: 0, max: niceMax(Math.max(...ps.map((p) => p.hops_med)) + 1), format: String },
          series: [{ id: "ots", label: "OTS", color: C().ots, marker: "all", emphasis: true, points: ps.map((p) => [Math.round(p.p_d * 100), p.hops_med]) }],
          tooltip: ptip((p) => [{ value: d1(p.hops_med) + " hops", label: "OTS median to lock", color: C().ots }, { value: fmt.int(p.hops.length), label: "correct locks counted" }]),
        };
      }
      function renderD() {
        const ps = byP(), worst = ps[ps.length - 1];
        const ahead = ps.every((p) => p.ots_acc > p.base_acc);
        Dd.k.textContent = `Even when the sensor misses ${missedIn10(worst.p_d)}, OTS is right ${pct1(worst.ots_acc)} of the time`
          + (worst.hops_med != null ? ` and locks on in a median of ${d1(worst.hops_med)} hops` : "");
        Dd.d.textContent = (ahead ? "OTS stays ahead of the standard test at every sensor level. " : "")
          + "Further right on each chart means a worse sensor: it catches fewer hops and hears more ghosts. A worse sensor costs OTS a few more hops, not the answer.";
        Dd.foot.textContent = `${fmt.int(all.n)} runs: ${fmt.int(all.n / nCond)} per sensor condition. Hops to lock counts the ${fmt.int(all.hops.length)} runs that locked on the right answer. Same for every filter.`;
        Dd.table = () => table("By sensor condition (all patterns)",
          [{ h: "Sensor condition" }, { h: "Catches" }, { h: "OTS right", us: true }, { h: "Standard test right" }, { h: "OTS hops to lock", us: true }],
          ps.map((p) => [p.name, Math.round(p.p_d * 100) + "%", pct1(p.ots_acc), pct1(p.base_acc), p.hops_med == null ? "—" : d1(p.hops_med)]));
      }

      // ========================================================= claim 5
      const bm = DATA.proof && DATA.proof.bm_vs_filtered;
      const Ee = shell(5, "The textbook method breaks on the first missed hop");
      const eBox = el("div");
      Ee.body.append(OTS.chart.legend([{ label: "OTS (keeps every state that fits)", color: C().ots }, { label: "Textbook (Berlekamp–Massey)", color: C().base }]), eBox);
      const bmRows = bm ? [...bm.rows].sort((a, b) => b.p_d - a.p_d) : [];
      const pdTicks = (rows) => rows.map((r) => Math.round(r.p_d * 100));
      function specE() {
        return {
          height: 270, ariaLabel: "Share of shift-register secrets recovered as the sensor catches fewer hops: textbook Berlekamp-Massey versus OTS",
          x: caughtAxis(Math.min(...pdTicks(bmRows)), pdTicks(bmRows)),
          y: { label: "LFSR secret recovered (%)", min: 0, max: 100, ticks: [0, 25, 50, 75, 100], format: String },
          series: [
            { id: "ots", label: "OTS", color: C().ots, marker: "all", emphasis: true, points: bmRows.map((r) => [Math.round(r.p_d * 100), 100 * r.state_filtered]) },
            { id: "bm", label: "Textbook", color: C().base, marker: "all", points: bmRows.map((r) => [Math.round(r.p_d * 100), 100 * r.bm]) }],
          tooltip: (x) => { const r = bmRows.find((q) => Math.round(q.p_d * 100) === x); return { title: `Sensor catches ${x}% of hops`, rows: [
            { value: pct0(r.state_filtered), label: "OTS recovered", color: C().ots }, { value: pct0(r.bm), label: "Textbook recovered", color: C().base }] }; },
        };
      }
      function renderE() {
        if (!bmRows.length) { Ee.k.textContent = "This experiment is not in the data."; return; }
        const firstGap = bmRows.find((r) => r.p_d < 1) || bmRows[0], worst = bmRows[bmRows.length - 1];
        Ee.k.textContent = `Miss ${missedIn10(firstGap.p_d)} and the textbook method recovers ${pct0(firstGap.bm)} of secrets; OTS still recovers ${pct0(worst.state_filtered)} with ${worst.p_d === 0.5 ? "half the hops" : Math.round((1 - worst.p_d) * 100) + "% of hops"} missed`;
        Ee.d.textContent = "Berlekamp–Massey is the classic way to crack a shift-register (LFSR) radio, but it assumes every hop is heard. OTS keeps every register state that could explain the gaps.";
        Ee.foot.textContent = `Separate controlled test: ${bm.length}-bit LFSR, ${bm.n_seeds} seeds per point, no ghost signals. Same for every filter.`;
        Ee.table = () => table("LFSR secret recovered",
          [{ h: "Sensor catches" }, { h: "OTS", us: true }, { h: "Textbook" }],
          bmRows.map((r) => [Math.round(r.p_d * 100) + "%", pct0(r.state_filtered), pct0(r.bm)]));
      }

      // ========================================================= claim 6
      const ab = DATA.proof && DATA.proof.sensor_ablation;
      const Ff = shell(6, "Modelling sensor mistakes matters");
      const fBox = el("div");
      Ff.body.append(OTS.chart.legend([{ label: "OTS (expects missed hops and ghosts)", color: C().ots }, { label: "Same bank, sensor-naive", color: C().base }]), fBox);
      const abRows = ab ? [...ab.rows].sort((a, b) => b.p_d - a.p_d) : [];
      function specF() {
        const lo = Math.min(...abRows.flatMap((r) => [r.sensor_aware.accuracy, r.naive.accuracy])) * 100;
        return {
          height: 270, ariaLabel: "Share of runs with the right answer as the sensor catches fewer hops: sensor-aware versus sensor-naive scoring",
          x: caughtAxis(Math.min(...pdTicks(abRows)), pdTicks(abRows)),
          y: { label: "Right answer (%)", min: Math.min(50, Math.floor((lo - 5) / 10) * 10), max: 100, format: String },
          series: [
            { id: "aware", label: "OTS", color: C().ots, marker: "all", emphasis: true, points: abRows.map((r) => [Math.round(r.p_d * 100), 100 * r.sensor_aware.accuracy]) },
            { id: "naive", label: "Naive", color: C().base, marker: "all", points: abRows.map((r) => [Math.round(r.p_d * 100), 100 * r.naive.accuracy]) }],
          tooltip: (x) => { const r = abRows.find((q) => Math.round(q.p_d * 100) === x); return { title: `Sensor catches ${x}% of hops`, rows: [
            { value: pct0(r.sensor_aware.accuracy), label: "OTS right", color: C().ots }, { value: pct0(r.naive.accuracy), label: "Sensor-naive right", color: C().base },
            r.sensor_aware.mean_hops != null ? { value: d1(r.sensor_aware.mean_hops), label: "OTS hops to lock (mean)" } : null,
            r.naive.mean_hops != null ? { value: d1(r.naive.mean_hops), label: "Naive hops to lock (mean)" } : null].filter(Boolean) }; },
        };
      }
      function renderF() {
        if (!abRows.length) { Ff.k.textContent = "This experiment is not in the data."; return; }
        const w = abRows[abRows.length - 1];
        const half = w.p_d === 0.5 ? "half the hops" : Math.round((1 - w.p_d) * 100) + "% of hops";
        Ff.k.textContent = `With ${half} missed, OTS is right ${pct0(w.sensor_aware.accuracy)} of the time; the same bank without its sensor model, ${pct0(w.naive.accuracy)}`;
        const faster = w.sensor_aware.mean_hops != null && w.naive.mean_hops != null && w.sensor_aware.mean_hops < w.naive.mean_hops;
        Ff.d.textContent = "Same experts, same detections: the only change is whether the scoring expects missed hops and ghost signals."
          + (faster ? ` It also locks on sooner (${Math.round(w.sensor_aware.mean_hops)} vs ${Math.round(w.naive.mean_hops)} hops on average at ${Math.round(w.p_d * 100)}%).` : "");
        const gens = ab.generators.map((g) => OTS.emitName(g)).join(", ").replace(/, ([^,]*)$/, " and $1");
        Ff.foot.textContent = `Separate controlled test: ${gens}, ${ab.n_seeds} seeds each (${ab.n_seeds * ab.generators.length} runs per point), about ${ghosts(ab.p_fa)} ghost signals per hop. Same for every filter.`;
        Ff.table = () => table("Sensor-aware vs sensor-naive scoring",
          [{ h: "Sensor catches" }, { h: "OTS right", us: true }, { h: "Naive right" }, { h: "OTS hops", us: true }, { h: "Naive hops" }],
          abRows.map((r) => [Math.round(r.p_d * 100) + "%", pct0(r.sensor_aware.accuracy), pct0(r.naive.accuracy),
            r.sensor_aware.mean_hops == null ? "—" : d1(r.sensor_aware.mean_hops), r.naive.mean_hops == null ? "—" : d1(r.naive.mean_hops)]));
      }

      // ========================================================= claim 7
      const G = shell(7, "Calls the next hop");
      const gBox = el("div");
      G.body.append(legend([sq("OTS", C().ots), sq("Frequency statistics", C().base), { label: `Random guess (${chanceTxt()})`, kind: "dash" }], "▲ higher is better"), gBox);
      function specG() {
        const rows = byEmitterK(filter), by = new Map(rows.map((r) => [r.id, r]));
        return {
          height: 330, ariaLabel: "Share of next hops called correctly for each hidden pattern: OTS versus a frequency-statistics forecaster",
          categories: rows.map((r) => ({ id: r.id, label: r.name })),
          series: [{ id: "ots", label: "OTS", color: C().ots }, { id: "mk", label: "Frequency statistics", color: C().base }],
          value: (c, s) => 100 * (s === "ots" ? by.get(c).top1 : by.get(c).markov_top1),
          y: { label: "Next hop called right (%)", max: 100, ticks: [0, 25, 50, 75, 100], format: String },
          x: { label: "Hidden hop pattern" }, barWidth: 20, valueLabels: roomy(gBox, rows.length), valueFormat: (x) => String(Math.round(x)),
          refLines: [{ y: 100 * S.chance() }],
          tooltip: (c) => { const r = by.get(c.id); return { title: `${r.name} · ${condName(filter)}`, rows: [
            { value: pct0(r.top1), label: "OTS, whole run", color: C().ots, kind: "square" },
            r.top1_locked != null ? { value: pct0(r.top1_locked), label: "OTS, after it locks on" } : null,
            { value: pct0(r.markov_top1), label: "Frequency statistics", color: C().base, kind: "square" },
            { value: chanceTxt(), label: "Random guess" }].filter(Boolean) }; },
        };
      }
      function renderG() {
        const o = overK(filter), ch = S.chance(), pre = filter === "all" ? "" : condName(filter) + " sensor: ";
        G.k.textContent = pre + `OTS calls ${pct0(o.top1)} of next hops right: ${times(o.top1 / o.markov_top1)}× as many as frequency statistics (${pct0(o.markov_top1)}) and ${times(o.top1 / ch)}× a random guess`;
        const cs = emitRow("csprng", filter);
        G.d.textContent = "Counted over each whole run, including the hops before OTS has locked on"
          + (o.top1_locked != null ? `; once locked, it calls ${pct0(o.top1_locked)} right.` : ".")
          + (cs && Math.abs(cs.top1 - ch) < 0.02 ? ` On the unbreakable ${csName} nobody beats a random guess, as it should be.` : "");
        const rows = byEmitterK(filter);
        G.foot.textContent = `${fmt.int(o.n)} runs on ${condPhrase(filter)}; each run calls hops 2 to ${DATA.window}. Frequency statistics: the usual forecaster that counts which channel tends to follow which, on the same detections.`;
        G.table = () => table(`Next hop called right · ${condName(filter)}`,
          [{ h: "Hidden pattern" }, { h: "Runs" }, { h: "OTS whole run", us: true }, { h: "OTS after lock", us: true }, { h: "Frequency stats" }],
          rows.map((r) => [r.name, fmt.int(r.n), pct0(r.top1), r.top1_locked == null ? "never locked" : pct0(r.top1_locked), pct0(r.markov_top1)]));
      }

      // ========================================================= claim 8
      const H = shell(8, "Predicts an algorithm it was never built for");
      const hBox = el("div"), hVerd = el("div", { class: "proof-verdicts" });
      H.body.append(legend([sq("OTS predictor · after it learns the rule", C().ots),
        { label: `Random guess (${chanceTxt()})`, kind: "dash" }], "▲ higher is better"), hBox, hVerd);
      const hasNew = () => byEmitterNew("all").length > 0;
      function specH() {
        const rows = byEmitterNew(filter), by = new Map(rows.map((r) => [r.id, r]));
        return {
          height: 300, ariaLabel: "Share of next hops the predictor calls right for each new algorithm the bank has no expert for",
          categories: rows.map((r) => ({ id: r.id, label: r.name })),
          series: [{ id: "pred", label: "OTS predictor", color: C().ots }],
          value: (c) => 100 * (by.get(c).pred_top1_steady == null ? 0 : by.get(c).pred_top1_steady),
          y: { label: "Next hop called right (%)", max: 100, ticks: [0, 25, 50, 75, 100], format: String },
          x: { label: "New algorithm · no expert in the bank" }, barWidth: 28,
          valueLabels: roomy(hBox, rows.length), valueFormat: (x) => String(Math.round(x)),
          refLines: [{ y: 100 * S.chance() }],
          tooltip: (c) => { const r = by.get(c.id); return { title: `${r.name} · ${condName(filter)}`, rows: [
            { value: pct0(r.pred_top1_steady), label: "OTS predictor, after it learns", color: C().ots, kind: "square" },
            { value: pct0(r.pred_top1), label: "OTS predictor, whole run" },
            { value: pct0(r.ots_top1), label: "bank-only forecast" },
            { value: chanceTxt(), label: "random guess" }] }; },
        };
      }
      function renderH() {
        const rows = byEmitterNew(filter);
        if (!rows.length) { H.k.textContent = "No new-algorithm runs in this data."; return; }
        const oN = overN(filter), ch = S.chance(), pre = filter === "all" ? "" : condName(filter) + " sensor: ";
        const best = rows.reduce((a, b) => ((b.pred_top1_steady || 0) > (a.pred_top1_steady || 0) ? b : a));
        H.k.textContent = pre + `With no expert for these schemes, OTS still calls ${pct0(oN.pred_top1_steady)} of next hops right once it has learned the rule`
          + ` — up to ${pct0(best.pred_top1_steady)} on the ${best.name}, about ${times(oN.pred_top1_steady / ch)}× a random guess`;
        if (!pre) H.k.textContent = H.k.textContent[0].toUpperCase() + H.k.textContent.slice(1);
        H.d.textContent = "These schemes are held out of the bank entirely — nothing in it was built to recognise them, so OTS cannot reliably name which one it is. "
          + "The universal predictor learns each one's rule from the hops it hears and forecasts the next channel anyway: not knowing the algorithm's name does not make the next hop unpredictable.";
        fill(hVerd, rows.map((r) => status("good", "✓",
          `${r.name}: ${pct0(r.pred_top1_steady)} of next hops called right, with no expert built for it`)));
        H.foot.textContent = `${fmt.int(oN.n)} runs on ${condPhrase(filter)} across ${rows.length} schemes with no expert in the bank. `
          + "“After it learns” counts hops in the second half of each run; the whole-run and bank-only forecast numbers are in the tooltip and table.";
        H.table = () => table(`New-algorithm next-hop prediction · ${condName(filter)}`,
          [{ h: "New algorithm" }, { h: "Runs" }, { h: "After it learns", us: true }, { h: "Whole run", us: true }, { h: "Bank-only forecast" }, { h: "Random guess" }],
          rows.map((r) => [r.name, fmt.int(r.n), pct0(r.pred_top1_steady), pct0(r.pred_top1), pct0(r.ots_top1), chanceTxt()]));
      }

      // ---------------------------------------------------------- layout
      const page = el("div", { class: "page proof" },
        hero, bar, tiles,
        section("Naming the hidden pattern", "Which algorithm drives the radio? A wrong answer wastes effort; a false “I cracked it” is worse."),
        A.node, el("div", { class: "grid-2" }, B.node, Cc.node),
        section("When the sensor gets worse", "Real sensors miss hops and hear ghosts. These claims show what that does to each method."),
        Dd.node, el("div", { class: "grid-2" }, Ee.node, Ff.node),
        section("Calling the next hop", "The payoff of naming a pattern: once it and its secret are known, every next hop can be predicted."),
        G.node,
        hasNew() ? section("Predicting a brand-new algorithm", "The bank is limited by design. When the radio uses a scheme with no expert at all, naming it fails — yet the next channel can still be predicted.") : null,
        hasNew() ? H.node : null);
      root.append(page);

      // charts mount after the page is in the DOM so they know their width
      const mk = (box, make, spec) => { const h = make(box, spec()); handles.push(h); return { h, spec }; };
      mk(dAcc, OTS.chart.line, specDAcc);
      mk(dHops, OTS.chart.line, specDHops);
      if (bmRows.length) mk(eBox, OTS.chart.line, specE);
      if (abRows.length) mk(fBox, OTS.chart.line, specF);
      /** Column charts that follow the filter (and re-check value-label room on resize). */
      const filtered = [mk(aBox, OTS.chart.columns, specA), mk(bBox, OTS.chart.columns, specB),
        mk(cBox, OTS.chart.columns, specC), mk(gBox, OTS.chart.columns, specG),
        ...(hasNew() ? [mk(hBox, OTS.chart.columns, specH)] : [])];
      const redrawFiltered = () => { for (const c of filtered) c.h.update(c.spec()); };

      // ---------------------------------------------------------- render
      const cards = [A, B, Cc, Dd, Ee, Ff, G, ...(hasNew() ? [H] : [])];
      function renderTables() {
        for (const c of cards) {
          c.tbl.hidden = !OTS.prefs.tables || !c.table;
          if (!c.tbl.hidden) fill(c.tbl, c.table()); else OTS.clear(c.tbl);
        }
        tblInp.checked = !!OTS.prefs.tables;
      }
      function renderFiltered() {
        for (const b of segBtns) {
          const on = b.dataset.f === filter;
          b.classList.toggle("sel", on); b.setAttribute("aria-pressed", String(on));
        }
        const p = OTS.preset(filter);
        barHelp.textContent = p ? `Catches ${Math.round(p.p_d * 100)}% of hops · ~${ghosts(p.p_fa)} ghosts per hop`
          : `${fmt.int(all.n)} runs, ${nCond} sensors, strong to contested`;
        renderTiles(); renderA(); renderB(); renderC(); renderG();
        if (hasNew()) renderH();
        redrawFiltered();
        renderTables();
      }
      renderD(); renderE(); renderF();
      renderFiltered();

      offs.push(OTS.on("prefs", renderTables));

      // value labels switch on/off with the available width (phones vs laptops)
      let raf = 0;
      const onResize = () => {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(redrawFiltered);
      };
      window.addEventListener("resize", onResize);

      // keep the sticky filter row just under the (sticky) top bar
      const topbar = OTS.$(".topbar");
      const setTop = () => page.style.setProperty("--proof-top", (topbar ? topbar.offsetHeight : 0) + "px");
      const ro = new ResizeObserver(setTop);
      if (topbar) ro.observe(topbar);
      setTop();

      return () => {
        offs.forEach((off) => off());
        handles.forEach((h) => h.destroy());
        window.removeEventListener("resize", onResize);
        cancelAnimationFrame(raf);
        ro.disconnect();
        OTS.tip.hide();
      };
    },
  });
})();
