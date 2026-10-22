/* Decision analysis view: WHY OTS decided, for the same engagement as the Live view.
 *
 * Reads the shared OTS.player (trace + hops shown t) and redraws on every 'load' / 'tick':
 *   tiles   hops to lock · candidate settings left · sensor mistakes absorbed
 *   card 1  how sure OTS is about each pattern (posterior lines, 95% lock line)
 *   card 2  needle in a haystack (candidate secret settings still possible, log scale)
 *   card 3  calling the next hop (cumulative hit rate + one square per hop since the lock)
 *   card 4  the standard test (pairwise SPRT, no "unknown" option) on the same hops
 * Every number is computed from the trace; the context notes use the 600-run scoreboard.
 */
(function () {
  "use strict";
  const { el, fmt, T } = OTS;

  /** "Show all N hops" toggle for the decision charts; kept across visits in this session. */
  let showAll = false;

  // ------------------------------------------------------------------ styles
  const CSS = `
.analysis-tiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.analysis-soft { color: var(--ink-2); font-size: 26px; font-weight: 600; letter-spacing: 0; }
.analysis-toggle { margin-left: auto; padding-top: 2px; }
.analysis-chart { margin-top: 4px; }
.analysis-foot { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin-top: 12px; padding-top: 12px;
  border-top: 1px solid var(--grid); font-size: 13.5px; color: var(--ink-2); }
.analysis-foot .mono { color: var(--ink); font-size: 13px; }
.analysis-calls { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--grid); }
.analysis-calls .legend { margin: 0; }
.analysis-strip { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 10px; }
.analysis-strip i { width: 10px; height: 10px; border-radius: 2px; }
.analysis-strip i.hit { background: var(--accent); }
.analysis-strip i.miss { background: var(--muted); opacity: 0.5; }
.analysis-sum { font-size: 13.5px; color: var(--ink-2); margin: 10px 0 0; }
.analysis-sum b { color: var(--ink); font-weight: 600; }
.analysis-truth { font-size: 14px; color: var(--ink-2); margin: 0 0 12px; }
.analysis-truth b { color: var(--ink); font-weight: 600; }
.analysis-vs { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.analysis-side { border: 1px solid var(--line); border-radius: 11px; background: var(--surface); padding: 14px 16px;
  display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.analysis-side.ots { border-color: rgba(34, 211, 238, 0.4); }
.analysis-who { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; font-weight: 600; font-size: 14px; }
.analysis-who small { color: var(--muted); font-weight: 400; font-size: 12.5px; }
.analysis-what { font: 650 22px/1.25 var(--sans); color: var(--ink); overflow-wrap: anywhere; }
.analysis-when { color: var(--ink-2); font-size: 13.5px; }
.analysis-side .status { align-self: flex-start; margin-top: 4px; }
.analysis-note { margin: 14px 0 0; font-size: 13.5px; color: var(--ink-2); border-left: 3px solid var(--line-2);
  padding: 3px 0 3px 12px; max-width: 92ch; }
.analysis-note b { color: var(--ink); font-weight: 600; }
@media (max-width: 700px) {
  .analysis-tiles, .analysis-vs { grid-template-columns: minmax(0, 1fr); }
  .analysis-what { font-size: 19px; }
  .analysis-soft { font-size: 22px; }
  .analysis-toggle { margin-left: 0; }
}`;
  function injectCss() {
    if (document.getElementById("css-analysis")) return;
    document.head.append(el("style", { id: "css-analysis" }, CSS));
  }

  // ----------------------------------------------------------------- helpers
  const isUnknown = (id) => id === "unknown" || id === "H0";
  /** Pattern name for sentences: plain "Unknown" instead of "Unknown (H₀)". */
  const plainName = (id) => (isUnknown(id) ? "Unknown" : OTS.hypName(id));
  const quoted = (id) => (isUnknown(id) ? "“Unknown”" : plainName(id));
  /** Patterns that carry a recoverable secret (seed, register state, map parameter). */
  const hasSecret = (id) => !isUnknown(id) && id !== "markov";
  /** Did the lock name the right pattern and recover the radio's actual secret? */
  const foundSecret = (tr) => tr.correct && (tr.secret_match === "exact" || tr.secret_match === "equivalent");
  const plural = (n, one, many) => fmt.int(n) + " " + (n === 1 ? one : many || one + "s");
  const pctAxis = (v) => Math.round(v * 100) + "%";
  const aliveAt = (tr, t) => (t <= 0 ? tr.search_space : tr.alive[t - 1]);
  const narrow = () => window.innerWidth < 700;
  const status = (kind, icon, text) => el("span", { class: "status " + kind },
    el("span", { class: "ic", "aria-hidden": "true" }, icon), el("span", {}, text));
  const key = (color) => el("i", { class: "sq", style: `background:${color}` });
  /** Replace a node's children; nested arrays are flattened and null/false skipped (like OTS.el). */
  const fill = (node, ...kids) => {
    OTS.clear(node).append(...kids.flat(Infinity).filter((k) => k != null && k !== false && k !== ""));
    return node;
  };

  /** Whole-number hop ticks (the chart kit's nice ticks may land on 2.5). */
  function hopTicks(min, max) {
    const span = max - min, step = [1, 2, 5, 10, 20, 25, 50, 100, 200].find((s) => span / s <= 6) || 500;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max; v += step) out.push(v);
    if (min === 0 && out[0] !== 0) out.unshift(0);
    return out;
  }
  /** X range of the decision charts: zoomed to the lock (or the play head) unless "show all". */
  function decisionMax(tr, t) {
    if (showAll) return tr.window;
    const anchor = T.lockedAt(tr, t) ? tr.tau : t;
    return Math.min(tr.window, Math.max(25, anchor + 12));
  }

  /** Next-hop calls per slot j >= 1 (hop j + 1): {hit, ch, heard}. Cached per trace. */
  const callCache = new WeakMap();
  function callsFor(tr) {
    if (callCache.has(tr)) return callCache.get(tr);
    const out = [];
    for (let j = 1; j < tr.window; j++) {
      const f = T.forecastTop(tr, j);
      if (f) out[j] = { hit: f.ch === tr.truth[j], ch: f.ch, heard: tr.dets[j].includes(tr.truth[j]) };
    }
    callCache.set(tr, out);
    return out;
  }

  /** Where the run stands at hop t, for both methods. */
  function stateOf(tr, t) {
    const locked = T.lockedAt(tr, t), ended = t >= tr.window;
    const b = tr.baseline, bTau = b.tau == null ? tr.window : b.tau;
    return { locked, ended, otsDone: locked || ended, bTau, baseDone: t >= bTau };
  }

  // ===================================================================== view
  OTS.views.register({
    id: "analysis", nav: "Decision analysis", title: "Inside the decision", hint: "Confidence and search, same run",
    subtitle: "Why OTS decided, hop by hop, on the same run as the Live demo",
    icon: "M3 3v18h18M7 15l4-4 3 3 5-6", usesPlayer: true,
    mount(root) {
      injectCss();
      const offs = [], charts = {};
      let lastTrace = null, resizeRaf = 0;

      // ----------------------------------------------------------- skeleton
      const tile = (label) => {
        const v = el("div", { class: "tile-v num" }), s = el("div", { class: "tile-s" });
        return { node: el("div", { class: "tile" }, el("div", { class: "tile-l" }, label), v, s), v, s };
      };
      const tiles = { lock: tile("Hops to lock"), alive: tile("Candidate settings left"), sensor: tile("Sensor mistakes absorbed") };

      const head = (title, extra) => {
        const k = el("p", { class: "card-k" }), d = el("p", { class: "card-d" });
        return { node: el("div", { class: "card-h" }, el("div", {}, el("p", { class: "card-t" }, title), k, d), extra || null), k, d };
      };

      // card 1: confidence per pattern
      const allBox = el("input", { type: "checkbox" });
      allBox.checked = showAll;
      allBox.addEventListener("change", () => { showAll = allBox.checked; render(); });
      const allText = el("span", {}, "Show all hops");
      const h1 = head("How sure is OTS about each pattern?", el("label", { class: "switch analysis-toggle" }, allBox, allText));
      const legend1 = el("div");
      const box1 = el("div", { class: "analysis-chart" });
      const card1 = el("section", { class: "card" }, h1.node, legend1, box1);

      // card 2: needle in a haystack
      const h2 = head("Needle in a haystack");
      const box2 = el("div", { class: "analysis-chart" });
      const foot2 = el("div", { class: "analysis-foot" });
      const card2 = el("section", { class: "card" }, h2.node, box2, foot2);

      // card 3: calling the next hop
      const h3 = head("Calling the next hop");
      const box3 = el("div", { class: "analysis-chart" });
      const strip = el("div", { class: "analysis-strip", role: "img" });
      const sum3 = el("p", { class: "analysis-sum" });
      const calls3 = el("div", { class: "analysis-calls" },
        OTS.chart.legend([{ label: "Called right", color: OTS.tok("accent"), kind: "square" }, { label: "Missed", color: OTS.tok("muted"), kind: "square" }]),
        strip, sum3);
      const card3 = el("section", { class: "card" }, h3.node, box3, calls3);

      // card 4: the standard test on the same hops
      const h4 = head("Same hops, standard test");
      const truth4 = el("p", { class: "analysis-truth" });
      const side = (cls, color, who, tag) => {
        const what = el("div", { class: "analysis-what" }), when = el("div", { class: "analysis-when" }), chip = el("div");
        return { node: el("div", { class: "analysis-side " + cls }, el("div", { class: "analysis-who" }, key(color), who, el("small", {}, tag)), what, when, chip), what, when, chip };
      };
      const sOts = side("ots", OTS.tok("accent"), "OTS", "can answer “Unknown”");
      const sBase = side("", OTS.tok("baseline"), "Standard test", "must always name a pattern");
      const note4 = el("p", { class: "analysis-note" });
      const card4 = el("section", { class: "card" }, h4.node, truth4, el("div", { class: "analysis-vs" }, sOts.node, sBase.node), note4);

      const empty = el("div", { class: "card empty", hidden: true }, "No engagement is loaded. Pick a radio pattern in Settings.");
      const page = el("div", { class: "page analysis" },
        el("div", { class: "analysis-tiles" }, tiles.lock.node, tiles.alive.node, tiles.sensor.node),
        card1, el("div", { class: "grid-2" }, card2, card3), card4);
      root.append(empty, page);

      // ------------------------------------------------------------- tiles
      function setTile(tl, value, sub, hot) {
        fill(tl.v, value);
        tl.s.textContent = sub;
        tl.node.classList.toggle("hot", !!hot);
      }
      function renderTiles(tr, t, st) {
        const soft = (txt) => el("span", { class: "analysis-soft" }, txt);
        const L = T.leader(tr, t);
        if (st.locked) {
          setTile(tiles.lock, [String(tr.tau), el("small", {}, tr.tau === 1 ? " hop" : " hops")],
            `Locked on ${plainName(tr.decision)}, ${fmt.pct(T.post(tr, tr.tau, tr.decision))} sure`, tr.correct);
        } else if (t === 0) setTile(tiles.lock, soft("—"), "Press Play to start the run", false);
        else if (!st.ended) setTile(tiles.lock, soft("searching…"), `Hop ${t}: ${plainName(L.id)} leads at ${fmt.pct(L.p)} (needs 95%)`, false);
        else setTile(tiles.lock, soft("No lock"), isUnknown(tr.decision)
          ? `Answered “Unknown” after ${tr.window} hops` : `Best guess after ${tr.window} hops: ${plainName(tr.decision)}`, false);

        const a = aliveAt(tr, t);
        const sub = t === 0 ? "possible secret settings before any hop"
          : a === 0 ? "no exact setting fits the hops"
          : a === 1 && st.locked && foundSecret(tr) ? "only one still fits: the secret"
          : (a === 1 ? "setting" : "settings") + " not yet ruled out";
        setTile(tiles.alive, [fmt.int(a), el("small", {}, " of " + fmt.int(tr.search_space))], sub, a === 1 && st.locked && tr.correct);

        const ss = T.sensorStats(tr, t);
        setTile(tiles.sensor, fmt.int(ss.miss + ss.ghost), `${plural(ss.miss, "missed hop")} · ${plural(ss.ghost, "ghost signal")}`, false);
      }

      // ------------------------------------------------- card 1: confidence
      function confidenceHeadline(tr, t, st) {
        const n = tr.labels.length;
        if (t === 0) return `All ${n} patterns start level at ${fmt.pct(1 / n)}. Press Play to watch OTS weigh them`;
        if (st.locked) {
          const k = T.idx(tr, tr.decision), p = tr.post[t - 1];
          if (p[k] < 0.9) {
            const L = T.leader(tr, t);
            return L.k !== k && L.p >= 0.5
              ? `OTS locked on ${plainName(tr.decision)} at hop ${tr.tau}; later hops shifted it to ${plainName(L.id)} (${fmt.pct(L.p)})`
              : `OTS locked on ${plainName(tr.decision)} at hop ${tr.tau}; noisy hops since then have shaken it to ${fmt.pct(p[k])}`;
          }
          const ruled = p.every((v, i) => i === k || v < 0.01);
          return `OTS crossed the 95% line on ${plainName(tr.decision)} at hop ${tr.tau}` +
            (ruled ? ` and ruled out the other ${n - 1} patterns` : ", and the rivals share the rest");
        }
        if (!st.ended) {
          const L = T.leader(tr, t);
          if (isUnknown(L.id)) return `After ${plural(t, "hop")}, no known pattern fits: “Unknown” leads at ${fmt.pct(L.p)}`;
          return `After ${plural(t, "hop")}, ${plainName(L.id)} leads at ${fmt.pct(L.p)}. OTS waits for 95% before it commits`;
        }
        if (isUnknown(tr.decision)) return `No known pattern reached 95% in ${tr.window} hops, so OTS answered “Unknown”` + (tr.correct ? ": the right call" : "");
        return `No pattern reached 95% in ${tr.window} hops; the best guess, ${plainName(tr.decision)}, was never confirmed`;
      }
      function renderConfidence(tr, t, st) {
        const xmax = decisionMax(tr, t), upto = Math.min(t, xmax), lead = T.leader(tr, upto);
        h1.k.textContent = confidenceHeadline(tr, t, st);
        const series = tr.labels.map((id, k) => {
          const pts = [];
          for (let h = 1; h <= upto; h++) pts.push([h, tr.post[h - 1][k]]);
          const top = upto > 0 && k === lead.k, last = pts.length ? pts[pts.length - 1][1] : 0;
          return { id, label: OTS.hypName(id), color: OTS.hypColor(id), points: pts,
            width: top ? 3 : 2, opacity: top ? 1 : 0.6, emphasis: top, marker: top ? "end" : null,
            endLabel: top || last >= 0.05 };   // no labels piled up on the floor for ~0% lines
        });
        const spec = {
          height: narrow() ? 260 : 300,
          ariaLabel: `Line chart: chance of each of ${tr.labels.length} patterns, hop by hop. ${h1.k.textContent}.`,
          x: { label: "Hop number", min: 1, max: xmax, ticks: hopTicks(1, xmax) },
          // a little headroom above 100% keeps the lock label clear of the leading line
          y: { label: "Chance it is this pattern (%)", min: 0, max: 1.12, ticks: [0, 0.25, 0.5, 0.75, 1], format: pctAxis },
          // the 95% line is a refLine; its label is a direct end label (a one-point series) at the right edge
          series: [...series, { id: "lock", label: narrow() ? "Lock 95%" : "Lock threshold 95%", color: OTS.tok("ink-2"), points: [[xmax, 0.95]] }],
          endLabels: (narrow() ? 2 : 3) + 1,
          refLines: [{ y: 0.95 }],
          tooltip: (x) => x > upto ? { title: `Hop number: ${x}`, rows: [{ value: "—", label: "not reached yet" }] } : {
            title: `Hop number: ${x}`,
            rows: tr.labels.map((id, k) => ({ value: fmt.pct(tr.post[x - 1][k]), label: OTS.hypName(id), color: OTS.hypColor(id), y: tr.post[x - 1][k] }))
              .sort((p, q) => q.y - p.y),
          },
          vLines: st.locked && tr.tau <= xmax ? [{ x: tr.tau, label: `Locked (hop ${tr.tau})` }] : [],
          emptyText: "Press Play: the lines grow as hops arrive",
        };
        charts.conf ? charts.conf.update(spec) : (charts.conf = OTS.chart.line(box1, spec));
      }

      // ------------------------------------------------- card 2: haystack
      function firstHopAtOrBelow(tr, t, v) {
        for (let h = 1; h <= t; h++) if (tr.alive[h - 1] <= v) return h;
        return t;
      }
      function renderHaystack(tr, t, st) {
        const S = tr.search_space, a = aliveAt(tr, t), when = firstHopAtOrBelow(tr, t, a);
        const after = `after ${plural(when, "hop")}`;
        const claimable = st.locked && hasSecret(tr.decision);
        let k, d;
        if (t === 0) {
          k = `${fmt.int(S)} possible secret settings to search`;
          d = `Each pattern OTS knows has a secret setting, such as a seed or a starting state. Before any hop all ${fmt.int(S)} are possible; every hop crosses off the ones that could not have produced it.`;
        } else if (a === 0) {
          k = `${fmt.int(S)} possible secret settings → none fit ${after}`;
          d = "No exact secret setting explains these hops. That is what a statistical (Markov) or unbreakable radio looks like: there is no small secret to find" +
            (tr.decision === "markov" && st.otsDone ? ", so OTS learns the radio's hop-to-hop habits instead." : isUnknown(tr.decision) && st.otsDone ? ", so OTS answers “Unknown”." : ".");
        } else if (st.locked || !st.ended) {
          k = `${fmt.int(S)} possible secret settings → ${fmt.int(a)} ${after}`;
          d = a === 1 && claimable && foundSecret(tr) ? "Only one setting still fits every hop: the radio's secret, found by elimination."
            : a === 1 && claimable ? "Only one setting still fits every hop. Below: how it compares with the radio's real secret."
            : claimable ? `The ${fmt.int(a)} settings left all fit every hop heard so far. OTS keeps them all, weighted by how well each fits.`
            : st.locked ? "A Markov hopper has no secret setting, only habits; OTS learns those instead."
            : "Every hop crosses off settings that could not have produced it. A short list is not proof: OTS claims no secret until one pattern passes 95%.";
        } else {
          k = `No secret claimed: ${plural(a, "setting")} not ruled out, but no pattern reached 95%`;
          d = "Hops crossed off most settings, but no pattern ever reached 95%, so OTS claims no secret at all rather than guess one.";
        }
        h2.k.textContent = k;
        h2.d.textContent = d;

        const xmax = decisionMax(tr, t), upto = Math.min(t, xmax), cur = aliveAt(tr, upto);
        const FLOOR = 0.5, pts = [[0, Math.max(1, S)]];
        for (let h = 1; h <= upto; h++) pts.push([h, Math.max(FLOOR, tr.alive[h - 1])]);   // log scale: 0 is drawn on the floor
        const accent = OTS.tok("accent");
        const spec = {
          height: 300,
          ariaLabel: `Line chart, log scale: candidate secret settings still possible, hop by hop. ${k}.`,
          x: { label: "Hop number", min: 0, max: xmax, ticks: hopTicks(0, xmax) },
          y: { label: "Candidate settings still possible (log scale)", min: FLOOR, max: Math.max(10, S) * 1.6, log: true, format: fmt.int },
          series: [{ id: "alive", label: cur === 0 ? "None fit" : `${fmt.int(cur)} left`, color: accent, points: pts, marker: "end" }],
          endLabels: 1,
          vLines: st.locked && tr.tau <= xmax ? [{ x: tr.tau, label: `Locked (hop ${tr.tau})` }] : [],
          tooltip: (x) => {
            const v = aliveAt(tr, x);
            return { title: x === 0 ? "Before the first hop" : `Hop number: ${x}`,
              rows: [{ value: v === 0 ? "none" : fmt.int(v), label: v === 0 ? "no exact setting fits" : "settings still possible", color: accent }] };
          },
        };
        charts.hay ? charts.hay.update(spec) : (charts.hay = OTS.chart.line(box2, spec));

        // the needle: the secret recovered at the lock, checked against the real one
        if (st.locked && tr.secret_recovered) {
          const m = tr.secret_match;
          if (m === "model") {
            fill(foot2, status("", "≈", "No single secret"), el("span", {}, "OTS learned the radio's hop-to-hop habits instead."));
          } else {
            const chip = { exact: ["good", "✔", "Matches the real secret"], equivalent: ["good", "✔", "Different key, identical hops"],
              family: ["warn", "≈", "Right map, nearby setting"], mismatch: ["bad", "✖", "Not the real secret"] }[m];
            fill(foot2, el("span", {}, "Secret found: ", el("span", { class: "mono" }, tr.secret_recovered)),
              chip ? status(...chip) : null,
              m !== "exact" ? el("span", { class: "muted" }, "Real: ", el("span", { class: "mono" }, tr.secret_true)) : null);
          }
        } else {
          fill(foot2, el("span", { class: "muted" }, st.otsDone ? "No secret is claimed without a lock."
            : "The secret is revealed here once OTS locks."));
        }
      }

      // -------------------------------------------------- card 3: next hop
      function renderCalls(tr, t, st) {
        const calls = callsFor(tr), chance = 1 / tr.M;
        const pts = [];
        let hit = 0, n = 0;
        for (let h = 2; h <= t; h++) {
          const c = calls[h - 1]; if (!c) continue;
          n++; if (c.hit) hit++;
          pts.push([h, hit / n]);
        }
        const hs = T.hitStats(tr, t), times = (r) => Math.round(r / chance);
        let k;
        if (t < 2) k = "Before each hop, OTS names the channel it expects next";
        else if (st.locked && hs.n === 0) k = "Locked: from here on, watch OTS call each hop before it happens";
        else if (st.locked && hs.hit === hs.n) k = `Since the lock, OTS called all ${plural(hs.n, "next hop")} right`;
        else if (st.locked) k = `Since the lock, OTS called ${fmt.pct0(hs.rate)} of next hops right` +
          (hs.rate >= 2 * chance ? `: about ${times(hs.rate)}× better than guessing` : ", close to guessing");
        else if (!st.ended) k = `No lock yet: so far ${fmt.pct0(hs.rate)} of hops called right (random guessing: ${fmt.pct0(chance)})`;
        else k = hs.rate <= 2 * chance ? `OTS never locked, and its guesses stayed near random (${fmt.pct0(hs.rate)} vs ${fmt.pct0(chance)})`
          : `OTS never locked, but its best guesses still hit ${fmt.pct0(hs.rate)} (random: ${fmt.pct0(chance)})`;
        h3.k.textContent = k;
        h3.d.textContent = `The line is the share of all calls OTS got right so far. Guessing at random gets 1 in ${tr.M}.`;

        // same range as the decision charts while searching, then it follows the play head
        const xmax = showAll ? tr.window : Math.min(tr.window, Math.max(decisionMax(tr, t), t));
        const accent = OTS.tok("accent");
        const spec = {
          height: 300,
          ariaLabel: `Line chart: share of next hops OTS called right so far, hop by hop. ${k}.`,
          x: { label: "Hop number", min: 1, max: xmax, ticks: hopTicks(1, xmax) },
          y: { label: "Hops called right so far (%)", min: 0, max: 1.12, ticks: [0, 0.25, 0.5, 0.75, 1], format: pctAxis },
          series: [{ id: "ots", label: "OTS", color: accent, points: pts, marker: "end" },
            { id: "chance", label: "Random guess", color: OTS.tok("ink-2"), points: [[xmax, chance]] }],
          endLabels: 2,
          refLines: [{ y: chance }],
          vLines: st.locked ? [{ x: tr.tau, label: `Locked (hop ${tr.tau})` }] : [],
          tooltip: (x) => {
            if (x > t) return { title: `Hop number: ${x}`, rows: [{ value: "—", label: "not reached yet" }] };
            let hh = 0, nn = 0;
            for (let h = 2; h <= x; h++) { const c = calls[h - 1]; if (c) { nn++; if (c.hit) hh++; } }
            const c = calls[x - 1];
            return { title: `Hop number: ${x}`, rows: [
              { value: nn ? fmt.pct0(hh / nn) : "—", label: `called right so far (${hh} of ${nn})`, color: accent },
              c ? { value: c.hit ? "✔" : "✖", label: c.hit ? "this hop called right" : "this hop missed" } : null,
            ].filter(Boolean) };
          },
        };
        charts.calls ? charts.calls.update(spec) : (charts.calls = OTS.chart.line(box3, spec));

        // one square per hop since the lock
        OTS.clear(strip);
        if (st.locked) {
          let unheard = 0;
          for (let j = tr.tau; j < t; j++) {
            const c = calls[j]; if (!c) continue;
            if (c.hit && !c.heard) unheard++;
            strip.append(el("i", { class: c.hit ? "hit" : "miss",
              title: `Hop ${j + 1}: called channel ${fmt.ch(c.ch)}, it was ${fmt.ch(tr.truth[j])}${c.hit ? " ✔" : " ✖"}` }));
          }
          strip.setAttribute("aria-label", `${hs.hit} of ${hs.n} hops called right since the lock`);
          fill(sum3, hs.n
            ? ["Since the lock at hop ", String(tr.tau), ": ", el("b", {}, `${hs.hit} of ${hs.n}`), " called right. Random guessing would get about ",
              el("b", {}, fmt.int(hs.n * chance)), ".", unheard ? [" That includes ", el("b", {}, plural(unheard, "hop")), " the sensor never even heard."] : ""]
            : "One square per hop appears here as OTS calls them.");
        } else {
          strip.removeAttribute("aria-label");
          fill(sum3, st.ended ? "OTS never locked on this run, so it made no confident calls."
            : "One square per hop appears here once OTS locks.");
        }
      }

      // ----------------------------------------------- card 4: standard test
      function renderVersus(tr, t, st) {
        const b = tr.baseline;
        fill(truth4, "Right answer for this run: ", el("b", {}, quoted(tr.target)),
          OTS.emitName(tr.emitter) !== plainName(tr.target) ? el("span", { class: "muted" }, ` (the radio uses: ${OTS.emitName(tr.emitter)})`) : null);
        h4.d.textContent = `Both methods scored exactly the same hops. The standard test has no “unknown” option, so it must always name one of the ${tr.labels.length - 1} known patterns, even when none fits.`;

        // OTS side
        let oKind = null;   // 'right' | 'wrong' | 'safe' once decided
        if (!st.otsDone) {
          sOts.what.textContent = "Still weighing…";
          sOts.when.textContent = t ? `${plural(t, "hop")} heard, no pattern at 95% yet` : "Waiting for the first hop";
          sOts.chip.replaceChildren(status("", "…", "Testing"));
        } else {
          sOts.what.textContent = quoted(tr.decision);
          sOts.when.textContent = tr.declared ? `Locked at hop ${tr.tau}`
            : isUnknown(tr.decision) ? `No pattern reached 95% in ${tr.window} hops` : `Best guess after ${tr.window} hops (never reached 95%)`;
          oKind = tr.correct ? "right" : isUnknown(tr.decision) ? "safe" : "wrong";
          sOts.chip.replaceChildren(oKind === "right" ? status("good", "✔", "Right")
            : oKind === "safe" ? status("warn", "◐", "Missed, no false claim")
            : status("bad", "✖", T.isAlgebraic(tr.decision) && tr.declared ? "Wrong: false break" : "Wrong"));
        }

        // standard-test side
        let bKind = null;
        if (!st.baseDone) {
          sBase.what.textContent = "Still testing…";
          sBase.when.textContent = t ? `${plural(t, "hop")} heard, not sure yet` : "Waiting for the first hop";
          sBase.chip.replaceChildren(status("", "…", "Testing"));
        } else {
          sBase.what.textContent = plainName(b.decision);
          sBase.when.textContent = b.declared ? `Declared at hop ${b.tau}` : `Never sure; forced to pick after ${tr.window} hops`;
          bKind = b.correct ? "right" : T.isAlgebraic(b.decision) ? "break" : "wrong";
          sBase.chip.replaceChildren(bKind === "right" ? status("good", "✔", "Right")
            : status("bad", "✖", bKind === "break" ? "Wrong: false break" : "Wrong"));
        }

        // headline
        let k;
        if (!oKind && !bKind) k = "Both methods are still testing the same hops";
        else if (!bKind) k = "OTS has answered; the standard test is still testing";
        else if (!oKind) k = `The standard test has answered${bKind === "right" ? "" : " (wrongly)"}; OTS is still weighing the evidence`;
        else if (oKind === "right" && bKind !== "right") k = bKind === "break"
          ? `OTS got it right; the standard test claimed a false break (${plainName(b.decision)})`
          : "OTS got it right; the standard test named the wrong pattern";
        else if (oKind === "right") {
          const ot = tr.declared ? tr.tau : tr.window, bt = b.declared ? b.tau : tr.window, dd = Math.abs(ot - bt);
          k = "Both got it right" + (dd === 0 ? ", at the same hop" : bt < ot ? `; the standard test answered ${plural(dd, "hop")} sooner` : `; OTS answered ${plural(dd, "hop")} sooner`);
        } else if (bKind === "right") k = oKind === "safe" ? "The standard test got this one; OTS played safe and said “Unknown”" : "The standard test got this one; OTS did not";
        else k = oKind === "safe" && bKind === "break" ? "Neither got it right, but only the standard test made a false claim" : "Neither got this one right";
        h4.k.textContent = k;

        // context: why a miss happened, or why a false break matters (scoreboard over every run, no cherry-picking)
        const all = OTS.S.overall("all"), per = (r) => fmt.per100(r);
        let note = null;
        if (oKind && oKind !== "right") {
          if (tr.emitter === "lcg_full") {
            note = [el("b", {}, "Why OTS missed: "), "the full-state LCG is its known weak spot. Cracking it needs an unbroken run of clean hops, and a sensor that misses ",
              fmt.pct0(1 - tr.p_d), " of hops rarely delivers one. ", T.isAlgebraic(tr.decision) && tr.declared ? "" : "OTS still made no false claim of a break."];
          } else if (oKind === "safe") {
            note = [el("b", {}, "Why OTS missed: "), `no pattern reached 95% in ${tr.window} hops, so it answered “Unknown” rather than guess. A miss, but a safe one.`];
          } else if (tr.declared && T.isAlgebraic(tr.decision)) {
            note = [el("b", {}, "Why OTS missed: "), `on this run the first ${tr.tau} hops happened to fit ${plainName(tr.decision)} well enough to pass 95%. That is rare: across all ${fmt.int(all.n)} scored runs OTS made a false break `,
              el("b", {}, per(all.ots_fb)), " times per 100, the standard test ", el("b", {}, per(all.base_fb)), "."];
          } else {
            note = [el("b", {}, "Why OTS missed: "), tr.declared ? `on this run the first ${tr.tau} hops fitted ${plainName(tr.decision)} better than the real pattern.`
              : "no pattern reached 95%, and the best guess was wrong, but OTS never claimed it.",
              ` Across all ${fmt.int(all.n)} scored runs OTS names the right answer ${fmt.pct0(all.ots_acc)} of the time, the standard test ${fmt.pct0(all.base_acc)}.`];
          }
        } else if (bKind === "break") {
          note = [el("b", {}, "A false break"), " is the dangerous mistake: claiming to have cracked a pattern that is not there. Across all ",
            fmt.int(all.n), " scored runs: OTS ", el("b", {}, per(all.ots_fb)), " per 100 runs, the standard test ", el("b", {}, per(all.base_fb)), "."];
        }
        note4.hidden = !note;
        fill(note4, note);
      }

      // ------------------------------------------------------------ render
      function render() {
        const tr = OTS.player.trace;
        page.hidden = !tr; empty.hidden = !!tr;
        if (!tr) return;
        const t = OTS.player.t, st = stateOf(tr, t);
        if (tr !== lastTrace) {
          lastTrace = tr;
          OTS.clear(legend1).append(OTS.chart.legend(tr.labels.map((id) => ({ label: OTS.hypName(id), color: OTS.hypColor(id) }))));
          allText.textContent = `Show all ${tr.window} hops`;
          h1.d.textContent = `Each line is one possible pattern: the ${tr.labels.length - 1} OTS knows, plus “Unknown” for anything else. ` +
            "After every hop OTS re-scores all of them, and it commits only when one passes the dashed 95% line.";
        }
        renderTiles(tr, t, st);
        renderConfidence(tr, t, st);
        renderHaystack(tr, t, st);
        renderCalls(tr, t, st);
        renderVersus(tr, t, st);
      }

      render();

      offs.push(OTS.on("load", render), OTS.on("tick", render), OTS.on("resize", render));
      // chart heights and label counts depend on phone vs desktop width
      const onResize = () => { cancelAnimationFrame(resizeRaf); resizeRaf = requestAnimationFrame(render); };
      window.addEventListener("resize", onResize);

      return () => {
        offs.forEach((off) => off());
        window.removeEventListener("resize", onResize);
        cancelAnimationFrame(resizeRaf);
        for (const h of Object.values(charts)) h.destroy();
      };
    },
  });
})();
