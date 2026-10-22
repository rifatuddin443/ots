/* Story view (stub with chart-kit smoke test). */
OTS.views.register({
  id: "story", nav: "Story", title: "Why it matters", hint: "The 60-second version for anyone",
  subtitle: "", icon: "M3 12h4l3 8 4-16 3 8h4", usesPlayer: false,
  mount(root) {
    const { el, S, fmt } = OTS;
    const c1 = el("div"), c2 = el("div");
    root.append(el("div", { class: "page" },
      el("div", { class: "card" }, el("div", { class: "card-h" }, el("div", {}, el("p", { class: "card-t" }, "Test"), el("p", { class: "card-k" }, "Wrong answers per 100"))),
        OTS.chart.legend([{ label: "OTS", color: OTS.tok("accent"), kind: "square" }, { label: "Standard test", color: OTS.tok("baseline"), kind: "square" }]), c1),
      el("div", { class: "card" }, c2)));
    const rows = S.byEmitter("all");
    const h1 = OTS.chart.columns(c1, { categories: rows.map((r) => ({ id: r.id, label: r.name })),
      series: [{ id: "o", label: "OTS", color: OTS.tok("accent") }, { id: "b", label: "Standard test", color: OTS.tok("baseline") }],
      value: (c, s) => { const r = rows.find((x) => x.id === c); return 100 * (1 - (s === "o" ? r.ots_acc : r.base_acc)); },
      y: { label: "Wrong answers per 100 runs", max: 100, format: (v) => String(Math.round(v)) }, x: { label: "Hidden hop generator" } });
    const tr = OTS.tracesFor("lfsr", "typical")[0];
    const h2 = OTS.chart.line(c2, { x: { label: "Hop number", min: 1, max: 30 }, y: { label: "Probability (%)", min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => Math.round(v * 100) + "%" },
      series: tr.labels.map((id, k) => ({ id, label: OTS.hypName(id), color: OTS.hypColor(id), points: tr.post.slice(0, 30).map((p, j) => [j + 1, p[k]]) })),
      refLines: [{ y: 0.95, label: "Lock 95%" }], vLines: [{ x: tr.tau, label: "Locked" }] });
    return () => { h1.destroy(); h2.destroy(); };
  },
});
