/* Decision analysis view (stub). */
OTS.views.register({
  id: "analysis", nav: "Decision analysis", title: "Inside the decision", hint: "Confidence and search, same run",
  subtitle: "", icon: "M3 3v18h18M7 15l4-4 3 3 5-6", usesPlayer: true,
  mount(root) {
    root.append(OTS.el("div", { class: "card" }, "Inside the decision: coming soon"));
    return () => {};
  },
});
