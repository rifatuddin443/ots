/* Proof view (stub). */
OTS.views.register({
  id: "proof", nav: "Proof", title: "Proof at scale", hint: "Hundreds of runs, side by side",
  subtitle: "", icon: "M4 20V10m6 10V4m6 16v-7m4 7H2", usesPlayer: false,
  mount(root) {
    root.append(OTS.el("div", { class: "card" }, "Proof at scale: coming soon"));
    return () => {};
  },
});
