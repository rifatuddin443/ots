/* How it works view (stub). */
OTS.views.register({
  id: "how", nav: "How it works", title: "How it works", hint: "The method in plain words",
  subtitle: "", icon: "M12 3l9 5-9 5-9-5 9-5zm-9 9l9 5 9-5M3 16l9 5 9-5", usesPlayer: false,
  mount(root) {
    root.append(OTS.el("div", { class: "card" }, "How it works: coming soon"));
    return () => {};
  },
});
