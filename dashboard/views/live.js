/* Live demo view (stub). */
OTS.views.register({
  id: "live", nav: "Live demo", title: "Watch it work", hint: "One engagement, hop by hop",
  subtitle: "", icon: "M5 3l14 9-14 9V3z", usesPlayer: true,
  mount(root) {
    root.append(OTS.el("div", { class: "card" }, "Watch it work: coming soon"));
    return () => {};
  },
});
