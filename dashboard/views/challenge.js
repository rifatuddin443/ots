/* Challenge view (stub). */
OTS.views.register({
  id: "challenge", nav: "Challenge", title: "Beat the algorithm", hint: "Can you guess the next hop?",
  subtitle: "", icon: "M12 2a10 10 0 100 20 10 10 0 000-20zm0 6v4l3 2", usesPlayer: false,
  mount(root) {
    root.append(OTS.el("div", { class: "card" }, "Beat the algorithm: coming soon"));
    return () => {};
  },
});
