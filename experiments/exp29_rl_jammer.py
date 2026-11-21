"""Experiment 29: the end-to-end learned jammer baseline (thesis Section 12).

The OTS jammer is a *pipeline*: a sensor-aware identification bank, a calibrated
next-channel predictor, and a threshold-aware allocator. A fair question is what a
*monolithic learned* jammer -- one policy mapping the sensor stream straight to a
power vector, with no identification, predictor, or calibration -- achieves against
the same adversaries, as the baseline that shows "what the pipeline's structure is
worth" (and what the distribution-free guarantee costs). :mod:`fhew.rl.jammer`
provides it; this experiment measures it.

Two panels:

* **A -- denial by jammer, per hopper family.** Scripted spot / barrage, the OTS
  pipeline, and the learned PPO jammer, each against an LFSR (structured,
  algebraically predictable), a peaky and a broad first-order Markov (statistically
  predictable), and a keyed CSPRNG (unpredictable). The learned jammer *matches* the
  pipeline where structure is first-order statistical (Markov) but *loses on the
  LFSR* -- which it cannot crack without the pipeline's algebraic expert -- and both
  collapse to the floor on the CSPRNG. That contrast is the point: end-to-end
  learning recovers the statistical structure for free but not the algebraic
  structure the pipeline's experts encode.
* **B -- learning curves.** Denial vs policy-gradient training iteration for the
  learned jammer against a learnable (peaky Markov) hopper and an unlearnable keyed
  CSPRNG: denial climbs and plateaus on the Markov source as the policy learns its
  transition structure, and stays pinned at the barrage floor on the CSPRNG -- the
  predictability boundary, reached by a learned jammer with no hand-built identify /
  predict stage.

A distributionally-robust (CVaR-over-adversaries) training objective is available in
the library (:func:`fhew.rl.jammer.train_jammer_robust`, and the per-episode
``PPOConfig.cvar_alpha``). It is not the headline here because, in this environment,
harder-to-predict adversaries are also inherently *less jammable*, so the worst-case
adversary has little headroom and the robust objective trades mean denial for only a
small worst-case gain -- the predictability boundary bounds robustness too.

Every jammer reads only the sensor stream and acts through the shared
``EffectModel`` (one g and tau), so the comparison is on one physics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.allocator import EffectModel
from fhew.channels import ChannelSet
from fhew.duel import OTSJammer
from fhew.generators import make_generator
from fhew.logging_utils import load_json, save_json
from fhew.rl.agents import PPOConfig
from fhew.rl.env import (EngagementEnv, GeneratorHopper, OTSJammerPolicy, ScriptedJammer,
                         run_engagement)
from fhew.rl.jammer import RLJammerAgent, train_jammer
from fhew.rl.psro import fast_ots_specs
from fhew.sensor import SensorModel

NAME = "exp29_rl_jammer"

# Hopper families, drawn with a FIXED key per family so the structure is learnable
# across episodes (a fresh key each episode would be a different emitter each time).
FAMILIES = {
    "lfsr": ("lfsr", {"length": 7}),
    "markov_peaky": ("markov", {"alpha": 0.05}),
    "markov_broad": ("markov", {"alpha": 0.3}),
    "csprng": ("csprng", {}),
}
FAM_LABELS = {"lfsr": "LFSR", "markov_peaky": "Markov (peaky)",
              "markov_broad": "Markov (broad)", "csprng": "CSPRNG"}


def _fixed_hopper(fam: str, key_seed: int):
    gen_name, kw = FAMILIES[fam]
    return make_generator(gen_name, 16, rng=np.random.default_rng(key_seed), **kw)


def _eval(jammer, fam: str, env, *, n_seeds: int, key_seed: int) -> list[float]:
    """Per-episode denial of a (reset-able) jammer against a fixed-key hopper family."""
    out = []
    for s in range(n_seeds):
        hop = GeneratorHopper(_fixed_hopper(fam, key_seed))
        rec = run_engagement(hop, jammer, env, np.random.default_rng(500 + s))
        out.append(rec.denied_frac())
    return out


def compute(*, M=C.DEFAULT_M, window=100, n_seeds=12, budget=3.0, max_tones=4,
            p_d=0.95, p_fa=0.01, train_iters=40, train_episodes=12, cvar_alpha=0.4,
            key_seed=77, quick=False, **_) -> dict:
    if quick:
        n_seeds, train_iters, train_episodes, window = 6, 20, 10, 80
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    effect = EffectModel(ch, tau=1.0)
    env = EngagementEnv(ch, sensor, effect, window=window)

    # Fixed jammers (no training).
    spot = ScriptedJammer(ch, effect, budget, mode="spot", label="spot")
    barrage = ScriptedJammer(ch, effect, budget, mode="barrage", n_tones=max_tones,
                             label="barrage")
    ots = OTSJammerPolicy(OTSJammer(ch, sensor, effect, budget=budget, max_tones=max_tones,
                                    specs=fast_ots_specs()), label="ots_pipeline")

    panelA = {}
    for fam in FAMILIES:
        def opp(fam=fam):
            return GeneratorHopper(_fixed_hopper(fam, key_seed))
        # A learned specialist jammer per family (it knows which family it faces).
        rlj = train_jammer(opp, ch, effect, env, budget=budget, max_tones=max_tones,
                           n_iters=train_iters, episodes_per_iter=train_episodes,
                           cfg=PPOConfig(), rng=np.random.default_rng(10)).freeze()
        row = {}
        for name, jam in (("spot", spot), ("barrage", barrage), ("ots_pipeline", ots),
                          ("rl_jammer", rlj)):
            d = _eval(jam, fam, env, n_seeds=n_seeds, key_seed=key_seed)
            row[name] = {"denied_mean": float(np.mean(d)), "denied_std": float(np.std(d))}
        panelA[fam] = row

    # Panel B: learning curves -- denial vs training iteration on a learnable
    # (peaky Markov) hopper and an unlearnable CSPRNG.
    eval_seeds = max(n_seeds, 16)
    total_iters = max(train_iters, 40)
    eval_every = max(4, total_iters // 10)
    curves = {}
    for fam in ("markov_peaky", "csprng"):
        curves[fam] = _learning_curve(ch, effect, env, fam, key_seed, budget, max_tones,
                                      total_iters=total_iters, eval_every=eval_every,
                                      episodes_per_iter=train_episodes, eval_seeds=eval_seeds,
                                      seed=20)
    panelB = {"curves": curves, "eval_every": eval_every, "total_iters": total_iters,
              "eval_seeds": eval_seeds}

    return {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "budget": budget,
                     "max_tones": max_tones, "p_d": p_d, "p_fa": p_fa,
                     "train_iters": train_iters, "train_episodes": train_episodes,
                     "cvar_alpha": cvar_alpha},
            "panelA": panelA, "panelB": panelB}


def _learning_curve(ch, effect, env, fam, key_seed, budget, max_tones, *, total_iters,
                    eval_every, episodes_per_iter, eval_seeds, seed) -> dict:
    """Denial vs training iteration for a learned jammer against one hopper family."""
    agent = RLJammerAgent(ch, effect, budget, rng=np.random.default_rng(seed),
                          max_tones=max_tones, training=True)
    base = _fixed_hopper(fam, key_seed)
    cfg = PPOConfig()
    rng = np.random.default_rng(seed)

    def eval_denial() -> float:
        fr = agent.freeze()
        return float(np.mean([run_engagement(GeneratorHopper(base), fr, env,
                     np.random.default_rng(800 + s)).denied_frac() for s in range(eval_seeds)]))

    iters, denial = [0], [eval_denial()]
    done = 0
    while done < total_iters:
        for _ in range(eval_every):
            for _ in range(episodes_per_iter):
                run_engagement(GeneratorHopper(base), agent, env, rng)
                agent.end_episode()
            agent.learn(cfg)
            done += 1
        iters.append(done)
        denial.append(eval_denial())
    return {"iters": iters, "denial": denial}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    # Panel A: grouped bars, jammer type per hopper family.
    fams = list(results["panelA"].keys())
    jammers = ["spot", "barrage", "ots_pipeline", "rl_jammer"]
    jlabels = {"spot": "spot", "barrage": "barrage", "ots_pipeline": "OTS pipeline",
               "rl_jammer": "learned PPO jammer"}
    colors = {"spot": style.PALETTE["grey"], "barrage": style.PALETTE["sky"],
              "ots_pipeline": style.PALETTE["blue"], "rl_jammer": style.PALETTE["vermillion"]}
    x = np.arange(len(fams)); w = 0.2
    for k, jm in enumerate(jammers):
        vals = [results["panelA"][f][jm]["denied_mean"] for f in fams]
        axA.bar(x + (k - 1.5) * w, vals, w, label=jlabels[jm], color=colors[jm])
    axA.set_xticks(x); axA.set_xticklabels([FAM_LABELS[f] for f in fams], fontsize=8)
    axA.set_ylabel("fraction of hops denied"); axA.set_ylim(0, 1.05)
    axA.set_title("A. Learned jammer vs pipeline, per hopper family")
    axA.legend(fontsize=8, loc="upper right", ncol=2)

    # Panel B: learning curves on a learnable vs unlearnable hopper.
    b = results["panelB"]
    cm = {"markov_peaky": style.PALETTE["vermillion"], "csprng": style.PALETTE["grey"]}
    for fam, c in b["curves"].items():
        axB.plot(c["iters"], c["denial"], "-o", ms=4, color=cm[fam], label=FAM_LABELS[fam])
    axB.axhline(1.0 / results["meta"]["M"], color=style.PALETTE["black"], ls=":", lw=1.0,
                alpha=0.6, label="1/M (chance)")
    axB.set_xlabel("policy-gradient training iteration")
    axB.set_ylabel("fraction of hops denied")
    axB.set_ylim(0, max(0.6, max(max(c["denial"]) for c in b["curves"].values()) + 0.08))
    axB.set_title("B. Learned jammer: learns structure, floors on CSPRNG")
    axB.legend(fontsize=8, loc="center right")

    fig.tight_layout()
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=12)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    if args.plot_only:
        results = load_json(jpath)
    else:
        results = compute(n_seeds=args.n_seeds, quick=args.quick)
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    for fam, row in results["panelA"].items():
        print(f"  {fam:13s} " + "  ".join(
            f"{k}={row[k]['denied_mean']:.2f}" for k in ("spot", "barrage", "ots_pipeline", "rl_jammer")))
    b = results["panelB"]
    for fam, c in b["curves"].items():
        print(f"  learning curve {fam:13s}: {c['denial'][0]:.2f} -> {c['denial'][-1]:.2f} "
              f"over {c['iters'][-1]} iters")


if __name__ == "__main__":
    main()
