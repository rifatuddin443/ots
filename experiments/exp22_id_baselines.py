"""Experiment 22: identification baselines on the paper's full bank.

The paper's identification bank (one expert per family, plus the generative
``H_0``) against three conventional identifiers, on *identical* episodes and
the *same* sensor-aware likelihoods:

* ``pairwise_sprt`` -- Wald-threshold pairwise SPRT over the same experts, no
  none-of-the-above class (a closed-set test);
* ``fixed_length``  -- a fixed-sample-size likelihood-ratio test (30 hops);
* ``learned``       -- a feature classifier trained on simulated episodes drawn
  from the same reconnaissance ranges, with an explicit ``unknown`` class,
  deciding once on the whole window (its best case).

Generators are the seven rows of the paper's identification table plus the
three held-out "new algorithm" families (target: ``unknown``; no method has an
expert or a training class for them). Reported per method and sensor:

* accuracy on the in-bank generators (declared family == true family);
* how often the keyed CSPRNG is correctly reported as ``unknown``;
* the *hard* false-ID rate over every episode -- naming an algebraic family
  that is not the truth, the dangerous error;
* mean hops to a correct declaration.

A paired bootstrap on per-episode correctness accompanies the comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.baselines import FixedLengthTest, LearnedClassifier, PairwiseSPRT, episode_features
from fhew.channels import ChannelSet
from fhew.config import GENERATOR_TARGET_FAMILY
from fhew.experts import build_hypothesis
from fhew.likelihood import SensorAwareEmission
from fhew.logging_utils import load_json, save_json
from fhew.metrics import bootstrap_ci, paired_bootstrap_diff
from fhew.sensor import SensorModel
from fhew.sim import decide_outcome, make_episode, run_bank, run_fixed_length, run_sprt

NAME = "exp22_id_baselines"

SENSORS = {"typical": (0.8, 0.02), "contested": (0.6, 0.05)}
IN_BANK = ("lfsr", "filtered_lfsr", "lcg", "weak_seed", "logistic", "markov")
CONTROL = ("csprng",)
HELD_OUT = ("lagged_fib", "quadratic_cg", "sine_map")
GENERATORS = IN_BANK + CONTROL + HELD_OUT
ALGEBRAIC = frozenset({"lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed"})
METHODS = ("bank", "pairwise_sprt", "fixed_length", "learned")


def _train_classifier(ch: ChannelSet, *, n_per_class: int, seed: int) -> tuple[LearnedClassifier, dict]:
    """Train the learned baseline on the bank's own reconnaissance ranges.

    Classes are the bank's families plus ``unknown`` (trained on the CSPRNG
    only: the held-out families stay held out for the classifier too).
    """
    rng = np.random.default_rng(seed)
    classes = sorted({GENERATOR_TARGET_FAMILY[g] for g in IN_BANK}) + ["unknown"]
    index = {c: i for i, c in enumerate(classes)}
    X, y = [], []
    for gen in IN_BANK + CONTROL:
        for _ in range(n_per_class):
            p_d = float(rng.choice([0.6, 0.8, 0.95, 1.0]))
            p_fa = float(rng.choice([0.0, 0.01, 0.02, 0.05]))
            w = int(rng.choice([40, 70, 100, 140]))
            ep = make_episode(gen, ch, SensorModel(p_d=p_d, p_fa=p_fa), w, rng,
                              gen_params=C.draw_episode_params(gen, rng))
            X.append(episode_features(ep.observations, ch))
            y.append(index[GENERATOR_TARGET_FAMILY[gen]])
    X, y = np.asarray(X), np.asarray(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    held = LearnedClassifier(classes, l2=3e-2, hidden=32, seed=seed).fit(X[perm[:cut]], y[perm[:cut]])
    test_acc = float((np.argmax(held.predict_proba(X[perm[cut:]]), 1) == y[perm[cut:]]).mean())
    clf = LearnedClassifier(classes, l2=3e-2, hidden=32, seed=seed).fit(X, y)
    return clf, {"classes": classes, "n_train": int(len(y)), "heldout_accuracy": test_acc}


def _run_learned(clf: LearnedClassifier, ep):
    """The classifier's best case: one decision on the whole window.

    (Letting it stop early on its own confidence makes it declare after two or
    three hops and collapse, which would be a strawman.)
    """
    proba = clf.predict_proba(episode_features(ep.observations, ep.channels)[None])[0]
    return decide_outcome(clf.label, ep, clf.classes[int(np.argmax(proba))], True, ep.window)


def _rate(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci": [float("nan")] * 2, "n": 0}
    lo, hi = bootstrap_ci(arr)
    return {"mean": float(arr.mean()), "ci": [float(lo), float(hi)], "n": int(arr.size)}


def _summarise(results) -> dict:
    inb = [r for r in results if r.true_label in IN_BANK]
    ctl = [r for r in results if r.true_label in CONTROL]
    hops = [r.tau_stop for r in inb if r.correct and r.declared]
    hard = [r.decision_label in ALGEBRAIC and r.decision_label != r.target_label
            for r in results]
    return {
        "in_bank_accuracy": _rate([r.correct for r in inb]),
        "csprng_unknown": _rate([r.decision_label == "unknown" for r in ctl]),
        "hard_false_id": _rate(hard),
        "hard_false_id_out_of_bank": _rate(
            [h for h, r in zip(hard, results) if r.true_label not in IN_BANK]),
        "mean_hops": float(np.mean(hops)) if hops else float("nan"),
        "per_generator": {
            g: {"accuracy": float(np.mean([r.correct for r in results if r.true_label == g])),
                "hard_false_id": float(np.mean([h for h, r in zip(hard, results)
                                                if r.true_label == g]))}
            for g in GENERATORS},
    }


def compute(*, M=C.DEFAULT_M, window=140, n_seeds=100, n_per_class=300,
            sensors=SENSORS, **_) -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.id_specs(M)
    clf, clf_info = _train_classifier(ch, n_per_class=n_per_class, seed=0)
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds,
                    "generators": list(GENERATORS), "experts": [s.label for s in specs],
                    "n_episodes": n_seeds * len(GENERATORS) * len(sensors),
                    "classifier": clf_info, "fixed_length_n": 30},
           "sensors": {}}
    for name, (p_d, p_fa) in sensors.items():
        sensor = SensorModel(p_d=p_d, p_fa=p_fa)
        emit = SensorAwareEmission(sensor)
        bank = C.make_bank(ch, window, sensor, specs)
        sprt = PairwiseSPRT([build_hypothesis(s, ch, window) for s in specs], emit, M)
        fixedl = FixedLengthTest([build_hypothesis(s, ch, window) for s in specs], emit, M,
                                 n_fixed=30)
        res = {m: [] for m in METHODS}
        for s in C.seeds(n_seeds):
            for gen in GENERATORS:
                ep = C.episode_for(gen, ch, sensor, window, s)
                ep.target_label = GENERATOR_TARGET_FAMILY.get(gen, "unknown")
                res["bank"].append(run_bank(bank, ep, eval_nexthop=False))
                res["pairwise_sprt"].append(run_sprt(sprt, ep))
                res["fixed_length"].append(run_fixed_length(fixedl, ep))
                res["learned"].append(_run_learned(clf, ep))
        summ = {m: _summarise(r) for m, r in res.items()}
        sig = {}
        inb = {m: [r for r in rr if r.true_label in IN_BANK] for m, rr in res.items()}
        for m in METHODS[1:]:
            sig[m] = paired_bootstrap_diff(inb["bank"], inb[m], metric="correct")
        out["sensors"][name] = {"p_d": p_d, "p_fa": p_fa, "methods": summ,
                                "in_bank_accuracy_bank_minus": sig}
    return out


LABELS = {"bank": "bank + $H_0$\n(proposed)", "pairwise_sprt": "pairwise\nSPRT",
          "fixed_length": "fixed-length\nLRT", "learned": "learned\nclassifier"}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    names = list(results["sensors"])
    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(METHODS))
    w = 0.38
    for ax, name in zip(axes, names):
        S = results["sensors"][name]
        acc = [S["methods"][m]["in_bank_accuracy"]["mean"] for m in METHODS]
        hard = [S["methods"][m]["hard_false_id"]["mean"] for m in METHODS]
        ax.bar(x - w / 2, acc, w, color=style.PALETTE["blue"], label="in-bank accuracy")
        ax.bar(x + w / 2, hard, w, color=style.PALETTE["vermillion"], label="hard false-ID")
        for xi, a, h in zip(x, acc, hard):
            ax.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=9)
            ax.text(xi + w / 2, h + 0.015, f"{h:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x, [LABELS[m] for m in METHODS], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{name} ($P_d$={S['p_d']:g}, $P_{{fa}}$={S['p_fa']:g})")
    axes[0].set_ylabel("rate")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}  ({results['meta']['n_episodes']} episodes)")
    for name, S in results["sensors"].items():
        print(f"  {name}:")
        for m in METHODS:
            d = S["methods"][m]
            print(f"    {m:14s} in-bank acc={d['in_bank_accuracy']['mean']:.3f} "
                  f"csprng->unknown={d['csprng_unknown']['mean']:.3f} "
                  f"hard false-ID={d['hard_false_id']['mean']:.3f} "
                  f"(out-of-bank {d['hard_false_id_out_of_bank']['mean']:.3f}) "
                  f"hops={d['mean_hops']:.1f}")


if __name__ == "__main__":
    main()
