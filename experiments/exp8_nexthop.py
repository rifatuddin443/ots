"""Experiment 8: next-hop prediction (logged for journal reuse, paper Section 8).

The bank's posterior-weighted next-hop distribution is reported here (next-hop
log-loss and top-k hit rate) against a uniform baseline and an online Markov
predictor, on the whole window (no early stop). The paper only reports
identification, but these runs are logged so the thesis mixture reuses them
without re-simulating.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import style
from fhew.channels import ChannelSet
from fhew.logging_utils import load_json, save_json
from fhew.metrics import nexthop_summary
from fhew.sensor import SensorModel
from fhew.sim import run_bank

NAME = "exp8_nexthop"


def _markov_uniform_baselines(ep, M):
    """Online first-order Markov predictor and a uniform predictor, on clean hops."""
    counts = np.ones((M, M))
    last = None
    ll_mk, ll_un, t1_mk = [], [], []
    true = ep.true_channels
    for t, o in enumerate(ep.observations):
        if t >= len(true) - 1:
            break
        nxt = int(true[t + 1])
        pmk = counts[last] / counts[last].sum() if last is not None else np.full(M, 1.0 / M)
        ll_mk.append(-math.log(max(pmk[nxt], 1e-12)))
        t1_mk.append(int(np.argmax(pmk) == nxt))
        ll_un.append(math.log(M))
        if len(o.detections) == 1:
            c = o.detections[0].channel
            if last is not None:
                counts[last, c] += 1
            last = c
    return (float(np.mean(ll_mk)), float(np.mean(t1_mk)), float(np.mean(ll_un)))


def compute(*, M=C.DEFAULT_M, window=C.DEFAULT_WINDOW, n_seeds=30, p_d=0.85, p_fa=0.02,
            generators=("lfsr", "weak_seed", "lcg", "markov", "csprng")) -> dict:
    ch = ChannelSet.of_size(M)
    specs = C.default_specs(M)
    tgt = C.default_target_labels(specs)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    bank = C.make_bank(ch, window, sensor, specs)
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "p_d": p_d, "p_fa": p_fa,
                    "uniform_logloss": math.log(M)}, "series": {}}
    for gen in generators:
        bank_res = []
        mk_ll, mk_t1, un_ll = [], [], []
        for s in C.seeds(n_seeds):
            ep = C.episode_for(gen, ch, sensor, window, s, target_labels=tgt)
            bank_res.append(run_bank(bank, ep, stop_early=False, eval_nexthop=True))
            a, b, c = _markov_uniform_baselines(ep, M)
            mk_ll.append(a); mk_t1.append(b); un_ll.append(c)
        bank_nh = nexthop_summary(bank_res)
        out["series"][gen] = {
            "bank": bank_nh,
            "markov_baseline": {"logloss": float(np.mean(mk_ll)), "top1": float(np.mean(mk_t1))},
            "uniform_baseline": {"logloss": float(np.mean(un_ll))},
        }
    return out


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    gens = list(results["series"].keys())
    x = np.arange(len(gens))
    w = 0.27
    bank = [results["series"][g]["bank"]["logloss"] for g in gens]
    mk = [results["series"][g]["markov_baseline"]["logloss"] for g in gens]
    un = results["meta"]["uniform_logloss"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x - w, bank, w, label="bank (posterior-weighted)", color=style.PALETTE["blue"])
    ax.bar(x, mk, w, label="online Markov", color=style.PALETTE["orange"])
    ax.axhline(un, color=style.PALETTE["grey"], ls="--", label=f"uniform (log M={un:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(gens)
    ax.set_ylabel("next-hop log-loss (nats, lower is better)")
    ax.set_title(f"Next-hop prediction ($P_d={results['meta']['p_d']}$, "
                 f"$P_{{fa}}={results['meta']['p_fa']}$)")
    ax.legend()
    return style.save(fig, path)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=30)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    jpath = C.RESULTS_DIR / f"{NAME}.json"
    results = load_json(jpath) if args.plot_only else compute(n_seeds=args.n_seeds)
    if not args.plot_only:
        save_json(results, jpath)
    fig = plot(results, C.FIG_DIR / f"{NAME}.png")
    print(f"[{NAME}] wrote {jpath} and {fig}")
    for g, d in results["series"].items():
        print(f"  {g:10s} bank_ll={d['bank']['logloss']:.2f} top1={d['bank']['top1']:.2f} "
              f"| markov_ll={d['markov_baseline']['logloss']:.2f} "
              f"| uniform_ll={results['meta']['uniform_logloss']:.2f}")


if __name__ == "__main__":
    main()
