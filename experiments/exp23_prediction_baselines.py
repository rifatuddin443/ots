"""Experiment 23: external next-channel prediction baselines on the lossy stream.

The proposed predictor is compared, on *identical* episodes and through the
same lossy sensor, against predictors that are not built from the bank:

* ``markov1`` -- an online first-order Markov counter (clean slots only);
* ``ppm``     -- an online variable-order PPM-C predictor (orders 0-4), the
  standard training-free sequence predictor;
* ``lstm``    -- a two-layer LSTM *trained offline* on simulated episodes of the
  bank's own families at the test sensor (the "trained per family" predictor of
  the FH-prediction literature), then run on the test stream. It reads the raw
  detection mask of every slot, so it sees everything the bank sees.

Alongside sit the paper's two internal reference points, built exactly as in
Experiment 14 and run on its episodes: ``proposed`` (the Fixed-Share predictor
carrying the training-free universal and rule-search experts) and ``id_only``
(the same mixture over the identification experts alone).

The question is generalisation: the LSTM is trained on the in-bank families and
tested on an in-family generator (LFSR), on the three held-out families it never
saw, and on the keyed CSPRNG control. Metric: steady-state top-1 next-channel
accuracy (final 40 slots), chance ``1/M``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import common as C
from experiments import exp14_next_channel_prediction as E14
from experiments import style
from fhew.baselines import torch_available
from fhew.channels import ChannelSet
from fhew.generators import make_generator
from fhew.logging_utils import load_json, save_json
from fhew.metrics import bootstrap_ci
from fhew.predict import ChannelPredictor
from fhew.sensor import SensorModel

NAME = "exp23_prediction_baselines"

SENSORS = {"strong": (0.95, 0.01), "typical": (0.8, 0.02), "contested": (0.6, 0.05)}
TEST_GENERATORS = ("lfsr", "lagged_fib", "quadratic_cg", "sine_map", "csprng")
TRAIN_FAMILIES = ("lfsr", "filtered_lfsr", "lcg", "weak_seed", "logistic", "markov")
METHODS = ("proposed", "id_only", "markov1", "ppm", "lstm")
STEADY = 40


# ------------------------------------------------------------- online baselines
def _clean(obs):
    return obs.detections[0].channel if len(obs.detections) == 1 else None


class Markov1:
    """Online first-order Markov counter fed by clean (single-detection) slots."""

    def __init__(self, M: int):
        self.M = M
        self.reset()

    def reset(self):
        self.counts = np.ones((self.M, self.M))
        self.last = None

    def observe(self, obs):
        c = _clean(obs)
        if c is None:
            self.last = None            # the chain is broken by a gap or a ghost
            return
        if self.last is not None:
            self.counts[self.last, c] += 1.0
        self.last = c

    def predict(self) -> np.ndarray:
        if self.last is None:
            return np.full(self.M, 1.0 / self.M)
        row = self.counts[self.last]
        return row / row.sum()


class PPM:
    """Online PPM-C (escape method C), orders 0..max_order, on the clean stream.

    A context is the last ``k`` *consecutive* clean channels; a gap or an
    ambiguous slot empties the context, as it must for any counter that has no
    sensor model.
    """

    def __init__(self, M: int, max_order: int = 4):
        self.M, self.K = M, max_order
        self.reset()

    def reset(self):
        self.tables = [dict() for _ in range(self.K + 1)]
        self.hist: list[int] = []

    def observe(self, obs):
        c = _clean(obs)
        if c is None:
            self.hist = []
            return
        for k in range(min(self.K, len(self.hist)) + 1):
            ctx = tuple(self.hist[len(self.hist) - k:]) if k else ()
            row = self.tables[k].get(ctx)
            if row is None:
                row = self.tables[k][ctx] = np.zeros(self.M)
            row[c] += 1.0
        self.hist = (self.hist + [c])[-self.K:]

    def predict(self) -> np.ndarray:
        p = np.full(self.M, 1.0 / self.M)
        for k in range(min(self.K, len(self.hist)) + 1):
            ctx = tuple(self.hist[len(self.hist) - k:]) if k else ()
            row = self.tables[k].get(ctx)
            if row is None:
                continue
            n, d = row.sum(), float(np.count_nonzero(row))
            p = (row + d * p) / (n + d)
        return p


# ------------------------------------------------------------------ LSTM baseline
def _sense_masks(true: np.ndarray, M: int, p_d: float, p_fa: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Vectorised twin of ``SensorModel.observe``: detection masks for a batch."""
    onehot = np.eye(M, dtype=bool)[true]                       # (B, T, M)
    hit = rng.random(true.shape)[..., None] < p_d
    fa = rng.random(true.shape + (M,)) < p_fa
    return ((onehot & hit) | (fa & ~onehot)).astype(np.float32)


def _training_sequences(M: int, window: int, n_per_family: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seqs = []
    for fam in TRAIN_FAMILIES:
        for _ in range(n_per_family):
            params = ({"keyspace": E14.WEAK_KEYSPACE} if fam == "weak_seed"
                      else C.draw_episode_params(fam, rng, lcg_keyspace=E14.LCG_SEED_MAX))
            g = make_generator(fam, M, rng=rng, **params)
            seqs.append(g.sequence(window))
    return np.asarray(seqs, dtype=np.int64)


def train_lstm(M: int, window: int, p_d: float, p_fa: float, *, n_per_family: int,
               steps: int, hidden: int = 128, batch: int = 128, seed: int = 0):
    """Train the LSTM next-channel predictor on the in-bank families at one sensor.

    Input per slot is the detection mask plus an "empty slot" flag; the target
    is the *true* next channel, which simulation supplies (a real system would
    have to learn from the lossy stream alone, so this is the generous setting).
    Sensor noise is redrawn for every batch.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    seqs = _training_sequences(M, window, n_per_family, seed)
    rng = np.random.default_rng(seed + 1)
    net = _make_net(M, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lossf = nn.CrossEntropyLoss()
    loss_hist = []
    for _ in range(steps):
        idx = rng.integers(0, len(seqs), size=batch)
        true = seqs[idx]
        x = torch.from_numpy(_lstm_input(_sense_masks(true, M, p_d, p_fa, rng)))
        y = torch.from_numpy(true[:, 1:])
        logits = net(x)[:, :-1]
        loss = lossf(logits.reshape(-1, M), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_hist.append(float(loss.detach()))
    net.eval()
    return net, {"n_train_sequences": int(len(seqs)), "steps": steps, "hidden": hidden,
                 "batch": batch, "final_loss": float(np.mean(loss_hist[-50:])),
                 "uniform_loss": float(np.log(M))}


def _make_net(M: int, hidden: int):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(M + 1, hidden, num_layers=2, batch_first=True)
            self.head = nn.Linear(hidden, M)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out)

    return Net()


def _lstm_input(masks: np.ndarray) -> np.ndarray:
    empty = (masks.sum(-1, keepdims=True) == 0).astype(np.float32)
    return np.concatenate([masks, empty], axis=-1)


def _lstm_calls(net, observations) -> np.ndarray:
    """The LSTM's next-channel call after each slot of one episode."""
    import torch

    masks = np.stack([o.detection_mask() for o in observations]).astype(np.float32)[None]
    with torch.no_grad():
        return net(torch.from_numpy(_lstm_input(masks)))[0].argmax(-1).numpy()


# ------------------------------------------------------------------ evaluation
def _steady_top1(calls: np.ndarray, true: np.ndarray) -> float:
    """calls[t] is the prediction for slot t+1 made after observing slot t."""
    hits = calls[:-1] == true[1:]
    return float(hits[-STEADY:].mean())


def _online_calls(model, observations) -> np.ndarray:
    model.reset()
    calls = np.empty(len(observations), dtype=np.int64)
    for t, o in enumerate(observations):
        model.observe(o)
        calls[t] = int(np.argmax(model.predict()))
    return calls


def _mixture_calls(pred: ChannelPredictor, observations) -> np.ndarray:
    pred.reset()
    calls = np.empty(len(observations), dtype=np.int64)
    for t, o in enumerate(observations):
        pred.observe(o)
        calls[t] = pred.predict_top1()
    return calls


def compute(*, M=C.DEFAULT_M, window=120, n_seeds=30, sensors=SENSORS,
            lstm_per_family=3000, lstm_steps=2000, **_) -> dict:
    ch = ChannelSet.of_size(M)
    use_lstm = torch_available()
    methods = [m for m in METHODS if m != "lstm" or use_lstm]
    out = {"meta": {"M": M, "window": window, "n_seeds": n_seeds, "chance": 1.0 / M,
                    "steady_window": STEADY, "methods": methods,
                    "train_families": list(TRAIN_FAMILIES),
                    "n_episodes": n_seeds * len(TEST_GENERATORS) * len(sensors)},
           "sensors": {}}
    for name, (p_d, p_fa) in sensors.items():
        sensor = SensorModel(p_d=p_d, p_fa=p_fa)
        proposed = ChannelPredictor(ch, sensor, specs=E14._pred_specs(M))
        id_only = ChannelPredictor(ch, sensor, specs=E14._id_specs(M))
        tgt = C.default_target_labels(E14._id_specs(M))
        markov1, ppm = Markov1(M), PPM(M)
        net, info = (train_lstm(M, window, p_d, p_fa, n_per_family=lstm_per_family,
                                steps=lstm_steps) if use_lstm else (None, {}))
        per_gen = {}
        for gen in TEST_GENERATORS:
            acc = {m: [] for m in methods}
            for s in C.seeds(n_seeds):
                ep = E14._episode(gen, ch, sensor, window, s, tgt)
                true = np.asarray(ep.true_channels)
                acc["proposed"].append(_steady_top1(_mixture_calls(proposed, ep.observations), true))
                acc["id_only"].append(_steady_top1(_mixture_calls(id_only, ep.observations), true))
                acc["markov1"].append(_steady_top1(_online_calls(markov1, ep.observations), true))
                acc["ppm"].append(_steady_top1(_online_calls(ppm, ep.observations), true))
                if use_lstm:
                    acc["lstm"].append(_steady_top1(_lstm_calls(net, ep.observations), true))
            per_gen[gen] = {m: {"mean": float(np.mean(v)),
                                "ci": [float(x) for x in bootstrap_ci(v)]}
                            for m, v in acc.items()}
        out["sensors"][name] = {"p_d": p_d, "p_fa": p_fa, "lstm": info, "generators": per_gen}
    return out


LABELS = {"proposed": "proposed", "id_only": "id-only bank", "markov1": "Markov-1",
          "ppm": "PPM-C", "lstm": "LSTM (trained)"}
GEN_LABELS = {"lfsr": "LFSR\n(in-bank)", "lagged_fib": "lagged-Fib.\n(held out)",
              "quadratic_cg": "quad.-CG\n(held out)", "sine_map": "sine map\n(held out)",
              "csprng": "CSPRNG\n(control)"}


def plot(results: dict, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    names = list(results["sensors"])
    methods = results["meta"]["methods"]
    colors = dict(zip(METHODS, (style.PALETTE["blue"], style.PALETTE["orange"],
                                style.PALETTE["grey"], style.PALETTE["green"],
                                style.PALETTE["vermillion"])))
    fig, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(TEST_GENERATORS))
    w = 0.8 / len(methods)
    for ax, name in zip(axes, names):
        S = results["sensors"][name]
        for i, m in enumerate(methods):
            vals = [S["generators"][g][m]["mean"] for g in TEST_GENERATORS]
            lo = [v - S["generators"][g][m]["ci"][0] for v, g in zip(vals, TEST_GENERATORS)]
            hi = [S["generators"][g][m]["ci"][1] - v for v, g in zip(vals, TEST_GENERATORS)]
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w, yerr=[lo, hi],
                   color=colors[m], label=LABELS[m], error_kw={"lw": 0.8})
        ax.axhline(results["meta"]["chance"], color="#555", ls=":", lw=1)
        ax.set_xticks(x, [GEN_LABELS[g] for g in TEST_GENERATORS], fontsize=8.5)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{name} ($P_d$={S['p_d']:g}, $P_{{fa}}$={S['p_fa']:g})")
    axes[0].set_ylabel("steady-state top-1")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
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
    print(f"[{NAME}] wrote {jpath} and {fig}  ({results['meta']['n_episodes']} episodes)")
    methods = results["meta"]["methods"]
    for name, S in results["sensors"].items():
        print(f"  {name}:  " + "  ".join(f"{m:>9s}" for m in methods))
        for g, d in S["generators"].items():
            print(f"    {g:13s}" + "  ".join(f"{d[m]['mean']:9.2f}" for m in methods))
        if S["lstm"]:
            print(f"    (LSTM train loss {S['lstm']['final_loss']:.2f} nats vs uniform "
                  f"{S['lstm']['uniform_loss']:.2f})")


if __name__ == "__main__":
    main()
