"""Training-data generation and parameter grounding.

Simulation is the deliverable, but parameter ranges are calibrated from public
sources rather than invented, and every parameter is labelled *calibrated* or
*assumed* (paper Section 11, "Datasets and grounding"). The :data:`GROUNDING`
table records that provenance; :func:`grounding_report` emits it next to
results. Replace the placeholder ranges with values measured from your own
captures (e.g. RFUAV for FHSS hop rate / dwell / band occupancy) before quoting
numbers as calibrated.

This module also builds the labelled episode set that trains the learned
classifier baseline. It never leaks ground truth into features: the classifier
sees only the sensor output, exactly like every other method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fhew.baselines.features import episode_features
from fhew.baselines.learned import LearnedClassifier
from fhew.channels import ChannelSet
from fhew.logging_utils import save_json
from fhew.sensor import SensorModel
from fhew.sim.episode import make_episode

# --------------------------------------------------------------------- grounding
# source: 'calibrated:<dataset>' once tied to a measured range, else 'assumed'.
GROUNDING = {
    "n_channels": {"value": 16, "source": "assumed",
                   "note": "power-of-two channel set; set from the real hop set size"},
    "hop_rate_hz": {"range": [100, 1000], "source": "assumed",
                    "note": "calibrate from RFUAV / drone-RF FHSS captures"},
    "dwell_ms": {"range": [1, 10], "source": "assumed",
                 "note": "calibrate from measured dwell times"},
    "p_d": {"range": [0.5, 1.0], "source": "assumed",
            "note": "detection probability; depends on receiver + geometry"},
    "p_fa": {"range": [0.0, 0.1], "source": "assumed",
             "note": "per-channel false-alarm rate; set from the sensor's ROC"},
    "window_slots": {"range": [40, 150], "source": "assumed",
                     "note": "short-engagement window; set from the scenario"},
    "lfsr_length": {"range": [5, 9], "source": "assumed",
                    "note": "register length searched by the recce prior"},
    "weak_keyspace": {"range": [64, 4096], "source": "assumed",
                      "note": "seed keyspace size; grades key strength"},
}


def grounding_report() -> dict:
    """Return the grounding table plus a summary of calibrated vs assumed."""
    calibrated = [k for k, v in GROUNDING.items() if v["source"].startswith("calibrated")]
    assumed = [k for k, v in GROUNDING.items() if v["source"] == "assumed"]
    return {"parameters": GROUNDING, "calibrated": calibrated, "assumed": assumed,
            "note": "Every number is labelled; replace 'assumed' ranges with measured ones."}


# --------------------------------------------------------- training-data sampling
#: how to sample each class's hidden parameters (the recce *ranges*, not the truth)
def sample_params(name: str, rng: np.random.Generator, n_channels: int) -> dict:
    if name == "lfsr":
        return {"length": int(rng.choice([5, 6, 7]))}
    if name == "filtered_lfsr":
        return {"length": int(rng.choice([9, 11]))}
    if name == "lcg":
        return {}
    if name == "weak_seed":
        return {"keyspace": int(rng.choice([128, 256, 512]))}
    if name == "markov":
        return {"alpha": float(rng.choice([0.1, 0.2, 0.3]))}
    if name == "csprng":
        return {}
    return {}


def generate_training_set(
    channels: ChannelSet,
    *,
    in_bank_classes=("lfsr", "weak_seed", "lcg", "markov"),
    unknown_classes=("csprng", "filtered_lfsr"),
    n_per_class: int = 200,
    windows=(40, 70, 100),
    p_d_grid=(0.6, 0.8, 1.0),
    p_fa_grid=(0.0, 0.02, 0.05),
    seed: int = 0,
):
    """Build a labelled feature set for the learned classifier.

    Returns ``(X, y, classes, episodes_meta)``. In-bank families keep their own
    label; the ``unknown_classes`` are all folded into a single ``'unknown'``
    label so the classifier has an explicit none-of-the-above class, matching the
    bank's H_0.
    """
    rng = np.random.default_rng(seed)
    classes = list(in_bank_classes) + ["unknown"]
    class_index = {c: i for i, c in enumerate(classes)}

    X, y, meta = [], [], []
    plan = list(in_bank_classes) + list(unknown_classes)
    for name in plan:
        target = name if name in in_bank_classes else "unknown"
        for _ in range(n_per_class):
            pd = float(rng.choice(p_d_grid))
            pfa = float(rng.choice(p_fa_grid))
            w = int(rng.choice(windows))
            sensor = SensorModel(p_d=pd, p_fa=pfa)
            ep = make_episode(name, channels, sensor, w, rng,
                              gen_params=sample_params(name, rng, channels.size))
            X.append(episode_features(ep.observations, channels))
            y.append(class_index[target])
            meta.append({"true": name, "target": target, "p_d": pd, "p_fa": pfa, "window": w})
    return np.asarray(X), np.asarray(y, dtype=np.int64), classes, meta


def train_classifier(channels: ChannelSet, *, seed: int = 0, n_per_class: int = 250,
                     l2: float = 3e-2, hidden: int = 32) -> tuple[LearnedClassifier, dict]:
    """Train the learned baseline with a held-out split for an honest accuracy.

    The baseline is deliberately a *feature* classifier (no sequence model); its
    characteristic failure -- folding LCG and weak-seed into 'unknown' because
    their summary statistics look random -- is the gap the sequential bank fills,
    and is reported as such.
    """
    X, y, classes, meta = generate_training_set(channels, n_per_class=n_per_class, seed=seed)
    rng = np.random.default_rng(seed + 7)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    clf = LearnedClassifier(classes, l2=l2, hidden=hidden, seed=seed).fit(X[tr], y[tr])
    acc_tr = float((np.argmax(clf.predict_proba(X[tr]), 1) == y[tr]).mean())
    acc_te = float((np.argmax(clf.predict_proba(X[te]), 1) == y[te]).mean())
    # refit on all data for deployment
    clf = LearnedClassifier(classes, l2=l2, hidden=hidden, seed=seed).fit(X, y)
    return clf, {"train_accuracy": acc_tr, "test_accuracy": acc_te,
                 "n": int(len(y)), "classes": classes, "hidden": hidden, "l2": l2}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate FH-EW training data and classifier.")
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="data")
    args = p.parse_args(argv)

    ch = ChannelSet.of_size(args.channels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clf, info = train_classifier(ch, seed=args.seed, n_per_class=args.n_per_class)
    save_json(clf.to_dict(), out / "learned_classifier.json")
    save_json(info, out / "learned_classifier_info.json")
    save_json(grounding_report(), out / "grounding.json")
    print(f"trained learned classifier: train_acc={info['train_accuracy']:.3f} "
          f"n={info['n']} classes={info['classes']}")
    print(f"wrote {out/'learned_classifier.json'}, grounding.json")


if __name__ == "__main__":
    main()
