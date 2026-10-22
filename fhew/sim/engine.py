"""The end-to-end slot loop (paper Section 9) and per-method runners.

``run_bank`` is the paper's Algorithm 1: for each slot, observe, score every
hypothesis through the sensor-aware forward update, do the Bayesian update,
form the posterior-weighted next-hop, and test the stop rule. The baseline
runners drive the same episodes so every method is compared on identical data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fhew.baselines.fixed_length import FixedLengthTest
from fhew.baselines.learned import LearnedClassifier
from fhew.baselines.pairwise_sprt import PairwiseSPRT
from fhew.experts.bank import UNKNOWN_LABEL, HypothesisBank
from fhew.logging_utils import RunLogger
from fhew.sim.episode import Episode


@dataclass
class EpisodeResult:
    method: str
    true_label: str
    target_label: str
    decision_label: str            # label used for accuracy (declared, else MAP at window end)
    declared: bool                 # did the stop rule fire before the window ended?
    tau_stop: int                  # slot of declaration, else the window length
    correct: bool                  # decision_label == target_label
    false_id: bool                 # named a specific *wrong* generator
    kappa_final: float = 0.0
    posterior_final: dict = field(default_factory=dict)
    nexthop_logloss: float = float("nan")
    nexthop_top1: float = float("nan")
    nexthop_top3: float = float("nan")
    per_slot: list = field(default_factory=list)


def decide_outcome(method: str, episode: Episode, decision_label: str, declared: bool,
                   tau_stop: int, **extra) -> EpisodeResult:
    target = episode.target_label
    correct = decision_label == target
    false_id = (decision_label != target) and (decision_label != UNKNOWN_LABEL)
    return EpisodeResult(
        method=method, true_label=episode.true_label, target_label=target,
        decision_label=decision_label, declared=declared,
        tau_stop=tau_stop, correct=correct, false_id=false_id, **extra,
    )


def _nexthop_eval(p_hat: np.ndarray, true_next: int) -> tuple[float, bool, bool]:
    ll = -float(np.log(max(float(p_hat[true_next]), 1e-12)))
    order = np.argsort(p_hat)[::-1]
    return ll, bool(true_next == order[0]), bool(true_next in order[:3])


# --------------------------------------------------------------------- the bank
def run_bank(bank: HypothesisBank, episode: Episode, *, stop_early: bool = True,
             record_perslot: bool = False, logger: RunLogger | None = None,
             eval_nexthop: bool = True) -> EpisodeResult:
    bank.reset()
    true = episode.true_channels
    window = episode.window
    ll, t1, t3 = [], [], []
    per_slot: list[dict] = []
    declared_label = None
    tau = None

    for t, o in enumerate(episode.observations):
        bank.update(o)
        if eval_nexthop and t < window - 1:
            p_hat = bank.predict()
            a, b, c = _nexthop_eval(p_hat, int(true[t + 1]))
            ll.append(a); t1.append(b); t3.append(c)
        if record_perslot or logger is not None:
            out = bank.output()
            rec = {
                "slot": t + 1, "method": bank.label,
                "true_label": episode.true_label,
                "posterior": {lab: float(p) for lab, p in zip(out.labels, out.posterior)},
                "kappa": out.kappa, "stop_score": out.stop_score,
                "declared": out.declared, "declared_label": out.declared_label,
            }
            if record_perslot:
                per_slot.append(rec)
            if logger is not None:
                logger.log(rec)
        if bank.declared and declared_label is None:
            declared_label = bank.declared_identity()
            tau = bank.tau_stop
            if stop_early:
                break

    out = bank.output()
    declared = declared_label is not None
    if declared:
        decision_label = declared_label
        tau_stop = int(tau)
    else:
        top = out.top_index
        decision_label = UNKNOWN_LABEL if top == 0 else out.labels[top]
        tau_stop = window
    return decide_outcome(
        bank.label, episode, decision_label, declared, tau_stop,
        kappa_final=out.kappa,
        posterior_final={lab: float(p) for lab, p in zip(out.labels, out.posterior)},
        nexthop_logloss=float(np.mean(ll)) if ll else float("nan"),
        nexthop_top1=float(np.mean(t1)) if t1 else float("nan"),
        nexthop_top3=float(np.mean(t3)) if t3 else float("nan"),
        per_slot=per_slot,
    )


# ------------------------------------------------------------------- baselines
def run_sprt(sprt: PairwiseSPRT, episode: Episode, *, stop_early: bool = True) -> EpisodeResult:
    sprt.reset()
    for o in episode.observations:
        sprt.update(o)
        if sprt.declared and stop_early:
            break
    if sprt.declared:
        decision_label = sprt.declared_identity()
        tau_stop = int(sprt.tau_stop)
        declared = True
    else:
        decision_label = sprt.map_identity()   # forced MAP; no unknown class
        tau_stop = episode.window
        declared = False
    return decide_outcome(sprt.label, episode, decision_label, declared, tau_stop)


def run_fixed_length(fl: FixedLengthTest, episode: Episode) -> EpisodeResult:
    fl.reset()
    for o in episode.observations:
        fl.update(o)
    decision_label = fl.declared_identity() or fl.labels[fl.tracker.map_index()]
    return decide_outcome(fl.label, episode, decision_label, fl.declared,
                          int(fl.tau_stop or episode.window))


def run_learned(clf: LearnedClassifier, episode: Episode, *, threshold: float = 0.95,
                step: int = 1) -> EpisodeResult:
    label, tau, final_proba = clf.declare_over_stream(
        episode.observations, episode.channels, threshold=threshold, step=step)
    if label is not None:
        decision_label, tau_stop, declared = label, int(tau), True
    else:
        decision_label = clf.classes[int(np.argmax(final_proba))]
        tau_stop, declared = episode.window, False
    return decide_outcome(clf.label, episode, decision_label, declared, tau_stop,
                          posterior_final={c: float(p) for c, p in zip(clf.classes, final_proba)})
