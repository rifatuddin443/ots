"""Metrics: identification, calibration, and next-hop prediction (+ CIs and significance).

This module merges what used to be the fhew/metrics/ subpackage; the public
names are unchanged, so `from fhew.metrics import X` still works.
"""

from __future__ import annotations

# ---- merged from fhew/metrics/identification.py ----
from dataclasses import dataclass, field

import numpy as np

from fhew.sim import EpisodeResult


@dataclass
class IdentificationSummary:
    n: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    false_id_rate: float
    false_id_ci: tuple[float, float]
    declaration_rate: float
    mean_hops_to_identify: float          # over correct declarations only
    hops_ci: tuple[float, float]
    n_correct_declared: int
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n": self.n, "accuracy": self.accuracy, "accuracy_ci": self.accuracy_ci,
            "false_id_rate": self.false_id_rate, "false_id_ci": self.false_id_ci,
            "declaration_rate": self.declaration_rate,
            "mean_hops_to_identify": self.mean_hops_to_identify, "hops_ci": self.hops_ci,
            "n_correct_declared": self.n_correct_declared, **self.extra,
        }


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of ``values``."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boots = stat(values[idx], axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def summarize(results: list[EpisodeResult], *, seed: int = 0) -> IdentificationSummary:
    n = len(results)
    if n == 0:
        return IdentificationSummary(0, float("nan"), (float("nan"),) * 2, float("nan"),
                                     (float("nan"),) * 2, float("nan"), float("nan"),
                                     (float("nan"),) * 2, 0)
    correct = np.array([r.correct for r in results], dtype=np.float64)
    false_id = np.array([r.false_id for r in results], dtype=np.float64)
    declared = np.array([r.declared for r in results], dtype=np.float64)
    # hops-to-identify: over episodes that were correctly declared early
    hops = [r.tau_stop for r in results if r.correct and r.declared]
    hops_arr = np.asarray(hops, dtype=np.float64)
    return IdentificationSummary(
        n=n,
        accuracy=float(correct.mean()),
        accuracy_ci=bootstrap_ci(correct, seed=seed),
        false_id_rate=float(false_id.mean()),
        false_id_ci=bootstrap_ci(false_id, seed=seed + 1),
        declaration_rate=float(declared.mean()),
        mean_hops_to_identify=float(hops_arr.mean()) if hops_arr.size else float("nan"),
        hops_ci=bootstrap_ci(hops_arr, seed=seed + 2) if hops_arr.size else (float("nan"),) * 2,
        n_correct_declared=int(len(hops)),
    )


def summarize_by_class(results: list[EpisodeResult], key="true_label", *,
                       seed: int = 0) -> dict[str, IdentificationSummary]:
    groups: dict[str, list[EpisodeResult]] = {}
    for r in results:
        groups.setdefault(getattr(r, key), []).append(r)
    return {k: summarize(v, seed=seed) for k, v in sorted(groups.items())}


def paired_bootstrap_diff(results_a: list[EpisodeResult], results_b: list[EpisodeResult],
                          metric="correct", n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired bootstrap on a per-episode metric difference (A - B).

    Assumes the two methods were run on the *same* episodes in the same order
    (which the experiment harness guarantees). Returns the mean difference, its
    CI, and a two-sided bootstrap p-value for H0: diff == 0.
    """
    a = np.array([getattr(r, metric) for r in results_a], dtype=np.float64)
    b = np.array([getattr(r, metric) for r in results_b], dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired bootstrap needs equal-length aligned result lists")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean_diff": float(diff.mean()), "ci": (float(lo), float(hi)),
            "p_value": float(min(1.0, p)), "metric": metric, "n": int(diff.size)}


def mcnemar_test(results_a: list[EpisodeResult], results_b: list[EpisodeResult]) -> dict:
    """Exact-ish McNemar test on paired correctness (binomial on discordants)."""
    from scipy.stats import binomtest

    a = np.array([r.correct for r in results_a], dtype=bool)
    b = np.array([r.correct for r in results_b], dtype=bool)
    b01 = int(np.sum(a & ~b))   # A correct, B wrong
    b10 = int(np.sum(~a & b))   # B correct, A wrong
    disc = b01 + b10
    if disc == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    p = binomtest(b01, disc, 0.5).pvalue
    return {"b01": b01, "b10": b10, "p_value": float(p)}

# ---- merged from fhew/metrics/calibration.py ----
import numpy as np


def reliability_curve(confidences, correct, n_bins: int = 10):
    """Return ``(bin_centers, bin_accuracy, bin_confidence, bin_count)``."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    acc = np.full(n_bins, np.nan)
    avg_conf = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=np.int64)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        count[b] = int(mask.sum())
        if count[b] > 0:
            acc[b] = float(corr[mask].mean())
            avg_conf[b] = float(conf[mask].mean())
    return centers, acc, avg_conf, count


def ece(confidences, correct, n_bins: int = 10) -> float:
    """Expected calibration error: weighted mean ``|accuracy - confidence|``."""
    conf = np.asarray(confidences, dtype=np.float64)
    _, acc, avg_conf, count = reliability_curve(conf, correct, n_bins)
    total = count.sum()
    if total == 0:
        return float("nan")
    mask = count > 0
    return float(np.sum(count[mask] / total * np.abs(acc[mask] - avg_conf[mask])))

# ---- merged from fhew/metrics/prediction.py ----
import numpy as np

from fhew.sim import EpisodeResult


def nexthop_summary(results: list[EpisodeResult]) -> dict:
    """Aggregate next-hop log-loss and top-k hit rates over episodes."""
    ll = np.array([r.nexthop_logloss for r in results], dtype=np.float64)
    t1 = np.array([r.nexthop_top1 for r in results], dtype=np.float64)
    t3 = np.array([r.nexthop_top3 for r in results], dtype=np.float64)
    ll = ll[~np.isnan(ll)]
    t1 = t1[~np.isnan(t1)]
    t3 = t3[~np.isnan(t3)]
    return {
        "logloss": float(ll.mean()) if ll.size else float("nan"),
        "top1": float(t1.mean()) if t1.size else float("nan"),
        "top3": float(t3.mean()) if t3.size else float("nan"),
        "n": int(ll.size),
    }
