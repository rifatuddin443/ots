"""Fair baselines: pairwise SPRT, fixed-length LRT, learned classifier (+ optional LSTM).

This module merges what used to be the fhew/baselines/ subpackage; the public
names are unchanged, so `from fhew.baselines import X` still works.
"""

from __future__ import annotations

# ---- merged from fhew/baselines/common.py ----
import numpy as np

from fhew.experts import Hypothesis
from fhew.likelihood import EmissionModel
from fhew.sensor import Observation


class LikelihoodTracker:
    """Drives a list of hypotheses and accumulates reduced log-likelihoods."""

    def __init__(self, hypotheses: list[Hypothesis], emission: EmissionModel,
                 n_channels: int):
        self.hyps = list(hypotheses)
        self.labels = [h.label for h in self.hyps]
        self.emission = emission
        self.n = len(self.hyps)
        self.n_channels = n_channels
        self.reset()

    def reset(self) -> None:
        for h in self.hyps:
            h.reset()
        self.cum_loglik = np.zeros(self.n, dtype=np.float64)
        self.t = 0
        self._next_hops = np.tile(np.full(self.n_channels, 1.0 / self.n_channels),
                                  (self.n, 1))

    def update(self, obs: Observation) -> None:
        log_e = self.emission.log_emission(obs)
        clean = obs.detections[0].channel if len(obs.detections) == 1 else None
        for k, h in enumerate(self.hyps):
            res = h.step(log_e)
            self.cum_loglik[k] += res.loglik
            self._next_hops[k] = res.predict_next
            observe_clean = getattr(h, "observe_clean", None)
            if observe_clean is not None:
                observe_clean(clean)
        self.t += 1

    def map_index(self) -> int:
        return int(np.argmax(self.cum_loglik))

    def next_hop(self, index: int) -> np.ndarray:
        return self._next_hops[index]

# ---- merged from fhew/baselines/features.py ----
import math

import numpy as np

from fhew.channels import ChannelSet
from fhew.recovery import channels_to_bits, linear_complexity_profile
from fhew.sensor import Observation


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def episode_features(obs_list: list[Observation], channels: ChannelSet) -> np.ndarray:
    """Return a fixed-length feature vector for an observation prefix."""
    M = channels.size
    n_slots = max(1, len(obs_list))

    hist = np.zeros(M, dtype=np.float64)
    n_det_total = 0
    n_empty = 0
    clean_channels: list[int] = []
    for o in obs_list:
        nd = len(o.detections)
        n_det_total += nd
        if nd == 0:
            n_empty += 1
        for d in o.detections:
            hist[d.channel] += 1.0
        if nd == 1:
            clean_channels.append(o.detections[0].channel)

    hist_norm = hist / hist.sum() if hist.sum() > 0 else np.full(M, 1.0 / M)
    occ_entropy = _entropy(hist_norm) / math.log(M)

    frac_empty = n_empty / n_slots
    mean_det = n_det_total / n_slots

    clean = np.asarray(clean_channels, dtype=np.int64)
    n_clean = clean.size
    if n_clean >= 2:
        repeat_rate = float(np.mean(clean[1:] == clean[:-1]))
        bigram_entropy = _bigram_entropy(clean, M) / math.log(M)
    else:
        repeat_rate = 0.0
        bigram_entropy = 1.0

    lc_ratio = 0.0
    if _is_power_of_two(M) and n_clean >= 8:
        b = int(round(math.log2(M)))
        bits = channels_to_bits(clean, b)
        prof = linear_complexity_profile(bits)
        lc = prof[-1] if prof else 0
        lc_ratio = min(1.0, lc / max(1, bits.size / 2))

    frac_clean = n_clean / n_slots
    log_slots = math.log1p(n_slots) / math.log1p(500)

    scalars = np.array([
        occ_entropy, frac_empty, mean_det, repeat_rate, bigram_entropy,
        lc_ratio, frac_clean, log_slots,
    ], dtype=np.float64)
    return np.concatenate([hist_norm, scalars])


def feature_dim(channels: ChannelSet) -> int:
    return channels.size + 8


def _entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def _bigram_entropy(clean: np.ndarray, M: int) -> float:
    counts = np.ones((M, M), dtype=np.float64)  # Laplace
    for a, b in zip(clean[:-1], clean[1:]):
        counts[a, b] += 1.0
    T = counts / counts.sum(axis=1, keepdims=True)
    # weight rows by how often each state is visited
    visit = np.bincount(clean[:-1], minlength=M).astype(np.float64)
    visit = visit / visit.sum() if visit.sum() > 0 else np.full(M, 1.0 / M)
    row_ent = np.array([_entropy(T[i]) for i in range(M)])
    return float((visit * row_ent).sum())

# ---- merged from fhew/baselines/fixed_length.py ----
from fhew.experts import Hypothesis
from fhew.likelihood import EmissionModel
from fhew.sensor import Observation


class FixedLengthTest:
    label = "fixed_length"

    def __init__(self, hypotheses: list[Hypothesis], emission: EmissionModel,
                 n_channels: int, n_fixed: int = 30):
        self.tracker = LikelihoodTracker(hypotheses, emission, n_channels)
        self.labels = self.tracker.labels
        self.n_fixed = int(n_fixed)
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self.declared = False
        self.declared_index: int | None = None
        self.tau_stop: int | None = None

    def update(self, obs: Observation) -> None:
        if self.tracker.t >= self.n_fixed:
            return
        self.tracker.update(obs)
        if self.tracker.t >= self.n_fixed and not self.declared:
            self.declared = True
            self.declared_index = self.tracker.map_index()
            self.tau_stop = self.n_fixed

    def declared_identity(self) -> str | None:
        if self.declared_index is None:
            return None
        return self.labels[self.declared_index]

# ---- merged from fhew/baselines/learned.py ----
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

from fhew.channels import ChannelSet
from fhew.sensor import Observation


class LearnedClassifier:
    """Feature-based classifier: softmax regression or a 1-hidden-layer MLP.

    ``hidden=0`` is L2-regularised multinomial logistic regression; ``hidden>0``
    adds one tanh layer, which is enough to make it a credible stand-in for the
    LSTM baseline without a deep-learning dependency. Both are trained with
    L-BFGS on scaled features.
    """

    label = "learned"

    def __init__(self, classes: list[str], l2: float = 1e-2, hidden: int = 64,
                 seed: int = 0):
        self.classes = list(classes)
        self.C = len(self.classes)
        self.l2 = float(l2)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.params_: np.ndarray | None = None
        self.D_: int | None = None

    # ------------------------------------------------------------------- train
    def _shapes(self, D: int):
        H, C = self.hidden, self.C
        if H == 0:
            return [((D + 1, C), (D + 1) * C)]
        return [((D + 1, H), (D + 1) * H), ((H + 1, C), (H + 1) * C)]

    def _unpack(self, w: np.ndarray, D: int):
        out, off = [], 0
        for shape, size in self._shapes(D):
            out.append(w[off:off + size].reshape(shape))
            off += size
        return out

    def _forward(self, w: np.ndarray, Xs: np.ndarray):
        D = Xs.shape[1]
        mats = self._unpack(w, D)
        Xb = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
        if self.hidden == 0:
            logits = Xb @ mats[0]
            return logits, (Xb,)
        W1, W2 = mats
        h = np.tanh(Xb @ W1)
        hb = np.hstack([np.ones((h.shape[0], 1)), h])
        logits = hb @ W2
        return logits, (Xb, h, hb, W1, W2)

    def fit(self, X: np.ndarray, y: np.ndarray, maxiter: int = 400) -> "LearnedClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        Xs = (X - self.mean_) / self.std_
        self.D_ = Xs.shape[1]
        N = Xs.shape[0]
        Y = np.eye(self.C)[y]
        rng = np.random.default_rng(self.seed)
        total = sum(size for _, size in self._shapes(self.D_))
        w0 = rng.normal(scale=0.1, size=total) if self.hidden else np.zeros(total)

        def loss_grad(w):
            logits, cache = self._forward(w, Xs)
            logp = log_softmax(logits, axis=1)
            nll = -(Y * logp).sum() / N
            P = softmax(logits, axis=1)
            dlogits = (P - Y) / N
            if self.hidden == 0:
                (Xb,) = cache
                W = self._unpack(w, self.D_)[0]
                g = Xb.T @ dlogits
                g[1:] += self.l2 * W[1:]
                reg = 0.5 * self.l2 * (W[1:] ** 2).sum()
                return nll + reg, g.ravel()
            Xb, h, hb, W1, W2 = cache
            gW2 = hb.T @ dlogits
            dh = (dlogits @ W2[1:].T) * (1 - h ** 2)
            gW1 = Xb.T @ dh
            gW1[1:] += self.l2 * W1[1:]
            gW2[1:] += self.l2 * W2[1:]
            reg = 0.5 * self.l2 * ((W1[1:] ** 2).sum() + (W2[1:] ** 2).sum())
            return nll + reg, np.concatenate([gW1.ravel(), gW2.ravel()])

        res = minimize(loss_grad, w0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        self.params_ = res.x
        return self

    # ----------------------------------------------------------------- predict
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("classifier is not fitted")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Xs = (X - self.mean_) / self.std_
        logits, _ = self._forward(self.params_, Xs)
        return softmax(logits, axis=1)

    def predict_label(self, X: np.ndarray) -> list[str]:
        idx = np.argmax(self.predict_proba(X), axis=1)
        return [self.classes[i] for i in idx]

    # ------------------------------------------------------- streaming helpers
    def declare_over_stream(self, obs_list: list[Observation], channels: ChannelSet,
                            threshold: float = 0.95, min_slots: int = 2,
                            step: int = 1) -> tuple[str | None, int | None, np.ndarray]:
        """Declare the first slot where top-class probability crosses threshold.

        Returns ``(label, tau, final_proba)``. Mirrors the bank's anytime use so
        the hops-to-identify comparison is fair.
        """
        proba = None
        for t in range(1, len(obs_list) + 1):
            if t < min_slots or (t % step != 0 and t != len(obs_list)):
                continue
            feats = episode_features(obs_list[:t], channels)
            proba = self.predict_proba(feats)[0]
            top = int(np.argmax(proba))
            if proba[top] >= threshold:
                return self.classes[top], t, proba
        return None, None, (proba if proba is not None else np.full(self.C, 1.0 / self.C))

    # --------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "classes": self.classes, "l2": self.l2, "hidden": self.hidden,
            "D": self.D_,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
            "params": None if self.params_ is None else self.params_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedClassifier":
        obj = cls(d["classes"], l2=d.get("l2", 1e-2), hidden=d.get("hidden", 64))
        obj.D_ = d.get("D")
        obj.mean_ = None if d["mean"] is None else np.asarray(d["mean"])
        obj.std_ = None if d["std"] is None else np.asarray(d["std"])
        obj.params_ = None if d["params"] is None else np.asarray(d["params"])
        return obj

# ---- merged from fhew/baselines/pairwise_sprt.py ----
import numpy as np

from fhew.experts import Hypothesis
from fhew.likelihood import EmissionModel
from fhew.sensor import Observation


class PairwiseSPRT:
    label = "pairwise_sprt"

    def __init__(self, hypotheses: list[Hypothesis], emission: EmissionModel,
                 n_channels: int, alpha: float = 0.05, beta: float = 0.05):
        self.tracker = LikelihoodTracker(hypotheses, emission, n_channels)
        self.labels = self.tracker.labels
        # Wald's A bound on the log-likelihood-ratio for a target error.
        self.A = float(np.log((1.0 - beta) / alpha))
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self.declared = False
        self.declared_index: int | None = None
        self.tau_stop: int | None = None

    def update(self, obs: Observation) -> None:
        self.tracker.update(obs)
        if self.declared:
            return
        cl = self.tracker.cum_loglik
        k = int(np.argmax(cl))
        # leader's smallest margin over any competitor
        margins = cl[k] - cl
        margins[k] = np.inf
        if float(margins.min()) >= self.A:
            self.declared = True
            self.declared_index = k
            self.tau_stop = self.tracker.t

    def declared_identity(self) -> str | None:
        if self.declared_index is None:
            return None
        return self.labels[self.declared_index]

    def map_identity(self) -> str:
        return self.labels[self.tracker.map_index()]

# ---- merged from fhew/baselines/lstm.py ----
import numpy as np

try:  # pragma: no cover - exercised only when torch is present
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def torch_available() -> bool:
    return _HAS_TORCH


if _HAS_TORCH:  # pragma: no cover - only when torch installed

    class _LSTMNet(nn.Module):
        def __init__(self, n_channels: int, n_classes: int, hidden: int = 64):
            super().__init__()
            self.embed = nn.Embedding(n_channels + 1, 16)  # +1 for "empty slot"
            self.lstm = nn.LSTM(16, hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x, lengths=None):
            e = self.embed(x)
            out, _ = self.lstm(e)
            return self.head(out[:, -1, :])

    class LSTMClassifier:
        label = "lstm"

        def __init__(self, n_channels: int, classes: list[str], hidden: int = 64,
                     lr: float = 1e-2, device: str = "cpu"):
            self.n_channels = n_channels
            self.classes = list(classes)
            self.device = device
            self.net = _LSTMNet(n_channels, len(classes), hidden).to(device)
            self.lr = lr

        def _encode(self, obs_list):
            # dominant detected channel per slot, or n_channels for an empty slot
            seq = []
            for o in obs_list:
                if len(o.detections) == 0:
                    seq.append(self.n_channels)
                else:
                    seq.append(o.detections[0].channel)
            return torch.tensor(seq, dtype=torch.long)

        def fit(self, episodes, labels, epochs: int = 15, batch: int = 64):
            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            lossf = nn.CrossEntropyLoss()
            y = torch.tensor(labels, dtype=torch.long)
            seqs = [self._encode(e) for e in episodes]
            L = max(len(s) for s in seqs)
            X = torch.full((len(seqs), L), self.n_channels, dtype=torch.long)
            for i, s in enumerate(seqs):
                X[i, : len(s)] = s
            n = len(seqs)
            for _ in range(epochs):
                perm = torch.randperm(n)
                for i in range(0, n, batch):
                    idx = perm[i : i + batch]
                    opt.zero_grad()
                    out = self.net(X[idx])
                    loss = lossf(out, y[idx])
                    loss.backward()
                    opt.step()
            return self

        def predict_proba_stream(self, obs_list):
            self.net.eval()
            with torch.no_grad():
                x = self._encode(obs_list).unsqueeze(0)
                logits = self.net(x)
                return torch.softmax(logits, dim=1).cpu().numpy()[0]
