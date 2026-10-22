"""A feature-based learned classifier baseline (numpy, offline-trained).

Multinomial logistic regression over the :mod:`fhew.baselines.features` vector,
trained offline on labelled simulated episodes (the training data from
:mod:`fhew.datagen`). It is a legitimate "LSTM classifier" stand-in that runs
with no deep-learning dependency; a torch-LSTM drop-in lives in
:mod:`fhew.baselines.lstm` and is used automatically when torch is installed.

The classifier has an explicit ``unknown`` class, trained on crypto and
out-of-bank episodes, so the comparison against the bank's H_0 is apples to
apples. Like every baseline it only ever sees the sensor output.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

from fhew.baselines.features import episode_features, feature_dim
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
