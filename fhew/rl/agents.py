"""Dependency-free (numpy) reinforcement-learning cores.

No torch, no autograd: the policies here are linear-softmax over hand-built
features, trained with an analytically-differentiated **PPO-clip** surrogate and a
linear value baseline. That is enough for the short-horizon, small-channel
engagements this package studies, it runs in milliseconds, and -- crucially -- it
is exactly reproducible from a seed, so the exploitability certificate
(:mod:`fhew.rl.psro`) is a figure a reviewer can regenerate rather than a training
run that drifts.

The pieces:

* :class:`LinearSoftmaxPolicy` -- ``p(a|s) = softmax(W phi(s))``. Exposes the
  analytic score ``d log p / d logits = onehot(a) - p`` used by both the PPO
  surrogate and the entropy bonus.
* :class:`LinearBaseline` -- a ridge-fit linear value ``V(s) = v . phi(s)`` for
  variance reduction (advantages ``A = R - V``).
* :func:`ppo_update` -- one PPO-clip optimisation phase (several epochs of
  minibatches) over a batch of episodes, with an entropy bonus and optional CVaR
  tail weighting.
* :func:`cvar_weights` -- per-episode weights that concentrate the update on the
  worst-``alpha`` tail of returns, turning plain PPO into a **CVaR-PPO** risk
  objective (the Section 12 robust-jammer baseline).

A policy-gradient agent (hopper or jammer) owns feature construction and a
trajectory buffer and calls :func:`ppo_update`; see :mod:`fhew.rl.hopper` and
:mod:`fhew.rl.jammer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class LinearSoftmaxPolicy:
    """A linear-softmax categorical policy ``p(a|s) = softmax(W phi(s))``.

    Parameters
    ----------
    n_actions:
        Size of the (discrete) action space.
    n_features:
        Dimension of the feature vector ``phi(s)``.
    rng:
        Seeded generator for action sampling and (tiny) weight init.
    init_scale:
        Std of the Gaussian weight initialisation. ``0`` starts from the uniform
        policy, which is the right prior for a hopper (uniform hopping is the
        unpredictable floor) and keeps early exploration broad.
    """

    def __init__(self, n_actions: int, n_features: int, *,
                 rng: np.random.Generator | None = None, init_scale: float = 0.0):
        self.A = int(n_actions)
        self.d = int(n_features)
        self.rng = rng or np.random.default_rng()
        self.W = (init_scale * self.rng.standard_normal((self.A, self.d))
                  if init_scale > 0 else np.zeros((self.A, self.d)))

    # ------------------------------------------------------------------ forward
    def logits(self, phi: np.ndarray) -> np.ndarray:
        return self.W @ phi

    def probs(self, phi: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
        z = self.W @ phi
        if mask is not None:
            z = np.where(mask, z, -1e30)
        return _softmax(z)

    def sample(self, phi: np.ndarray, *, mask: np.ndarray | None = None
               ) -> tuple[int, float]:
        p = self.probs(phi, mask=mask)
        a = int(self.rng.choice(self.A, p=p))
        return a, float(np.log(max(p[a], 1e-300)))

    def logp(self, phi: np.ndarray, action: int) -> float:
        p = self.probs(phi)
        return float(np.log(max(p[int(action)], 1e-300)))

    def copy_weights(self) -> np.ndarray:
        return self.W.copy()

    def load_weights(self, W: np.ndarray) -> None:
        self.W = np.asarray(W, dtype=np.float64).reshape(self.A, self.d).copy()


class LinearBaseline:
    """Ridge-regression linear value baseline ``V(s) = v . phi(s)``."""

    def __init__(self, n_features: int, *, ridge: float = 1.0):
        self.d = int(n_features)
        self.v = np.zeros(self.d)
        self.ridge = float(ridge)

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        return Phi @ self.v

    def fit(self, Phi: np.ndarray, returns: np.ndarray) -> None:
        if Phi.shape[0] == 0:
            return
        A = Phi.T @ Phi + self.ridge * np.eye(self.d)
        b = Phi.T @ returns
        self.v = np.linalg.solve(A, b)


@dataclass
class PPOConfig:
    """PPO-clip hyper-parameters (defaults tuned for these short engagements)."""

    lr: float = 0.08
    clip: float = 0.2
    epochs: int = 6
    minibatch: int = 256
    gamma: float = 0.9              # reward-to-go discount (mild credit assignment)
    entropy_coef: float = 0.01
    max_grad: float = 5.0           # L2 clip on the weight gradient
    cvar_alpha: float = 0.0         # 0 disables CVaR; else the worst-tail fraction


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """Reward-to-go ``R_t = sum_{k>=t} gamma^{k-t} r_k`` for one episode."""
    out = np.empty(len(rewards), dtype=np.float64)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        out[i] = acc
    return out


def cvar_weights(episode_returns: np.ndarray, alpha: float) -> np.ndarray:
    """Per-episode weights concentrating on the worst-``alpha`` tail of returns.

    Returns weights that are uniform over the lowest-``alpha`` fraction of episodes
    (by total return) and zero elsewhere, normalised to mean 1 over the whole
    batch. Multiplying advantages by these turns the PPO objective into a CVaR
    objective: the policy is improved for the engagements where it did *worst*,
    which is the risk-averse (robust) criterion the Section 12 CVaR-PPO jammer
    wants -- optimise the bad tail, not the average.
    """
    n = len(episode_returns)
    if alpha <= 0.0 or n == 0:
        return np.ones(n)
    k = max(1, int(np.ceil(alpha * n)))
    order = np.argsort(episode_returns)          # ascending: worst first
    w = np.zeros(n)
    w[order[:k]] = 1.0
    w *= n / w.sum()                             # keep mean weight 1
    return w


def ppo_update(policy: LinearSoftmaxPolicy, baseline: LinearBaseline,
               episodes: list[dict], cfg: PPOConfig) -> dict:
    """Run one PPO-clip optimisation phase over a batch of episodes.

    Each episode dict carries ``phi`` (T, d), ``action`` (T,), ``reward`` (T,).
    Returns a small diagnostics dict (mean return, policy entropy, approx KL).
    """
    if not episodes:
        return {"mean_return": float("nan"), "entropy": float("nan"), "n": 0}

    # Report the mean return over the WHOLE batch (before any CVaR filtering).
    all_returns = np.array([float(np.sum(ep["reward"])) for ep in episodes])

    # CVaR: restrict the update to the worst-alpha tail of episodes, then optimise
    # that sub-batch normally. Filtering *before* fitting the baseline is what makes
    # this work: the baseline and advantages are then computed WITHIN the tail, so a
    # good action inside a bad engagement still gets a positive advantage and is
    # reinforced -- the risk-averse (robust) objective. Weighting a whole-batch
    # advantage instead would leave every tail-episode advantage below the global
    # baseline and merely diffuse the policy on the hard episodes.
    train_eps = episodes
    if cfg.cvar_alpha > 0.0 and len(episodes) > 1:
        k = max(1, int(np.ceil(cfg.cvar_alpha * len(episodes))))
        tail = np.argsort(all_returns)[:k]
        train_eps = [episodes[i] for i in tail]

    # Flatten the (possibly tail-filtered) episodes into one batch.
    Phi_list, act_list, ret_list = [], [], []
    for ep in train_eps:
        R = discounted_returns(list(ep["reward"]), cfg.gamma)
        Phi_list.append(np.asarray(ep["phi"], dtype=np.float64))
        act_list.append(np.asarray(ep["action"], dtype=np.int64))
        ret_list.append(R)
    Phi = np.concatenate(Phi_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    returns = np.concatenate(ret_list, axis=0)

    # Baseline + advantages (standardised for a stable step size).
    baseline.fit(Phi, returns)
    adv = returns - baseline.predict(Phi)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Old log-probs under the pre-update policy (PPO ratio denominator).
    old_logp = _batch_logp(policy, Phi, actions)

    N = Phi.shape[0]
    idx_all = np.arange(N)
    ent_last = 0.0
    for _ in range(cfg.epochs):
        policy.rng.shuffle(idx_all)
        for start in range(0, N, cfg.minibatch):
            mb = idx_all[start:start + cfg.minibatch]
            grad, ent = _ppo_grad(policy, Phi[mb], actions[mb], adv[mb],
                                   old_logp[mb], cfg)
            gnorm = np.linalg.norm(grad)
            if gnorm > cfg.max_grad:
                grad = grad * (cfg.max_grad / gnorm)
            policy.W = policy.W + cfg.lr * grad
            ent_last = ent
    new_logp = _batch_logp(policy, Phi, actions)
    approx_kl = float(np.mean(old_logp - new_logp))
    return {"mean_return": float(all_returns.mean()), "entropy": float(ent_last),
            "approx_kl": approx_kl, "n": int(N)}


def _batch_logp(policy: LinearSoftmaxPolicy, Phi: np.ndarray, actions: np.ndarray
                ) -> np.ndarray:
    Z = Phi @ policy.W.T                           # (N, A)
    Z = Z - Z.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(Z).sum(axis=1))
    return Z[np.arange(len(actions)), actions] - logZ


def _ppo_grad(policy: LinearSoftmaxPolicy, Phi: np.ndarray, actions: np.ndarray,
              adv: np.ndarray, old_logp: np.ndarray, cfg: PPOConfig
              ) -> tuple[np.ndarray, float]:
    """Gradient of the PPO-clip surrogate (+ entropy bonus) w.r.t. W.

    The clip detaches the gradient wherever the clipped term is the binding one, so
    only samples inside the trust region (or pushing back toward it) contribute --
    the standard PPO behaviour, here written out by hand for the linear softmax.
    """
    N = Phi.shape[0]
    Z = Phi @ policy.W.T
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    P = P / P.sum(axis=1, keepdims=True)           # (N, A)
    logp = np.log(np.clip(P[np.arange(N), actions], 1e-300, 1.0))
    ratio = np.exp(logp - old_logp)

    # PPO-clip: the surrogate is min(ratio*adv, clip(ratio)*adv). The unclipped
    # branch carries gradient iff it is the binding (smaller) term.
    clipped = np.clip(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
    unclipped_obj = ratio * adv
    clipped_obj = clipped * adv
    active = (unclipped_obj <= clipped_obj).astype(np.float64)  # 1 where unclipped binds

    # d(ratio*adv)/d logits_k = adv * ratio * (onehot(a) - p_k)
    coeff = (active * adv * ratio)                 # (N,)
    onehot = np.zeros_like(P)
    onehot[np.arange(N), actions] = 1.0
    dlogits = coeff[:, None] * (onehot - P)        # (N, A)

    # Entropy bonus gradient: dH/d logits_k = -p_k (log p_k + H).
    H = -(P * np.log(np.clip(P, 1e-300, 1.0))).sum(axis=1, keepdims=True)
    dH = -P * (np.log(np.clip(P, 1e-300, 1.0)) + H)
    dlogits = dlogits + cfg.entropy_coef * dH

    grad = dlogits.T @ Phi / N                     # (A, d)
    return grad, float(H.mean())
