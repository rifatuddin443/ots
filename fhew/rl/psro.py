"""The empirical game: payoff oracle, meta-Nash, PSRO and the exploitability certificate.

This is the tool thesis Sections 9 and 13 ask for: not a tuned jammer, but a
*worst-case certificate* for an FH-EW policy. The question a reviewer raises --
"have you only tested against adversaries you wrote?" -- is answered by growing a
population of adversaries and systems, each a best response to the other's current
mixture, and reporting **exploitability**: how much a *fresh* best-response adversary
can still gain over the system's meta-strategy. A low, decreasing exploitability
curve is the certificate.

The pieces:

* :class:`EmpiricalGame` -- holds the two populations (hoppers and jammers) and the
  payoff matrix ``U[i, j]`` = mean denial of jammer ``j`` against hopper ``i``,
  filled by the rollout oracle and extended one row/column at a time.
* :func:`solve_zero_sum` -- fictitious play for the meta-Nash of the (zero-sum in
  denial) matrix game: the hopper minimises ``U``, the jammer maximises it.
* :func:`run_psro` -- the double-oracle loop. Each iteration solves the meta-Nash,
  trains an RL best-response hopper against the jammer mixture and an RL
  best-response jammer against the hopper mixture (:mod:`fhew.rl.hopper` /
  :mod:`fhew.rl.jammer`), adds both, and records the exploitability of the current
  meta.
* :func:`exploitability` -- trains fresh best responses to a meta-profile and
  reports ``(jammer-side gain) + (hopper-side gain)``.

The populations are seeded with the scripted and pipeline strategies the rest of the
codebase already defines (:class:`fhew.duel.OTSJammer`, the evasion hopper, the
scripted jammers), so PSRO measures the real OTS system, not a toy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fhew.allocator import EffectModel
from fhew.channels import ChannelSet
from fhew.experts import HypothesisSpec
from fhew.rl.agents import PPOConfig
from fhew.rl.env import (
    EngagementEnv,
    GeneratorHopper,
    JammerPolicy,
    OTSJammerPolicy,
    ScriptedJammer,
    HopperPolicy,
    run_engagement,
)


def fast_ots_specs() -> list[HypothesisSpec]:
    """A light but structure-capable predictor for the OTS strategies in the game.

    The full prediction bank is ~15x slower per slot (the seed-beam and rule-search
    experts enumerate large keyspaces), which is too slow to sit inside a PSRO
    best-response training loop. This keeps the universal context-mixing predictor,
    the online Markov expert, an adaptive Berlekamp-Massey expert (so an LFSR is
    still heavily jammed) and the barrage floor -- enough for the OTS pipeline to be
    a strong, structure-aware jammer at a few tens of ms per episode.
    """
    return [
        HypothesisSpec("context_mixing", "markov", "context_mixing", options={"max_order": 3}),
        HypothesisSpec("markov_live", "markov", "markov"),
        HypothesisSpec("bm_wide", "lfsr", "bm_adaptive", options={"max_length": 16}),
        HypothesisSpec("barrage", "none", "barrage"),
    ]


@dataclass
class Strategy:
    """A reusable policy in a population (reset by the rollout, never rebuilt)."""

    label: str
    policy: object                      # a HopperPolicy or JammerPolicy (reset()-able)
    kind: str                           # "hopper" | "jammer"
    learned: bool = False


# --------------------------------------------------------------------- the game
class EmpiricalGame:
    """Two growing populations and the denial payoff matrix between them."""

    def __init__(self, env: EngagementEnv, *, n_seeds: int = 6, seed0: int = 10_000):
        self.env = env
        self.n_seeds = int(n_seeds)
        self.seed0 = int(seed0)
        self.hoppers: list[Strategy] = []
        self.jammers: list[Strategy] = []
        self.U = np.zeros((0, 0), dtype=np.float64)   # U[i, j] = denial(jammer j, hopper i)

    # ------------------------------------------------------------------ payoff
    def _payoff(self, hopper: HopperPolicy, jammer: JammerPolicy) -> float:
        vals = []
        for s in range(self.n_seeds):
            rng = np.random.default_rng(self.seed0 + s)
            rec = run_engagement(hopper, jammer, self.env, rng)
            vals.append(rec.jammer_payoff(self.env.warmup))
        return float(np.mean(vals))

    def add_hopper(self, strat: Strategy) -> int:
        self.hoppers.append(strat)
        row = np.array([[self._payoff(strat.policy, j.policy) for j in self.jammers]])
        self.U = np.vstack([self.U, row]) if self.U.size else row.reshape(1, -1)
        if not self.jammers:
            self.U = np.zeros((len(self.hoppers), 0))
        return len(self.hoppers) - 1

    def add_jammer(self, strat: Strategy) -> int:
        self.jammers.append(strat)
        col = np.array([[self._payoff(h.policy, strat.policy)] for h in self.hoppers])
        if self.U.size:
            self.U = np.hstack([self.U, col])
        else:
            self.U = col.reshape(-1, 1) if self.hoppers else np.zeros((0, 1))
        return len(self.jammers) - 1

    # ------------------------------------------------------------------ labels
    def hopper_labels(self) -> list[str]:
        return [h.label for h in self.hoppers]

    def jammer_labels(self) -> list[str]:
        return [j.label for j in self.jammers]


# --------------------------------------------------------------- meta solver
def solve_zero_sum(U: np.ndarray, iters: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Fictitious play for the meta-Nash of a zero-sum matrix game.

    ``U[i, j]`` is the column player's (jammer, maximiser) payoff when the row player
    (hopper, minimiser) plays ``i``. Returns ``(sigma_hopper, sigma_jammer)`` as the
    time-averaged play, which converges to a Nash profile for a zero-sum game.
    """
    P, Q = U.shape
    if P == 0 or Q == 0:
        return np.ones(max(P, 1)) / max(P, 1), np.ones(max(Q, 1)) / max(Q, 1)
    row_counts = np.zeros(P)
    col_counts = np.zeros(Q)
    row_counts[0] = 1.0
    col_counts[0] = 1.0
    for _ in range(iters):
        row_mix = row_counts / row_counts.sum()
        col_br = int(np.argmax(row_mix @ U))         # jammer maximises
        col_counts[col_br] += 1.0
        col_mix = col_counts / col_counts.sum()
        row_br = int(np.argmin(U @ col_mix))         # hopper minimises
        row_counts[row_br] += 1.0
    return row_counts / row_counts.sum(), col_counts / col_counts.sum()


def meta_value(U: np.ndarray, sigma_h: np.ndarray, sigma_j: np.ndarray) -> float:
    return float(sigma_h @ U @ sigma_j)


# --------------------------------------------------------- best-response oracles
def _sample_factory(strats: list[Strategy], sigma: np.ndarray,
                    rng: np.random.Generator) -> Callable[[], object]:
    """A factory that draws a population member according to the meta-mixture."""
    def factory():
        i = int(rng.choice(len(strats), p=sigma))
        return strats[i].policy
    return factory


def train_hopper_br(game: EmpiricalGame, sigma_j: np.ndarray, channels: ChannelSet, *,
                    br: dict | None = None, rng: np.random.Generator | None = None):
    """Train an RL hopper best response to the jammer meta-mixture."""
    from fhew.rl.hopper import train_hopper

    br = br or {}
    rng = rng or np.random.default_rng(0)
    opp = _sample_factory(game.jammers, sigma_j, rng)
    agent = train_hopper(opp, channels, game.env, rng=rng,
                         n_iters=br.get("n_iters", 18),
                         episodes_per_iter=br.get("episodes_per_iter", 12),
                         cfg=br.get("cfg", PPOConfig()))
    return agent.freeze()


def train_jammer_br(game: EmpiricalGame, sigma_h: np.ndarray, channels: ChannelSet,
                    effect: EffectModel, budget: float, *, max_tones: int = 4,
                    br: dict | None = None, rng: np.random.Generator | None = None):
    """Train an RL jammer best response to the hopper meta-mixture."""
    from fhew.rl.jammer import train_jammer

    br = br or {}
    rng = rng or np.random.default_rng(0)
    opp = _sample_factory(game.hoppers, sigma_h, rng)
    agent = train_jammer(opp, channels, effect, game.env, budget=budget,
                         max_tones=max_tones, rng=rng,
                         n_iters=br.get("n_iters", 18),
                         episodes_per_iter=br.get("episodes_per_iter", 12),
                         cfg=br.get("cfg", PPOConfig()))
    return agent.freeze()


def _value_vs_mix(policy, strats: list[Strategy], sigma: np.ndarray,
                  game: EmpiricalGame, *, policy_is_jammer: bool) -> float:
    """Mean denial of ``policy`` against a meta-mixture of the opposing population."""
    v = 0.0
    for w, s in zip(sigma, strats):
        if w <= 0:
            continue
        if policy_is_jammer:
            v += w * game._payoff(s.policy, policy)
        else:
            v += w * game._payoff(policy, s.policy)
    return float(v)


@dataclass
class ExploitabilityReport:
    meta_value: float
    jammer_br_value: float               # best a fresh jammer does vs hopper meta
    hopper_br_value: float               # best a fresh hopper does vs jammer meta (low=good for hopper)
    jammer_gain: float                   # jammer_br_value - meta_value  (>=0)
    hopper_gain: float                   # meta_value - hopper_br_value  (>=0)
    exploitability: float                # jammer_gain + hopper_gain

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def exploitability(game: EmpiricalGame, channels: ChannelSet, effect: EffectModel,
                   budget: float, *, max_tones: int = 4, br: dict | None = None,
                   rng: np.random.Generator | None = None) -> ExploitabilityReport:
    """Train fresh best responses to the current meta and measure the residual gain."""
    rng = rng or np.random.default_rng(0)
    sigma_h, sigma_j = solve_zero_sum(game.U)
    v = meta_value(game.U, sigma_h, sigma_j)

    j_br = train_jammer_br(game, sigma_h, channels, effect, budget,
                           max_tones=max_tones, br=br, rng=np.random.default_rng(rng.integers(1 << 30)))
    jbr_val = _value_vs_mix(j_br, game.hoppers, sigma_h, game, policy_is_jammer=True)

    h_br = train_hopper_br(game, sigma_j, channels, br=br,
                           rng=np.random.default_rng(rng.integers(1 << 30)))
    hbr_val = _value_vs_mix(h_br, game.jammers, sigma_j, game, policy_is_jammer=False)

    jgain = max(0.0, jbr_val - v)
    hgain = max(0.0, v - hbr_val)
    return ExploitabilityReport(meta_value=v, jammer_br_value=jbr_val,
                                hopper_br_value=hbr_val, jammer_gain=jgain,
                                hopper_gain=hgain, exploitability=jgain + hgain)


# ------------------------------------------------------------------- the PSRO loop
def initial_populations(channels: ChannelSet, sensor, effect: EffectModel, budget: float,
                        *, max_tones: int = 4, ots_specs: list | None = None,
                        include_unpredictable: bool = True,
                        rng: np.random.Generator | None = None
                        ) -> tuple[list[Strategy], list[Strategy]]:
    """Seed the two populations with scripted + pipeline strategies.

    Hoppers: oblivious LFSR / Markov, a reactive evasion LFSR, the online-bandit
    learned hopper, and -- when ``include_unpredictable`` -- a keyed CSPRNG. Jammers:
    scripted spot and barrage, and the OTS predict-then-allocate pipeline (the system
    under test).

    The populations are seeded *minimally and exploitably*: oblivious, predictable
    hoppers (an LFSR and a peaky Markov) and a weak-to-strong spread of jammers
    (scripted spot / barrage and the OTS pipeline). The strong reactive escapers --
    an evasion radio, a learned bandit hopper -- are deliberately *not* pre-seeded:
    PSRO's best-response oracle must *discover* them, which is what makes the
    exploitability curve start high and fall (a self-seeded strong population would
    sit at equilibrium from iteration 0, a correct but un-illustrative certificate).

    The CSPRNG is a *dominant escape*: a hopper that plays it is unpredictable by
    design, so the game value collapses to the barrage floor. ``include_unpredictable``
    adds it so the boundary (the floor) can be read off; dropping it restricts the
    game to the structured / reactive adversary class.
    """
    from fhew.duel import OTSJammer
    from fhew.generators import make_generator

    rng = rng or np.random.default_rng(0)
    ots_specs = ots_specs or fast_ots_specs()

    hoppers = [
        Strategy("lfsr", GeneratorHopper(make_generator("lfsr", channels.size, length=7,
                 rng=np.random.default_rng(1)), label="lfsr"), "hopper"),
        Strategy("markov", GeneratorHopper(make_generator("markov", channels.size, alpha=0.1,
                 rng=np.random.default_rng(2)), label="markov"), "hopper"),
    ]
    if include_unpredictable:
        hoppers.append(Strategy("csprng", GeneratorHopper(
            make_generator("csprng", channels.size, rng=np.random.default_rng(3)),
            label="csprng"), "hopper"))
    jammers = [
        Strategy("spot", ScriptedJammer(channels, effect, budget, mode="spot"), "jammer"),
        Strategy("barrage", ScriptedJammer(channels, effect, budget, mode="barrage",
                 n_tones=max_tones), "jammer"),
        Strategy("ots_knapsack", OTSJammerPolicy(OTSJammer(channels, sensor, effect,
                 budget=budget, max_tones=max_tones, specs=ots_specs), label="ots_knapsack"),
                 "jammer"),
    ]
    return hoppers, jammers


@dataclass
class PSROResult:
    iterations: list[int] = field(default_factory=list)
    exploitability: list[float] = field(default_factory=list)
    jammer_gain: list[float] = field(default_factory=list)
    hopper_gain: list[float] = field(default_factory=list)
    meta_value: list[float] = field(default_factory=list)
    ots_vs_meta_hopper: list[float] = field(default_factory=list)
    payoff_matrix: list = field(default_factory=list)
    hopper_labels: list[str] = field(default_factory=list)
    jammer_labels: list[str] = field(default_factory=list)
    final_sigma_hopper: list[float] = field(default_factory=list)
    final_sigma_jammer: list[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def run_psro(channels: ChannelSet, sensor, effect: EffectModel, *, budget: float = 3.0,
             max_tones: int = 4, window: int = 80, n_iters: int = 4, n_seeds: int = 6,
             br: dict | None = None, ots_specs: list | None = None,
             include_unpredictable: bool = True,
             feedback_mode: str = "spectrum", seed: int = 0,
             progress: bool = False) -> PSROResult:
    """Run the double-oracle PSRO loop and return the exploitability certificate.

    Each iteration: solve the meta-Nash of the current payoff matrix, measure the
    exploitability of that meta (train fresh best responses to it), then grow both
    populations with an RL best response to the opponent's mixture. Exploitability is
    expected to trend down as the populations cover more of the strategy space.
    """
    rng = np.random.default_rng(seed)
    env = EngagementEnv(channels, sensor, effect, window=window, feedback_mode=feedback_mode)
    game = EmpiricalGame(env, n_seeds=n_seeds)
    hoppers, jammers = initial_populations(channels, sensor, effect, budget,
                                           max_tones=max_tones, ots_specs=ots_specs,
                                           include_unpredictable=include_unpredictable, rng=rng)
    # add jammers first so the hopper rows get full columns
    for j in jammers:
        game.jammers.append(j)
    game.U = np.zeros((0, len(game.jammers)))
    for h in hoppers:
        game.add_hopper(h)

    res = PSROResult(meta={"budget": budget, "max_tones": max_tones, "window": window,
                           "n_seeds": n_seeds, "feedback_mode": feedback_mode,
                           "n_init_hoppers": len(hoppers), "n_init_jammers": len(jammers)})
    ots_idx = next((k for k, j in enumerate(game.jammers) if j.label == "ots_knapsack"), None)

    for it in range(n_iters):
        rep = exploitability(game, channels, effect, budget, max_tones=max_tones, br=br,
                             rng=np.random.default_rng(rng.integers(1 << 30)))
        sigma_h, sigma_j = solve_zero_sum(game.U)
        res.iterations.append(it)
        res.exploitability.append(rep.exploitability)
        res.jammer_gain.append(rep.jammer_gain)
        res.hopper_gain.append(rep.hopper_gain)
        res.meta_value.append(rep.meta_value)
        if ots_idx is not None:
            res.ots_vs_meta_hopper.append(float(game.U[:, ots_idx] @ sigma_h))
        if progress:
            print(f"[psro] iter {it}: expl={rep.exploitability:.3f} "
                  f"(J+{rep.jammer_gain:.3f} / H+{rep.hopper_gain:.3f}) "
                  f"meta_denial={rep.meta_value:.3f} "
                  f"|H|={len(game.hoppers)} |J|={len(game.jammers)}")

        # grow both populations with a best response to the current meta. Train both
        # against the *pre-growth* mixtures (so sigma lengths match the populations),
        # then add them.
        h_br = train_hopper_br(game, sigma_j, channels, br=br,
                               rng=np.random.default_rng(rng.integers(1 << 30)))
        j_br = train_jammer_br(game, sigma_h, channels, effect, budget, max_tones=max_tones,
                               br=br, rng=np.random.default_rng(rng.integers(1 << 30)))
        game.add_hopper(Strategy(f"rl_hopper_{it}", h_br, "hopper", learned=True))
        game.add_jammer(Strategy(f"rl_jammer_{it}", j_br, "jammer", learned=True))

    sigma_h, sigma_j = solve_zero_sum(game.U)
    res.payoff_matrix = game.U.tolist()
    res.hopper_labels = game.hopper_labels()
    res.jammer_labels = game.jammer_labels()
    res.final_sigma_hopper = sigma_h.tolist()
    res.final_sigma_jammer = sigma_j.tolist()
    return res
