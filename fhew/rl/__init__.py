"""Reinforcement-learning layer: learned adversaries, a learned jammer, and the
empirical-game / PSRO exploitability certificate (thesis Sections 9, 11, 12).

Everything in :mod:`fhew` outside this package treats the engagement as either a
fixed hop tape (the identification / prediction experiments) or a *scripted*
closed loop (:mod:`fhew.duel` with :mod:`fhew.evasion`). This package adds the
missing *learned* half of the game and the tool to evaluate it:

    * :mod:`fhew.rl.env`      -- a minimal, history-resolved engagement env + the
      ``HopperPolicy`` / ``JammerPolicy`` interfaces both sides share, with the
      information-parity rule enforced structurally (neither side ever sees the
      other's current move).
    * :mod:`fhew.rl.agents`   -- dependency-free (numpy) learning cores: a linear
      softmax policy with a PPO-clip update, a linear value baseline, and a CVaR
      objective wrapper. No torch required.
    * :mod:`fhew.rl.hopper`   -- the learned anti-jam frequency hopper (the "DQN /
      PPO adversary" of thesis Section 11), both an offline-trained policy and an
      online bandit variant, plus registry generators so a learned hopper is a
      first-class truth-side emitter.
    * :mod:`fhew.rl.jammer`   -- the end-to-end learned jammer (the "PPO / CVaR-PPO
      jammer" baseline of Section 12), a ``JammerPolicy`` trained to deny hops,
      reusing the shared ``EffectModel`` so g and tau match the pipeline.
    * :mod:`fhew.rl.psro`     -- the empirical game: a payoff oracle, a replicator
      meta-Nash solver, the PSRO loop that grows both populations with best-response
      oracles, and the **exploitability certificate** (thesis Sections 9 / 13).

The design mirrors the rest of the codebase: pure numpy, fixed-seed reproducible,
and every learner consumes only what the modelled sensor / radio receiver can
deliver. A learned hopper reads only *jamming feedback* (was my hop denied, which
channels carried jamming energy); a learned jammer reads only *sensor
observations*. Neither reads the opponent's policy, internal belief, or the action
it is about to take -- so the RL results respect the same information-parity rule
the identification and prediction results do.
"""

from __future__ import annotations

from fhew.rl.env import (
    EngagementEnv,
    EpisodeRecord,
    GeneratorHopper,
    HopperObs,
    HopperPolicy,
    JammerPolicy,
    OTSJammerPolicy,
    ScriptedJammer,
    run_engagement,
)

__all__ = [
    # env + interfaces
    "EngagementEnv",
    "EpisodeRecord",
    "HopperObs",
    "HopperPolicy",
    "JammerPolicy",
    "run_engagement",
    # adapters
    "GeneratorHopper",
    "OTSJammerPolicy",
    "ScriptedJammer",
]


def __getattr__(name):
    # Lazy re-exports so importing fhew.rl stays light.
    if name in ("LinearSoftmaxPolicy", "PPOConfig", "LinearBaseline", "cvar_weights"):
        import fhew.rl.agents as a
        return getattr(a, name)
    if name in ("RLHopperAgent", "OnlineBanditHopper", "LearnedHopperGenerator",
                "train_hopper"):
        import fhew.rl.hopper as h
        return getattr(h, name)
    if name in ("RLJammerAgent", "train_jammer"):
        import fhew.rl.jammer as j
        return getattr(j, name)
    if name in ("EmpiricalGame", "PSROResult", "run_psro", "replicator_nash",
                "exploitability"):
        import fhew.rl.psro as p
        return getattr(p, name)
    raise AttributeError(f"module 'fhew.rl' has no attribute {name!r}")
