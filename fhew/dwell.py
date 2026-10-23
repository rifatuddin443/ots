"""Dwell-time estimation: a cheap predictability feature and the reactive trigger.

Staying on one channel too long gives two separate weapons (thesis Section 7 /
the feasibility map):

    1. It is *predictable*. Dwell time per channel is a cheap, strong feature:
       a long dwell pushes the predictability score kappa up directly, because
       the next hop is very likely the current channel.

    2. It is *reactively jammable*. If the emitter sits on a channel longer than
       the jammer's reaction latency, the hop can be detected and then jammed
       while it is still there -- no prediction needed at all. "Unpredictable"
       (a keyed CSPRNG) is not the same as "un-jammable": a slow crypto hopper is
       still beaten by detect-and-follow.

This module estimates the current dwell from the observation stream (clean,
single-detection slots), both online (:class:`DwellTracker`) and in batch
(:func:`estimate_dwell`), and maps a dwell to a predictability contribution
(:func:`dwell_predictability`). It consumes only :class:`Observation` objects,
so it respects information parity like everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fhew.sensor import Observation


def _clean_channel(obs: Observation) -> int | None:
    if len(obs.detections) == 1:
        return obs.detections[0].channel
    return None


def dwell_predictability(dwell: float, *, reaction_latency: float = 1.0) -> float:
    """Map an estimated dwell to a predictability contribution in ``[0, 1)``.

    With a dwell of ``D`` slots, repeating the current channel is correct for
    ``D-1`` of every ``D`` hops, so a "stay" predictor is right with probability
    ``1 - 1/D``. We also require the dwell to exceed the reaction latency to
    count it as *exploitable* (below the latency the jammer cannot react in
    time), which is exactly the reactive-follower condition.
    """
    d = max(1.0, float(dwell))
    if d <= float(reaction_latency):
        return 0.0
    return float(1.0 - 1.0 / d)


@dataclass
class DwellTracker:
    """Online estimate of the emitter's dwell (mean run length on one channel).

    Updated from clean slots only. ``mean_dwell`` is an EWMA over completed
    runs; ``current_run`` is how long the current channel has been held so far
    (the quantity a reactive jammer compares against its reaction latency *now*).
    """

    ewma: float = 0.3
    mean_dwell: float = 1.0
    current_channel: int | None = None
    current_run: int = 0
    _completed: list[int] = field(default_factory=list)

    def update(self, obs: Observation) -> None:
        c = _clean_channel(obs)
        if c is None:
            return                      # gap/phantom slot carries no clean run info
        if c == self.current_channel:
            self.current_run += 1
        else:
            if self.current_channel is not None and self.current_run > 0:
                self._completed.append(self.current_run)
                self.mean_dwell = ((1.0 - self.ewma) * self.mean_dwell
                                   + self.ewma * self.current_run)
            self.current_channel = c
            self.current_run = 1

    def estimate(self) -> float:
        """Best current dwell estimate (blends completed runs and the live run)."""
        return float(max(self.mean_dwell, self.current_run))

    def reactive_ready(self, reaction_latency: float) -> bool:
        """True iff the emitter has already sat longer than the reaction latency,
        so a detect-and-follow jam would land before it hops."""
        return self.current_run >= float(reaction_latency)


def estimate_dwell(observations, *, min_runs: int = 1) -> float:
    """Batch mean dwell over an observation stream (clean slots only)."""
    runs: list[int] = []
    cur, run = None, 0
    for o in observations:
        c = _clean_channel(o)
        if c is None:
            continue
        if c == cur:
            run += 1
        else:
            if cur is not None and run > 0:
                runs.append(run)
            cur, run = c, 1
    if cur is not None and run > 0:
        runs.append(run)
    if len(runs) < min_runs:
        return 1.0
    return float(np.mean(runs))
