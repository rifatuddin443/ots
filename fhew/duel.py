"""The closed-loop engagement: the OTS jammer against a reactive adversary.

Experiments exp1-exp25 score the OTS against a *fixed tape* -- a hop sequence drawn
once and sensed, indifferent to the jammer. That is right for measuring
identification and prediction, but it cannot measure what happens when the two sides
actually interact: when the transmitter evades the jam it is receiving
(:mod:`fhew.evasion`) and feeds the collector deceptive detections
(:mod:`fhew.deception`), and the jammer in turn re-predicts and re-allocates against
a moving, lying target. This module runs that loop.

Each slot is resolved from **history alone**, so neither side sees the other's
current move:

    1. The jam power committed for this slot (planned last slot from the prediction)
       lands on the emitter's current channel -> the realised denial.
    2. The sensor observes the emitter's current channel; deception overlays decoy
       detections; the OTS ingests the (corrupted) observation.
    3. The OTS predicts the next channel and commits next slot's jam plan.
    4. The emitter receives *this slot's* jamming feedback (denial + sensed band),
       folds it into its pressure/hot-map, and -- without any sight of the plan just
       committed -- chooses next slot's channel (deflect / spread / re-key).

So the jammer predicts where the emitter will go; the emitter reacts to where it has
already been jammed; they move simultaneously. The result records, per slot, the
realised jam effect on the true channel (the operational "fraction denied"), the
OTS's next-hop hit, its predictability ``kappa`` and tone count, and the adversary's
pressure and re-key count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fhew.allocator import (
    AllocationController,
    CDTController,
    EffectModel,
    predictability,
)
from fhew.channels import ChannelSet
from fhew.conformal import ConformalChannelPredictor, jam_set_from_conformal
from fhew.deception import DeceptionEmitter
from fhew.evasion import AdaptiveEvasionGenerator, JammingFeedback
from fhew.predict import ChannelPredictor
from fhew.sensor import Observation, SensorModel


class OTSJammer:
    """The OTS side of the duel: predict the next channel, then plan a jam over it.

    Wraps a :class:`~fhew.predict.ChannelPredictor` and an
    :class:`~fhew.allocator.EffectModel` + allocation. ``plan`` picks how the
    predictive distribution becomes a power vector:

    * ``"knapsack"`` -- the threshold-aware greedy knapsack with a
      confidence-scaled tone cap (the Control-room behaviour);
    * ``"conformal"`` -- fund the ACI conformal set (the calibrated jam plan of
      ``exp24``), with the tone count set by the data;
    * ``"reactive"`` -- follow the observed channel when the emitter dwells longer
      than the reaction latency, else fall back to the knapsack.

    It consumes only observations (possibly deception-corrupted) and never the true
    channel.
    """

    def __init__(self, channels: ChannelSet, sensor: SensorModel, effect: EffectModel, *,
                 budget: float = 3.0, specs=None, plan: str = "knapsack",
                 max_tones: int = 3, reaction_latency: float = 2.0,
                 alpha: float = 0.1, gamma: float = 0.03, lam0: float = 0.3,
                 monitor: bool = False):
        if plan not in ("knapsack", "conformal", "reactive"):
            raise ValueError("plan must be 'knapsack', 'conformal' or 'reactive'")
        self.channels = channels
        self.M = channels.size
        self.effect = effect
        self.budget = float(budget)
        self.plan = plan
        self.max_tones = int(max_tones)
        self._alpha, self._gamma, self._lam0 = float(alpha), float(gamma), float(lam0)
        self._reaction_latency = float(reaction_latency)
        self.pred = ChannelPredictor(channels, sensor, specs=specs, monitor=monitor)
        self.cpred = (ConformalChannelPredictor(self.pred, alpha=alpha, gamma=gamma)
                      if plan == "conformal" else None)
        # The allocator + its CDT knob are cheap to rebuild, which is how reset()
        # returns them to a clean state so one OTSJammer can be reused across duels
        # (the expensive predictor is only reset, never rebuilt).
        self._new_allocator()
        from fhew.dwell import DwellTracker
        self.dwell = DwellTracker()
        self._last_p = channels.uniform()

    def _new_allocator(self) -> None:
        self.alloc = AllocationController(
            self.effect, self.budget, reaction_latency=self._reaction_latency,
            cdt=CDTController(alpha=self._alpha, lam0=self._lam0), max_tones=self.max_tones)

    def reset(self) -> None:
        self.pred.reset()
        if self.cpred is not None:
            self.cpred.alpha_t = self.cpred.alpha
        self._new_allocator()
        from fhew.dwell import DwellTracker
        self.dwell = DwellTracker()
        self._last_p = self.channels.uniform()

    def observe(self, obs: Observation) -> None:
        self.pred.observe(obs)
        self.dwell.update(obs)

    def predict_next(self) -> np.ndarray:
        self._last_p = self.pred.predict_next()
        return self._last_p

    @staticmethod
    def _clean(obs: Observation) -> int | None:
        return obs.detections[0].channel if len(obs.detections) == 1 else None

    def plan_power(self, p: np.ndarray, observed_channel: int | None = None
                   ) -> tuple[np.ndarray, int, float, list[int] | None]:
        """Return ``(power, n_tones, kappa, conformal_set)`` for the next slot."""
        kappa = predictability(p, self.channels)
        conf_set = None
        if self.plan == "conformal":
            conf_set = self.cpred.prediction_set()
            power = jam_set_from_conformal(conf_set, self.effect, self.budget,
                                           channels=self.channels)
        elif self.plan == "reactive":
            power, _mode = self.alloc.allocate(
                p, observed_channel=observed_channel,
                dwell_estimate=self.dwell.estimate(), current_run=self.dwell.current_run)
        else:  # knapsack
            power, _mode = self.alloc.allocate(p)
        n_tones = int((power > 1e-12).sum())
        return power, n_tones, kappa, conf_set

    def calibrate_conformal(self, covered: bool) -> None:
        if self.cpred is not None:
            self.cpred.calibrate(covered)


@dataclass
class DuelSlot:
    """One slot of the engagement (truth fields are for scoring only)."""

    slot: int
    true_channel: int
    effect_on_true: float          # realised jam effect on the channel actually used
    denied: bool
    predicted_next: int            # OTS top-1 call for the next channel
    pred_hit: bool                 # did the top-1 call match the realised next channel?
    kappa: float
    n_tones: int
    pressure: float                # adversary's denial pressure after this slot
    rekeys: int                    # cumulative re-keys so far
    n_decoys: int


@dataclass
class DuelResult:
    slots: list[DuelSlot] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def aggregate(self, warmup: int = 5) -> dict:
        """Headline metrics over slots past ``warmup`` (the OTS cold-start)."""
        s = [r for r in self.slots if r.slot >= warmup]
        if not s:
            return {}
        eff = np.array([r.effect_on_true for r in s])
        den = np.array([r.denied for r in s], dtype=float)
        hit = np.array([r.pred_hit for r in s if r.predicted_next is not None], dtype=float)
        kap = np.array([r.kappa for r in s])
        ton = np.array([r.n_tones for r in s])
        return {
            "denied_frac": float(den.mean()),
            "mean_effect": float(eff.mean()),
            "nexthop_top1": float(hit.mean()) if hit.size else float("nan"),
            "kappa_mean": float(kap.mean()),
            "mean_tones": float(ton.mean()),
            "rekeys": int(self.slots[-1].rekeys),
            "pressure_final": float(self.slots[-1].pressure),
            "n_slots": len(s),
        }


def run_duel(emitter: AdaptiveEvasionGenerator, jammer: OTSJammer,
             sensor: SensorModel, channels: ChannelSet, rng: np.random.Generator, *,
             window: int = 160, deception: DeceptionEmitter | None = None,
             deny_threshold: float = 0.5) -> DuelResult:
    """Run one closed-loop engagement and return the per-slot record + metadata.

    ``emitter`` evades; ``jammer`` predicts and jams; optional ``deception`` overlays
    decoy detections on what the jammer senses. Both sides are resolved from history
    only (see the module docstring), so the result is a fair game, not a replay.
    """
    emitter.reset()
    jammer.reset()
    if deception is not None:
        deception.reset()
    M = channels.size

    power_cur = np.zeros(M, dtype=np.float64)   # committed for slot 0 (nothing yet)
    c_cur = emitter.next_channel(None)          # c_0: no feedback has arrived
    result = DuelResult(meta={"window": window, "plan": jammer.plan,
                              "deception": deception is not None,
                              "feedback_mode": emitter.cfg.feedback_mode})
    for t in range(window):
        # 1. realised jamming on the current slot (power committed last slot)
        eff = jammer.effect.effect_on(power_cur, c_cur)
        denied = bool(eff >= deny_threshold)

        # 2. sense the true channel, then let deception corrupt what the OTS sees
        o_real = sensor.observe([c_cur], channels, rng, start_slot=t)[0]
        if deception is not None:
            o, decoys = deception.corrupt(o_real, rng)
        else:
            o, decoys = o_real, ()

        # 3. OTS ingests the (corrupted) observation and plans next slot's jam
        jammer.observe(o)
        p_next = jammer.predict_next()
        power_next, n_tones, kappa, _set = jammer.plan_power(
            p_next, observed_channel=jammer._clean(o))
        pred_top1 = int(np.argmax(p_next))

        # 4. adversary gets THIS slot's feedback and picks the next channel
        jammed_mask = (power_cur > 1e-12) if emitter.cfg.feedback_mode == "spectrum" else None
        fb = JammingFeedback(slot=t, effect_on_self=float(eff), denied=denied,
                             jammed_mask=jammed_mask)
        c_next = emitter.next_channel(fb) if t < window - 1 else None

        # conformal calibration: did the committed jam set cover the realised next hop?
        if jammer.cpred is not None and c_next is not None and _set is not None:
            jammer.calibrate_conformal(c_next in _set)

        result.slots.append(DuelSlot(
            slot=t, true_channel=int(c_cur), effect_on_true=float(eff), denied=denied,
            predicted_next=pred_top1,
            pred_hit=bool(c_next is not None and pred_top1 == c_next),
            kappa=float(kappa), n_tones=int(n_tones), pressure=float(emitter.pressure),
            rekeys=int(emitter.rekeys), n_decoys=len(decoys)))

        power_cur = power_next
        c_cur = c_next if c_next is not None else c_cur
    return result
