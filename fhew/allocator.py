"""Layer 4: threshold-aware multi-tone power allocation (thesis Section 7).

A single next-channel prediction is not enough to act on: under any real
predictor the mass is spread over several channels, and jamming is a *threshold*
effect -- below a J/S threshold ``tau`` a channel is not denied at all, and power
beyond ``tau`` is wasted. So the action is a power *vector* over channels (a
multi-tone / multiple-frequency jam), and the real question per channel is
whether it is worth paying the power ``c_i`` needed to cross ``tau`` there. That
makes the allocation a knapsack, not a water-filling, problem -- the main change
of Layer 4.

The behaviour the thesis wants falls out of one allocator plus one calibrated
knob ``lambda`` (no operator-selected mode):

    * **Confident prediction** -> a few channels hold the mass, the budget
      denies them fully: *spot* jamming.
    * **Uncertain prediction** -> the mass is spread, the budget denies as many
      channels as it can afford at ``tau``: *barrage*-like.

``lambda`` (the conservatism knob, calibrated online by :class:`CDTController`,
Section 6.3) slides between the two by raising a floor on the predictive before
ranking. This module also carries the **reactive / follower** allocation
(Section 7 and the feasibility map): when the emitter dwells longer than the
reaction latency, jam the *observed* channel directly -- no prediction needed,
whatever kappa says.

On top of ``lambda`` there is an explicit **multiple-transmit-frequency** cap:
``max_tones`` bounds how many channels (tones) the predictive allocator may light
up in one slot. The *active* count scales with prediction confidence
(:func:`tones_from_confidence`) -- a sharp prediction concentrates to a single
spot tone, and an unsure one opens out to ``max_tones``. ``max_tones`` is the
operator control ("how many frequencies may we transmit on at once when we are
not sure?"); a real jammer has a finite number of tone generators, and this is
that limit.

Everything here is pure numpy over the predictive distribution and a shared
effect model; the same ``g`` and ``tau`` drive the conformal loss, so the risk
guarantee and the allocation agree (the "one thing to check", Section 15).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fhew.channels import ChannelSet


@dataclass
class EffectModel:
    """The jamming effect function ``g`` and the denial threshold ``tau``.

    ``gain`` is the per-channel power-to-J/S gain (geometry + the hopper's signal
    strength); a channel's cost to cross ``tau`` is ``c_i = tau / gain_i``. The
    effect of placing power ``p`` on a channel is a steep logistic around
    ``tau`` (``sharpness`` controls how step-like), clipped to ``[0, 1]`` -- so
    below ``tau`` the effect is ~0 and at/above it the channel is ~denied. The
    *same* instance is read by the allocator and by the conformal loss, which is
    what keeps the risk guarantee describing the allocation (Section 15).
    """

    channels: ChannelSet
    tau: float = 1.0
    sharpness: float = 8.0
    gain: np.ndarray | None = None
    cost_margin: float = 1.3        # fund a little above tau so crossing ~denies

    def __post_init__(self) -> None:
        M = self.channels.size
        if self.gain is None:
            self.gain = np.ones(M, dtype=np.float64)
        else:
            self.gain = np.asarray(self.gain, dtype=np.float64)
            if self.gain.shape != (M,):
                raise ValueError("gain must have one entry per channel")
        if self.tau <= 0:
            raise ValueError("tau must be positive")

    def costs(self) -> np.ndarray:
        """Power needed to cross ``tau`` (to ~denial) on each channel.

        ``tau`` is the half-effect point; funding ``cost_margin * tau`` of J/S
        places the channel on the denied side of the knee, so a funded channel
        is effectively (not just marginally) jammed.
        """
        return self.cost_margin * self.tau / self.gain

    def effect(self, power: np.ndarray) -> np.ndarray:
        """Per-channel effect ``g(power)`` in ``[0, 1]``."""
        js = np.asarray(power, dtype=np.float64) * self.gain
        return 1.0 / (1.0 + np.exp(-self.sharpness * (js - self.tau) / self.tau))

    def effect_on(self, power: np.ndarray, channel: int) -> float:
        """Realised effect on the channel the adversary actually used."""
        return float(self.effect(power)[int(channel)])


def _effective_mask(channels: ChannelSet) -> np.ndarray:
    mask = np.zeros(channels.size, dtype=bool)
    mask[channels.effective_indices()] = True
    return mask


def predictability(p: np.ndarray, channels: ChannelSet | None = None) -> float:
    """Confidence score ``kappa = 1 - H(p)/log M`` of a predictive vector.

    Matches :meth:`fhew.mixture.FixedShareMixture.predictability` (Section 4.7):
    ``kappa = 1`` means the next channel is pinned down (sharp prediction),
    ``kappa = 0`` means the mass is uniform (we are *not sure* at all). The
    multi-tone cap below reads this to decide how many frequencies to light up.
    """
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    M = p.size if channels is None else channels.size
    return float(np.clip(1.0 - H / np.log(M), 0.0, 1.0))


def tones_from_confidence(kappa: float, max_tones: int, *, min_tones: int = 1) -> int:
    """How many transmit frequencies to light up given prediction confidence.

    This is the "multiple transmit frequency" rule: **when we are not too sure
    about the prediction, spread across more frequencies, up to an operator-set
    maximum.** The count grows linearly as confidence falls --

        n = round(min_tones + (max_tones - min_tones) * (1 - kappa))

    so a sharp prediction (``kappa -> 1``) collapses to ``min_tones`` (a single
    spot tone) and a flat one (``kappa -> 0``) opens all ``max_tones`` tones. The
    result is clamped to ``[min_tones, max_tones]``. ``max_tones`` is the knob
    exposed in the controls; raising it lets the system hedge across more
    channels when it is uncertain, at the cost of less power per tone.
    """
    max_tones = max(int(max_tones), int(min_tones))
    kappa = float(np.clip(kappa, 0.0, 1.0))
    n = min_tones + (max_tones - min_tones) * (1.0 - kappa)
    return int(np.clip(round(n), min_tones, max_tones))


def greedy_knapsack(p: np.ndarray, effect: EffectModel, budget: float, *,
                    lam: float = 0.0, channels: ChannelSet | None = None,
                    max_tones: int | None = None) -> np.ndarray:
    """Greedy threshold-aware knapsack allocation (main method, Section 7.2).

    1. Raise a conservatism floor on the predictive: ``p' = (1-lam) p + lam u``
       over the effective channel set. ``lam=0`` keeps the raw prediction
       (spot-leaning); ``lam->1`` flattens it toward barrage.
    2. Rank channels by value density ``p'_i / c_i``.
    3. Fully fund channels in that order until the budget runs out; spend any
       remainder partially on the next channel.

    ``max_tones`` caps the number of channels (transmit frequencies) that may be
    funded in one slot. ``None`` (default) leaves the count implicit, bounded
    only by the budget -- the original behaviour. A finite cap models a jammer
    with a fixed number of tone generators: no more than ``max_tones`` channels
    are ever lit, and any budget left after the cap is reached goes unspent
    (power beyond ``tau`` on a funded channel is wasted anyway).
    """
    p = np.asarray(p, dtype=np.float64)
    M = p.size
    channels = channels or effect.channels
    costs = effect.costs()
    eff_mask = _effective_mask(channels)
    u = np.zeros(M)
    u[eff_mask] = 1.0 / max(1, int(eff_mask.sum()))
    lam = float(np.clip(lam, 0.0, 1.0))
    p_eff = (1.0 - lam) * p + lam * u
    p_eff = np.where(eff_mask, p_eff, 0.0)

    density = np.where(costs > 0, p_eff / costs, 0.0)
    order = np.argsort(density)[::-1]
    power = np.zeros(M, dtype=np.float64)
    remaining = float(budget)
    cap = M if max_tones is None else max(1, int(max_tones))
    funded = 0
    for i in order:
        if remaining <= 0 or p_eff[i] <= 0 or funded >= cap:
            break
        pay = min(costs[i], remaining)
        power[i] = pay
        remaining -= pay
        funded += 1
    return power


def water_filling(p: np.ndarray, effect: EffectModel, budget: float, *,
                  weighted: bool = False, channels: ChannelSet | None = None) -> np.ndarray:
    """Concave-``g`` water-filling baseline (Section 7.3).

    Allocates proportionally to the predictive (``weighted`` also folds in the
    per-channel gain), ignoring the threshold. It over-spreads below ``tau`` and
    is the baseline the knapsack is meant to beat when the prediction is sharp.
    """
    p = np.asarray(p, dtype=np.float64)
    channels = channels or effect.channels
    eff_mask = _effective_mask(channels)
    w = np.where(eff_mask, p, 0.0)
    if weighted:
        w = w * effect.gain
    s = w.sum()
    if s <= 0:
        w = eff_mask.astype(np.float64)
        s = w.sum()
    return budget * w / s


def equal_power(effect: EffectModel, budget: float, support: np.ndarray | None = None,
                *, channels: ChannelSet | None = None) -> np.ndarray:
    """Equal power across a support set (simple baseline, Section 7.3)."""
    channels = channels or effect.channels
    if support is None:
        support = channels.effective_indices()
    support = np.asarray(support, dtype=np.int64)
    power = np.zeros(channels.size, dtype=np.float64)
    if support.size:
        power[support] = budget / support.size
    return power


def spot(p: np.ndarray, effect: EffectModel, budget: float) -> np.ndarray:
    """Put the whole budget on the single most-likely channel (fixed-spot)."""
    power = np.zeros_like(p, dtype=np.float64)
    power[int(np.argmax(p))] = budget
    return power


def reactive_follower(observed_channel: int, effect: EffectModel, budget: float, *,
                      p: np.ndarray | None = None, lam: float = 0.0,
                      channels: ChannelSet | None = None) -> np.ndarray:
    """Reactive / follower allocation (Section 7, the feasibility floor).

    Jam the channel the emitter is *currently* sitting on: fund it to ``tau``
    first (detect-and-follow, no prediction), then spend any remaining budget via
    the knapsack over the predictive. This is the mode that works whenever the
    reaction latency is shorter than the dwell, and it is the floor against a
    slow keyed hopper that prediction can never beat.
    """
    channels = channels or effect.channels
    costs = effect.costs()
    power = np.zeros(channels.size, dtype=np.float64)
    c = int(observed_channel)
    pay = min(costs[c], float(budget))
    power[c] = pay
    remaining = float(budget) - pay
    if remaining > 1e-12 and p is not None:
        resid = np.asarray(p, dtype=np.float64).copy()
        resid[c] = 0.0                       # already funded
        power = power + greedy_knapsack(resid, effect, remaining, lam=lam, channels=channels)
    return power


class CDTController:
    """Conformal Decision Theory controller on the conservatism knob ``lambda``
    (thesis Section 6.3).

    Updates ``lambda_{t+1} = lambda_t + eta (L_t - alpha)`` where ``L_t`` is the
    realised *effect forgone* in ``[0, 1]`` and ``alpha`` the target. The
    guarantee is a long-run average: the mean realised loss stays at or below
    ``alpha + O(1/T)`` for *any* sequence, because ``lambda`` is bounded and the
    update is a bounded-gradient online step. ``lambda`` rising spreads the
    allocation (more barrage); falling concentrates it (more spot).
    """

    def __init__(self, alpha: float = 0.2, eta: float = 0.05, *, lam0: float = 0.3,
                 lam_min: float = 0.0, lam_max: float = 1.0):
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.lam = float(np.clip(lam0, lam_min, lam_max))
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.losses: list[float] = []

    def update(self, loss_forgone: float) -> float:
        L = float(np.clip(loss_forgone, 0.0, 1.0))
        self.losses.append(L)
        self.lam = float(np.clip(self.lam + self.eta * (L - self.alpha),
                                 self.lam_min, self.lam_max))
        return self.lam

    @property
    def mean_loss(self) -> float:
        return float(np.mean(self.losses)) if self.losses else float("nan")


@dataclass
class AllocationStep:
    """One slot's allocation output."""

    power: np.ndarray
    mode: str                  # 'predictive' | 'reactive'
    lam: float
    effect_on_true: float      # realised jamming effect on the true next channel
    loss_forgone: float        # 1 - effect_on_true, in [0, 1]
    n_tones: int = 0           # transmit frequencies actually lit this slot
    kappa: float = float("nan")  # prediction confidence that set the tone count


class AllocationController:
    """Picks the allocation mode and drives the CDT knob (Sections 6.3 / 7).

    Mode selection is the explicit second weapon of Section 7: if the emitter
    has dwelt longer than the reaction latency, use reactive / follower jamming
    on the observed channel *regardless of kappa*; otherwise allocate from the
    predictive with the greedy knapsack, with ``lambda`` deciding spot vs
    barrage. The controller is online: after the true next channel is revealed
    it books the realised effect-forgone loss and steps ``lambda``.
    """

    def __init__(self, effect: EffectModel, budget: float, *,
                 reaction_latency: float = 1.0, cdt: CDTController | None = None,
                 allocator: str = "greedy_knapsack", max_tones: int | None = None):
        self.effect = effect
        self.channels = effect.channels
        self.budget = float(budget)
        self.reaction_latency = float(reaction_latency)
        self.cdt = cdt or CDTController()
        self.allocator = allocator
        # Maximum number of transmit frequencies (tones) the predictive allocator
        # may light up in one slot. ``None`` leaves the count budget-bounded
        # (original behaviour). When set, the active count scales with prediction
        # confidence (:func:`tones_from_confidence`): sharp -> 1 tone, unsure ->
        # up to ``max_tones``.
        self.max_tones = None if max_tones is None else max(1, int(max_tones))
        self.last_n_tones = 0
        self.last_kappa = float("nan")

    def allocate(self, p: np.ndarray, *, observed_channel: int | None = None,
                 dwell_estimate: float = 1.0, current_run: int = 0) -> tuple[np.ndarray, str]:
        """Return ``(power_vector, mode)`` for this slot's action.

        Reactive / follower mode is chosen when the emitter's estimated dwell
        exceeds the reaction latency (the feasibility condition, independent of
        kappa) and a current channel has been observed to follow.

        In predictive mode with ``max_tones`` set, the number of frequencies lit
        this slot is driven by the prediction confidence ``kappa``: a confident
        prediction concentrates to a single tone, an unsure one spreads across up
        to ``max_tones`` tones.
        """
        self.last_kappa = predictability(p, self.channels)
        reactive = (observed_channel is not None
                    and dwell_estimate > self.reaction_latency
                    and current_run >= self.reaction_latency)
        if reactive:
            power = reactive_follower(observed_channel, self.effect, self.budget,
                                      p=p, lam=self.cdt.lam, channels=self.channels)
            self.last_n_tones = int((power > 1e-12).sum())
            return power, "reactive"
        if self.allocator == "water_filling":
            power = water_filling(p, self.effect, self.budget, channels=self.channels)
        elif self.allocator == "weighted_water_filling":
            power = water_filling(p, self.effect, self.budget, weighted=True,
                                  channels=self.channels)
        elif self.allocator == "equal":
            power = equal_power(self.effect, self.budget, channels=self.channels)
        elif self.allocator == "spot":
            power = spot(p, self.effect, self.budget)
        else:
            cap = (None if self.max_tones is None
                   else tones_from_confidence(self.last_kappa, self.max_tones))
            power = greedy_knapsack(p, self.effect, self.budget, lam=self.cdt.lam,
                                    channels=self.channels, max_tones=cap)
        self.last_n_tones = int((power > 1e-12).sum())
        return power, "predictive"

    def step(self, p: np.ndarray, true_next: int, *, observed_channel: int | None = None,
             dwell_estimate: float = 1.0, current_run: int = 0) -> AllocationStep:
        """Allocate, score against the revealed next channel, and update lambda."""
        power, mode = self.allocate(p, observed_channel=observed_channel,
                                    dwell_estimate=dwell_estimate, current_run=current_run)
        eff = self.effect.effect_on(power, int(true_next))
        loss = 1.0 - eff
        self.cdt.update(loss)
        return AllocationStep(power=power, mode=mode, lam=self.cdt.lam,
                              effect_on_true=eff, loss_forgone=loss,
                              n_tones=self.last_n_tones, kappa=self.last_kappa)
