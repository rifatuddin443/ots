"""The feasibility map (thesis Section 10), with the reactive-jamming region.

The feasibility map names which strategy is achievable in which regime, so "what
if the enemy comes fast / hops unpredictably?" has a principled answer: it names
the regime and shows the system selects the best strategy available there. The
merged map had three regions -- exact recovery, short-horizon prediction, and
barrage. Revision 3 adds a fourth, orthogonal one:

    * **reactive / follower** -- feasible whenever the dwell exceeds the reaction
      latency, *independent of predictability*. This is the region where a slow
      keyed hopper (uncrackable by prediction) is still jammable by
      detect-and-follow. Its counter is the entropy-maximising hopper that both
      spreads its channels (defeats prediction) and shortens its dwell (defeats
      reactive jamming) -- the true predictability boundary.

:func:`classify` returns the single best available strategy for a regime;
:func:`dwell_latency_grid` and :func:`window_hoprate_grid` build the 2-D maps the
experiment plots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ordered best-to-worst: the system always takes the strongest feasible one.
REGIONS = ("exact_recovery", "short_horizon_prediction", "reactive_follower", "barrage")

REGION_COLORS = {
    "exact_recovery": "#0072B2",
    "short_horizon_prediction": "#56B4E9",
    "reactive_follower": "#009E73",
    "barrage": "#999999",
}


@dataclass
class RegimeParams:
    """The knobs that decide which strategy is feasible."""

    window: int                     # engagement window T, in slots
    hop_rate: float = 1.0           # hops per slot (>=1 means multiple hops/slot)
    dwell: float = 1.0              # slots the emitter holds each channel
    reaction_latency: float = 1.0   # detect-to-emit latency, in slots
    kappa: float = 0.0              # measured predictability in [0, 1]
    register_length: int = 7        # L, for the ~2L clean-hop recovery budget
    p_d: float = 1.0                # detection probability (thins the clean run)
    key_strength: float = 0.0       # >0 marks a keyed/unbreakable generator


def can_recover_exactly(rp: RegimeParams) -> bool:
    """Enough clean consecutive hops within the window for algebraic recovery.

    Berlekamp-Massey needs about ``2L`` clean *consecutive* outputs; detection
    thinning (``p_d``) and a long dwell (fewer distinct hops per slot) shrink the
    usable budget. A keyed generator is never exactly recoverable.
    """
    if rp.key_strength > 0:
        return False
    distinct_hops = rp.window * rp.hop_rate / max(1.0, rp.dwell)
    clean_consecutive = distinct_hops * (rp.p_d ** 2)   # a gap breaks the run
    return clean_consecutive >= 2 * rp.register_length


def can_predict_short_horizon(rp: RegimeParams, *, kappa_min: float = 0.25) -> bool:
    """Predictable enough to concentrate power on a short-horizon next-hop set."""
    return rp.key_strength <= 0 and rp.kappa >= kappa_min


def can_react(rp: RegimeParams) -> bool:
    """Reactive / follower feasible: the dwell outlasts the reaction latency.

    This holds *whatever* kappa says -- it is the floor against an unpredictable
    but slow hopper.
    """
    return rp.dwell > rp.reaction_latency and rp.reaction_latency >= 1


def classify(rp: RegimeParams, *, kappa_min: float = 0.25) -> str:
    """The single strongest strategy feasible in this regime."""
    if can_recover_exactly(rp):
        return "exact_recovery"
    if can_predict_short_horizon(rp, kappa_min=kappa_min):
        return "short_horizon_prediction"
    if can_react(rp):
        return "reactive_follower"
    return "barrage"


def region_index(region: str) -> int:
    return REGIONS.index(region)


def dwell_latency_grid(dwells, latencies, *, base: RegimeParams | None = None,
                       kappa_min: float = 0.25) -> np.ndarray:
    """Region index over a (dwell, reaction-latency) grid.

    Rows index ``dwells``, columns index ``latencies``. With a low-kappa base
    (an unpredictable hopper) this isolates the reactive-follower region as its
    own band: the upper-left (long dwell, short latency) is jammable by
    detect-and-follow even though prediction is hopeless.
    """
    base = base or RegimeParams(window=100, kappa=0.0, key_strength=1.0)
    grid = np.empty((len(dwells), len(latencies)), dtype=np.int64)
    for i, d in enumerate(dwells):
        for j, lat in enumerate(latencies):
            rp = RegimeParams(**{**base.__dict__, "dwell": float(d),
                                 "reaction_latency": float(lat)})
            grid[i, j] = region_index(classify(rp, kappa_min=kappa_min))
    return grid


def window_hoprate_grid(windows, hop_rates, *, base: RegimeParams | None = None,
                        kappa_min: float = 0.25, evidence_kappa: bool = True,
                        evidence_ref: float = 120.0, kappa_cap: float = 0.95) -> np.ndarray:
    """Region index over a (window, hop-rate) grid (the classic Section 10 map).

    With ``evidence_kappa`` (default) the *exploitable* predictability grows with
    the evidence seen -- ``kappa_eff = min(kappa_cap, base.kappa · hops / ref)``
    where ``hops = window · hop_rate`` -- so a short engagement window, which
    yields too few hops either to recover a generator or to build a confident
    prediction, correctly falls to the barrage region. This is what gives the
    classic three-band partition (barrage / short-horizon / exact-recovery)
    instead of a single region, and it matches the short-window constraint that
    reshapes the whole thesis (fewer hops observed = less predictability to spend).
    """
    base = base or RegimeParams(window=100, kappa=0.9, register_length=7, p_d=0.9)
    grid = np.empty((len(windows), len(hop_rates)), dtype=np.int64)
    for i, w in enumerate(windows):
        for j, hr in enumerate(hop_rates):
            kappa = base.kappa
            if evidence_kappa:
                hops = float(w) * float(hr)
                kappa = min(kappa_cap, base.kappa * hops / evidence_ref)
            rp = RegimeParams(**{**base.__dict__, "window": int(w), "hop_rate": float(hr),
                                 "kappa": kappa})
            grid[i, j] = region_index(classify(rp, kappa_min=kappa_min))
    return grid
