"""The detection-level sensor model and the frozen observation schema (v1).

The input to the system is a stream of sensor outputs, one per slot, *not* the
true channel sequence (paper Section 4). The sensor is modelled at the detection
level, not at IQ level, which keeps the paper focused on the inference problem.

Emission model (used twice: to *generate* observations and inside *every*
likelihood -- this double use is the core of the sensor-aware claim):

    * the true channel ``c_t`` produces a detection with probability ``P_d``;
    * every other channel independently produces a false detection with
      probability ``P_fa``.

Observation schema v1 (frozen): a detection is
``(slot, channel, power, bearing, emitter_id)`` with ``power``, ``bearing`` and
``emitter_id`` optional and single-emitter here. Freezing it now means the
journal's multi-emitter and probing additions are extensions, not rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from fhew.channels import ChannelSet

SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class Detection:
    """A single detection in one slot (observation schema v1).

    ``power``, ``bearing`` and ``emitter_id`` are optional fields the journal
    uses (RF fingerprinting, track-aware sorting, multi-emitter). They are logged
    but not required by the conference bank, which reads only ``channel``.
    """

    slot: int
    channel: int
    power: Optional[float] = None
    bearing: Optional[float] = None
    emitter_id: Optional[int] = None


@dataclass(frozen=True)
class Observation:
    """The sensor output ``o_t`` for one slot: a set of detections.

    The bank consumes only :class:`Observation` objects (never the true channel),
    which is how the information-parity rule (paper Section 11) is made
    structural rather than a convention.
    """

    slot: int
    detections: tuple[Detection, ...]
    n_channels: int

    @property
    def detected_channels(self) -> frozenset[int]:
        return frozenset(d.channel for d in self.detections)

    def detection_mask(self) -> np.ndarray:
        """Boolean mask over channels: True where at least one detection fell."""
        mask = np.zeros(self.n_channels, dtype=bool)
        for d in self.detections:
            mask[d.channel] = True
        return mask

    def is_empty(self) -> bool:
        return len(self.detections) == 0


class SensorModel:
    """Detection-level sensor: miss probability ``P_d`` and false-alarm ``P_fa``.

    The same instance is used to generate observations (:meth:`observe`) and to
    build the reduced emission vector (:meth:`emission_logits`) consumed by every
    likelihood, so the inference matches the sensor by construction.

    Parameters
    ----------
    p_d:
        Probability the true hop is detected, in ``(0, 1]``.
    p_fa:
        False-alarm rate per channel per slot, in ``[0, 1)``.
    power_model / bearing_model:
        Optional callables ``(rng) -> float`` for the optional schema-v1 fields.
        When ``None``, those fields are left unset. They never affect the
        conference likelihood; they exist so logged runs carry the fields the
        journal needs.
    """

    version = SCHEMA_VERSION

    def __init__(
        self,
        p_d: float,
        p_fa: float,
        *,
        power_model=None,
        bearing_model=None,
        clip_eps: float = 1e-9,
    ) -> None:
        if not 0.0 < p_d <= 1.0:
            raise ValueError("p_d must be in (0, 1]")
        if not 0.0 <= p_fa < 1.0:
            raise ValueError("p_fa must be in [0, 1)")
        self.p_d = float(p_d)
        self.p_fa = float(p_fa)
        self.power_model = power_model
        self.bearing_model = bearing_model
        self.clip_eps = clip_eps

    # ------------------------------------------------------------------ generate
    def observe(
        self,
        true_channels: Sequence[int],
        channels: ChannelSet,
        rng: np.random.Generator,
        *,
        emitter_id: Optional[int] = 0,
        start_slot: int = 0,
    ) -> list[Observation]:
        """Generate the lossy observation stream for a true channel sequence."""
        obs: list[Observation] = []
        M = channels.size
        for i, c in enumerate(true_channels):
            slot = start_slot + i
            dets: list[Detection] = []
            # True-channel detection with probability P_d.
            if rng.random() < self.p_d:
                dets.append(self._make_detection(slot, int(c), rng, emitter_id))
            # Independent false alarms on every *other* channel with prob P_fa.
            if self.p_fa > 0.0:
                fa_mask = rng.random(M) < self.p_fa
                fa_mask[int(c)] = False  # the true channel is handled above
                for ch in np.nonzero(fa_mask)[0]:
                    dets.append(self._make_detection(slot, int(ch), rng, emitter_id))
            obs.append(Observation(slot=slot, detections=tuple(dets), n_channels=M))
        return obs

    def _make_detection(
        self, slot: int, channel: int, rng: np.random.Generator, emitter_id
    ) -> Detection:
        power = float(self.power_model(rng)) if self.power_model is not None else None
        bearing = float(self.bearing_model(rng)) if self.bearing_model is not None else None
        return Detection(slot=slot, channel=channel, power=power, bearing=bearing,
                         emitter_id=emitter_id)

    # --------------------------------------------------------------- emission
    def emission_loglikelihood(self, obs: Observation) -> np.ndarray:
        r"""Log of the *reduced* emission vector ``log e_t(s)`` over channels ``s``.

        The exact per-slot emission factorises as

            P(o_t | true = s) = F_t * ratio(s),

        where ``F_t = prod_c P_fa^{1[c in D]} (1-P_fa)^{1[c notin D]}`` is the
        all-false-alarm baseline (common to every hypothesis and every state, so
        it cancels in the posterior over hypotheses and in all likelihood ratios)
        and

            ratio(s) = (P_d / P_fa)       if channel s was detected,
                       (1-P_d)/(1-P_fa)   otherwise.

        We return ``log ratio(s)`` -- the reduced emission. Because ``F_t`` is
        dropped, per-slot likelihoods are comparable *across hypotheses and across
        slots* but are not raw probabilities; this is exactly what the Bayesian
        posterior and the e-process need, and it is numerically stable.

        When ``P_fa == 0`` the ratio for a detected channel is defined by a
        detected false alarm being impossible; we fall back to an exact
        probability-based emission (see :meth:`_emission_log_pfa_zero`).
        """
        if self.p_fa == 0.0:
            return self._emission_log_pfa_zero(obs)

        p_d = np.clip(self.p_d, self.clip_eps, 1.0 - self.clip_eps) if self.p_d < 1.0 else 1.0 - self.clip_eps
        p_fa = np.clip(self.p_fa, self.clip_eps, 1.0 - self.clip_eps)
        detected = obs.detection_mask()
        log_ratio = np.empty(obs.n_channels, dtype=np.float64)
        # detected channels: log(P_d / P_fa); undetected: log((1-P_d)/(1-P_fa))
        log_ratio[detected] = np.log(p_d) - np.log(p_fa)
        log_ratio[~detected] = np.log(1.0 - p_d) - np.log(1.0 - p_fa)
        return log_ratio

    def _emission_log_pfa_zero(self, obs: Observation) -> np.ndarray:
        """Emission when there are no false alarms (``P_fa == 0``).

        With no false alarms, a detection pins the true channel exactly, while an
        empty slot is an informative miss. We still return a *reduced* vector (up
        to a per-slot constant) so downstream code is uniform.
        """
        p_d = np.clip(self.p_d, self.clip_eps, 1.0 - self.clip_eps) if self.p_d < 1.0 else 1.0 - self.clip_eps
        detected = obs.detection_mask()
        log_e = np.empty(obs.n_channels, dtype=np.float64)
        n_det = int(detected.sum())
        if n_det == 0:
            # No detection: every state explains it only through a miss.
            log_e[:] = np.log(1.0 - p_d)
        else:
            # A detection can only come from the true channel (no false alarms),
            # so any state not on a detected channel is impossible (log -> -inf,
            # clipped). Multiple detections with P_fa==0 cannot be explained by a
            # single-emitter model; we treat each detected channel as admissible.
            very_small = np.log(self.clip_eps)
            log_e[:] = very_small
            log_e[detected] = np.log(p_d)
        return log_e

    def emission_vector(self, obs: Observation) -> np.ndarray:
        """Linear (non-log) reduced emission vector, for the naive-free path."""
        return np.exp(self.emission_loglikelihood(obs))

    # --------------------------------------------------------------- summaries
    def expected_false_alarms(self, channels: ChannelSet) -> float:
        """Expected phantom hops per slot, ~ P_fa * M (paper Section 4)."""
        return self.p_fa * channels.size

    def as_dict(self) -> dict:
        return {"p_d": self.p_d, "p_fa": self.p_fa, "schema": self.version}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SensorModel(p_d={self.p_d:.3f}, p_fa={self.p_fa:.4f})"


def clean_subsequence(observations: Iterable[Observation]) -> list[tuple[int, int]]:
    """Return ``(slot, channel)`` pairs for slots with exactly one detection.

    These are the "clean" slots a naive/algebraic recovery would trust. They are
    *not* guaranteed to be the true channel (a false alarm can be the sole
    detection), which is exactly why plain Berlekamp-Massey is brittle under a
    lossy sensor -- the gap-robust variant exists to handle this.
    """
    out: list[tuple[int, int]] = []
    for o in observations:
        if len(o.detections) == 1:
            out.append((o.slot, o.detections[0].channel))
    return out
