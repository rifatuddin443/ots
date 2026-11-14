"""Closed-loop jammer *evasion*: a reactive anti-jam transmitter (thesis red team).

Everything else in this package treats the emitter as **open loop** -- it draws a
hop sequence and is sensed, oblivious to whether anyone is jamming it. That is the
fair setting for the identification and prediction claims, but it is not the setting
a fielded adversary operates in: a real frequency-hopping radio has a receiver, so
it *knows* when a link is being denied and where the interference sits, and it
adapts. This module adds that missing half of the engagement -- the emitter that
fights back -- so the OTS predict-then-jam loop can be evaluated against an
adversary that is trying to escape it, not just against a fixed tape.

The transmitter is given only **jamming feedback** (:class:`JammingFeedback`), the
physical-layer signal a radio genuinely has:

    * ``denied`` / ``effect_on_self`` -- was *my* hop jammed? (a missing ACK / a
      failed CRC: always available to a transmitter);
    * ``jammed_mask`` -- which channels carried jamming energy this slot, sensed by
      the radio's own receiver (available only in the ``"spectrum"`` feedback mode;
      a receiver-less transmitter runs in ``"denial_only"`` mode and learns only
      about channels it actually sat on).

It is *never* given the OTS's internal prediction, posterior, or the power it is
about to place next slot -- that would be clairvoyance, and the duel
(:mod:`fhew.duel`) is careful to resolve both sides from history alone. On a slot
with no feedback (or none passed), the evader reproduces its base generator exactly,
so switching evasion on changes nothing until jamming actually appears.

Three escalating behaviours, driven by a single scalar **pressure** ``P in [0,1]``
(an EWMA of how often this radio's own hops are being denied):

    1. **Deflection.** Keep a decaying map ``h`` of recently jammed channels. When
       the base generator proposes a channel that is hot, hop to a cold one instead.
       Cheap, and enough against a slow spot jammer.
    2. **Entropy escalation.** As pressure rises, draw the evasive hop from a
       *flatter* distribution over the cold channels (a temperature that grows with
       ``P``). This is the direct attack on the predictor: it drives the OTS
       predictability ``kappa`` down toward the barrage floor -- "if you can predict
       me, I stop being predictable."
    3. **Re-key.** When pressure crosses ``rekey_pressure`` the radio changes its
       hop pattern wholesale (a fresh key / seed via :attr:`rekey_factory`), which
       invalidates any lock the OTS has formed on the old deterministic sequence and
       forces it to re-identify from scratch. Pressure is cooled after a re-key so
       the radio does not thrash.

The evader wraps *any* base :class:`~fhew.generators.Generator`, so evasion is
orthogonal to the hop algorithm: an LFSR, a Markov hopper and a CSPRNG can each be
made reactive. It consumes only feedback and advances only its own state, so it
respects information parity exactly as the inference side does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from fhew.channels import ChannelSet
from fhew.generators import Generator, make_generator

#: emitter_id stamped on the transmitter's own detections (vs decoys, see deception)
REAL_EMITTER_ID = 0


@dataclass(frozen=True)
class JammingFeedback:
    """What a transmitter learns about one slot's jamming, and nothing more.

    ``jammed_mask`` is the set of channels that carried jamming energy this slot,
    sensed by the radio's receiver (``spectrum`` feedback). ``effect_on_self`` is
    the realised denial effect on the channel the radio actually used, in ``[0, 1]``;
    ``denied`` is the thresholded version (a lost packet). A receiver-less radio
    gets only ``effect_on_self`` / ``denied`` and leaves ``jammed_mask`` ``None``.
    """

    slot: int
    effect_on_self: float
    denied: bool
    jammed_mask: Optional[np.ndarray] = None


@dataclass
class EvasionConfig:
    """Aggressiveness knobs for the reactive transmitter.

    The defaults are a moderately aggressive radio: it tolerates a little jamming
    before deflecting, spreads its hops once it is losing half its packets, and
    re-keys only under sustained denial.
    """

    memory: float = 0.6            # EWMA weight for the hot-channel map (recent jam dominates)
    pressure_gain: float = 0.34    # EWMA weight for pressure from self-denial
    avoid_threshold: float = 0.35  # deflect when the proposed channel is at least this hot
    entropy_pressure: float = 0.45  # above this pressure, spread hops (flatten -> low kappa)
    rekey_pressure: float = 0.8    # re-key when pressure crosses this
    cool_after_rekey: float = 0.5  # multiply pressure down after a re-key (anti-thrash)
    temp_lo: float = 0.12          # deflection temperature at zero pressure (near-deterministic)
    temp_hi: float = 1.6           # deflection temperature at full pressure (near-uniform)
    feedback_mode: str = "spectrum"  # "spectrum" (sense the band) | "denial_only" (own hops)
    deny_threshold: float = 0.5    # effect above this counts as a denied hop

    def __post_init__(self) -> None:
        if self.feedback_mode not in ("spectrum", "denial_only"):
            raise ValueError("feedback_mode must be 'spectrum' or 'denial_only'")


#: An :class:`EvasionConfig` whose every trigger threshold exceeds 1, so none ever
#: fires: the wrapped generator hops exactly as its base does. Use it to run the
#: duel with an *oblivious* emitter (the fixed-tape baseline) through the same code
#: path as an evading one.
NO_EVASION_CFG = EvasionConfig(avoid_threshold=9.0, entropy_pressure=9.0, rekey_pressure=9.0)


class AdaptiveEvasionGenerator:
    """A reactive anti-jam transmitter wrapping any base hop generator.

    Drive it one slot at a time with :meth:`next_channel`, passing the previous
    slot's :class:`JammingFeedback` (or ``None`` before any jamming is observed).
    It returns the channel to transmit on *now*, having first folded the feedback
    into its pressure and hot-channel map and, if warranted, re-keyed.

    Parameters
    ----------
    base:
        The underlying hop generator (the radio's nominal hopping).
    channels:
        The channel set; its effective (non-blacklisted) indices bound every hop.
    config:
        :class:`EvasionConfig` aggressiveness knobs.
    rekey_factory:
        Optional ``() -> Generator`` that mints a freshly keyed base generator.
        Provided, a re-key swaps the whole hop pattern; omitted, a re-key falls back
        to a large random phase jump on the existing generator (still breaks a lock,
        but a weaker move). :meth:`from_family` wires this up automatically.
    rng:
        Randomness for evasive sampling and phase jumps (seeded for reproducibility).
    """

    def __init__(self, base: Generator, channels: ChannelSet, *,
                 config: EvasionConfig | None = None,
                 rekey_factory: Callable[[], Generator] | None = None,
                 rng: np.random.Generator | None = None):
        self.base = base
        self.channels = channels
        self.M = channels.size
        self.cfg = config or EvasionConfig()
        self._rekey_factory = rekey_factory
        self._rng = rng or np.random.default_rng()
        self._eff = channels.effective_indices()
        if self._eff.size == 0:
            raise ValueError("channel set has no effective channels to hop on")
        self._eff_mask = np.zeros(self.M, dtype=bool)
        self._eff_mask[self._eff] = True
        self.reset()

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_family(cls, gen_name: str, channels: ChannelSet,
                    rng: np.random.Generator, *, base_params: dict | None = None,
                    config: EvasionConfig | None = None) -> "AdaptiveEvasionGenerator":
        """Build an evader over a registered generator family.

        ``base_params`` are forwarded to :func:`fhew.generators.make_generator`.
        Leave any *seed / state* parameter out of ``base_params`` so each re-key
        draws a fresh key from ``rng`` (that is what makes a re-key actually change
        the hop pattern); pin them only if you deliberately want re-keys to recur to
        the same sequence.
        """
        params = dict(base_params or {})

        def factory() -> Generator:
            return make_generator(gen_name, channels.size, rng=rng, **params)

        return cls(factory(), channels, config=config, rekey_factory=factory, rng=rng)

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        self.base.reset()
        self.h = np.zeros(self.M, dtype=np.float64)   # decaying jammed-channel memory
        self.pressure = 0.0
        self.rekeys = 0
        self.t = 0
        self.last_channel: int | None = None

    # ------------------------------------------------------------------ feedback
    def observe_feedback(self, fb: JammingFeedback) -> None:
        """Fold one slot's jamming feedback into pressure and the hot-channel map."""
        cfg = self.cfg
        if cfg.feedback_mode == "spectrum" and fb.jammed_mask is not None:
            jm = np.asarray(fb.jammed_mask, dtype=np.float64)
            self.h = (1.0 - cfg.memory) * self.h + cfg.memory * jm
        else:  # denial_only: the radio only learns about the channel it sat on
            self.h *= (1.0 - cfg.memory)
            if self.last_channel is not None:
                self.h[self.last_channel] = (
                    (1.0 - cfg.memory) * self.h[self.last_channel]
                    + cfg.memory * float(fb.denied)
                )
        d = float(fb.denied)
        self.pressure = (1.0 - cfg.pressure_gain) * self.pressure + cfg.pressure_gain * d

    # ------------------------------------------------------------------ re-keying
    def _rekey(self) -> None:
        """Change the hop pattern wholesale to shake off an established lock."""
        if self._rekey_factory is not None:
            self.base = self._rekey_factory()
            self.base.reset()
        else:  # no factory: jump the phase of the existing generator instead
            self.base.reset()
            skip = int(self._rng.integers(1, 97))
            for _ in range(skip):
                self.base.next_channel()
        self.rekeys += 1
        self.pressure *= self.cfg.cool_after_rekey

    # ------------------------------------------------------------------ hopping
    def next_channel(self, feedback: JammingFeedback | None = None) -> int:
        """Advance one slot and return the channel to transmit on now.

        ``feedback`` describes the *previous* slot's jamming (``None`` before any is
        seen). The method never reads anything about the current or future jam plan.
        """
        if feedback is not None:
            self.observe_feedback(feedback)
        if self.pressure >= self.cfg.rekey_pressure:
            self._rekey()
        c_base = int(self.base.next_channel())          # the nominal hop (state advances)
        c = self._choose(c_base)
        self.last_channel = c
        self.t += 1
        return c

    def _choose(self, c_base: int) -> int:
        """Pick the hop: the nominal one when safe, else an evasive cold channel."""
        cfg = self.cfg
        base_ok = self._eff_mask[c_base]
        threatened = (
            (base_ok and self.h[c_base] >= cfg.avoid_threshold)
            or self.pressure >= cfg.entropy_pressure
            or not base_ok
        )
        if not threatened:
            return c_base
        # Evade: prefer cold channels; temperature (hence spread) grows with pressure.
        cold = np.where(self._eff_mask, 1.0 - self.h, -np.inf)
        temp = cfg.temp_lo + (cfg.temp_hi - cfg.temp_lo) * float(np.clip(self.pressure, 0.0, 1.0))
        logits = cold / max(temp, 1e-6)
        logits -= np.max(logits[np.isfinite(logits)])
        w = np.where(np.isfinite(logits), np.exp(logits), 0.0)
        s = w.sum()
        if s <= 0:                                       # degenerate: fall back to uniform
            w = self._eff_mask.astype(np.float64)
            s = w.sum()
        return int(self._rng.choice(self.M, p=w / s))

    # ------------------------------------------------------------------ diagnostics
    def state(self) -> dict:
        """A small snapshot for logging / the dashboard (never fed back to inference)."""
        return {"slot": self.t, "pressure": float(self.pressure), "rekeys": int(self.rekeys),
                "hot_mean": float(self.h[self._eff_mask].mean()) if self._eff.size else 0.0}
