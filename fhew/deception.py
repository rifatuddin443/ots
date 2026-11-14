"""Counter-reconnaissance *deception*: corrupt what the OTS sensor sees.

The inference side of this package is deliberately honest about its sensor -- it
handles missed hops and random false alarms *inside* the likelihood. But a real
adversary does not emit only random false alarms: it emits **structured** ones. A
cheap second transmitter, a towed decoy, or a handful of repeaters can lay down
detections that are *time-coherent with a known hop algorithm*, so to a passive
collector they do not look like noise -- they look like a second, perfectly
ordinary emitter. That is a deception attack on the whole OTS pipeline, and it is
mounted here at the one interface the OTS actually consumes: the stream of
:class:`~fhew.sensor.Observation` objects.

Two mechanisms, both at the detection level so nothing downstream has to change:

    * **Coherent mimicry (counter-ID).** Run a *decoy* generator -- an in-bank
      family such as an LFSR -- in lockstep with the real emitter and add its
      channel to every slot. The OTS bank now sees an emitter that matches a known
      algorithm every slot, so the matching hypothesis's likelihood climbs and the
      bank is pulled toward naming the decoy. This is the attack the mandatory
      none-of-the-above class ``H_0`` exists to blunt: a bank *with* ``H_0`` can
      still answer "unknown" rather than commit to the planted break, while a
      pairwise test with no ``H_0`` is forced to name the decoy (see
      ``experiments/exp27_deception``). It is the active-adversary counterpart of
      the passive ``exp4`` none-of-the-above result.

    * **Decoy tones (counter-jam).** The same planted detections also widen the
      predictor's belief and so the conformal / knapsack **jam plan**: tones are
      spent covering channels the real emitter never uses, and the realised hop is
      denied less often per unit power. Optional unstructured ``chaff`` adds plain
      extra false alarms on top.

Everything records which detections were planted (``emitter_id = DECOY_ID``) and
returns the decoy channels, so an experiment can score "tones wasted on decoys" and
"false-ID induced" against the hidden truth -- the truth is used only for scoring,
never shown to the OTS, exactly as elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fhew.channels import ChannelSet
from fhew.generators import Generator, make_generator
from fhew.sensor import Detection, Observation, SensorModel
from fhew.sim import Episode

#: emitter_id stamped on planted (decoy) detections, so scoring can tell them apart
DECOY_ID = 99


@dataclass
class DeceptionConfig:
    """How much deception to lay down per slot.

    ``n_decoys`` coherent decoy streams are run in lockstep (the first follows
    ``decoy_gen``; any extras follow ``extra_gens``), each planting one detection per
    slot. ``chaff_rate`` adds independent unstructured false alarms per channel per
    slot on top (plain noise, to thicken the picture). ``p_present`` is the
    probability the decoy is actually radiating on a given slot (a decoy that blinks
    is harder to gate out than one that is always on).
    """

    n_decoys: int = 1
    chaff_rate: float = 0.0
    p_present: float = 1.0

    def __post_init__(self) -> None:
        if self.n_decoys < 1:
            raise ValueError("n_decoys must be >= 1")
        if not 0.0 <= self.chaff_rate < 1.0:
            raise ValueError("chaff_rate must be in [0, 1)")
        if not 0.0 < self.p_present <= 1.0:
            raise ValueError("p_present must be in (0, 1]")


class DeceptionEmitter:
    """Overlay time-coherent decoy detections onto a real observation stream.

    Parameters
    ----------
    decoy_gen:
        The generator the primary decoy follows (an in-bank family makes the
        strongest counter-ID attack; any family makes a counter-jam attack).
    channels:
        The channel set.
    config:
        :class:`DeceptionConfig`.
    extra_gens:
        Optional further decoy generators for ``n_decoys > 1`` (one per extra
        stream). If fewer are supplied than requested, the remainder are drawn as
        independent copies of ``decoy_gen``'s family via ``decoy_family``.
    decoy_family / decoy_params:
        Used to mint extra decoy generators when ``extra_gens`` is short.
    """

    def __init__(self, decoy_gen: Generator, channels: ChannelSet,
                 config: DeceptionConfig | None = None, *,
                 extra_gens: list[Generator] | None = None,
                 decoy_family: str | None = None, decoy_params: dict | None = None,
                 rng: np.random.Generator | None = None):
        self.channels = channels
        self.M = channels.size
        self.cfg = config or DeceptionConfig()
        self._rng = rng or np.random.default_rng()
        gens = [decoy_gen] + list(extra_gens or [])
        while len(gens) < self.cfg.n_decoys:
            if decoy_family is None:
                raise ValueError("need decoy_family to mint extra decoy streams")
            gens.append(make_generator(decoy_family, self.M, rng=self._rng,
                                       **dict(decoy_params or {})))
        self.decoy_gens = gens[: self.cfg.n_decoys]

    def reset(self) -> None:
        for g in self.decoy_gens:
            g.reset()

    # ------------------------------------------------------------------ one slot
    def corrupt(self, obs: Observation, rng: np.random.Generator | None = None
                ) -> tuple[Observation, tuple[int, ...]]:
        """Return ``(corrupted_observation, decoy_channels)`` for one slot.

        The decoy generators advance exactly once per call, so calling
        :meth:`corrupt` in slot order keeps every decoy stream coherent. The real
        detections are preserved verbatim; decoys are appended with
        ``emitter_id = DECOY_ID``.
        """
        rng = rng or self._rng
        dets = list(obs.detections)
        decoy_channels: list[int] = []
        for g in self.decoy_gens:
            d = int(g.next_channel())                 # advance the decoy's own clock
            if rng.random() < self.cfg.p_present:
                dets.append(Detection(slot=obs.slot, channel=d, emitter_id=DECOY_ID))
                decoy_channels.append(d)
        if self.cfg.chaff_rate > 0.0:
            mask = rng.random(self.M) < self.cfg.chaff_rate
            for ch in np.nonzero(mask)[0]:
                dets.append(Detection(slot=obs.slot, channel=int(ch), emitter_id=DECOY_ID))
                decoy_channels.append(int(ch))
        return (Observation(slot=obs.slot, detections=tuple(dets), n_channels=self.M),
                tuple(decoy_channels))

    # ------------------------------------------------------------------ one stream
    def deceive_stream(self, observations, rng: np.random.Generator | None = None
                       ) -> tuple[list[Observation], list[tuple[int, ...]]]:
        """Corrupt a whole observation list; returns the new stream + decoy channels."""
        self.reset()
        rng = rng or self._rng
        new_obs, decoys = [], []
        for o in observations:
            co, dc = self.corrupt(o, rng)
            new_obs.append(co)
            decoys.append(dc)
        return new_obs, decoys


def make_deception_episode(real: Episode, decoy_family: str, *,
                           config: DeceptionConfig | None = None,
                           decoy_params: dict | None = None,
                           sensor: SensorModel | None = None,
                           rng: np.random.Generator | None = None) -> Episode:
    """Overlay a coherent decoy onto an existing episode's observation stream.

    The returned :class:`~fhew.sim.Episode` keeps the real hidden ``true_channels``
    and ``true_label`` (so scoring is unchanged) but its ``observations`` are the
    deceived stream, and ``gen_params`` carries ``decoy_family`` and the per-slot
    ``decoy_channels`` so an experiment can measure wasted tones and induced
    false-ID. The OTS still sees only observations, as always.
    """
    # deterministic from the episode seed (no string hashing, which is per-process)
    rng = rng or np.random.default_rng((int(real.seed) + 1) * 2654435761 % (2 ** 32))
    decoy_gen = make_generator(decoy_family, real.channels.size, rng=rng,
                               **dict(decoy_params or {}))
    dec = DeceptionEmitter(decoy_gen, real.channels, config, decoy_family=decoy_family,
                           decoy_params=decoy_params, rng=rng)
    new_obs, decoys = dec.deceive_stream(real.observations, rng)
    params = dict(real.gen_params)
    params["deception"] = {"decoy_family": decoy_family,
                           "decoy_channels": [list(d) for d in decoys],
                           "config": vars(config) if config else vars(DeceptionConfig())}
    ep = Episode(true_label=real.true_label, true_channels=real.true_channels,
                 observations=new_obs, window=real.window, channels=real.channels,
                 p_d=real.p_d, p_fa=real.p_fa, seed=real.seed, gen_params=params)
    ep.target_label = real.target_label
    return ep
