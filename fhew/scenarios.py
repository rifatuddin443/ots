"""Non-stationary and long-dwell episode builders (thesis Sections 6.4 / 11).

The conference :func:`fhew.sim.make_episode` draws one generator and senses it.
The thesis needs two stressors on top of that, without touching the gym core:

    * **Regime switches** (``make_switch_episode``) -- the adversary changes its
      generator family mid-engagement. This is the "regime-switcher" red-team
      adversary and the input to the recovery-time metric: the mixture must
      re-point itself onto the new generator within a few hops (Section 6.4).

    * **Long dwell** (``make_dwell_episode`` / ``with_dwell``) -- the emitter
      stays on each drawn channel for ``dwell`` slots. A long dwell is both a
      predictability feature (it raises kappa) and, more importantly, the regime
      where reactive / follower jamming works regardless of kappa: detect the
      hop, then jam it while it is still sitting there, whenever the reaction
      latency is shorter than the dwell (Section 7 / the feasibility map).

Everything returns a plain :class:`fhew.sim.Episode`, so the mixture, the
regime controller and the allocator consume these exactly like any episode.
``switch_at`` / ``dwell`` are recorded in ``gen_params`` so downstream metrics
know where the truth changed and how long each channel was held.
"""

from __future__ import annotations

import numpy as np

from fhew.channels import ChannelSet
from fhew.generators import Generator, make_generator
from fhew.sensor import SensorModel
from fhew.sim import Episode


def with_dwell(true_channels: np.ndarray, dwell: int, window: int | None = None) -> np.ndarray:
    """Hold each channel for ``dwell`` consecutive slots (then truncate)."""
    dwell = max(1, int(dwell))
    held = np.repeat(np.asarray(true_channels, dtype=np.int64), dwell)
    if window is not None:
        if held.size < window:                 # draw was too short; tile the tail
            reps = int(np.ceil(window / held.size))
            held = np.tile(held, reps)
        held = held[:window]
    return held


def make_dwell_episode(gen_name: str, channels: ChannelSet, sensor: SensorModel,
                       window: int, rng: np.random.Generator, *, dwell: int = 1,
                       gen_params: dict | None = None, seed: int | None = None,
                       label: str | None = None) -> Episode:
    """An episode whose emitter dwells ``dwell`` slots per channel."""
    gen_params = dict(gen_params or {})
    g: Generator = make_generator(gen_name, channels.size, rng=rng, **gen_params)
    # draw enough underlying hops to fill the window after dwelling
    n_draw = int(np.ceil(window / max(1, dwell))) + 1
    base = g.sequence(n_draw)
    true = with_dwell(base, dwell, window)
    obs = sensor.observe(true, channels, rng)
    params = {**gen_params, "dwell": int(dwell)}
    return Episode(true_label=label or gen_name, true_channels=true, observations=obs,
                   window=window, channels=channels, p_d=sensor.p_d, p_fa=sensor.p_fa,
                   seed=int(seed) if seed is not None else -1, gen_params=params)


def make_switch_episode(segments: list[tuple[str, dict]], channels: ChannelSet,
                        sensor: SensorModel, window: int, rng: np.random.Generator, *,
                        switch_at: int | None = None, dwell: int = 1,
                        seed: int | None = None) -> Episode:
    """Concatenate several generators into one non-stationary episode.

    ``segments`` is a list of ``(gen_name, gen_params)``. With two segments and
    an explicit ``switch_at`` the episode is the first generator for
    ``switch_at`` slots and the second thereafter; with more segments the window
    is split evenly. The switch point(s) are recorded in ``gen_params`` so the
    recovery-time metric knows where the truth changed.
    """
    if not segments:
        raise ValueError("need at least one segment")
    n = len(segments)
    if n == 1:
        switch_at = window
    if switch_at is None:
        bounds = [round(window * i / n) for i in range(n + 1)]
    elif n == 2:
        bounds = [0, int(switch_at), window]
    else:
        raise ValueError("pass switch_at only for the two-segment case")

    pieces: list[np.ndarray] = []
    labels: list[str] = []
    for (name, params), lo, hi in zip(segments, bounds[:-1], bounds[1:]):
        length = max(0, hi - lo)
        if length == 0:
            labels.append(name)
            continue
        g = make_generator(name, channels.size, rng=rng, **dict(params or {}))
        n_draw = int(np.ceil(length / max(1, dwell))) + 1
        seg = with_dwell(g.sequence(n_draw), dwell, length)
        pieces.append(seg)
        labels.append(name)
    true = np.concatenate(pieces)[:window] if pieces else np.zeros(window, dtype=np.int64)
    if true.size < window:                       # pad defensively
        true = np.concatenate([true, np.full(window - true.size, true[-1] if true.size else 0)])
    obs = sensor.observe(true, channels, rng)
    params = {"segments": [s[0] for s in segments],
              "switch_at": int(switch_at) if (n == 2 and switch_at is not None) else bounds,
              "dwell": int(dwell)}
    return Episode(true_label="+".join(labels), true_channels=true, observations=obs,
                   window=window, channels=channels, p_d=sensor.p_d, p_fa=sensor.p_fa,
                   seed=int(seed) if seed is not None else -1, gen_params=params)
