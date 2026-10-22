"""The FH-EW gym core: episodes and the end-to-end slot loop.

This is the "gym core: sensor model and slot loop" row of the reuse table
(paper Section 12): unchanged, the base of the full thesis system. The journal
adds moving emitters, look-through blanking and probing on top of exactly this
loop.
"""

from fhew.sim.episode import Episode, make_episode
from fhew.sim.engine import (
    EpisodeResult,
    decide_outcome,
    run_bank,
    run_fixed_length,
    run_learned,
    run_sprt,
)

__all__ = [
    "Episode",
    "make_episode",
    "EpisodeResult",
    "run_bank",
    "run_sprt",
    "run_fixed_length",
    "run_learned",
    "decide_outcome",
]
