"""Dashboard data for the Revision-3 layer: adaptation, jamming and feasibility.

Produces the compact JSON the ``Adapts & jams`` page reads (``OTS_DATA.adaptive``):

    * ``regime``      -- recovery time after a mid-engagement generator switch,
                         for the Fixed-Share mixture with the context flush, with
                         a weights-only reset, and for the cumulative bank alone
                         (thesis Section 6.4 / experiment exp15).
    * ``allocation``  -- multi-tone power-on-target reward: the greedy
                         threshold-aware knapsack vs water-filling across adversary
                         predictability, and reactive/follower jamming vs dwell on
                         an unpredictable hopper (Section 7 / experiment exp16).
    * ``feasibility`` -- the strategy feasibility map with the reactive-follower
                         region (Section 10 / experiment exp17).

The numbers are the same ones the experiments report; this module just runs them
at dashboard-sized seed counts and hands back JSON-serialisable dicts.
"""

from __future__ import annotations

import time

from experiments import exp15_regime_switch as R
from experiments import exp16_allocation as A
from experiments import exp17_feasibility as F


def build(n_regime: int = 10, n_alloc: int = 6, *, verbose: bool = True) -> dict:
    """Compute the three result sets for the dashboard."""
    t0 = time.time()
    if verbose:
        print(f"  adaptive: regime-switch recovery ({n_regime} seeds)")
    regime = R.compute(n_seeds=n_regime)
    if verbose:
        print(f"  adaptive: multi-tone allocation ({n_alloc} seeds)")
    allocation = A.compute(n_seeds=n_alloc, window=120)
    if verbose:
        print("  adaptive: feasibility map (analytic)")
    feasibility = F.compute()
    return {
        "regime": regime,
        "allocation": allocation,
        "feasibility": feasibility,
        "params": {"regime_seeds": n_regime, "alloc_seeds": n_alloc},
        "build_seconds": round(time.time() - t0, 1),
    }
