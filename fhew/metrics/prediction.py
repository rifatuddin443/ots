"""Next-hop prediction metrics (logged for journal reuse, paper Section 8)."""

from __future__ import annotations

import numpy as np

from fhew.sim.engine import EpisodeResult


def nexthop_summary(results: list[EpisodeResult]) -> dict:
    """Aggregate next-hop log-loss and top-k hit rates over episodes."""
    ll = np.array([r.nexthop_logloss for r in results], dtype=np.float64)
    t1 = np.array([r.nexthop_top1 for r in results], dtype=np.float64)
    t3 = np.array([r.nexthop_top3 for r in results], dtype=np.float64)
    ll = ll[~np.isnan(ll)]
    t1 = t1[~np.isnan(t1)]
    t3 = t3[~np.isnan(t3)]
    return {
        "logloss": float(ll.mean()) if ll.size else float("nan"),
        "top1": float(t1.mean()) if t1.size else float("nan"),
        "top3": float(t3.mean()) if t3.size else float("nan"),
        "n": int(ll.size),
    }
