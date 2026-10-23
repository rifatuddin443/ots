# Implementation log

A running record of what has been built on top of the conference identification
bank, so the repo always documents itself. **Append a dated entry here for every
future change** (newest first). Deep detail lives in the methodology write-up and
the README; this file is the index to it.

- Methodology & results: [`docs/methodology_and_results.md`](methodology_and_results.md)
- Package overview & quick starts: [`../README.md`](../README.md)
- Figures: [`docs/figures/`](figures/)

---

## 2026-10-23 — Confidence-adaptive layer: mixture, regime monitor, allocation, dashboard

Built the Revision-3 layer on top of the conference identification bank. All of
it is **additive**: the gym core and the bank are untouched, and the bank enters
the mixture through its existing `Expert` interface.

### Why
Two behaviours the bank alone could not provide:
1. **Re-point without manual switching when the adversary changes generator
   mid-engagement.** The bank weights by *cumulative* evidence, so it stays stuck
   on the old winner. The gap flagged in review: a regime monitor that resets the
   mixture weights is not enough — the universal predictor's context is still full
   of the old generator's patterns and keeps mispredicting until it is forgotten.
2. **Act on a prediction.** A single next-channel prediction is not actionable;
   the action is a multi-tone power vector, and a long dwell is a separate weapon
   (reactive/follow jamming) that works even against an unpredictable radio.

### What was added (modules)
| Module | Purpose | Thesis |
| --- | --- | --- |
| `fhew/mixture.py` | Fixed-Share expert mixture (bank + universal context-mixing predictor + online Markov + barrage floor); share floor `>0` lets a dead expert recover | §4.2 |
| `fhew/regime.py` | CUSUM / BOCPD / conformal-e change detectors on the loss stream + controller that, on alarm, **resets weights AND flushes the universal predictor's context** (`forget(retain)`); `recovery_time` metric | §6.4 |
| `fhew/dwell.py` | Online dwell estimation: predictability feature + the reactive-jamming trigger | §7 |
| `fhew/allocator.py` | Threshold-aware greedy knapsack over the whole predictive (spot↔barrage from one CDT-calibrated knob `λ`), reactive/follower mode, water-filling baselines | §6.3 / §7 |
| `fhew/feasibility.py` | Strategy feasibility map; adds the reactive-follower region (dwell > reaction latency, regardless of κ) | §10 |
| `fhew/scenarios.py` | Regime-switch and long-dwell episode builders | §6.4 / §11 |
| `fhew/generators.py` | New `entropy_max` adversary (spreads channels; with short dwell defeats both weapons — the true predictability boundary) | §11 |
| `fhew/experts.py` | `forget(retain)` added to every stateful expert (context-mixing tables + window, Markov/occupancy counts, phase-space history, bank cumulative evidence) | §6.4 |

### Experiments (wired into `scripts/run_all`)
- `experiments/exp15_regime_switch.py` — recovery time after a switch: context flush vs weights-only vs bank-alone. **Flush recovers 1.2–2.3× faster**; monitor fires on ~78 % of switches, ~0.12 false alarms/episode. Figure: `docs/figures/exp15_regime_switch.png`.
- `experiments/exp16_allocation.py` — knapsack vs water-filling across predictability (knapsack holds a barrage floor where water-filling collapses to 0), and reactive jamming vs dwell on an uncrackable CSPRNG (reward 0.11→0.82 as dwell grows). Figure: `docs/figures/exp16_allocation.png`.
- `experiments/exp17_feasibility.py` — feasibility map with the reactive region (~71 % of the dwell/latency plane on a κ=0 hopper) + the classic three-band window/hop-rate map. Figure: `docs/figures/exp17_feasibility.png`.

### Dashboard
- New page **"Adapts & jams"** (`dashboard/views/adaptive.js`) rendering the three results above from real runs.
- Data producer `dashboard/adaptive.py`, wired into `dashboard/build.py` (`--regime-seeds`, `--alloc-seeds`; `--quick` shrinks them). Rebuild with `python -m dashboard.build` (or `python -m dashboard.build --standalone ots_dashboard.html`).

### Tests
- `tests/test_mixture_allocation.py` — 29 tests covering context forgetting (incl. a guard that an alarm actually flushes the context model), the share floor, the detectors, dwell, the allocator (spot/barrage/reactive/CDT) and the feasibility map.

### Notes for future changes
- Reconciled against the `next-channel prediction` + dashboard work already on the branch: these experiments are numbered **exp15/16/17** (exp14 is `next_channel_prediction`). Keep new experiments numbered sequentially and add a row to the README experiments table.
- Detector defaults (`exp15_regime_switch._DETECTOR`) were tuned for a low false-alarm rate; when the monitor does not fire, the flush config simply equals weights-only, so the context flush never hurts.
