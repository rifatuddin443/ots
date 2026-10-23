# fhew — Sensor-Aware Hypothesis Bank for FH Generator Identification

A reusable simulation codebase for the conference-paper slice of the
Confidence-Adaptive Multi-Channel EW thesis. The headline capability is
**predicting the next channel** a frequency-hopping emitter will hop to, from a
noisy passive sensor — and doing so **even when the emitter uses an algorithm the
bank has no expert for.** Identifying the known mechanism (in as few observed hops
as possible, and deciding when it is *not* in the bank) is the fast path that
makes that prediction sharp when a known generator is present.

**Primary metric: next-channel top-1 accuracy.** Because the bank is necessarily
limited, prediction rides a Fixed-Share expert mixture that always carries a
zero-shot *universal predictor*; on a brand-new generator it keeps predicting the
next channel where an identification-only bank collapses to the uniform floor (see
`exp14`, and `fhew.predict.ChannelPredictor`).

Everything here is software-only and simulation-only. The code is organised so
that the gym core, the generator registry, the HMM likelihood, and the bank
carry over unchanged into the full thesis (the journal adds experts, probing and
allocation *on top of* this, without editing it).

---

## What this implements (conference paper)

A sensor-aware Bayesian hypothesis bank with anytime stopping:

1. **Sensor-aware likelihoods.** Every hypothesis is scored through the sensor
   model (detection probability `P_d`, false-alarm rate `P_fa`) with an HMM
   forward pass, so missed hops and phantom detections are handled *inside* the
   likelihood instead of corrupting it.
2. **A mandatory none-of-the-above class `H_0`.** Keeps the bank from naming the
   wrong generator with false confidence when the true one is absent (e.g. a
   keyed CSPRNG).
3. **Anytime, multi-hypothesis stopping.** A posterior over the whole bank with
   MSPRT stopping, plus an optional e-process rule whose false-identification
   bound holds whenever the run stops.

---

## Install

```bash
python -m pip install -e .            # core: numpy, scipy, pyyaml
python -m pip install -e ".[plot,dev]" # + matplotlib/pandas (figures) + pytest
# optional: ".[torch]" enables the LSTM baseline drop-in; a numpy classifier is used otherwise
```

Python ≥ 3.10.

## Quick start

```python
import numpy as np
from fhew import ChannelSet, SensorModel
from fhew.config import build_bank, default_bank_specs, in_bank_labels
from fhew.sim import make_episode, run_bank

ch = ChannelSet.of_size(16)
sensor = SensorModel(p_d=0.85, p_fa=0.02)            # lossy passive sensor
specs = default_bank_specs(16)                        # LFSR, weak-seed, LCG, Markov
bank = build_bank(ch, horizon=100, sensor=sensor, specs=specs)

rng = np.random.default_rng(0)
ep = make_episode("weak_seed", ch, sensor, 100, rng,
                  gen_params={"keyspace": 256}).set_target(in_bank_labels(specs))
result = run_bank(bank, ep)
print(result.decision_label, result.tau_stop, result.correct)   # e.g. weak_seed 6 True
```

### Predict the next channel (the headline use)

```python
from fhew import ChannelSet, SensorModel, ChannelPredictor

ch = ChannelSet.of_size(16)
sensor = SensorModel(p_d=0.95, p_fa=0.01)
predictor = ChannelPredictor(ch, sensor)      # carries a zero-shot universal predictor

for obs in stream:                            # the lossy sensor stream, slot by slot
    predictor.observe(obs)
    guess = predictor.predict_top1()          # best guess for the NEXT channel
    shortlist = predictor.predict_topk(3)     # or a few channels to cover
    kappa = predictor.predictability()        # 0 = unpredictable, 1 = pinned down
```

This keeps predicting even for a generator no expert was built for (e.g.
`lagged_fib`, `quadratic_cg`, `sine_map`); against a keyed CSPRNG it correctly
falls back to chance. It never sees the true channel — only `Observation`s.

## Reproduce the paper figures

```bash
python -m fhew.datagen --n-per-class 250         # train the learned baseline + grounding
python -m scripts.run_all                        # all 17 experiments -> results/ + results/figures/
python -m scripts.run_all --quick                # fast smoke run (fewer seeds)
python -m scripts.run_all --only exp1 exp3       # a subset
python -m experiments.exp1_hops_vs_pd --plot-only # re-render a figure from saved JSON (no re-sim)
```

Results land in `results/*.json` (regenerable figures from saved numbers),
figures in `results/figures/*.png`, and a consolidated `results/summary.json`.

---

## Demo dashboard

A presentation-ready, multi-page dashboard that shows the bank working, aimed at a
non-technical audience. Every number on it comes from real `fhew.HypothesisBank`
runs recorded by `python -m dashboard.build`:

| Page | What it shows |
| --- | --- |
| **Story** | the problem in three steps and the headline results |
| **Live demo** | one engagement hop by hop: a labelled hop chart (detections, OTS's next-hop call before each hop arrives), the race between the hypotheses, the lock and the recovered secret next to the true one |
| **Decision analysis** | the same run: confidence per hypothesis over time, the search shrinking from 2,025 candidate settings to one, next-hop calls, and the standard test on the same hops |
| **Challenge** | the audience tries to guess the next hop against OTS |
| **Proof** | Monte-Carlo evidence over 600 runs: wrong answers and dangerous false claims vs the standard test, "unknown" on the CSPRNG, robustness to a worse sensor, textbook Berlekamp-Massey vs OTS under missed hops, sensor-aware vs naive, next-hop accuracy |
| **How it works** | the method in plain words: sense, score, decide, predict; the expert bank; the two safety rules |

```bash
python -m dashboard.build      # record engagements, scoreboard and proofs -> dashboard/data.js (~2 min)
# then open dashboard/index.html in a browser: replays offline, no server or external scripts
python -m dashboard.serve      # http://127.0.0.1:8050 : same pages, plus new engagements run live
python -m dashboard.build --reuse --standalone ots_demo.html   # one self-contained file to share
```

`dashboard/data.js` is committed, so the pages open straight from a fresh clone.
Scenario, sensor quality, recorded run, display toggles and speed live in the
settings drawer of the Live and Analysis pages. Press **G** for the guided demo
(LFSR, weak seed under a contested sensor, Hénon chaos, filtered LFSR, then the
CSPRNG control where the bank answers "unknown" and the standard test names a
generator anyway). Keys: `1`–`6` pages, `Space` play, `←`/`→` step, `R` restart,
`N` next run, `T` sensor mistakes, `S` settings, `F` full screen.

The dashboard replays the exact episodes `experiments.common.episode_for` draws
for each seed (checked in the tests), runs the one-expert-per-family
identification bank of exp12/exp13 at MSPRT `δ = 0.05`, and keeps the bank on
observations only; the truth overlays and secrets are read from the generator
afterwards, for display. The two proof ablations in `dashboard/proof.py` follow
the exp6/exp7 protocols with the same seeds.

## Experiments ↔ paper sections

| Script | Paper | What it shows |
| --- | --- | --- |
| `exp1_hops_vs_pd` | §11.1 (headline) | Hops-to-identify vs `P_d`, per generator |
| `exp2_accuracy_vs_pfa` | §11.2 | Accuracy & false-ID vs `P_fa`, in/out-of-bank |
| `exp3_method_comparison` | §11.3 | Bank+MSPRT vs pairwise SPRT, fixed-length, learned (+ significance) |
| `exp4_none_of_the_above` | §11.4 | `H_0` vs a no-`H_0` SPRT on CSPRNG / out-of-bank |
| `exp5_stale_bank` | §11.5 | Degrade to blind, not to a false algebraic break |
| `exp6_bm_vs_filtered` | §11.6 / §10 | Plain BM vs gap-robust state-filtered recovery vs `P_d` |
| `exp7_sensor_ablation` | §11.7 | Sensor-aware vs naive likelihood |
| `exp8_nexthop` | §8 / §11.8 | Next-hop log-loss & top-k (logged for journal reuse) |
| `exp9_eprocess_bound` | §10 | Empirical false-ID stays under the nominal Ville bound |
| `exp10_calibration` | §11 metrics | Posterior reliability diagram + ECE |
| `exp11_stopping_law` | §10 | E[τ] ∝ log(1/δ) and ∝ 1/`P_d` (detection thinning) |
| `exp12_full_bank_confusion` | thesis §5 | Full expert-bank coverage confusion matrix (every expert fires on its target) |
| `exp13_coverage` | thesis §5 | Large-scale per-type identification over the whole library (7,500 episodes) |
| `exp14_next_channel_prediction` | thesis §4.2 / §12 (headline) | **Next-channel top-1 accuracy**, incl. new algorithms the bank has no expert for; proposed (universal) vs bank-only vs Markov vs uniform |
| `exp15_regime_switch` | thesis §4.2 / §6.4 | Recovery time after a mid-engagement generator switch: Fixed-Share + context flush vs weights-only vs bank-alone |
| `exp16_allocation` | thesis §6.3 / §7 | Multi-tone power allocation: spot↔barrage knapsack vs water-filling, and reactive/follower jamming vs dwell |
| `exp17_feasibility` | thesis §10 | Feasibility map with the reactive-jamming region (dwell vs reaction latency, and window vs hop rate) |

### Headline findings (30 seeds, `M=16`, `T=100`)

- In-bank generators (LFSR, weak-seed, LCG) are identified in **4–13 hops** at
  100% accuracy; hops-to-identify rises as `P_d` falls, matching the
  stopping-time law.
- The bank **never falsely claims a deterministic/algebraic break** on a CSPRNG
  (0%) and almost never on out-of-bank or stale emitters (0–10%), while a
  pairwise SPRT with no `H_0` false-identifies **100%** of the time.
- Plain Berlekamp-Massey collapses the instant `P_d < 1` (any gap); the
  gap-robust state-filtered variant stays at 100% recovery.
- The e-process false-identification rate stays under the nominal `β` line.

### Next-channel prediction, including new algorithms (`exp14`)

- Known generators (LFSR/weak-seed/LCG) are predicted at **100% top-1**; a keyed
  CSPRNG collapses to chance (~1/16), the predictability boundary.
- On **`lagged_fib`**, a new order-2 generator the bank has **no expert for**, the
  universal predictor reaches **0.88 top-1** vs **0.42** for an identification-only
  bank — more than double — and stays far above it across the whole `P_d` sweep.
  This is the "if a new algorithm is used, still predict the next channel" result.

## The full thesis expert bank (beyond the conference slice)

The conference paper (exp1–exp11) uses one hypothesis per family. The **full
thesis expert bank** (thesis §5) implements every expert in the expert table,
each identifying its target across the whole generator library:

| Expert | Target | Implemented as |
| --- | --- | --- |
| Hypothesis bank | known candidates | `HypothesisBank` (posterior + MSPRT / e-process) |
| Berlekamp-Massey | LFSR | `AlgebraicLFSRHypothesis` (collapsing, needs ~2L clean *consecutive*) |
| State-filtered BM | LFSR through gaps | `DeterministicHypothesis` (particle filter over register states) |
| LCG cryptanalysis | linear congruential | `LCGCryptanalysisHypothesis` (algebraic `(a,c)` recovery on `lcg_full`) |
| Seed search | weak-seed PRNG | `DeterministicHypothesis` over the keyspace |
| Correlation attack | filtered LFSR | `DeterministicHypothesis` enumerating a known filter family's state |
| Parametric chaos ID | logistic / tent / Hénon | `MixtureHMMHypothesis` over induced symbolic-dynamics transitions |
| Phase-space predictor | unknown chaotic | `PhaseSpacePredictorHypothesis` (delay embedding + kNN) |
| Markov / n-gram | empirical regularity | `MarkovHypothesis` |
| Zero-shot neural | any learnable, gap-robust | `ContextMixingHypothesis` (training-free variable-order mixture) |
| Behaviour model | reactive / learned policy | `BehaviourHypothesis` (recency-weighted; reduced TD-MPC2 stand-in) |
| Prior-weighted barrage | none assumed | `BarrageHypothesis` (the zero-hop floor) |

Build it with `fhew.config.identification_bank_specs(M)` (one best expert per
family) or `full_bank_specs(M)` (all experts, for the coverage matrix).

### Full-bank findings (exp12 / exp13, 7,500-episode sweep)

- Every family is identified: at `P_d=0.95` the correct expert wins on **100%**
  of episodes for all 9 identifiable families; CSPRNG routes to `H_0`.
- Across 10 families × 5 `P_d` × 3 `P_fa` × 50 seeds, per-type accuracy stays
  **≥0.92** (LFSR/filtered-LFSR/LCG/weak-seed/chaos ≈ 1.0), with **hard
  (algebraic) false-ID ≤0.05 everywhere, mostly ≤0.01**.
- `lcg_full` degrades gracefully at low `P_d` (its period-16 cycle at `M=16` is
  legitimately caught by the Markov expert — a soft, non-dangerous miss).
- The chaotic maps (logistic, tent, Hénon) are all identified by the single
  parametric-chaos-ID expert, which recovers the map parameter.

---

## Confidence-adaptive mixture, regime monitor and allocation (thesis journal)

On top of the identification bank sit the pieces that let the system **track a
changing adversary** and **act** on a prediction. They are additive — the gym
core and the bank are untouched, and the bank enters the mixture through its
existing `Expert` interface.

| Component | Module | What it adds |
| --- | --- | --- |
| Fixed-Share mixture | `fhew/mixture.py` | Weights experts (bank + universal + Markov + barrage) by *recent* log-loss with a share floor `α>0`, so a dead expert recovers after a switch |
| Regime monitor | `fhew/regime.py` | CUSUM / BOCPD / conformal-e change detector on the loss stream; on an alarm **resets the weights AND flushes the universal predictor's context** (exponential forgetting) — closing the gap where a weights-only reset left stale context mispredicting |
| Dwell feature | `fhew/dwell.py` | Online dwell estimate; a predictability contribution and the reactive-jamming trigger |
| Multi-tone allocator | `fhew/allocator.py` | Threshold-aware greedy knapsack over the predictive (spot↔barrage from one knob `λ`, CDT-calibrated), reactive/follower mode, water-filling baselines |
| Feasibility map | `fhew/feasibility.py` | Adds the reactive-follower region (dwell > reaction latency, regardless of κ) |
| Scenarios | `fhew/scenarios.py` | Regime-switch and long-dwell episode builders + the entropy-max adversary |

```python
from fhew import ChannelSet, SensorModel
from fhew.mixture import build_default_mixture
from fhew.regime import RegimeController, make_detector, recovery_time
from fhew.scenarios import make_switch_episode
import numpy as np

ch, sensor = ChannelSet.of_size(16), SensorModel(0.97, 0.01)
ep = make_switch_episode([("markov", {"alpha": 0.03}), ("markov", {"alpha": 0.03})],
                         ch, sensor, 240, np.random.default_rng(0), switch_at=120)
mix = build_default_mixture(ch, 240, sensor)
ctrl = RegimeController(mix, make_detector("cusum"), forget_context=True)  # flush on alarm
for o in ep.observations:
    ctrl.update(o)                      # drives mixture + monitor; flushes context on a switch
```

### Findings (exp15 / exp16 / exp17)

- **Regime switch (exp15).** On a near-deterministic Markov hopper that switches
  mid-engagement, resetting the weights **and** flushing the context recovers the
  top-1 prediction **1.2–2.3× faster** than resetting the weights alone; the
  cumulative bank is slowest. The CUSUM monitor fires on ≈ 78 % of switches with
  ≈ 0.12 false alarms per episode.
- **Allocation (exp16).** The knapsack tracks the best allocation across the
  predictability range and degrades to a barrage floor (reward ≈ 0.11) where
  water-filling collapses to 0. Against an *uncrackable* CSPRNG, reactive/follower
  jamming lifts reward from 0.11 (dwell 1) to 0.82 (dwell 8) — unpredictable is
  not un-jammable.
- **Feasibility (exp17).** Against a `κ=0` hopper, ≈ 71 % of the dwell/latency
  plane is still jammable by detect-and-follow.

---

## Package layout

Each concern is a single flat module in `fhew/` (the former subpackages were
merged into one file each; public import paths are unchanged — `from
fhew.generators import make_generator` still works).

```
fhew/
  channels.py        channel set C
  sensor.py          detection-level sensor + frozen observation schema v1
  generators.py      LFSR, filtered-LFSR, LCG, full-state LCG, weak-seed, Markov,
                     chaotic maps (logistic/tent/Hénon), CSPRNG, and the held-out
                     "new algorithm" families (lagged_fib/quadratic_cg/sine_map)
  recovery.py        Berlekamp-Massey + bit/channel expansion
  likelihood.py      reduced emission (sensor-aware / naive) + forward filter
  experts.py         Expert interface, the full thesis expert bank, H_0, the bank,
                     and the Fixed-Share mixture (next-channel predictor)
  predict.py         ChannelPredictor: the next-channel prediction front end
  stopping.py        MSPRT + e-process
  baselines.py       pairwise SPRT, fixed-length LRT, learned classifier (+ LSTM)
  sim.py             episode + end-to-end slot loop (paper Algorithm 1)
  metrics.py         identification / calibration / next-hop + CIs + significance
  config.py          dataclasses, YAML, bank builders (conference / id / full)
  datagen.py         training data + parameter grounding provenance
  mixture.py         Fixed-Share expert mixture (+ Expert adapter) — thesis §4.2
  regime.py          change detectors (CUSUM/BOCPD/conformal-e) + controller — §6.4
  dwell.py           dwell estimation: predictability feature + reactive trigger
  allocator.py       threshold-aware knapsack + reactive/follower + CDT knob — §7
  feasibility.py     strategy feasibility map incl. the reactive region — §10
  scenarios.py       regime-switch / long-dwell episode builders
configs/             conference.yaml, quick.yaml
experiments/         exp1..exp17 + shared style/harness
scripts/run_all.py   run everything
dashboard/           demo dashboard: recorder, proofs, build, live server, pages
tests/test_fhew.py                67 tests incl. the information-parity audit
tests/test_mixture_allocation.py  29 tests for the mixture / regime / allocator
```

## Design rules that keep it reuse-ready (paper §12)

1. The bank has the same interface as every expert: `update(obs)`, `predict()`
   (next-hop distribution), `posterior()`, and a `declared` flag — so the thesis
   Fixed-Share mixture consumes it as one expert unchanged.
2. The next-hop distribution and its log-loss are logged in every run.
3. The sensor model and observation schema are frozen as **v1**, with the
   optional fields (`power`, `bearing`, `emitter_id`) the journal needs already
   present.
4. `H_0` is **pluggable** (`uniform` / `occupancy` / `markov`), so the journal
   swaps in a learned open-set score without touching the bank.
5. Generators live in a **registry**; everything runs from config with fixed
   seeds.
6. **Information parity** is enforced structurally (the bank sees only
   `Observation` objects and its own recce ranges) and audited in the tests.

## Information parity

No likelihood uses anything the modelled sensor cannot deliver — never the true
seed, register length, or generator family. The bank is built from the
reconnaissance *ranges*; a secret outside those ranges is provably not
identifiable (see `tests/test_information_parity.py`). Recovery numbers obtained
with illegitimate side-information would be void, and this is the first place a
careful reviewer probes.

## Documentation

- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — running implementation log (what was built, where, and why); **append a dated entry for every change**.
- [`docs/methodology_and_results.md`](docs/methodology_and_results.md) — full methodology and per-experiment results with figures.

## Grounding & ethics

Parameter ranges are meant to be calibrated from public FHSS/drone-RF captures,
not trained on. `fhew/datagen.py::GROUNDING` labels every parameter *calibrated*
or *assumed*; replace the placeholder ranges with measured values before quoting
numbers as calibrated. This is offensive-EW-adjacent research: keep it generic
and simulation-only, and clear publication with your supervisor and institution.

## Tests

```bash
python -m pytest        # 96 tests, ~25s
```
