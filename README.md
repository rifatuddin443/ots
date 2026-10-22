# fhew — Sensor-Aware Hypothesis Bank for FH Generator Identification

A reusable simulation codebase for the conference-paper slice of the
Confidence-Adaptive Multi-Channel EW thesis: **identify which known
hop-generation mechanism a frequency-hopping emitter is using, from a noisy
passive sensor, in as few observed hops as possible — and decide when the
generator is not in the bank at all.**

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

## Reproduce the paper figures

```bash
python -m fhew.datagen --n-per-class 250         # train the learned baseline + grounding
python -m scripts.run_all                        # all 11 experiments -> results/ + results/figures/
python -m scripts.run_all --quick                # fast smoke run (fewer seeds)
python -m scripts.run_all --only exp1 exp3       # a subset
python -m experiments.exp1_hops_vs_pd --plot-only # re-render a figure from saved JSON (no re-sim)
```

Results land in `results/*.json` (regenerable figures from saved numbers),
figures in `results/figures/*.png`, and a consolidated `results/summary.json`.

---

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

---

## Package layout

Each concern is a single flat module in `fhew/` (the former subpackages were
merged into one file each; public import paths are unchanged — `from
fhew.generators import make_generator` still works).

```
fhew/
  channels.py        channel set C
  sensor.py          detection-level sensor + frozen observation schema v1
  generators.py      LFSR, filtered-LFSR, LCG, weak-seed, Markov, CSPRNG (registry)
  recovery.py        Berlekamp-Massey + bit/channel expansion
  likelihood.py      reduced emission (sensor-aware / naive) + forward filter
  experts.py         Expert interface, hypotheses, H_0, the bank
  stopping.py        MSPRT + e-process
  baselines.py       pairwise SPRT, fixed-length LRT, learned classifier (+ LSTM)
  sim.py             episode + end-to-end slot loop (paper Algorithm 1)
  metrics.py         identification / calibration / next-hop + CIs + significance
  config.py          dataclasses, YAML, bank builders
  datagen.py         training data + parameter grounding provenance
configs/             conference.yaml, quick.yaml
experiments/         exp1..exp11 + shared style/harness
scripts/run_all.py   run everything
tests/test_fhew.py   46 tests incl. the information-parity audit
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

## Grounding & ethics

Parameter ranges are meant to be calibrated from public FHSS/drone-RF captures,
not trained on. `fhew/datagen.py::GROUNDING` labels every parameter *calibrated*
or *assumed*; replace the placeholder ranges with measured values before quoting
numbers as calibrated. This is offensive-EW-adjacent research: keep it generic
and simulation-only, and clear publication with your supervisor and institution.

## Tests

```bash
python -m pytest        # 46 tests, ~4s
```
