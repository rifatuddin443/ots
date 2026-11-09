# Experiment review: what has been run, what is missing, what to tighten

*Written 2026-11-04 against the working tree on `claude/vibrant-dirac-22pndv`.
Numbers come from `docs/methodology_and_results.md`; only `exp19` has a committed
result JSON (`results/` is git-ignored), so every other number is as recorded in
the write-up, not re-run for this review.*

---

## 1. The 19 experiments, grouped by the claim each one supports

All experiments are simulation-only: `M = 16` channels, one emitter, a slot-synchronous
detection-level sensor (`P_d`, `P_fa`), fixed seeds, and figures re-rendered from JSON.

### A. Identification: speed and the stopping law

| Exp | Question | Protocol | Key result |
| --- | --- | --- | --- |
| **exp1** hops vs `P_d` | How many hops to name an in-bank generator, and how does that scale with detection loss? | Conference bank (LFSR, weak-seed, LCG, Markov, `H_0`), MSPRT `δ=0.05`, `P_fa=0.02`, `P_d ∈ {1.0…0.5}`, 30 seeds | LFSR/weak-seed/LCG at 100 % accuracy in 4.2→10.9–13.3 hops; Markov 46→74 hops |
| **exp11** stopping law | Does `E[τ]` follow `log(1/δ) / (P_d · KL)`? | Markov source (finite KL), 40 seeds, sweep `δ` and `P_d` | Linear in `log(1/δ)` (R²=0.984) and in `1/P_d` (R²=0.997) |

### B. Robustness to the sensor

| Exp | Question | Protocol | Key result |
| --- | --- | --- | --- |
| **exp7** sensor ablation | Does putting `P_d`/`P_fa` inside the likelihood matter? | Same bank, sensor-aware vs naive clean-match emission, `P_fa=0.05`, 30 seeds/gen | At `P_d=0.5`: 0.98 vs 0.60 accuracy, 15.3 vs 21.1 hops |
| **exp6** plain vs state-filtered BM | Does LFSR recovery survive missed hops? | LFSR `L=7`, `P_fa=0`, 40 seeds | Plain BM 1.00 at `P_d=1`, 0.00 for any `P_d<1`; state enumeration 1.00 everywhere |
| **exp2** accuracy vs `P_fa` | Do phantom detections cause false IDs? | `P_d=0.8`, `P_fa ∈ {0…0.1}`, 30 seeds | In-bank 1.00 / false-ID 0.00 throughout; CSPRNG→unknown 0.63–0.87 |

### C. Safety: not naming the wrong generator

| Exp | Question | Protocol | Key result |
| --- | --- | --- | --- |
| **exp4** none-of-the-above | Does `H_0` stop false claims on unbreakable / unmodelled sources? | CSPRNG and filtered-LFSR vs conference bank, 40 seeds | Bank: hard false-ID 0.00, CSPRNG→unknown 87 %; no-`H_0` SPRT false-IDs 100 % |
| **exp5** stale bank | What if the secret is outside the searched range? | LCG seed, weak-seed keyspace 2¹⁶, LFSR `L=10`, all outside recce, 40 seeds | Hard false-ID 0.00 / 0.03 / 0.03; SPRT 100 % |
| **exp3** method comparison | Bank vs standard sequential/fixed tests and a learned classifier | Mixed in/out-of-bank set, `P_d=0.8`, `P_fa=0.02`, 30 seeds, paired bootstrap + McNemar | Bank 0.79 acc / 0.19 false-ID; SPRT 0.66/0.34; fixed-length 0.67/0.33; learned 0.38/0.11; all p<1e-3 |
| **exp12** full-bank confusion | Does every expert fire on its own target? | 13-expert thesis bank, `P_d=0.95`, `P_fa=0.01`, 40 seeds | Clean diagonal; CSPRNG 0.925 → `H_0`; `lcg_full` 0.975 |
| **exp13** coverage at scale | Does safety hold across the whole operating grid? | 10 families × 5 `P_d` × 3 `P_fa` × 50 seeds = 7,500 episodes | Accuracy ≥ 0.92 except `lcg_full` (0.50, soft misses); hard false-ID ≤ 0.05 everywhere |

### D. Statistical guarantees and calibration

| Exp | Question | Protocol | Key result |
| --- | --- | --- | --- |
| **exp9** e-process bound | Does empirical false-ID stay under nominal `β`? | In-bank truths, `P_d=0.7`, `P_fa=0.05`, ≥100 seeds/family | All points on/under `y=β` — **stale**: run with the old mixture denominator |
| **exp10** calibration | Is the declared posterior trustworthy? | 80 seeds × 6 families, `P_d=0.75`, `P_fa=0.03` | ECE = 0.154 over 180 declarations |

### E. Next-channel prediction (the headline metric)

| Exp | Question | Protocol | Key result |
| --- | --- | --- | --- |
| **exp8** bank next-hop | How good is the identification bank's own next-hop distribution? | Full window, `P_d=0.85`, `P_fa=0.02`, 30 seeds | Log-loss 0.07–0.09 / top-1 0.97–0.98 on deterministic families; CSPRNG 0.06 |
| **exp14** prediction incl. new algorithms | Can it predict generators it has no expert for? | Proposed (Fixed-Share + universal) vs bank-only vs Markov vs uniform, `P_d=0.95`, `P_fa=0`, 12 seeds, + `P_d` sweep | `lagged_fib` 0.88 vs 0.42; known families 1.00; CSPRNG 0.07 — **stale**, near-perfect sensor only |
| **exp19** sequence coverage | Every structured sequence type, and secrets far outside recce | Before/after structured experts, two sensors, 8 seeds | All structured types 0.84–1.00 clean, 0.71–1.00 lossy; LFSR-16 lossy 0.12; CSPRNG at chance. Only committed JSON |

### F. Non-stationary adversary

| Exp | Question | Protocol | Key result |
| --- | --- | --- | --- |
| **exp15** regime switch | After a mid-engagement switch, how fast does prediction come back? | Markov→Markov switch at slot 120/240, 30 seeds × 4 `α`, CUSUM | Weights+context flush 1.2–2.3× faster than weights-only — **stale** |
| **exp18** switch tracking | Detect, date and re-identify arbitrary algorithm switches | 6 random plans × 400 hops over 13 algorithms, 4 sensors | 90–100 % detected, median delay 0–11 hops, 0–3 % dangerous locks — **stale** |

### G. Downstream use (jamming allocation) — thesis/journal material

| Exp | Question | Protocol | Key result |
| --- | --- | --- | --- |
| **exp16** allocation | Does the threshold-aware knapsack beat water-filling? Does reactive jamming help on a CSPRNG? | 30 seeds, `P_J = 3τ` | Knapsack 0.11 vs 0.00 at κ≈0; reactive 0.11→0.82 as dwell 1→8 |
| **exp17** feasibility map | Which strategy is feasible where? | Analytic | ~71 % of dwell/latency plane reactive-jammable — **stale** after the `P_d^{2L}` fix |

---

## 2. What must be fixed before submission (blocking)

1. **Re-run the five stale experiments** — exp9 (new max-competitor e-process),
   exp14/exp15/exp18 (pre-structured-expert predictor), exp17 (clean-run rule fix).
   Their tables in the draft section are marked `\stale` until then.
2. **Commit the result JSONs** (un-ignore `results/*.json` or copy them into
   `docs/results/`). Right now a reviewer can reproduce only exp19.
3. **Report confidence intervals for every zero.** "Hard false-ID 0.00 over 40
   seeds" has a 95 % upper bound of ≈ 7.5 % (rule of three, 3/n). Use Wilson or
   Clopper-Pearson intervals in every table; exp13's 7,500 episodes are the only
   place a near-zero rate is tightly bounded (≈ 750 episodes per family → upper
   bound ≈ 0.4 %).
4. **Raise seed counts on the headline prediction results.** exp14 has 12 seeds,
   exp19 has 8, exp18 has 6 plans per sensor. Target ≥ 50 for exp14/exp19 and
   ≥ 30 plans for exp18, with bootstrap CIs and a paired test (proposed vs bank-only)
   on the out-of-bank rows.
5. **Run out-of-bank prediction at realistic sensors.** Every "new algorithm"
   number is at `P_d=0.95, P_fa=0`. Repeat exp14 at the typical (0.8, 0.02) and
   contested (0.6, 0.05) sensors; the claim "predicts generators it has no expert
   for" must be shown where the sensor is lossy.

## 3. Experiments a reviewer will ask for (new)

Ranked by how much each closes an attack on the paper.

1. **Mis-specified sensor.** The bank scores with the same `P_d`/`P_fa` that
   generated the data. Sweep assumed vs true (e.g. assumed 0.9 when true is 0.6,
   and `P_fa` off by 5×), and add an online `P_d`/`P_fa` estimator (EM on the clean
   slots) as the fix. Without this, "sensor-aware" is "sensor-oracle".
2. **Correlated (bursty) misses.** Replace i.i.d. misses with a Gilbert-Elliott
   channel at the same average `P_d`. Fades and interference are bursty; this is
   exactly what breaks consecutive-run methods and should favour the state filter
   even more.
3. **A fair open-set baseline.** The no-`H_0` SPRT is a straw man: a closed-set test
   must name something. Add (a) closed-set MAP + reject threshold on the posterior
   (Chow), (b) SPRT with a uniform null added, (c) a trained sequence classifier
   with max-softmax / energy rejection. Report accuracy-vs-false-ID curves, not
   single points.
4. **A strong learned predictor baseline.** The exp3 learned classifier is a small
   MLP on summary statistics. For next-hop prediction, add an LSTM/GRU or small
   transformer trained on the generator library (and, for the held-out families, an
   in-context transformer in the style of Tao et al., ICML 2025). Show where training
   helps and where the training-free mixture wins (held-out families, lossy sensor).
5. **Sensitivity of the free design constants.** `H_0` prior floor (0.15), MSPRT `δ`,
   Fixed-Share `α` and `η`, lock rule (95 % on two hops), rule-search prune margin
   (25 nats). One table or a tornado plot; reviewers distrust hand-set constants
   that are never varied.
6. **Scaling in `M` and window `T`.** All results are `M=16`. Run `M ∈ {16, 64, 256}`
   (fixes the `lcg_full` period-16 confound too) and report hops-to-identify,
   accuracy and per-hop runtime. Also `T` from 30 to 200.
7. **Runtime vs hop rate.** The switch tracker costs 40–90 ms/hop. State the hop
   rates this supports (≈ 11–25 hops/s) and compare with real systems (Bluetooth
   1,600 hops/s, tactical VHF ≈ 100+ hops/s). Profile the bank alone vs the tracker.
8. **A pre-registered held-out generator set.** The same author wrote the generators
   and the experts. Freeze a set the experts were never tuned on (xorshift, PCG,
   truncated Mersenne Twister, Bluetooth BR/EDR hop kernel, BLE CSA#2, a
   `quadratic_cg` with larger modulus) before running, and report all of them.
9. **Semi-realistic sensing.** Drive `P_d`/`P_fa` from an energy detector at a given
   SNR on simulated IQ (or on public recordings, e.g. RFUAV / drone FHSS captures),
   so the operating points map to physical conditions. Even one IQ-derived
   experiment removes "everything is assumed".
10. **Timing errors and multiple emitters.** Slot jitter (insertions/deletions) and
    two interleaved emitters. If out of scope, say so in Limitations — but at least
    one jitter sweep shows where the slot-synchronous assumption breaks.
11. **Adversarial generator choice.** An adversary that picks the generator to
    maximise the bank's error (e.g. a Markov hopper shaped to mimic an LFSR prefix,
    or the entropy-max hopper). This is the worst-case counterpart to exp13.
12. **Switch baselines for exp18.** No external baseline exists today. Compare with
    plain BOCPD on the loss stream, CUSUM + full restart, and an oracle restarted at
    the true switch. exp15 only switches Markov→Markov; include family switches.
13. **Prediction calibration.** exp10 calibrates the identification posterior only.
    Add a reliability diagram for the next-hop distribution `p̂_t` (top-1 confidence
    vs hit rate). This is what any downstream allocator consumes.
14. **Allocation claims (only if kept in the paper).** A CDT run reporting realised
    loss vs target `α` (with an `α` inside the achievable range), and a Monte-Carlo
    for `tones_from_confidence` (coverage vs tones lit, against fixed-k).

## 4. Improvements to existing experiments

| Exp | Change |
| --- | --- |
| exp1 | Report accuracy beside hops; Markov dips to 0.83 at `P_d=0.5`. Add a `P_fa` axis or say why 0.02 is fixed. |
| exp2 | Explain the `filtered_lfsr` "wrong" as soft; split false-ID into hard vs soft as in exp13. |
| exp3 | Add per-class breakdown (in-bank vs out-of-bank) so the headline is not a mix-weighted average; add the fair baselines above; McNemar p-values in the table. |
| exp4/5 | Wilson CIs on the zeros; sweep the `H_0` prior floor. |
| exp6 | Add `P_fa > 0` (ghosts are the harder failure for BM), and longer `L` where enumeration becomes expensive — show the cost curve. |
| exp7 | Add the mis-specified-sensor arm here (aware-but-wrong). |
| exp9 | Re-run with max-competitor e-values; also show `τ` cost vs MSPRT, and an out-of-bank row to show the bound is not claimed there. |
| exp10 | ECE with bootstrap CI and per-bin counts; consider a temperature-scaled posterior as a fix and report post-fix ECE. |
| exp11 | Also fit the predicted slope `1/KL` from the computed KL and compare to the measured slope, not only R². |
| exp13 | Add hops-to-identify per family per cell; plot accuracy heat-maps over (`P_d`, `P_fa`). |
| exp14 | More seeds, lossy sensors, paired test, include the structured experts (current predictor), show top-3 too. |
| exp15 | Family-level switches, more detectors (BOCPD, conformal-e) side by side, false-alarm rate vs delay curve (ARL). |
| exp18 | More plans, CIs, an oracle-restart baseline, and per-pair (from→to) detection matrix. |
| exp19 | More seeds; report lock hops against the information minimum (e.g. `log_M` keyspace, `P+1`) as an explicit column. |

## 5. Wording to keep the paper defensible

- "state-enumeration recovery under a detection-level channel", not "gap-robust BM".
- "training-free online universal predictor", not "zero-shot".
- "avoids hard false identification", not "detects unknown generators".
- `κ` and the feasibility map are operational heuristics, not predictability bounds.
- e-process bound: per in-bank competitor (`β`), `(K−1)β` for any wrong one; nothing
  claimed out of bank.
