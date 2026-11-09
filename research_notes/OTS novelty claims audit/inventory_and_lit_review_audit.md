# OTS/fhew — Contribution Inventory, Citation Audit, and Missing Closest Prior Work

Cross-cutting notes for the novelty audit. Scope: everything NOT owned by the four
per-slice reviewers (sensor likelihoods/H_0/MSPRT+e-process; Fixed-Share/universal
predictor/switch tracking; cryptanalytic experts LCG/BM/seed-beam/rule-search;
confidence-adaptive multi-tone jamming). I cover the full inventory, the lit-review
citation check, and missing prior work that threatens the claims.

Repo read: `README.md`, `docs/CHANGELOG.md`, `docs/methodology_and_results.md`,
`docs/Thesis_Architecture_Revision_3.1_prediction_first.md`, all `fhew/*.py`,
`experiments/*.py`, `tests/*.py`, `scripts/run_all.py`, `dashboard/*.py`, `git log`.
As-of 2026-11-04. Nothing in the repo was modified.

---

## PART 1 — CONTRIBUTION INVENTORY

### 1.0 Status key and a load-bearing caveat on "evidence"

- **E** = implemented AND evaluated by an experiment whose numbers are quoted in the docs.
- **R** = implemented and tested (unit tests pass) but its headline figure/JSON is NOT
  re-run after later code changes (stale numbers).
- **D** = described in the thesis/lit-review as a contribution but NOT implemented in code.

**Reproducibility caveat (applies to every "E" row).** `results/` is git-ignored
(`.gitignore`: `results/`). The only committed result JSON is
`results/exp19_sequence_coverage.json`. Every other headline number in
`docs/methodology_and_results.md` and the README lives only in prose / committed PNGs
under `docs/figures/` — the underlying JSON is not in the repo and I did not re-run the
sims. So all exp1–exp18 numbers are "as last recorded by the author," not independently
reproduced here. A reviewer can regenerate them with `python -m scripts.run_all`, but as
checked-in artefacts they are unverified. Treat the figures as the primary evidence.

### 1.1 Contribution inventory table

| # | Candidate contribution | Where (file / function) | Evidence experiment | Headline number | Status |
|---|---|---|---|---|---|
| C1 | Sensor-aware reduced-emission likelihood (detection-level HMM; `P_d`/`P_fa` folded into the likelihood, all-false-alarm constant cancels) | `fhew/likelihood.py` (`SensorAwareEmission`, `NaiveEmission`, `DeterministicTrajectoryFilter`, `ChannelHMMFilter`) | exp7 sensor ablation | sensor-aware acc 0.98 vs naive 0.60 at `P_d=0.5`; 15.3 vs 21.1 hops | E |
| C2 | Bayesian hypothesis bank + MSPRT anytime stopping | `fhew/experts.py` (`HypothesisBank`), `fhew/stopping.py` (`MSPRT`) | exp1, exp3, exp11 | in-bank ID in 4–13 hops @100%; bank 0.79 acc vs SPRT 0.66 | E |
| C3 | Mandatory none-of-the-above class `H_0` (pluggable uniform/occupancy/markov, prior floor 0.15) | `fhew/experts.py` (`H0Hypothesis`, `_OccupancyFloor`) | exp4, exp5 | CSPRNG routed to unknown 87%, hard false-ID 0.00; SPRT false-IDs 100% | E |
| C4 | E-process (anytime-valid) stopping with Ville false-ID bound | `fhew/stopping.py` (`EProcess`) | exp9 | empirical false-ID ≤ nominal β at every β | E |
| C5 | Stopping-time law characterization (E[τ] ∝ log(1/δ) and ∝ 1/P_d) | exp11 harness | exp11 | R²=0.984 (log 1/δ), R²=0.997 (1/P_d) | E |
| C6 | Full thesis expert bank (13 experts, one best per family) | `fhew/experts.py`; `fhew/config.py` (`full_bank_specs`, `identification_bank_specs`) | exp12, exp13 | clean diagonal @ `P_d=0.95`; 7,500-episode sweep per-type acc ≥0.92, hard false-ID ≤0.05 | E |
| C7 | Deterministic experts as particle filter over parameter space = gap-robust state-filtered BM, LCG cryptanalysis, seed search, correlation attack | `fhew/experts.py` (`DeterministicHypothesis`, `AlgebraicLFSRHypothesis`, `LCGCryptanalysisHypothesis`, `_filtered_lfsr_trajectories`); `fhew/recovery.py` | exp6 | plain BM collapses to 0.00 at any `P_d<1`; state-filtered holds 1.00 | E |
| C8 | Parametric chaos-ID via induced symbolic-dynamics transitions | `fhew/experts.py` (`MixtureHMMHypothesis`, `_induced_transition`, `_build_chaos_id`) | exp12, exp13 | logistic/tent/Hénon all ID'd by one expert; chaos acc ≈0.95–1.00 | E |
| C9 | Phase-space (delay-embedding kNN) predictor for unknown chaos | `fhew/experts.py` (`PhaseSpacePredictorHypothesis`) | exp12 (fallback column, empty) | exists as fallback; not a primary winner | R |
| C10 | Fixed-Share expert mixture as the next-channel predictor (recent-skill weighting, share floor α, tracking-regret framing) | `fhew/mixture.py` (`FixedShareMixture`, `build_default_mixture`), `fhew/predict.py` (`ChannelPredictor`) | exp14 | — (see C11) | E |
| C11 | Zero-shot universal predictor carrying prediction on NEW out-of-bank algorithms | `fhew/experts.py` (`ContextMixingHypothesis`, gap-robust clean-run context) | exp14 | `lagged_fib` 0.88 top-1 vs 0.42 id-only; known fams ≈1.00; CSPRNG ≈1/16 | E |
| C12 | Regime monitor + context flush (CUSUM / BOCPD / conformal-e; resets weights AND forgets universal context) | `fhew/regime.py` (`CUSUMDetector`, `BOCPDDetector`, `ConformalEDetector`, `RegimeController`); `forget()` on every stateful expert | exp15 | flush recovers 1.2–2.3× faster than weights-only; CUSUM fires ~78% switches, ~0.12 false alarms/ep | E |
| C13 | Online dwell estimation as predictability feature + reactive trigger | `fhew/dwell.py` (`DwellTracker`, `dwell_predictability`, `estimate_dwell`) | exp16 panel B, exp17 | — (feeds C14/C15) | E |
| C14 | Threshold-aware greedy knapsack multi-tone allocator (spot↔barrage from one knob λ) + water-filling/weighted-WF/equal/spot baselines | `fhew/allocator.py` (`greedy_knapsack`, `water_filling`, `equal_power`, `spot`, `EffectModel`, `AllocationController`) | exp16 panel A | knapsack 0.11 vs WF 0.00 @ κ=0; tie at κ=1 | E |
| C15 | Reactive/follower allocation (jam observed channel when dwell>latency) | `fhew/allocator.py` (`reactive_follower`) | exp16 panel B | reward 0.11 (dwell 1) → 0.82 (dwell 8) on uncrackable CSPRNG | E |
| C16 | CDT (Conformal Decision Theory) controller on λ: λ_{t+1}=λ_t+η(L_t−α) | `fhew/allocator.py` (`CDTController`) | only `tests/test_mixture_allocation.py::test_cdt_keeps_mean_loss_near_alpha` | mean loss ≤ α+0.05 in a unit test; NOT in any experiment figure | R |
| C17 | Confidence-capped multi-tone count (`tones_from_confidence`, `max_tones`) — spread when unsure | `fhew/allocator.py` (`tones_from_confidence`, `predictability`) | Control room only; `tests/test_mixture_allocation.py` (5 tests) | sharp→1 tone, flat→max_tones; no Monte-Carlo experiment | R |
| C18 | Analytic feasibility map incl. reactive-follower region | `fhew/feasibility.py` (`classify`, `dwell_latency_grid`, `window_hoprate_grid`) | exp17 | ~71% of dwell/latency plane jammable at κ=0 | E |
| C19 | Scenario/episode builders: regime-switch, long-dwell, entropy-max adversary | `fhew/scenarios.py`; `fhew/generators.py` (`EntropyMaxGenerator`) | exp15, exp16 | entropy-max @ dwell 1 defeats both weapons (stated boundary) | E |
| C20 | SwitchTracker: BOCPD over anchored bank segments (detector + marginalised identifier + run-length-weighted predictor) | `fhew/tracker.py` (`SwitchTracker`, `_Segment`, `ExpertLibrary`, `universal_specs`) | exp18 | 90–100% switches detected; next-hop 55–79% by hop 6–10; 0–3% dangerous locks | E |
| C21 | Modular linear-recurrence learner (zero-shot) for the tracker | `fhew/experts.py` (`RecurrenceHypothesis`) | exp18 (bracketed before/after) | lagged perfect-sensor hops 6–10: 66→79%; full-state LCG typical 0→100% | E |
| C22 | Structured-sequence experts: Bayesian rule search, full-keyspace seed beam, adaptive-window wide BM | `fhew/experts.py` (`RuleSearchHypothesis`, `default_rule_library`, `SeedBeamHypothesis`, `AdaptiveBMHypothesis`) | exp19 | every structured type → ≈1.00 top-1; weak-seed 2^18 locks 5–7 hops; 16-bit LFSR 1.00 clean / 0.12 lossy | E |
| C23 | Structured generator library (arithmetic, polynomial, linear/nonlinear recurrence, lookup table, hybrids, noisy rule) | `fhew/generators.py` (8 new generator classes) | exp19 | generator laws unit-tested (`tests/test_sequences.py`) | E |
| C24 | Held-out "new algorithm" generators (lagged_fib, quadratic_cg, sine_map) to measure prediction generalisation | `fhew/generators.py` (`LaggedFibonacciGenerator`, `QuadraticCongruentialGenerator`, `SineMapGenerator`) | exp14 | prediction-generalisation testbed | E |
| C25 | Baselines: pairwise SPRT, fixed-length LRT, numpy-MLP learned classifier w/ explicit `unknown`, optional torch-LSTM drop-in | `fhew/baselines.py` (`PairwiseSPRT`, `FixedLengthTest`, `LearnedClassifier`, torch block) | exp3 | learned clf 0.38 acc; LSTM path exists only if torch installed (optional) | E (numpy) / R (LSTM) |
| C26 | Textbook Berlekamp–Massey (GF(2)) reference implementation + linear-complexity profile | `fhew/recovery.py` (`berlekamp_massey_gf2`, `linear_complexity_profile`, `recover_and_predict`) | exp6 | the brittle baseline that collapses under gaps | E |
| C27 | Calibration metrics + significance machinery (reliability/ECE, paired bootstrap, McNemar) | `fhew/metrics.py` (`ece`, `reliability_curve`, `paired_bootstrap_diff`, `mcnemar_test`, `bootstrap_ci`) | exp10, exp3 | ECE 0.154; bank beats SPRT +0.139 (p<1e-3) | E |
| C28 | Information-parity principle, structurally enforced + audited | bank sees only `Observation`; `tests/test_fhew.py` (information-parity audit) | exp5 (secret outside recce range not identifiable) | hard false-ID 0.00–0.03 on out-of-range secrets | E |
| C29 | Grounding/provenance table (calibrated vs assumed params) | `fhew/datagen.py` (`GROUNDING`, `grounding_report`) | — | ALL parameters currently `source:"assumed"` (RFUAV named only as a placeholder note; nothing is actually calibrated yet) | R |
| C30 | Interactive dashboard / Control room demo artefact (7 pages + live server + standalone HTML + CLI) | `dashboard/` (`build.py`, `serve.py`, `control.py`, `views/*.js`); `ots_dashboard.html` (1.95 MB committed) | replays exp12/13 episodes, exp6/exp7 proof ablations | demo only; numbers come from real bank runs recorded by `dashboard.build` | E (as demo) |
| C31 | Config-driven, fixed-seed experiment harness + runner | `experiments/common.py`, `scripts/run_all.py`, `configs/*.yaml` | all exp | `--quick`, `--only`, `--plot-only` supported | E |

Test totals (from `pytest --collect-only`): `test_fhew.py` 79, `test_mixture_allocation.py`
36, `test_sequences.py` 23, `test_tracker.py` 16 = **154 tests** (README/CHANGELOG still
say "126" / "67+29+16" — stale, see §1.4).

### 1.2 Components described in the thesis/lit-review but NOT implemented (status D)

These are claimed as contributions in `Thesis_Architecture_Revision_3.1` and/or the
attached lit review, but **grep across `fhew/`, `experiments/`, `tests/`, `dashboard/`
finds no implementation**. A reviewer who reads the thesis then the code will see the gap.

- **Mamba-2 / S4 sequence encoder** (thesis §3.3, lit-review II-B/II-D "the backbone our
  encoder adopts"): no `mamba`, `s4`, or encoder module anywhere. The actual predictor is a
  numpy variable-order context-mixing model. **This is the single biggest doc↔code gap:**
  the lit review builds a whole subsection around the SSM backbone that does not exist.
- **SAOCP / DtACI / ACI online conformal set calibration** (thesis §6.2, lit-review II-C):
  no conformal *set* layer. `fhew/predict.py` mentions "APS" only in a comment. The only
  conformal object in code is the CDT scalar controller (C16) and the conformal-e change
  detector (C12). The APS/RAPS prediction set, SAOCP, DtACI, plain-ACI ablation — none exist.
- **Conformal Risk Control (CRC) batch baseline** (thesis §6.3 "replaces batch CRC"): not implemented.
- **Global PSRO / exploitability certificate / red-team population** (thesis §9, §13): no
  PSRO, no exploitability metric, no RL hopper population. The DQN/PPO/CVaR-PPO adversaries
  and "learned (DQN, PPO)" library entries (thesis §11) are **not** in `fhew/generators.py`
  (registry has no DQN/PPO generator; `cvar|ppo|dqn` only appears as a comment in
  `allocator.py`).
- **Track-aware GNN emitter sorter + Kalman tracker / moving emitters** (thesis §3.2, L0,
  lit-review II-D): no GNN, no Kalman, no multi-emitter deinterleaving. `bearing`/`emitter_id`
  exist as frozen-schema optional fields only; the sim is single-emitter.
- **Active info-gain probing bandit** (thesis §8 "the most original component"): not
  implemented. `probe` appears only in `sensor.py` as an unrelated token.
- **TD-MPC2 adaptive expert** (thesis §9): only a `BehaviourHypothesis` recency-weighted
  "reduced stand-in" (`fhew/experts.py:492`), explicitly labelled a stand-in; no world model.
- **Intermittent-feedback conformal variant** (thesis §6.5): not implemented.

Net: the **implemented system is Layers 2 + (a slice of) 3–4**: identification bank,
Fixed-Share predictor, CDT knob + knapsack allocator, regime monitor, switch tracker,
feasibility map. Layers 0 (sorting), 1 (Mamba-2 encoder), the conformal *set* layer, the
probe (L5), and the PSRO game (§9) are **paper-only**. The lit review surveys all of them as
if they are the contribution.

### 1.3 Attached-draft claims now outdated / superseded by later commits

The attached lit review is dated 21 Oct 2026 and matches the thesis Revision 3.1 at
commit `f990a9f` (22 Oct). Commits after it (switch tracker `341e97b`, multi-tone jam
`5488d8b`, structured sequences `90e2e86`) added code the lit review does not mention:

- The lit review's II-B gap ("Published FH predictors are fit per sequence… assume clean
  consecutive outputs… emit a point prediction") is now **partly answered by the author's
  own later code** (C22 structured rule search, C20/C11 zero-shot gap-robust predictor,
  C21 recurrence learner) — i.e. the "gap" framing predates the structured-sequence and
  switch-tracker work and should be re-read against it.
- The lit review claims the predictor rides a **Mamba-2 backbone** (II-B, II-D). The built
  predictor is a numpy context-mixing model (C11). Any "Mamba-2" wording is unsupported by
  the current codebase and is the most exposed single statement.
- "Multi-tone" novelty (II-A gap) is now backed by C17 (`tones_from_confidence`/`max_tones`),
  added in `5488d8b` AFTER the review — but C17 has **no Monte-Carlo experiment**, only the
  Control room and 5 unit tests. The allocation experiment exp16 (C14) is single-tone
  effect-on-true-channel reward, not a multi-tone-coverage metric.
- CHANGELOG itself flags `**Not re-run:** exp14/exp15/exp18 figures still show the
  pre-change predictor` (2026-11-03 entry) — so the committed exp14/15/18 PNGs predate C22,
  i.e. the headline prediction figures are stale relative to the latest predictor.
- Test-count claims are stale: README says "126 tests"; actual 154.

---

## PART 2 — CITATION AUDIT

Method: spot-checked the load-bearing / recent citations via web search (IEEE Xplore,
arXiv, NeurIPS/ICLR proceedings, publisher pages). "Verified" = title + authors + venue
confirmed against a primary or near-primary record. The review's own closing note already
asks to confirm [13]–[15],[29],[31]–[32],[40]–[45],[48]–[50]; findings below.

### 2.1 Verified (exist and are characterised correctly)

- **[19] Massey 1969**, "Shift-register synthesis and BCH decoding," IEEE Trans. Inf.
  Theory 15(1):122–127 — canonical, correct. (Note: review attributes Berlekamp–Massey to
  Massey alone; the "Berlekamp" half is Berlekamp 1968. Minor, conventional.)
- **[20] Strickland & Haddad**, "Predicting Jamming Systems Frequency Hopping Sequences
  Using Artificial Neural Networks," IEEE SoutheastCon 2023, pp. 313–318, DOI
  10.1109/SoutheastCon51012.2023.10115067. Verified; correctly characterised as LFSR/m-sequence
  reverse-engineering from few samples. [Georgia Southern Scholars record](https://scholars.georgiasouthern.edu/en/publications/predicting-jamming-systems-frequency-hopping-sequences-using-arti/)
- **[13] Rao et al.**, "Efficient Jamming Resource Allocation Against FHSS in WSNs with
  Asynchronous DRL," IEEE Sensors J. 24(8), 2024, DOI 10.1109/JSEN.2024.3369038 (pp.
  13560–13577). Verified; A3C + partial-band characterisation correct. (Review gives no
  page/DOI; fill in.)
- **[14] Rao et al.**, meta-DRL follow-up, IEEE TAES 2024, DOI 10.1109/TAES.2024.3418944.
  Verified via OpenAlex; "meta/trust-region to generalise across hop-set distributions"
  correct.
- **[25] Li, Gao et al.**, "LSTM-based Frequency Hopping Sequence Prediction," WCSP 2020,
  Nanjing, IEEE Xplore doc 9299717. Verified; authors Gao Li, Jianliang Xu, Weiguo Shen,
  Wei Wang, Zitong Liu, Guoru Ding. Characterisation correct.
- **[26] Li et al.**, "Frequency-Hopping Frequency Reconnaissance and Prediction for
  Non-cooperative Communication Network," China Commun. 18(12):51–64, 2021. Verified as a
  distinct paper from [25] (confirmed via MDPI reference list). Correct.
- **[27] Zhu et al.**, "Deep-Learning-Based Recovery of Frequency-Hopping Sequences for
  Anti-Jamming Applications," Electronics 12(3):496, 2023, MDPI. Verified; CNN-GRU + transfer
  learning correct.
- **[38] Gibbs & Candès**, "Adaptive Conformal Inference Under Distribution Shift," NeurIPS
  2021 (oral), arXiv 2106.00170. Verified; single-parameter online quantile update correct.
- **[42] Angelopoulos, Bates, Fisch, Lei, Schuster**, "Conformal Risk Control," ICLR 2024,
  arXiv 2208.02814. Verified; "extends conformal from coverage to any monotone bounded loss"
  correct.
- **[45] Lekeufack, Angelopoulos, Bajcsy, Jordan, Malik**, "Conformal Decision Theory: Safe
  Autonomous Decisions from Imperfect Predictions," arXiv 2310.05921 (2023), also IEEE ICRA
  2024. Verified; the λ_{t+1}=λ_t+η(α−ℓ_t) conformal-controller mechanism and the
  arbitrary/adversarial-sequence claim are correctly stated. This is the flagship citation
  and it checks out.

### 2.2 Could not verify / likely needs correction (FLAG)

- **[15] J. Zhang & X.-F. Wu**, "Cooperative jamming over DRL-based frequency hopping… a
  one-leader multi-follower Stackelberg game," **IEEE TIFS vol. 20, 2025.** **NOT CONFIRMED.**
  Two independent searches for this exact title/authors/venue returned nothing matching. A
  real Zhang–Wu FH-jamming item is a *bi-matrix faking-slot* game (different structure); the
  "one-leader multi-follower Stackelberg" phrasing matches other authors' UAV anti-jamming
  papers, not a Zhang–Wu TIFS 2025 cooperative-jamming paper. **This is a load-bearing
  citation** — it is the review's "closest multi-tone / cooperative-jamming prior art." Must
  be verified against IEEE Xplore directly or replaced before submission.
- **[48] "Authors", "Global policy-space response oracles for two-player zero-sum games,"
  arXiv, 2026.** Placeholder author list (the review admits this). I could not confirm a 2026
  "Global PSRO" paper. Real, citable neighbours: McAleer et al. "Anytime (Optimal) PSRO for
  Two-Player Zero-Sum Games," arXiv 2201.07700 / 2202.00633 (2022); and McAleer et al.
  "Self-Play PSRO"/"Global PSRO" line. Confirm the exact paper and fix authors/year.
- **[49] "Authors", "Radar signal sorting via graph convolutional network and semi-supervised
  learning," 2024** and **[50] "Authors", "Temporal graph neural network for ISAC target
  detection and tracking," arXiv, 2026.** Both have placeholder authors; not verified.
  Plausible (GCN radar-pulse deinterleaving is a real active area) but must have real
  author/venue filled in. Flag as unverifiable as written.
- **[24] Kim et al.**, "Spreading code estimation using inherent linear relation in
  non-cooperative contexts," IEEE TAES, 2025. Not directly confirmed in search; plausible and
  topically consistent (LFSR relation + Berlekamp–Massey for spreading-code estimation is a
  known thread) but verify the exact title/year/DOI.
- **[29] M. Deng et al.**, "Nonlinear hop frequency prediction based on HT-HMM in navigation
  confrontation scenarios," VTC2025-Fall. Not confirmed in search; the review's own note flags
  it. The review text calls it "homotopy-HMM" while the citation says "HT-HMM" — reconcile the
  expansion (homotopy vs Hilbert-transform vs other) and confirm the venue.
- **[7]/[1] Cheng, Ling, Zhu**, "DRL-based anti-jamming approach for fast frequency hopping
  systems," IEEE OJ-COMS 2025 — appears as BOTH [1] and [7] (the review double-lists it, with
  [1] as a stand-in for "cognitive radio/radar foundations"). Confirmed to exist (DOAJ record),
  but using the same 2025 OJ-COMS paper as the citation for the *historical* cognitive-radio/
  radar lineage [1] is a mischaracterisation — [1] should point to Haykin's cognitive radar /
  Mitola cognitive radio, not to a 2025 FFH paper. **Flag [1] as misattributed.**

### 2.3 Not individually re-verified (lower risk, standard references)

[2] Han/Xiao/Poor ICASSP 2017, [3] Liu et al. IEEE Commun. Lett. 2018, [21] Meier &
Staffelbach J. Cryptol. 1989, [22] Courtois & Meier EUROCRYPT 2003, [30] Gu/Goel/Ré S4
ICLR 2022, [31] Gu & Dao Mamba arXiv 2312.00752, [32] Dao & Gu "Transformers are SSMs"
ICML 2024, [39] Gibbs & Candès JMLR 2024 (DtACI), [40] Bhatnagar et al. SAOCP ICML 2023,
[46] Perdomo et al. performative prediction ICML 2020, [47] Lanctot et al. PSRO NeurIPS
2017 — all are well-known and almost certainly correct as cited; confirm volume/page only.
**General risk:** the review relies heavily on 2024–2026 preprints (the thesis §14 admits
this). Several ([43] Phil. Trans. R. Soc. A 2026, [44] arXiv 2026, [48]–[50]) are
forward-dated / preprint and must be checked for published status before submission.

---

## PART 3 — MISSING CLOSEST PRIOR WORK (and how it threatens each claim)

These are works a reviewer would cite *against* the paper that the review misses. Grouped by
which OTS claim they bound.

### 3.1 Against C10/C11/C20 (zero-shot / in-context sequence prediction of PRNG hops)

- **Tao, Doshi, Kalra, He, Barkeshli, "(How) Can Transformers Predict Pseudo-Random
  Numbers?", arXiv 2502.10390, ICML 2025.** THE most threatening missing citation. Decoder-only
  transformers predict next LCG output x_{t+1}=a·x_t+c mod m, generalise to unseen moduli up
  to 2^16, and recover the modulus + prime factorisation in-context.
  [arXiv](https://arxiv.org/abs/2502.10390) · [ICML 2025](https://icml.cc/virtual/2025/poster/44774).
  **Threat:** directly pre-empts OTS's "zero-shot, in-context, gap-robust hop predictor" and
  "LCG falls to our seed-beam/cryptanalysis" novelty — a 2025 paper already does in-context
  LCG prediction with a learned sequence model, which is exactly the OTS universal-predictor
  story but with a transformer and a cleaner generalisation result. It also undercuts the
  thesis's stated "Mamba-2 backbone" motivation (they use a vanilla transformer and analyse
  *why* it works). OTS must cite and differentiate (noisy passive sensor + gaps + bank
  coupling) or be scooped on the headline.
- **Classic "neural network predicts LCG" line** (cited inside 2502.10390): feed-forward nets
  predicting modified-LCG outputs (~2021). Bounds the novelty of "a learned model can predict
  a weak PRNG's next output."

### 3.2 Against C30/C19 (open FH-EW environment) and C23/C24 (benchmark suite)

- **Turing Synthetic Radar Dataset (TSRD) + Turing Deinterleaving Challenge**, Alan Turing
  Institute & Dstl, arXiv 2602.03856 (2026). 6,000 pulse trains, ~3B pulses, staring/scanning
  receivers, V-measure scoring. [arXiv](https://arxiv.org/abs/2602.03856). **Threat:** weakens
  "no open environment of this kind exists" (thesis §11, §13) — it is a released, scored,
  Dstl-backed benchmark for the adjacent deinterleaving/sorting problem. Not FH-specific, so
  the claim survives if narrowed to "FH-EW with a dumb-to-adaptive generator library + passive
  sensor + jamming action," but the blanket "no open environment" is overclaimed.
- **Qixian Zhao, Radar Signal Deinterleaving & Modulation-Type Recognition Dataset**, IEEE
  DataPort, DOI 10.21227/k9zh-8y52 (~10k files). Same bounding effect for the sorting layer
  (which is anyway unimplemented — §1.2).
- **Gap I could not close:** no clearly open, FH-sequence-specific benchmark/dataset found
  (IEEE DataPort / Papers-with-Code searches returned radar-pulse sets only). So OTS's
  environment claim is *defensible if scoped to FH* — but note this is unverified absence, not
  confirmed absence, and the simulator currently has NO calibrated grounding (C29: all params
  "assumed"; RFUAV named but not used). A reviewer will press on "calibrated against public
  datasets" (thesis §11) vs the code where nothing is calibrated.

### 3.3 Against the whole pipeline (passive intercept → predict → jam, "cognitive EW")

- **Bluetooth / BLE hop-sequence prediction attacks — the biggest missing *application*
  precedent.** Cauquil's **BtleJack / BLE 5 CSA#2 reverse engineering** (channel-selection PRNG
  recovered from sniffing two channels, then connection hijack/jam) and
  **Albazrqaoe, Huang, Xing, "A Practical Bluetooth Frequency Hopping Attack,"** (MSU; models
  the hop-selection kernel, exposes the clock, predicts the sequence to capture packets — IEEE/ACM
  ToN 2018 line). [MSU thesis](https://d.lib.msu.edu/etd/1032) · [BtleJack writeup](https://hackaday.com/2019/08/10/new-bluetooth-5-channel-hopping-reverse-engineered-for-jamming-and-hijacking).
  **Threat:** this is a real, deployed instance of "intercept a FH link, predict its next
  channel from the generator structure, then jam/hijack" — the exact OTS loop, on a real
  protocol, years earlier. OTS's novelty is NOT the concept of predicting hops to jam (BLE
  attacks own that); it must be reframed as "generator-agnostic, bank+universal, under a lossy
  passive sensor, with calibrated budgeted multi-tone allocation." The review cites zero
  protocol-attack literature, which a security/EW reviewer will immediately raise.
- **USPTO patents on tracking/deriving future FH frequencies to mitigate interference**
  (e.g. US 6,941,110; and the SDR FH-jamming patent family US 8,903,304 / 9,246,629 / 9,749,085
  / 10,256,940). **Threat:** prior-art patents that explicitly "derive future hop frequencies"
  and "jam FH comms with SDR." Bound the "to our knowledge first" wording; at minimum a
  freedom-to-operate / prior-art acknowledgement is expected.
- **Cognitive-radar anti-jamming DQN line** (Ak & Brüggenwirth, "Avoiding Jammers: A RL
  Approach," arXiv 1911.08874; Li/Jiu/Liu IEEE RadarConf DQN frequency-agile radar; Zhang et
  al. J. Radars 2024). [arXiv 1911.08874](https://arxiv.org/abs/1911.08874). **Threat:** these
  are the radar-countermeasures mirror of the comms anti-jamming work the review already covers;
  a radar-EW reviewer expects them. They also pre-empt "DRL to choose jamming/hop frequency
  against an agile emitter." Bounds C14/C15 framing (RL already selects jamming frequency), not
  the calibration coupling.
- **No true end-to-end CEW survey found** linking recon+prediction+jamming in one reference
  (searches returned patents + narrow MDPI pieces). So the review's implicit claim that the
  four programs "matured in isolation" is *plausible* — but Dabcevic et al. (J. Signal Process.
  Syst. 2016, cognitive-radio communications-EW) and the general CEW closed-loop framing exist
  and should anchor II-A's opening instead of a 2025 FFH paper ([1] misattribution, §2.2).

### 3.4 Against C16 (CDT-calibrated jamming) — the flagship

- The flagship coupling (online-calibrated decision knob → threshold-aware budgeted
  multi-tone jam, distribution-free forgone-effect bound) is **genuinely not pre-empted** by
  anything I found: CDT [45] is robot-motion/trading/manufacturing; conformal methods have not
  been applied to RF resource allocation. This supports the review's II-C gap.
  **BUT the threat is internal, not external:** C16 is status **R** — it exists only as
  `CDTController` exercised by ONE unit test (`test_cdt_keeps_mean_loss_near_alpha`), with no
  experiment measuring realised forgone-effect vs the α target over adversarial sequences, no
  conformal *set* layer (SAOCP/DtACI/CRC all unimplemented, §1.2), and no PSRO exploitability
  evaluation. So the single most novel claim is the least evidenced in code. A reviewer who
  diffs thesis↔repo finds the flagship is a ~30-line online-gradient controller plus a passing
  assertion, not the calibrated-decision result the lit review promises.

### 3.5 Summary of the sharpest reviewer attacks (priority order)

1. **Mamba-2 backbone is claimed but absent** (§1.2) — and a 2025 transformer-predicts-PRNG
   paper (§3.1) already does the in-context prediction the thesis motivates with Mamba-2.
2. **[15] Zhang–Wu TIFS 2025 cannot be verified** (§2.2) — the review's closest multi-tone/
   cooperative-jamming prior art may not exist as cited.
3. **BLE/Bluetooth hop-prediction attacks** (§3.3) are the obvious application precedent and
   are entirely missing.
4. **Flagship CDT claim is under-implemented** (§3.4) and its whole conformal-set + PSRO
   scaffolding is paper-only (§1.2).
5. **"No open FH-EW environment" is bounded** by the Turing TSRD benchmark (§3.2) unless
   scoped to FH; and the simulator's grounding is entirely "assumed," contradicting
   "calibrated against public datasets."
6. **Placeholder citations [48]–[50]** and the **[1] misattribution** are easy credibility hits.

---

## Gaps (what I could not resolve)

- Could not reproduce any exp1–exp18 numbers (results JSON git-ignored; sims not re-run here).
  All headline numbers are "as recorded by the author," evidenced only by committed PNGs/prose.
- Could not confirm existence of [15] Zhang–Wu TIFS 2025, [24] Kim TAES 2025, [29] Deng
  VTC2025-Fall, or the placeholder [48]–[50] against primary records.
- Could not find a confirmed open, FH-sequence-specific public benchmark/dataset (absence is
  suggestive, not proven).
