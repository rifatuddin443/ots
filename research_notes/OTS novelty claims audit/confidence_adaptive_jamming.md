# Confidence-Adaptive Multi-Channel Jamming / Allocation — Novelty Audit

Scope: three candidate novelties in the OTS/fhew codebase, checked against published
electronic-warfare and anti-jamming literature up to 2026. Verdicts are deliberately
conservative (the goal is to avoid over-claiming). Candidates:

1. `tones_from_confidence(kappa, max_tones)` — number of simultaneous jamming tones set
   linearly from a predictability score `kappa = 1 - H(p)/log M` of the next-hop
   predictive distribution (sharp → 1 spot tone, flat → spread up to an operator cap).
2. `greedy_knapsack` allocation of jamming power over channels from the predictive
   distribution, with a cap on transmit frequencies; `AllocationController` predictive mode.
3. The overall loop: passive identification → probabilistic next-hop prediction →
   confidence-adaptive spot/partial-band jamming ("predictive, not follower/reactive,
   jamming whose spread adapts to calibrated uncertainty").

Code reviewed (not modified): `fhew/allocator.py` (functions `predictability`,
`tones_from_confidence`, `greedy_knapsack`, `water_filling`, `spot`, `reactive_follower`,
`CDTController`, `AllocationController`); `docs/CHANGELOG.md` 2026-10-30 entries;
`docs/Thesis_Architecture_Revision_3.1_prediction_first.md` §7, §13.

---

## Candidate 1: number of jamming tones set from prediction confidence (`tones_from_confidence`)

### Takeaway
I found no paper that sets the *number of simultaneous jamming tones* as an explicit
function of the predictive distribution's entropy/confidence. The closest prior art maps
prediction *probability* to per-channel attack inclusion or to power, not to a tone count,
and the classic "optimal number of tones" literature derives the count from a fixed J/S
budget, not from predictor uncertainty. Verdict: **NOVEL-IN-COMBINATION-OR-APPLICATION**
(the specific `kappa → tone-count` mapping appears novel; the underlying idea "hedge wider
when less sure" is not).

### Cited Findings
- Classic FH/MFSK multitone-jamming theory derives the optimum *number of tones* from a
  fixed jamming-power / J-S budget and the modulation, with at most one tone per FH band,
  tones of equal power — not from any predictor's uncertainty. Houston's partial-band tone
  jammers are described as optimal among energy-restricted threats against orthodox MFSK;
  Levitt (JPL, 1985) analyses n-tone bands with 1 ≤ n < M and finds the n=1 band-multitone
  jammer the best nonadaptive ECM strategy against FH/MFSK with optimum diversity for K>2 —
  [Levitt JPL report, NTRS](https://ntrs.nasa.gov/api/citations/19850015886/downloads/19850015886.pdf); Levitt, "FH/MFSK performance in multitone jamming," IEEE JSAC 3(5), 1985, DOI 10.1109/JSAC.1985.1146253 (abstract verified via search; full text not fetched).
- The idea that multiple tones beat a single tone against FH is standard textbook material,
  and multitone jamming is modelled with a per-frequency, per-time power-allocation term —
  but driven by a power budget, not predictor confidence — [MSU "Modeling and Detection of Hostile Jamming in Spread Spectrum Systems"](https://www.egr.msu.edu/bawc/publications/Conference%20Papers/Modeling%20and%20Detection%20of%20Hostile%20Jamming%20in%20Spread%20Spectrum%20Systems.pdf).
- The nearest confidence-to-action mapping in a *predictive jammer*: Chen & Zhou, "Model-
  Adaptive and Risk-Constrained Frequency Hopping Against Predictive Jammers," arXiv:2609.06514
  (Sep 2026) — the jammer's predictor q_t ∈ Δ_K is turned into per-channel inclusion
  probabilities h_t(k)=min{1, λ_t q_t(k)} with λ_t set so they sum to a **fixed** budget M
  (default M=1), then dependent-rounded to a jam set. Inclusion is proportional to predicted
  preference, but the *number* of jammed channels M is fixed, not a function of entropy —
  [arXiv:2609.06514 HTML](https://arxiv.org/html/2609.06514).
- Erpek, Sagduyu, Shi, "Deep Learning for Launching and Mitigating Wireless Jamming Attacks,"
  IEEE TCCN / arXiv:1807.02567 (2018): "The deep learning classification scores are used by
  the jammer for power control subject to an average power constraint." This scales *power*
  with a prediction score, not tone *count*, and is single-target (jam the predicted
  successful transmission) — [arXiv:1807.02567](https://arxiv.org/abs/1807.02567).
- Feng et al., "Fight Against Smart Communication Rival," IEEE WCL 2022: multiple DL
  classifiers predict channel status and *selectively* choose the jamming channel; jamming
  *power* optimised by a multi-armed bandit under an energy constraint — selection and power,
  not an uncertainty-sized tone count — [Consensus record](https://consensus.app/papers/details/ec1222661dd1575ba1606616f12c27c2/?utm_source=claude_desktop).

### Inferences
- The mechanism's *spirit* (spread jamming energy wider when the next channel is less
  certain, concentrate when certain) is implicit in partial-band-vs-spot theory and in
  proportional-inclusion schemes like arXiv:2609.06514. What appears genuinely unoccupied is
  the *explicit, closed-form* control `n = round(min + (max-min)(1-kappa))` with `kappa` the
  normalised entropy of a *calibrated* predictive vector. I found no prior work stating this
  rule.
- Because the rule itself is a simple linear map, the defensible novelty is the *coupling*
  (confidence score → tone count within an operator cap), not the map as an algorithm.

### Gaps
- Could not fetch full text of Houston (IEEE Trans. Commun. ~1975) or Levitt 1985 to confirm
  whether either ever indexes tone count to an *estimated* probability vector (abstracts
  strongly suggest not: both use a power budget).
- Non-English / classified EW literature not searchable; frame as "to our knowledge."

---

## Candidate 2: greedy threshold-aware knapsack power allocation from the predictive distribution, with a tone cap

### Takeaway
Jamming power allocation across channels is a well-established problem, but it is almost
always solved as a game-equilibrium, RL policy, or water-filling / partial-band-bandwidth
problem. A *greedy threshold-aware knapsack* (rank channels by `p_i / c_i`, fund to a J/S
threshold τ until budget exhausts, capped at `max_tones` generators) driven by a *predicted*
hop distribution is an uncommon, arguably novel, formulation — but "knapsack-style jamming
resource allocation" and "allocate proportional to predicted channel probability" each have
precedents. Verdict: **NOVEL-IN-COMBINATION-OR-APPLICATION**.

### Cited Findings
- Rao et al. explicitly frame jamming resource allocation against FHSS as a high-dimensional
  **combinatorial optimization / NP-hard** problem, solved by DRL (A3C+LSTM; TRPO/meta-TRPO),
  allocating partial-band noise center frequency and bandwidth across jamming nodes — the
  recognised "closest prior art" on the jammer side, but bandwidth/node allocation via RL,
  not a threshold knapsack over a predictive vector — [Rao, IEEE Sensors J. 2024](https://consensus.app/papers/details/b873d077d3cd5c9ba1474ab27dac29a1/?utm_source=claude_desktop); [Rao, IEEE TAES 2024](https://consensus.app/papers/details/b00453860a0d52269865574245d56ba7/?utm_source=claude_desktop).
- Peng et al., "Dynamic multi-target jamming channel allocation and power decision-making,"
  China Communications 2025: multi-agent DRL (FBDDQN) for joint jamming *channel allocation
  and power* — the allocation is an RL policy, not a knapsack, and not driven by a calibrated
  next-hop distribution — [Consensus record](https://consensus.app/papers/details/51ccc33f752c5f4caf8e05a9b6a4fd45/?utm_source=claude_desktop).
- Zhang & Wu, "Cooperative Jamming Over DRL-Based FH… One-Leader Multi-Follower Stackelberg,"
  IEEE TIFS 2025: multi-jammer frequency+power coordination as a potential game; compares
  against a "multi-tone intelligent jamming method." Multi-tone operation and power allocation
  exist here, but as a game equilibrium, not a prediction-driven knapsack — [Consensus record](https://consensus.app/papers/details/746e4296fa995ff799f5d974772f7f89/?utm_source=claude_desktop).
- Power allocation over parallel channels in jamming/eavesdropping settings is typically
  **water-filling**-style (concave objective), not a threshold knapsack — e.g. Xu et al.,
  "Power Allocation for Cooperative Jamming… Over Parallel Channels," IEEE TIFS 2023 — [Consensus record](https://consensus.app/papers/details/c2e98d55ab70571182dfe1d320b85dc7/?utm_source=claude_desktop). The thesis itself positions plain/weighted water-filling as the baseline the knapsack beats (`fhew/allocator.py::water_filling`, thesis §7.3).
- The threshold-effect premise (below a J/S threshold a channel is not denied, power beyond
  it is wasted) is the standard J/S / denial model underlying partial-band and tone-jamming
  optimisation (Houston/Levitt above; Torrieri, *Principles of Spread-Spectrum Communication
  Systems*, and Poisel, *Modern Communications Jamming*, are the textbook anchors — not
  independently fetched here).

### Inferences
- Casting the per-slot allocation as a 0/1-ish knapsack over a *threshold cost* `c_i = τ/gain_i`
  with value density `p_i/c_i` is a clean reframing that I did not find stated elsewhere for
  FH jamming; prior combinatorial treatments (Rao) go straight to RL rather than exposing a
  greedy knapsack with an optimality reference (MILP/DP).
- The `max_tones` cap as a hardware tone-generator limit is a sensible engineering constraint
  but not itself a research contribution; its novelty rides on being *coupled* to `kappa`
  (Candidate 1).
- Defensible wording: "a threshold-aware greedy knapsack allocation of jamming power over
  channels, ranked by predicted-probability-per-unit-denial-cost, with a transmit-frequency
  cap" — claim as a *formulation* novel to our knowledge, not as the first multi-channel
  jamming allocator.

### Gaps
- Did not locate any paper using the exact "knapsack" framing for FH jamming power
  allocation; absence of evidence, not proven absence. A targeted IEEE Xplore full-text
  search for "knapsack" + "jamming" + "frequency hopping" would strengthen this.
- Torrieri and Poisel textbooks not directly consulted; they should be checked for any
  threshold-budget tone-count argument that could anticipate the knapsack.

---

## Candidate 3: the overall loop — passive ID → probabilistic next-hop prediction → confidence-adaptive spot/partial-band jamming (predictive, not reactive)

### Takeaway
Each ingredient exists in isolation (hop-sequence prediction; predictive/learning jammers;
partial-band vs spot trade-off; uncertainty-aware allocation), but I found no system that
chains *passive self-identification of the hop generator* → *calibrated probabilistic next-hop
prediction* → *uncertainty-adaptive multi-tone allocation* as one pipeline. The author's own
framing ("coupling of calibrated prediction to a threshold-aware budgeted allocation") is the
honest level of claim. Verdict: **NOVEL-IN-COMBINATION** (strongest as a combination/application
claim; weak as any single-component "first").

### Cited Findings
- Predicting a hopper's next channel is established: Strickland et al., "Predicting Jamming
  Systems Frequency Hopping Sequences Using Artificial Neural Networks," IEEE SoutheastCon
  2023 (reverse-engineers LFSR hoppers; reports driving jamming probability toward 0 in
  simulation) — [Georgia Southern thesis PDF](https://digitalcommons.georgiasouthern.edu/cgi/viewcontent.cgi?article=3805&context=etd); plus LSTM/CNN-GRU/TCN/HMM predictors (Li 2020/2021; Zhu 2023; Deng 2022/2025; Cao transformer 2025). These emit point predictions / hard classifications, fit per sequence, assuming clean consecutive outputs — not calibrated distributions feeding an allocator.
- Predictive/learning jammers that jam the *predicted* channel exist: FOLPETTI (Bout,
  Brighente, Conti, Loscri, arXiv:2205.03915, 2022) — a MAB attack that follows/predicts the
  victim's channel in real time — [arXiv:2205.03915](https://arxiv.org/abs/2205.03915v1); Erpek/Sagduyu/Shi 2018 (DL classifier predicts successful transmissions, jams them, score-based power control) — [arXiv:1807.02567](https://arxiv.org/abs/1807.02567); Jamming Bandits (Amuru, Tekin, van der Schaar, Buehrer, IEEE TWC 2016) — learns optimal PHY jamming parameters, sublinear regret — [arXiv:1411.3652](https://arxiv.org/pdf/1411.3652); Chen & Zhou arXiv:2609.06514 (2026) formalises a predictive jammer with proportional inclusion. None self-identify the generator first, and none size the attack from predictive *entropy*.
- The spot-vs-barrage / partial-band trade-off and follower (reactive/repeater) jamming are
  classic (Torrieri; Poisel; Simon/Omura/Scholtz/Levitt *Spread Spectrum Communications
  Handbook*). The codebase keeps the reactive/follower path (`reactive_follower`) as the
  feasibility floor and explicitly makes spot-vs-barrage emerge from the allocator rather than
  being an operator mode (thesis §7.2). That "emergent, not mode-switched" framing is a
  presentation novelty, not an algorithmic first.
- Uncertainty-aware adversarial RF resource allocation is emerging but defender-side and
  game-theoretic, e.g. Tang et al., "Uncertainty-Aware Jamming Mitigation With Active RIS: A
  Robust Stackelberg Game," IEEE TIFS 2026 — [Consensus record](https://consensus.app/papers/details/2a21c150cea95c189eb1e402966e93d6/?utm_source=claude_desktop). Here "uncertainty" is channel-state uncertainty, not predictor entropy driving jammer spread.
- Reactive jamming as a recognised class (Wilhelm et al., "Short Paper: Reactive Jamming in
  Wireless Networks — How Realistic is the Threat?", ACM WiSec 2011) anchors the
  reactive/follower baseline the system contrasts itself against (cited widely; not
  independently fetched this session).

### Inferences
- The combination claim is defensible precisely because the surveyed jammer-side works split
  into two camps: (a) predictors that output point hop guesses with no principled allocation,
  and (b) RL/game jammers that allocate power/bandwidth with no calibrated predictive
  distribution and no self-identification stage. The OTS pipeline sits at the seam.
- The strongest, safest claim wording: "To our knowledge, no prior system couples a
  calibrated probabilistic next-hop predictor to an uncertainty-adaptive, threshold-aware
  multi-channel jamming allocation; prior predictive jammers either jam a single/fixed-size
  predicted set or allocate power without quantified predictive uncertainty." Avoid any
  unqualified "first predictive jammer" or "first multi-tone jammer" phrasing — both are
  anticipated.

### Gaps
- Could not verify whether any defence/classified program (e.g. MILCOM, DTIC-restricted)
  already fields an ID→predict→adapt-spread loop; DTIC and classified venues are out of reach.
- The distribution-free / conformal risk-control coupling (CDTController, the thesis's
  flagship) was out of this slice's scope and should be audited separately; the author's own
  §13 already flags conformal-for-jamming as the lead novelty and here I only note that the
  allocation side's novelty is combinational, not standalone.

---

## Summary of verdicts

| Candidate | Verdict | Closest prior art | Defensible claim wording |
|---|---|---|---|
| 1. Tone count from confidence `kappa` | NOVEL-IN-COMBINATION/APPLICATION | arXiv:2609.06514 (proportional inclusion, fixed budget); Erpek 2018 (score→power) | "number of simultaneous tones set from the normalised entropy of a calibrated next-hop distribution — novel to our knowledge; the spread-when-unsure principle is classical" |
| 2. Threshold-aware greedy knapsack from predictive vector, tone cap | NOVEL-IN-COMBINATION/APPLICATION | Rao 2024 (DRL combinatorial JRA); water-filling allocators | "a threshold-aware greedy knapsack ranked by predicted-prob-per-denial-cost with a transmit-freq cap — a novel formulation, not the first multi-channel jamming allocator" |
| 3. ID → probabilistic predict → confidence-adaptive spot/partial-band loop | NOVEL-IN-COMBINATION | Strickland 2023 (predict); FOLPETTI 2022, Jamming Bandits 2016, Erpek 2018, Chen&Zhou 2026 (predictive jammers) | "to our knowledge no prior system couples calibrated probabilistic next-hop prediction to uncertainty-adaptive threshold-aware multi-channel allocation" |

Overall: none of the three is safely claimable as an outright algorithmic first. All three
are defensible as **novel in combination / application**, matching the thesis's own "to our
knowledge" posture (§13). The single most exposed over-claim to avoid is any wording implying
the *idea* of predictive or multi-tone FH jamming is new — it is not. The genuinely
unoccupied seam is the explicit **confidence(entropy)→tone-count→threshold-knapsack** chain
fed by a *calibrated* predictive distribution.

### Full-text items not verified this session (flag before relying on)
- Houston (IEEE Trans. Commun., ~1975) and Levitt (IEEE JSAC 3(5), 1985) — abstracts only.
- Torrieri *Principles of Spread-Spectrum Communication Systems*; Poisel *Modern
  Communications Jamming*; Simon/Omura/Scholtz/Levitt *Spread Spectrum Communications
  Handbook* — textbook anchors, cited from general knowledge, not fetched.
- Wilhelm et al., WiSec 2011 (reactive jamming) — cited from general knowledge, not fetched.
- Erpek 2018 threshold/power-scaling exact rule — abstract-level only (full PDF not read).
