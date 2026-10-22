# Thesis Architecture — Revision 3.1 (Confidence-Adaptive Multi-Channel EW)

Oct 21, 2026 · @Ehsan · *Revision 3.1 (prediction-first correction), Oct 2026*

## Changes in this revision (3.1 — prediction-first)

This revision corrects the document so it matches the built-and-tested system,
whose **primary achievement metric is next-channel prediction accuracy**, not
generator identification. Identifying ("cracking") the generator is only the fast
path that sharpens prediction when a *known* mechanism is present; the system is
scored on how accurately it predicts the **next channel** the emitter hops to,
and it must keep predicting when a **new algorithm outside the limited bank** is
used. What changed, and where:

| # | Change | Where it lands |
| --- | --- | --- |
| 1 | **Next-channel prediction is the headline metric** (top-1 accuracy), with identification demoted to a supporting/ fast-path role. | §1, §4, §12 |
| 2 | The **Fixed-Share expert mixture is realised as the next-channel predictor** (`fhew.experts.FixedShareMixture` / `fhew.predict.ChannelPredictor`): experts are weighted by *recent* predictive skill with a share-floor, and the mixed distribution is the system's next-hop output. | §4.2 |
| 3 | The mixture **always carries a zero-shot universal predictor** (a gap-robust variable-order context-mixing model, plus a phase-space predictor and the barrage floor), so prediction survives a **limited bank**: on a generator no expert targets, the universal expert carries the prediction instead of collapsing to uniform. | §4.2, §5 |
| 4 | The universal predictor is made **gap-robust**: its context is formed only from a *contiguous run of clean single-detection slots*, so missed hops and false alarms never corrupt the order-`k` rule it learns under a lossy sensor. | §4.2, §5 |
| 5 | Added three **held-out "new algorithm" generators** (`lagged_fib`, `quadratic_cg`, `sine_map`) that no expert targets, to measure prediction generalisation. | §11 |
| 6 | Added the prediction benchmark **`exp14`**: next-channel top-1 for the proposed predictor vs an identification-only bank vs online-Markov vs uniform, across in-bank and new algorithms, plus a detection-probability robustness sweep. | §12 |

**Key measured result.** On `lagged_fib` — a new generator whose next channel
depends on the last *two* channels, which no bank expert targets and which a
first-order Markov expert cannot model — the universal predictor reaches **0.88
top-1** vs **0.42** for an identification-only bank (`P_d = 0.95`), and stays far
above it across the whole `P_d` sweep. Known generators are predicted at ~100%,
and a keyed CSPRNG correctly falls to chance (~1/16). The rest of the
architecture (Layers 0, 3, 4, 5; calibration; allocation; game frame) is
unchanged by this revision.

## Summary

Revision 3 keeps the same idea: read how predictable a frequency-hopping
adversary is right now, and spend a fixed power budget to match it. The
system concentrates power when it can predict and spreads it when it
cannot, from the first slot of contact. It is software-only and
simulation-only, and the encoder is still Mamba-2.

What changes is the machinery in Layers 0, 2, 3 and 4. Each change
replaces a part whose assumptions the setting breaks, or one a reviewer
would challenge, with a better-founded alternative.

**Flagship contribution (restated).** An online, adversarially valid
calibration loop tunes a single conservatism knob λ. That knob drives a
threshold-aware knapsack allocation of jamming power. The guarantee is
that the long-run average jamming effect forgone stays at or below α,
for any hop sequence. The loss and the allocator share one effect model.

**Primary achievement metric (restated for 3.1): next-channel prediction.**
Everything downstream — the power allocation and its guarantee — is only as good
as the predicted next-channel distribution it is fed. The system is therefore
scored first on **how accurately it predicts the next channel** (top-1 accuracy),
and this must hold even when the emitter uses an algorithm the recce bank has no
expert for. Identifying ("cracking") a known generator is the *fast path* that
makes the prediction a near-point-mass; it is not the goal. Because the bank is
necessarily limited, the predictor is a **Fixed-Share expert mixture that always
carries a zero-shot universal predictor** (§4.2): on a new, unseen algorithm the
universal expert carries the prediction, so the next-channel distribution stays
sharp instead of collapsing to the uniform floor, while a keyed CSPRNG correctly
pins it at chance.

### What changed from the merged architecture

  -------------------------------------------------------------------------------
  Area         Merged          Revision 3                      Why
               architecture                                    
  ------------ --------------- ------------------------------- ------------------
  Layer 0      GNN over static Track-aware GNN with a Kalman   A moving drone's
  sorting      per-hop         tracker; drift-robust features  bearing and power
               features        (timing, RF fingerprint)        drift, so static
                               weighted up; moving emitters in similarity breaks
                               the gym                         

  Hypothesis   Sequential      Bayesian posterior over the     SPRT is a
  bank (recce) probability     bank, MSPRT stopping,           two-hypothesis
               ratio test      sensor-aware likelihoods (HMM   test; the bank has
               (SPRT)          forward pass), mandatory        many
                               none-of-the-above, optional     
                               e-process stopping              

  Prediction   Split           SAOCP (main) or DtACI, online,  Split conformal
  set          conformal + ACI over an APS score               needs
               with hand-tuned                                 exchangeability;
               step                                            adaptive hoppers
                                                               break it

  Risk         Batch conformal Conformal Decision Theory (CDT) Gives E[loss] ≤
  guarantee    risk control    controller on λ, the            α on average for
                               allocator's conservatism knob  any sequence, and
                                                               calibrates the
                                                               decision itself

  Change       BOCPD resets    BOCPD on the score and loss     BOCPD's
  detection    the calibration stream, plus a conformal        within-regime
               window          e-detector; alarms reset expert i.i.d. assumption
                               weights and boost the           fails; adaptive
                               calibrator, not wipe it         calibrators
                                                               self-recover

  Allocator    Water-filling   Greedy threshold-aware knapsack Jamming is a
               (main), MLP,    (main), exact MILP/DP as        threshold effect:
               diffusion       reference; water-filling and    a channel is
                               weighted water-filling as       denied only above
                               baselines; MLP only as a fast   a J/S threshold
                               approximation; diffusion        
                               dropped                         

  Baselines    Ablations only  Adds fixed spot, fixed barrage, Shows what the
                               softmax-threshold heuristic,    guarantee buys
                               split CP + batch CRC, and an    
                               end-to-end PPO / CVaR-PPO       
                               jammer                          
  -------------------------------------------------------------------------------

Everything else (Layer 1, the expert bank, probing, the game frame, the
gym, the feasibility map) carries over with small edits noted in place.

## 1. Problem and constraints

**Thesis question.** Can one software system deliver useful
electronic-attack effect against an adversary whose hop-generation
mechanism is unknown? It must do so inside an engagement window too
short to learn that mechanism from scratch, and degrade gracefully, by
design, when the mechanism is unbreakable.

The system is an offensive cognitive-EW node studied entirely in
simulation. A modelled passive SIGINT sensor observes one or more
unknown frequency-hopping emitters. A characterization stage estimates
how predictable each one currently is. A calibrated allocator splits a
fixed power budget across the likely channels. There is no hopper of our
own and no hardware.

  --------------------------------------------------------------------------
  Constraint        What it rules out        What it forces
  ----------------- ------------------------ -------------------------------
  Software-only     SDR testbed,             The environment and models are
                    over-the-air validation, the deliverable; the gym
                    hardware claims          becomes a contribution

  Adversaries span  Any single fixed         Self-identification of the
  deterministic to  exploitation strategy;   regime, and calibration valid
  adaptive          guarantees that assume   for arbitrary sequences
                    exchangeable data        

  Multi-frequency   Single-channel action    A power vector over channels as
  emission under a  spaces                   the one action, allocated power
  total power                                summing to at most P_J
  budget                                     

  Short engagement  Convergence-time         Anytime operation, offline
  window            metrics, methods needing pre-training, reconnaissance
                    thousands of slots       priors
  --------------------------------------------------------------------------

The fourth constraint reshapes the thesis most. The system is judged on
effect delivered per hop observed, not on performance at convergence.

**The unifying idea.** A calibrated prediction drives a budgeted,
threshold-aware power allocation. When probability mass is concentrated,
the budget fully denies a few channels (spot). When it is spread, the
budget covers as many channels as it can at the denial threshold
(barrage-like). Spot versus barrage is not an operator-selected mode. It
falls out of the predictive distribution and one calibrated knob λ.

## 2. System architecture

The spine runs from the simulated adversary through the passive sensor
to multi-tone emission. Characterization in the middle produces the
predictive distribution and the predictability score κ that every later
stage reads. The shaded layers (L0, L2, L3, L4) are the ones Revision 3
changes; L1 and L5 carry over.

![Revision 3 spine · 6 layers, 2 side-loops, changed layers
shaded](media/image1.png){width="6.0in" height="3.5in"}

Revision 3 spine · 6 layers, 2 side-loops, changed layers shaded

Hops flow down the spine. On the right, reconnaissance priors warm Layer
2 and the barrage floor before first contact, and the Layer 5 probe is a
side-loop that injects a causal adaptivity feature. Dashed tiers are
severable, each with a named fallback, so Tiers 1 to 3 stand without
them.

## 3. Layers 0 and 1: sensing, emitter sorting and the Mamba-2 encoder

The sensor model decides what is observable. The sorter splits
interleaved emitters into tracks, and a Mamba-2 encoder compresses each
track's recent window into one shared latent state z.

### 3.1 Sensor model

The sensor is modelled, not simulated at IQ level. Its output per
detection is time, channel index, received power and bearing, with gaps,
timing jitter and false detections.

  -------------------------------------------------------------------------
  Parameter      Meaning                 Why it matters downstream
  -------------- ----------------------- ----------------------------------
  P_d            Probability a hop is    Below 1 the sequence has gaps;
                 detected                algebraic recovery and naive
                                         likelihoods break

  P_fa           False-alarm rate per    Phantom hops corrupt recovery and
                 channel per slot        the hypothesis test

  Reaction       Detect-to-emit, end to  Sets the prediction horizon
  latency        end                     

  Sensing lead   Observation before the  Free observation; removes the cold
  time           emitter is in range     start

  Bearing error  Angular uncertainty     Affects emitter association

  Frequency      How much of the band is A partial-band receiver lowers
  resolution and watched                 effective P_d
  span                                   

  Look-through   Whether the sensor is   Decides whether feedback to the
  blanking (new) blind while the node    calibrator is full or intermittent
                 emits                   (Section 6.5)
  -------------------------------------------------------------------------

P_d and P_fa are used twice: to generate observations, and inside every
likelihood (Section 4.4), so the inference matches the sensor.

### 3.2 Layer 0: track-aware emitter sorting (optional, severable)

With several emitters present, a single channel stream is ill-posed, so
hops must be assigned to emitters first. Revision 3 treats each emitter
as a moving source whose features drift smoothly, not as a fixed point
in feature space.

-   **Graph.** Each detected hop is a node with features time, channel,
    power, bearing and RF fingerprint (carrier offset, rise time). A
    graph-attention network (GAT) scores edges meaning "same emitter".

-   **Local edges only.** Edges connect hops close in time. Association
    is a chain of short links, so slow drift over seconds does not break
    it.

-   **Kalman tracker.** A Kalman filter per track predicts where the
    next hop's bearing and power should be. The GAT scores a hop
    against that prediction, not against the track's history. This is
    standard multi-target tracking.

-   **Drift-robust features weighted up.** Hop timing (dwell, hop rate)
    and RF fingerprint stay stable as a drone moves; bearing and power
    do not.

-   **Moving emitters in the gym.** Simulated drones fly trajectories,
    including crossing paths. Training on static emitters would
    overstate performance.

-   **Single emitter.** The sorter is an identity pass-through, so it
    never hurts the base case.

-   **Baselines.** DBSCAN, hierarchical clustering and HDW, plus a
    tracker-only (no GNN) baseline.

-   **Hard cases reported.** Crossing tracks (identity swaps), fast
    close emitters, deep fading, and decoys (Section 11).

### 3.3 Layer 1: the Mamba-2 encoder

The encoder reads per-channel energy over the last N slots and a coarse
spectrogram patch. Its core is a Mamba-2 selective state-space model,
which gives linear-time inference on long, noisy hop histories under a
per-slot latency budget.

-   **Pretraining.** Masked-hop modelling across the whole adversary
    library, offline. Inference only at engagement time.

-   **One shared state.** The mixture, the zero-shot expert, the
    open-set score and the probe all read the same latent z.

-   **Ablation.** Mamba-2 against CNN-GRU against a Transformer, on
    next-hop log-loss and per-slot latency.

-   **Fallback.** CNN-GRU if a usable Mamba-2 implementation is not
    available in time. Nothing else depends on the choice.

## 4. Layer 2: adversary characterization

The system never classifies the adversary and commits. Two mechanisms
run side by side and produce one predictive distribution. A Fixed-Share
mixture weights every expert by recent predictive skill. A Bayesian
hypothesis bank tracks a posterior over known generators from
reconnaissance. Their outputs are combined into p_t\^mix, and the
predictability score κ is read off it.

### 4.1 Why not a hard router

A hard router must be right to be useful. It fails when it
misclassifies, when the generator is a hybrid, and when the adversary
changes behaviour mid-engagement.

### 4.2 Fixed-Share expert mixture

Each expert emits a distribution over the next channel. Weights update
multiplicatively on log-loss:

w_{t+1}(i) \\propto w_t(i)\\, \\exp\\!\\big(-\\eta\\, \\ell_t(i)\\big),
\\qquad \\ell_t(i) = -\\log p_i\\big(c_t \\mid \\mathcal{H}_{t-1}\\big)

Fixed-Share adds a small weight floor so the mixture recovers when the
regime switches.

**This mixture is the next-channel predictor (3.1).** It is implemented as
`fhew.experts.FixedShareMixture`, wrapped by `fhew.predict.ChannelPredictor`, and
its weighted output `p̂_t(c) = Σ_i w_t(i) p_i(c)` *is* the system's next-channel
distribution — the headline deliverable. Under the lossy sensor the per-slot loss
is the sensor-aware predictive log-likelihood of the *observation* (not of the
hidden channel), which keeps information parity; the per-slot constant the reduced
emission drops is identical across experts and cancels in the weight update.
Because the mixture weights by *recent* skill with a share-floor, it tracks
whichever expert predicts best right now and recovers within a logarithmic number
of steps after a regime switch — the Bayesian bank (§4.3), by contrast, weights by
*cumulative* likelihood and can only predict well once an in-bank hypothesis wins.

**The universal predictor makes a limited bank enough (3.1).** The mixture always
includes, alongside the identification experts, a **zero-shot universal
predictor** — a gap-robust variable-order (context-mixing) model that recovers
in-context regularity of *any* order — plus a phase-space predictor for unseen
chaotic dynamics and the barrage floor. So when a **new algorithm the bank has no
expert for** is used, the universal expert accrues the weight and carries the
prediction; the next-channel distribution stays sharp rather than collapsing to
uniform. Gap-robustness is structural: the universal expert forms a context only
from a *contiguous run of clean single-detection slots*, so a missed hop or a
false-alarm slot breaks the run and it falls back to lower orders rather than
learning a context that straddles a gap, keeping its order-`k` tables uncorrupted
under detection loss. This is validated in `exp14` (§12): on the order-2
`lagged_fib`, prediction top-1 is **0.88** vs **0.42** for an identification-only
bank.

**Proposition to prove and report.** Against the best partition of the
engagement into k segments, Fixed-Share's cumulative log-loss exceeds
the best segmented expert by at most roughly

\\frac{k \\log N + k \\log (T/k)}{\\eta}

over T slots with N experts. Take the exact constants from the standard
proof. The matching experiment is empirical mixture regret per adversary
class.

### 4.3 Bayesian hypothesis bank with MSPRT stopping (replaces SPRT)

The classic SPRT is a two-hypothesis test, and the reconnaissance bank
holds many candidate generators. Revision 3 keeps a posterior over all
of them and updates it after every observation:

\\pi_t(k) \\propto \\pi_0(k)\\; P\\big(o_{1:t} \\mid H_k\\big), \\qquad
k \\in \\{1, \\dots, K, \\varnothing\\}

-   **Any number of hypotheses.** One update rule covers the whole bank.

-   **No forced early decision.** The system can act on the posterior at
    any time, which suits the short window.

-   **MSPRT stopping.** Declare identification when the top posterior
    passes a threshold (for example 0.95). This is the multi-hypothesis
    SPRT, which has known asymptotic optimality properties.

-   **Feeds the predictor directly.** The bank's next-hop distribution
    is the posterior-weighted mix of each generator's prediction. It
    enters the Fixed-Share mixture as one expert, so it is weighted by
    realised skill like any other.

### 4.4 Sensor-aware likelihoods

Each hypothesis H_k has hidden parameters (seed, register state, phase).
Likelihoods must handle them and the sensor's imperfections.

-   **Hidden state.** Marginalise over seeds or states when the space is
    small. For LFSR-type generators, use exact recovery
    (Berlekamp-Massey, about 2L clean outputs) to collapse the state,
    then score the remaining sequence.

-   **Missed and false detections.** Compute P(o_{1:t} | H_k) with a
    hidden Markov model whose hidden variable is the generator's true
    hop and whose emission model is the sensor (P_d, P_fa). The forward
    algorithm gives the likelihood with gaps and phantom hops handled
    exactly.

-   **Same sensor model as the gym.** The information-parity rule
    (Section 12.4) applies: no likelihood uses information the modelled
    sensor cannot deliver.

### 4.5 Mandatory none-of-the-above

The bank includes H_∅, an unknown-generator hypothesis with a flat or
learned likelihood (uniform, or a fitted Markov model). Without it, a
stale bank confidently picks the wrong generator. When H_∅ dominates,
the bank's expert collapses towards the live experts and the system
runs blind. A recce-stale condition (bank deliberately mismatched) is
reported to show performance degrades to blind, not below it.

### 4.6 Anytime-valid stopping (optional, for rigor)

Replace the fixed posterior threshold with an e-process: a
likelihood-ratio martingale for each candidate against the alternatives.
Ville's inequality gives a false-identification bound that holds
whenever the system stops, which fits "stop when the window ends".
This matches the distribution-free spirit of Layer 3.

### 4.7 The predictability score

The four-way label is replaced by one continuous quantity:

\\kappa_t = 1 - \\frac{H\\big(p_t\^{\\text{mix}}\\big)}{\\log
|\\mathcal{C}|}

κ runs from 0 (uniform) to 1 (fully determined) and is estimated on
held-out slots. A keyed CSPRNG pins κ near zero however long the system
watches. That sustained reading, not a classifier, tells the system to
stop chasing structure.

### 4.8 Open-set handling and the demoted classifier

A distance or energy score in the encoder's embedding flags
out-of-distribution emitters. The flag suppresses the classifier prior
and raises the prior on H_∅. The classifier head gates nothing; it only
sets the mixture's initial weights for the first few hops.

## 5. The expert bank

Experts differ by what they assume and how many hops they need before
saying anything useful. Under a short window the hops-needed column
matters most. The only addition in Revision 3 is the hypothesis-bank
expert from Section 4.3.

**The bank is limited — the zero-shot expert is what predicts the rest (3.1).**
No finite bank can hold an expert for every hop-generation scheme, so the
**zero-shot universal predictor** row below is load-bearing for the prediction
metric: it is the only expert that predicts a generator *no other expert targets*.
In the built system it is a training-free, gap-robust variable-order (context-
mixing) model; because it recovers in-context regularity of any order, it predicts
new deterministic schemes (e.g. the additive `lagged_fib`, where it reaches 0.88
top-1 vs 0.42 for the rest of the bank) and keeps the next-channel distribution
sharp where every dedicated expert would sit at the uniform floor. Its failure
signature — collapse to uniform — is exactly the correct behaviour against an
unpredictable (keyed-CSPRNG) source.

  ------------------------------------------------------------------------------------------
  Expert             Target         Hops needed     Output on success    Signature on
                     structure                                           failure
  ------------------ -------------- --------------- -------------------- -------------------
  Hypothesis bank    Known          A handful       Posterior-weighted   Posterior moves to
  (new)              candidates                     next-hop             none-of-the-above
                     from recce                     distribution         

  Berlekamp-Massey   LFSR           About 2L,       Exact recurrence     Linear complexity
                                    consecutive and                      keeps climbing near
                                    clean                                n/2

  State-filtered BM  LFSR through   More than BM,   Register-state       Posterior stays
                     gaps or errors degrading with  posterior            diffuse
                                    P_d                                  

  LCG cryptanalysis  Linear         A small handful Modulus, multiplier, Lattice reduction
                     congruential                   increment            finds no consistent
                     generator                                           parameters

  Seed search        Weakly keyed   Few, given a    The seed, hence the  Keyspace exhausted
                     seed           bounded         future               
                                    keyspace                             

  Correlation attack Weakly         Moderate, grows Partial state        Correlation at
                     filtered LFSR  with filter                          chance
                                    degree                               

  Parametric chaos   Known chaotic  Tens            Map parameters       Model selection
  ID                 map family                                          rejects all

  Phase-space        Unknown        Tens            Short-horizon        Error reaches
  predictor          chaotic                        forecast             uniform within one
                     dynamics                                            step

  Markov / n-gram    Empirical      Tens            Transition           Flat transition
                     regularity,                    distribution         matrix
                     blacklist AFH                                       

  Zero-shot neural   Any learnable  Few, amortised  In-context           Collapses to
                     regularity,    in pretraining  distribution         uniform
                     gap-robust                                          

  Behaviour model    Reactive or    Hundreds, fewer Belief-conditioned   Predictions drift
  (optional TD-MPC2) learned policy with            distribution         as the policy
                                    meta-training                        adapts

  Prior-weighted     None assumed   Zero            Spread over the      Not applicable;
  barrage                                           channel plan         this is the floor
  ------------------------------------------------------------------------------------------

**Berlekamp-Massey** is the strongest and the most brittle expert. It
recovers an unobfuscated LFSR exactly, but missed detections and any
nonlinear filter between register and channel index defeat it. The
linear-complexity profile turns failure into information: it separates
"no linear structure" from "not enough data yet".

**Seed search** is the most realistic route to a genuine break. Keyspace
size is an explicit parameter, so results are graded by key strength.

**What no expert can do.** Against a properly keyed CSPRNG no expert
beats uniform. The correct behaviour is to detect it quickly (sustained
κ near 0) and spend the budget on the best available barrage.

## 6. Layer 3: online calibration and the decision controller

This layer carries the flagship contribution. The previous design used
split conformal prediction with Adaptive Conformal Inference (ACI), then
batch Conformal Risk Control (CRC). Both give finite-sample guarantees
only under exchangeability, and an adaptive hopper that reacts to
jamming makes the data non-exchangeable, so those guarantees are
formally void here. Revision 3 moves to online methods whose guarantees
hold for arbitrary, even adversarial, sequences.

### 6.1 Score and set

For each channel, a conformity score is computed from the mixture's
predictive distribution, using an APS or RAPS rule (accumulated
probability until the true channel is covered). A calibrated threshold
turns the scores into a set S_t of likely channels. The set is kept as a
diagnostic and for interpretability; the allocation does not depend on
the set alone (Section 7).

### 6.2 Online set calibration (replaces ACI)

Instead of one hand-tuned ACI step size, the threshold is tuned by a
strongly adaptive online learner:

-   **SAOCP (main):** Strongly Adaptive Online Conformal Prediction. It
    runs experts on dyadic time intervals and combines them, giving
    near-optimal strongly adaptive regret, so coverage is controlled on
    every window, not only on average. This matters because an adversary
    can concentrate misses right after a switch.

-   **DtACI (alternative):** runs several ACI step sizes in parallel and
    reweights them. It is a drop-in, principled version of "ACI with an
    online-tuned step size".

-   **Plain ACI (ablation):** kept to show the improvement.

### 6.3 Decision calibration (replaces batch CRC)

The quantity to control is jamming effect forgone, not miss-rate. A wide
low-confidence set that spreads power below the denial threshold on
every channel can score worse than a tight set that fully denies a few.
Revision 3 calibrates the decision directly with a Conformal Decision
Theory (CDT) controller on a single conservatism knob λ:

\\lambda_{t+1} = \\lambda_t + \\eta\\,\\big(L_t - \\alpha\\big)

Here L_t is the realised effect forgone in [0,1], α is the target, and
λ_t sets how conservative (spread) the allocation is. The guarantee is a
long-run average: the mean realised loss stays at or below α + O(1/T)
for any sequence. The step size η is itself self-tuned with a
DtACI-style grid. Spot versus barrage then emerges from λ_t and the
predictive distribution. This removes the set-to-allocation mismatch,
because a 90%-coverage set does not imply 90% of the jamming effect.

**The claim to defend.** Coupling an online-calibrated decision knob to
a budgeted, threshold-aware multi-tone allocation, with a
distribution-free bound on forgone effect, is the flagship novelty. The
first thing to rule out in the novelty check is any existing system that
already sizes a multi-tone jam from a calibrated set or decision.

### 6.4 Regime monitor (BOCPD demoted)

BOCPD assumed i.i.d. data within a regime, which a Markov or chaotic
hopper violates, and it was used to wipe the calibration window. In
Revision 3 it is an auxiliary signal, run on the score and loss stream
rather than raw hops:

-   **BOCPD (CUSUM as cheap fallback)** on residuals flags a likely
    switch.

-   **Conformal e-detector** gives a distribution-free alarm with
    average-run-length control on the same scores.

-   On an alarm, the system resets the Fixed-Share weights and boosts
    the calibrator's learning rate (restarts a decaying-step schedule).
    It does not discard calibration state, because the strongly adaptive
    calibrator already recovers within a logarithmic number of steps.

### 6.5 Feedback and exchangeability, stated honestly

-   **Intermittent feedback.** If look-through blanking hides the
    emitter while the node jams, the true hop is not always revealed.
    Use an intermittent-feedback online conformal variant and state the
    assumption.

-   **Exchangeability.** The online guarantees are long-run averages and
    need truthful feedback. They can hide bursts of misses, so empirical
    coverage and realised risk are reported per class on sliding
    windows, including where they fall short of nominal. A guarantee
    claimed where its assumptions fail is worse than a measured
    shortfall reported honestly.

### 6.6 Set width still responds to three things

Adversary predictability (low κ widens it), prediction horizon (more
look-ahead widens it), and evidence accumulated (wide early, narrowing
as experts resolve). The third gives the anytime behaviour: at first
contact the allocation barrages, and as hops accumulate it concentrates,
with no explicit mode switch.

**Fallbacks.** Fixed-coverage split conformal for the set, batch CRC for
the risk guarantee (shown as the failure case), a fixed α for the
controller.

## 7. Layer 4: threshold-aware power allocation

Jamming only works if enough power lands on the channel the hopper
actually uses. Below a J/S threshold the effect is essentially nothing,
and power beyond the threshold yields little. So the real question per
channel is whether it is worth paying the power needed to cross its
threshold. This makes the allocation a knapsack problem, not a
water-filling problem, and that is the main change in Layer 4.

### 7.1 The effect model

Let g map power on a channel to effect on it. The denial threshold is τ:
below τ, effect is near zero; at or above τ, the channel is effectively
denied. The conformal loss in Section 6.3 uses this same g and τ, so the
risk guarantee and the allocation agree. If they used different models,
the guarantee would not describe the allocator.

### 7.2 The greedy knapsack allocator (main method)

1.  For each candidate channel i, take the probability p_i that the
    hopper lands there, from the predictor.

2.  Compute the power c_i needed to cross τ on channel i (depends on
    geometry and the hopper's signal strength).

3.  Rank channels by value density p_i / c_i.

4.  Fully fund channels in that order until the budget P_J runs out;
    spend any remainder on the next channel partially.

The behaviour emerges automatically:

-   **Confident prediction.** A few channels hold most of the
    probability, and the budget denies them fully. That is spot jamming.

-   **Uncertain prediction.** Probability is spread, so the budget
    denies as many channels as it can afford at τ. That is barrage-like.

The conservatism knob λ from Section 6.3 enters here. It tunes how
aggressively low-probability channels are included, which is the
online-calibrated control of the spot-to-barrage balance.

### 7.3 Reference and baselines

  -----------------------------------------------------------------------
  Allocator                   Role
  --------------------------- -------------------------------------------
  Exact MILP or DP            Optimal reference; shows the greedy is
                              near-optimal

  Greedy knapsack             Main method

  Weighted water-filling      Baseline; concave-g allocator with recce
                              weights

  Plain water-filling         Baseline; concave-g, no weights

  Equal power across the set  Simple baseline

  Small MLP head              Optional; only as a fast approximation of
                              the knapsack, reported with its speedup and
                              its gap to the exact solution
  -----------------------------------------------------------------------

The diffusion-policy head is dropped. The problem is small enough that
an exact MILP or DP solution exists, so a reviewer would ask why a
diffusion sampler is needed.

### 7.4 Barrage floor with reconnaissance weighting

When the set is wide the floor is not uniform. A known channel plan,
measured channel-usage statistics and blacklisted channels shrink the
effective channel set and raise the floor. The barrage bound is the
budget divided by the effective channel count, scaled by these weights.
This is the one lever against an otherwise unpredictable adversary.

### 7.5 Reward in simulation

The reward is the power placed on the channel the adversary actually
used, passed through g. The simulator knows the true next channel, so
this is measured directly. Averaged over an engagement it is hit
probability weighted by power-on-target.

## 8. Layer 5: active probing as an information-gain test

Passive observation cannot separate a deterministic hopper from an
adaptive one that has so far behaved identically. Probing does something
to the adversary and watches the response, and it is the most original
component.

**The causal test.** The node emits a brief, low-power jam on a
predicted channel and watches whether the hop statistics change. An
open-loop hopper ignores it; a reactive one avoids the channel or shifts
pattern. Formally the probe is an intervention: compare the post-probe
hop distribution against the no-probe counterfactual. A significant
divergence is evidence of a closed control loop, added to the encoder's
embedding as a causal feature no passive classifier can obtain.

**The probe-or-not rule.** Probing is not free: it reveals presence,
spends budget, and trains a learning adversary. The probe value is
expected information gain minus an exposure cost:

V_{\\text{probe}}(x_t) = I\\big(A ; O \\mid x_t\\big) - \\lambda\\,
c(x_t)

-   **Context x:** the predictability score κ, the set width, and the
    time remaining.

-   **Actions:** probe or not, learned by a contextual bandit with the
    mutual-information reward.

-   **Learned shape:** probing is most worthwhile at mid-uncertainty,
    least at the extremes.

-   **Information link.** The same information-gain rule picks which
    hypothesis-bank candidates to separate (Section 4.3), so probing and
    identification share one objective.

-   **Fallback:** a fixed, sparse probe schedule, which is also the
    ablation baseline.

## 9. Adaptive adversaries: game frame and exploitability

The thesis frames the reactive-adversary interaction and measures the
system against it. It does not claim a tuned procedure for defeating a
reactive emitter. The deliverable is the game formulation and a
worst-case certificate.

**The correct frame.** When the adversary reacts, the next-hop
distribution depends on the system's own action. The passive-prediction
assumption of Layers 2 and 3 breaks, and the interaction becomes a
leader-follower (Stackelberg) game, equivalently a
performative-prediction problem. The online calibration in Layer 3
survives this because its guarantees hold for arbitrary sequences.

**Exploitability via Global PSRO.** The adaptive branch is a population
game, evaluated with Policy-Space Response Oracles.

-   Grow a population of adversary policies and system policies, each a
    best response to the other's meta-distribution.

-   Use Global PSRO, which minimises population exploitability and
    reaches a low-exploitability solution in fewer iterations than
    vanilla PSRO. This matters because each iteration trains a fresh RL
    hopper.

-   Report exploitability, how much a fresh best-response adversary can
    gain over the system's meta-strategy, as the worst-case
    certificate. A low, decreasing curve answers "have you only tested
    against adversaries you wrote?"

-   Fallbacks: vanilla PSRO, then a scripted best response. The tier can
    be cut entirely.

**Optional adaptive expert (TD-MPC2).** A severable TD-MPC2 latent world
model, meta-trained offline across learned hoppers, enters the bank as a
predictor only. The Fixed-Share mixture weights it like any other
expert, so it takes over only when it predicts well. A model-based
expert is preferred because the short window does not allow a model-free
predictor to adapt. The Markov expert and the mixture already route to a
reactive fallback, so the spine does not depend on it.

## 10. Reconnaissance priors, anytime operation and the feasibility map

These three make the short-engagement constraint a feature.
Reconnaissance turns blind recovery into a short multiple-choice test
(the hypothesis bank of Section 4.3), the system acts from the first
slot, and the feasibility map names which strategy is achievable in
which regime.

**Reconnaissance artifacts.** A hypothesis bank of candidate generators
with parameters; a channel plan (hop set, dwell, rate, guard channels)
that seeds the barrage floor; per-emitter RF fingerprints; seed and key
behaviour; and protocol structure (sync and beacon channels). The bank
is a prior to be overridden by live evidence, never ground truth. A
recce-stale condition is reported to show the system degrades to blind
performance, not below it.

**Anytime operation.** The system never waits for a confident
classification. The evidence-to-action ladder is not coded as stages; it
falls out of set width contracting as the mixture sharpens.

  --------------------------------------------------------------------------
  Evidence so far        Set width     Action
  ---------------------- ------------- -------------------------------------
  Zero hops              Maximal       Prior-weighted barrage over the recce
                                       channel plan

  A few hops             Contracting   Bank posterior narrows to surviving
                                       hypotheses

  Enough for the active  Narrow        Concentrate power on the recovered or
  expert                               predicted next-hop set

  κ pinned low           Maximal       Detect unpredictable adversary;
                                       commit to the weighted barrage floor
  --------------------------------------------------------------------------

**Feasibility map.** Plotting achievable strategy against window length
and hop rate partitions the plane into regions where exact recovery is
possible, where only short-horizon prediction is, and where only barrage
remains. It gives a principled answer to "what if the enemy comes
fast": it names the regime and shows the system selects the best
strategy available there. The headline it enables is hit-rate against
time-since-contact, one curve blind and one reconnaissance-primed; the
gap is the measured value of reconnaissance.

## 11. The FH-EW gym, adversary library and red-team adversaries

The software-only constraint makes the simulation environment a
contribution. No open Gymnasium environment for this problem exists, so
a credible one is itself a result.

**Environment.** Configurable sensor (P_d, P_fa, reaction latency, lead
time, bearing error, resolution, blanking); channels (count, hop rate,
dwell, guard channels); action (power vector with budget P_J and the
effectiveness function g with threshold τ); episodes with short,
randomly drawn windows and the adversary appearing at a random time.
Emitters follow moving trajectories (Section 3.2).

**Adversary library (authored by the thesis).** Deterministic (LFSR,
LCG, nonlinear-filtered LFSR); weakly keyed (time-derived, reused,
short-key seeds); chaotic (logistic, tent, Henon); statistical (Markov,
blacklist AFH); learned (DQN, PPO, including reactive variants);
cryptographic (keyed CSPRNG as the ceiling).

**Held-out "new algorithm" generators (3.1, no expert in the bank).** To measure
whether the system predicts a scheme it was *not* built for, three generators are
drawn as unseen adversaries with no dedicated expert: `lagged_fib`
(`c_t = c_{t-1} + c_{t-k} mod M`, an order-`k` additive rule a first-order Markov
expert cannot model), `quadratic_cg` (a nonlinear congruential generator), and
`sine_map` (a chaotic map outside the parametric-chaos-ID grid). They are the
truth side of `exp14`; only the universal predictor can predict their next
channel.

**Grounding.** Calibrated against public datasets (RFUAV for FHSS
parameters, drone-RF datasets for timing and occupancy) to set parameter
ranges, not to train on. State which parameters were calibrated and
which assumed, alongside every performance number.

**Red-team adversaries (held out, not authored by the solver).**

  -----------------------------------------------------------------------
  Red-team adversary      What it does            Component it stresses
  ----------------------- ----------------------- -----------------------
  Keyed-alternate evader  Reacts to sensed        Game frame (9) and
                          jamming by switching    probing (8)
                          among pre-agreed        
                          channels                

  Decoy emitter           Emits fake hops to draw Track-aware sorter
                          budget and confuse      (3.2) and open-set flag
                          association             (4.8)

  Cognition-masking       Acts slightly           Behaviour expert and
  hopper                  sub-optimally to hide   the adaptivity feature
                          its goal                

  Regime-switcher         Changes generator       Fixed-Share recovery
                          family mid-engagement   (4.2) and regime
                                                  monitor (6.4)

  Entropy-maximising      Spreads its own         Predictability
  hopper                  distribution to         boundary: should pin κ
                          minimise predictability low and force barrage

  Mobile crossing pair    Two drones whose tracks The Kalman tracker and
  (new)                   cross                   identity-swap handling
                                                  (3.2)
  -----------------------------------------------------------------------

The entropy-maximising hopper and the keyed CSPRNG are the adversaries
that should defeat prediction. Including them is how the thesis shows it
fails gracefully and by design, not silently.

## 12. Experiments, metrics and ablations

Evaluation rewards fast effect under short windows, not performance at
convergence.

**Primary metric (3.1): next-channel prediction accuracy.** The headline number
is **top-1 next-channel accuracy** — the fraction of slots where the single
most-probable channel is the one the emitter actually hops to — reported
whole-window and at steady state, with top-3 and next-hop log-loss alongside, and
measured on both in-bank families *and* the held-out new algorithms (§11). This is
the metric that captures the thesis goal directly: predict the next channel, for a
known *or* unknown generator. The experiment that reports it is **`exp14`**
(next-channel prediction): it compares the proposed predictor (Fixed-Share +
universal) against an identification-only bank, an online first-order Markov
predictor, and the uniform floor, with a detection-probability robustness sweep
over the new algorithms. Headline: known generators ≈100% top-1; the new order-2
`lagged_fib` **0.88 vs 0.42** for the identification-only bank; keyed CSPRNG at
chance.

**Headline results.** Next-channel top-1 accuracy, in-bank and for new algorithms
(`exp14`); hit-rate against adversary intelligence (the
system tracks the best per-class baseline across the range while any
fixed strategy fails at one end); hit-rate against time-since-contact
(blind versus recce-primed); exploitability against PSRO iteration.

**Metrics.** **Next-channel top-1 / top-3 / log-loss (primary);** cumulative
jammed hops within window T; time-to-lock per
class; hit-rate against hops observed; power-on-target efficiency;
exact-recovery rate and bits recovered; empirical conformal coverage and
realised risk, on sliding windows; predictability score per class;
mixture regret against the Section 4.2 bound; hops-to-identify, accuracy
and false-identification rate for the hypothesis bank; exploitability;
feasibility region.

**Ablations.** Soft mixture vs hard router; Mamba-2 vs CNN-GRU vs
Transformer; **zero-shot universal expert in/out — i.e. the proposed predictor vs
an identification-only bank on next-channel accuracy (`exp14`), the ablation that
isolates the limited-bank-vs-new-algorithm result;** SAOCP vs DtACI vs plain ACI vs
fixed-coverage conformal; CDT decision control vs batch CRC; greedy
knapsack vs exact MILP vs water-filling vs weighted water-filling vs
equal power; with/without probing, and info-gain vs fixed schedule;
with/without recce priors; Bayesian bank + MSPRT vs pairwise SPRT vs
fixed-length test vs an LSTM classifier; track-aware GNN vs DBSCAN vs
hierarchical vs HDW vs tracker-only; meta-trained TD-MPC2 vs cold world
model vs model-free; Global PSRO vs vanilla PSRO.

**Baselines beyond ablations.** Fixed spot, fixed barrage,
softmax-threshold heuristic, split CP + batch CRC (shows the
exchangeability failure), and an end-to-end PPO or CVaR-PPO jammer
(shows what the guarantee buys).

**The information-parity rule.** Because the simulator generates both
adversaries and experts, every expert and every likelihood gets only
what the modelled sensor delivers, never the true register length or
generator family. State this rule and audit each expert against it.
Recovery numbers obtained with illegitimate side-information are
worthless, and a careful reviewer probes exactly here.

## 13. Novelty and positioning

The flagship novelty is the coupling of online-calibrated,
decision-aware prediction to a threshold-aware budgeted allocation, with
a distribution-free bound. These are claims to check, not settled facts,
and a formal novelty check must precede any "first to" wording. A
literature scan found no prior work coupling calibrated hop-prediction
sets or decisions to jammer power allocation; frame it as "to our
knowledge", since IEEE-only and non-English venues were not
exhaustively searched.

  ---------------------------------------------------------------------------
  Claim                          Why plausibly novel  What would undercut it
  ------------------------------ -------------------- -----------------------
  Online-calibrated decision     The coupling and the A prior system sizing a
  knob (CDT) driving a           online risk          multi-tone jam from a
  threshold-aware knapsack jam,  objective are absent calibrated set or
  with a distribution-free bound from surveyed FH and decision
  on forgone effect              jamming work         

  One system that                Jamming RL is        An existing
  self-identifies the regime via overwhelmingly       intercept-and-jam
  a sensor-aware Bayesian bank   single-strategy,     system that
  and a Fixed-Share mixture,     single-channel       self-selects
  with a tracking-regret bound                        exploitation mode

  A zero-shot, in-context,       Published FH         A synthetic-pretrained,
  gap-robust hop predictor on a  predictors are fit   gap-robust FH predictor
  Mamba-2 backbone               per sequence and     already published
                                 assume clean         
                                 consecutive outputs  

  Intervention-based (causal)    No passive           Prior active-probing
  discrimination of adaptive     classifier can       work testing hopper
  hoppers with an info-gain      obtain this          adaptivity this way
  probe                                               

  Exploitability-certified       Anti-jamming RL      A prior FH study
  evaluation of an FH-EW policy  rarely reports       reporting the same
  via Global PSRO                worst-case           certificate
                                 exploitability       

  An open FH-EW Gymnasium        No open environment  An environment released
  environment with configurable  of this kind was     since the survey
  sensor, moving emitters and a  found                
  dumb-to-adaptive library                            
  ---------------------------------------------------------------------------

The predictability boundary, where crypto and entropy-maximising hoppers
defeat prediction, is presented as a result, not a gap. Characterizing
exactly where a method stops working is itself a contribution.

## 14. Limitations and the predictability boundary

Results are simulation-only, and the architecture is explicit about
where prediction stops working.

-   **Cryptographic hoppers defeat prediction.** Against a keyed CSPRNG
    no predictor beats uniform, by design. The correct behaviour is to
    detect it via sustained κ near 0 and fall to the weighted barrage
    floor.

-   **Keyed or filtered LFSRs break algebraic recovery.** A nonlinear
    filter between register and channel index defeats Berlekamp-Massey
    even though an LFSR is still underneath. Reported on its own, graded
    by filter strength.

-   **Exchangeability fails under adaptation and probing.** The online
    calibration guarantees are long-run averages that need truthful
    feedback, and they can hide short bursts of misses. Report
    sliding-window coverage and realised risk, including where they fall
    short.

-   **Intermittent feedback.** Look-through blanking can hide the true
    hop. The intermittent-feedback conformal variant mitigates this, and
    the assumption is stated.

-   **Moving-emitter sorting is imperfect.** Crossing tracks can cause
    identity swaps, and fast or close drones outrun the tracker. The
    stable features (timing, fingerprint) carry these cases, and the
    failure rate is reported.

-   **Non-stationarity has no clean guarantee.** A reactive adversary
    makes the interaction a non-stationary game. Change detection and a
    randomized policy limit exploitability, but no convergence guarantee
    is claimed, which is why Section 9 reports exploitability instead of
    optimality.

-   **Recent-preprint dependencies.** Mamba-2, SAOCP/DtACI, conformal
    decision theory, conformal e-detectors, TD-MPC2 and Global PSRO are
    fast-moving, and several key sources are 2026 preprints not yet peer
    reviewed. Confirm a usable implementation exists for each before
    committing, and keep the named fallback.

## 15. Build order and fallbacks

The build order is chosen so a defensible thesis exists even if later
tiers run out of time. Each tier produces a standalone result, and
upgrades slot in top-down.

  ----------------------------------------------------------------------------
  Tier   Build                            Produces            If time runs
                                                              short
  ------ -------------------------------- ------------------- ----------------
  1      Gym with sensor model,           The spine end to    A submittable
         deterministic and crypto         end, plus           thesis on its
         adversaries; BM and seed-search  exact-recovery and  own
         experts; greedy knapsack         feasibility-map     
         allocator; fixed-coverage        results             
         conformal set                                        

  2      Soft Fixed-Share mixture with    The                 Keep the mixture
         the regret proposition; Bayesian dumb-and-adaptive   and bank; drop
         hypothesis bank + MSPRT;         headline plot       Mamba-2 to
         predictability score; open-set   across the range    CNN-GRU; drop
         flag; Mamba-2 encoder; zero-shot                     chaos to Markov
         expert; chaotic and Markov                           
         experts                                              

  3      Online calibration: SAOCP set +  The flagship        Fall back to
         CDT decision controller; regime  calibration result  fixed-coverage
         monitor; recce bank sequential   and the recce-value conformal +
         test; anytime ladder;            result              batch CRC; a
         time-since-contact metrics                           small fixed bank

  3.5    Track-aware GNN emitter sorting  Multi-emitter       Cut; the sorter
         with Kalman tracker; moving      sorting and         is an identity
         emitters in the gym              prediction          pass-through

  4      Info-gain probing bandit;        Causal adaptivity   All cut; the
         optional TD-MPC2 expert;         detection, adaptive mixture routes
         optional MLP allocator head      learning curve      to a reactive
                                                              fallback and the
                                                              knapsack

  5      Global PSRO evaluation with the  An exploitability   Cut; always a
         red-team set                     certificate beyond  second
                                          hit-rate            contribution
  ----------------------------------------------------------------------------

**Named fallback for each risky part.** CNN-GRU for the encoder and
predictor; fixed-coverage conformal + batch CRC for calibration; exact
MILP or weighted water-filling for the allocator; pairwise SPRT or a
fixed bank for identification; tracker-only sorting for Layer 0; a fixed
probe schedule for the prober; vanilla PSRO or a scripted best response
for the game.

**One thing to check.** The conformal effect-forgone loss (Section 6.3)
and the allocator (Section 7.1) must use the same g and threshold τ. If
they diverge, the risk guarantee no longer describes the allocation.