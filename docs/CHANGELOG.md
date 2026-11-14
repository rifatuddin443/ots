# Implementation log

A running record of what has been built on top of the conference identification
bank, so the repo always documents itself. **Append a dated entry here for every
future change** (newest first). Deep detail lives in the methodology write-up and
the README; this file is the index to it.

- Methodology & results: [`docs/methodology_and_results.md`](methodology_and_results.md)
- Package overview & quick starts: [`../README.md`](../README.md)
- Figures: [`docs/figures/`](figures/)

---

## 2026-11-14 — Control room drives the closed-loop adversary (dashboard)

The red-team components (evasion, deception) are now playable in the dashboard's
**Control room**, so the two-sided engagement can be driven by hand, not only in the
batch experiments.

- `dashboard/control.py` — `ControlSession` gains `evade`, `deceive`, `decoy`,
  `n_decoys`, `max_tones`. With either adversary flag on it runs a genuine closed
  loop: the jammer's barrage for a hop is backend-authoritative (`_barrage`, the same
  spot↔barrage rule the UI uses), the transmitter is wrapped in
  `fhew.evasion.AdaptiveEvasionGenerator` and reacts to the *previous* hop's jamming
  (deflect / spread; a manual algorithm switch is the re-key, so auto-rekey is off),
  and `fhew.deception.DeceptionEmitter` overlays coherent decoys on the sensor
  stream. Each record now carries `lit` / `next_lit` / `denied` / `decoys` /
  `pressure`. With no flags the session is byte-for-byte the old open-loop one.
- `dashboard/serve.py` — `/api/control/new` parses the new flags.
- `dashboard/views/control.js` — a **Adversary · red team** settings section (Evade /
  Deceive toggles, decoy family + count); the barrage overlay and denial now read the
  backend's closed-loop plan (`lit` / `next_lit`); planted decoys are drawn as purple
  diamonds (legend + tooltip); the accuracy banner reframes to **jammer denial** with
  a "transmitter fighting back" line; the system view shows evasion pressure and
  decoys-per-hop; the event log notes the red-team mode. All closed-loop fields are
  session-scoped, so toggling one starts a new session (like channels/hops).
- CLI: `python -m dashboard.control --evade` / `--deceive --decoy lfsr --n-decoys 3`
  (prints a jammer-denial line). Tests: 4 new in `tests/test_tracker.py` (shape,
  back-compat, evasion escapes the jammer, deception drags the ID to a false break).

Observed in the Control room (M=16, perfect/strong sensor): evasion turns a fully
jammed LFSR (denial ≈ 1.0, next-hop ≈ 0.98) into one denied ≈ 0.15–0.22 with next-hop
≈ 0.10; three LFSR decoys flip a keyed CSPRNG's identification from the correct
*unknown* to a false *LFSR* break.

---

## 2026-11-14 — Closed-loop adversary: jammer evasion + counter-ID deception (exp26, exp27)

The whole repo up to here scores the OTS against a **fixed tape** — a hop sequence
drawn once and sensed, indifferent to the jammer. That is right for the
identification and prediction claims, but it leaves the system untested against the
thing a fielded emitter actually does: *fight back*. This revision adds the missing
half of the engagement — a reactive adversary — and the closed loop that pits it
against the OTS. Everything is additive; no existing module changed, and both sides
still move from history alone (no clairvoyance), so the games are fair, not replays.

### `fhew/evasion.py` — reactive anti-jam transmitter (jammer evasion)
A radio has a receiver, so it knows when it is being denied and (optionally) where
the jamming sits, and it adapts. `AdaptiveEvasionGenerator` wraps **any** base
generator and, from jamming feedback only (`JammingFeedback`: `denied` /
`effect_on_self`, and in `"spectrum"` mode a sensed `jammed_mask`), escalates on one
scalar **pressure** `P`:
- **deflection** off a decaying hot-channel map; **entropy escalation** (a sampling
  temperature that grows with `P`) that drives the OTS predictability `kappa` down;
  and a **re-key** (fresh seed via `rekey_factory`) that invalidates any lock the OTS
  holds on the old deterministic sequence.
- With no jamming it reproduces the base generator exactly (`NO_EVASION_CFG` is a
  passthrough), so switching evasion on never perturbs the baseline experiments.

### `fhew/deception.py` — counter-reconnaissance deception (counter-ID + counter-jam)
Deception is mounted at the one interface the OTS consumes — the `Observation`
stream. `DeceptionEmitter` overlays **time-coherent decoy detections** that follow a
chosen in-bank family (an LFSR), so to a single-emitter collector they look like a
second ordinary emitter rather than noise, plus optional unstructured `chaff`.
`make_deception_episode` wraps any episode; the hidden truth is untouched (scoring
only), decoys are tagged `DECOY_ID`.

### `fhew/duel.py` — the closed-loop engagement
`OTSJammer` (predictor + knapsack/conformal/reactive allocation) vs the adversary,
slot by slot: committed jam lands → sensor + deception → OTS re-predicts and
re-plans → emitter reacts to *this* slot's jamming and picks the next channel. Both
decide from history ≤ t; neither sees the other's current move. `run_duel` returns a
per-slot record and the headline aggregates (fraction denied, next-hop top-1, mean
`kappa`, tones, re-keys).

### Findings
- **exp26 (evasion).** Against the strong sensor, closed-loop evasion collapses the
  fraction of hops denied from **~1.00 → ~0.17** on an LFSR (`kappa` 0.95 → 0.02,
  several forced re-keys) and **~0.99 → ~0.14** on the held-out lagged-Fibonacci
  source; the CSPRNG control is unmoved (0.13 → 0.13 — already at the barrage floor,
  nothing to evade). *Predictable is not un-evadable.*
- **exp27 (deception).** *Counter-ID:* against a keyed CSPRNG (truth = unknown) with
  an LFSR decoy, a pairwise SPRT with **no `H_0`** is false-identified **100 %** at
  every decoy persistence, while the bank's mandatory `H_0` routes an intermittent
  decoy to *unknown* (hard-false-ID ≈ 0.03 at `p_present ≤ 0.35`). The honest limit:
  a near-always-on coherent decoy (`p ≥ 0.7`) embeds a real LFSR and defeats even
  `H_0` — the motivation for emitter-ID sorting (journal). *Counter-jam:* a handful
  of coherent decoys split the predictive jammer's belief and drive denial on the
  real hop from **~1.0 to ~0.1** (it preferentially chases the decoy, which under a
  lossy sensor is a *cleaner* signal than the real emitter), with tones wasted on
  ghosts.

Modules: `fhew/evasion.py`, `fhew/deception.py`, `fhew/duel.py`;
`experiments/exp26_evasion.py`, `experiments/exp27_deception.py`;
`tests/test_evasion.py` (13), `tests/test_deception.py` (9), `tests/test_duel.py` (7).
Lazy re-exports added to `fhew/__init__.py`. Figures in `docs/figures/exp26_*.png`,
`docs/figures/exp27_*.png`.

---

## 2026-11-10 — Conformal next-channel prediction sets + an identification converse (exp20, exp21)

Two new, additive contributions on top of the existing predictor and bank. Both
read only observations (information parity); the hidden truth is used only to
score. Nothing in the existing modules changed.

### exp20 — conformal next-channel sets, coupled to the jam plan (new in the literature)
The predictor emitted a point guess and a *fixed* `top-k` shortlist with no
coverage guarantee. `fhew/conformal.py` adds a **calibrated prediction set**
whose size adapts so the true next channel is covered with a controlled
long-run probability `1 - alpha`, on any emitter, with no distributional
assumption:
- `aps_set(p, level)` — the adaptive-prediction-set builder (smallest
  top-ranked set with cumulative mass ≥ level).
- `ConformalChannelPredictor` — wraps a `ChannelPredictor` with **Adaptive
  Conformal Inference** (Gibbs–Candès): an online effective level `alpha_t`
  updated by `alpha_{t+1} = clip(alpha_t + gamma*(alpha - err_t), 0, 1)`. The
  calibration bit is *erasure feedback* — it fires only on clean slots, which is
  all a passive node sees (the FH twist; standard split-conformal assumes
  exchangeability, which the stream violates).
- `jam_set_from_conformal(set, effect, budget)` — **D1**: the set *is* the
  multi-tone jam plan. Funding it makes the realised next hop denied at the
  coverage rate, so the fraction-of-hops-jammed inherits the distribution-free
  `1 - alpha` bound, with the tone count set by the data instead of a hand-set
  `max_tones`.

Measured (`exp20`, M=16, P_d=0.95, P_fa=0.01, target 0.90): conformal coverage
0.89–1.00 across LFSR/weak-seed/LCG/logistic/markov, the three **new algorithms**
and the CSPRNG, while a fixed top-3 under-covers badly on markov (0.51) and
CSPRNG (0.19). Set size tracks predictability: ~1 tone on a pinned-down LFSR
(κ≈0.95) up to ~14.4 on a CSPRNG (κ≈0.01). Jam-hit rate equals coverage (D1).

### exp21 — a lower bound (converse) on hops-to-identify
`exp11` showed the achievability side (E[τ] linear in log(1/δ) and 1/P_d).
`exp21` adds the matching **information-theoretic floor** no identifier can beat
— the sequential-test / change-of-measure bound
`E[τ] ≥ log((1-δ)/δ) / D`, with the per-slot posterior-log-odds drift `D ≈ P_d·KL*`
estimated directly from the runs (so it holds for the real sensor-aware
emission, detection-thinning included). Measured: the bank sits **at or above**
the floor at every operating point, efficiency ≈ 1.4–2.3× (mean 2.28×) — the
cost is within a small constant of optimal, and the bound rises as 1/P_d exactly
as the drift falls.

### Wiring / tests
- `experiments/exp20_conformal_sets.py`, `experiments/exp21_identification_converse.py`;
  both registered in `scripts/run_all.py` (+ `_headline`).
- `tests/test_conformal.py` — 10 tests (aps-set monotonicity, ACI direction,
  coverage≈target, set-size adaptivity, the D1 jam coupling, converse-is-a-bound).
- README experiments table, package layout and test count updated; methodology
  write-up gains Sections 5.19 (conformal sets) and 5.20 (converse).

Rebuild figures without re-simulating: `python -m experiments.exp20_conformal_sets --plot-only`.

---

## 2026-11-09 — Control room: run/pause on the chart, barrage calls when unsure

Two fixes to the Control room (`dashboard/views/control.js`, standalone rebuilt):
- **Run / Pause on the hop chart.** The run control was only in the settings
  panel, which is closed on many layouts, so the hops could not be stopped at a
  glance. A `❚❚ Pause` / `▶ Resume` button now sits in the hop-chart header. It
  shares one `toggleRun()` with the panel button and Space. Pressing an algorithm
  while paused switches the radio but leaves the run paused; only the first
  algorithm of a session starts it.
- **Barrage, not one channel, when unsure.** The system called a single channel
  even at 36% sure, so every miss on that one channel scored as wrong. Now: when the top channel is under
  90% (`SURE_P`) the system calls a barrage of at least two channels together,
  growing with uncertainty up to the "max transmit freqs" control (default 3). A
  sure call stays one channel (spot). The jammer lights exactly the barrage, the
  hop chart draws it (ring = top channel, amber bars = extras), and the tooltip
  and system view name it.
- **Accuracy scores the barrage.** A next-hop hit is "true channel ∈ barrage".
  The exact top-1 rate is shown beside it, and the random baseline is the chance
  of a barrage of that size. Measured on the real engine (4 plans × 2 sensors,
  300 hops, 16 channels, max 3): exact top-1 averages **51%**, the barrage
  **67%**, with an average barrage of 2.0 channels.

Checks: all 131 tests pass; a headless browser run confirmed pause stops the
hop counter, Space resumes, switching while paused stays paused, and every
sampled unsure call (top-1 under 90%) shows 2–3 channels.

## 2026-11-04 — Fixes from the novelty audit: clean-run feasibility, per-competitor e-process

Two correctness fixes found while checking the paper's claims against the literature
(audit: `reports/OTS novelty claims audit.md`).

### Feasibility (`fhew/feasibility.py`)
- `can_recover_exactly` credited `P_d²` of the hops as one clean run, but a run of
  `2L` hops is fully detected with probability `P_d^{2L}` (0.81 vs ≈0.23 at
  `L = 7`, `P_d = 0.9`). New `prob_clean_run(n, k, p)` gives the exact probability
  that `n` hops contain `k` detected in a row; exact recovery is feasible when a
  `2L` run is seen with probability ≥ `min_prob` (default 0.5).

### E-process (`fhew/stopping.py`)
- `EProcess` now defaults to the e-value against the strongest competitor,
  `p_k / max_{j≠k} p_j`, giving a Ville bound of `β` against *each* in-bank
  competitor (union: `(K−1)β` for any wrong one). The old prior-weighted mixture
  denominator, whose bound held only under that mixture, is kept as
  `EProcess(against="mixture")`. Stopping is stricter, so `τ` grows.

### Tests
- `tests/test_fhew.py` — 2 added: the max denominator ignores a competitor's prior
  weight; Monte-Carlo check that each wrong hypothesis crosses `1/β` at most `β`
  of the time.
- `tests/test_mixture_allocation.py` — 1 added: the clean-run probability and the
  exact-recovery rule at short/long windows and a lossy sensor.

**Not re-run:** exp9 (e-process) and exp17 (feasibility) figures and tables predate
these fixes. With exp17's Panel B base (`P_d = 0.55`, `L = 14`) the exact-recovery
band now vanishes; see `methodology_and_results.md` §5.17.

---

## 2026-11-03 — Structured sequence coverage: rule search, seed beam, wide BM

Closes the gaps in which hop-sequence types the predictor covers, aiming at the
fewest hops to lock. Results: [`methodology_and_results.md` §5.18](methodology_and_results.md).

### Generators (`fhew/generators.py`)
- `arithmetic`, `polynomial`, `linear_rec` (order-k affine), `nonlinear_rec`
  (bilinear order 2), `lookup_table`, `hybrid_sum` (table + drift),
  `hybrid_switch` (rule changes every `segment` hops), `noisy_rule` (affine rule
  with random jumps). `_PRIMITIVE_TAPS` gains a 16-bit polynomial.

### Experts (`fhew/experts.py`)
- `RuleSearchHypothesis` + `default_rule_library` — Bayesian program search over
  vectorised rule families (affine order 1–3, poly finite differences, quadratic,
  bilinear, XOR, rotate-XOR, lag-P tables), Occam family prior, pruning,
  restart-with-carry on rule switches, ghost-aware re-sync. `heavy=False` gives a
  ~10k-rule library.
- `SeedBeamHypothesis` — pruned full-keyspace seed search (`weak_seed`, `lcg`).
- `AdaptiveBMHypothesis` — Berlekamp-Massey with an adaptive window (LFSRs to
  length 32), drops a fit after two contradictions.
- `build_hypothesis` kinds: `rule_search`, `seed_beam`, `bm_adaptive`.

### Wiring
- `prediction_expert_specs(..., structured=True)` adds the four experts (BM only
  for power-of-two `M`); `structured=False` restores the old set.
- The switch tracker's universal experts swap `recurrence` for the light
  `rule_search` (shared read-only tables across segments).
- `STRUCTURED_GENERATORS`, target-family maps updated.

### Experiment & tests
- `exp19_sequence_coverage` (registered in `scripts/run_all.py`).
- `tests/test_sequences.py` (23 tests: generator laws, lock-hop budgets,
  lossy-sensor accuracy, rule-switch re-lock, CSPRNG humility, seed beam, BM).
- Fixed a Windows-only failure in
  `test_dashboard_standalone_inlines_every_local_asset` (read the page as UTF-8).

**Not re-run:** exp14/exp15/exp18 figures still show the pre-change predictor.

## 2026-10-31 — Control room: scrubbable hop chart, transmitter in the side panel

UI refinements to the Control room (`dashboard/views/control.js`):
- **Scrub past hops.** A slider under the hop chart scrolls back through earlier
  hops without stopping the run (`viewEnd` pins the window; `⏭` resumes
  following). The chart draws up to the scrubbed hop; the live run keeps
  recording underneath.
- **Transmitter moved into the side panel.** The algorithm picker + transport now
  live in the settings panel next to the sensor/jammer controls, freeing the main
  column for the chart (the redundant settings chip is dropped in that layout).
- **Jam tooltip clarified.** The per-hop `jam` line now reads "jam → next hop" and
  shows whether the lit tones actually caught the following hop (✓/✗). The jam is
  the plan for the *next* hop, not the current one, which the old label conflated.

## 2026-10-30 (latest) — Multiple transmit frequency: jam more channels when unsure

Implements "transmit on several frequencies at once when the prediction is not
confident, up to an operator-set maximum" — both in the allocator core and as a
live control in the Control room.

### Core (`fhew/allocator.py`)
- `tones_from_confidence(kappa, max_tones)` — the rule: a sharp prediction
  (`kappa→1`) collapses to one spot tone, a flat one (`kappa→0`) opens out to
  `max_tones`, linearly in between (clamped to `[1, max_tones]`).
- `greedy_knapsack(..., max_tones=None)` — caps the number of funded channels
  (transmit frequencies); `None` keeps the original budget-bounded behaviour, so
  nothing else changes.
- `predictability(p, channels)` — the `κ` of a predictive vector (matches the
  mixture's definition), so the allocator can read confidence on its own.
- `AllocationController(max_tones=…)` — in predictive mode the active tone count
  is `tones_from_confidence(κ, max_tones)` each slot; `AllocationStep` now carries
  `n_tones` and `kappa`. Default `max_tones=None` ⇒ backward compatible.

### Control room (the interactive "controls" surface)
- `dashboard/control.py` records a per-hop `jam` ranking (the most-likely next
  channels, best first) and reports `jam_cap = min(M, 8)` in the session info.
- `dashboard/views/control.js` adds a **"Jammer · max transmit freqs"** slider to
  the settings panel (1..`jam_cap`, display-only so it updates live without a
  re-run). The hop chart's NEXT column lights the extra transmit frequencies
  (amber) around the cyan top-1 call with a `jam ×N` tag; the system-view card
  shows "Jamming next: N frequencies" with the lit channels; the accuracy card
  shows a running multi-tone coverage stat (how often a lit tone caught the true
  next hop). The client's top-`n` selection equals the knapsack under the Control
  room's uniform per-channel cost, so the picture matches the allocator.

### Tests
- `tests/test_mixture_allocation.py` — 5 added: the confidence→tones mapping, the
  `max_tones` cap (and that `None` is unchanged), the `predictability` helper, and
  the controller's tone count following confidence (sharp→1, flat→max).

Rebuild the standalone with `python -m dashboard.build --reuse --standalone ots_dashboard.html`.

---

## 2026-10-30 (later) — Control-room settings panel; faster re-learning after a switch

### Settings in a side panel
The Control room's setup (channels, hops, sensor incl. *Perfect*, new session),
playback speed and chart overlays moved out of the page into a **settings panel
docked on the right**. "⇥ Pop in" hides it; the edge tab, the settings chip in the
transmitter card or `S` brings it back; the choice is remembered. Below 1100 px it
slides over the page as a drawer (`Esc` closes). The panel sits outside the page's
CSS query container, because a query container is the containing block of
`position: fixed` children, which the drawer and the tab need to escape.

### Accuracy after a switch
Measured first against an oracle restarted at the exact switch hop: with a perfect
sensor the tracker already matched it from hop 6, so the dip was the experts'
re-learning time, not switch handling (a higher switch prior did not help either).
Changes:
- `fhew/experts.py::RecurrenceHypothesis` — a zero-shot learner of modular linear
  recurrences `c_t = a1 c_{t-1} + a2 c_{t-2} + b (mod M)` (all rules at once; each
  re-syncs to clean detections and rolls its own prediction across missed hops;
  sensor-aware scoring; `|a| <= 8` above 32 channels). Added to the tracker's
  universal experts (`fhew/tracker.py::universal_specs`); it names no generator,
  so identification is unchanged. Lagged generator, perfect sensor: hops 6–10
  after a switch 5 % -> 90 %; full-state LCG under the typical sensor 0 % -> 100 %.
- `LCGCryptanalysisHypothesis` now drops leading outputs that do not fit before
  recovering `(a, c)`: a segment starting one hop before the LCG came on air no
  longer blocks recovery for good (it had cost exact identification once the
  recurrence learner made such segments competitive).
- `SwitchTracker(merge=20)`: a takeover within 20 hops of a detected switch
  re-dates it instead of counting a second switch.
- `exp18` now scores next-hop accuracy by hops since each switch (panel C).

Next-hop top-1 by hops since a switch (before -> after, same plans): perfect
6–10: 66 -> 79 %, 11–20: 64 -> 76 %; strong 11–20: 60 -> 74 %; typical 11–20:
58 -> 68 %, 21–40: 60 -> 72 %; contested about unchanged. Switches detected: perfect
93 -> 100 %, strong 90 -> 97 %, typical 90 -> 93 %. Dangerous locks unchanged (0 %,
contested 1 of 36). Typical-sensor false alarms 4 -> 6 per 2,400 hops.

Tests: 4 added (recurrence learner exact / noisy / no invented pattern; tracker
predicts a new lagged generator within 8 hops of the switch); 126 pass.

---

## 2026-10-30 — Control room: drive the transmitter yourself, switch algorithms mid-test

A new dashboard page where the operator chooses the hop algorithm the radio runs
— and switches it whenever they like — while the system, seeing only the sensor
output, has to notice the switch, name the new algorithm and keep predicting the
next hop. Channels, hops and the sensor (including a **perfect, 100%-accurate**
one) are chosen per session.

### Why a new tracker was needed
The identification bank's deterministic experts are anchored in time: candidate
`j` predicts `traj[j, t]` counted from the start of the engagement. A freshly keyed
weak-seed PRNG or truncated LCG switched in at hop 137 emits its own counter-0
output there, which no candidate anchored at hop 0 produces — and flushing on an
alarm (`forget`) cannot help because the alarm lands a few hops late, so the
anchor is still off. The existing `ChannelPredictor` + regime monitor therefore
re-pointed the *prediction* through its universal learner but could not
*re-identify* such generators after a switch.

### What was added
| Where | What |
| --- | --- |
| `fhew/tracker.py` | `SwitchTracker`: Bayesian online change-point detection whose run-length hypotheses each carry an **anchored segment** (a fresh identification bank + the universal context-mixing / phase-space experts, started at the hop the segment assumes the generator began). The run-length posterior is the switch detector (a switch is declared when ≥ 90 % of the mass sits on segments younger than the current regime; the most probable dates it), the identifier (family posterior **marginalised over segments**, so a newborn segment that fits a few noisy hops by luck cannot produce a false lock), and the predictor (run-length-weighted mixture of segment forecasts). Lock at 95 % with hysteresis (released below 80 %). Segments share the experts' read-only trajectory tables via an `ExpertLibrary` (built once per channel count; a segment birth is a cheap copy). Newborn segments are protected for `ceil(9 / clean-hop rate)` hops (10–32), so the segment born at a switch survives until it can lock under a worse sensor. Register experts are dropped for a non-power-of-two channel count. |
| `dashboard/control.py` | `ControlSession`: the transmitter you drive (re-keyed from the recce ranges on every switch), the sensor, the tracker, per-hop records (truth for the audience only), per-algorithm summary; library cache; CLI `python -m dashboard.control --channels 16 --hops 300 --sensor perfect --plan lfsr:0 weak_seed:80 markov:160`. |
| `dashboard/serve.py` | `GET /api/control/new` (channels, hops, p_d, p_fa, seed) and `GET /api/control/step` (session, algo, n); warms the 16-channel library at start. |
| `dashboard/views/control.js` | The **Control room** page: setup (channels 4–64, hops 20–1000, sensor Perfect / Strong / Typical / Contested / Custom), transmitter buttons grouped known / new / control (on-air indicator; register algorithms disabled off a power of two), run / pause / 1-hop / speed; hop chart with the system's belief band over the transmitter's truth band and switch / detection markers; the system's view (identification, lock, estimated start, recovered vs true secret, next-hop call, family race, switch posterior); whole-session timeline (truth vs belief + rolling hit rate); event log; per-algorithm table. Controls are built once and updated in place so a click is never lost to a re-render while running. Offline (standalone file) it explains how to start the live server. |
| `experiments/exp18_switch_tracking.py` | Random switch plans over all 13 algorithms per sensor: detection rate / delay, dating, false alarms, identification by the end of each regime, dangerous locks. Wired into `scripts/run_all.py`. Figure: `docs/figures/exp18_switch_tracking.png`. |
| `tests/test_tracker.py` | 12 tests: power-of-two bank specs, protection vs sensor, shared tables / private state, LFSR → weak-seed switch detected, dated exactly and re-identified, a noisy chaos → LCG switch, no false switch or algebraic claim on a CSPRNG, the control session end to end, channel-count and settings validation. |

### Measured (`exp18`, 6 random switch plans x 400 hops per sensor, `M=16`, 36 regimes / 30 switches each)
| Sensor | Switches detected | Delay (median / 90th pct) | Dated within ±1 hop | False alarms / 100 hops | Named exactly by regime end (+ closest family) | Correct lock (median hops) | Dangerous locks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Perfect (`P_d=1, P_fa=0`) | 93 % | 1 / 5 hops | 96 % | 0.04 | 81 % (+19 %) | 72 % of regimes (6) | 0 % |
| Strong (0.95, 0.01) | 90 % | 2 / 5 | 96 % | 0.08 | 72 % (+28 %) | 72 % (7) | 0 % |
| Typical (0.8, 0.02) | 90 % | 4 / 10 | 81 % | 0.17 | 75 % (+25 %) | 64 % (9) | 0 % |
| Contested (0.6, 0.05) | 90 % | 10 / 22 | 59 % | 0.08 | 75 % (+17 %) | 58 % (20) | 3 % (1 of 36) |

- **Every switch *into* an algorithm the bank has an expert for was detected**, on every
  sensor. All missed switches are between two sources with no exact structure for the
  bank (Markov → CSPRNG, CSPRNG → lagged generator), where the old segment's online
  learners simply adapt and the prediction carries on.
- "Closest family" is the graceful answer for the new algorithms (e.g. the
  quadratic PRNG described as a Markov hopper, the sine map as a chaotic map) and
  for the CSPRNG; it is never an algebraic claim.
- The one dangerous lock (contested sensor) is an LFSR candidate fitting a stretch
  of a Markov hopper's noisy hops, mostly through ghost detections — the bank's own
  residual false-ID rate under that sensor (exp13: hard false-ID <= 0.05), slightly
  amplified because a segment may start where the luck is best. Requiring the lock
  on two consecutive hops removed the perfect-sensor case (a candidate matching the
  first four hops after a switch to a CSPRNG exactly, by chance).
- Cost: 40–85 ms per hop at `M=16` (more segments are protected under a worse sensor).

Also: `index.html` registers the page after Live demo (shortcut hint now `1`–`8`);
`ots_dashboard.html` rebuilt with `--reuse`; README updated (dashboard table,
experiments table, a switch-tracker section, package layout, test count).

---

## 2026-10-23 (later) — Live demo: playable regime switches + prediction-accuracy readout

Made the adaptive behaviour playable in the **Live demo** page and surfaced a
running next-hop prediction accuracy on every run.

### Prediction accuracy (all live runs)
- An always-on **"Next-hop prediction"** banner shows the predictor's running
  accuracy (`X of Y right · Z%`) with a bar against the 1/M random baseline,
  updated hop by hop. Uses the already-recorded `fc_pred` (the Fixed-Share
  predictor's forecast) vs the truth.

### Playable regime-switch scenarios (the adaptive part, hop by hop)
- New picker group **"The radio changes recipe (adaptive)"** in the Live-demo
  settings, with two scenarios: `lfsr_to_markov` and `chaos_to_markov`. The radio
  switches its generator at hop 100; you watch accuracy dip and the predictor
  re-point.
- Scenario render mode in `dashboard/views/live.js`: scenario narration, an
  event log (on A → switched to B → change detected/memory flushed), the live
  dominant expert, and a **flush vs no-flush** accuracy comparison, plus an
  **accuracy-over-time** timeline with switch/flush markers. The bank
  identification race/verdict are swapped out for scenario traces.
- Data: `dashboard/traces.py::record_switch_scenario` runs the predictor twice on
  the same observations (monitor on/off) so the demo shows the flush recovers
  faster; `dashboard/build.py::build_scenarios` records them; metadata in
  `OTS_DATA.scenarios`. Scenarios are kept **out of `OTS.emitters`** so the
  scoreboard and other pages are unaffected.

### Enabling the context flush in the live predictor
- Added `FixedShareMixture.forget(retain)` to `fhew/experts.py` (the mixture the
  live `ChannelPredictor` uses) — flushes member contexts and re-flattens weights.
- `ChannelPredictor` (`fhew/predict.py`) gained an **opt-in regime monitor**
  (`monitor=True`): a CUSUM on the mixture surprise that flushes the context on a
  detected switch (`alarms` records the slots). Off by default, so stationary-source
  behaviour is unchanged.
- Tests: 2 added to `tests/test_mixture_allocation.py` (mixture forget; predictor
  monitor flush recovers at least as fast).

Rebuild the dashboard data + standalone with `python -m dashboard.build`
(`--scenario-seeds N`; `--quick` shrinks it).

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

## Conference-paper additions: baselines, response result, compute time

Four standalone experiments back results added to the IEEE paper
(`docs/paper/ieee/`). They are not wired into `scripts/run_all` (exp23 trains an
LSTM per sensor and needs `torch`); run each with `python -m experiments.<name>`,
then `python -m scripts.paper_additions`.

- `exp22_id_baselines` — bank + `H_0` vs pairwise SPRT / fixed-length LRT / learned
  classifier, 2,000 episodes. Typical sensor: in-bank accuracy 0.99 / 0.98 / 0.94 /
  0.61; hard false-ID on out-of-bank emitters 0.005 / 0.235 / 0.230 / 0.482.
- `exp23_prediction_baselines` — Markov-1, PPM-C and a trained LSTM on exp14's
  episodes (450). Its `proposed` / `id_only` columns reproduce `extra_predict.json`.
  The LSTM reaches 1.00 on the in-family LFSR and 0.01–0.13 on the held-out
  lagged-Fibonacci and quadratic generators; PPM-C gets 0.74 / 0.42 / 0.21 on
  lagged-Fibonacci at the strong / typical / contested sensor.
- `exp24_denial` — fraction of hops denied vs tone budget on exp20's protocol, 540
  episodes. Unlimited budget: 0.90–1.00 denied for every generator; mean tones on
  the typical sensor 1.2 (LFSR) to 14.5 (CSPRNG).
- `exp25_runtime` — median per-slot time on one 2.1 GHz Xeon core, `M=16`: bank
  0.79 ms, predictor 1.86 ms (95th percentile 17 ms), conformal set 0.06 ms, switch
  tracker 46 ms.

### Paper text changes
- Abstract: the identification claim is qualified by sensor and excludes the
  full-state LCG; the episode total (15,690) is accounted for in Section IV.
- Table IV (prediction): the Strong column held the `P_fa = 0` numbers of the default
  exp14 run; it now holds the strong-sensor `(0.95, 0.01)` numbers from
  `data/extra_predict.json` (identification-only bank on lagged-Fibonacci 0.38, not
  0.35; quadratic 0.90, not 1.00).
- Fig. 2 (per-slot flowchart, a duplicate of Algorithm 1) was dropped and float
  spacing tightened to make room; the body fits six pages, the last references
  run onto a seventh.
