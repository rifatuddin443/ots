"""Record engagements of the identification bank for the demo dashboard.

An engagement is one emitter observed through the lossy sensor for ``window``
slots. :func:`record_engagement` runs the real bank (the one-expert-per-family
identification bank of exp12/exp13) slot by slot and keeps everything the
dashboard animates: the sensor's detections, the hidden truth (for the audience
only -- the bank never sees it), the posterior over every hypothesis, the
posterior-weighted next-hop forecast, the number of candidate generator states
still plausible, and the recovered secret next to the true one. The pairwise
SPRT baseline (no ``H_0``) is run on the *same* observations for comparison.

Episodes are drawn exactly as ``experiments.common.episode_for`` draws them, so
a dashboard trace is the same episode the experiments score for that seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments import common as C
from fhew.baselines import PairwiseSPRT
from fhew.channels import ChannelSet
from fhew.config import GENERATOR_TARGET_FAMILY, NEW_ALGORITHM_GENERATORS, prediction_expert_specs
from fhew.experts import (
    DeterministicHypothesis,
    HypothesisBank,
    LCGCryptanalysisHypothesis,
    MixtureHMMHypothesis,
    build_hypothesis,
)
from fhew.generators import make_generator
from fhew.likelihood import SensorAwareEmission
from fhew.predict import ChannelPredictor
from fhew.scenarios import make_switch_episode
from fhew.sensor import SensorModel
from fhew.sim import Episode, decide_outcome, run_sprt

M = C.DEFAULT_M
WINDOW = C.DEFAULT_WINDOW
DELTA = 0.05

#: a candidate counts as still plausible while its joint posterior
#: P(family) * P(parameters | family) stays above this
ALIVE_THRESHOLD = 1e-6

#: naming a family that asserts an exact algebraic break when it is not the truth
ALGEBRAIC_FAMILIES = frozenset({"lfsr", "filtered_lfsr", "lcg", "lcg_full", "weak_seed"})

HYPOTHESES = [
    {"id": "H0", "name": "Unknown (H₀)",
     "desc": "None of the above: random or out-of-bank source"},
    {"id": "lfsr", "name": "LFSR",
     "desc": "Shift register · state-filtered Berlekamp–Massey"},
    {"id": "filtered_lfsr", "name": "Filtered LFSR",
     "desc": "Nonlinear combiner · correlation attack"},
    {"id": "lcg", "name": "LCG",
     "desc": "Linear congruential · seed search"},
    {"id": "lcg_full", "name": "Full-state LCG",
     "desc": "Exposed state · algebraic (a, c) recovery"},
    {"id": "weak_seed", "name": "Weak-seed PRNG",
     "desc": "Strong mixer, small keyspace · seed search"},
    {"id": "chaos", "name": "Chaotic map",
     "desc": "Logistic / tent / Hénon · symbolic-dynamics ID"},
    {"id": "markov", "name": "Markov hopper",
     "desc": "Statistical pattern · online transition fit"},
]

EMITTERS = [
    {"id": "lfsr", "name": "LFSR", "kind": "Shift register",
     "blurb": "A 5–7 bit linear-feedback shift register picks each hop."},
    {"id": "filtered_lfsr", "name": "Filtered LFSR", "kind": "Nonlinear register",
     "blurb": "A 9-bit register hidden behind a nonlinear Boolean combiner."},
    {"id": "lcg", "name": "LCG", "kind": "Congruential PRNG",
     "blurb": "A truncated linear congruential generator with a secret seed."},
    {"id": "lcg_full", "name": "Full-state LCG", "kind": "Congruential PRNG",
     "blurb": "An LCG whose whole state is the channel; (a, c) are secret."},
    {"id": "weak_seed", "name": "Weak-seed PRNG", "kind": "Mixed PRNG",
     "blurb": "A well-mixed PRNG (SplitMix64) keyed from only 256 seeds."},
    {"id": "logistic", "name": "Logistic map", "kind": "Chaotic",
     "blurb": "The chaotic logistic map x ← r·x(1−x), quantised to channels."},
    {"id": "tent", "name": "Tent map", "kind": "Chaotic",
     "blurb": "The chaotic tent map, quantised to channels."},
    {"id": "henon", "name": "Hénon map", "kind": "Chaotic",
     "blurb": "The 2-D Hénon attractor, quantised to channels."},
    {"id": "markov", "name": "Markov hopper", "kind": "Statistical",
     "blurb": "Each hop drawn from a hidden transition matrix."},
    {"id": "lagged_fib", "name": "Lagged generator", "kind": "New algorithm", "new_algo": True,
     "blurb": "A new scheme with no expert in the bank: each hop is the sum of the "
              "last two. Not a first-order pattern, so only the universal predictor catches it."},
    {"id": "quadratic_cg", "name": "Quadratic PRNG", "kind": "New algorithm", "new_algo": True,
     "blurb": "A new nonlinear congruential generator the bank has no expert for; "
              "still predictable once its cycle is learnt."},
    {"id": "sine_map", "name": "Sine map", "kind": "New algorithm", "new_algo": True,
     "blurb": "A chaotic map outside the bank's chaos family; predictable above chance "
              "only through the universal predictor."},
    {"id": "csprng", "name": "Keyed CSPRNG", "kind": "Cryptographic control",
     "blurb": "SHA-256 counter mode with a 128-bit key: nothing to exploit."},
]

PRESETS = [
    {"id": "clear", "name": "Strong signal", "p_d": 0.95, "p_fa": 0.01},
    {"id": "typical", "name": "Typical", "p_d": 0.80, "p_fa": 0.02},
    {"id": "harsh", "name": "Contested", "p_d": 0.60, "p_fa": 0.05},
]

#: the guided demo: identification, a hostile sensor, chaos, a new algorithm, the H_0 test
TOUR = [
    {"emitter": "lfsr", "preset": "typical"},
    {"emitter": "weak_seed", "preset": "harsh"},
    {"emitter": "henon", "preset": "typical"},
    {"emitter": "lagged_fib", "preset": "clear"},
    {"emitter": "csprng", "preset": "typical"},
]


@dataclass
class Engine:
    """Bank, baseline SPRT and next-channel predictor for one sensor condition.

    The bank *identifies* the generator; the :class:`~fhew.predict.ChannelPredictor`
    (the thesis Fixed-Share mixture, carrying a zero-shot universal predictor)
    *predicts the next channel* — and keeps predicting for a new algorithm the
    bank has no expert for. Both read the same observation stream.
    """

    channels: ChannelSet
    sensor: SensorModel
    window: int
    bank: HypothesisBank
    sprt: PairwiseSPRT
    predictor: ChannelPredictor

    @classmethod
    def build(cls, p_d: float, p_fa: float, *, window: int = WINDOW) -> "Engine":
        ch = ChannelSet.of_size(M)
        sensor = SensorModel(p_d=p_d, p_fa=p_fa)
        specs = C.id_specs(M)
        bank = C.make_bank(ch, window, sensor, specs, stopping={"kind": "msprt", "delta": DELTA})
        sprt = PairwiseSPRT([build_hypothesis(s, ch, window) for s in specs],
                            SensorAwareEmission(sensor), M)
        # The predictor's identification experts use the same recce ranges as the
        # bank (so it predicts known generators exactly as well), plus the
        # zero-shot universal predictor for everything outside the bank.
        predictor = ChannelPredictor(ch, sensor, specs=prediction_expert_specs(M),
                                     horizon=window + 4)
        return cls(ch, sensor, window, bank, sprt, predictor)

    def search_space(self) -> int:
        return sum(_n_candidates(h) for h in self.bank.hypotheses)


# ------------------------------------------------------------------ episodes
def draw_episode(engine: Engine, gen_name: str, seed: int):
    """Draw the episode ``experiments.common.episode_for`` would, keeping the generator.

    The same RNG calls happen in the same order as ``make_episode``, so the
    stream is identical; the generator instance is kept only to reveal its
    secret to the audience after the fact.
    """
    rng = np.random.default_rng(seed)
    params = C.draw_episode_params(gen_name, rng)
    gen = make_generator(gen_name, engine.channels.size, rng=rng, **params)
    true = gen.sequence(engine.window)
    obs = engine.sensor.observe(true, engine.channels, rng)
    ep = Episode(true_label=gen_name, true_channels=true, observations=obs,
                 window=engine.window, channels=engine.channels, p_d=engine.sensor.p_d,
                 p_fa=engine.sensor.p_fa, seed=int(seed), gen_params=params)
    ep.target_label = GENERATOR_TARGET_FAMILY.get(gen_name, "unknown")
    return ep, gen


def _n_candidates(h) -> int:
    if isinstance(h, DeterministicHypothesis):
        return h.n_particles
    if isinstance(h, MixtureHMMHypothesis):
        return len(h.log_w)
    return 0


def _alive(bank: HypothesisBank, post: np.ndarray) -> int:
    n = 0
    for k, h in enumerate(bank.hypotheses):
        if isinstance(h, DeterministicHypothesis):
            w = h.candidate_posterior()
        elif isinstance(h, MixtureHMMHypothesis):
            w = np.exp(h.log_w)
        elif isinstance(h, LCGCryptanalysisHypothesis) and h._recovered:
            w = np.ones(1)
        else:
            continue
        n += int(np.count_nonzero(w * post[k] >= ALIVE_THRESHOLD))
    return n


# ------------------------------------------------------------------- secrets
_CHAOS_NAMES = {"logistic": "Logistic map", "tent": "Tent map", "henon": "Hénon map"}


def true_secret(gen) -> str:
    """The generator's hidden parameters, phrased for the audience."""
    name = gen.name
    if name in ("lfsr", "filtered_lfsr"):
        return f"L={gen.length} · state 0x{gen._init_state:X}"
    if name in ("lcg", "weak_seed"):
        return f"seed {gen._seed}"
    if name == "lcg_full":
        return f"a={gen.a}, c={gen.c}"
    if name == "logistic":
        return f"Logistic map · r={gen.r:g}"
    if name == "tent":
        return f"Tent map · μ={gen.mu:g}"
    if name == "henon":
        return f"Hénon map · a={gen.a:g}, b={gen.b:g}"
    if name == "markov":
        return "hidden 16×16 transition matrix"
    if name == "lagged_fib":
        return f"sum of last {gen.lag} · start {'·'.join(str(s) for s in gen._seed)}"
    if name == "quadratic_cg":
        return f"a={gen.a}, b={gen.b}, c={gen.c} · start {gen._seed}"
    if name == "sine_map":
        return f"Sine map · r={gen.r:g}"
    if name == "csprng":
        return f"{8 * len(gen.key)}-bit key"
    return "—"


def recovered_secret(h, truth: np.ndarray, true_text: str) -> tuple[str | None, str | None]:
    """What the hypothesis' best candidate says the secret is, and how it matches.

    The match is ``exact`` (same secret), ``equivalent`` (a different key that
    produces the identical hop sequence), ``family`` (right chaotic map, nearby
    parameter), ``model`` (a statistical fit, no single secret) or ``mismatch``.
    """
    text = None
    if isinstance(h, DeterministicHypothesis) and h.candidate_labels:
        lab = h.candidate_labels[h.map_candidate()]
        if lab[0] in ("lfsr", "filtered_lfsr"):
            text = f"L={lab[1]} \u00b7 state 0x{lab[-1]:X}"
        elif lab[0] in ("lcg", "weak_seed"):
            text = f"seed {lab[1]}"
        if text is not None and text != true_text:
            same = np.array_equal(h.map_trajectory()[:truth.size], truth)
            return text, "equivalent" if same else "mismatch"
    elif isinstance(h, MixtureHMMHypothesis):
        fam, p = h.map_candidate()
        if fam == "henon":
            text = f"H\u00e9non map \u00b7 a={p[0]:g}, b={p[1]:g}"
        else:
            sym = "r" if fam == "logistic" else "\u03bc"
            text = f"{_CHAOS_NAMES.get(fam, fam)} \u00b7 {sym}={p:g}"
        if text != true_text:
            return text, "family" if true_text.startswith(_CHAOS_NAMES[fam]) else "mismatch"
    elif isinstance(h, LCGCryptanalysisHypothesis) and h._recovered:
        text = f"a={h._a}, c={h._c}"
    elif getattr(h, "label", "") == "markov":
        return "16\u00d716 transition fitted online", "model"
    if text is None:
        return None, None
    return text, "exact" if text == true_text else "mismatch"


# ---------------------------------------------------------------- engagement
def _top1(forecasts: list[np.ndarray], truth: np.ndarray, *, start: int = 0) -> float | None:
    """Share of slots where the argmax forecast equals the true next channel."""
    n = hit = 0
    for j, f in enumerate(forecasts):
        if j < start or j + 1 >= len(truth):
            continue
        n += 1
        hit += int(int(np.argmax(f)) == int(truth[j + 1]))
    return round(hit / n, 4) if n else None


def record_engagement(engine: Engine, gen_name: str, seed: int, *, preset: str = "custom") -> dict:
    """Run one engagement slot by slot and return the dashboard trace."""
    ep, gen = draw_episode(engine, gen_name, seed)
    bank = engine.bank
    bank.reset()
    predictor = engine.predictor
    predictor.reset()
    labels = bank.labels
    dets, post, fc, fc_pred, kappa, kappa_pred, alive = [], [], [], [], [], [], []
    pred_forecasts = []
    truth = np.asarray(ep.true_channels)
    secret_true = true_secret(gen)
    recovered = (None, None)
    for o in ep.observations:
        bank.update(o)
        predictor.observe(o)
        p = bank.posterior()
        nxt = bank.predict()
        pnxt = predictor.predict_next()
        pred_forecasts.append(pnxt)
        dets.append(sorted(int(c) for c in o.detected_channels))
        post.append([round(float(x), 5) for x in p])
        fc.append([int(round(1000 * float(x))) for x in nxt])
        fc_pred.append([int(round(1000 * float(x))) for x in pnxt])
        nz = nxt[nxt > 0]
        kappa.append(round(float(1.0 - (-(nz * np.log(nz)).sum()) / np.log(M)), 4))
        kappa_pred.append(round(float(predictor.predictability()), 4))
        alive.append(_alive(bank, p))
        if bank.declared and bank.tau_stop == bank.t:
            recovered = recovered_secret(bank.hypotheses[bank.declared_index], truth, secret_true)

    if bank.declared:
        decision = bank.declared_identity()
        tau = int(bank.tau_stop)
    else:
        top = int(np.argmax(bank.posterior()))
        decision = "unknown" if top == 0 else labels[top]
        tau = None
        if top != 0:
            recovered = recovered_secret(bank.hypotheses[top], truth, secret_true)

    base = run_sprt(engine.sprt, ep)
    half = engine.window // 2
    return {
        "emitter": gen_name, "preset": preset, "seed": int(seed),
        "p_d": engine.sensor.p_d, "p_fa": engine.sensor.p_fa,
        "M": M, "window": engine.window, "delta": DELTA,
        "labels": list(labels), "target": ep.target_label,
        "truth": [int(c) for c in truth],
        "dets": dets, "post": post, "fc": fc, "kappa": kappa, "alive": alive,
        # the Fixed-Share predictor's next-channel forecast + its sharpness, slot by slot
        "fc_pred": fc_pred, "kappa_pred": kappa_pred,
        "pred_top1": _top1(pred_forecasts, truth),
        "pred_top1_steady": _top1(pred_forecasts, truth, start=half),
        "bank_top1": _top1([np.asarray(f, dtype=float) for f in fc], truth),
        "new_algo": gen_name in NEW_ALGORITHM_GENERATORS,
        "search_space": engine.search_space(),
        "declared": bool(bank.declared), "tau": tau, "decision": decision,
        "correct": decision == ep.target_label,
        "secret_true": secret_true, "secret_recovered": recovered[0],
        "secret_match": recovered[1],
        "baseline": {"decision": base.decision_label, "declared": base.declared,
                     "tau": base.tau_stop, "correct": base.correct},
    }


# ----------------------------------------------------------- regime scenarios
#: a longer window than the identification demo, so the switch and the recovery
#: both have room to play out.
SCENARIO_WINDOW = 200
SCENARIO_SWITCH = 100

#: Playable "the radio changes recipe mid-contact" scenarios for the live demo.
#: Each is two generator segments concatenated at ``SCENARIO_SWITCH``; the
#: next-channel predictor (Fixed-Share + universal) re-points on its own, and the
#: regime monitor flushes its memory so the re-point is fast.
SWITCH_SCENARIOS = [
    {"id": "lfsr_to_markov", "name": "Register → statistical", "kind": "Recipe change",
     "from": ("lfsr", {"length": 7}), "to": ("markov", {"alpha": 0.04}),
     "from_label": "Shift register (LFSR)", "to_label": "Statistical hopper",
     "blurb": "A shift-register radio (predicted near-perfectly) switches to a "
              "statistical hopper. Accuracy dives at the switch, then recovers."},
    {"id": "chaos_to_markov", "name": "Chaos → statistical", "kind": "Recipe change",
     "from": ("logistic", {"r": 3.95}), "to": ("markov", {"alpha": 0.04}),
     "from_label": "Chaotic map", "to_label": "Statistical hopper",
     "blurb": "A chaotic radio switches to a statistical hopper mid-engagement."},
]


#: detector tuned for the demo: one clean alarm near the switch, few false ones
SCENARIO_DETECTOR = {"h": 8.0, "warmup": 30, "k": 0.5, "ewma": 0.03}


def _pred_run(ch, sensor, specs, ep, *, monitor, window):
    """Run the next-channel predictor over an episode; return per-slot records."""
    pred = ChannelPredictor(ch, sensor, specs=specs, horizon=window + 4,
                            monitor=monitor, retain=0.0,
                            detector_kwargs=SCENARIO_DETECTOR if monitor else None)
    truth = np.asarray(ep.true_channels)
    fc, kappa, dom, hit = [], [], [], []
    for t, o in enumerate(ep.observations):
        pred.observe(o)
        nxt = pred.predict_next()
        fc.append([int(round(1000 * float(x))) for x in nxt])
        kappa.append(round(float(pred.predictability()), 4))
        dom.append(pred.dominant_expert())
        if t < window - 1:
            hit.append(int(int(np.argmax(nxt)) == int(truth[t + 1])))
    return {"fc": fc, "kappa": kappa, "dom": dom, "hit": hit, "alarms": list(pred.alarms)}


def record_switch_scenario(scenario: dict, p_d: float, p_fa: float, seed: int, *,
                           preset: str = "custom", window: int = SCENARIO_WINDOW,
                           switch_at: int = SCENARIO_SWITCH) -> dict:
    """Record one regime-switch engagement for the live demo.

    The predictor is run twice on the *same* observations: once with the regime
    monitor (memory flush on a detected switch), once without, so the demo can
    show that flushing the context re-points the prediction faster.
    """
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    specs = prediction_expert_specs(M)
    rng = np.random.default_rng(seed)
    ep = make_switch_episode([scenario["from"], scenario["to"]], ch, sensor, window, rng,
                             switch_at=switch_at)
    flush = _pred_run(ch, sensor, specs, ep, monitor=True, window=window)
    plain = _pred_run(ch, sensor, specs, ep, monitor=False, window=window)
    truth = np.asarray(ep.true_channels)

    def _acc(hit, start):
        xs = hit[start:]
        return round(float(np.mean(xs)), 4) if xs else None

    return {
        "emitter": scenario["id"], "scenario": True, "preset": preset, "seed": int(seed),
        "p_d": p_d, "p_fa": p_fa, "M": M, "window": window,
        "switch_at": int(switch_at),
        "from_label": scenario["from_label"], "to_label": scenario["to_label"],
        "alarms": flush["alarms"],
        "truth": [int(c) for c in truth],
        "dets": [sorted(int(c) for c in o.detected_channels) for o in ep.observations],
        # the flush predictor's forecast drives the hop chart + strip (fc / fc_pred)
        "fc": flush["fc"], "fc_pred": flush["fc"],
        "kappa_pred": flush["kappa"], "dom": flush["dom"],
        "hit_flush": flush["hit"], "hit_noflush": plain["hit"],
        "acc_flush_after": _acc(flush["hit"], switch_at),
        "acc_noflush_after": _acc(plain["hit"], switch_at),
        "acc_before": _acc(flush["hit"][:switch_at], 0),
        # bank-centric fields the shared live code may touch (kept inert here)
        "labels": ["H0"], "target": "", "declared": False, "tau": None,
        "decision": "unknown", "correct": None, "new_algo": False,
        "post": [], "alive": [], "fc_bank": [],
        "secret_true": None, "secret_recovered": None, "secret_match": None,
        "baseline": None,
    }


# ---------------------------------------------------------------- scoreboard
def _rate(xs) -> float | None:
    return round(float(np.mean(xs)), 4) if xs else None


def score_episode(bank: HypothesisBank, ep: Episode) -> tuple:
    """One full-window run: the ``run_bank`` decision plus next-hop hits per slot.

    The decision is formed exactly as :func:`fhew.sim.run_bank` forms it (the
    first declaration, else the MAP at the window end); running past the
    declaration only adds the forecasts made after the lock.
    """
    bank.reset()
    true = ep.true_channels
    hits = []
    for t, o in enumerate(ep.observations):
        bank.update(o)
        if t < ep.window - 1:
            hits.append(int(np.argmax(bank.predict())) == int(true[t + 1]))
    if bank.declared:
        label, tau = bank.declared_identity(), int(bank.tau_stop)
    else:
        top = int(np.argmax(bank.posterior()))
        label, tau = ("unknown" if top == 0 else bank.labels[top]), ep.window
    res = decide_outcome(bank.label, ep, label, bank.declared, tau)
    # forecasts made after the slot that declared (hits[t] predicts slot t + 1)
    after = hits[tau - 1:] if bank.declared else []
    return res, hits, after


def markov_top1(ep: Episode, M: int) -> float:
    """Top-1 of a one-step-ahead online Markov predictor fitted on clean hops.

    The conventional frequency-statistics forecaster: it counts transitions
    between clean single-detection slots and calls the most frequent successor
    of the latest clean channel.
    """
    counts = np.ones((M, M))
    last, hits = None, []
    true = ep.true_channels
    for t, o in enumerate(ep.observations[:-1]):
        if len(o.detections) == 1:
            c = o.detections[0].channel
            if last is not None:
                counts[last, c] += 1
            last = c
        hits.append(last is not None and int(np.argmax(counts[last])) == int(true[t + 1]))
    return float(np.mean(hits))


def predict_top1(predictor: ChannelPredictor, ep: Episode) -> tuple[float, float]:
    """Next-channel top-1 of the Fixed-Share predictor: (whole run, steady state).

    Steady state is the second half of the window, after the universal predictor
    has had time to learn an unseen rule.
    """
    predictor.reset()
    true = ep.true_channels
    half = ep.window // 2
    whole, steady = [], []
    for t, o in enumerate(ep.observations):
        predictor.observe(o)
        if t + 1 >= ep.window:
            continue
        hit = int(predictor.predict_top1() == int(true[t + 1]))
        whole.append(hit)
        if t >= half:
            steady.append(hit)
    return (float(np.mean(whole)) if whole else 0.0,
            float(np.mean(steady)) if steady else 0.0)


def scoreboard(n_seeds: int, *, presets=PRESETS, emitters=EMITTERS, window: int = WINDOW,
               progress=None) -> dict:
    """Monte-Carlo scores per emitter and preset: bank vs pairwise SPRT, plus next-hop."""
    rows = []
    for pr in presets:
        engine = Engine.build(pr["p_d"], pr["p_fa"], window=window)
        for em in emitters:
            ots, base, top1, top1_lock, mk, pred, pred_steady = [], [], [], [], [], [], []
            for s in C.seeds(n_seeds):
                ep, _ = draw_episode(engine, em["id"], s)
                res, hits, after = score_episode(engine.bank, ep)
                ots.append(res)
                top1.append(float(np.mean(hits)))
                if after:
                    top1_lock.append(float(np.mean(after)))
                mk.append(markov_top1(ep, M))
                pw, ps = predict_top1(engine.predictor, ep)
                pred.append(pw); pred_steady.append(ps)
                base.append(run_sprt(engine.sprt, ep))
            hops = [int(r.tau_stop) for r in ots if r.correct and r.declared]
            rows.append({
                "emitter": em["id"], "preset": pr["id"], "n": len(ots),
                "new_algo": bool(em.get("new_algo")),
                "ots_acc": _rate([r.correct for r in ots]),
                "ots_false_break": _rate([r.decision_label in ALGEBRAIC_FAMILIES
                                          and not r.correct for r in ots]),
                "ots_unknown": _rate([r.decision_label == "unknown" for r in ots]),
                "ots_locked": _rate([r.declared for r in ots]),
                "ots_hops": hops,
                "ots_top1": _rate(top1),
                "ots_top1_locked": _rate(top1_lock),
                "markov_top1": _rate(mk),
                # the Fixed-Share predictor: the system's actual next-channel output
                "pred_top1": _rate(pred),
                "pred_top1_steady": _rate(pred_steady),
                "base_acc": _rate([r.correct for r in base]),
                "base_false_break": _rate([r.decision_label in ALGEBRAIC_FAMILIES
                                           and not r.correct for r in base]),
            })
            if progress is not None:
                progress(pr["id"], em["id"], rows[-1])
    return {"n_seeds": n_seeds, "window": window, "chance_top1": round(1.0 / M, 4), "rows": rows}


def search_sizes(engine: Engine) -> dict:
    """Candidate parameterisations each expert enumerates (0 for algebraic/statistical)."""
    return {lab: _n_candidates(h) for lab, h in zip(engine.bank.labels, engine.bank.hypotheses)}
