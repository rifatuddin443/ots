"""Record (and serve live) the RL *game* for the dashboard's Game page.

Two things the Game page shows, both produced here so the page works offline from
``data.js`` and live from the server:

* **Engagement replays.** Hop-by-hop traces of a hopper against the OTS predict-then-
  jam system: an oblivious LFSR that gets jammed almost every hop, a *learned* hopper
  that escapes it, and a keyed CSPRNG at the floor. Each hop logs the true channel,
  the jammer's lit tones, whether the hop was denied, and the jammer's confidence --
  enough to draw the chart and the running denial rate.
* **The exploitability certificate.** A PSRO run (:func:`fhew.rl.psro.run_psro`): the
  exploitability curve, the meta-Nash denial, and the final payoff matrix with its
  meta-Nash support -- an interactive version of ``exp28``.

The live server (:mod:`dashboard.serve`) reuses :func:`run_duel_trace` for a fresh
duel on demand and :class:`GameSession` to *step* PSRO one iteration at a time, so the
certificate curve can be watched growing rather than waiting a minute for a full run.
"""

from __future__ import annotations

import time
import uuid

import numpy as np

from fhew.allocator import EffectModel
from fhew.channels import ChannelSet
from fhew.duel import OTSJammer
from fhew.generators import make_generator
from fhew.rl.env import EngagementEnv, GeneratorHopper, HopperObs
from fhew.rl.hopper import OnlineBanditHopper
from fhew.sensor import SensorModel

M_DEFAULT = 16
BUDGET = 3.0
MAX_TONES = 4
DENY = 0.5

#: The hoppers the Game page can replay, each against the OTS jammer.
HOPPERS = {
    "lfsr_oblivious": {"name": "Oblivious LFSR", "kind": "scripted",
                       "blurb": "A 7-bit LFSR that ignores the jammer — fully predictable."},
    "learned_bandit": {"name": "Learned hopper", "kind": "learned",
                       "blurb": "An online-bandit radio that learns to dodge the jammer from denial feedback."},
    "evasion_lfsr": {"name": "Reactive LFSR", "kind": "scripted",
                     "blurb": "An LFSR wrapped in the hand-coded evasion rule (deflect / spread / re-key)."},
    "csprng": {"name": "Keyed CSPRNG", "kind": "scripted",
               "blurb": "Cryptographically keyed — unpredictable by design, already at the floor."},
}


def _env(M=M_DEFAULT, p_d=0.95, p_fa=0.01, window=56):
    ch = ChannelSet.of_size(M)
    sensor = SensorModel(p_d=p_d, p_fa=p_fa)
    effect = EffectModel(ch, tau=1.0)
    env = EngagementEnv(ch, sensor, effect, window=window)
    return ch, sensor, effect, env


def _make_hopper(hopper_id: str, ch: ChannelSet, seed: int):
    rng = np.random.default_rng(seed)
    if hopper_id == "lfsr_oblivious":
        return GeneratorHopper(make_generator("lfsr", ch.size, length=7, rng=rng))
    if hopper_id == "csprng":
        return GeneratorHopper(make_generator("csprng", ch.size, rng=rng))
    if hopper_id == "learned_bandit":
        return OnlineBanditHopper(ch, rng=rng)
    if hopper_id == "evasion_lfsr":
        from fhew.evasion import AdaptiveEvasionGenerator, EvasionConfig
        return GeneratorHopper(AdaptiveEvasionGenerator.from_family(
            "lfsr", ch, rng, base_params={"length": 7}, config=EvasionConfig()))
    raise ValueError(f"unknown hopper '{hopper_id}'")


def run_duel_trace(hopper_id: str, *, M=M_DEFAULT, p_d=0.95, p_fa=0.01, window=56,
                   seed: int = 0, ots: OTSJammer | None = None) -> dict:
    """Run one hopper-vs-OTS engagement and return a hop-by-hop trace.

    Resolved from history (the jammer's power for a hop was planned the hop before),
    so neither side sees the other's current move -- the same fair-game rule as
    :func:`fhew.rl.env.run_engagement`. The per-hop record carries the true channel,
    the jammer's lit tones, the realised denial, the jammer's next-hop call and its
    confidence (kappa) -- everything the page draws.
    """
    from fhew.rl.psro import fast_ots_specs

    ch, sensor, effect, _env_obj = _env(M, p_d, p_fa, window)
    rng = np.random.default_rng(1000 + seed)
    hopper = _make_hopper(hopper_id, ch, seed)
    if ots is None:
        ots = OTSJammer(ch, sensor, effect, budget=BUDGET, max_tones=MAX_TONES,
                        specs=fast_ots_specs())
    hopper.reset()
    ots.reset()

    power_cur = np.zeros(M, dtype=np.float64)
    c_cur = int(hopper.act(None))
    hops = []
    denied_n = 0
    hit_n = 0
    scored = 0
    for t in range(window):
        eff = float(effect.effect_on(power_cur, c_cur))
        denied = bool(eff >= DENY)
        lit = [int(i) for i in np.nonzero(power_cur > 1e-12)[0]]
        obs = sensor.observe([c_cur], ch, rng, start_slot=t)[0]
        ots.observe(obs)
        p = ots.predict_next()
        power_next, n_tones, kappa, _set = ots.plan_power(p, observed_channel=ots._clean(obs))
        top1 = int(np.argmax(p))
        hops.append({"t": t, "ch": int(c_cur), "lit": lit, "den": denied,
                     "kappa": round(float(kappa), 3), "tones": int(n_tones),
                     "next": top1, "detected": sorted(int(c) for c in obs.detected_channels)})
        if t >= 1:
            denied_n += int(denied)
            scored += 1
        fb = HopperObs(slot=t, denied=denied, effect_on_self=eff, last_channel=c_cur,
                       jammed_mask=(power_cur > 1e-12))
        c_next = int(hopper.act(fb)) if t < window - 1 else c_cur
        # did the jammer's PREVIOUS next-hop call land on this realised channel?
        if t >= 1 and hops[t - 1]["next"] == c_cur:
            hit_n += 1
        power_cur = power_next
        c_cur = c_next

    denied_frac = denied_n / scored if scored else 0.0
    return {
        "hopper": hopper_id, "name": HOPPERS[hopper_id]["name"],
        "kind": HOPPERS[hopper_id]["kind"], "blurb": HOPPERS[hopper_id]["blurb"],
        "M": M, "window": window, "p_d": p_d, "p_fa": p_fa, "seed": seed,
        "hops": hops,
        "summary": {"denied_frac": round(denied_frac, 3),
                    "nexthop_top1": round(hit_n / scored, 3) if scored else 0.0,
                    "kappa_mean": round(float(np.mean([h["kappa"] for h in hops])), 3)},
    }


def _psro_floor(res, ch, sensor, effect, window, n_seeds) -> float:
    """CSPRNG denial against the final jammer meta (the boundary floor)."""
    from fhew.rl.env import OTSJammerPolicy, ScriptedJammer, run_engagement
    from fhew.rl.psro import fast_ots_specs

    env = EngagementEnv(ch, sensor, effect, window=window)
    sj = np.asarray(res.final_sigma_jammer)
    val, wsum = 0.0, 0.0
    for lab, w in zip(res.jammer_labels, sj):
        if w <= 1e-6 or lab.startswith("rl_jammer"):
            continue
        if lab == "spot":
            jam = ScriptedJammer(ch, effect, BUDGET, mode="spot")
        elif lab == "barrage":
            jam = ScriptedJammer(ch, effect, BUDGET, mode="barrage", n_tones=MAX_TONES)
        elif lab == "ots_knapsack":
            jam = OTSJammerPolicy(OTSJammer(ch, sensor, effect, budget=BUDGET,
                                           max_tones=MAX_TONES, specs=fast_ots_specs()))
        else:
            continue
        ds = [run_engagement(GeneratorHopper(make_generator("csprng", ch.size,
              rng=np.random.default_rng(900 + s))), jam, env, np.random.default_rng(900 + s)
              ).jammer_payoff() for s in range(n_seeds)]
        val += w * float(np.mean(ds)); wsum += w
    return float(val / wsum) if wsum > 0 else float("nan")


def record_psro(*, M=M_DEFAULT, window=64, n_iters=5, n_seeds=5, quick=False,
                verbose=True) -> dict:
    """Record a PSRO run (exploitability certificate + payoff game) for the page."""
    from fhew.rl.agents import PPOConfig
    from fhew.rl.psro import run_psro

    if quick:
        n_iters, n_seeds, window = 3, 4, 56
    ch, sensor, effect, _ = _env(M, window=window)
    br = {"n_iters": 8 if quick else 14, "episodes_per_iter": 8 if quick else 12,
          "cfg": PPOConfig()}
    t0 = time.time()
    res = run_psro(ch, sensor, effect, budget=BUDGET, max_tones=MAX_TONES, window=window,
                   n_iters=n_iters, n_seeds=n_seeds, br=br, include_unpredictable=False,
                   seed=1, progress=verbose)
    floor = _psro_floor(res, ch, sensor, effect, window, n_seeds)
    if verbose:
        print(f"  [game.psro] {time.time() - t0:.1f}s  expl {res.exploitability[0]:.3f}"
              f" -> {res.exploitability[-1]:.3f}")
    return {
        "iterations": res.iterations, "exploitability": res.exploitability,
        "jammer_gain": res.jammer_gain, "hopper_gain": res.hopper_gain,
        "meta_value": res.meta_value, "ots_vs_meta_hopper": res.ots_vs_meta_hopper,
        "payoff_matrix": res.payoff_matrix, "hopper_labels": res.hopper_labels,
        "jammer_labels": res.jammer_labels, "final_sigma_hopper": res.final_sigma_hopper,
        "final_sigma_jammer": res.final_sigma_jammer, "csprng_floor": floor,
        "window": window, "n_seeds": n_seeds,
    }


def record_game(*, quick=False, verbose=True) -> dict:
    """The whole Game block for ``data.js``: duel replays + the PSRO certificate."""
    window = 48 if quick else 56
    duels = []
    ch, sensor, effect, _ = _env(window=window)
    from fhew.rl.psro import fast_ots_specs
    ots = OTSJammer(ch, sensor, effect, budget=BUDGET, max_tones=MAX_TONES,
                    specs=fast_ots_specs())
    order = ["lfsr_oblivious", "learned_bandit", "evasion_lfsr", "csprng"]
    for hid in order:
        tr = run_duel_trace(hid, window=window, seed=0, ots=ots)
        duels.append(tr)
        if verbose:
            s = tr["summary"]
            print(f"  [game.duel] {hid:15s} denied {s['denied_frac']:.2f}  "
                  f"kappa {s['kappa_mean']:.2f}")
    psro = record_psro(window=64 if not quick else 56, quick=quick, verbose=verbose)
    return {"duels": duels, "psro": psro, "hoppers": HOPPERS,
            "budget": BUDGET, "max_tones": MAX_TONES, "deny_threshold": DENY}


# ---------------------------------------------------------------- live PSRO sessions
_SESSIONS: dict[str, "GameSession"] = {}


class GameSession:
    """A live PSRO game that advances one iteration per :meth:`step` (for the server)."""

    def __init__(self, *, M=M_DEFAULT, window=56, n_seeds=4, seed=1):
        from fhew.rl.psro import EmpiricalGame, initial_populations

        self.id = uuid.uuid4().hex[:12]
        self.ch, self.sensor, self.effect, _ = _env(M, window=window)
        self.env = EngagementEnv(self.ch, self.sensor, self.effect, window=window)
        self.game = EmpiricalGame(self.env, n_seeds=n_seeds)
        hoppers, jammers = initial_populations(self.ch, self.sensor, self.effect, BUDGET,
                                               max_tones=MAX_TONES, include_unpredictable=False,
                                               rng=np.random.default_rng(seed))
        for j in jammers:
            self.game.jammers.append(j)
        self.game.U = np.zeros((0, len(self.game.jammers)))
        for h in hoppers:
            self.game.add_hopper(h)
        self.rng = np.random.default_rng(seed)
        self.iters: list[int] = []
        self.exploitability: list[float] = []
        self.meta_value: list[float] = []
        self._it = 0
        _SESSIONS[self.id] = self

    def snapshot(self) -> dict:
        from fhew.rl.psro import solve_zero_sum
        sh, sj = solve_zero_sum(self.game.U) if self.game.U.size else ([], [])
        return {"session": self.id, "iter": self._it, "iterations": self.iters,
                "exploitability": self.exploitability, "meta_value": self.meta_value,
                "payoff_matrix": self.game.U.tolist(),
                "hopper_labels": self.game.hopper_labels(),
                "jammer_labels": self.game.jammer_labels(),
                "sigma_hopper": list(sh), "sigma_jammer": list(sj),
                "done": self._it >= 6}

    def step(self) -> dict:
        from fhew.rl.agents import PPOConfig
        from fhew.rl.psro import (Strategy, exploitability, solve_zero_sum,
                                  train_hopper_br, train_jammer_br)
        br = {"n_iters": 10, "episodes_per_iter": 10, "cfg": PPOConfig()}
        rep = exploitability(self.game, self.ch, self.effect, BUDGET, max_tones=MAX_TONES,
                             br=br, rng=np.random.default_rng(self.rng.integers(1 << 30)))
        sh, sj = solve_zero_sum(self.game.U)
        self.iters.append(self._it)
        self.exploitability.append(round(rep.exploitability, 4))
        self.meta_value.append(round(rep.meta_value, 4))
        h_br = train_hopper_br(self.game, sj, self.ch, br=br,
                               rng=np.random.default_rng(self.rng.integers(1 << 30)))
        j_br = train_jammer_br(self.game, sh, self.ch, self.effect, BUDGET, max_tones=MAX_TONES,
                               br=br, rng=np.random.default_rng(self.rng.integers(1 << 30)))
        self.game.add_hopper(Strategy(f"rl_hopper_{self._it}", h_br, "hopper", learned=True))
        self.game.add_jammer(Strategy(f"rl_jammer_{self._it}", j_br, "jammer", learned=True))
        self._it += 1
        return self.snapshot()


def psro_new(query: dict) -> dict:
    sess = GameSession()
    return sess.snapshot()


def psro_step(query: dict) -> dict:
    sid = (query.get("session") or [""])[0] if isinstance(query.get("session"), list) else query.get("session")
    sess = _SESSIONS.get(sid)
    if sess is None:
        sess = GameSession()
    if sess._it >= 6:
        return sess.snapshot()
    return sess.step()


def duel_request(query: dict) -> dict:
    def _q(k, default):
        v = query.get(k, default)
        return v[0] if isinstance(v, list) else v
    hopper = _q("hopper", "learned_bandit")
    if hopper not in HOPPERS:
        hopper = "learned_bandit"
    seed = int(_q("seed", 0))
    window = max(24, min(120, int(_q("window", 56))))
    return run_duel_trace(hopper, window=window, seed=seed)
