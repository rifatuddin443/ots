"""Control room: drive the transmitter yourself and watch the system follow it.

You choose the hop algorithm the radio runs, and switch it whenever you like in
the middle of the engagement. Every switch re-keys the radio with a fresh secret
drawn from the same reconnaissance ranges the dashboard's other pages use
(``experiments.common.draw_episode_params``), so a known algorithm is always
identifiable. The system is :class:`fhew.tracker.SwitchTracker`: it sees only the
sensor's detections and must notice the switch, name the new algorithm and keep
predicting the next hop. The truth (the algorithm on air, its secret, the true
channel) is recorded next to every hop for the audience only.

Used by ``python -m dashboard.serve`` (the Control room page) and runnable on
its own with a scripted switch plan::

    python -m dashboard.control --channels 16 --hops 300 --sensor perfect \\
        --plan lfsr:0 weak_seed:80 markov:160 lagged_fib:240
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from dashboard import traces as T
from experiments.common import draw_episode_params
from fhew.channels import ChannelSet
from fhew.config import GENERATOR_TARGET_FAMILY
from fhew.generators import make_generator
from fhew.sensor import SensorModel
from fhew.tracker import ExpertLibrary, SwitchTracker, is_power_of_two

MIN_CHANNELS, MAX_CHANNELS = 4, 64
MIN_HOPS, MAX_HOPS = 20, 1000

#: most transmit frequencies (jam tones) the control room will ever light at once;
#: the per-hop ``jam`` ranking is capped at this, and the UI's "max transmit freqs"
#: control runs 1..min(M, JAM_TONES_CAP).
JAM_TONES_CAP = 8

#: algorithms the transmitter can run (the dashboard's emitter list)
ALGORITHMS = [{**e, "target": GENERATOR_TARGET_FAMILY[e["id"]],
               "needs_pow2": e["id"] in ("lfsr", "filtered_lfsr")} for e in T.EMITTERS]
_ALGO_IDS = {a["id"] for a in ALGORITHMS}

#: sensor presets: a perfect sensor (every hop caught, no false alarms) + the dashboard's
SENSORS = [{"id": "perfect", "name": "Perfect", "p_d": 1.0, "p_fa": 0.0}] + T.PRESETS

#: algebraic families: naming one of these when it is not the truth is the dangerous error
ALGEBRAIC = T.ALGEBRAIC_FAMILIES

_LIBRARIES: dict[int, ExpertLibrary] = {}


def algorithms_for(n_channels: int) -> list[dict]:
    """The algorithm list with availability for a channel count."""
    pow2 = is_power_of_two(n_channels)
    return [{**a, "available": pow2 or not a["needs_pow2"],
             "reason": None if pow2 or not a["needs_pow2"]
             else "needs a power-of-two channel count (4, 8, 16, 32, 64)"} for a in ALGORITHMS]


def library_for(n_channels: int, hops: int) -> ExpertLibrary:
    """Pristine experts for ``n_channels``, cached and grown to fit ``hops``.

    The horizon is rounded up to a power of two so a few sizes cover every
    session; building the trajectory tables is the only slow step (seconds).
    """
    need = max(128, 1 << math.ceil(math.log2(hops + 2)))
    lib = _LIBRARIES.get(n_channels)
    if lib is None or lib.horizon < hops + 2:
        lib = ExpertLibrary.build(ChannelSet.of_size(n_channels), need)
        _LIBRARIES[n_channels] = lib
    return lib


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


class ControlSession:
    """One engagement whose transmitter algorithm you choose hop by hop."""

    def __init__(self, n_channels: int = 16, hops: int = 300, p_d: float = 1.0,
                 p_fa: float = 0.0, seed: int | None = None):
        if not MIN_CHANNELS <= n_channels <= MAX_CHANNELS:
            raise ValueError(f"channels must be in [{MIN_CHANNELS}, {MAX_CHANNELS}]")
        if not MIN_HOPS <= hops <= MAX_HOPS:
            raise ValueError(f"hops must be in [{MIN_HOPS}, {MAX_HOPS}]")
        if not (0.3 <= p_d <= 1.0 and 0.0 <= p_fa <= 0.2):
            raise ValueError("p_d must be in [0.3, 1], p_fa in [0, 0.2]")
        self.M, self.hops = int(n_channels), int(hops)
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % 2 ** 31)
        self.rng = np.random.default_rng(self.seed)
        self.channels = ChannelSet.of_size(self.M)
        self.sensor = SensorModel(p_d=p_d, p_fa=p_fa)
        t0 = time.time()
        self.tracker = SwitchTracker(self.channels, self.sensor, library=library_for(self.M, self.hops))
        self.build_seconds = round(time.time() - t0, 2)
        self.labels = ["unknown" if lab == "H0" else lab for lab in self.tracker.labels]
        self.t = 0
        self.algo: str | None = None
        self.gen = None
        self.truth: list[int] = []
        self.regimes: list[dict] = []     # what the transmitter ran, for the audience
        self.records: list[dict] = []

    # ---------------------------------------------------------------- metadata
    def info(self) -> dict:
        return {
            "channels": self.M, "hops": self.hops, "seed": self.seed,
            "p_d": self.sensor.p_d, "p_fa": self.sensor.p_fa,
            "labels": self.labels, "algorithms": algorithms_for(self.M),
            "segments_protected": self.tracker.protect, "build_seconds": self.build_seconds,
            "jam_cap": min(self.M, JAM_TONES_CAP),
        }

    @property
    def done(self) -> bool:
        return self.t >= self.hops

    # --------------------------------------------------------------- the radio
    def _switch(self, algo: str) -> dict:
        if algo not in _ALGO_IDS:
            raise ValueError(f"unknown algorithm {algo!r}")
        if not next(a for a in algorithms_for(self.M) if a["id"] == algo)["available"]:
            raise ValueError(f"{algo} needs a power-of-two channel count")
        params = draw_episode_params(algo, self.rng)
        self.gen = make_generator(algo, self.M, rng=self.rng, **params)
        self.algo = algo
        regime = {"algo": algo, "start_hop": self.t + 1, "target": GENERATOR_TARGET_FAMILY[algo],
                  "secret": T.true_secret(self.gen)}
        self.regimes.append(regime)
        return regime

    def step(self, algo: str, n: int = 1) -> list[dict]:
        """Advance ``n`` hops with the transmitter on ``algo`` (switching if it changed)."""
        out = []
        for _ in range(max(1, int(n))):
            if self.done:
                break
            switched = algo != self.algo
            if switched:
                self._switch(algo)
            out.append(self._hop(switched))
        return out

    def _hop(self, switched: bool) -> dict:
        tr = self.tracker
        p = tr.predict_next()
        call = int(np.argmax(p)) if self.t > 0 else None
        top3 = [int(c) for c in np.argsort(p)[::-1][:3]] if self.t > 0 else []
        c = int(self.gen.next_channel())
        obs = self.sensor.observe([c], self.channels, self.rng, start_slot=self.t)[0]
        st = tr.observe(obs)
        self.truth.append(c)
        idn = st.ident
        label = "unknown" if idn.label == "H0" else idn.label
        nxt = int(np.argmax(st.forecast))
        # Jam ranking: the most-likely next channels, best first. When the system
        # is unsure (low kappa) the UI lights several of these at once -- a
        # multi-tone / multiple-frequency jam, capped by the "max transmit freqs"
        # control; a confident forecast collapses it to the single top channel.
        jam = [int(c) for c in np.argsort(st.forecast)[::-1][:min(self.M, JAM_TONES_CAP)]]
        rec = {
            "hop": self.t + 1, "algo": self.algo, "switched": switched,
            "truth": c, "dets": sorted(int(d) for d in obs.detected_channels),
            "call": call, "call_p": _r(p[call], 3) if call is not None else None,
            "hit": (call == c) if call is not None else None,
            "top3_hit": (c in top3) if call is not None else None,
            "next": nxt, "next_p": _r(st.forecast[nxt], 3), "kappa": _r(tr.predictability(), 3),
            "jam": jam,
            "id": {"label": label, "p": _r(idn.p), "locked": idn.locked,
                   "lock_hop": idn.lock_slot + 1 if idn.lock_slot is not None else None,
                   "start_hop": idn.start_slot + 1, "carrier": idn.carrier},
            "post": [_r(x, 3) for x in idn.posterior],
            "switch_detected": st.switch_detected, "p_switch": _r(st.p_switch, 3),
        }
        if switched:
            rec["regime"] = self.regimes[-1]
        if idn.locked and label != "unknown":
            rec["id"]["recovered"], rec["id"]["match"] = self._recovered(idn.label)
        self.t += 1
        self.records.append(rec)
        return rec

    def _recovered(self, bank_label: str) -> tuple[str | None, str | None]:
        """The leading segment's recovered secret, checked against the true one."""
        seg = self.tracker.leading_segment()
        h = seg.bank.hypotheses[seg.bank.labels.index(bank_label)]
        truth = np.asarray(self.truth[seg.birth:], dtype=np.int64)
        return T.recovered_secret(h, truth, self.regimes[-1]["secret"])

    # ---------------------------------------------------------------- summary
    def summary(self) -> list[dict]:
        """Per transmitter regime: detection delay, identification, accuracy."""
        out = []
        recs = self.records
        for i, reg in enumerate(self.regimes):
            lo = reg["start_hop"] - 1
            hi = self.regimes[i + 1]["start_hop"] - 1 if i + 1 < len(self.regimes) else len(recs)
            rs = recs[lo:hi]
            if not rs:
                continue
            det = next((r["hop"] for r in rs if r["switch_detected"]), None) if i > 0 else None
            from_hop = det if det is not None else (reg["start_hop"] if i == 0 else None)
            lock = None
            if from_hop is not None:
                lock = next((r["id"]["lock_hop"] for r in rs if r["hop"] >= from_hop
                             and r["id"]["locked"] and r["id"]["label"] == reg["target"]), None)
            final = rs[-1]["id"]
            hits = [r["hit"] for r in rs if r["hit"] is not None]
            tail = hits[-30:]
            out.append({
                "algo": reg["algo"], "start_hop": reg["start_hop"], "end_hop": rs[-1]["hop"],
                "target": reg["target"], "secret": reg["secret"],
                "detected_hop": det, "detect_delay": det - reg["start_hop"] if det else None,
                "final_label": final["label"], "final_p": final["p"], "final_locked": final["locked"],
                "id_correct": final["label"] == reg["target"],
                "lock_hops": lock - reg["start_hop"] + 1 if lock else None,
                "top1": _r(np.mean(hits)) if hits else None,
                "top1_last30": _r(np.mean(tail)) if tail else None,
            })
        return out


# ---------------------------------------------------------------------- CLI
def _parse_plan(items: list[str]) -> list[tuple[str, int]]:
    plan = []
    for it in items:
        algo, _, hop = it.partition(":")
        plan.append((algo, int(hop or 0)))
    plan.sort(key=lambda x: x[1])
    if not plan or plan[0][1] != 0:
        raise SystemExit("the plan must start at hop 0, e.g. lfsr:0 markov:100")
    return plan


def run_plan(session: ControlSession, plan: list[tuple[str, int]]) -> list[dict]:
    """Drive a session through ``[(algo, from_hop), ...]`` to the end; return its summary."""
    for i, (algo, start) in enumerate(plan):
        end = plan[i + 1][1] if i + 1 < len(plan) else session.hops
        session.step(algo, max(0, end - start))
    return session.summary()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--hops", type=int, default=300)
    p.add_argument("--sensor", default="perfect", choices=[s["id"] for s in SENSORS] + ["custom"])
    p.add_argument("--p-d", type=float, help="custom detection probability")
    p.add_argument("--p-fa", type=float, help="custom false-alarm rate per channel")
    p.add_argument("--seed", type=int)
    p.add_argument("--plan", nargs="+", default=["lfsr:0", "weak_seed:80", "markov:160", "lagged_fib:240"],
                   help="algo:from_hop items, e.g. lfsr:0 markov:100")
    args = p.parse_args(argv)
    preset = next((s for s in SENSORS if s["id"] == args.sensor), {"p_d": 0.9, "p_fa": 0.01})
    p_d = args.p_d if args.p_d is not None else preset["p_d"]
    p_fa = args.p_fa if args.p_fa is not None else preset["p_fa"]
    t0 = time.time()
    s = ControlSession(args.channels, args.hops, p_d, p_fa, seed=args.seed)
    rows = run_plan(s, _parse_plan(args.plan))
    print(f"[control] {args.channels} channels, {args.hops} hops, P_d={p_d} P_fa={p_fa}, "
          f"seed {s.seed} ({time.time() - t0:.1f}s)")
    print(f"{'hops':>11}  {'transmitter':14} {'detected':>9} {'identified as':16} {'locked':>7} {'top-1':>6} {'last 30':>8}")
    for r in rows:
        det = "-" if r["start_hop"] == 1 else (f"+{r['detect_delay']}" if r["detect_delay"] is not None else "missed")
        ok = "ok" if r["id_correct"] else ("~" if r["target"] == "unknown" and r["final_label"] not in ALGEBRAIC else "X")
        lock = f"{r['lock_hops']} hops" if r["lock_hops"] else "-"
        print(f"{r['start_hop']:>4}-{r['end_hop']:<6} {r['algo']:14} {det:>9} {r['final_label'][:13]:13} {ok:2} "
              f"{lock:>7} {r['top1'] or 0:6.2f} {r['top1_last30'] or 0:8.2f}")
    hits = [r["hit"] for r in s.records if r["hit"] is not None]
    print(f"next-hop top-1 over the run: {np.mean(hits):.2f}  (random guessing: {1 / s.M:.2f})")


if __name__ == "__main__":
    main()
