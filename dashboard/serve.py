"""Serve the dashboard and run new engagements live through the real bank.

Usage:
    python -m dashboard.serve                 # http://127.0.0.1:8050
    python -m dashboard.serve --port 9000 --open

Static files come from ``dashboard/``; ``GET /api/run`` records one engagement
on demand (``emitter``, ``p_d``, ``p_fa``, ``seed``) and returns the same trace
format ``data.js`` holds. The server is single-threaded on purpose: banks are
stateful and are reused across requests for the same sensor condition.

The Control room page drives a transmitter interactively (``dashboard.control``):
``GET /api/control/new`` (``channels``, ``hops``, ``p_d``, ``p_fa``, ``seed``)
opens a session and ``GET /api/control/step`` (``session``, ``algo``, ``n``)
advances it ``n`` hops with the transmitter on ``algo``, switching it when the
algorithm changed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import webbrowser
from collections import OrderedDict
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from dashboard import control as CR
from dashboard import traces as T
from dashboard.build import HERE

_EMITTER_IDS = {e["id"] for e in T.EMITTERS}
_ENGINES: OrderedDict = OrderedDict()
_MAX_ENGINES = 6


def engine_for(p_d: float, p_fa: float) -> T.Engine:
    key = (round(p_d, 3), round(p_fa, 3))
    if key in _ENGINES:
        _ENGINES.move_to_end(key)
    else:
        _ENGINES[key] = T.Engine.build(*key)
        if len(_ENGINES) > _MAX_ENGINES:
            _ENGINES.popitem(last=False)
    return _ENGINES[key]


def run_request(query: dict) -> dict:
    emitter = query.get("emitter", ["lfsr"])[0]
    if emitter not in _EMITTER_IDS:
        raise ValueError(f"unknown emitter {emitter!r}")
    p_d = float(query.get("p_d", ["0.8"])[0])
    p_fa = float(query.get("p_fa", ["0.02"])[0])
    seed = int(query.get("seed", ["1000"])[0])
    if not (0.3 <= p_d <= 1.0 and 0.0 <= p_fa <= 0.2 and 0 <= seed < 2 ** 31):
        raise ValueError("p_d must be in [0.3, 1], p_fa in [0, 0.2], seed in [0, 2^31)")
    return T.record_engagement(engine_for(p_d, p_fa), emitter, seed, preset="live")


_SESSIONS: OrderedDict = OrderedDict()
_MAX_SESSIONS = 4
_SESSION_IDS = itertools.count(1)


def _arg(query: dict, key: str, default: str) -> str:
    return query.get(key, [default])[0]


def control_new(query: dict) -> dict:
    session = CR.ControlSession(
        n_channels=int(_arg(query, "channels", "16")), hops=int(_arg(query, "hops", "300")),
        p_d=float(_arg(query, "p_d", "1")), p_fa=float(_arg(query, "p_fa", "0")),
        seed=int(query["seed"][0]) if "seed" in query else None)
    sid = f"s{next(_SESSION_IDS)}"
    _SESSIONS[sid] = session
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    return {"session": sid, **session.info()}


def control_step(query: dict) -> dict:
    sid = _arg(query, "session", "")
    if sid not in _SESSIONS:
        raise ValueError("unknown or expired session; start a new one")
    _SESSIONS.move_to_end(sid)
    session = _SESSIONS[sid]
    n = max(1, min(25, int(_arg(query, "n", "1"))))
    records = session.step(_arg(query, "algo", "lfsr"), n)
    return {"session": sid, "t": session.t, "done": session.done, "records": records}


_API = {"/api/run": run_request, "/api/control/new": control_new, "/api/control/step": control_step}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server naming
        url = urlparse(self.path)
        if url.path == "/api/ping":
            return self._json(200, {"live": True})
        if url.path in _API:
            try:
                return self._json(200, _API[url.path](parse_qs(url.query)))
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
        if url.path.endswith(".py") or "/__pycache__" in url.path:
            return self.send_error(404)
        return super().do_GET()

    def log_request(self, code="-", size="-"):
        if self.path.startswith("/api/"):
            super().log_request(code, size)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = p.parse_args(argv)
    print("[dashboard] warming up banks for the sensor presets ...")
    for pr in T.PRESETS:
        engine_for(pr["p_d"], pr["p_fa"])
    print("[dashboard] building the control-room experts for 16 channels ...")
    CR.library_for(16, 300)
    server = HTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[dashboard] live at {url}  (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
