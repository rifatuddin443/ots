"""Run logging for metrics and journal reuse (paper Section 12, rule 2).

Every run logs the posterior and the next-hop distribution per slot, even though
the conference paper only reports identification metrics, so the journal reuses
these runs without re-simulating. Records are written as JSON Lines and can also
be kept in memory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return [_to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if is_dataclass(x) and not isinstance(x, type):
        return _to_jsonable(asdict(x))
    return x


class RunLogger:
    """Collects per-slot records; optionally streams them to a JSONL file."""

    def __init__(self, path: str | Path | None = None, keep_in_memory: bool = True):
        self.path = Path(path) if path is not None else None
        self.keep_in_memory = keep_in_memory
        self.records: list[dict] = []
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w")

    def log(self, record: dict) -> None:
        rec = _to_jsonable(record)
        if self.keep_in_memory:
            self.records.append(rec)
        if self._fh is not None:
            self._fh.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(_to_jsonable(obj), fh, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open() as fh:
        return json.load(fh)
