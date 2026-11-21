"""The dashboard's ``paper`` block (Proof and How-it-works pages).

``dashboard/paper.py`` condenses the paper's saved result JSON for the pages. These
checks need only the standard library, like the module itself: the block must carry
every saved result, agree with the raw files, and survive a ``--reuse`` refresh of
``data.js`` without touching the recorded runs.
"""

from __future__ import annotations

import json

import pytest

from dashboard import paper as PAPER

RAW = PAPER.PAPER_DATA


def _raw(name):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def block():
    return PAPER.load()


def test_block_has_every_saved_result(block):
    for key, (name, _) in PAPER._FILES.items():
        if (RAW / name).exists():
            assert key in block, key
            assert block["files"][key] == name
    assert set(block["sensors"]) == {"perfect", "strong", "typical", "contested"}
    assert block["documented"]["episodes"]["total"] == sum(p["n"] for p in block["documented"]["episodes"]["parts"])


def test_confusion_rows_count_every_run(block):
    c = block["confusion"]
    for s in c["by_sensor"].values():
        assert len(s["counts"]) == len(c["generators"])
        for row in s["counts"]:
            assert len(row) == len(c["families"])
            assert sum(row) == c["n_seeds"]


def test_numbers_match_the_raw_files(block):
    raw = _raw("exp22_id_baselines.json")["sensors"]["typical"]["methods"]
    got = block["id_baselines"]["by_sensor"]["typical"]["methods"]
    for m in ("bank", "pairwise_sprt", "fixed_length", "learned"):
        assert got[m]["hard_oob"] == pytest.approx(raw[m]["hard_false_id_out_of_bank"]["mean"], abs=1e-4)
        assert got[m]["unknown"] == pytest.approx(raw[m]["csprng_unknown"]["mean"], abs=1e-4)
    raw = _raw("extra_predict.json")["sensors"]["typical"]["operating_point"]
    got = block["predict"]["by_sensor"]["typical"]
    for g, v in raw.items():
        assert got[g]["steady"] == pytest.approx(v["proposed_steady"], abs=1e-4)
        assert got[g]["id_only"] == pytest.approx(v["id_only_steady"], abs=1e-4)
    raw = _raw("exp24_denial.json")["sensors"]["typical"]["generators"]
    got = block["denial"]["by_sensor"]["typical"]
    for g, v in raw.items():
        assert len(got[g]["denied"]) == len(block["denial"]["budgets"])
        assert got[g]["mean_tones"] == pytest.approx(v["mean_tones"], abs=1e-3)


def test_documented_results_are_consistent(block):
    cv = block["documented"]["converse"]
    assert len(cv["p_d"]) == len(cv["measured"]) == len(cv["bound"]) == len(cv["drift"])
    assert all(m >= b for m, b in zip(cv["measured"], cv["bound"])), "measured cost must sit at or above the floor"
    for r in block["documented"]["evasion"]["rows"]:
        assert 0 <= r["denied_evading"] <= r["denied_passive"] <= 1
    for doc in ("converse", "evasion", "deception"):
        assert block["documented"][doc]["source"]


def test_reuse_refreshes_only_the_paper_block(tmp_path, monkeypatch):
    from dashboard import build as B

    data = {"version": 3, "generated": "2000-01-01", "traces": [{"x": 1}], "paper": {"stale": True}}
    path = tmp_path / "data.js"
    path.write_text(B.to_js(data), encoding="utf-8")
    monkeypatch.setattr(B, "DATA_JS", path)
    B.main(["--reuse", "-q"])
    out = B.read_data_js(path)
    assert out["traces"] == data["traces"] and out["generated"] == "2000-01-01"
    assert "stale" not in out["paper"] and "confusion" in out["paper"]
