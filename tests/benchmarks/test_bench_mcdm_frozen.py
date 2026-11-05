"""bulanik against scores frozen from pymcdm and pyrepo-mcda; versions are in the result file."""

import json
from pathlib import Path

import pytest
from _bench_support import spearman
from _mcdm_cases import bulanik_score

import bulanik.mcdm as bm

DATA = json.loads((Path(__file__).parent / "data" / "mcdm_results.json").read_text(encoding="utf-8"))
FROZEN = DATA["frozen"]
LIBS = ("pymcdm", "pyrepo-mcda")
AGREE = ["topsis/vector", "vikor", "codas", "waspas", "edas", "marcos", "cocoso", "entropy", "critic"]


def _ours(method, p):
    return bulanik_score(method.split("/")[0], p["matrix"], p["weights"], p["types"])


def test_provenance_is_recorded():
    who = DATA["produced_by"]
    assert who["pymcdm"] == "1.4.0" and who["pyrepo-mcda"] == "0.1.15"
    assert all(r["classification"] != "UNCLASSIFIED" for r in DATA["comparison"])
    assert all(r["classification"] != "UNCLASSIFIED" for r in DATA["fixtures"])


@pytest.mark.parametrize("lib", LIBS)
@pytest.mark.parametrize("method", AGREE)
def test_same_scores_as_library(lib, method):
    for p, theirs in zip(FROZEN["problems"], FROZEN["scores"][lib][method]):
        assert _ours(method, p) == pytest.approx(theirs, abs=1e-9)


def test_copras_matches_pyrepo():
    for p, theirs in zip(FROZEN["problems"], FROZEN["scores"]["pyrepo-mcda"]["copras"]):
        assert _ours("copras", p) == pytest.approx(theirs, abs=1e-9)


def test_copras_matches_pymcdm_only_in_its_sum_form():
    for p, theirs in zip(FROZEN["problems"], FROZEN["scores"]["pymcdm"]["copras"]):
        r = bm.copras(p["matrix"], p["weights"], p["types"])
        q = [a + b for a, b in zip(r.extras["S_plus"].values(), r.extras["S_minus"].values())]
        assert [v / max(q) for v in q] == pytest.approx(theirs, abs=1e-9)


@pytest.mark.parametrize("lib", LIBS)
def test_library_default_topsis_is_a_different_normalisation(lib):
    gaps = [
        max(abs(a - b) for a, b in zip(_ours("topsis", p), theirs))
        for p, theirs in zip(FROZEN["problems"], FROZEN["scores"][lib]["topsis"])
    ]
    assert max(gaps) > 0.05


@pytest.mark.parametrize("lib", LIBS)
def test_ahp_weights(lib):
    for p, theirs in zip(FROZEN["pairwise"], FROZEN["scores"][lib]["ahp"]):
        got = list(bm.ahp(p["matrix"]).weights.values())
        assert got == pytest.approx(theirs, abs=1e-9)
        assert spearman(got, theirs) == pytest.approx(1.0)


def test_tie_order_keeps_input_order():
    ties = DATA["tie_handling"]
    assert ties["bulanik_ranking_positions"] == [3, 4, 1, 2]
    assert ties["pymcdm_rankdata"] == [3.5, 3.5, 1.0, 2.0]
    assert ties["pyrepo_rank_preferences"] == [3, 3, 1, 2]
