"""bulanik against the expected values in the pymcdm 1.4.0 and pyrepo-mcda 0.1.15 test suites."""

import pytest
from _mcdm_cases import FIXTURES, bulanik_score

import bulanik.mcdm as bm

KNOWN_DIFFERENT = {"pymcdm-copras-1", "pymcdm-copras-2", "pymcdm-topsis", "pymcdm-waspas"}
BY_ID = {f["id"]: f for f in FIXTURES}
AGREEING = [f for f in FIXTURES if f["id"] not in KNOWN_DIFFERENT]


def _score(f):
    """Rounded study weights need normalize_weights when they miss 1 by more than the tolerance."""
    try:
        return bulanik_score(f["method"], f["matrix"], f["weights"], f["types"], **f["params"])
    except ValueError as exc:
        assert "weights must sum to 1" in str(exc)
        return bulanik_score(f["method"], f["matrix"], f["weights"], f["types"], normalize_weights=True, **f["params"])


@pytest.mark.parametrize("f", AGREEING, ids=[f["id"] for f in AGREEING])
def test_fixture_within_printed_rounding(f):
    got = _score(f)
    tol = 0.5 * 10 ** -f["decimals"] + 1e-9
    assert all(abs(a - b) <= tol for a, b in zip(got, f["expected"]))


@pytest.mark.parametrize("fid", ["pymcdm-copras-1", "pymcdm-copras-2"])
def test_pymcdm_copras_fixture_is_its_sum_form(fid):
    """pymcdm's Q reduces to S+ + S-; rebuilt from bulanik's extras it matches, the default does not."""
    f = BY_ID[fid]
    r = bm.copras(f["matrix"], f["weights"], f["types"])
    q = [a + b for a, b in zip(r.extras["S_plus"].values(), r.extras["S_minus"].values())]
    rebuilt = [v / max(q) for v in q]
    assert all(abs(a - b) <= 0.5e-4 + 1e-9 for a, b in zip(rebuilt, f["expected"]))
    assert max(abs(a - b) for a, b in zip(_score(f), f["expected"])) > 0.05


def test_pymcdm_topsis_fixture_is_min_max():
    f = BY_ID["pymcdm-topsis"]
    assert _score(f) == pytest.approx([0.76214, 0.72218, 0.23786], abs=1e-5)  # vector normalisation
    assert f["expected"] == [0.5, 0.617, 0.5]  # pymcdm's min-max default


def test_pymcdm_waspas_fixture_weights_are_refused_then_rank_alike():
    f = BY_ID["pymcdm-waspas"]
    with pytest.raises(ValueError, match="weights must sum to 1"):
        bulanik_score("waspas", f["matrix"], f["weights"], f["types"])
    got = _score(f)
    order = lambda v: sorted(range(len(v)), key=lambda i: -v[i])  # noqa: E731
    assert order(got) == order(f["expected"])
    assert 0.005 < max(abs(a - b) for a, b in zip(got, f["expected"])) < 0.006
