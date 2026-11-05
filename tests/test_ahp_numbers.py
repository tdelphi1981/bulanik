"""Tests for ahp_fuzzy_numbers (generic AHP over any fuzzy number kind)."""

import math

import pytest
from pytest import approx

from bulanik import IT2TrFN, IVIFN, IVNeutrosophicFN, PFN, TFN
from bulanik.aggregate import wg
from bulanik.mcdm import ahp, ahp_fuzzy_numbers


def test_crisp_degenerate_equals_row_geometric_mean():
    m = [[1, 3, 5], [1 / 3, 1, 2], [1 / 5, 1 / 2, 1]]
    r = ahp_fuzzy_numbers([[TFN(x, x, x) for x in row] for row in m])
    g = [math.prod(row) ** (1 / 3) for row in m]
    assert list(r.weights.values()) == approx([x / sum(g) for x in g])
    assert r.consistency_ratio == approx(ahp(m).consistency_ratio)
    assert r.operators["consistency"] == "saaty_ri_on_scores"
    assert r.operators["weights"] == "row_wg"


def _it2(points, h=0.5):
    return IT2TrFN(tuple(points) + (1, 1), tuple(points) + (h, h))


def test_it2_dtrat_hand_derived_golden():
    # HAND-DERIVED golden (NOT the published example of Kahraman, Oeztaysi,
    # Ucal Sari and Turanoglu 2014, KBS 59:48-57, doi:10.1016/j.knosys.2014.02.001,
    # whose worked example is paywalled and could not be obtained).
    # Every entry is a trapezoid (a1..a4) with upper heights (1,1) and lower
    # heights (0.5,0.5) on the same points. Rows (componentwise cube root of
    # the product, heights unchanged by the power, min across the product):
    #   r1 = [1, (1,2,4,8), (1,4,16,64)] -> (1,2,4,8)
    #   r2 = [1, 1, 1]                   -> (1,1,1,1)
    #   r3 = [1, 1, (1,8,8,8)]           -> (1,2,2,2)
    # DTraT per trapezoid X: ((a4-a1) + (h1 a2 - a1) + (h2 a3 - a1))/4 + a1.
    #   r1: U = (7+1+3)/4+1 = 3.75, L = (7+0+1)/4+1 = 3      -> 3.375
    #   r2: U = 1,                  L = (0-.5-.5)/4+1 = 0.75 -> 0.875
    #   r3: U = (1+1+1)/4+1 = 1.75, L = (1+0+0)/4+1 = 1.25   -> 1.5
    # sum = 5.75, weights = 27/46, 7/46, 6/23.
    one = _it2((1, 1, 1, 1))
    m = [
        [one, _it2((1, 2, 4, 8)), _it2((1, 4, 16, 64))],
        [one, one, one],
        [one, one, _it2((1, 8, 8, 8))],
    ]
    r = ahp_fuzzy_numbers(m, ["a", "b", "c"])
    assert list(r.weights.values()) == approx([27 / 46, 7 / 46, 6 / 23], abs=1e-9)
    assert r.operators["score"] == "dtrat"
    assert r.fuzzy_weights["a"].upper[:4] == approx((1, 2, 4, 8))


def _ivifn_strong(mu=0.6):
    return IVIFN(mu, mu + 0.1, 0.1, 0.2)


def test_ivifn_and_ivnn_run_with_normalized_score():
    s, w = _ivifn_strong(0.6), _ivifn_strong(0.3)
    m = [[s, w, s], [w, s, w], [s, s, w]]
    r = ahp_fuzzy_numbers(m, score="normalized")
    assert sum(r.weights.values()) == approx(1.0)
    for i, name in enumerate(r.criteria):
        assert r.fuzzy_weights[name] == wg(m[i])
    assert r.operators["score"] == "normalized"
    n1 = IVNeutrosophicFN(0.6, 0.7, 0.1, 0.2, 0.2, 0.3)
    n2 = IVNeutrosophicFN(0.3, 0.4, 0.3, 0.4, 0.4, 0.5)
    r2 = ahp_fuzzy_numbers([[n1, n2], [n2, n1]])
    assert sum(r2.weights.values()) == approx(1.0)
    assert r2.operators["score"] == "average"


def test_negative_scores_are_rejected_with_hint():
    bad = IVIFN(0.1, 0.2, 0.6, 0.7)
    with pytest.raises(ValueError, match='score="normalized"'):
        ahp_fuzzy_numbers([[bad, bad], [bad, bad]])
    n = IVNeutrosophicFN(0.0, 0.0, 1.0, 1.0, 1.0, 1.0)  # average score 0
    with pytest.raises(ValueError, match="positive row scores"):
        ahp_fuzzy_numbers([[n, n], [n, n]])


def test_consistency_matrix_and_none_branches():
    s, w = _ivifn_strong(0.6), _ivifn_strong(0.3)
    m = [[s, w, s], [w, s, w], [s, s, w]]
    saaty = [[1, 3, 5], [1 / 3, 1, 2], [1 / 5, 1 / 2, 1]]
    r = ahp_fuzzy_numbers(m, score="normalized", consistency_matrix=saaty)
    assert r.consistency_ratio == approx(ahp(saaty).consistency_ratio)
    assert r.operators["consistency"] == "saaty_ri_on_given_matrix"
    with pytest.raises(ValueError):
        ahp_fuzzy_numbers(m, score="normalized", consistency_matrix=[[1, 2], [0.5, 1]])
    # signed entry scores (difference) but positive row scores -> no CR
    weak = IVIFN(0.1, 0.2, 0.6, 0.7)
    big = IVIFN(0.9, 0.95, 0.0, 0.02)
    m2 = [[big, big, weak], [big, big, big], [big, big, big]]
    r2 = ahp_fuzzy_numbers(m2)
    assert r2.consistency_ratio is None
    assert r2.operators["consistency"] == "none"


def test_wa_aggregation():
    m = [[TFN(1, 1, 1), TFN(2, 3, 4)], [TFN(0.25, 1 / 3, 0.5), TFN(1, 1, 1)]]
    r = ahp_fuzzy_numbers(m, aggregation="wa")
    assert r.operators["weights"] == "row_wa"
    assert r.fuzzy_weights["C1"].m == approx(2.0)


def test_errors():
    a, b = TFN(1, 1, 1), PFN(0.5, 0.3)
    with pytest.raises(TypeError):
        ahp_fuzzy_numbers([[a, a], [a, b]])
    with pytest.raises(ValueError, match="square"):
        ahp_fuzzy_numbers([[a, a, a], [a, a, a]])
    with pytest.raises(ValueError, match="at most 10"):
        ahp_fuzzy_numbers([[a] * 11] * 11)
    with pytest.raises(ValueError, match="valid: wa, wg"):
        ahp_fuzzy_numbers([[a]], aggregation="foo")
    with pytest.raises(TypeError):
        ahp_fuzzy_numbers([[1.0]])
    with pytest.raises(ValueError, match="valid"):
        ahp_fuzzy_numbers([[a]], score="nope")
