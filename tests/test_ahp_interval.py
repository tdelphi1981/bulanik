"""Interval-valued AHP: the weights are checked against the formulas written out in full."""

import math

import pytest
from hypothesis import given, settings, strategies as st

from bulanik import IVFFN, IVIFN, IVPFN, IVQROFN
from bulanik.mcdm import ahp_interval_valued

# Illustrative linguistic scales: mu_l, mu_u, nu_l, nu_u
IF_TERMS = {
    "EQ": (.45, .45, .45, .45), "M": (.5, .6, .2, .3), "S": (.6, .7, .15, .25),
    "W": (.2, .3, .5, .6), "VW": (.15, .25, .6, .7),
}
PF_TERMS = {
    "EQ": (.5, .6, .5, .6), "M": (.6, .7, .3, .4), "S": (.7, .8, .2, .3),
    "W": (.3, .4, .6, .7), "VW": (.2, .3, .7, .8),
}
FF_TERMS = {
    "EQ": (.55, .6, .55, .6), "M": (.65, .75, .3, .4), "S": (.75, .85, .2, .3),
    "W": (.3, .4, .65, .75), "VW": (.2, .3, .75, .85),
}
TERMS3 = [["EQ", "M", "S"], ["W", "EQ", "M"], ["VW", "W", "EQ"]]
TERMS4 = [["EQ", "S", "M", "W"], ["VW", "EQ", "W", "VW"], ["W", "M", "EQ", "W"], ["M", "S", "M", "EQ"]]


def build(kind, scale, terms):
    return [[kind(*scale[t]) for t in row] for row in terms]


def multiplicative_by_formula(scale, terms, q):
    """s = 1000^(d/q), h the hesitancy, t = (s_l + s_u)/2 h, w_i the row share of t."""
    t = []
    for row in terms:
        out = []
        for term in row:
            ml, mu, nl, nu = scale[term]
            s_l = 1000.0 ** ((ml**q - nu**q) / q)
            s_u = 1000.0 ** ((mu**q - nl**q) / q)
            out.append((s_l + s_u) / 2 * (1 - (mu**q - ml**q) - (nu**q - nl**q)))
        t.append(out)
    total = sum(map(sum, t))
    return [sum(row) / total for row in t]


def possibility_by_formula(scale, terms):
    """Exponential interval matrix, row-sum interval weights, w_j = sum_i p_ij + n/2 - 1."""
    n = len(terms)
    lo = [[math.exp(scale[t][0] - scale[t][3]) for t in row] for row in terms]
    hi = [[math.exp(scale[t][1] - scale[t][2]) for t in row] for row in terms]
    all_lo, all_hi = sum(map(sum, lo)), sum(map(sum, hi))
    iv = [(sum(lo[i]) / all_hi, sum(hi[i]) / all_lo) for i in range(n)]
    raw = []
    for j in range(n):
        total = n / 2 - 1
        for i in range(n):
            den = (iv[i][1] - iv[i][0]) + (iv[j][1] - iv[j][0])
            total += (max(0.0, iv[j][1] - iv[i][0]) - max(0.0, iv[j][0] - iv[i][1])) / den
        raw.append(total)
    return [x / sum(raw) for x in raw]


def test_multiplicative_q2_matches_formula():
    r = ahp_interval_valued(build(IVPFN, PF_TERMS, TERMS3))
    assert r.method == "ahp_interval_valued"
    assert r.operators["method"] == "multiplicative"
    assert list(r.weights.values()) == pytest.approx(multiplicative_by_formula(PF_TERMS, TERMS3, 2))


def test_multiplicative_order_four_matches_formula():
    r = ahp_interval_valued(build(IVPFN, PF_TERMS, TERMS4))
    assert list(r.weights.values()) == pytest.approx(multiplicative_by_formula(PF_TERMS, TERMS4, 2))


def test_multiplicative_q3_matches_formula():
    r = ahp_interval_valued(build(IVFFN, FF_TERMS, TERMS3), ["a", "b", "c"])
    assert r.operators == {"method": "multiplicative", "q": "3"}
    assert list(r.weights.values()) == pytest.approx(multiplicative_by_formula(FF_TERMS, TERMS3, 3))


def test_multiplicative_two_by_two_by_hand():
    # (.6,.7,.3,.4), q = 2: d_l = .36 - .16 = .2, d_u = .49 - .09 = .4, s = 1000^.1 and 1000^.2,
    # h = 1 - .13 - .07 = .8. (.3,.4,.6,.7) mirrors it: d_l = -.4, d_u = -.2, the same h.
    one = IVPFN(.6, .7, .3, .4)
    two = IVPFN(.3, .4, .6, .7)
    t_one = (1000**.1 + 1000**.2) / 2 * .8
    t_two = (1000**-.2 + 1000**-.1) / 2 * .8
    r = ahp_interval_valued([[one, one], [two, two]], method="multiplicative")
    assert r.weights["C1"] == pytest.approx(t_one / (t_one + t_two))


def test_possibility_matches_formula():
    r = ahp_interval_valued(build(IVIFN, IF_TERMS, TERMS3))
    assert r.operators == {"method": "possibility", "q": "1"}
    assert list(r.weights.values()) == pytest.approx(possibility_by_formula(IF_TERMS, TERMS3))


def test_possibility_order_four_matches_formula():
    r = ahp_interval_valued(build(IVIFN, IF_TERMS, TERMS4))
    assert list(r.weights.values()) == pytest.approx(possibility_by_formula(IF_TERMS, TERMS4))


def test_possibility_all_equal_gives_equal_weights():
    m = build(IVIFN, IF_TERMS, [["EQ", "EQ"], ["EQ", "EQ"]])
    r = ahp_interval_valued(m)
    assert r.weights["C1"] == pytest.approx(0.5)
    assert r.weights["C2"] == pytest.approx(0.5)


def test_possibility_crisp_matrix_orders_by_value():
    one = IVIFN(.5, .5, .5, .5)
    big = IVIFN(.9, .9, .1, .1)
    r = ahp_interval_valued([[one, big], [one, one]], method="possibility")
    assert r.weights["C1"] > r.weights["C2"]


def test_possibility_rejects_q2():
    m = build(IVPFN, PF_TERMS, [["EQ", "W"], ["M", "EQ"]])
    with pytest.raises(ValueError, match="IVIFN only"):
        ahp_interval_valued(m, method="possibility")


def test_mixed_kinds_rejected():
    a = IVIFN(.5, .5, .5, .5)
    b = IVPFN(.5, .5, .5, .5)
    with pytest.raises(TypeError, match="IVPFN"):
        ahp_interval_valued([[a, a], [a, b]])
    with pytest.raises(TypeError):
        ahp_interval_valued([[a, 1.0], [a, a]])


def test_shape_rejected():
    a = IVIFN(.5, .5, .5, .5)
    with pytest.raises(ValueError):
        ahp_interval_valued([[a, a, a], [a, a, a]])
    with pytest.raises(ValueError):
        ahp_interval_valued([[a]])


def test_unknown_method_lists_valid():
    a = IVIFN(.5, .5, .5, .5)
    with pytest.raises(ValueError, match="multiplicative, possibility"):
        ahp_interval_valued([[a, a], [a, a]], method="nope")


def test_default_method_by_q():
    for kind, expected in ((IVIFN, "possibility"), (IVPFN, "multiplicative"), (IVFFN, "multiplicative")):
        a = kind(.3, .4, .2, .3)
        assert ahp_interval_valued([[a, a], [a, a]]).operators["method"] == expected
    g = IVQROFN(.3, .4, .2, .3, 4)
    assert ahp_interval_valued([[g, g], [g, g]]).operators == {"method": "multiplicative", "q": "4"}


def test_extras_keys():
    m = build(IVPFN, PF_TERMS, TERMS3)
    r = ahp_interval_valued(m)
    for k in ("d_lower", "d_upper", "s_lower", "s_upper", "h", "t", "scored_matrix", "consistency_ratio"):
        assert k in r.extras
    assert len(r.extras["t"]) == 3 and len(r.extras["t"][0]) == 3
    m = build(IVIFN, IF_TERMS, TERMS3)
    r = ahp_interval_valued(m)
    for k in ("interval_weights", "possibility", "scored_matrix", "consistency_ratio"):
        assert k in r.extras
    assert len(r.extras["interval_weights"]) == 3 and len(r.extras["interval_weights"][0]) == 2
    assert r.extras["possibility"][0][0] == pytest.approx(0.5)


def test_consistency_given_matrix():
    m = build(IVIFN, IF_TERMS, TERMS3)
    crisp = [[1, 1 / 3, 3], [3, 1, 5], [1 / 3, 1 / 5, 1]]
    r = ahp_interval_valued(m, consistency_matrix=crisp)
    assert r.consistency_ratio is not None and r.consistency_ratio < 0.1
    assert r.extras["consistency_ratio"] == r.consistency_ratio
    with pytest.raises(ValueError):
        ahp_interval_valued(m, consistency_matrix=[[1, 2], [0.5, 1]])


def test_criteria_names():
    m = build(IVIFN, IF_TERMS, [["EQ", "M"], ["W", "EQ"]])
    r = ahp_interval_valued(m, ["a", "b"])
    assert r.ranking == ("a", "b")
    with pytest.raises(ValueError):
        ahp_interval_valued(m, ["a"])


def _iv(u, v, fl, fn):
    return (u * fl, u, v * fn, v)


@st.composite
def pythagorean_matrix(draw):
    n = draw(st.integers(2, 5))
    unit = st.floats(0, 1)
    rows = []
    for _ in range(n):
        row = []
        for _ in range(n):
            u, v = draw(st.floats(0, 0.7)), draw(st.floats(0, 0.7))
            row.append(IVPFN(*_iv(u, v, draw(unit), draw(unit))))
        rows.append(row)
    return rows


@given(pythagorean_matrix())
@settings(max_examples=60, deadline=None)
def test_weights_sum_to_one(m):
    r = ahp_interval_valued(m)
    w = list(r.weights.values())
    assert all(x > 0 for x in w)
    assert sum(w) == pytest.approx(1.0, abs=1e-12)


def test_non_number_matrix_rejected():
    with pytest.raises(TypeError, match="IVQROFN"):
        ahp_interval_valued([[1.0, 2.0], [0.5, 1.0]])


def test_zero_total_rejected():
    full = IVPFN(0.0, 1.0, 0.0, 0.0)  # h = 0: a maximally hesitant judgment
    with pytest.raises(ValueError, match="non-positive"):
        ahp_interval_valued([[full, full], [full, full]])
