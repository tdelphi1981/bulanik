import dataclasses

import pytest
from hypothesis import given, strategies as st
from pytest import approx

from bulanik import IFN, TFN
from bulanik.mcdm import RankingResult, WeightingResult, ahp, entropy, fuzzy_ahp, swara
from bulanik.mcdm._common import (
    check_matrix,
    check_types,
    coerce_weights,
    default_names,
    rank_order,
)


def test_ahp_consistent():
    r = ahp([[1, 2, 4], [0.5, 1, 2], [0.25, 0.5, 1]], ["a", "b", "c"])
    assert list(r.weights.values()) == approx([4 / 7, 2 / 7, 1 / 7])
    assert r.consistency_ratio == approx(0, abs=1e-9)
    assert r.ranking == ("a", "b", "c") and r.method == "ahp"
    assert r.operators == {"weights": "eigenvector", "consistency": "saaty_ri"}


def test_ahp_inconsistent_closed_form():
    # 3x3 reciprocal: eigenvector = row geometric means; lmax = 1 + c^(1/3) + c^(-1/3), c = a12*a23/a13 = 3*.25/.5 = 1.5
    r = ahp([[1, 3, 0.5], [1 / 3, 1, 0.25], [2, 4, 1]])
    gm = [(1 * 3 * 0.5) ** (1 / 3), (1 / 3 * 0.25) ** (1 / 3), 8 ** (1 / 3)]
    assert list(r.weights.values()) == approx([g / sum(gm) for g in gm])  # .319618, .121957, .558425
    lmax = 1 + 1.5 ** (1 / 3) + 1.5 ** (-1 / 3)  # 3.018295
    assert r.consistency_ratio == approx((lmax - 3) / 2 / 0.58)  # .015771
    assert r.ranking == ("C3", "C1", "C2")


def test_ahp_two_criteria_cr_zero():
    r = ahp([[1, 3], [1 / 3, 1]])
    assert r.weights["C1"] == approx(0.75) and r.consistency_ratio == 0


def test_ahp_validation():
    with pytest.raises(ValueError, match="reciprocal"):
        ahp([[1, 2], [2, 1]])
    with pytest.raises(ValueError, match="10"):
        ahp([[1.0] * 11 for _ in range(11)])
    with pytest.raises(ValueError, match="square"):
        ahp([[1, 2, 3], [1, 1, 1]])
    with pytest.raises(ValueError, match="positive"):
        ahp([[1, -2], [-0.5, 1]])
    with pytest.raises(ValueError, match="finite"):
        ahp([[1, float("inf")], [0.0, 1]])
    with pytest.raises(ValueError, match="diagonal"):
        ahp([[2, 1], [1, 2]])
    with pytest.raises(ValueError, match="names"):
        ahp([[1, 2], [0.5, 1]], ["only"])
    with pytest.raises(ValueError, match="unique"):
        ahp([[1, 2], [0.5, 1]], ["x", "x"])


def test_fuzzy_ahp_buckley_golden():
    m = [[TFN(1, 1, 1), TFN(1, 2, 3)], [TFN(1 / 3, 1 / 2, 1), TFN(1, 1, 1)]]
    r = fuzzy_ahp(m, ["c1", "c2"])
    # r1=(1,sqrt2,sqrt3), r2=(sqrt(1/3),sqrt(.5),1); sum=(1.577350,2.121320,2.732051)
    # w1=(1/2.732051, sqrt2/2.121320, sqrt3/1.577350)=(.366025,.666667,1.098076); w2=(.211325,.333333,.633975)
    # centroids .710256 and .392878 -> normalised
    assert r.weights["c1"] == approx(0.6438531421) and r.weights["c2"] == approx(0.3561468579)
    f = r.fuzzy_weights["c1"]
    assert (f.l, f.m, f.u) == approx((0.3660254038, 2 / 3, 1.0980762114))
    assert r.consistency_ratio == 0 and r.operators["defuzzification"] == "centroid"
    assert r.operators["weights"] == "buckley_geometric_mean"
    with pytest.raises(ValueError, match="reciprocal"):
        fuzzy_ahp([[TFN(1, 1, 1), TFN(1, 2, 3)], [TFN(1, 2, 3), TFN(1, 1, 1)]])


def test_fuzzy_ahp_defuzzification_and_errors():
    m = [[TFN(1, 1, 1), TFN(1, 2, 3)], [TFN(1 / 3, 1 / 2, 1), TFN(1, 1, 1)]]
    r = fuzzy_ahp(m, defuzzification="graded_mean")
    assert r.operators["defuzzification"] == "graded_mean"
    assert sum(r.weights.values()) == approx(1)
    with pytest.raises(ValueError, match="valid: centroid"):
        fuzzy_ahp(m, defuzzification="foo")
    with pytest.raises(TypeError):
        fuzzy_ahp([[1, 2], [0.5, 1]])
    with pytest.raises(ValueError, match="square"):
        fuzzy_ahp([[TFN(1, 1, 1)], [TFN(1, 1, 1)]])


def test_fuzzy_ahp_inconsistent_has_positive_cr():
    # crisp-like TFNs reusing the closed-form inconsistent AHP matrix: CR must equal ahp's
    raw = [[1, 3, 0.5], [1 / 3, 1, 0.25], [2, 4, 1]]
    m = [[TFN(x, x, x) for x in r] for r in raw]
    assert fuzzy_ahp(m).consistency_ratio == approx(ahp(raw).consistency_ratio)


def test_swara_golden():
    r = swara([0.2, 0.5], ["A", "B", "C"])  # k=(1,1.2,1.5) q=(1,5/6,5/9) sum=43/18
    assert list(r.weights.values()) == approx([18 / 43, 15 / 43, 10 / 43])
    assert r.ranking == ("A", "B", "C") and r.operators == {"score": "crisp"}


def test_swara_fuzzy_via_score():
    r = swara([TFN(0.1, 0.2, 0.3), TFN(0.4, 0.5, 0.6)], ["A", "B", "C"])  # centroids .2 and .5
    assert list(r.weights.values()) == approx([18 / 43, 15 / 43, 10 / 43])
    assert r.operators["score"] == "centroid"
    with pytest.raises(ValueError, match="normalized"):
        swara([IFN(0.2, 0.6)], ["A", "B"])  # score -.4
    assert swara([IFN(0.2, 0.6)], ["A", "B"], score="normalized").weights["A"] == approx(
        1 / (1 + 1 / 1.3)
    )  # s=.3
    with pytest.raises(ValueError):
        swara([0.2], ["A", "B", "C"])
    with pytest.raises(ValueError, match="unknown score"):
        swara([TFN(0.1, 0.2, 0.3)], ["A", "B"], score="foo")
    with pytest.raises(ValueError, match=">= 0"):
        swara([-0.1], ["A", "B"])


def test_entropy_golden():
    # col1 (1,2,3): p=(1/6,1/3,1/2) E=.920620 d=.079380 ; col2 (4,4,1): p=(4/9,4/9,1/9) E=.878347 d=.121653
    r = entropy([[1, 4], [2, 4], [3, 1]])
    assert list(r.weights.values()) == approx([0.3948612456, 0.6051387544])
    with pytest.raises(ValueError, match="discriminat"):
        entropy([[1, 2], [1, 2]])


def test_entropy_zero_entries_and_errors():
    # col (0,1): p=(0,1), 0 ln 0 = 0, E=0, d=1; col (1,1): d=0 -> all weight on col 1
    r = entropy([[0, 1], [1, 1]], ["x", "y"])
    assert r.weights == approx({"x": 1.0, "y": 0.0})
    with pytest.raises(ValueError, match="at least 2"):
        entropy([[1, 2]])
    with pytest.raises(ValueError, match="non-negative"):
        entropy([[1, -2], [1, 2]])
    with pytest.raises(ValueError, match="sums to zero"):
        entropy([[0, 2], [0, 3]])


@given(st.lists(st.floats(0.0, 3.0), min_size=1, max_size=6))
def test_swara_weights_sum_and_order(s):
    r = swara(s, [f"C{i}" for i in range(len(s) + 1)])
    w = list(r.weights.values())
    assert sum(w) == approx(1) and all(a >= b - 1e-12 for a, b in zip(w, w[1:]))


# -- results and common helpers ---------------------------------------------


def test_results_are_frozen():
    r = swara([0.2], ["A", "B"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.method = "x"
    rr = RankingResult("m", ("a",), {"a": 1.0}, ("a",), {})
    assert rr.extras == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        rr.method = "x"


def test_rank_order_ties_keep_input_order():
    assert rank_order({"a": 1, "b": 2, "c": 2, "d": 0}, True) == ("b", "c", "a", "d")
    assert rank_order({"a": 1, "b": 2, "c": 2, "d": 0}, False) == ("d", "a", "b", "c")


def test_default_names_and_types():
    assert default_names("A", 3) == ["A1", "A2", "A3"]
    assert check_types(["benefit", "cost"], 2) == ["benefit", "cost"]
    with pytest.raises(ValueError, match="valid: benefit, cost"):
        check_types(["max"], 1)
    with pytest.raises(ValueError, match="expected 2"):
        check_types(["cost"], 2)


def test_check_matrix():
    assert check_matrix([[1, 2], [3, 4]], 2, 2) == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError, match="same length"):
        check_matrix([[1, 2], [3]])
    with pytest.raises(ValueError, match="at least one"):
        check_matrix([])
    with pytest.raises(ValueError, match="alternatives"):
        check_matrix([[1, 2]], n_alt=2)
    with pytest.raises(ValueError, match="criteria"):
        check_matrix([[1, 2]], n_crit=3)
    with pytest.raises(ValueError, match="finite"):
        check_matrix([[1, float("nan")]])


def test_coerce_weights():
    crit = ["a", "b"]
    assert coerce_weights([0.3, 0.7], crit) == [0.3, 0.7]
    assert coerce_weights({"b": 0.7, "a": 0.3}, crit) == [0.3, 0.7]
    assert coerce_weights([1, 3], crit, normalize=True) == approx([0.25, 0.75])
    wr = swara([1.0], ["a", "b"])  # q=(1,.5) -> w=(2/3,1/3)
    assert coerce_weights(wr, crit) == approx([2 / 3, 1 / 3])
    with pytest.raises(ValueError, match="missing"):
        coerce_weights({"a": 1.0}, crit)
    with pytest.raises(ValueError, match="missing"):
        coerce_weights(swara([1.0], ["a", "z"]), crit)
    with pytest.raises(ValueError, match="sum to 1"):
        coerce_weights([1, 3], crit)
    assert isinstance(wr, WeightingResult)


# -- fix round 1 -----------------------------------------------------------


def _reciprocal(n, upper):
    a = [[1.0] * n for _ in range(n)]
    for (i, j), v in upper.items():
        a[i][j], a[j][i] = v, 1 / v
    return a


def test_ahp_non_convergence_raises():
    # extremely inconsistent: pairs ~1e6 in conflicting directions; power iteration
    # does not settle within 1000 iterations (found by search, verified by hand loop)
    up = {
        (0, 1): 1041586.2109754108, (0, 2): 1713443.06695905, (0, 3): 8.773749463924485e-07,
        (1, 2): 553601.6626045557, (1, 3): 1.9995781939179036e-06, (2, 3): 1.7055041647294022e-06,
    }
    with pytest.raises(ValueError, match="converge.*inconsistent"):
        ahp(_reciprocal(4, up))


def test_ahp_saaty_scale_still_converges():
    up = {(0, 1): 9, (0, 2): 1 / 9, (1, 2): 9, (0, 3): 7, (1, 3): 1 / 5, (2, 3): 3}
    r = ahp(_reciprocal(4, up))
    assert sum(r.weights.values()) == approx(1)


@pytest.mark.parametrize("bad", [True, "2", False])
def test_non_numeric_entries_rejected(bad):
    with pytest.raises(TypeError):
        ahp([[1, bad], [0.5, 1]])
    with pytest.raises(TypeError):
        swara([bad], ["A", "B"])
    with pytest.raises(TypeError):
        entropy([[1, bad], [2, 3]])
    with pytest.raises(TypeError):
        check_matrix([[1, bad]])


def test_swara_accepts_generator_and_needs_a_criterion():
    r = swara([0.2, 0.5], (c for c in "ABC"))
    assert list(r.weights) == ["A", "B", "C"]
    with pytest.raises(ValueError, match="at least one criterion"):
        swara([], [])
    assert swara([], ["A"]).weights == {"A": 1.0}


def test_fuzzy_ahp_needs_positive_lower_bounds():
    # like crisp ahp ("entries must be positive"): every TFN judgement needs l > 0
    neg = TFN(-2, -1, -0.5)
    with pytest.raises(ValueError, match=r"positive.*l > 0"):
        fuzzy_ahp([[TFN(1, 1, 1), neg], [neg, TFN(1, 1, 1)]])
    with pytest.raises(ValueError, match=r"positive.*l > 0"):
        fuzzy_ahp([[TFN(1, 1, 1), TFN(0, 1, 2)], [TFN(0.5, 1, 4), TFN(1, 1, 1)]])


def test_swara_score_name_and_mixed_inputs():
    # a given score is validated even when every input is crisp
    with pytest.raises(ValueError, match="unknown score 'foo'"):
        swara([0.2, 0.5], ["A", "B", "C"], score="foo")
    with pytest.raises(ValueError, match="unknown score 'foo'"):
        swara([], ["A"], score="foo")
    # a valid fuzzy score name on crisp inputs is harmless: k=(1,1.2,1.5) as in the golden
    r = swara([0.2, 0.5], ["A", "B", "C"], score="centroid")
    assert r.weights["A"] == approx(18 / 43) and r.operators["score"] == "crisp"
    with pytest.raises(ValueError, match="mix"):
        swara([0.2, TFN(0.1, 0.2, 0.3)], ["A", "B", "C"])
    with pytest.raises(ValueError, match="mix"):
        swara([IFN(0.2, 0.6), 0.5], ["A", "B", "C"], score="normalized")


def test_coerce_weights_accepts_bwm_result_and_rejects_unknown_keys():
    from bulanik.mcdm import BWMResult

    crit = ["a", "b"]
    bw = BWMResult(
        method="bwm_nonlinear", criteria=("b", "a"), weights={"b": 0.4, "a": 0.6},
        ranking=("a", "b"), xi=0.1, consistency_ratio=0.05, best="a", worst="b", operators={},
    )
    assert coerce_weights(bw, crit) == approx([0.6, 0.4])
    with pytest.raises(ValueError, match="unknown criteria: c, d"):
        coerce_weights({"a": 0.5, "b": 0.5, "c": 0.0, "d": 0.0}, crit)
    with pytest.raises(ValueError, match="unknown criteria: z"):
        coerce_weights(swara([1.0, 1.0], ["a", "b", "z"]), crit)


def test_entropy_weak_but_real_discrimination_is_kept():
    # deficits 1 - E ~ 2 delta^2 = 1.25e-13 and 5e-13: below the old absolute 1e-12 cutoff,
    # far above rounding noise, and in ratio 1:4 (delta = 2.5e-7 and 5e-7)
    r = entropy([[1.0, 1.0], [1.0 + 1e-6, 1.0 + 2e-6]])
    w1, w2 = r.weights["C1"], r.weights["C2"]
    assert w2 / w1 == approx(4.0, rel=0.05)


def test_entropy_rounding_noise_is_not_discrimination():
    # identical columns up to the last bits: entropy deficit is pure float noise
    with pytest.raises(ValueError, match="discriminat"):
        entropy([[1.0, 3.0], [1.0 + 2e-16, 3.0 + 4e-16]])


def test_fuzzy_ahp_reciprocity_tolerance_is_relative():
    one = TFN(1, 1, 1)
    # large judgements: a mere 1e-12 relative rounding difference is accepted
    big = TFN(1e10, 2e10, 3e10)
    inv = big.reciprocal()
    ok = TFN(inv.l * (1 + 1e-12), inv.m, inv.u)
    assert sum(fuzzy_ahp([[one, big], [ok, one]]).weights.values()) == approx(1)
    # same tolerance with the large value in the lower triangle, a[1][0]
    ok_low = TFN(big.l * (1 + 1e-12), big.m, big.u)
    assert sum(fuzzy_ahp([[one, inv], [ok_low, one]]).weights.values()) == approx(1)
    # small reciprocals: a 10 % error is only 1e-11 in absolute terms, yet must be rejected
    bad = TFN(inv.l * 1.1, inv.m, inv.u)
    with pytest.raises(ValueError, match="reciprocal"):
        fuzzy_ahp([[one, big], [bad, one]])
