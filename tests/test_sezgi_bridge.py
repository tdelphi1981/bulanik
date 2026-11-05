import importlib
import math
import sys

import pytest
from pytest import approx

sezgi = pytest.importorskip("sezgi")

from bulanik import TFN  # noqa: E402
from bulanik.anfis import ANFIS  # noqa: E402
from bulanik.mcdm.bwm import fuzzy_bwm_consistency_index  # noqa: E402
from bulanik.scales import FUZZY_BWM_TFN  # noqa: E402
from bulanik.sezgi import (  # noqa: E402
    ANFISProblem,
    ANFISTrainResult,
    BWMProblem,
    FuzzyBWMProblem,
    train_anfis,
)

BO, OW = {"C1": 1, "C2": 2, "C3": 4}, {"C1": 4, "C2": 2, "C3": 1}


def test_bwm_evaluate_golden():
    p = BWMProblem(BO, OW)
    assert p.evaluate([4 / 7, 2 / 7, 1 / 7]) == approx(0.0, abs=1e-12)
    # max(|.5/.3-2|, |.5/.2-4|, |.3/.2-2|) = max(.3333, 1.5, .5); Σ=1 so no penalty
    assert p.evaluate([0.5, 0.3, 0.2]) == approx(1.5)
    assert p.evaluate([1.0, 0.5, 0.25]) == approx(10 * 0.75)  # ratios exact, |Σ-1|=.75


def test_bwm_solve_consistent():
    r = BWMProblem(BO, OW).solve(seed=1)
    assert list(r.weights.values()) == approx([4 / 7, 2 / 7, 1 / 7], abs=1e-3) and r.xi < 1e-3
    assert r.method == "bwm_nonlinear" and r.best == "C1" and r.worst == "C3"
    assert r.operators == {"model": "nonlinear", "consistency": "rezaei2015_ci",
                           "algorithm": "DifferentialEvolution"}
    assert sum(r.weights.values()) == approx(1, abs=1e-12) and r.ranking == ("C1", "C2", "C3")


def test_bwm_solve_inconsistent_closed_form():
    # binding: w1/w2 = 2-ξ, w1/w3 = 4+ξ, w2/w3 = 3-ξ -> (2-ξ)(3-ξ) = 4+ξ -> ξ²-6ξ+2=0 -> ξ = 3-√7 = .354249
    # w3 = 1/(1+√7+(7-√7)) = 1/8, w2 = √7/8, w1 = (7-√7)/8
    r = BWMProblem(BO, {"C1": 4, "C2": 3, "C3": 1}).solve(seed=1)
    assert r.xi == approx(3 - 7 ** 0.5, abs=1e-3)
    assert list(r.weights.values()) == approx([(7 - 7 ** 0.5) / 8, 7 ** 0.5 / 8, 1 / 8], abs=2e-3)
    assert r.consistency_ratio == approx((3 - 7 ** 0.5) / ((9 - 33 ** 0.5) / 2), abs=1e-3)  # CI(4)=1.627719 -> .217635


def test_bwm_consistency_ratio_zero_index():
    # a_BW = 1 gives CI = 0 (both roots of ξ²-3ξ+0 are 0 and 3; smaller is 0). Consistent
    # comparisons (ξ <= 1e-6) report CR = 0
    r = BWMProblem({"C1": 1, "C2": 1}, {"C1": 1, "C2": 1}, best="C1", worst="C2").solve()
    assert r.xi <= 1e-6 and r.consistency_ratio == 0.0
    assert list(r.weights.values()) == approx([0.5, 0.5], abs=1e-3)
    assert "consistency_note" not in r.extras


def test_bwm_consistency_ratio_none_when_index_zero_but_inconsistent():
    # a_BW = 1 (CI = 0) yet a_AB = 3, a_BC = 2 contradict w_A ≈ w_C. With x = w_A/w_C,
    # t = w_B/w_C: x <= 1+ξ, t >= 2-ξ, x/t >= 3-ξ -> (1+ξ) >= (3-ξ)(2-ξ) -> ξ²-6ξ+5 <= 0 -> ξ* = 1
    r = BWMProblem({"A": 1, "B": 3, "C": 1}, {"A": 1, "B": 2, "C": 1}, best="A", worst="C").solve()
    assert r.xi == approx(1.0, abs=1e-3)
    assert r.consistency_ratio is None
    assert "inconsistent" in r.extras["consistency_note"] and "a_BW = 1" in r.extras["consistency_note"]


def test_bwm_problem_validates():
    with pytest.raises(ValueError, match="penalty"):
        BWMProblem(BO, OW, penalty=-1)
    with pytest.raises(ValueError, match="must be >= 1"):
        BWMProblem({"C1": 1, "C2": 0.5}, {"C1": 0.5, "C2": 1})
    with pytest.raises(ValueError, match="expected 3"):
        BWMProblem(BO, OW).evaluate([0.5, 0.5])
    with pytest.raises(ValueError, match="budget"):
        BWMProblem(BO, OW).solve(budget=0)
    with pytest.raises(ValueError, match="> 0"):
        BWMProblem(BO, OW).evaluate([0.5, 0.0, 0.5])


def test_fuzzy_ci():
    assert fuzzy_bwm_consistency_index(TFN(1, 1, 1)) == approx(3.0)  # roots of ξ²-3ξ
    assert fuzzy_bwm_consistency_index(FUZZY_BWM_TFN["WI"]) == approx((4 + 13 ** 0.5) / 2)  # 3.802776 (table 3.80)
    assert fuzzy_bwm_consistency_index(FUZZY_BWM_TFN["VI"]) == approx((8 + 29 ** 0.5) / 2)  # 6.692582 (table 6.69)
    # FI u=5/2: ξ²-6ξ+15/4 -> (6+√21)/2 = 5.291288 (table 5.29); AI u=9/2: ξ²-10ξ+63/4 -> (10+√37)/2 = 8.041381 (8.04)
    assert fuzzy_bwm_consistency_index(FUZZY_BWM_TFN["FI"]) == approx((6 + 21 ** 0.5) / 2)
    assert fuzzy_bwm_consistency_index(FUZZY_BWM_TFN["AI"]) == approx((10 + 37 ** 0.5) / 2)
    with pytest.raises(TypeError, match="TFN"):
        fuzzy_bwm_consistency_index(3.0)
    with pytest.raises(ValueError, match=">= 1"):
        fuzzy_bwm_consistency_index(TFN(0.5, 0.6, 0.8))


def test_fuzzy_bwm_scale_terms():
    assert FUZZY_BWM_TFN.terms == ("EI", "WI", "FI", "VI", "AI")
    assert FUZZY_BWM_TFN["FI"] == TFN(1.5, 2, 2.5) and "Guo" in FUZZY_BWM_TFN.source


def test_fuzzy_bwm_degenerate_equals_crisp_evaluate():
    T = lambda a: TFN(a, a, a)  # noqa: E731
    p = FuzzyBWMProblem({k: T(v) for k, v in BO.items()}, {k: T(v) for k, v in OW.items()})
    w = [4 / 7, 2 / 7, 1 / 7]
    x = [v for wi in w for v in (wi, wi, wi)]
    assert p.evaluate(x) == approx(0.0, abs=1e-12)  # R(w̃)=w, Σ=1, ordered


def test_fuzzy_bwm_ordering_penalty():
    T = lambda a: TFN(a, a, a)  # noqa: E731
    bo, ow = {k: T(v) for k, v in BO.items()}, {k: T(v) for k, v in OW.items()}
    # C3 = (1/7 + .01, 1/7, 1/7 - .01): ΣR is unchanged (l and u shift by ±.01, R = (l+4m+u)/6),
    # so the only penalised term is the ordering one: max(0, l - m) + max(0, m - u) = .01 + .01
    x = [4 / 7] * 3 + [2 / 7] * 3 + [1 / 7 + 0.01, 1 / 7, 1 / 7 - 0.01]
    lo, hi = FuzzyBWMProblem(bo, ow, penalty=10.0), FuzzyBWMProblem(bo, ow, penalty=30.0)
    # evaluate = deviation + penalty * .02, so the penalty slope is exactly .02 per unit
    assert (hi.evaluate(x) - lo.evaluate(x)) / 20.0 == approx(0.02, abs=1e-12)
    assert lo.evaluate(x) == approx(lo.deviation(x) + 10.0 * 0.02, abs=1e-12)
    # the deviation alone is the C1/C3 lower row |4/7 - 4(1/7 - .01)| / (1/7 - .01), the largest of all
    assert lo.deviation(x) == approx(0.04 / (1 / 7 - 0.01), rel=1e-9)
    # the ordered version of the same sum has no penalty at all
    ordered = [4 / 7] * 3 + [2 / 7] * 3 + [1 / 7] * 3
    assert lo.evaluate(ordered) == approx(0.0, abs=1e-12)


def test_fuzzy_bwm_solve_golden():
    S = FUZZY_BWM_TFN
    bo = {"C1": S["EI"], "C2": S["FI"], "C3": S["VI"]}
    ow = {"C1": S["VI"], "C2": S["FI"], "C3": S["EI"]}
    # the middle values alone form crisp BWM a12=2, a13=3, a23=2: (2-ξ)(2-ξ)=3+ξ -> ξ²-5ξ+1=0 -> ξ=(5-√21)/2
    # that is a lower bound on k*, and it is attained (prototype seeds 1-3: k*=.208712)
    r = FuzzyBWMProblem(bo, ow).solve(seed=1)
    assert r.xi == approx((5 - 21 ** 0.5) / 2, abs=1e-3)
    assert r.consistency_ratio == approx(((5 - 21 ** 0.5) / 2) / ((8 + 29 ** 0.5) / 2), abs=1e-3)  # .031186
    assert sum(r.weights.values()) == approx(1, abs=1e-9) and r.ranking == ("C1", "C2", "C3")
    assert all(f.l <= f.m <= f.u for f in r.fuzzy_weights.values())
    assert r.method == "fuzzy_bwm" and r.operators["defuzzification"] == "graded_mean"


def test_fuzzy_bwm_validates():
    S = FUZZY_BWM_TFN
    with pytest.raises(TypeError, match="TFN"):
        FuzzyBWMProblem({"C1": S["EI"], "C2": 2}, {"C1": S["FI"], "C2": S["EI"]})
    with pytest.raises(ValueError, match="must be >= 1"):
        FuzzyBWMProblem({"C1": S["EI"], "C2": TFN(0.5, 0.8, 3)}, {"C1": TFN(0.5, 0.8, 3), "C2": S["EI"]})
    with pytest.raises(ValueError, match="l > 0"):
        FuzzyBWMProblem({"C1": S["EI"], "C2": TFN(0, 2, 3)}, {"C1": TFN(0, 2, 3), "C2": S["EI"]})
    with pytest.raises(ValueError, match="expected 6"):
        FuzzyBWMProblem({"C1": S["EI"], "C2": S["FI"]}, {"C1": S["FI"], "C2": S["EI"]}).evaluate([0.5] * 3)


def test_train_anfis_linear_target():
    X = [[i / 20] for i in range(21)]
    y = [2 * x[0] + 1 for x in X]
    a = ANFIS(1, 2, ranges=[(0, 1)])
    before = a.get_params()
    res = train_anfis(a, X, y, budget=20000, seed=1)
    assert res.rmse < 0.05 and a.get_params() == before and res.evals_used <= 20000  # prototype: 3.7e-4
    assert isinstance(res, ANFISTrainResult) and res.anfis is not a
    assert res.anfis.rmse(X, y) == approx(res.rmse)


def test_anfis_problem_decode_and_inf_penalty():
    a = ANFIS(1, 2, mf="bell", ranges=[(0, 10)])
    p = ANFISProblem(a, [[0.0], [10.0]], [0.0, 0.0], coef_bound=4.0)
    lay = a.param_layout()
    lo = p.decode([0.0] * len(lay))
    hi = p.decode([1.0] * len(lay))
    # span 10: a in [.1, 10], b in [.5, 5], c in [0, 10], consequents in [-4, 4]
    want_lo = {"a": 0.1, "b": 0.5, "c": 0.0, "p": -4.0, "r": -4.0}
    want_hi = {"a": 10.0, "b": 5.0, "c": 10.0, "p": 4.0, "r": 4.0}
    for (kind, _, _), v_lo, v_hi in zip(lay, lo, hi):
        assert v_lo == approx(want_lo[kind]) and v_hi == approx(want_hi[kind])
    # all consequents 0 at z = .5 -> output 0 = target -> RMSE 0
    assert p.evaluate([0.5] * len(lay)) == approx(0.0, abs=1e-12)
    g = ANFISProblem(ANFIS(1, 1, ranges=[(0, 1)]), [[0.0]], [0.0])
    assert [k for k, _, _ in ANFIS(1, 1, ranges=[(0, 1)]).param_layout()] == ["c", "sigma", "p", "r"]
    # gaussian sigma lower bound is span/100 = .01
    assert g.decode([0.0, 0.0, 0.5, 0.5])[1] == approx(0.01)
    # a far-away input underflows every firing strength -> inf, never NaN
    far = ANFISProblem(ANFIS(1, 1, ranges=[(0, 1)]), [[1e200]], [0.0])
    assert far.evaluate([0.0, 0.0, 0.5, 0.5]) == math.inf


def test_anfis_problem_validates():
    with pytest.raises(RuntimeError, match="ranges"):
        ANFISProblem(ANFIS(1, 2), [[0.0]], [0.0])
    with pytest.raises(TypeError, match="ANFIS"):
        ANFISProblem("net", [[0.0]], [0.0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ANFISProblem(ANFIS(1, 1, ranges=[(0, 1)]), [[0.0]], [0.0]).decode([0.0, 1.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="coef_bound"):
        ANFISProblem(ANFIS(1, 2, ranges=[(0, 1)]), [[0.0]], [0.0], coef_bound=0)


def test_missing_extra_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "sezgi", None)
    monkeypatch.delitem(sys.modules, "bulanik.sezgi", raising=False)
    with pytest.raises(ImportError, match=r'pip install "bulanik\[sezgi\]"'):
        importlib.import_module("bulanik.sezgi")


def test_module_restored_after_missing_extra():
    assert sys.modules["bulanik.sezgi"].BWMProblem is BWMProblem


def test_anfis_problem_decode_survives_huge_coef_bound():
    # hi - lo = 2 * coef_bound overflows above 8.9e307; the convex form must not
    p = ANFISProblem(ANFIS(1, 1, ranges=[(0, 1)]), [[0.0]], [0.0], coef_bound=1.7e308)
    lo, hi = p.decode([0.5, 0.5, 0.0, 0.0]), p.decode([0.5, 0.5, 1.0, 1.0])
    assert lo[2:] == [-1.7e308, -1.7e308] and hi[2:] == [1.7e308, 1.7e308]
    mid = p.decode([0.5, 0.5, 0.5, 0.5])
    assert mid[2:] == [0.0, 0.0] and all(math.isfinite(v) for v in mid)


def test_train_anfis_infers_ranges_like_fit():
    X = [[i / 10, 5 + i] for i in range(11)]
    y = [x[0] + 0.1 * x[1] for x in X]
    a = ANFIS(2, 2)  # no ranges
    res = train_anfis(a, X, y, budget=4000, seed=1)
    assert a.ranges is None  # the original is untouched
    assert res.anfis.ranges == [(0.0, 1.0), (5.0, 15.0)]
    assert res.rmse < 0.5
    with pytest.raises(ValueError, match="constant"):
        train_anfis(ANFIS(1, 2), [[1.0], [1.0]], [0.0, 1.0], budget=10)
    with pytest.raises(TypeError, match="ANFIS"):
        train_anfis("net", [[0.0]], [0.0])
