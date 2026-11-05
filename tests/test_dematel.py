import math

import pytest
from pytest import approx

from bulanik import IVIFN, TFN
from bulanik.mcdm import DEMATELResult, WeightingResult, dematel, topsis
from bulanik.mcdm.dematel import cfcs

Z = TFN(0, 0, 0)


def test_dematel_hand_golden():
    r = dematel([[0, 2], [1, 0]])  # s=2, N=[[0,1],[.5,0]], (I-N)^-1=[[2,2],[1,2]], T=[[1,2],[1,1]]
    assert isinstance(r, DEMATELResult) and r.method == "dematel"
    assert r.total == ((approx(1), approx(2)), (approx(1), approx(1)))
    assert r.prominence == approx({"F1": 5, "F2": 5}) and r.relation == approx({"F1": 1, "F2": -1})
    assert r.d == approx({"F1": 3, "F2": 2}) and r.r == approx({"F1": 2, "F2": 3})
    assert r.causes == ("F1",) and r.effects == ("F2",) and r.threshold == approx(1.25)
    assert r.relations == (("F1", "F2", approx(2.0)),)
    assert r.weights == approx({"F1": 0.5, "F2": 0.5})
    assert r.operators == {
        "normalization": "max_row_col", "defuzzification": "crisp",
        "threshold": "mean", "weights": "sqrt",
    }


def test_chain_golden_hand_derived():
    # HAND-DERIVED golden (no published table could be verified).
    # A = [[0,2,0],[0,0,1],[0,0,0]]: s = max(row sums 2,1,0; col sums 0,2,1) = 2,
    # N = [[0,1,0],[0,0,.5],[0,0,0]] is nilpotent, so T = N + N^2 = [[0,1,.5],[0,0,.5],[0,0,0]].
    # D = (1.5, .5, 0), R = (0, 1, 1): D+R = (1.5, 1.5, 1), D-R = (1.5, -.5, -1);
    # threshold = 2/9, so relations are the three positive entries (row-major).
    r = dematel([[0, 2, 0], [0, 0, 1], [0, 0, 0]], ["a", "b", "c"], normalization="max_row")
    # max_row: s = 2 as well
    assert r.total == ((approx(0), approx(1), approx(0.5)), (approx(0),) * 2 + (approx(0.5),), (approx(0),) * 3)
    assert r.prominence == approx({"a": 1.5, "b": 1.5, "c": 1.0})
    assert r.relation == approx({"a": 1.5, "b": -0.5, "c": -1.0})
    assert r.causes == ("a",) and r.effects == ("b", "c")
    assert r.threshold == approx(2 / 9)
    assert r.relations == (("a", "b", approx(1.0)), ("a", "c", approx(0.5)), ("b", "c", approx(0.5)))
    raw = [math.hypot(1.5, 1.5), math.hypot(1.5, 0.5), math.hypot(1.0, 1.0)]
    assert list(r.weights.values()) == approx([x / sum(raw) for x in raw])
    assert r.ranking == ("a", "b", "c")
    p = dematel([[0, 2, 0], [0, 0, 1], [0, 0, 0]], weighting="prominence", threshold=0.6)
    assert list(p.weights.values()) == approx([1.5 / 4, 1.5 / 4, 1 / 4])
    assert p.operators["threshold"] == "0.6" and p.operators["weights"] == "prominence"
    assert [x[:2] for x in p.relations] == [("F1", "F2")]


def test_normalization_choices_differ():
    a = [[0, 2], [1, 0]]
    assert dematel(a, normalization="max_row").total == dematel(a).total  # row max 2 == col max 2
    # c: row sums (2,0,0), column sums (0,1,1) -> both normalisations divide by 2
    c = [[0, 1, 1], [0, 0, 0], [0, 0, 0]]
    assert dematel(c, normalization="max_row").total[0][1] == approx(0.5)
    assert dematel(c).total[0][1] == approx(0.5)
    # d: row sums (0,1,1), column sums (2,0,0): max_row uses s=1 (N=A, T=A, N^2=0), max_row_col uses s=2
    d = [[0, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert dematel(d, normalization="max_row").total[1][0] == approx(1.0)
    assert dematel(d).total[1][0] == approx(0.5)
    assert dematel(d, normalization="max_row").operators["normalization"] == "max_row"
    with pytest.raises(ValueError, match="valid:"):
        dematel(a, normalization="x")
    with pytest.raises(ValueError, match="valid:"):
        dematel(a, weighting="x")


def test_cfcs_hand_golden():
    # HAND-DERIVED from Opricovic & Tzeng (2003) formulas, per cell across experts.
    # Single expert (1,2,4): min=1, max=4, D=3; xl=0, xm=1/3, xr=1; xls=.25, xrs=.6;
    # x=(.25*.75+.36)/(1-.25+.6)=.405556; z=1+3x=2.216667.
    got = cfcs([[[TFN(1, 2, 4), Z], [Z, Z]]])
    assert got[0][0] == approx(2.2166667, abs=1e-6) and got[0][1] == 0.0
    # Two experts (1,2,3),(2,3,5): min l=1, max r=5, D=4.
    # e1: xl=0,xm=.25,xr=.5 -> xls=.2, xrs=.4, x=.32/1.2, z=1+4x=31/15
    # e2: xl=.25,xm=.5,xr=1 -> xls=.4, xrs=2/3, x=(.24+4/9)/(19/15), z=1+4x
    z1 = 31 / 15
    z2 = 1 + 4 * ((0.24 + 4 / 9) / (19 / 15))
    two = cfcs([[[TFN(1, 2, 3), Z], [Z, Z]], [[TFN(2, 3, 5), Z], [Z, Z]]])
    assert two[0][0] == approx((z1 + z2) / 2) and z2 == approx(3.161404, abs=1e-6)
    # crisp-like TFN (3,3,3): D=0 -> z=min
    assert cfcs([[[TFN(3, 3, 3), Z], [Z, Z]]])[0][0] == 3.0
    with pytest.raises(ValueError):
        cfcs([])
    with pytest.raises(ValueError):
        cfcs([[[Z, Z], [Z, Z]], [[Z, Z]]])
    with pytest.raises(TypeError):
        cfcs([[[1.0, Z], [Z, Z]]])


def test_fuzzy_dematel_matches_crisp_on_cfcs_values():
    f = [[Z, TFN(1, 2, 4)], [TFN(2, 3, 5), Z]]
    r = dematel(f)
    crisp = dematel(cfcs([f]))
    assert r.direct == crisp.direct and r.total == crisp.total
    assert r.direct[0][1] == approx(2.2166667, abs=1e-6)
    assert r.operators["defuzzification"] == "cfcs"
    c = dematel(f, defuzzification="centroid")
    assert c.direct == ((0.0, approx(7 / 3)), (approx(10 / 3), 0.0))
    assert c.operators["defuzzification"] == "centroid"


def test_other_kinds_via_score():
    zero = IVIFN(0, 0, 1, 1)
    a = IVIFN(0.6, 0.7, 0.1, 0.2)
    b = IVIFN(0.3, 0.4, 0.3, 0.4)
    r = dematel([[zero, a], [b, zero]], defuzzification="normalized")
    assert r.operators["defuzzification"] == "normalized"
    assert r.direct[0][1] == approx(a.score("normalized"))
    with pytest.raises(ValueError, match="valid:"):
        dematel([[zero, a], [b, zero]], defuzzification="cfcs")
    with pytest.raises(ValueError, match="valid:"):
        dematel([[Z, TFN(1, 2, 3)], [TFN(1, 2, 3), Z]], defuzzification="nope")
    # default score of IVIFN is signed: negatives rejected with a hint
    neg = IVIFN(0.1, 0.2, 0.6, 0.7)
    with pytest.raises(ValueError, match='defuzzification="normalized"'):
        dematel([[zero, neg], [neg, zero]])


def test_singular_and_invalid_matrices():
    with pytest.raises(ValueError, match="singular"):
        dematel([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="spectral radius"):
        dematel([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="zero"):
        dematel([[0, 0], [0, 0]])
    with pytest.raises(ValueError, match="negative|>= 0"):
        dematel([[0, -1], [1, 0]])
    with pytest.raises(ValueError, match="finite"):
        dematel([[0, math.inf], [1, 0]])
    with pytest.raises(ValueError, match="square"):
        dematel([[0, 1, 2], [1, 0, 1]])
    with pytest.raises(ValueError, match="at least 2"):
        dematel([[1]])
    with pytest.raises(TypeError):
        dematel([[0, True], [1, 0]])
    with pytest.raises(TypeError):
        dematel([[0, TFN(1, 2, 3)], [1, 0]])
    with pytest.raises(TypeError):
        dematel([[Z, TFN(1, 2, 3)], [IVIFN(0, 0, 1, 1), Z]])
    with pytest.raises(ValueError, match="factor"):
        dematel([[0, 1], [2, 0]], ["a"])
    with pytest.raises(ValueError, match="unique"):
        dematel([[0, 1], [2, 0]], ["a", "a"])
    with pytest.raises(ValueError, match="threshold"):
        dematel([[0, 1], [2, 0]], threshold=math.nan)


def test_as_weights_feeds_ranking():
    r = dematel([[0, 2, 1], [1, 0, 0], [0, 1, 0]], ["x", "y", "z"])
    w = r.as_weights()
    assert isinstance(w, WeightingResult) and w.method == "dematel"
    assert sum(w.weights.values()) == approx(1.0) and w.criteria == ("x", "y", "z")
    assert w.ranking == r.ranking and w.operators == r.operators
    t = topsis([[1, 2, 3], [3, 2, 1]], w, ["benefit"] * 3)
    assert set(t.ranking) == {"A1", "A2"}


def test_zero_relation_counts_as_effect():
    # Symmetric A with rows sums < max so I-N is regular: D == R up to rounding.
    r = dematel([[0, 1, 2], [1, 0, 1], [2, 1, 0]], normalization="max_row")
    assert all(abs(v) < 1e-12 for v in r.relation.values())
    assert r.causes == () and r.effects == ("F1", "F2", "F3")


def test_ill_conditioned_inverse_is_rejected(monkeypatch):
    import importlib

    mod = importlib.import_module("bulanik.mcdm.dematel")

    monkeypatch.setattr(mod, "inverse", lambda m: [[-5.0, 0.0], [0.0, -5.0]])
    with pytest.raises(ValueError, match="ill-conditioned"):
        dematel([[0, 2], [1, 0]])


def test_rounding_noise_does_not_make_a_cause():
    # Symmetric A: the exact D-R is 0 for every factor, but in floats F3 gets +4.4e-16.
    a = [[0, 3, 1, 0, 3], [3, 0, 0, 3, 3], [1, 0, 0, 4, 0], [0, 3, 4, 0, 5], [3, 3, 0, 5, 0]]
    r = dematel(a)
    assert r.relation["F3"] > 0  # the noise is really there
    assert r.causes == () and r.effects == ("F1", "F2", "F3", "F4", "F5")


def test_crisp_rejects_defuzzification_and_bad_threshold():
    assert dematel([[0, 2], [1, 0]], defuzzification="crisp").method == "dematel"
    assert dematel([[0, 2], [1, 0]], defuzzification=None).method == "dematel"
    with pytest.raises(ValueError, match="valid: crisp"):
        dematel([[0, 2], [1, 0]], defuzzification="cfcs")
    for bad in (True, "0.5"):
        with pytest.raises(TypeError, match="threshold"):
            dematel([[0, 2], [1, 0]], threshold=bad)


def test_r_dematel_vignette_cross_check():
    # R package `dematel` vignette 5x5 example (stats.bris.ac.uk/R/web/packages/dematel).
    # Printed values are approximate (abs=1e-4); bulanik matches exact rational arithmetic (rel=1e-12).
    a = [[0, 2, 4, 3, 4], [3, 0, 2, 0, 0], [3, 3, 0, 0, 0], [3, 0, 4, 0, 2], [4, 0, 2, 0, 0]]
    r = dematel(a)
    f = ["F1", "F2", "F3", "F4", "F5"]
    printed_p = [4.860978, 2.437040, 3.603359, 2.658148, 2.738514]
    printed_r = [0.01399868, -0.14637765, -1.02666409, 1.07807102, 0.08097203]
    exact_p = [4.860957964325808, 2.437004698349341, 3.603334580682716, 2.658130639058667, 2.738505675439691]
    exact_r = [0.013999417903621471, -0.14637229221238202, -1.0266600141366264, 1.0780632821920086, 0.08096960625337823]
    assert [r.prominence[k] for k in f] == approx(printed_p, abs=1e-4)
    assert [r.relation[k] for k in f] == approx(printed_r, abs=1e-4)
    assert [r.prominence[k] for k in f] == approx(exact_p, rel=1e-12)
    assert [r.relation[k] for k in f] == approx(exact_r, rel=1e-12)
    assert r.total[0][0] == approx(0.4889668, abs=1e-4) and r.total[0][2] == approx(0.7026745, abs=1e-4)
    assert r.total[0][0] == approx(0.48896511579560104, rel=1e-12)
    assert r.total[0][2] == approx(0.702673485509958, rel=1e-12)
    assert r.causes == ("F1", "F4", "F5")


def test_nearly_singular_matrix_is_rejected():
    # spectral radius 1 - 5e-11 after normalisation: T ~ 1e10, which is rounding noise
    with pytest.raises(ValueError, match="ill-conditioned"):
        dematel([[0, 1], [1 - 1e-10, 0]])
    # clearly away from singular stays accepted
    r = dematel([[0, 1], [0.9, 0]])
    assert max(max(row) for row in r.total) < 1e3
