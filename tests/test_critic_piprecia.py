import math

import pytest
from hypothesis import assume, given, strategies as st
from pytest import approx

from bulanik import IFN, TFN
from bulanik.mcdm import WeightingResult, critic, piprecia

# -- CRITIC -------------------------------------------------------------------


def test_critic_shared_case_golden():
    # tests/test_ranking.py case X = [[7, 9], [8, 7], [9, 8]], T = [benefit, cost].
    # Normalised (max-min, cost reversed): col1 = (0, .5, 1), col2 = (0, 1, .5).
    # Both have mean .5, population variance (.25 + 0 + .25) / 3 = 1/6.
    # cov sum = (-.5)(-.5) + 0(.5) + (.5)(0) = .25 and var sums are .5, so rho = .5.
    # C_j = sigma (1 - 1 + 1 - .5) is equal for both columns, so w = (.5, .5).
    r = critic([[7, 9], [8, 7], [9, 8]], ["a", "b"], criteria_types=["benefit", "cost"])
    assert isinstance(r, WeightingResult) and r.method == "critic"
    assert list(r.weights.values()) == approx([0.5, 0.5])
    assert r.extras["std"] == approx({"a": math.sqrt(1 / 6), "b": math.sqrt(1 / 6)})
    assert r.extras["correlation"]["a"]["b"] == approx(0.5)
    assert r.extras["correlation"]["a"]["a"] == approx(1.0)
    assert r.extras["information"]["a"] == approx(math.sqrt(1 / 6) * 0.5)
    assert r.operators == {"weights": "critic"}


def test_critic_golden():
    # X columns a = (1, 2, 3), b = (1, 3, 2), c = (0, 1, 5), all benefit.
    # Max-min: a = (0, .5, 1), b = (0, 1, .5), c = (0, .2, 1).
    # sigma_a = sigma_b = sqrt(1/6); c has mean .4, deviations (-.4, -.2, .6),
    # sum of squares .56, sigma_c = sqrt(.56 / 3).
    # Deviation products: ab .25, ac .5, bc .1; norms: aa = bb = .5, cc = .56.
    # rho_ab = .5, rho_ac = .5 / sqrt(.28), rho_bc = .1 / sqrt(.28).
    X = [[1, 1, 0], [2, 3, 1], [3, 2, 5]]
    sa = sb = math.sqrt(1 / 6)
    sc = math.sqrt(0.56 / 3)
    rab, rac, rbc = 0.5, 0.5 / math.sqrt(0.28), 0.1 / math.sqrt(0.28)
    c = [sa * (2 - rab - rac), sb * (2 - rab - rbc), sc * (2 - rac - rbc)]
    r = critic(X, ["a", "b", "c"])
    assert list(r.weights.values()) == approx([x / sum(c) for x in c])
    assert r.extras["information"]["c"] == approx(c[2])
    assert r.extras["correlation"]["b"]["c"] == approx(rbc)
    assert r.extras["correlation"]["c"]["b"] == approx(rbc)
    assert r.ranking == tuple(sorted("abc", key=lambda k: -r.weights[k]))
    # the default criteria types are all benefit
    assert critic(X, ["a", "b", "c"], criteria_types=["benefit"] * 3).weights == r.weights


def test_critic_cost_reverses_normalisation():
    # reversing a column is the same as negating it: rho flips sign, sigma is unchanged
    X = [[1, 1, 0], [2, 3, 1], [3, 2, 5]]
    cost = critic(X, criteria_types=["benefit", "benefit", "cost"])
    neg = critic([[a, b, -c] for a, b, c in X])
    assert list(cost.weights.values()) == approx(list(neg.weights.values()))
    assert cost.weights != critic(X).weights


def test_critic_population_vs_sample_std_is_irrelevant():
    # sigma (n) and s (n - 1) differ by the constant sqrt(m / (m - 1)) for every column,
    # which cancels in C_j / sum C. Check with the sample deviation by hand.
    X = [[1, 1, 0], [2, 3, 1], [3, 2, 5], [4, 9, 2]]
    r = critic(X)
    m = 4
    scale = math.sqrt(m / (m - 1))
    sample = [s * scale for s in r.extras["std"].values()]
    rho = r.extras["correlation"]
    names = list(rho)
    c = [s * sum(1 - rho[a][b] for b in names) for s, a in zip(sample, names)]
    assert list(r.weights.values()) == approx([x / sum(c) for x in c])


def test_critic_formula_as_printed_in_a_review():
    # Diakoulaki et al. (1995) is paywalled and its printed example could not be read
    # from an open copy, so this pins the formula against the equations as printed in
    # Wang, Nabavi & Rangaiah (arXiv:2407.09931), Eqs. 15.3-15.6: Pearson rho, population
    # sigma, c_j = sigma_j sum_k (1 - rho_jk), w_j = c_j / sum c.
    X = [[3.0, 8.0, 1.0], [5.0, 2.0, 4.0], [4.0, 6.0, 9.0], [9.0, 1.0, 5.0]]
    r = critic(X)
    cols = [[row[j] for row in X] for j in range(3)]
    norm = [[(v - min(c)) / (max(c) - min(c)) for v in c] for c in cols]
    mean = [sum(c) / 4 for c in norm]
    sd = [math.sqrt(sum((v - m) ** 2 for v in c) / 4) for c, m in zip(norm, mean)]

    def rho(j, k):
        num = sum((a - mean[j]) * (b - mean[k]) for a, b in zip(norm[j], norm[k]))
        return num / math.sqrt(
            sum((a - mean[j]) ** 2 for a in norm[j]) * sum((b - mean[k]) ** 2 for b in norm[k])
        )

    cj = [sd[j] * sum(1 - rho(j, k) for k in range(3)) for j in range(3)]
    assert list(r.weights.values()) == approx([x / sum(cj) for x in cj])


def test_critic_degenerate_inputs():
    X = [[1, 1, 0], [2, 3, 1], [3, 2, 5]]
    base = critic(X, ["a", "b", "c"])
    # a constant column gets weight 0 and changes no other weight
    with_const = critic([r[:1] + [7] + r[1:] for r in X], ["a", "k", "b", "c"])
    assert with_const.weights["k"] == 0.0
    assert [with_const.weights[k] for k in "abc"] == approx([base.weights[k] for k in "abc"])
    assert with_const.extras["std"]["k"] == 0.0 and with_const.extras["information"]["k"] == 0.0
    assert "k" not in with_const.extras["correlation"]["a"]
    assert with_const.ranking[-1] == "k"
    # a single informative column carries all the weight
    one = critic([[1, 4], [2, 4], [3, 4]], ["a", "k"])
    assert one.weights == {"a": 1.0, "k": 0.0}
    assert critic([[1], [2], [4]]).weights == {"C1": 1.0}
    with pytest.raises(ValueError, match="at least 2 alternatives"):
        critic([[1, 2, 3]])
    with pytest.raises(ValueError, match="no criterion discriminates"):
        critic([[1, 2], [1, 2], [1, 2]])
    # perfectly redundant columns: the conflict term is zero everywhere
    with pytest.raises(ValueError, match="perfectly correlated"):
        critic([[1, 10], [2, 20], [3, 30]])
    # two alternatives give rho = +-1 only
    with pytest.raises(ValueError, match="perfectly correlated"):
        critic([[1, 2], [3, 4]])
    assert critic([[1, 4], [3, 2]]).weights == approx({"C1": 0.5, "C2": 0.5})


def test_critic_input_errors():
    with pytest.raises(ValueError, match="criteria types"):
        critic([[1, 2], [2, 1]], criteria_types=["benefit"])
    with pytest.raises(ValueError, match="unknown criterion type"):
        critic([[1, 2], [2, 1]], criteria_types=["benefit", "min"])
    with pytest.raises(ValueError, match="criteria names"):
        critic([[1, 2], [2, 1]], ["only"])
    with pytest.raises(ValueError, match="finite"):
        critic([[1, float("nan")], [2, 1]])
    with pytest.raises(TypeError):
        critic([[1, True], [2, 1]])
    with pytest.raises(ValueError):
        critic([[1, 2], [2]])


_matrix = st.integers(2, 4).flatmap(
    lambda n: st.lists(
        st.lists(st.integers(0, 20), min_size=n, max_size=n), min_size=3, max_size=6
    )
)


@given(
    _matrix,
    st.lists(st.floats(0.5, 5.0), min_size=4, max_size=4),
    st.lists(st.floats(-10.0, 10.0), min_size=4, max_size=4),
    st.lists(st.sampled_from(["benefit", "cost"]), min_size=4, max_size=4),
)
def test_critic_affine_invariance(X, scales, shifts, types):
    n = len(X[0])
    try:
        base = critic(X, criteria_types=types[:n])
    except ValueError:
        assume(False)
    moved = [[x * scales[j] + shifts[j] for j, x in enumerate(row)] for row in X]
    got = critic(moved, criteria_types=types[:n])
    assert list(got.weights.values()) == approx(list(base.weights.values()), abs=1e-7)


# -- PIPRECIA -----------------------------------------------------------------


def test_piprecia_golden():
    r = piprecia([1.2, 0.9], ["a", "b", "c"])
    # k = (1, 2 - 1.2, 2 - .9) = (1, .8, 1.1); q = (1, 1/.8, 1.25/1.1) = (1, 1.25, 1.136364)
    q = [1.0, 1.25, 1.25 / 1.1]
    assert list(r.weights.values()) == approx([x / sum(q) for x in q])
    assert sum(q) == approx(3.386364, abs=1e-6)
    assert r.method == "piprecia" and r.ranking == ("b", "c", "a")
    assert r.operators == {"score": "crisp", "inverse": "none"} and r.fuzzy_weights is None
    assert piprecia([], ["only"]).weights == {"only": 1.0}


def test_piprecia_published():
    # Stanujkic, Karabasevic & Popovic (2021), J. Process Management and New Technologies
    # 9(3-4), 62-68, Table 1 (eqs. (1)-(4) of Stanujkic et al. 2017), printed to 2 decimals.
    # s = (.9, 1.2, .95, .9, 1) -> k = (1.1, .8, 1.05, 1.1, 1) -> q = (.91, 1.14, 1.08, .98, .98).
    r = piprecia([0.9, 1.2, 0.95, 0.9, 1.0], ["C1", "C2", "C3", "C4", "C5", "C6"])
    assert list(r.weights.values()) == approx([0.16, 0.15, 0.19, 0.18, 0.16, 0.16], abs=0.005)
    assert r.ranking[0] == "C3" and r.ranking[1] == "C4"


# Stevic et al. (2018) fuzzy PIPRECIA as restated with a worked example in Dalic, Stevic,
# Karamasa & Puska (2020), Decision Making: Applications in Management and Engineering
# 3(1), 126-145, Table 5 ("forward" PIPRECIA, 3 decimals).
_S = [
    TFN(1.100, 1.150, 1.200),
    TFN(1.050, 1.075, 1.125),
    TFN(0.310, 0.367, 0.450),
    TFN(1.150, 1.225, 1.275),
    TFN(0.310, 0.367, 0.450),
    TFN(1.050, 1.075, 1.125),
]
_W = [
    (0.131, 0.154, 0.173),
    (0.146, 0.181, 0.216),
    (0.153, 0.195, 0.247),
    (0.091, 0.120, 0.160),
    (0.107, 0.154, 0.220),
    (0.063, 0.094, 0.142),
    (0.066, 0.102, 0.162),
]
_DEFUZZ = [0.153, 0.181, 0.197, 0.121, 0.157, 0.097, 0.106]


def test_fuzzy_piprecia_published():
    names = [f"C{i}" for i in range(1, 8)]
    r = piprecia(_S, names)
    for name, (l, m, u) in zip(names, _W):
        f = r.fuzzy_weights[name]
        assert (f.l, f.m, f.u) == approx((l, m, u), abs=1.5e-3)
    # the source defuzzifies with the graded mean (l + 4m + u) / 6; weights are normalised
    assert r.operators == {"score": "graded_mean", "inverse": "none"}
    assert list(r.weights.values()) == approx([d / sum(_DEFUZZ) for d in _DEFUZZ], abs=3e-3)
    assert sum(r.weights.values()) == approx(1.0)
    assert r.ranking[0] == "C3" and r.ranking[1] == "C2"
    # intermediate k_2 = (2 - 1.2, 2 - 1.15, 2 - 1.1) = (.8, .85, .9) so q_2 = (1.111, 1.176, 1.25)
    one = piprecia(_S[:1], ["C1", "C2"])
    assert one.fuzzy_weights["C2"].m / one.fuzzy_weights["C1"].m == approx(1 / 0.85)


def test_fuzzy_piprecia_scores_and_crisp_agreement():
    names = ["a", "b", "c"]
    # a degenerate TFN (point numbers) reproduces the crisp method exactly
    pts = piprecia([TFN(1.2, 1.2, 1.2), TFN(0.9, 0.9, 0.9)], names)
    assert list(pts.weights.values()) == approx(list(piprecia([1.2, 0.9], names).weights.values()))
    s = [TFN(1.1, 1.2, 1.3), TFN(0.8, 0.9, 1.0)]
    for score in ("centroid", "graded_mean", "mean"):
        r = piprecia(s, names, score=score)
        assert r.operators == {"score": score, "inverse": "none"}
        got = [r.fuzzy_weights[k].score(score) for k in names]
        assert list(r.weights.values()) == approx([g / sum(got) for g in got])
    with pytest.raises(ValueError, match="unknown score 'foo'"):
        piprecia(s, names, score="foo")
    # crisp inputs ignore a valid score name but still reject a misspelt one
    assert piprecia([1.2, 0.9], names, score="centroid").operators["score"] == "crisp"
    with pytest.raises(ValueError, match="unknown score 'foo'"):
        piprecia([1.2, 0.9], names, score="foo")


def test_piprecia_errors():
    names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="k_j"):
        piprecia([2.0, 1.0], names)  # k = 0
    with pytest.raises(ValueError, match="k_j"):
        piprecia([1.0, 2.5], names)  # k < 0
    with pytest.raises(ValueError, match="k_j"):
        piprecia([TFN(1.5, 1.8, 2.0), TFN(1, 1, 1)], names)  # fuzzy k reaches 0
    with pytest.raises(ValueError, match="mix"):
        piprecia([1.2, TFN(0.8, 0.9, 1.0)], names)
    with pytest.raises(TypeError, match="TFN"):
        piprecia([IFN(0.5, 0.2), IFN(0.4, 0.3)], names)
    with pytest.raises(TypeError, match="TFN"):
        piprecia([TFN(1, 1, 1), IFN(0.4, 0.3)], names)
    with pytest.raises(ValueError, match="expected 2 relative importances"):
        piprecia([1.2], names)
    with pytest.raises(ValueError, match="at least one criterion"):
        piprecia([], [])
    with pytest.raises(ValueError, match="unique"):
        piprecia([1.2], ["a", "a"])
    with pytest.raises(TypeError):
        piprecia([True], ["a", "b"])
    with pytest.raises(TypeError):
        piprecia(["1.2"], ["a", "b"])
    with pytest.raises(ValueError, match="finite"):
        piprecia([float("nan")], ["a", "b"])
    # criteria may be listed in any order, any generator works
    assert piprecia([1.2], (c for c in "ab")).weights.keys() == {"a", "b"}


@given(st.lists(st.floats(0.05, 1.9), min_size=1, max_size=8))
def test_piprecia_weights_sum_to_one(s):
    r = piprecia(s, [f"C{i}" for i in range(len(s) + 1)])
    assert sum(r.weights.values()) == approx(1.0)
    assert all(w > 0 for w in r.weights.values())


@given(st.lists(st.tuples(st.floats(0.1, 1.6), st.floats(0, 0.2), st.floats(0, 0.2)), min_size=1, max_size=6))
def test_fuzzy_piprecia_weights_sum_to_one(parts):
    assume(all(m - a > 0 for m, a, _ in parts))  # the valid domain is s^l > 0
    s = [TFN(m - a, m, m + b) for m, a, b in parts]
    r = piprecia(s, [f"C{i}" for i in range(len(s) + 1)])
    assert sum(r.weights.values()) == approx(1.0)
    for f in r.fuzzy_weights.values():
        assert 0 < f.l <= f.m <= f.u


@given(_matrix)
def test_critic_weights_sum_to_one(X):
    try:
        r = critic(X)
    except ValueError:
        assume(False)
    assert sum(r.weights.values()) == approx(1.0)
    assert all(w >= 0 for w in r.weights.values())


# -- inverse PIPRECIA ---------------------------------------------------------
# Dalic et al. (2020) Table 5 ("P-I" block) and Table 6. s'_j compares criterion j with
# j + 1 (assessed from the penultimate criterion backwards), so s_inverse has n - 1
# entries in the order of `criteria`: s'_1 .. s'_{n-1}.
_SI = [
    TFN(0.517, 0.708, 0.917),
    TFN(0.583, 0.833, 1.000),
    TFN(1.350, 1.525, 1.575),
    TFN(0.450, 0.583, 0.833),
    TFN(1.350, 1.525, 1.575),
    TFN(0.583, 0.833, 1.000),
]
_WI = [
    (0.025, 0.152, 0.750),
    (0.038, 0.197, 0.812),
    (0.053, 0.229, 0.812),
    (0.035, 0.109, 0.345),
    (0.054, 0.154, 0.403),
    (0.035, 0.073, 0.171),
    (0.049, 0.086, 0.171),
]
_II = [0.231, 0.273, 0.297, 0.136, 0.179, 0.083, 0.094]  # Table 6 column II
_FINAL = [0.192, 0.227, 0.247, 0.129, 0.168, 0.090, 0.100]  # Table 6 column w_j


def test_fuzzy_inverse_piprecia_published():
    names = [f"C{i}" for i in range(1, 8)]
    r = piprecia(_S, names, s_inverse=_SI)
    inv = r.extras["fuzzy_inverse"]
    for name, (l, m, u) in zip(names, _WI):
        f = inv[name]
        assert (f.l, f.m, f.u) == approx((l, m, u), abs=3e-3)
    # Table 6: I and II are the graded-mean scores of the forward and inverse weights
    assert list(r.extras["forward"].values()) == approx(_DEFUZZ, abs=2e-3)
    assert list(r.extras["inverse"].values()) == approx(_II, abs=4e-3)
    # the final weight is their mean (the source's sums to 1.153, so it is renormalised)
    assert list(r.weights.values()) == approx([f / sum(_FINAL) for f in _FINAL], abs=4e-3)
    assert sum(r.weights.values()) == approx(1.0)
    assert r.ranking == ("C3", "C2", "C1", "C5", "C4", "C7", "C6")  # Table 6 ranks 1..7
    assert r.operators == {"score": "graded_mean", "inverse": "mean_of_forward_and_inverse"}
    # fuzzy_weights stay the forward ones
    assert r.fuzzy_weights["C1"].m == approx(0.154, abs=1.5e-3)


def test_inverse_piprecia_crisp_golden():
    # hand-derived (no printed crisp inverse example found).
    # forward s = (1.2, .9): k = (1, .8, 1.1), q = (1, 1.25, 1.25/1.1).
    # inverse s' = (1.1, .7), s'_1 compares a with b, s'_2 b with c:
    # k' = (2 - 1.1, 2 - .7, 1) = (.9, 1.3, 1); q'_3 = 1, q'_2 = 1/1.3, q'_1 = q'_2 / .9.
    q = [1.0, 1.25, 1.25 / 1.1]
    qi = [1 / 1.3 / 0.9, 1 / 1.3, 1.0]
    w = [x / sum(q) for x in q]
    wi = [x / sum(qi) for x in qi]
    r = piprecia([1.2, 0.9], ["a", "b", "c"], s_inverse=[1.1, 0.7])
    assert list(r.extras["forward"].values()) == approx(w)
    assert list(r.extras["inverse"].values()) == approx(wi)
    assert list(r.weights.values()) == approx([(a + b) / 2 for a, b in zip(w, wi)])
    assert r.operators == {"score": "crisp", "inverse": "mean_of_forward_and_inverse"}
    assert "fuzzy_inverse" not in r.extras
    assert piprecia([1.2, 0.9], ["a", "b", "c"]).operators["inverse"] == "none"
    assert piprecia([1.2, 0.9], ["a", "b", "c"]).extras == {}
    # a single criterion has no judgements
    assert piprecia([], ["only"], s_inverse=[]).weights == {"only": 1.0}


def test_inverse_piprecia_errors():
    names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="expected 2 inverse relative importances"):
        piprecia([1.2, 0.9], names, s_inverse=[1.1])
    with pytest.raises(ValueError, match="mix"):
        piprecia([1.2, 0.9], names, s_inverse=[TFN(1, 1, 1), TFN(1, 1, 1)])
    with pytest.raises(ValueError, match="mix"):
        piprecia([TFN(1, 1, 1), TFN(1, 1, 1)], names, s_inverse=[1.1, 0.7])
    with pytest.raises(ValueError, match="mix"):
        piprecia([1.2, 0.9], names, s_inverse=[1.1, TFN(1, 1, 1)])
    with pytest.raises(TypeError, match="TFN"):
        piprecia([TFN(1, 1, 1)], names[:2], s_inverse=[IFN(0.4, 0.3)])
    with pytest.raises(ValueError, match=r"k'_j"):
        piprecia([1.2, 0.9], names, s_inverse=[2.0, 0.7])
    with pytest.raises(ValueError, match=r"k'_j"):
        piprecia([TFN(1, 1, 1)], names[:2], s_inverse=[TFN(1.5, 1.8, 2.0)])
    with pytest.raises(TypeError):
        piprecia([1.2, 0.9], names, s_inverse=[True, 0.7])


@given(
    st.lists(st.tuples(st.floats(0.1, 1.9), st.floats(0.1, 1.9)), min_size=1, max_size=6),
    st.booleans(),
)
def test_inverse_piprecia_weights_sum_to_one(pairs, as_tfn):
    n = len(pairs) + 1
    f = [a for a, _ in pairs]
    i = [b for _, b in pairs]
    if as_tfn:
        f = [TFN(0.9 * v, v, 1.05 * v) for v in f]
        i = [TFN(0.9 * v, v, 1.05 * v) for v in i]
    try:
        r = piprecia(f, [f"C{j}" for j in range(n)], s_inverse=i)
    except ValueError:
        assume(False)  # a fuzzy k reached 0
    assert sum(r.weights.values()) == approx(1.0) and all(w > 0 for w in r.weights.values())


@pytest.mark.parametrize("bad", [-5, 0, -0.0, -1e-9])
def test_piprecia_rejects_non_positive_judgements(bad):
    # k = 2 - s > 0 alone accepted s <= 0: [-5] gave {a: .875, b: .125}, [0] gave 2/3, 1/3
    with pytest.raises(ValueError, match=r"s_j > 0.*s_2 = "):
        piprecia([bad], ["a", "b"])
    with pytest.raises(ValueError, match=r"s_3 = "):
        piprecia([1.2, bad], ["a", "b", "c"])
    with pytest.raises(ValueError, match=r"s'_j > 0.*s'_2 = "):
        piprecia([1.2, 0.9], ["a", "b", "c"], s_inverse=[1.1, bad])
    with pytest.raises(ValueError, match=r"s'_1 = "):
        piprecia([1.2, 0.9], ["a", "b", "c"], s_inverse=[bad, 0.7])


def test_piprecia_rejects_non_positive_tfn_judgements():
    with pytest.raises(ValueError, match=r"s_j > 0.*s_2 = "):
        piprecia([TFN(-1.0, 0.5, 1.2)], ["a", "b"])
    with pytest.raises(ValueError, match=r"s'_1 = "):
        piprecia([TFN(1.1, 1.2, 1.3)], ["a", "b"], s_inverse=[TFN(0.0, 0.5, 1.0)])
    # a positive lower bound is fine
    assert piprecia([TFN(0.1, 0.5, 1.2)], ["a", "b"]).fuzzy_weights
