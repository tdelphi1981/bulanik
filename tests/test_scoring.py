import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pytest import approx

from bulanik.mcdm import cocoso, codas, copras, electre, marcos, wisp

# shared crisp case (as in test_ranking): C1 benefit, C2 cost; w=(.6,.4)
X, W, T = [[7, 9], [8, 7], [9, 8]], [0.6, 0.4], ["benefit", "cost"]


def test_codas_golden():
    # n: C1 /9 -> (7/9, 8/9, 1); C2 7/x -> (7/9, 1, 7/8); r = w*n:
    #   A1 (.46667, .31111)  A2 (.53333, .40000)  A3 (.60000, .35000); ns = (.46667, .31111)
    # E = (0, 1/9, 5/36) = (0, .11111, .13889); T = (0, .15556, .17222)
    # all |dE| >= .02: H1 = -(.25) - (.32778) = -.57778 ; H2 = (.11111+.15556) + (-.02778-.01667) = .22222
    # H3 = (.13889+.17222) + (.02778+.01667) = .35556  (H sums to 0)
    r = codas(X, W, T)
    assert list(r.scores.values()) == approx([-0.57778, 0.22222, 0.35556], abs=5e-6)
    assert r.ranking == ("A3", "A2", "A1")
    assert r.method == "codas"
    assert r.extras["E"] == approx({"A1": 0.0, "A2": 1 / 9, "A3": 5 / 36})
    assert r.extras["T"]["A3"] == approx(0.17222, abs=5e-6)
    assert r.operators == {"normalization": "linear_max", "tau": "0.02", "psi": "equation"}


def test_codas_published():
    # Keshavarz Ghorabaee et al. (2016), Table 1 (data and weights of Example 1, robot selection).
    # The paper's printed headers of the 2nd and 3rd column are swapped relative to its data and
    # to Tables 2-3: column 2 (0.4, 0.15, ...) is the cost, column 3 the benefit, weights
    # (.036, .192, .326, .326, .12). Table 3 gives E and T to 4 dp.
    robots = [
        [60, 0.4, 2540, 500, 990],
        [6.35, 0.15, 1016, 3000, 1041],
        [6.8, 0.10, 1727.2, 1500, 1676],
        [10, 0.2, 1000, 2000, 965],
        [2.5, 0.10, 560, 500, 915],
        [4.5, 0.08, 1016, 350, 508],
        [3, 0.1, 1778, 1000, 920],
    ]
    w = [0.036, 0.192, 0.326, 0.326, 0.120]
    r = codas(robots, w, ["benefit", "cost", "benefit", "benefit", "benefit"])
    e = [0.2593, 0.3032, 0.2415, 0.1947, 0.1199, 0.1644, 0.2087]
    t = [0.3394, 0.4510, 0.4762, 0.3114, 0.1606, 0.2133, 0.3720]
    assert list(r.extras["E"].values()) == approx(e, abs=6e-5)
    assert list(r.extras["T"].values()) == approx(t, abs=6e-5)
    # the paper's ranking A2 > A3 > A1 > A7 > A4 > A6 > A5 (Table 4)
    assert r.ranking == ("A2", "A3", "A1", "A7", "A4", "A6", "A5")


def test_codas_published_walkthrough():
    # Wang & Rangaiah (2025), arXiv:2508.16087, Sec. 7.7: Tables 7.1 and 7.14-7.17, implements
    # psi(x) = 1 if |x| >= tau (Eq. 10 of the 2016 paper), tau = .02. Printed to 4 dp.
    acm = [[0.185, 2.33, 454], [0.317, 1.08, 298], [0.555, 6.45, 174], [0.731, 8.88, 849], [0.948, 7.39, 517]]
    r = codas(acm, [0.25, 0.4, 0.35], ["benefit", "cost", "benefit"])
    assert list(r.extras["E"].values()) == approx([0.1790, 0.3568, 0.0993, 0.3133, 0.2461], abs=6e-5)
    assert list(r.extras["T"].values()) == approx([0.2522, 0.4373, 0.1159, 0.4223, 0.3524], abs=6e-5)
    assert list(r.scores.values()) == approx([-0.6187, 1.1957, -1.6986, 0.9034, 0.2183], abs=2e-4)
    assert r.ranking == ("A2", "A4", "A5", "A1", "A3")


ROBOTS = [
    [60, 0.4, 2540, 500, 990],
    [6.35, 0.15, 1016, 3000, 1041],
    [6.8, 0.10, 1727.2, 1500, 1676],
    [10, 0.2, 1000, 2000, 965],
    [2.5, 0.10, 560, 500, 915],
    [4.5, 0.08, 1016, 350, 508],
    [3, 0.1, 1778, 1000, 920],
]


def test_codas_psi_tables_reproduces_the_paper_example_1():
    # Keshavarz Ghorabaee et al. (2016), Table 4: the H column, computed with the taxicab term
    # added when |dE| < tau (the paper's tables), not as printed in Eq. (10)
    r = codas(ROBOTS, [0.036, 0.192, 0.326, 0.326, 0.120], ["benefit", "cost", "benefit", "benefit", "benefit"],
              psi="tables")
    assert list(r.scores.values()) == approx(
        [0.1865, 0.6304, 0.3357, -0.1895, -0.6521, -0.3411, 0.0300], abs=2e-4
    )
    assert r.ranking == ("A2", "A3", "A1", "A7", "A4", "A6", "A5")
    assert r.operators["psi"] == "tables"
    # the default (printed equation) gives a different H on the same data
    d = codas(ROBOTS, [0.036, 0.192, 0.326, 0.326, 0.120], ["benefit", "cost", "benefit", "benefit", "benefit"])
    assert d.operators["psi"] == "equation" and d.scores["A1"] == approx(0.5122, abs=2e-4)


def test_codas_psi_tables_flips_the_threshold_branch():
    base = codas(X, W, T)  # |dE| = .02778 >= .02 for the A2/A3 pair
    e, t = base.extras["E"], base.extras["T"]
    big = codas(X, W, T, psi="tables", tau=0.5)  # all |dE| < .5 -> taxicab counts everywhere
    assert list(big.scores.values()) == approx(list(codas(X, W, T, tau=0.0).scores.values()))
    # tau = 0: psi="tables" never adds the taxicab term, so H_i = sum_k (E_i - E_k)
    tab0 = codas(X, W, T, psi="tables", tau=0.0)
    assert tab0.scores["A2"] == approx(3 * e["A2"] - sum(e.values()))
    assert t["A2"] > 0


def test_codas_psi_unknown_name():
    with pytest.raises(ValueError, match="unknown psi 'both' for codas; valid: equation, tables"):
        codas(X, W, T, psi="both")


def test_codas_antisymmetry():
    assert sum(codas(X, W, T).scores.values()) == approx(0, abs=1e-12)


def test_codas_ties_and_identical_alternatives():
    # identical rows tie exactly and keep input order
    r = codas([[7, 9], [7, 9], [9, 8]], W, T, alternatives=["p", "q", "r"])
    assert r.scores["p"] == r.scores["q"] and r.ranking == ("r", "p", "q")
    # |dE| == tau exactly counts as "different enough" (psi uses >=): taking tau from the extras
    base = codas(X, W, T)
    e = base.extras["E"]
    tau = abs(e["A3"] - e["A2"])
    at = codas(X, W, T, tau=tau)
    above = codas(X, W, T, tau=math.nextafter(tau, 1.0))
    # with psi(A2,A3) = 1 the scores equal the default (tau .02 < dE=.02778 there too)
    assert list(at.scores.values()) == approx(list(base.scores.values()))
    # with psi(A2,A3) = 0 only the E difference counts for that pair: H2 loses (T2-T3)
    t = base.extras["T"]
    assert above.scores["A2"] - at.scores["A2"] == approx(-(t["A2"] - t["A3"]))
    assert above.scores["A3"] - at.scores["A3"] == approx(-(t["A3"] - t["A2"]))
    # tau = 0 and tau = 1 are the allowed extremes
    assert codas(X, W, T, tau=0.0).ranking == ("A3", "A2", "A1")
    assert codas(X, W, T, tau=1.0).ranking == ("A3", "A2", "A1")
    assert codas(X, W, T, tau=1.0).operators["tau"] == "1.0"


def test_codas_constant_column_is_neutral():
    r = codas([[7, 5], [8, 5], [9, 5]], W, T)
    assert r.ranking == ("A3", "A2", "A1")
    assert r.extras["E"]["A1"] == 0.0


def test_codas_accepts_zero_in_benefit_column():
    r = codas([[0, 16], [200, 12], [300, 32]], [0.5, 0.5], ["benefit", "benefit"])
    assert len(r.ranking) == 3 and r.ranking[0] == "A3"
    # alternative 1 normalises to 0 in column 1, so its weighted entry is the column minimum
    assert r.extras["E"]["A1"] > 0.0


def test_codas_rejects_zero_in_cost_column():
    with pytest.raises(ValueError, match="cost") as ei:
        codas([[5, 0], [6, 1]], W, T, criteria=["price", "risk"])
    assert "'risk'" in str(ei.value) and 'score="normalized"' in str(ei.value)


def test_codas_rejects_all_zero_benefit_column():
    with pytest.raises(ValueError, match="positive") as ei:
        codas([[0, 1], [0, 2]], W, ["benefit", "benefit"], criteria=["gain", "other"])
    assert "'gain'" in str(ei.value)


def test_codas_rejects_negative_entry():
    for types in (["benefit", "cost"], ["benefit", "benefit"]):
        with pytest.raises(ValueError, match="negative"):
            codas([[-1, 1], [1, 1]], W, types)


def test_codas_degenerate_inputs():
    with pytest.raises(ValueError, match="positive") as ei:
        codas([[1, 0], [1, 1]], W, T)
    assert 'score="normalized"' in str(ei.value) and "CODAS" in str(ei.value)
    with pytest.raises(ValueError, match="positive"):
        codas([[-1, 1], [1, 1]], W, T)
    for bad in (1.5, -0.1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            codas(X, W, T, tau=bad)
    with pytest.raises(ValueError, match="finite"):
        codas(X, W, T, tau=float("nan"))
    with pytest.raises(TypeError):
        codas(X, W, T, tau=True)


def test_codas_zero_weight_ignored():
    a = codas(X, [1.0, 0.0], T)
    b = codas([[7], [8], [9]], [1.0], ["benefit"])
    assert list(a.scores.values()) == approx(list(b.scores.values()))


def test_copras_golden():
    # col sums 24, 24. d1 = .6 x/24 = (.175, .2, .225) benefit ; d2 = .4 x/24 = (.15, .11667, .13333) cost
    # sum S- = .4 ; sum 1/S- = 6.66667 + 8.57143 + 7.5 = 22.73810
    # Q1 = .175 + .4/(.15 * 22.7381) = .29228 ; Q2 = .2 + .4/(.11667*22.7381) = .35079
    # Q3 = .225 + .4/(.13333*22.7381) = .35694 ; utility = 100 Q/Qmax
    r = copras(X, W, T)
    assert list(r.scores.values()) == approx([0.29228, 0.35079, 0.35694], abs=5e-6)
    assert r.ranking == ("A3", "A2", "A1") and r.method == "copras"
    assert r.extras["S_plus"] == approx({"A1": 0.175, "A2": 0.2, "A3": 0.225})
    assert r.extras["S_minus"] == approx({"A1": 0.15, "A2": 0.35 / 3, "A3": 0.4 / 3})
    assert r.extras["utility"]["A3"] == approx(100.0)
    assert r.extras["utility"]["A1"] == approx(100 * 0.29228 / 0.35694, abs=1e-3)
    assert r.operators == {"normalization": "sum"}


def test_copras_published():
    # Wang & Rangaiah (2025), arXiv:2509.06388, Sec. 8.6, Tables 8.1 and 8.14 (4 dp, rounded inputs)
    acm = [[0.93, 600, 8.25], [0.51, 700, 6.33], [0.77, 500, 3.16], [0.82, 400, 2.98]]
    r = copras(acm, [0.25, 0.33, 0.42], ["benefit", "benefit", "cost"])
    assert list(r.extras["S_plus"].values()) == approx([0.1667, 0.1471, 0.1385, 0.1277], abs=6e-5)
    assert list(r.extras["S_minus"].values()) == approx([0.1672, 0.1283, 0.0641, 0.0604], abs=6e-5)
    assert list(r.scores.values()) == approx([0.2214, 0.2183, 0.2813, 0.2790], abs=2e-4)
    assert r.ranking == ("A3", "A4", "A1", "A2")


def test_copras_only_cost_and_only_benefit():
    # only benefit: Q = S+ (col sums 24 -> .6 x/24 + .4 x/24 ... here one column each)
    b = copras(X, W, ["benefit", "benefit"])
    assert b.scores == approx(b.extras["S_plus"])
    assert b.scores["A3"] == approx(0.6 * 9 / 24 + 0.4 * 8 / 24)
    # only cost: S+ = 0, Q = sum S- / (S-_i sum 1/S-)
    c = copras(X, W, ["cost", "cost"])
    s = c.extras["S_minus"]
    inv = sum(1 / v for v in s.values())
    assert c.scores["A2"] == approx(sum(s.values()) / (s["A2"] * inv))
    assert set(c.extras["S_plus"].values()) == {0.0}


def test_copras_ties_and_identical_alternatives():
    r = copras([[7, 9], [7, 9], [9, 8]], W, T, alternatives=["p", "q", "r"])
    assert r.scores["p"] == r.scores["q"]
    assert r.ranking.index("p") < r.ranking.index("q")
    # all alternatives equal -> equal scores, order kept
    e = copras([[3, 4], [3, 4]], W, T)
    assert e.ranking == ("A1", "A2") and e.extras["utility"]["A1"] == approx(100.0)


def test_copras_zero_entries_and_zero_weight():
    # a zero in a benefit column is fine; a zero benefit column sum is not
    assert copras([[0, 9], [8, 7], [9, 8]], W, T).ranking[0] in {"A2", "A3"}
    # a zero-weight cost criterion is ignored, so its zeros are harmless and Q = S+
    z = copras([[7, 0], [8, 0], [9, 0]], [1.0, 0.0], T)
    assert z.scores == approx(z.extras["S_plus"])


def test_copras_degenerate_inputs():
    # zero cost entry in a single-cost model -> S-_i = 0, names the alternative
    with pytest.raises(ValueError, match="alternative 'bad'.*cost") as ei:
        copras([[7, 9], [8, 0], [9, 8]], W, T, alternatives=["ok", "bad", "fine"])
    assert "COPRAS" in str(ei.value)
    with pytest.raises(ValueError, match="non-negative"):
        copras([[7, 9], [-8, 7], [9, 8]], W, T)
    with pytest.raises(ValueError, match="positive sum.*'C1'"):
        copras([[0, 9], [0, 7], [0, 8]], W, T)
    with pytest.raises(ValueError, match="finite"):
        copras([[7, float("inf")], [8, 7]], W, T)


def test_copras_huge_and_tiny_entries():
    # sum(col) = 2.4e308 overflows; COPRAS is column scale-invariant
    big = copras([[1.6e308, 1], [0.8e308, 2]], [0.5, 0.5], ["benefit", "cost"])
    small = copras([[1.6, 1], [0.8, 2]], [0.5, 0.5], ["benefit", "cost"])
    assert list(big.scores.values()) == approx(list(small.scores.values()))
    t = 5e-324
    tiny = copras([[t, 1], [2 * t, 2], [4 * t, 5]], [0.5, 0.5], ["benefit", "cost"])
    plain = copras([[1, 1], [2, 2], [4, 5]], [0.5, 0.5], ["benefit", "cost"])
    assert list(tiny.scores.values()) == approx(list(plain.scores.values()), abs=1e-12)
    # S- of 1e-300 would overflow 1/S-; the ratio form stays finite
    r = copras([[1, 1e-300], [2, 1e-300], [3, 3e-300]], [0.5, 0.5], ["benefit", "cost"])
    assert all(math.isfinite(v) for v in r.scores.values())


def test_codas_huge_entries_are_safe():
    big = codas([[1.6e308, 1], [0.8e308, 2]], [0.5, 0.5], ["benefit", "benefit"])
    small = codas([[1.6, 1], [0.8, 2]], [0.5, 0.5], ["benefit", "benefit"])
    assert list(big.scores.values()) == approx(list(small.scores.values()))
    t = 5e-324
    tiny = codas([[t, 1], [2 * t, 2], [4 * t, 5]], [0.5, 0.5], ["benefit", "cost"])
    plain = codas([[1, 1], [2, 2], [4, 5]], [0.5, 0.5], ["benefit", "cost"])
    assert list(tiny.scores.values()) == approx(list(plain.scores.values()), abs=1e-12)


def test_weights_forms_and_names():
    from bulanik.mcdm import swara

    fns = (codas, copras, marcos, wisp)
    base = [fn(X, W, T) for fn in fns]
    for fn, ref in zip(fns, base):
        assert fn(X, {"C1": 0.6, "C2": 0.4}, T).scores == ref.scores
        assert fn(X, [3, 2], T, alternatives=["x", "y", "z"], criteria=["a", "b"],
                  normalize_weights=True).ranking == ("z", "y", "x")
        wr = swara([0.3], ["q", "p"])
        assert fn(X, wr, T).scores == fn(X, [wr.weights["q"], wr.weights["p"]], T).scores
        with pytest.raises(ValueError, match="valid: benefit, cost"):
            fn(X, W, ["benefit", "profit"])
        with pytest.raises(ValueError, match="sum to 1"):
            fn(X, [0.6, 0.6], T)


_cell = st.floats(min_value=0.1, max_value=100.0, allow_nan=False)


@st.composite
def _problem(draw):
    m = draw(st.integers(3, 5))
    n = draw(st.integers(2, 4))
    x = [[draw(_cell) for _ in range(n)] for _ in range(m)]
    types = [draw(st.sampled_from(["benefit", "cost"])) for _ in range(n)]
    raw = [draw(st.floats(min_value=0.1, max_value=1.0)) for _ in range(n)]
    return x, [r / sum(raw) for r in raw], types


@pytest.mark.parametrize("fn", [codas, copras, marcos, cocoso, wisp, electre])
@given(prob=_problem(), c=st.floats(min_value=0.1, max_value=100.0), data=st.data())
@settings(max_examples=60, deadline=None)
def test_invariant_to_positive_column_scaling(fn, prob, c, data):
    x, w, types = prob
    j = data.draw(st.integers(0, len(w) - 1))
    y = [[v * c if k == j else v for k, v in enumerate(row)] for row in x]
    try:
        a, b = fn(x, w, types), fn(y, w, types)
    except ValueError:
        assume(fn is not cocoso)  # CoCoSo rejects an alternative that is worst everywhere
        raise
    assert list(b.scores.values()) == approx(list(a.scores.values()), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("fn", [codas, copras, marcos, cocoso, wisp])
@given(prob=_problem(), seed=st.randoms(use_true_random=False))
@settings(max_examples=60, deadline=None)
def test_permutation_equivariance(fn, prob, seed):
    x, w, types = prob
    names = [f"a{i}" for i in range(len(x))]
    order = list(range(len(x)))
    seed.shuffle(order)
    try:
        a = fn(x, w, types, alternatives=names)
    except ValueError:
        assume(fn is not cocoso)
        raise
    b = fn([x[i] for i in order], w, types, alternatives=[names[i] for i in order])
    for name in names:
        assert b.scores[name] == approx(a.scores[name], rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------- MARCOS


def test_marcos_golden():
    # AI = (9, 7), AAI = (7, 9) (C1 benefit, C2 cost). n: C1 x/9, C2 7/x; v = w n:
    #   AAI (7/9, 7/9) -> S = 7/9 = .77778 ; AI (1, 1) -> S = 1
    #   A1 (7/9, 7/9) S = .77778 ; A2 (8/9, 1) S = .6*8/9 + .4 = .93333 ; A3 (1, 7/8) S = .6 + .35 = .95
    # K- = S/S_AAI = (1, 1.2, 1.22143) ; K+ = S/S_AI = (.77778, .93333, .95)
    # f(K-) = K+/(K+ + K-) = .4375 and f(K+) = .5625 for every alternative (K+ = K- S_AAI/S_AI)
    # f(K) = (K+ + K-)/(1 + f(K-)/f(K+) + f(K+)/f(K-)) = (K+ + K-)/(1 + 7/9 + 9/7) -> .58031, .69637, .70881
    r = marcos(X, W, T)
    assert list(r.scores.values()) == approx([0.58031, 0.69637, 0.70881], abs=5e-6)
    assert r.ranking == ("A3", "A2", "A1") and r.method == "marcos"
    assert list(r.extras["K_minus"].values()) == approx([1.0, 1.2, 1.2214286], abs=5e-7)
    assert list(r.extras["K_plus"].values()) == approx([7 / 9, 14 / 15, 0.95], abs=5e-7)
    assert list(r.extras["S"].values()) == approx([7 / 9, 14 / 15, 0.95])
    assert r.operators == {"normalization": "ideal_ratio"}


def test_marcos_published():
    # Ulutas, Karabasevic, Popovic, Stanujkic, Nguyen & Karakoy (2020), Mathematics 8(10), 1672
    # (MARCOS of Stevic et al. 2020 on 8 stackers): Table 1 (matrix), Table 4 (combined weights,
    # printed to 4 dp and summing to 1.0001, hence normalize_weights), Table 8 (Z-, Z+, f(Z), 4 dp).
    # The paper computed from unrounded weights: allow 5e-4 on Z, 2e-4 on f(Z).
    stackers = [
        [660, 1000, 1600, 18, 1200],
        [800, 1000, 1600, 24, 900],
        [980, 1000, 2500, 24, 900],
        [920, 1500, 1600, 24, 900],
        [1380, 1500, 1500, 24, 1150],
        [1230, 1000, 1600, 24, 1150],
        [680, 1500, 1600, 18, 1100],
        [960, 2000, 1600, 12, 1150],
    ]
    w = [0.1061, 0.3476, 0.3330, 0.1185, 0.0949]
    r = marcos(stackers, w, ["cost", "benefit", "benefit", "benefit", "benefit"], normalize_weights=True)
    assert list(r.extras["K_minus"].values()) == approx(
        [1.2199, 1.1970, 1.3843, 1.3331, 1.2988, 1.1774, 1.3567, 1.4128], abs=5e-4
    )
    assert list(r.extras["K_plus"].values()) == approx(
        [0.6767, 0.6640, 0.7679, 0.7395, 0.7205, 0.6531, 0.7526, 0.7837], abs=5e-4
    )
    assert list(r.scores.values()) == approx(
        [0.5649, 0.5543, 0.6410, 0.6173, 0.6014, 0.5452, 0.6283, 0.6542], abs=2e-4
    )
    # Table 8 ranking: Stc8, Stc3, Stc7, Stc4, Stc5, Stc1, Stc2, Stc6
    assert r.ranking == ("A8", "A3", "A7", "A4", "A5", "A1", "A2", "A6")


def test_marcos_degenerate_inputs():
    with pytest.raises(ValueError, match="positive") as ei:
        marcos([[0, 1], [1, 1]], W, T)
    assert 'score="normalized"' in str(ei.value) and "MARCOS" in str(ei.value)
    with pytest.raises(ValueError, match="positive"):
        marcos([[-1, 1], [1, 1]], W, T)
    with pytest.raises(ValueError, match="finite"):
        marcos([[1, float("nan")], [1, 1]], W, T)


def test_marcos_ties_constant_column_and_zero_weight():
    r = marcos([[7, 9], [7, 9], [9, 8]], W, T, alternatives=["p", "q", "r"])
    assert r.scores["p"] == r.scores["q"] and r.ranking == ("r", "p", "q")
    # a constant column adds the same weight to AAI, AI and every alternative
    c = marcos([[7, 5], [8, 5], [9, 5]], W, T)
    assert c.ranking == ("A3", "A2", "A1")
    a = marcos(X, [1.0, 0.0], T)
    b = marcos([[7], [8], [9]], [1.0], ["benefit"])
    assert list(a.scores.values()) == approx(list(b.scores.values()))
    # all alternatives identical: all equal, input order
    e = marcos([[3, 4], [3, 4]], W, T)
    assert e.ranking == ("A1", "A2") and e.scores["A1"] == approx(e.scores["A2"])


def test_marcos_extreme_scales_are_safe():
    big = marcos([[1.6e308, 1], [0.8e308, 2]], [0.5, 0.5], ["benefit", "cost"])
    small = marcos([[1.6, 1], [0.8, 2]], [0.5, 0.5], ["benefit", "cost"])
    assert list(big.scores.values()) == approx(list(small.scores.values()))
    t = 5e-324
    tiny = marcos([[t, 1], [2 * t, 2], [4 * t, 5]], [0.5, 0.5], ["benefit", "cost"])
    plain = marcos([[1, 1], [2, 2], [4, 5]], [0.5, 0.5], ["benefit", "cost"])
    assert list(tiny.scores.values()) == approx(list(plain.scores.values()), abs=1e-12)
    # ratios so extreme that S_AAI underflows to 0 give a clear error, not a ZeroDivisionError
    with pytest.raises(ValueError, match="MARCOS"):
        marcos([[1e-300, 1e300], [1e300, 1e-300]], [0.5, 0.5], ["benefit", "benefit"])


# ---------------------------------------------------------------- CoCoSo

# X, W, T itself has an alternative (A1) that is worst on both criteria, which CoCoSo rejects;
# the golden uses the matrix below instead: C1 benefit, C2 cost
XC = [[7, 8], [8, 7], [9, 9]]


def test_cocoso_golden():
    # C1 (7, 8, 9): r = (x-7)/2 = (0, .5, 1) ; C2 cost (8, 7, 9): r = (9-x)/2 = (.5, 1, 0)
    # S = .6 r1 + .4 r2 = (.2, .7, .6) ; P = r1^.6 + r2^.4 = (0 + .5^.4, .5^.6 + 1, 1 + 0)
    #   = (.757858, 1.659754, 1) ; lambda = .5
    # k_a = (P+S)/sum(P+S) : P+S = (.957858, 2.359754, 1.6), sum 4.917612 -> (.194781, .479858, .325361)
    # k_b = S/min S + P/min P = (1+1, 3.5+2.190058, 3+1.319508) -> (2, 5.690058, 4.319508)
    # k_c = (.5 S + .5 P)/(.5 max S + .5 max P) = (P+S)/(.7+1.659754) -> (.405914, 1, .678037)
    # k = (k_a k_b k_c)^(1/3) + (k_a + k_b + k_c)/3
    r = cocoso(XC, W, T)
    assert list(r.extras["S"].values()) == approx([0.2, 0.7, 0.6])
    assert list(r.extras["P"].values()) == approx([0.757858, 1.659754, 1.0], abs=5e-7)
    assert list(r.extras["k_a"].values()) == approx([0.194781, 0.479858, 0.325361], abs=5e-7)
    assert list(r.extras["k_b"].values()) == approx([2.0, 5.690058, 4.319508], abs=5e-7)
    assert list(r.extras["k_c"].values()) == approx([0.405914, 1.0, 0.678037], abs=5e-7)
    k = [(a * b * c) ** (1 / 3) + (a + b + c) / 3 for a, b, c in zip(
        r.extras["k_a"].values(), r.extras["k_b"].values(), r.extras["k_c"].values())]
    assert list(r.scores.values()) == approx(k)
    assert r.ranking == ("A2", "A3", "A1") and r.method == "cocoso"
    assert r.operators == {"normalization": "min_max", "lambda": "0.5"}


def test_cocoso_published():
    # Reference-implementation output, NOT the paper's printed tables (Management Decision 57(9) is
    # paywalled): R package cocosoR (CRAN) README, example_cocoso (7 alternatives, 5 criteria,
    # lambda = 0.5, the robot-selection data), 6 dp; pymcdm 1.4.0 test_mcdm.TestCOCOSO agrees at 3 dp.
    robots = [
        [60, 0.4, 2540, 500, 990],
        [6.35, 0.15, 1016, 3000, 1041],
        [6.8, 0.10, 1727.2, 1500, 1676],
        [10, 0.2, 1000, 2000, 965],
        [2.5, 0.10, 560, 500, 915],
        [4.5, 0.08, 1016, 350, 508],
        [3, 0.1, 1778, 1000, 920],
    ]
    w = [0.036, 0.192, 0.326, 0.326, 0.120]
    r = cocoso(robots, w, ["benefit", "cost", "benefit", "benefit", "benefit"])
    assert list(r.extras["S"].values()) == approx(
        [0.4299734, 0.6082495, 0.6363392, 0.4470733, 0.2402679, 0.2683310, 0.5031435], abs=5e-7)
    assert list(r.extras["P"].values()) == approx(
        [3.291361, 4.390712, 4.502038, 4.205820, 2.260980, 2.505710, 4.199086], abs=5e-6)
    assert list(r.extras["k_a"].values()) == approx(
        [0.13062317, 0.17546935, 0.18036300, 0.16332196, 0.08779670, 0.09737208, 0.16505373], abs=5e-7)
    assert list(r.extras["k_b"].values()) == approx(
        [3.245282, 4.473498, 4.639646, 3.720904, 2.0, 2.225040, 3.951291], abs=5e-6)
    assert list(r.extras["k_c"].values()) == approx(
        [0.7242238, 0.9728678, 1.0, 0.9055181, 0.4867778, 0.5398672, 0.9151197], abs=5e-7)
    assert list(r.scores.values()) == approx(
        [2.041313, 2.787990, 2.882350, 2.416046, 1.298692, 1.443143, 2.519094], abs=5e-6)
    assert r.ranking == ("A3", "A2", "A7", "A4", "A1", "A6", "A5")


def test_cocoso_lambda_and_constant_column():
    for lam in (0.0, 1.0):
        r = cocoso(XC, W, T, lam=lam)
        assert all(math.isfinite(v) for v in r.scores.values())
        assert r.operators["lambda"] == str(lam)
    # lambda = 1: k_c = S/max S ; lambda = 0: k_c = P/max P
    one = cocoso(XC, W, T, lam=1.0).extras["k_c"]
    assert one == approx({"A1": 0.2 / 0.7, "A2": 1.0, "A3": 0.6 / 0.7})
    for bad in (1.2, -0.1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            cocoso(XC, W, T, lam=bad)
    with pytest.raises(ValueError, match="finite"):
        cocoso(XC, W, T, lam=float("nan"))
    with pytest.raises(TypeError):
        cocoso(XC, W, T, lam=True)
    # a constant column has r = 0 for everyone: S and P are the sums over the other columns
    c = cocoso([[7, 5, 8], [8, 5, 7], [9, 5, 9]], [0.5, 0.2, 0.3], ["benefit", "cost", "cost"])
    s = {"A1": 0.0 * 0.5 + 0.3 * 0.5, "A2": 0.5 * 0.5 + 0.3 * 1.0, "A3": 0.5 * 1.0 + 0.0}
    p = {"A1": 0.0 + 0.5 ** 0.3, "A2": 0.5 ** 0.5 + 1.0, "A3": 1.0 + 0.0}
    assert c.extras["S"] == approx(s) and c.extras["P"] == approx(p)


def test_cocoso_ties_and_identical_alternatives():
    # A1 is worst on every criterion: min S = 0, k_b would divide by zero
    with pytest.raises(ValueError, match="alternative 'A1' is worst on every criterion") as ei:
        cocoso(X, W, T)
    assert "CoCoSo" in str(ei.value) and "k_b" in str(ei.value)
    # a tie for worst on every criterion names the first one
    with pytest.raises(ValueError, match="alternative 'p' is worst on every criterion"):
        cocoso([[7, 9], [7, 9], [9, 7]], W, T, alternatives=["p", "q", "r"])
    # identical alternatives that are not worst tie exactly and keep input order
    r = cocoso([[7, 8], [8, 7], [8, 7], [9, 9]], W, T, alternatives=["a", "b", "c", "d"])
    assert r.scores["b"] == r.scores["c"] and r.ranking.index("b") < r.ranking.index("c")
    # a zero-weight criterion is ignored: A1 is worst on the only weighted one
    with pytest.raises(ValueError, match="alternative 'A1' is worst"):
        cocoso([[7, 8], [8, 7], [9, 9]], [1.0, 0.0], T)


def test_cocoso_degenerate_inputs():
    with pytest.raises(ValueError, match="finite"):
        cocoso([[1, float("inf")], [2, 3], [4, 5]], W, T)
    # negative and zero entries are fine for CoCoSo
    r = cocoso([[-5, 0], [0, -3], [5, 4]], [0.5, 0.5], ["benefit", "benefit"])
    assert r.ranking[0] == "A3"
    big = cocoso([[1.6e308, 1], [0.8e308, 1.5], [1.0e308, 2]], [0.5, 0.5], ["benefit", "cost"])
    assert all(math.isfinite(v) for v in big.scores.values())


# ---------------------------------------------------------------- WISP


def test_wisp_golden():
    # r = x / max (C1 benefit, C2 cost): A1 (7/9, 1) A2 (8/9, 7/9) A3 (1, 8/9) ; w = (.6, .4)
    # u_sd = .6 r1 - .4 r2 = (.194781?)...: (.46667-.4, .53333-.31111, .6-.35556) = (.06667, .22222, .24444)
    # u_pd = r1^.6 - r2^.4 = (-.13997, .02741, .04602)
    # u_sr = .6 r1/(.4 r2) = (1.16667, 1.71429, 1.6875)
    # u_pr = r1^.6 / r2^.4 = (.86003, 1.03031, 1.04824)
    # ubar = (1 + u)/(1 + max u) per measure, u = mean of the four ubar -> (.84642, .98890, .99753)
    r = wisp(X, W, T)
    assert list(r.extras["u_sd"].values()) == approx([0.06667, 0.22222, 0.24444], abs=5e-6)
    assert list(r.extras["u_pd"].values()) == approx([-0.13997, 0.02741, 0.04602], abs=5e-6)
    assert list(r.extras["u_sr"].values()) == approx([1.16667, 1.71429, 1.6875], abs=5e-6)
    assert list(r.extras["u_pr"].values()) == approx([0.86003, 1.03031, 1.04824], abs=5e-6)
    assert list(r.scores.values()) == approx([0.84642, 0.98890, 0.99753], abs=5e-6)
    assert r.ranking == ("A3", "A2", "A1") and r.method == "wisp"
    assert r.operators == {"normalization": "linear_max", "product": "power"}


CAND_W = [0.11, 0.13, 0.13, 0.15, 0.15, 0.15, 0.16]
CAND = [
    [4, 3, 3, 3, 4, 3, 4],
    [4, 3, 4, 4, 3, 4, 4],
    [5, 5, 4, 3, 3, 4, 3],
    [4, 5, 3, 4, 4, 5, 3],
    [5, 4, 3, 3, 4, 4, 3],
]


def test_wisp_published():
    # M. Stanujkic (2023), J. Process Management and New Technologies 11(3-4), 79-88 (CC BY),
    # Sec. 3: five candidates, seven benefit criteria (Table 1; its printed weights sum to 0.98 and
    # the tables were computed with the weights rescaled to 1, hence normalize_weights), Table 7: weighted sum u^s,
    # recalculated u-bar^s and overall utility u. For benefit criteria only, the four measures
    # collapse to two (u_sd = u_sr = weighted sum, u_pd = u_pr = weighted product), and the mean of
    # four equals Sec. 2.3.1's mean of two. (Table 7 prints A5 as .829; the formula gives .8306.) Table 5 gives the WASPAS weighted product 0.776, ...
    r = wisp(CAND, CAND_W, ["benefit"] * 7, normalize_weights=True)
    assert list(r.extras["u_sd"].values()) == approx([0.792, 0.856, 0.852, 0.904, 0.829], abs=2e-3)
    assert list(r.extras["u_sr"].values()) == approx(list(r.extras["u_sd"].values()))
    assert list(r.extras["ubar_sd"].values()) == approx([0.941, 0.975, 0.973, 1.000, 0.961], abs=8e-4)
    assert list(r.extras["u_pd"].values()) == approx([0.776, 0.843, 0.844, 0.896, 0.823], abs=2e-3)  # A5 .8245
    assert r.ranking == ("A4", "A2", "A3", "A5", "A1")  # Table 7 / Table 8


def test_wisp_product_times_reproduces_table_7():
    # The article's Table 7 product column (2e-7, ..., 5e-7) is prod(r_ij * w_j), not
    # prod(r_ij ^ w_j): u-bar^p = 0.9999997 ... 1.0 and u = 0.9706, 0.9874, 0.9863, 1.0000, 0.9805.
    r = wisp(CAND, CAND_W, ["benefit"] * 7, product="times", normalize_weights=True)
    assert r.operators["product"] == "times"
    assert list(r.extras["u_pd"].values()) == approx([2e-7, 3e-7, 4e-7, 5e-7, 3e-7], abs=6e-8)
    assert list(r.extras["ubar_pd"].values()) == approx([0.9999997, 0.9999998, 0.9999999, 1.0, 0.9999998], abs=2e-7)
    assert list(r.scores.values()) == approx([0.9706, 0.9874, 0.9863, 1.0, 0.9805], abs=4e-4)
    assert r.ranking == ("A4", "A2", "A3", "A5", "A1")
    with pytest.raises(ValueError, match="unknown product 'sum' for wisp; valid: power, times"):
        wisp(X, W, T, product="sum")


def test_wisp_degenerate_inputs():
    with pytest.raises(ValueError, match="positive") as ei:
        wisp([[0, 1], [1, 1]], W, T)
    assert 'score="normalized"' in str(ei.value) and "WISP" in str(ei.value)
    with pytest.raises(ValueError, match="positive"):
        wisp([[-1, 1], [1, 1]], W, T)
    big = wisp([[1.6e308, 1], [0.8e308, 2]], [0.5, 0.5], ["benefit", "cost"])
    small = wisp([[1.6, 1], [0.8, 2]], [0.5, 0.5], ["benefit", "cost"])
    assert list(big.scores.values()) == approx(list(small.scores.values()))
    # r^w can underflow to 0 for the cost product; the ratio must stay finite or raise clearly
    try:
        v = wisp([[1, 1e-300], [1, 1]], [0.1, 0.9], ["cost", "cost"]).scores
        assert all(math.isfinite(s) for s in v.values())
    except ValueError as e:
        assert "WISP" in str(e)


def test_wisp_benefit_only_and_cost_only():
    # documented convention: in a difference an empty group is 0; in a ratio an empty denominator is 1,
    # an empty numerator 0 (sum) or 1 (product)
    b = wisp(X, W, ["benefit", "benefit"])
    # r = (7/9, 1), (8/9, 7/9), (1, 8/9) ; u_sd = u_sr = sum, u_pd = u_pr = product
    sums = [0.6 * 7 / 9 + 0.4, 0.6 * 8 / 9 + 0.4 * 7 / 9, 0.6 + 0.4 * 8 / 9]
    assert list(b.extras["u_sd"].values()) == approx(sums)
    assert list(b.extras["u_sr"].values()) == approx(sums)
    prods = [(7 / 9) ** 0.6, (8 / 9) ** 0.6 * (7 / 9) ** 0.4, (8 / 9) ** 0.4]
    assert list(b.extras["u_pd"].values()) == approx(prods)
    assert list(b.extras["u_pr"].values()) == approx(prods)
    assert b.ranking == ("A3", "A1", "A2")  # sums .8667, .8444, .9556
    # cost only: u_sd = -sum, u_pd = -product, u_sr = 0, u_pr = 1 / product
    c = wisp(X, W, ["cost", "cost"])
    assert list(c.extras["u_sr"].values()) == approx([0.0, 0.0, 0.0])
    cs = [0.6 * 7 / 9 + 0.4 * 1, 0.6 * 8 / 9 + 0.4 * 7 / 9, 0.6 * 1 + 0.4 * 8 / 9]
    assert list(c.extras["u_sd"].values()) == approx([-v for v in cs])
    cp = [(7 / 9) ** 0.6 * 1.0, (8 / 9) ** 0.6 * (7 / 9) ** 0.4, (8 / 9) ** 0.4]
    assert list(c.extras["u_pd"].values()) == approx([-v for v in cp])
    assert list(c.extras["u_pr"].values()) == approx([1 / v for v in cp])
    assert all(math.isfinite(v) for v in c.scores.values())
    assert c.ranking == ("A2", "A1", "A3")  # the smallest weighted cost sum is best


def test_wisp_zero_weight_ignored_and_ties():
    a = wisp(X, [1.0, 0.0], T)
    b = wisp([[7], [8], [9]], [1.0], ["benefit"])
    assert list(a.scores.values()) == approx(list(b.scores.values()))
    r = wisp([[7, 9], [7, 9], [9, 8]], W, T, alternatives=["p", "q", "r"])
    assert r.scores["p"] == r.scores["q"] and r.ranking == ("r", "p", "q")


@pytest.mark.parametrize("types", [["cost", "cost"], ["benefit", "benefit"], ["benefit", "cost"]])
def test_wisp_constant_and_single_alternative(types):
    # regression: all r = 1 on cost-only makes u_sd = u_pd = -1 for everyone, so 1 + max u = 0;
    # such a measure is 0/0 and every alternative is equal, so ubar = 1 (equal scores, input order)
    for rows in ([[1, 1], [1, 1], [1, 1]], [[3, 4], [3, 4]], [[2, 5]]):
        r = wisp(rows, [0.5, 0.5], types)
        assert len(set(r.scores.values())) == 1 and all(math.isfinite(v) for v in r.scores.values())
        assert r.ranking == tuple(f"A{i + 1}" for i in range(len(rows)))


@pytest.mark.parametrize("types", [["cost", "cost"], ["benefit", "benefit"], ["benefit", "cost"]])
def test_marcos_constant_and_single_alternative(types):
    for rows in ([[1, 1], [1, 1], [1, 1]], [[2, 5]]):
        r = marcos(rows, [0.5, 0.5], types)
        assert len(set(r.scores.values())) == 1 and r.scores["A1"] == approx(2 / 3)  # K- = K+ = 1


@pytest.mark.parametrize("rows", [[[1, 1], [1, 1], [1, 1]], [[2, 5]]])
def test_cocoso_constant_and_single_alternative_raise(rows):
    # every r = 0, so min S = 0: k_b divides by zero (documented), the error names the alternative
    with pytest.raises(ValueError, match="alternative 'A1' is worst on every criterion"):
        cocoso(rows, [0.5, 0.5], ["benefit", "cost"])


EXTREME = [[1e300, 1e-300], [1e-300, 1e300], [1, 1]]


@pytest.mark.parametrize("method, name", [(codas, "CODAS"), (marcos, "MARCOS"), (wisp, "WISP")])
@pytest.mark.parametrize(
    "types", [["benefit", "benefit"], ["cost", "benefit"], ["cost", "cost"], ["benefit", "cost"]]
)
def test_extreme_magnitude_spread_raises_named_value_error(method, name, types):
    # _unit_scale underflows 1e-300 / 1e300 to 0: raw ZeroDivisionError / "math domain error"
    with pytest.raises(ValueError, match=rf"{name}: criterion 'C1' spans too many orders"):
        method(EXTREME, [0.5, 0.5], types)


@pytest.mark.parametrize("method", [copras, cocoso])
@pytest.mark.parametrize(
    "types", [["benefit", "benefit"], ["cost", "benefit"], ["cost", "cost"], ["benefit", "cost"]]
)
def test_extreme_magnitude_spread_copras_cocoso_no_raw_error(method, types):
    try:
        method(EXTREME, [0.5, 0.5], types)
    except ValueError:
        pass  # a named, explained ValueError is acceptable


def test_cocoso_rounding_noise_at_column_minimum_is_zero():
    # 0.10000000000000002 is 0.1 up to rounding; its min-max value must be 0,
    # not ~2e-17, whose power r**w (w = 0.25) would be ~7e-5 and break the
    # invariance to positive column scaling.
    x = [[0.1, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [0.10000000000000002, 1.0, 1.0, 2.0]]
    w = [0.25] * 4
    t = ["benefit"] * 4
    a = cocoso(x, w, t)
    b = cocoso([[0.375 * r[0], *r[1:]] for r in x], w, t)
    assert a.extras["P"]["A1"] == a.extras["P"]["A3"]
    for name in a.scores:
        assert a.scores[name] == pytest.approx(b.scores[name], abs=1e-12)
