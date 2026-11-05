import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pytest import approx

from bulanik.mcdm import electre

# shared crisp case (as in test_ranking): C1 benefit, C2 cost; w=(.6,.4)
X, W, T = [[7, 9], [8, 7], [9, 8]], [0.6, 0.4], ["benefit", "cost"]


def test_electre_golden():
    # both columns have norm sqrt(194), so v = w*x/sqrt(194): column differences are .6u*dx1 and .4u*dx2, u = 1/sqrt(194)
    # (1,2): C1 7<8 -> D, C2 (cost) 9>7 -> D;      c = 0,  d = 1 (the worst of D is the worst overall)
    # (2,1): C1 8>=7, C2 7<=9 both concordant;      c = 1,  d = 0 (D empty)
    # (1,3): C1 7<9, C2 9>8 both discordant;        c = 0,  d = 1 ; (3,1): c = 1, d = 0
    # (2,3): C1 8<9 discordant (.6u), C2 7<=8 ok;   c = .4, d = .6u/max(.6u, .4u) = 1
    # (3,2): C1 9>=8 ok, C2 8>7 discordant (.4u);   c = .6, d = .4u/.6u = 2/3
    # means over the 6 ordered pairs: c-bar = (0+0+1+.4+1+.6)/6 = .5 ; d-bar = (1+1+0+1+0+2/3)/6 = 11/18
    # outranks: (2,1) and (3,1) pass both tests; (3,2) has c = .6 >= .5 but d = 2/3 > 11/18 -> no; (2,3) c = .4 < .5 -> no
    # net C = row sum - column sum: A1 -2, A2 (1+.4)-(0+.6) = .8, A3 (1+.6)-(0+.4) = 1.2
    # net D: A1 2, A2 (0+1)-(1+2/3) = -2/3, A3 (0+2/3)-(1+1) = -4/3 ; score = C - D = -4, 1.46667, 2.53333
    r = electre(X, W, T)
    assert r.method == "electre"
    assert list(r.scores.values()) == approx([-4.0, 0.8 + 2 / 3, 1.2 + 4 / 3])
    assert r.ranking == ("A3", "A2", "A1")
    assert r.extras["net_concordance"] == approx({"A1": -2.0, "A2": 0.8, "A3": 1.2})
    assert r.extras["net_discordance"] == approx({"A1": 2.0, "A2": -2 / 3, "A3": -4 / 3})
    c, d = r.extras["concordance"], r.extras["discordance"]
    for table, want in ((c, {"A1": {"A2": 0.0, "A3": 0.0}, "A2": {"A1": 1.0, "A3": 0.4}, "A3": {"A1": 1.0, "A2": 0.6}}),
                        (d, {"A1": {"A2": 1.0, "A3": 1.0}, "A2": {"A1": 0.0, "A3": 1.0}, "A3": {"A1": 0.0, "A2": 2 / 3}})):
        assert all(table[k] == approx(want[k]) for k in want) and set(table) == set(want)
    assert r.extras["outranks"] == {"A1": (), "A2": ("A1",), "A3": ("A1",)}
    assert r.operators == {
        "normalization": "vector",
        "cost": "reverse",
        "score": "net_concordance_minus_net_discordance",
        "c_threshold": "mean",
        "d_threshold": "mean",
    }


# Cakin & Ozdemir (2013), Afyon Kocatepe Univ. IIBF Dergisi 15(2), 339-361: Tablo 6 (net indices) and the ELECTRE
# matrix of EK 3 (12 suppliers, 15 criteria; weights from ANP, EK 2 / Tablo 5). The paper does not print the criterion
# types: the four obvious cost criteria (defect rate, price, delivery time, distance) are taken as cost.
SUPPLIERS = [
    [1, 99, 6, 3.03, 7, 1, 5, 98, 74, 9.33, 7.66, 9.66, 101, 527, 8.33],
    [3, 97, 4, 3.05, 30, 2, 7, 97, 39, 8.66, 7.33, 8.33, 19, 527, 9.00],
    [1, 99, 3, 3.00, 60, 3, 5, 99, 20, 8.00, 9.66, 8.66, 56, 527, 9.66],
    [4, 96, 2, 3.07, 30, 2, 7, 95, 16, 7.33, 7.33, 7.00, 23, 551, 7.00],
    [3, 96, 2, 3.05, 10, 4, 7, 96, 37, 7.66, 8.33, 8.33, 43, 605, 7.33],
    [5, 90, 2, 3.09, 30, 3, 15, 95, 17, 6.33, 5.66, 5.66, 12, 519, 5.33],
    [2, 99, 4, 3.03, 7, 4, 5, 99, 23, 9.00, 8.66, 9.33, 27, 528, 9.33],
    [3, 96, 2, 3.05, 60, 3, 7, 97, 29, 7.33, 8.33, 7.33, 27, 551, 7.33],
    [5, 94, 1, 3.07, 60, 3, 10, 95, 19, 6.00, 6.33, 5.66, 10, 20, 5.33],
    [2, 97, 2, 3.04, 30, 4, 7, 96, 39, 7.66, 8.66, 7.66, 33, 20, 7.66],
    [4, 97, 3, 3.06, 10, 5, 7, 98, 18, 7.00, 7.33, 7.33, 19, 20, 8.00],
    [5, 90, 2, 3.09, 30, 4, 15, 97, 26, 6.33, 7.00, 6.33, 16, 20, 7.00],
]
SUP_W = [0.170, 0.231, 0.032, 0.030, 0.010, 0.005, 0.090, 0.005, 0.051, 0.106, 0.077, 0.081, 0.060, 0.007, 0.045]
SUP_T = ["cost", "benefit", "benefit", "cost", "benefit", "benefit", "cost", "benefit", "benefit", "benefit",
         "benefit", "benefit", "benefit", "cost", "benefit"]
TABLO6_CP = [8.483, 2.479, 7.965, -3.759, 0.820, -9.080, 7.015, -0.382, -8.026, 3.443, -1.363, -7.595]
TABLO6_DP = [-10.025, -1.927, -8.310, 4.978, -1.654, 10.325, -5.276, 0.265, 7.434, -4.696, 2.398, 6.488]
TABLO7 = (1, 3, 7, 10, 2, 5, 8, 11, 4, 12, 9, 6)  # ranking by Cp, printed in Tablo 7


def test_electre_published():
    r = electre(SUPPLIERS, SUP_W, SUP_T, alternatives=[f"S{i}" for i in range(1, 13)])
    cp = list(r.extras["net_concordance"].values())
    dp = list(r.extras["net_discordance"].values())
    # The paper's Cp is reproduced to about 0.2 (largest gap 0.23, supplier 9): the weights are printed to
    # 3 decimals, the paper's cost types are not stated, and its own rounding is unknown.
    assert cp == approx(TABLO6_CP, abs=0.25)
    # net discordance agrees only roughly with cost="reverse": largest gap 1.57 (supplier 12), sum of
    # |gaps| 8.23, rms 0.80; see test_electre_published_inverse for the paper's own treatment
    assert dp == approx(TABLO6_DP, abs=1.6)
    assert sum(cp) == approx(0, abs=1e-9) and sum(dp) == approx(0, abs=1e-9)
    # Tablo 7: ranking by Cp, which the paper's Dp ranking repeats
    by_cp = sorted(range(12), key=lambda i: -cp[i])
    assert tuple(i + 1 for i in by_cp) == TABLO7
    # the best supplier (1) and the three worst (12, 9, 6) are the same with the combined score
    assert r.ranking[0] == "S1" and set(r.ranking[-3:]) == {"S12", "S9", "S6"}
    assert r.ranking[-1] == "S6"


@given(
    x=st.lists(st.lists(st.floats(0.1, 100, allow_nan=False), min_size=3, max_size=3), min_size=2, max_size=5),
    w=st.lists(st.floats(0.1, 1.0), min_size=3, max_size=3),
    t=st.lists(st.sampled_from(["benefit", "cost"]), min_size=3, max_size=3),
)
@settings(max_examples=80, deadline=None)
def test_concordance_complement_property(x, w, t):
    # ties count for both directions, so c_kl + c_lk >= 1 (weights normalised to sum 1)
    r = electre(x, w, t, normalize_weights=True)
    c = r.extras["concordance"]
    for k in c:
        for l in c[k]:
            assert c[k][l] + c[l][k] >= 1 - 1e-12
            assert 0 <= r.extras["discordance"][k][l] <= 1
    assert sum(r.extras["net_concordance"].values()) == approx(0, abs=1e-9)
    assert sum(r.extras["net_discordance"].values()) == approx(0, abs=1e-9)


def test_electre_ties_and_identical_alternatives():
    # identical rows: all columns concordant both ways, no discordance (0/0 -> 0), equal scores, input order kept
    r = electre([[7, 9], [7, 9], [9, 8]], W, T, alternatives=["p", "q", "r"])
    assert r.extras["concordance"]["p"]["q"] == approx(1.0) and r.extras["concordance"]["q"]["p"] == approx(1.0)
    assert r.extras["discordance"]["p"]["q"] == 0.0 and r.extras["discordance"]["q"]["p"] == 0.0
    assert r.scores["p"] == r.scores["q"]
    assert r.ranking == ("r", "p", "q")
    # all alternatives identical: every pair concordant and indifferent, mean thresholds 1 and 0 pass them all
    s = electre([[3, 4], [3, 4], [3, 4]], W, T)
    assert list(s.scores.values()) == [0.0, 0.0, 0.0] and s.ranking == ("A1", "A2", "A3")
    assert s.extras["outranks"] == {"A1": ("A2", "A3"), "A2": ("A1", "A3"), "A3": ("A1", "A2")}
    # a zero column contributes 0 to v, so it is concordant (tie) and never discordant
    z = electre([[0, 9], [0, 7], [0, 8]], W, T)
    assert z.extras["discordance"]["A2"]["A1"] == 0.0


def test_electre_constant_column_is_neutral():
    a = electre([[7, 5], [8, 5], [9, 5]], [0.6, 0.4], ["benefit", "benefit"])
    b = electre([[7], [8], [9]], [1.0], ["benefit"])
    # the constant column adds .4 to every concordance but changes no discordance
    assert a.ranking == b.ranking == ("A3", "A2", "A1")
    assert a.extras["discordance"] == b.extras["discordance"]


def test_electre_zero_weight_ignored():
    a = electre(X, [1.0, 0.0], T)
    b = electre([[7], [8], [9]], [1.0], ["benefit"])
    assert list(a.scores.values()) == approx(list(b.scores.values()))


def test_thresholds_and_errors():
    # default means: c-bar = .5, d-bar = 11/18 (see the golden). Explicit thresholds change the outranking relation.
    r = electre(X, W, T, c_threshold=0.0, d_threshold=1.0)  # everything outranks everything
    assert all(len(v) == 2 for v in r.extras["outranks"].values())
    assert r.operators["c_threshold"] == "0.0" and r.operators["d_threshold"] == "1.0"
    # c >= .6 and d <= .7 admits (3,2) too
    s = electre(X, W, T, c_threshold=0.6, d_threshold=0.7)
    assert s.extras["outranks"] == {"A1": (), "A2": ("A1",), "A3": ("A1", "A2")}
    # the scores do not depend on the thresholds
    assert s.scores == r.scores
    # a pair sitting exactly on the boundary counts as outranking (>=, <=), despite float sums
    t = electre(X, W, T, c_threshold=0.6, d_threshold=2 / 3)
    assert t.extras["outranks"]["A3"] == ("A1", "A2")
    for bad in (1.5, -0.1):
        with pytest.raises(ValueError, match="c_threshold must be between 0 and 1"):
            electre(X, W, T, c_threshold=bad)
        with pytest.raises(ValueError, match="d_threshold must be between 0 and 1"):
            electre(X, W, T, d_threshold=bad)
    with pytest.raises(ValueError, match="finite"):
        electre(X, W, T, c_threshold=float("nan"))
    with pytest.raises(TypeError):
        electre(X, W, T, d_threshold=True)
    with pytest.raises(ValueError, match="at least two alternatives"):
        electre([[1, 2]], W, T)
    # shared validation
    with pytest.raises(ValueError, match="valid: benefit, cost"):
        electre(X, W, ["benefit", "profit"])
    with pytest.raises(ValueError, match="sum to 1"):
        electre(X, [0.6, 0.6], T)
    with pytest.raises(ValueError, match="finite"):
        electre([[1, float("inf")], [2, 3]], W, T)
    with pytest.raises(TypeError):
        electre([[1, True], [2, 3]], W, T)


def test_electre_weight_forms():
    base = electre(X, W, T)
    assert electre(X, {"C1": 0.6, "C2": 0.4}, T).scores == base.scores
    assert electre(X, [3, 2], T, alternatives=["x", "y", "z"], criteria=["a", "b"],
                   normalize_weights=True).ranking == ("z", "y", "x")


def test_electre_overflow_is_not_silent():
    # entries near the float limit: vector norm via hypot does not overflow to inf
    big = 1e300
    r = electre([[big, 2 * big], [2 * big, big]], [0.5, 0.5], ["benefit", "benefit"])
    assert all(math.isfinite(s) for s in r.scores.values())


def test_cost_option():
    # inverse: the cost column is replaced by 1/x before the vector norm; concordance is unchanged
    # C2 cost, x = (9, 7, 8) -> 1/x = (.11111, .14286, .125), norm = sqrt(.012346 + .020408 + .015625) = .219951
    # v2 = .4/x/norm = (.20207, .25979, .22737): pair (1,2): C1 .6u*1 = .04307 (u = 1/sqrt(194)), C2 |.20207-.25979| = .05772
    # both discordant, d = 1 as before; pair (3,2): C2 discordant only, gap .03242 vs C1 gap .04307 -> d = .03242/.04307
    rev = electre(X, W, T)
    inv = electre(X, W, T, cost="inverse")
    assert inv.operators["cost"] == "inverse" and rev.operators["cost"] == "reverse"
    assert inv.extras["concordance"] == rev.extras["concordance"]
    u = 1 / math.sqrt(194)
    n2 = math.sqrt((1 / 9) ** 2 + (1 / 7) ** 2 + (1 / 8) ** 2)
    v2 = [0.4 * (1 / x) / n2 for x in (9, 7, 8)]
    assert inv.extras["discordance"]["A3"]["A2"] == approx(abs(v2[2] - v2[1]) / (0.6 * u))
    assert inv.extras["discordance"]["A1"]["A2"] == approx(1.0)
    assert inv.extras["discordance"] != rev.extras["discordance"]
    # without a cost criterion the option changes nothing
    a = electre([[7, 5], [8, 6], [9, 4]], W, ["benefit", "benefit"])
    b = electre([[7, 5], [8, 6], [9, 4]], W, ["benefit", "benefit"], cost="inverse")
    assert a.scores == b.scores
    with pytest.raises(ValueError, match=r"unknown cost handling 'flip' for electre; valid: reverse, inverse"):
        electre(X, W, T, cost="flip")
    for bad in (0, -1):
        with pytest.raises(ValueError, match="strictly positive") as ei:
            electre([[7, bad], [8, 7], [9, 8]], W, T, cost="inverse")
        assert "ELECTRE" in str(ei.value) and "criterion 2" in str(ei.value)
    # reverse accepts zeros and negatives in a cost column; a nonpositive benefit column is fine for inverse
    electre([[7, 0], [8, 7], [9, 8]], W, T)
    electre([[0, 9], [8, 7], [9, 8]], W, T, cost="inverse")


def test_electre_published_inverse():
    r = electre(SUPPLIERS, SUP_W, SUP_T, alternatives=[f"S{i}" for i in range(1, 13)], cost="inverse")
    dp = list(r.extras["net_discordance"].values())
    # cost="inverse" is the paper's treatment (1/x before the vector norm) but does NOT reproduce Tablo 6
    # much more closely than "reverse": largest gap 1.50 (supplier 9; supplier 2 is off by 1.02), sum of
    # |gaps| 5.31 (reverse: 8.23), rms 0.60 (0.80). Net concordance is the same as for "reverse". The
    # residual is not explained by the 3-dp weights alone, so the published test stays on the default.
    assert dp == approx(TABLO6_DP, abs=1.51)
    assert sum(abs(a - b) for a, b in zip(dp, TABLO6_DP)) < 5.4
    assert r.ranking[0] == "S1" and r.ranking[-1] == "S6"


def test_ties_are_relative_and_scale_invariant():
    # regression: a 1-ulp difference must not decide concordance, or scaling a column flips it by w_j
    x = [[0.10000000000000002, 0.1], [0.1, 0.1], [0.125, 1.0]]
    scaled = [[a * 0.1015625, b] for a, b in x]
    a = electre(x, [0.5, 0.5], ["benefit", "benefit"])
    b = electre(scaled, [0.5, 0.5], ["benefit", "benefit"])
    assert list(b.scores.values()) == approx(list(a.scores.values()), rel=1e-9, abs=1e-12)
    # the first two rows tie on both criteria: concordant both ways, no discordance
    assert a.extras["concordance"]["A1"]["A2"] == approx(1.0) and a.extras["concordance"]["A2"]["A1"] == approx(1.0)
    assert a.extras["discordance"]["A1"]["A2"] == 0.0 and a.extras["discordance"]["A2"]["A1"] == 0.0
    # cost criteria too
    c = electre(x, [0.5, 0.5], ["cost", "cost"])
    assert c.extras["concordance"]["A1"]["A2"] == approx(1.0) and c.extras["concordance"]["A2"]["A1"] == approx(1.0)
    # a difference above the tolerance still counts
    z = electre([[1.0 + 1e-9, 1.0], [1.0, 1.0]], [0.5, 0.5], ["benefit", "benefit"])
    assert z.extras["concordance"]["A2"]["A1"] == approx(0.5)
