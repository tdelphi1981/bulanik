import itertools

import pytest
from pytest import approx

from bulanik import IFN, IT2TrFN, IVIFN, PFN, TFN, NeutrosophicFN, PictureFN, TrFN
from bulanik.aggregate import aggregate_experts
from bulanik.mcdm import (
    DecisionMatrix,
    cocoso,
    fuzzy_cocoso,
    fuzzy_codas,
    fuzzy_waspas,
    rank_fuzzy,
)
from bulanik.mcdm.scoring import _cocoso_appraisal
from bulanik.scales import EXAMPLE_IFN_5

# The 2x2 example used by the golden tests, weights (0.6, 0.4), both benefit.
X = [[IFN(0.6, 0.3), IFN(0.4, 0.5)], [IFN(0.5, 0.4), IFN(0.7, 0.2)]]
W = [0.6, 0.4]
BB = ["benefit", "benefit"]
# Algebraic IFN laws (Xu 2007): k*a = (1-(1-mu)^k, nu^k), a^k = (mu^k, 1-(1-nu)^k),
# a+b = (mu1+mu2-mu1 mu2, nu1 nu2), a*b = (mu1 mu2, nu1+nu2-nu1 nu2).


def _pairs(xs):
    return [(round(x.mu, 12), round(x.nu, 12)) for x in xs]


def test_fuzzy_codas_golden_2x2():
    # u11 = 0.6*(0.6,0.3) = (1-0.4^0.6, 0.3^0.6) = (0.422920, 0.485593), score -0.062673
    # u12 = 0.4*(0.4,0.5) = (1-0.6^0.4, 0.5^0.4) = (0.184807, 0.757858), score -0.573051
    # u21 = 0.6*(0.5,0.4) = (0.340246, 0.577080), score -0.236834
    # u22 = 0.4*(0.7,0.2) = (0.382199, 0.525306), score -0.143106
    # ns = (u21, u12): the lower score in each column
    # A1: d_E(u11,u21) = sqrt((.082674^2+.091486^2+.008812^2)/2) = 0.087414, d_E(u12,u12) = 0
    #     d_H = (.082674+.091486+.008812)/2 = 0.091487 -> E1 = 0.087414, T1 = 0.091487
    # A2: d_E(u22,u12) = 0.217118, d_H = 0.232553 -> E2 = 0.217118, T2 = 0.232553
    # |E1-E2| = 0.129704 >= tau = 0.02, so h12 = (E1-E2)+(T1-T2) = -0.270770 = -H2
    r = fuzzy_codas(X, W, BB)
    assert r.method == "fuzzy_codas"
    assert list(r.scores.values()) == approx([-0.27077036494151485, 0.27077036494151485], abs=1e-9)
    assert r.ranking == ("A2", "A1")
    assert list(r.extras["E"].values()) == approx([0.08741409109181743, 0.2171183211915684], abs=1e-9)
    assert list(r.extras["T"].values()) == approx([0.09148658753268168, 0.23255272237444558], abs=1e-9)
    ns = r.extras["negative_ideal"]
    assert list(ns) == ["C1", "C2"]
    assert _pairs(ns.values()) == _pairs([IFN(0.5, 0.4).scale(0.6), IFN(0.4, 0.5).scale(0.4)])
    assert r.operators == {"ops": "algebraic", "score": "difference", "tau": "0.02", "psi": "equation"}


def test_fuzzy_waspas_golden_2x2():
    # Q1_1 = 0.6*(0.6,0.3) + 0.4*(0.4,0.5) = (0.529568, 0.368011)
    # Q2_1 = (0.6,0.3)^0.6 * (0.4,0.5)^0.4 = (0.6^0.6 0.4^0.4, 1-0.7^0.6 0.5^0.4) = (0.510170, 0.388147)
    # Q_1 = 0.5*Q1_1 + 0.5*Q2_1 = (1-sqrt(.470432*.489830), sqrt(.368011*.388147)) = (0.519967, 0.377945)
    # S(Q_1) = 0.142022 (difference score); A2 likewise: Q_2 = (0.582342, 0.314762), S = 0.267580
    r = fuzzy_waspas(X, W, BB)
    assert r.method == "fuzzy_waspas"
    assert list(r.scores.values()) == approx([0.14202198699019863, 0.26758004280237757], abs=1e-9)
    assert r.ranking == ("A2", "A1")
    q1 = r.extras["Q1"]["A1"]
    assert (q1.mu, q1.nu) == approx((0.5295683909901305, 0.36801096140891665), abs=1e-9)
    q2 = r.extras["Q2"]["A1"]
    assert (q2.mu, q2.nu) == approx((0.5101698002503163, 0.3881473776277705), abs=1e-9)
    assert list(r.extras["Q1_score"].values()) == approx([0.16155742958121388, 0.2892601318949589], abs=1e-9)
    assert list(r.extras["Q2_score"].values()) == approx([0.12202242262254581, 0.24520626519708116], abs=1e-9)
    assert r.extras["Q"]["A2"].mu == approx(0.5823424821148742, abs=1e-9)
    assert r.operators == {"ops": "algebraic", "score": "difference", "lambda": "0.5", "combine": "fuzzy"}


def test_fuzzy_waspas_combine_scores():
    # combine="scores": Q = 0.5 S(Q1) + 0.5 S(Q2) with the Q1/Q2 scores of the golden test,
    # A1: (0.161557 + 0.122022)/2 = 0.141790, A2: (0.289260 + 0.245206)/2 = 0.267233
    r = fuzzy_waspas(X, W, BB, combine="scores")
    assert list(r.scores.values()) == approx([0.14178992610187985, 0.26723319854602003], abs=1e-9)
    assert r.operators["combine"] == "scores" and "Q" not in r.extras
    assert set(r.extras) == {"Q1", "Q2", "Q1_score", "Q2_score"}
    lam = 0.3
    t = fuzzy_waspas(X, W, BB, combine="scores", lam=lam)
    q1, q2 = t.extras["Q1_score"], t.extras["Q2_score"]
    assert t.scores == approx({k: lam * q1[k] + (1 - lam) * q2[k] for k in q1}, abs=1e-12)
    with pytest.raises(ValueError, match="unknown combine 'x' for fuzzy_waspas; valid: fuzzy, scores"):
        fuzzy_waspas(X, W, BB, combine="x")


def test_fuzzy_waspas_fuzzy_combine_is_scale_and_add():
    # Q = lam Q1 (+) (1 - lam) Q2, then defuzzify
    m = [[IVIFN(0.5, 0.6, 0.2, 0.3), IVIFN(0.3, 0.4, 0.4, 0.5)],
         [IVIFN(0.4, 0.5, 0.3, 0.4), IVIFN(0.6, 0.7, 0.1, 0.2)]]
    for lam in (0.2, 0.5, 0.8):
        r = fuzzy_waspas(m, W, BB, lam=lam, score="defuzzified")
        for name in ("A1", "A2"):
            q1, q2 = r.extras["Q1"][name], r.extras["Q2"][name]
            q = q1.scale(lam).add(q2.scale(1 - lam))
            assert r.scores[name] == approx(q.score("defuzzified"), abs=1e-12)
        assert r.operators["score"] == "defuzzified"


def test_fuzzy_waspas_lambda_ends_pick_one_model():
    a = fuzzy_waspas(X, W, BB, lam=1.0)
    b = fuzzy_waspas(X, W, BB, lam=0.0)
    assert a.scores == approx(a.extras["Q1_score"])
    assert b.scores == approx(b.extras["Q2_score"])


def test_fuzzy_cocoso_golden_2x2():
    # S_i = Q1_i and P_i = Q2_i of the WASPAS test; normalized scores (1+mu-nu)/2:
    # S = (0.580779, 0.644630), P = (0.561011, 0.622603)
    # k_a = (S+P)/sum = (0.473964, 0.526036); k_b = S/min S + P/min P = (2, 2.219728)
    # k_c = (S+P)/(max S+max P) = (0.901010, 1); k = (k_a k_b k_c)^(1/3) + (k_a+k_b+k_c)/3
    r = fuzzy_cocoso(X, W, BB)
    assert r.method == "fuzzy_cocoso"
    assert list(r.scores.values()) == approx([2.0737773973089326, 2.301612592601466], abs=1e-9)
    assert r.ranking == ("A2", "A1")
    assert list(r.extras["S_score"].values()) == approx([0.5807787147906069, 0.6446300659474795], abs=1e-9)
    assert list(r.extras["P_score"].values()) == approx([0.5610112113112729, 0.6226031325985406], abs=1e-9)
    assert list(r.extras["k_b"].values()) == approx([2.0, 2.2197282554645925], abs=1e-9)
    s = r.extras["S"]["A1"]
    assert (s.mu, s.nu) == approx((0.5295683909901305, 0.36801096140891665), abs=1e-9)
    assert r.extras["P"]["A2"].mu == approx(0.5720331779293615, abs=1e-9)
    assert r.operators == {"ops": "algebraic", "score": "normalized", "lambda": "0.5"}


def test_fuzzy_cocoso_is_wa_wg_normalized_then_crisp_appraisal():
    # S = WA(row), P = WG(row) (a weighted
    # geometric product, not a sum of powers), no normalisation, both scored "normalized",
    # then crisp k_a, k_b, k_c, k with lambda.
    from bulanik.aggregate import wa, wg

    for lam in (0.3, 0.5, 0.9):
        r = fuzzy_cocoso(X, W, BB, lam=lam)
        s = [wa(row, W).score("normalized") for row in X]
        p = [wg(row, W).score("normalized") for row in X]
        assert list(r.scores.values()) == approx(_cocoso_appraisal(s, p, lam), abs=1e-12)
        sum_of_powers = X[0][0].power(0.6).add(X[0][1].power(0.4))
        assert r.extras["P"]["A1"].mu != approx(sum_of_powers.mu, abs=1e-6)


def test_cocoso_appraisal_matches_crisp_cocoso():
    m = [[7, 8], [8, 7], [9, 9]]
    r = cocoso(m, [0.6, 0.4], ["benefit", "cost"], alternatives=["A", "B", "C"])
    s, p = list(r.extras["S"].values()), list(r.extras["P"].values())
    assert _cocoso_appraisal(s, p, 0.5) == approx(list(r.scores.values()), abs=1e-12)
    r7 = cocoso(m, [0.6, 0.4], ["benefit", "cost"], lam=0.7)
    s7, p7 = list(r7.extras["S"].values()), list(r7.extras["P"].values())
    assert _cocoso_appraisal(s7, p7, 0.7) == approx(list(r7.scores.values()), abs=1e-12)


@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
def test_fuzzy_codas_matches_crisp_when_numbers_are_crisp_like(perm):
    # nu = 1 - mu: no hesitancy, so the numbers behave like crisp grades in [0, 1].
    # The ranking must not depend on the order in which the alternatives are listed.
    rows = [[IFN(0.8, 0.2), IFN(0.3, 0.7)], [IFN(0.5, 0.5), IFN(0.6, 0.4)], [IFN(0.2, 0.8), IFN(0.9, 0.1)]]
    names = ["P", "Q", "R"]
    base = fuzzy_codas(rows, [0.5, 0.5], BB, alternatives=names)
    r = fuzzy_codas([rows[i] for i in perm], [0.5, 0.5], BB, alternatives=[names[i] for i in perm])
    assert r.ranking == base.ranking
    assert r.scores == approx(base.scores, abs=1e-12)
    for i in range(3):
        for j in range(2):
            assert rows[i][j].hesitancy == approx(0.0, abs=1e-12)


@pytest.mark.parametrize("fn", [fuzzy_codas, fuzzy_waspas, fuzzy_cocoso])
def test_cost_column_uses_complement(fn):
    cost = fn(X, W, ["benefit", "cost"])
    flipped = [[r[0], r[1].complement()] for r in X]
    benefit = fn(flipped, W, BB)
    assert cost.scores == approx(benefit.scores, abs=1e-12)
    assert cost.ranking == benefit.ranking


def test_cost_column_neutrosophic_complement():
    m = [[NeutrosophicFN(0.7, 0.2, 0.1), NeutrosophicFN(0.4, 0.3, 0.5)],
         [NeutrosophicFN(0.5, 0.4, 0.3), NeutrosophicFN(0.6, 0.1, 0.2)]]
    cost = fuzzy_cocoso(m, W, ["cost", "benefit"])
    benefit = fuzzy_cocoso([[r[0].complement(), r[1]] for r in m], W, BB)
    assert cost.scores == approx(benefit.scores, abs=1e-12)
    assert cost.operators["score"] == "average"  # no "normalized": the kind's default


@pytest.mark.parametrize("fn", [fuzzy_waspas, fuzzy_cocoso])
def test_cost_with_tfn_raises_typeerror(fn):
    m = [[TFN(1, 2, 3), TFN(2, 3, 4)], [TFN(2, 3, 4), TFN(1, 2, 3)]]
    with pytest.raises(TypeError, match=r"TFN has no complement.*rank_fuzzy"):
        fn(m, W, ["benefit", "cost"])


def test_tfn_benefit_works_where_distances_are_not_needed():
    m = [[TFN(1, 2, 3), TFN(2, 3, 4)], [TFN(2, 3, 4), TFN(3, 4, 5)]]
    assert fuzzy_waspas(m, W, BB).ranking == ("A2", "A1")
    r = fuzzy_cocoso(m, W, BB)
    assert r.ranking == ("A2", "A1") and r.operators["score"] == "centroid"
    assert fuzzy_codas(m, W, BB).ranking == ("A2", "A1")


@pytest.mark.parametrize("fn", [fuzzy_codas, fuzzy_waspas, fuzzy_cocoso])
def test_mixed_kinds_raise(fn):
    with pytest.raises(TypeError, match=r"IFN\(q=1\).*PFN\(q=2\)"):
        fn([[IFN(0.6, 0.3), PFN(0.6, 0.3)], [IFN(0.5, 0.4), IFN(0.7, 0.2)]], W, BB)
    with pytest.raises(TypeError, match=r"IVIFN.*IFN"):
        fn([[IVIFN(0.5, 0.6, 0.2, 0.3), IFN(0.6, 0.3)]], W, BB)


@pytest.mark.parametrize("fn", [fuzzy_codas, fuzzy_waspas, fuzzy_cocoso])
def test_matrix_shape_and_entry_type(fn):
    with pytest.raises(ValueError, match="at least one row"):
        fn([], W, BB)
    with pytest.raises(ValueError, match="same length"):
        fn([[IFN(0.6, 0.3), IFN(0.4, 0.5)], [IFN(0.5, 0.4)]], W, BB)
    with pytest.raises(TypeError, match="fuzzy number entries, got float"):
        fn([[0.5, 0.4], [0.3, 0.2]], W, BB)
    with pytest.raises(ValueError, match="expected 2 alternatives"):
        fn(X, W, BB, alternatives=["a"])
    with pytest.raises(ValueError, match="sum to 1"):
        fn(X, [0.6, 0.6], BB)
    assert fn(X, [3, 2], BB, normalize_weights=True).scores == approx(fn(X, W, BB).scores)
    named = fn(X, {"price": 0.6, "quality": 0.4}, BB)
    assert named.scores == approx(fn(X, W, BB).scores)


def test_cocoso_zero_sequence_guard():
    # A2 is (0, 1) everywhere: S_2 = P_2 = (0, 1), whose normalized score is 0,
    # so min S = 0 and k_b would divide by zero; crisp cocoso's guard and message apply.
    m = [[IFN(0.6, 0.3), IFN(0.4, 0.5)], [IFN(0.0, 1.0), IFN(0.0, 1.0)]]
    msg = r"CoCoSo: alternative 'A2' is worst on every criterion, so k_b divides by zero"
    with pytest.raises(ValueError, match=r"CoCoSo: alternative 'A2'.*S or P is 0.*k_b divides by zero"):
        fuzzy_cocoso(m, W, BB)
    # the same guard reached directly through the shared helper with these scores
    s_scores = [IFN(0.6, 0.3).scale(0.6).add(IFN(0.4, 0.5).scale(0.4)).score("normalized"), 0.0]
    with pytest.raises(ValueError, match=msg):
        _cocoso_appraisal(s_scores, [0.56, 0.0], 0.5, ["A1", "A2"])


def test_cocoso_negative_score_raises_with_hint():
    with pytest.raises(ValueError, match=r'non-negative.*score="normalized"'):
        fuzzy_cocoso([[IFN(0.2, 0.7)], [IFN(0.6, 0.3)]], [1.0], ["benefit"], score="difference")


def test_waspas_lam_bounds():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="lam must be between 0 and 1"):
            fuzzy_waspas(X, W, BB, lam=bad)
        with pytest.raises(ValueError, match="lam must be between 0 and 1"):
            fuzzy_cocoso(X, W, BB, lam=bad)
    with pytest.raises(TypeError):
        fuzzy_waspas(X, W, BB, lam=True)


def test_codas_tau_and_psi_validation():
    with pytest.raises(ValueError, match="tau must be between 0 and 1"):
        fuzzy_codas(X, W, BB, tau=2)
    with pytest.raises(ValueError, match="unknown psi 'x' for fuzzy_codas; valid: equation, tables"):
        fuzzy_codas(X, W, BB, psi="x")
    # with tau above |E1 - E2| the taxicab term drops out under "equation"
    r = fuzzy_codas(X, W, BB, tau=0.5)
    assert list(r.scores.values()) == approx([-0.12970423009975097, 0.12970423009975097], abs=1e-9)
    assert fuzzy_codas(X, W, BB, tau=0.5, psi="tables").scores == approx(fuzzy_codas(X, W, BB).scores)


@pytest.mark.parametrize("fn", [fuzzy_codas, fuzzy_waspas, fuzzy_cocoso])
def test_unknown_ops_and_score_names(fn):
    with pytest.raises(ValueError, match="unknown ops 'hamacher' for IFN; valid: algebraic, einstein"):
        fn(X, W, BB, ops="hamacher")
    with pytest.raises(ValueError, match="unknown score 'foo' for IFN; valid: difference, normalized"):
        fn(X, W, BB, score="foo")


@pytest.mark.parametrize("fn", [fuzzy_codas, fuzzy_waspas, fuzzy_cocoso])
def test_einstein_ops_recorded_and_used(fn):
    a = fn(X, W, BB, ops="einstein")
    b = fn(X, W, BB)
    assert a.operators["ops"] == "einstein"
    assert a.scores != approx(b.scores, abs=1e-6)


def test_interval_valued_kind():
    m = [[IVIFN(0.5, 0.6, 0.2, 0.3), IVIFN(0.3, 0.4, 0.4, 0.5)],
         [IVIFN(0.4, 0.5, 0.3, 0.4), IVIFN(0.6, 0.7, 0.1, 0.2)]]
    # Golden, fuzzy_cocoso with C2 a cost criterion. The IVIF laws act on the lower and
    # upper bounds: complement (a, b, c, d) -> (c, d, a, b); k*x lower/upper mu 1-(1-mu)^k,
    # nu^k; x^k mu^k, 1-(1-nu)^k; normalized score (1 + (mu_l+mu_u-nu_l-nu_u)/2)/2.
    # A1: S = 0.6*(.5,.6,.2,.3) + 0.4*(.4,.5,.3,.4) = (.462173, .562655, .235216, .336587) -> 0.613256
    #     P = (.5,.6,.2,.3)^0.6 * (.4,.5,.3,.4)^0.4 = (.457305, .557800, .241609, .341858) -> 0.607910
    # A2: S = 0.6*(.4,.5,.3,.4) + 0.4*(.1,.2,.6,.7) -> 0.448683; P -> 0.397658
    # k_b = S/min S + P/min P = (2.895516, 2); k = (2.691259, 1.862016)
    r = fuzzy_cocoso(m, W, ["benefit", "cost"])
    assert list(r.scores.values()) == approx([2.6912590527698033, 1.8620159066692104], abs=1e-9)
    assert list(r.extras["S_score"].values()) == approx([0.6132564859544023, 0.44868276854266065], abs=1e-9)
    assert list(r.extras["P_score"].values()) == approx([0.6079095001479666, 0.3976584132554265], abs=1e-9)
    assert r.ranking == ("A1", "A2")
    # every method: the cost column equals the complemented benefit column
    flipped = [[row[0], row[1].complement()] for row in m]
    for fn in (fuzzy_codas, fuzzy_waspas, fuzzy_cocoso):
        cost, benefit = fn(m, W, ["benefit", "cost"]), fn(flipped, W, BB)
        assert cost.scores == approx(benefit.scores, abs=1e-12)
        assert cost.ranking == benefit.ranking


DM_IFN = DecisionMatrix.from_terms(
    EXAMPLE_IFN_5,
    ["a1", "a2", "a3"],
    ["c1", "c2"],
    [
        [["H", "L"], ["M", "M"], ["VH", "VL"]],
        [["M", "L"], ["H", "M"], ["H", "L"]],
    ],
)


@pytest.mark.parametrize(
    ("method", "fn", "kwargs"),
    [("fuzzy_codas", fuzzy_codas, {"tau": 0.05}), ("fuzzy_waspas", fuzzy_waspas, {"lam": 0.3}),
     ("fuzzy_cocoso", fuzzy_cocoso, {})],
)
@pytest.mark.parametrize("aggregation", ["wa", "wg"])
def test_rank_fuzzy_fuzzy_methods_skip_early_scoring(method, fn, kwargs, aggregation):
    w = {"c1": 0.7, "c2": 0.3}
    types = ["benefit", "cost"]
    r = rank_fuzzy(DM_IFN, w, types, method=method, aggregation=aggregation, **kwargs)
    agg = aggregate_experts(DM_IFN.experts, method=aggregation)
    direct = fn(agg, w, types, alternatives=list(DM_IFN.alternatives), **kwargs)
    assert r.method == method
    assert r.scores == approx(direct.scores, abs=1e-12) and r.ranking == direct.ranking
    assert r.operators == direct.operators | {"aggregation": aggregation}
    if method == "fuzzy_waspas":
        s = rank_fuzzy(DM_IFN, w, types, method=method, aggregation=aggregation, combine="scores")
        assert s.operators["combine"] == "scores"


def test_rank_fuzzy_fuzzy_codas_skips_early_scoring():
    w = [0.7, 0.3]
    r = rank_fuzzy(DM_IFN, w, ["benefit", "benefit"], method="fuzzy_codas", score="normalized")
    direct = fuzzy_codas(aggregate_experts(DM_IFN.experts), w, ["benefit", "benefit"],
                         alternatives=["a1", "a2", "a3"], criteria=["c1", "c2"], score="normalized")
    assert r.scores == approx(direct.scores, abs=1e-12)
    assert r.extras["E"] == approx(direct.extras["E"], abs=1e-12)
    assert r.operators["score"] == "normalized" and r.operators["aggregation"] == "wa"
    with pytest.raises(ValueError, match="unknown score 'foo' for IFN"):
        rank_fuzzy(DM_IFN, w, ["benefit", "benefit"], method="fuzzy_cocoso", score="foo")
    with pytest.raises(ValueError, match="fuzzy_codas"):
        rank_fuzzy(DM_IFN, w, ["benefit", "benefit"], method="nope")


def test_fuzzy_codas_picture_weight_error_names_method_and_alternatives():
    m = [[PictureFN(0.5, 0.2, 0.2), PictureFN(0.4, 0.3, 0.2)],
         [PictureFN(0.6, 0.1, 0.2), PictureFN(0.5, 0.2, 0.2)]]
    msg = (r"fuzzy_codas: PictureFN cannot be weighted by 0\.5 for criterion 'price'.*"
           r"fuzzy_waspas.*fuzzy_cocoso.*method=\"codas\", score=\"normalized\"")
    with pytest.raises(ValueError, match=msg):
        fuzzy_codas(m, [0.5, 0.5], BB, criteria=["price", "quality"])


def test_fuzzy_codas_it2_distance_error_hints_rank_fuzzy():
    a = IT2TrFN((1, 2, 3, 4, 1, 1), (1.5, 2, 3, 3.5, 0.8, 0.8))
    b = IT2TrFN((2, 3, 4, 5, 1, 1), (2.5, 3, 4, 4.5, 0.8, 0.8))
    with pytest.raises(TypeError, match=r"euclidean.*hamming.*IT2TrFN has: vertex.*rank_fuzzy"):
        fuzzy_codas([[a, b], [b, a]], W, BB)


def test_fuzzy_cocoso_zero_p_message_names_the_product():
    m = [[IFN(0.0, 1.0), IFN(0.6, 0.3)], [IFN(0.6, 0.3), IFN(0.4, 0.5)]]
    with pytest.raises(ValueError, match=r"alternative 'A1'.*S or P is 0.*\(0, 1\).*k_b divides by zero") as ei:
        fuzzy_cocoso(m, W, BB)
    assert "worst on every criterion" not in str(ei.value)


# --- fuzzy CODAS on triangular and trapezoidal numbers --------------------------------
# Keshavarz Ghorabaee, Amiri, Zavadskas, Hooshmand & Antucheviciene (2017), Journal of
# Business Economics and Management 18(1), 1-19: normalise by the largest defuzzified
# value of the column, n = x / max D (benefit) and n = 1 - x / max D (cost), weight,
# take the rating with the lowest D as negative ideal, ED_i = sum_j d_E, HD_i = sum_j d_H
# with Li's (2007) weighted distances, then the crisp CODAS assessment.
# The example matrices below are made up for these tests.

TFN_X = [
    [TFN(2, 3, 4), TFN(1, 2, 3)],
    [TFN(4, 5, 6), TFN(3, 4, 6)],
    [TFN(1, 2, 4), TFN(2, 3, 4)],
]
BC = ["benefit", "cost"]


def _reference_codas(points, w, types, tau=0.02, point_weights=None):
    """Independent numpy implementation of fuzzy CODAS (2017) on (m, n, k) point arrays."""
    np = pytest.importorskip("numpy")
    x = np.asarray(points, dtype=float)
    k = x.shape[2]
    pw = np.asarray(point_weights or ((1, 4, 1) if k == 3 else (1, 2, 2, 1)), dtype=float) / 6
    if k == 3:
        d = x.mean(axis=2)  # centroid of a triangle
    else:
        a, b, c, dd = (x[..., i] for i in range(4))
        d = (a + b + c + dd - (c * dd - a * b) / ((c + dd) - (a + b))) / 3  # Eq. (11)
    top = d.max(axis=0)
    nrm = x / top[None, :, None]
    for j, t in enumerate(types):
        if t == "cost":
            nrm[:, j, :] = 1 - nrm[:, j, ::-1]
    r = nrm * np.asarray(w)[None, :, None]
    if k == 3:
        dr = r.mean(axis=2)
    else:
        a, b, c, dd = (r[..., i] for i in range(4))
        dr = (a + b + c + dd - (c * dd - a * b) / ((c + dd) - (a + b))) / 3
    ns = r[dr.argmin(axis=0), np.arange(r.shape[1])]
    diff = r - ns[None]
    e = np.sqrt((pw * diff**2).sum(axis=2)).sum(axis=1)
    t = (pw * np.abs(diff)).sum(axis=2).sum(axis=1)
    de = e[:, None] - e[None, :]
    dt = t[:, None] - t[None, :]
    h = (de + (np.abs(de) >= tau) * dt).sum(axis=1)
    return h, e, t, ns


def _points(m):
    return [[x._pts() for x in row] for row in m]


def test_fuzzy_codas_tfn_hand_computed_steps():
    # C1 (benefit): centroids 3, 5, 7/3 -> max 5; n = (.4,.6,.8), (.8,1,1.2), (.2,.4,.8)
    # C2 (cost): centroids 2, 13/3, 3 -> max 13/3; n = 1 - x/(13/3) reversed:
    #   A1 (1-9/13, 1-6/13, 1-3/13) = (4/13, 7/13, 10/13), A2 (-5/13, 1/13, 4/13), A3 (1/13, 4/13, 7/13)
    # weighted (0.6, 0.4): ns_1 = A3 (lowest centroid 0.28), ns_2 = A2 (centroid 0)
    r = fuzzy_codas(TFN_X, [0.6, 0.4], BC)
    ns = r.extras["negative_ideal"]
    assert ns["C1"]._pts() == approx((0.12, 0.24, 0.48))
    assert ns["C2"]._pts() == approx((-0.4 * 5 / 13, 0.4 / 13, 1.6 / 13))
    # A1 on C1: differences (.12, .12, 0) -> d_E = sqrt((.0144 + 4*.0144)/6), d_H = (.12 + .48)/6
    # A1 on C2: differences 0.4*(9/13, 6/13, 6/13) -> d_E = 0.4/13*sqrt((81+144+36)/6)
    e1 = (5 * 0.0144 / 6) ** 0.5 + 0.4 / 13 * (261 / 6) ** 0.5
    t1 = 0.6 / 6 + 0.4 / 13 * (9 + 24 + 6) / 6
    assert r.extras["E"]["A1"] == approx(e1)
    assert r.extras["T"]["A1"] == approx(t1)
    assert sum(r.scores.values()) == approx(0.0, abs=1e-12)


def test_fuzzy_codas_tfn_matches_independent_numpy():
    h, e, t, ns = _reference_codas(_points(TFN_X), [0.6, 0.4], BC)
    r = fuzzy_codas(TFN_X, [0.6, 0.4], BC, alternatives=["P", "Q", "R"], criteria=["cap", "price"])
    assert list(r.scores.values()) == approx(list(h), abs=1e-12)
    assert list(r.extras["E"].values()) == approx(list(e), abs=1e-12)
    assert list(r.extras["T"].values()) == approx(list(t), abs=1e-12)
    for got, want in zip(r.extras["negative_ideal"].values(), ns):
        assert got._pts() == approx(tuple(want), abs=1e-12)
    assert list(r.extras["negative_ideal"]) == ["cap", "price"]
    order = sorted(range(3), key=lambda i: -h[i])
    assert r.ranking == tuple("PQR"[i] for i in order)
    assert r.operators == {"ops": "algebraic", "score": "centroid", "tau": "0.02",
                           "psi": "equation", "normalization": "max_score"}


@pytest.mark.parametrize("types", [["benefit", "benefit"], ["cost", "cost"], ["cost", "benefit"]])
def test_fuzzy_codas_tfn_benefit_and_cost_against_numpy(types):
    m = [
        [TFN(3, 5, 7), TFN(5, 7, 9)],
        [TFN(5, 7, 9), TFN(1, 3, 5)],
        [TFN(7, 9, 9), TFN(3, 5, 7)],
        [TFN(1, 1, 3), TFN(7, 9, 9)],
    ]
    w = [0.55, 0.45]
    h, e, t, _ = _reference_codas(_points(m), w, types)
    r = fuzzy_codas(m, w, types)
    assert list(r.scores.values()) == approx(list(h), abs=1e-12)
    assert list(r.extras["E"].values()) == approx(list(e), abs=1e-12)
    assert list(r.extras["T"].values()) == approx(list(t), abs=1e-12)


def test_fuzzy_codas_tfn_cost_reverses_benefit_order():
    m = [[TFN(1, 2, 3)], [TFN(2, 3, 4)], [TFN(4, 5, 6)]]
    assert fuzzy_codas(m, [1.0], ["benefit"]).ranking == ("A3", "A2", "A1")
    assert fuzzy_codas(m, [1.0], ["cost"]).ranking == ("A1", "A2", "A3")


def test_fuzzy_codas_trfn_matches_independent_numpy():
    m = [
        [TrFN(1, 2, 3, 4), TrFN(0, 1, 1, 2), TrFN(5, 6, 7, 8)],
        [TrFN(2, 3, 4, 5), TrFN(1, 2, 2, 3), TrFN(2, 3, 4, 5)],
        [TrFN(4, 5, 5, 6), TrFN(2, 3, 4, 5), TrFN(7, 8, 8, 9)],
    ]
    w, types = [0.5, 0.3, 0.2], ["benefit", "cost", "benefit"]
    h, e, t, ns = _reference_codas(_points(m), w, types)
    r = fuzzy_codas(m, w, types)
    assert list(r.scores.values()) == approx(list(h), abs=1e-12)
    assert list(r.extras["E"].values()) == approx(list(e), abs=1e-12)
    assert list(r.extras["T"].values()) == approx(list(t), abs=1e-12)
    for got, want in zip(r.extras["negative_ideal"].values(), ns):
        assert got._pts() == approx(tuple(want), abs=1e-12)


def test_fuzzy_codas_tfn_equals_trfn_of_the_same_numbers():
    tr = [[x.to_trfn() for x in row] for row in TFN_X]
    a = fuzzy_codas(TFN_X, [0.6, 0.4], BC)
    b = fuzzy_codas(tr, [0.6, 0.4], BC)
    assert list(a.scores.values()) == approx(list(b.scores.values()), abs=1e-12)


def test_fuzzy_codas_tfn_tau_switches_the_taxicab_term():
    # tau = 1 drops every taxicab term (all |E_i - E_k| < 1): H_i = m E_i - sum_k E_k
    r = fuzzy_codas(TFN_X, [0.6, 0.4], BC, tau=1.0)
    e = list(r.extras["E"].values())
    assert list(r.scores.values()) == approx([3 * v - sum(e) for v in e])
    h, *_ = _reference_codas(_points(TFN_X), [0.6, 0.4], BC, tau=1.0)
    assert list(r.scores.values()) == approx(list(h))
    tables = fuzzy_codas(TFN_X, [0.6, 0.4], BC, tau=1.0, psi="tables")
    t = list(tables.extras["T"].values())
    assert list(tables.scores.values()) == approx(
        [3 * a - sum(e) + 3 * b - sum(t) for a, b in zip(e, t)])


def test_fuzzy_codas_tfn_score_selects_the_defuzzification():
    r = fuzzy_codas(TFN_X, [0.6, 0.4], BC, score="graded_mean")
    assert r.operators["score"] == "graded_mean"
    # graded mean of the C2 column: (1+8+3)/6 = 2, (3+16+6)/6 = 25/6, (2+12+4)/6 = 3 -> max 25/6
    assert r.extras["negative_ideal"]["C2"]._pts() == approx(
        (0.4 * (1 - 6 / (25 / 6)), 0.4 * (1 - 4 / (25 / 6)), 0.4 * (1 - 3 / (25 / 6))))


def test_fuzzy_codas_tfn_needs_a_positive_column_maximum():
    m = [[TFN(-3, -2, -1), TFN(1, 2, 3)], [TFN(-2, -1, 0), TFN(2, 3, 4)]]
    with pytest.raises(ValueError, match=r"fuzzy_codas: criterion 'C1'.*largest centroid score.*-1"):
        fuzzy_codas(m, W, BB)


def test_fuzzy_codas_tfn_weights_mapping_and_normalize():
    a = fuzzy_codas(TFN_X, {"x": 3, "y": 2}, BC, normalize_weights=True)
    b = fuzzy_codas(TFN_X, [0.6, 0.4], BC)
    assert list(a.scores.values()) == approx(list(b.scores.values()))
    assert list(a.extras["negative_ideal"]) == ["x", "y"]


def test_tfn_li_distances_reproduce_published_ed_hd():
    # Keshavarz Ghorabaee et al. (2017), Table 6: the weighted normalised matrix (printed to
    # two decimals) of segments S1..S5 and the negative ideal, over the 16 criteria, and the
    # published ED_i / HD_i. Summing Li's distances over the rounded table must give them back
    # to within the rounding of the table.
    rows = [
        [(.01, .04, .06, .12), (.05, .11, .15, .25), (.01, .04, .06, .12), (.09, .18, .23, .35), (.02, .05, .08, .15), (.01, .04, .06, .12)],
        [(.38, .53, .61, .79), (.49, .65, .73, .89), (.28, .4, .45, .6), (.41, .56, .63, .81), (.04, .09, .13, .25), (.04, .09, .13, .25)],
        [(.28, .41, .49, .62), (.28, .41, .49, .62), (.07, .14, .21, .33), (.66, .85, .97, 1.06), (.47, .63, .7, .84), (.07, .14, .21, .33)],
        [(.36, .51, .59, .74), (.35, .5, .56, .72), (.28, .41, .45, .61), (.24, .36, .42, .58), (.38, .54, .6, .75), (.24, .36, .42, .58)],
        [(.07, .15, .21, .34), (.1, .19, .27, .43), (.03, .08, .12, .22), (.05, .11, .15, .26), (.07, .15, .21, .34), (.03, .08, .12, .22)],
        [(.41, .56, .63, .81), (.23, .36, .41, .57), (.03, .07, .13, .24), (.51, .68, .77, .93), (.54, .71, .79, .95), (.03, .07, .13, .24)],
        [(.05, .11, .17, .3), (.06, .12, .2, .34), (.31, .46, .54, .74), (.31, .46, .54, .74), (.02, .06, .12, .24), (.02, .06, .12, .24)],
        [(.07, .13, .18, .28), (.17, .3, .35, .49), (.08, .15, .18, .29), (.17, .29, .34, .48), (.03, .08, .11, .2), (.03, .08, .11, .2)],
        [(.4, .55, .65, .79), (.4, .55, .65, .79), (.07, .15, .2, .31), (.34, .48, .57, .73), (.31, .44, .53, .69), (.07, .15, .2, .31)],
        [(.03, .08, .14, .28), (.03, .07, .13, .25), (.02, .06, .12, .23), (.02, .06, .1, .21), (0, .01, .03, .09), (0, .01, .03, .09)],
        [(.02, .06, .09, .18), (.03, .08, .11, .22), (.01, .03, .05, .11), (.02, .06, .09, .18), (.05, .12, .17, .29), (.01, .03, .05, .11)],
        [(.52, .72, .84, 1.04), (.36, .54, .62, .81), (.43, .62, .7, .9), (.55, .75, .81, 1.01), (.61, .83, .89, 1.1), (.36, .54, .62, .81)],
        [(.45, .62, .69, .88), (.54, .72, .84, 1.01), (.42, .58, .67, .85), (.45, .62, .69, .88), (.2, .33, .38, .53), (.2, .33, .38, .53)],
        [(.07, .14, .18, .28), (.21, .33, .39, .54), (.1, .17, .21, .32), (.28, .41, .49, .62), (.23, .36, .42, .56), (.07, .14, .18, .28)],
        [(.14, .25, .34, .47), (.03, .08, .11, .2), (.01, .03, .05, .12), (.06, .13, .18, .29), (.04, .09, .12, .21), (.01, .03, .05, .12)],
        [(.42, .57, .64, .8), (.31, .44, .5, .67), (.27, .4, .45, .62), (.35, .5, .58, .76), (.43, .59, .67, .83), (.27, .4, .45, .62)],
    ]
    published_ed = [3.067, 3.080, 1.293, 4.102, 2.588]
    published_hd = [3.020, 3.026, 1.271, 4.052, 2.560]
    for i in range(5):
        ed = sum(TrFN(*row[i]).distance(TrFN(*row[5]), "euclidean") for row in rows)
        hd = sum(TrFN(*row[i]).distance(TrFN(*row[5]), "hamming") for row in rows)
        assert ed == approx(published_ed[i], abs=0.02)
        assert hd == approx(published_hd[i], abs=0.02)


def test_codas_assessment_reproduces_published_scores():
    # Keshavarz Ghorabaee et al. (2017), Table 7, from the ED/HD of Table 6 with theta = 0.02
    from bulanik.mcdm.scoring import _codas_assessment

    ed = [3.067, 3.080, 1.293, 4.102, 2.588]
    hd = [3.020, 3.026, 1.271, 4.052, 2.560]
    assert _codas_assessment(ed, hd, 0.02, "equation") == approx(
        [2.380, 2.464, -15.237, 12.711, -2.318], abs=5e-3)  # inputs rounded to 3 places


def _tfn_dm():
    e1 = [[TFN(2, 3, 4), TFN(1, 2, 3)], [TFN(4, 5, 6), TFN(3, 4, 6)], [TFN(1, 2, 4), TFN(2, 3, 4)]]
    e2 = [[TFN(3, 4, 5), TFN(1, 1, 2)], [TFN(3, 4, 5), TFN(4, 5, 6)], [TFN(2, 3, 5), TFN(1, 2, 3)]]
    return DecisionMatrix(["P", "Q", "R"], ["cap", "price"], [e1, e2]), e1, e2


def test_rank_fuzzy_fuzzy_codas_on_tfn_decision_matrix():
    dm, e1, e2 = _tfn_dm()
    r = rank_fuzzy(dm, {"cap": 0.6, "price": 0.4}, BC, method="fuzzy_codas")
    avg = [[tuple((p + q) / 2 for p, q in zip(a._pts(), b._pts())) for a, b in zip(r1, r2)]
           for r1, r2 in zip(e1, e2)]
    h, e, t, _ = _reference_codas(avg, [0.6, 0.4], BC)
    assert list(r.scores.values()) == approx(list(h), abs=1e-12)
    assert list(r.extras["E"].values()) == approx(list(e), abs=1e-12)
    assert r.alternatives == ("P", "Q", "R")
    assert r.operators == {"ops": "algebraic", "score": "centroid", "tau": "0.02", "psi": "equation",
                           "normalization": "max_score", "aggregation": "wa"}
    direct = fuzzy_codas(aggregate_experts(dm.experts, None), [0.6, 0.4], BC,
                         alternatives=["P", "Q", "R"], criteria=["cap", "price"])
    assert r.scores == approx(direct.scores)


def test_rank_fuzzy_fuzzy_codas_tfn_passes_tau_and_score():
    dm, *_ = _tfn_dm()
    r = rank_fuzzy(dm, [0.6, 0.4], BC, method="fuzzy_codas", tau=0.05, score="graded_mean",
                   aggregation="wg")
    assert r.operators["tau"] == "0.05" and r.operators["score"] == "graded_mean"
    assert r.operators["aggregation"] == "wg"
