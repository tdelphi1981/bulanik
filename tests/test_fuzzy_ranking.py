import itertools

import pytest
from pytest import approx

from bulanik import IFN, IVIFN, PFN, TFN, NeutrosophicFN, PictureFN
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


@pytest.mark.parametrize("fn", [fuzzy_codas, fuzzy_waspas, fuzzy_cocoso])
def test_cost_with_tfn_raises_typeerror(fn):
    m = [[TFN(1, 2, 3), TFN(2, 3, 4)], [TFN(2, 3, 4), TFN(1, 2, 3)]]
    with pytest.raises(TypeError, match=r"TFN has no complement.*rank_fuzzy"):
        fn(m, W, ["benefit", "cost"])


def test_tfn_benefit_works_where_distances_are_not_needed():
    m = [[TFN(1, 2, 3), TFN(2, 3, 4)], [TFN(2, 3, 4), TFN(3, 4, 5)]]
    assert fuzzy_waspas(m, W, BB).ranking == ("A2", "A1")
    r = fuzzy_cocoso(m, W, BB)
    assert r.ranking == ("A2", "A1") and r.operators["score"] == "centroid"
    with pytest.raises(TypeError, match=r"euclidean.*hamming.*TFN has: vertex"):
        fuzzy_codas(m, W, BB)


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


def test_fuzzy_codas_tfn_distance_error_hints_rank_fuzzy():
    m = [[TFN(1, 2, 3), TFN(2, 3, 4)], [TFN(2, 3, 4), TFN(3, 4, 5)]]
    with pytest.raises(TypeError, match=r"TFN has: vertex.*rank_fuzzy"):
        fuzzy_codas(m, W, BB)


def test_fuzzy_cocoso_zero_p_message_names_the_product():
    m = [[IFN(0.0, 1.0), IFN(0.6, 0.3)], [IFN(0.6, 0.3), IFN(0.4, 0.5)]]
    with pytest.raises(ValueError, match=r"alternative 'A1'.*S or P is 0.*\(0, 1\).*k_b divides by zero") as ei:
        fuzzy_cocoso(m, W, BB)
    assert "worst on every criterion" not in str(ei.value)
