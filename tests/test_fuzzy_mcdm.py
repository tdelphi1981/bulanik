import pytest
from pytest import approx

from bulanik import IFN, IT2TrFN, IVIFN, IVNeutrosophicFN, TFN
from bulanik.aggregate import aggregate_experts
from bulanik.mcdm import (
    DecisionMatrix,
    cocoso,
    codas,
    copras,
    electre,
    fuzzy_topsis,
    marcos,
    rank_fuzzy,
    swara,
    topsis,
    wisp,
)
from bulanik.scales import EXAMPLE_IFN_5

DM = DecisionMatrix(["a1", "a2"], ["c1"], [[[IFN(0.6, 0.3)], [IFN(0.5, 0.4)]]])


def test_fuzzy_topsis_chen_golden():
    M = [[TFN(1, 3, 5), TFN(5, 7, 9)], [TFN(3, 5, 7), TFN(1, 3, 5)]]
    w = [TFN(0.5, 0.7, 0.9), TFN(0.3, 0.5, 0.7)]
    # c*_1=7 ; a-_2=1 ; A1: r11=(1/7,3/7,5/7) v=(.071429,.3,.642857) ; r12=(1/9,1/7,1/5) v=(.033333,.071429,.14)
    # d* = Σ vertex(v,(1,1,1)) = 1.621801 ; d- = .504410 -> CC .237234 ; A2 CC .443214
    r = fuzzy_topsis(M, w, ["benefit", "cost"])
    assert list(r.scores.values()) == approx([0.2372341722, 0.4432137357]) and r.ranking == ("A2", "A1")
    assert r.operators["distance"] == "vertex" and set(r.extras) >= {"d_star", "d_minus"}


def test_fuzzy_topsis_crisp_weights_and_validation():
    M = [[TFN(1, 3, 5)], [TFN(3, 5, 7)]]
    # crisp w=1 equals TFN(1,1,1): same scores
    a = fuzzy_topsis(M, [1.0], ["benefit"])
    b = fuzzy_topsis(M, [TFN(1, 1, 1)], ["benefit"])
    assert list(a.scores.values()) == approx(list(b.scores.values()))
    assert a.ranking == ("A2", "A1")
    with pytest.raises(ValueError, match="valid: benefit, cost"):
        fuzzy_topsis(M, [1.0], ["profit"])
    with pytest.raises(ValueError, match="expected 1 weights"):
        fuzzy_topsis(M, [1.0, 1.0], ["benefit"])
    with pytest.raises(ValueError, match="positive"):
        fuzzy_topsis([[TFN(0, 1, 2)], [TFN(1, 2, 3)]], [1.0], ["cost"])
    with pytest.raises(ValueError, match="l >= 0"):
        fuzzy_topsis([[TFN(-3, -2, -1)], [TFN(-2, -1, 0)]], [1.0], ["benefit"])
    with pytest.raises(ValueError):
        fuzzy_topsis([[TFN(1, 2, 3)], [TFN(1, 2, 3), TFN(1, 2, 3)]], [1.0], ["benefit"])
    with pytest.raises(TypeError):
        fuzzy_topsis([[1.0], [2.0]], [1.0], ["benefit"])


def test_rank_fuzzy_reports_names():
    dm = DecisionMatrix.from_terms(
        EXAMPLE_IFN_5, ["s1", "s2"], ["q", "c"], [[["H", "L"], ["M", "M"]], [["VH", "L"], ["M", "H"]]]
    )
    r = rank_fuzzy(dm, [0.5, 0.5], ["benefit", "cost"], method="topsis", aggregation="wg", ops="einstein")
    assert r.operators == {
        **r.operators,
        "aggregation": "wg",
        "ops": "einstein",
        "score": "difference",
        "normalization": "vector",
    }
    assert set(r.ranking) == {"s1", "s2"}


def test_rank_fuzzy_equals_manual_pipeline():
    dm = DecisionMatrix(
        ["a1", "a2"],
        ["c1", "c2"],
        [
            [[IFN(0.6, 0.3), IFN(0.5, 0.4)], [IFN(0.4, 0.5), IFN(0.7, 0.2)]],
            [[IFN(0.5, 0.4), IFN(0.6, 0.3)], [IFN(0.5, 0.3), IFN(0.6, 0.2)]],
        ],
        expert_weights=[0.6, 0.4],
    )
    manual = topsis(
        [[x.score() for x in row] for row in aggregate_experts(dm.experts, [0.6, 0.4])],
        [0.5, 0.5],
        ["benefit", "benefit"],
        alternatives=["a1", "a2"],
    )
    assert rank_fuzzy(dm, [0.5, 0.5], ["benefit", "benefit"]).scores == approx(manual.scores)


def test_rank_fuzzy_unknown_method():
    with pytest.raises(ValueError, match="unknown method 'promethee'"):
        rank_fuzzy(DM, [1.0], ["benefit"], method="promethee")


def test_rank_fuzzy_other_methods_and_kwargs():
    for m in ("vikor", "waspas", "edas"):
        r = rank_fuzzy(DM, [1.0], ["benefit"], method=m, score="normalized")
        assert r.ranking == ("a1", "a2") and r.operators["score"] == "normalized"
    # normalized scores (1+mu-nu)/2: a1 .65, a2 .55
    with pytest.raises(ValueError, match="valid: difference, normalized"):
        rank_fuzzy(DM, [1.0], ["benefit"], score="foo")
    r = rank_fuzzy(DM, [2.0], ["benefit"], normalize_weights=True)
    assert r.ranking == ("a1", "a2")
    r = rank_fuzzy(DM, {"c1": 1.0}, ["benefit"], method="vikor", v=1.0)
    assert r.operators["v"] == "1.0"


def test_decision_matrix_is_frozen_and_validated():
    assert isinstance(DM.experts, tuple) and isinstance(DM.experts[0][0], tuple)
    with pytest.raises(AttributeError):
        DM.alternatives = ("x",)  # type: ignore[misc]
    one = IFN(0.6, 0.3)
    with pytest.raises(ValueError, match="alternatives"):
        DecisionMatrix(["a"], ["c"], [[[one], [one]]])
    with pytest.raises(ValueError, match="criteria"):
        DecisionMatrix(["a"], ["c"], [[[one, one]]])
    with pytest.raises(ValueError, match="expert"):
        DecisionMatrix(["a"], ["c"], [])
    with pytest.raises(ValueError, match="unique"):
        DecisionMatrix(["a", "a"], ["c"], [[[one], [one]]])
    with pytest.raises(ValueError, match="expert weights|weights"):
        DecisionMatrix(["a"], ["c"], [[[one]]], expert_weights=[0.5, 0.5])
    with pytest.raises(TypeError):
        DecisionMatrix(["a"], ["c"], [[[0.5]]])
    with pytest.raises(TypeError, match="cannot combine"):
        DecisionMatrix(["a", "b"], ["c"], [[[one], [TFN(1, 2, 3)]]])


def test_fuzzy_topsis_rejects_weights_outside_unit_interval():
    # Chen's FPIS (1,1,1) / FNIS (0,0,0) need every weighted value in [0, 1];
    # 0-10 scale weights (G, VG) used to rank the clearly worst A3 first.
    from bulanik.scales import EXAMPLE_TFN_7 as S

    M = [[S["F"], S["MG"]], [S["G"], S["VG"]], [S["P"], S["MP"]]]
    types = ["benefit", "benefit"]
    with pytest.raises(ValueError, match=r"criterion 'q'.*\[0, 1\].*normalis"):
        fuzzy_topsis(M, [S["G"], S["VG"]], types, criteria=["q", "c"])
    with pytest.raises(ValueError, match=r"criterion 'C2'.*\[0, 1\]"):
        fuzzy_topsis(M, [0.5, 1.5], types)
    with pytest.raises(ValueError, match=r"criterion 'C1'.*\[0, 1\]"):
        fuzzy_topsis(M, [TFN(-0.1, 0.2, 0.3), 1.0], types)
    with pytest.raises(TypeError):
        fuzzy_topsis(M, ["0.5", 0.5], types)
    with pytest.raises(TypeError):
        fuzzy_topsis(M, [True, 0.5], types)
    # the same weights normalised by 10 rank A2 (G, VG) first and A3 (P, MP) last
    r = fuzzy_topsis(M, [TFN(0.7, 0.9, 1.0), TFN(0.9, 1.0, 1.0)], types)
    assert r.ranking == ("A2", "A1", "A3")


def test_fuzzy_topsis_validates_rating_signs():
    # benefit columns need l >= 0 (a zero lower bound is fine); cost needs l > 0 (a-/x)
    ok = fuzzy_topsis([[TFN(0, 1, 2)], [TFN(1, 2, 3)]], [1.0], ["benefit"])
    assert ok.ranking == ("A2", "A1")
    with pytest.raises(ValueError, match=r"benefit criterion 'C1'.*l >= 0"):
        fuzzy_topsis([[TFN(-1, 1, 2)], [TFN(1, 2, 3)]], [1.0], ["benefit"])
    with pytest.raises(ValueError, match=r"cost criterion 'k'.*positive"):
        fuzzy_topsis([[TFN(0, 1, 2)], [TFN(1, 2, 3)]], [1.0], ["cost"], criteria=["k"])
    with pytest.raises(ValueError, match="positive upper bound"):
        fuzzy_topsis([[TFN(0, 0, 0)], [TFN(0, 0, 0)]], [1.0], ["benefit"])


def test_fuzzy_topsis_mapping_and_result_weights():
    M = [[TFN(1, 3, 5), TFN(5, 7, 9)], [TFN(3, 5, 7), TFN(1, 3, 5)]]
    types = ["benefit", "cost"]
    by_pos = fuzzy_topsis(M, [TFN(0.5, 0.7, 0.9), 0.4], types, criteria=["q", "p"])
    # mapping order is irrelevant: criteria pick the entries
    by_name = fuzzy_topsis(M, {"p": 0.4, "q": TFN(0.5, 0.7, 0.9)}, types, criteria=["q", "p"])
    assert by_name.scores == by_pos.scores
    # without criteria=, a mapping's keys name the columns in its order
    keyed = fuzzy_topsis(M, {"q": TFN(0.5, 0.7, 0.9), "p": 0.4}, types)
    assert keyed.scores == by_pos.scores
    # a WeightingResult is accepted (weights sum to 1, so each lies in [0, 1])
    wr = swara([0.3], ["q", "p"])
    from_result = fuzzy_topsis(M, wr, types, criteria=["q", "p"])
    assert from_result.scores == fuzzy_topsis(M, [wr.weights["q"], wr.weights["p"]], types).scores
    with pytest.raises(ValueError, match="missing criteria: p"):
        fuzzy_topsis(M, {"q": 0.5}, types, criteria=["q", "p"])
    with pytest.raises(ValueError, match="unknown criteria: z"):
        fuzzy_topsis(M, {"q": 0.5, "p": 0.5, "z": 0.1}, types, criteria=["q", "p"])


def test_fuzzy_topsis_shape_validation():
    M = [[TFN(1, 3, 5)], [TFN(3, 5, 7)]]
    with pytest.raises(ValueError, match="expected 1 criteria, got 2"):
        fuzzy_topsis(M, [1.0], ["benefit"], criteria=["a", "b"])
    with pytest.raises(ValueError, match="expected 1 weights, got 2"):
        fuzzy_topsis(M, [1.0, 1.0], ["benefit"])
    with pytest.raises(ValueError, match="2 unique names"):
        fuzzy_topsis(M, [1.0], ["benefit"], alternatives=["a", "a"])
    with pytest.raises(ValueError, match="at least one row and one column"):
        fuzzy_topsis([], [], [])
    with pytest.raises(ValueError, match="same length"):
        fuzzy_topsis([[TFN(1, 2, 3)], [TFN(1, 2, 3), TFN(1, 2, 3)]], [1.0], ["benefit"])


def test_decision_matrix_needs_alternatives_and_criteria():
    with pytest.raises(ValueError, match="at least one of alternatives"):
        DecisionMatrix([], ["c"], [[]])
    with pytest.raises(ValueError, match="at least one of criteria"):
        DecisionMatrix(["a"], [], [[[]]])
    with pytest.raises(ValueError, match="unique"):
        DecisionMatrix(["a", "a"], ["c"], [[[IFN(0.6, 0.3)], [IFN(0.5, 0.4)]]])


def test_rank_fuzzy_unknown_aggregation_names_aggregation():
    with pytest.raises(ValueError, match=r"unknown aggregation 'mean'.*valid: wa, wg"):
        rank_fuzzy(DM, [1.0], ["benefit"], aggregation="mean")


def test_fuzzy_topsis_infers_criteria_from_results_like_mappings():
    M = [[TFN(1, 3, 5), TFN(5, 7, 9)], [TFN(3, 5, 7), TFN(1, 3, 5)]]
    types = ["benefit", "cost"]
    wr = swara([0.3], ["q", "p"])
    inferred = fuzzy_topsis(M, wr, types)  # no criteria=: taken from wr.weights
    assert inferred.scores == fuzzy_topsis(M, wr, types, criteria=["q", "p"]).scores


def test_fuzzy_topsis_rejects_duplicate_criteria():
    M = [[TFN(1, 3, 5), TFN(5, 7, 9)], [TFN(3, 5, 7), TFN(1, 3, 5)]]
    with pytest.raises(ValueError, match="criteria names must be unique"):
        fuzzy_topsis(M, [0.5, 0.5], ["benefit", "cost"], criteria=["q", "q"])


_CRISP = {
    "codas": (codas, {"tau": 0.05}),
    "copras": (copras, {}),
    "marcos": (marcos, {}),
    "cocoso": (cocoso, {"lam": 0.7}),
    "wisp": (wisp, {"product": "power"}),
    "electre": (electre, {"c_threshold": 0.5, "d_threshold": 0.6}),
}


@pytest.mark.parametrize("name", list(_CRISP))
def test_rank_fuzzy_new_methods_match_crisp(name):
    fn, kw = _CRISP[name]
    dm = DecisionMatrix(
        ["A", "B", "C"],
        ["q", "p"],
        [
            [[TFN(5, 7, 9), TFN(3, 5, 7)], [TFN(7, 8, 9), TFN(1, 3, 5)], [TFN(3, 5, 7), TFN(2, 4, 6)]],
            [[TFN(4, 6, 8), TFN(2, 4, 6)], [TFN(6, 8, 9), TFN(2, 3, 4)], [TFN(2, 4, 6), TFN(2, 4, 6)]],
        ],
    )
    w, types = [0.6, 0.4], ["benefit", "cost"]
    agg = aggregate_experts(dm.experts, None, method="wa", ops="algebraic")
    crisp = [[x.score("centroid") for x in row] for row in agg]
    ref = fn(crisp, w, types, alternatives=list(dm.alternatives), criteria=list(dm.criteria), **kw)
    got = rank_fuzzy(dm, w, types, method=name, **kw)
    assert got.method == ref.method and got.ranking == ref.ranking
    assert list(got.scores.values()) == approx(list(ref.scores.values()))
    assert set(got.extras) == set(ref.extras)
    for key, val in ref.operators.items():
        if key == "score":  # ELECTRE's own ranking score moves to ranking_score
            assert got.operators["ranking_score"] == val
        else:
            assert got.operators[key] == val
    assert (got.operators["aggregation"], got.operators["ops"], got.operators["score"]) == (
        "wa",
        "algebraic",
        "centroid",
    )


def _two_experts(make):
    return DecisionMatrix(
        ["A", "B", "C"],
        ["q", "p"],
        [
            [[make(0.7), make(0.4)], [make(0.5), make(0.6)], [make(0.3), make(0.8)]],
            [[make(0.6), make(0.5)], [make(0.6), make(0.5)], [make(0.4), make(0.7)]],
        ],
    )


_MAKERS = {
    "ivifn": lambda a: IVIFN(a - 0.1, a, 0.1, 0.2),
    "ivnn": lambda a: IVNeutrosophicFN(a - 0.1, a, 0.1, 0.2, 0.2, 0.3),
    "it2": lambda a: IT2TrFN(
        (a - 0.2, a - 0.1, a + 0.1, a + 0.2, 1, 1), (a - 0.1, a, a, a + 0.1, 0.8, 0.8)
    ),
}


@pytest.mark.parametrize("kind", list(_MAKERS))
@pytest.mark.parametrize("aggregation", ["wa", "wg"])
@pytest.mark.parametrize("method", ["topsis", "codas"])
def test_rank_fuzzy_new_kinds(kind, aggregation, method):
    dm = _two_experts(_MAKERS[kind])
    # signed default scores are fine for topsis; CODAS needs positive ones
    kw = {"score": "normalized"} if kind == "ivifn" and method == "codas" else {}
    r = rank_fuzzy(dm, [0.5, 0.5], ["benefit", "cost"], method=method, aggregation=aggregation, **kw)
    assert sorted(r.ranking) == ["A", "B", "C"]
    assert r.operators["aggregation"] == aggregation


def test_ratio_methods_reject_signed_scores():
    # memberships below the non-memberships give negative "difference" scores
    dm = _two_experts(lambda a: IVIFN(0.0, 0.1, a * 0.7, a * 0.8 + 0.1))
    with pytest.raises(ValueError, match='score="normalized"'):
        rank_fuzzy(dm, [0.5, 0.5], ["benefit", "benefit"], method="marcos")
    r = rank_fuzzy(dm, [0.5, 0.5], ["benefit", "benefit"], method="marcos", score="normalized")
    assert r.operators["score"] == "normalized"


def test_unknown_method_lists_all_ten():
    with pytest.raises(ValueError) as e:
        rank_fuzzy(DM, [1.0], ["benefit"], method="nope")
    assert (
        "valid: topsis, vikor, waspas, edas, codas, copras, marcos, cocoso, wisp, electre"
        in str(e.value)
    )


def test_ratio_methods_hint_without_normalized_score():
    # IVNeutrosophicFN has only the "average" score: no score="normalized" to suggest
    # (MARCOS carries this: CODAS now accepts a zero in a benefit column)
    zero = IVNeutrosophicFN(0, 0, 1, 1, 1, 1)
    ok = IVNeutrosophicFN(0.5, 0.6, 0.1, 0.2, 0.1, 0.2)
    dm = DecisionMatrix(["A", "B"], ["q"], [[[zero], [ok]]])
    with pytest.raises(ValueError, match="use a positive-valued score") as e:
        rank_fuzzy(dm, [1.0], ["benefit"], method="marcos")
    assert "normalized" not in str(e.value)


def test_copras_negative_scores_get_the_remedy_hint():
    neg = IVIFN(0.1, 0.2, 0.6, 0.7)  # default difference score is negative
    dm = DecisionMatrix(["A", "B"], ["q"], [[[neg], [neg]]])
    with pytest.raises(ValueError, match='COPRAS needs non-negative entries.*score="normalized"'):
        rank_fuzzy(dm, [1.0], ["benefit"], method="copras", aggregation="wg")


def test_copras_hint_is_rewritten_for_kinds_without_normalized(monkeypatch):
    # the IVNeutrosophicFN average score lies in [0, 1], so force a negative one
    monkeypatch.setattr(IVNeutrosophicFN, "score", lambda self, name=None: -1.0)
    n = IVNeutrosophicFN(0.5, 0.6, 0.1, 0.2, 0.1, 0.2)
    dm = DecisionMatrix(["A", "B"], ["q"], [[[n], [n]]])
    with pytest.raises(ValueError, match="use a positive-valued score") as e:
        rank_fuzzy(dm, [1.0], ["benefit"], method="copras")
    assert "normalized" not in str(e.value)
