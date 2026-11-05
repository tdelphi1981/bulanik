import pytest
from pytest import approx

from bulanik.mcdm import edas, topsis, vikor, waspas

# shared crisp case: A1=(7,9) A2=(8,7) A3=(9,8); C1 benefit, C2 cost; w=(.6,.4)
X, W, T = [[7, 9], [8, 7], [9, 8]], [0.6, 0.4], ["benefit", "cost"]


def test_topsis_golden():
    # both column norms √194; ideal (9, 7) worst (7, 9) in raw terms
    # A2: d+=.6/√194, d-=√(.6²+.8²)/√194=1/√194 -> 1/1.6 ; A3: d+=.4/√194, d-=√1.6/√194 -> √1.6/(.4+√1.6)
    r = topsis(X, W, T)
    assert list(r.scores.values()) == approx([0.0, 0.625, 1.6**0.5 / (0.4 + 1.6**0.5)])  # .759747
    assert r.ranking == ("A3", "A2", "A1") and r.operators["normalization"] == "vector"
    assert set(r.extras) >= {"d_plus", "d_minus"}


def test_vikor_golden():
    # f*=(9,7) f-=(7,9): S=(1, .3, .2) R=(.6, .3, .2); Q_A2 = .5*(.1/.8)+.5*(.1/.4)
    r = vikor(X, W, T)
    assert list(r.scores.values()) == approx([1.0, 0.1875, 0.0]) and r.ranking == ("A3", "A2", "A1")
    assert r.extras["S"] == approx({"A1": 1.0, "A2": 0.3, "A3": 0.2}) and r.operators["v"] == "0.5"


def test_waspas_golden():
    # N: A1 (7/9, 7/9), A2 (8/9, 1), A3 (1, 7/8)
    # A2: WSM .6*8/9+.4 = .933333, WPM (8/9)^.6 = .931769 ; A3: WSM .95, WPM (7/8)^.4 = .947989
    r = waspas(X, W, T)
    assert list(r.scores.values()) == approx(
        [7 / 9, (0.6 * 8 / 9 + 0.4 + (8 / 9) ** 0.6) / 2, (0.95 + 0.875**0.4) / 2]
    )
    with pytest.raises(ValueError, match="positive"):
        waspas([[0, 1], [1, 1]], W, T)


def test_edas_golden():
    # AV=(8,8): A1 NDA (1/8,1/8) SN=.125 ; A2 PDA C2 1/8 SP=.05 ; A3 PDA C1 1/8 SP=.075 ; NSP (0,2/3,1) NSN (0,1,1)
    r = edas(X, W, T)
    assert list(r.scores.values()) == approx([0.0, 5 / 6, 1.0]) and r.ranking == ("A3", "A2", "A1")


def test_constant_column_is_neutral():
    Y = [[7, 5], [8, 5], [9, 5]]
    for fn in (topsis, vikor, waspas, edas):
        r = fn(Y, W, T)
        assert r.ranking == ("A3", "A2", "A1")


def test_ties_keep_input_order():
    r = topsis([[5, 5], [5, 5], [6, 4]], W, T, alternatives=["p", "q", "r"])
    assert r.ranking == ("r", "p", "q")


def test_validation_messages():
    with pytest.raises(ValueError, match="valid: benefit, cost"):
        topsis(X, W, ["benefit", "profit"])
    with pytest.raises(ValueError, match="sum to 1"):
        topsis(X, [0.6, 0.6], T)
    with pytest.raises(ValueError):
        topsis([[1, 2], [3]], W, T)
    assert topsis(X, {"C1": 0.6, "C2": 0.4}, T).ranking == ("A3", "A2", "A1")


def test_identical_alternatives_topsis_half():
    # all rows equal -> d+ = d- = 0 -> C = 0.5 by ruling
    r = topsis([[3, 4], [3, 4]], W, T)
    assert list(r.scores.values()) == [0.5, 0.5]


def test_edas_degenerate_branches():
    # all alternatives equal: SP = SN = 0 -> NSP = 0, NSN = 1 -> AS = 0.5
    r = edas([[3, 4], [3, 4]], W, T)
    assert list(r.scores.values()) == approx([0.5, 0.5])
    with pytest.raises(ValueError, match="average"):
        edas([[0, 1], [0, 1]], W, T)


def test_vikor_v_and_waspas_lambda():
    # v = 1 -> Q = (S-S*)/(S^- - S*) = (S - .2)/.8 : A1 1, A2 .125, A3 0
    r = vikor(X, W, T, v=1.0)
    assert list(r.scores.values()) == approx([1.0, 0.125, 0.0]) and r.operators["v"] == "1.0"
    # lam = 1 -> pure WSM; A1 = 7/9
    w = waspas(X, W, T, lam=1.0)
    assert w.scores["A3"] == approx(0.95) and w.operators["lambda"] == "1.0"
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            vikor(X, W, T, v=bad)
        with pytest.raises(ValueError, match="between 0 and 1"):
            waspas(X, W, T, lam=bad)


def test_names_and_normalize_weights():
    r = topsis(X, [3, 2], T, alternatives=["x", "y", "z"], criteria=["a", "b"], normalize_weights=True)
    assert r.ranking == ("z", "y", "x") and r.alternatives == ("x", "y", "z")
    with pytest.raises(ValueError, match="unique"):
        topsis(X, W, T, alternatives=["x", "x", "z"])
    with pytest.raises(ValueError, match="alternatives"):
        topsis(X, W, T, alternatives=["x", "y"])
    with pytest.raises(ValueError, match="criteria"):
        topsis(X, W, T, criteria=["a"])


def test_topsis_huge_entries_keep_every_criterion():
    # column 1 ~1e200: sum of squares overflowed to inf, so it silently dropped out.
    # TOPSIS is scale-invariant per column, so this equals [[1, 2], [3, 1]]:
    # v = (.5/√10, 1/√5), (1.5/√10, .5/√5); d+_A1 = 1/√10 = .316228, d-_A1 = .5/√5 = .223607
    # CC_A1 = .223607/.539835 = √2-1 ; CC_A2 = 2-√2
    r = topsis([[1e200, 2], [3e200, 1]], [0.5, 0.5], ["benefit", "benefit"])
    assert list(r.scores.values()) == approx([2**0.5 - 1, 2 - 2**0.5])
    assert r.ranking == ("A2", "A1")


def test_topsis_norm_overflow_raises():
    # hypot(1.7e308, 1.7e308) = 2.4e308 is not a float: refuse rather than drop the column
    with pytest.raises(ValueError, match="finite"):
        topsis([[1.7e308, 1], [1.7e308, 2]], [0.5, 0.5], ["benefit", "benefit"])


def test_vikor_huge_span_is_safe():
    # best - worst = 2e308 overflowed to inf; VIKOR is invariant to scaling a column,
    # so ±1e308 must give the same Q as ±1
    big = vikor([[-1e308, 2], [0.0, 1], [1e308, 3]], [0.5, 0.5], ["benefit", "benefit"])
    small = vikor([[-1.0, 2], [0.0, 1], [1.0, 3]], [0.5, 0.5], ["benefit", "benefit"])
    assert list(big.scores.values()) == approx(list(small.scores.values()))
    assert big.ranking == small.ranking


def test_edas_huge_average_is_safe():
    # sum(col) = 2.4e308 overflowed, making AV inf and PDA/NDA 0; EDAS is
    # scale-invariant per column, so this equals the 1.6 / 0.8 matrix
    big = edas([[1.6e308, 1], [0.8e308, 2]], [0.5, 0.5], ["benefit", "benefit"])
    small = edas([[1.6, 1], [0.8, 2]], [0.5, 0.5], ["benefit", "benefit"])
    assert list(big.scores.values()) == approx(list(small.scores.values()))
    # around 1e200 the old code already worked; keep it so (AV = 2e200, PDA = .5)
    mid = edas([[3e200, 1], [1e200, 2]], [0.5, 0.5], ["benefit", "benefit"])
    assert mid.extras["SP"]["A1"] == approx(0.5 * 0.5)


def test_bwm_result_works_as_weights():
    from bulanik.mcdm import BWMResult

    bw = BWMResult(
        method="bwm_linear", criteria=("q", "c"), weights={"c": 0.25, "q": 0.75},
        ranking=("q", "c"), xi=0.0, consistency_ratio=0.0, best="q", worst="c", operators={},
    )
    r = topsis([[1, 2], [3, 1]], bw, ["benefit", "benefit"], criteria=["q", "c"])
    plain = topsis([[1, 2], [3, 1]], [0.75, 0.25], ["benefit", "benefit"])
    assert list(r.scores.values()) == approx(list(plain.scores.values()))


def test_mapping_weights_reject_unknown_keys():
    with pytest.raises(ValueError, match="unknown criteria: z"):
        topsis([[1, 2], [3, 1]], {"q": 0.5, "c": 0.5, "z": 0.0}, ["benefit", "benefit"], criteria=["q", "c"])


@pytest.mark.parametrize("method", [vikor, edas])
def test_subnormal_columns_rank_like_their_scaled_selves(method):
    # VIKOR and EDAS are scale-invariant per column; halving 5e-324 used to round it to 0
    t = 5e-324
    types = ["benefit", "cost"]
    tiny = method([[t, 1], [2 * t, 2], [4 * t, 5]], [0.5, 0.5], types)
    plain = method([[1, 1], [2, 2], [4, 5]], [0.5, 0.5], types)
    assert list(tiny.scores.values()) == approx(list(plain.scores.values()), abs=1e-12)
    assert tiny.ranking == plain.ranking


def test_edas_all_subnormal_column_is_constant_not_an_error():
    r = edas([[5e-324, 1], [5e-324, 2]], [0.5, 0.5], ["benefit", "benefit"])
    # column 1 is constant, so only column 2 decides
    assert r.ranking == ("A2", "A1")


def test_edas_error_reports_the_unscaled_average():
    with pytest.raises(ValueError, match=r"column 1 has -0\.3333"):
        edas([[-3, 1], [1, 2], [1, 3]], W, T)


def test_crisp_rankers_infer_criteria_from_mapping_and_result_weights():
    from bulanik.mcdm import swara

    wr = swara([0.3], ["q", "p"])
    base = topsis(X, [wr.weights["q"], wr.weights["p"]], T)
    assert topsis(X, wr, T).scores == base.scores
    assert topsis(X, dict(wr.weights), T).scores == base.scores
    assert vikor(X, wr, T).scores == vikor(X, [wr.weights["q"], wr.weights["p"]], T).scores
