import math
import random
import warnings

import pytest
from pytest import approx

from bulanik.anfis import ANFIS, MIN_B, MIN_WIDTH, ExtrapolationWarning
from bulanik.inference import SugenoFIS

SIN_X = [[2 * math.pi * i / 49] for i in range(50)]
SIN_Y = [math.sin(x[0]) for x in SIN_X]


def test_rule_count_guard():
    assert ANFIS(2, [3, 4]).n_rules == 12
    with pytest.warns(UserWarning):
        ANFIS(3, 4)  # 64 > 27
    with pytest.raises(ValueError, match="allow_large"):
        ANFIS(2, 33)  # 1089 > 1024
    with pytest.warns(UserWarning):  # 1089 > 27 still warns when allowed
        assert ANFIS(2, 33, allow_large=True).n_rules == 1089
    with pytest.raises(ValueError, match="valid: gaussian, bell"):
        ANFIS(1, 2, mf="tri")


def test_linear_target_exact_after_one_hybrid_epoch():
    X = [[i / 10] for i in range(11)]
    y = [2 * x[0] + 1 for x in X]
    a = ANFIS(1, 2)
    h = a.fit(X, y, epochs=1)
    assert h[0] < 1e-6  # first-order Sugeno with Σw̄=1 represents 2x+1 exactly (p_r=2, r_r=1)


def test_sin_hybrid_reaches_threshold():
    a = ANFIS(1, 3)
    h = a.fit(SIN_X, SIN_Y, epochs=10, lr=0.01)
    assert len(h) == 10 and h[-1] < 0.01  # reference run: 0.0041
    assert a.history == h


def test_bell_variant():
    assert ANFIS(1, 3, mf="bell").fit(SIN_X, SIN_Y, epochs=10)[-1] < 0.05  # reference: 0.0386


def test_hybrid_not_worse_than_gradient():
    h = ANFIS(1, 3).fit(SIN_X, SIN_Y, epochs=20, method="hybrid")
    g = ANFIS(1, 3).fit(SIN_X, SIN_Y, epochs=20, method="gradient")
    assert h[-1] <= g[-1]


def test_gradient_method_decreases_rmse():
    # consequents start at 0, so the initial RMSE is RMS(sin) = sqrt(1/2) on a symmetric grid
    a = ANFIS(1, 3)
    g = a.fit(SIN_X, SIN_Y, epochs=20, method="gradient", lr=0.1)
    assert g[-1] < g[0] < math.sqrt(0.5) + 0.05


def _fd_check(a, X, y, n_prem):
    g = a.premise_gradient(X, y)
    p = a.get_params()
    assert len(g) == n_prem
    for k in range(n_prem):
        hi = list(p)
        hi[k] += 1e-6
        lo = list(p)
        lo[k] -= 1e-6
        a.set_params(hi)
        fh = a.rmse(X, y) ** 2
        a.set_params(lo)
        fl = a.rmse(X, y) ** 2
        a.set_params(p)
        assert g[k] == approx((fh - fl) / 2e-6, rel=1e-4, abs=1e-8), k


GRID2 = [[i / 4, j / 4] for i in range(5) for j in range(5)]
PROD2 = [x0 * x1 for x0, x1 in GRID2]


def test_premise_gradient_matches_finite_difference():
    a = ANFIS(2, 2, ranges=[(0, 1), (0, 1)])
    X, y = GRID2, PROD2
    a.fit(X, y, epochs=1)
    _fd_check(a, X, y, 8)


@pytest.mark.parametrize("mf, per_mf", [("gaussian", 2), ("bell", 3)])
def test_premise_gradient_matches_fd_at_random_params(mf, per_mf):
    rng = random.Random(11)
    a = ANFIS(2, 2, mf=mf, ranges=[(0, 1), (0, 1)])
    p = a.get_params()
    # widths/b scaled by [0.6, 1.4] (stay positive), centres shifted by +-0.2, consequents in [-2, 2]
    for k, (kind, _, _) in enumerate(a.param_layout()):
        if kind in ("sigma", "a", "b"):
            p[k] *= rng.uniform(0.6, 1.4)
        elif kind == "c":
            p[k] += rng.uniform(-0.2, 0.2)
        else:
            p[k] = rng.uniform(-2, 2)
    a.set_params(p)
    _fd_check(a, GRID2, PROD2, 2 * 2 * per_mf)


def test_export_to_sugeno_is_equivalent():
    a = ANFIS(1, 3)
    a.fit(SIN_X, SIN_Y, epochs=3)
    fis = a.to_sugeno(["x"])
    for x in (0.1, 1.0, 2.5, 4.0, 6.2):
        assert fis.evaluate({"x": x})["y"] == approx(a.predict([[x]])[0], abs=1e-9)


@pytest.mark.parametrize("mf", ["gaussian", "bell"])
def test_export_matches_predict_at_random_points(mf):
    rng = random.Random(5)
    a = ANFIS(2, [2, 3], mf=mf)
    X = [[rng.uniform(-1, 2), rng.uniform(0, 3)] for _ in range(60)]
    y = [math.sin(u) * v for u, v in X]
    a.fit(X, y, epochs=5)
    fis = a.to_sugeno(["u", "v"], output_name="z")
    assert isinstance(fis, SugenoFIS) and fis.and_method == "prod"
    (lo_u, hi_u), (lo_v, hi_v) = a.ranges
    pts = [[rng.uniform(lo_u, hi_u), rng.uniform(lo_v, hi_v)] for _ in range(200)]
    for p, f in zip(pts, a.predict(pts)):
        assert fis.evaluate({"u": p[0], "v": p[1]})["z"] == approx(f, abs=1e-12)


def test_export_structure_and_default_names():
    a = ANFIS(2, 2, ranges=[(0, 1), (0, 2)])
    fis = a.to_sugeno()
    assert [v.name for v in fis.inputs] == ["x1", "x2"]
    assert fis.inputs[1].universe == (0.0, 2.0) and fis.inputs[0].term_names == ("mf1", "mf2")
    # itertools.product order: (mf1, mf1), (mf1, mf2), (mf2, mf1), (mf2, mf2)
    assert str(fis.rules[1]) == "IF x1 IS mf1 AND x2 IS mf2 THEN y IS r2"
    assert fis.outputs[0].term_names == ("r1", "r2", "r3", "r4")
    with pytest.raises(ValueError, match="input_names"):
        a.to_sugeno(["only_one"])
    with pytest.raises(RuntimeError):
        ANFIS(1, 2).to_sugeno()


def test_predict_before_fit_without_ranges():
    with pytest.raises(RuntimeError):
        ANFIS(1, 2).predict([[0.5]])
    with pytest.raises(RuntimeError):
        ANFIS(1, 2).get_params()


def test_initial_params_and_layout():
    # centres linspace(0, 2, 3) = 0, 1, 2 ; sigma = (2-0)/(2*(3-1)) = .5 ; consequents 0
    a = ANFIS(1, 3, ranges=[(0, 2)])
    assert a.get_params() == [0, 0.5, 1, 0.5, 2, 0.5] + [0.0] * 6
    assert a.param_layout() == [
        ("c", 0, 0), ("sigma", 0, 0), ("c", 0, 1), ("sigma", 0, 1), ("c", 0, 2), ("sigma", 0, 2),
        ("p", 0, 0), ("r", -1, 0), ("p", 0, 1), ("r", -1, 1), ("p", 0, 2), ("r", -1, 2),
    ]
    # bell: a = width (4-0)/(2*(2-1)) = 2, b = 2, c = 0 and 4
    b = ANFIS(1, 2, mf="bell", ranges=[(0, 4)])
    assert b.get_params()[:6] == [2, 2, 0, 2, 2, 4]
    assert [t[0] for t in b.param_layout()[:3]] == ["a", "b", "c"]
    # one MF: centre linspace(0, 4, 1) = [0], sigma = (4-0)/2 = 2
    assert ANFIS(1, 1, ranges=[(0, 4)]).get_params() == [0, 2, 0, 0]
    # two inputs: premise input-major, consequents rule-major [p1, p2, r]
    lay = ANFIS(2, 2, ranges=[(0, 1), (0, 1)]).param_layout()
    assert len(lay) == 8 + 4 * 3 and lay[8:11] == [("p", 0, 0), ("p", 1, 0), ("r", -1, 0)]


def test_predict_hand_golden():
    # x = .5 is equidistant from c = 0 and c = 1 with equal sigma -> w̄ = (.5, .5);
    # rule 1: 2x + 0 = 1, rule 2: 0x + 3 = 3 -> output (1 + 3) / 2 = 2
    a = ANFIS(1, 2, ranges=[(0, 1)])
    a.set_params([0, 0.5, 1, 0.5, 2, 0, 0, 3])
    assert a.predict([[0.5]]) == approx([2.0])
    # rmse against y = 1: |2 - 1| = 1
    assert a.rmse([[0.5]], [1.0]) == approx(1.0)


def test_identical_fits_give_identical_histories():
    for method in ("hybrid", "gradient"):
        h1 = ANFIS(2, 2).fit(GRID2, PROD2, epochs=5, method=method)
        h2 = ANFIS(2, 2).fit(GRID2, PROD2, epochs=5, method=method)
        assert h1 == h2


def _fit_recording(a, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = a.fit(*args, **kwargs)
    return h, [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]


def _check_guarded(a, h, epochs):
    assert len(h) == epochs and all(math.isfinite(v) for v in h)
    p = a.get_params()
    assert all(math.isfinite(v) for v in p)
    span = 2 * math.pi
    for v, (kind, _, _) in zip(p, a.param_layout()):
        if kind in ("sigma", "a"):
            assert v >= MIN_WIDTH * span * (1 - 1e-12)
        if kind == "b":
            assert v >= MIN_B
    assert all(math.isfinite(v) for v in a.predict(SIN_X))


@pytest.mark.parametrize("mf", ["gaussian", "bell"])
@pytest.mark.parametrize("method, lr", [("hybrid", 1e6), ("gradient", 1e4)])
def test_huge_learning_rate_never_crashes_or_nans(mf, method, lr):
    a = ANFIS(1, 3, mf=mf)
    h, msgs = _fit_recording(a, SIN_X, SIN_Y, epochs=15, method=method, lr=lr)
    _check_guarded(a, h, 15)
    # deterministic: the halving guard absorbs these rates, so no step is skipped
    # (a skip would warn and repeat the previous RMSE exactly; see the next test)
    assert msgs == []
    assert all(prev != cur for prev, cur in zip(h, h[1:]))


@pytest.mark.parametrize("mf", ["gaussian", "bell"])
def test_absurd_learning_rate_skips_every_step_with_one_warning(mf):
    # lr = 1e300: even 60 halvings (1e300 * 2^-60 ~ 8.7e281) overflow, so every step is skipped
    a = ANFIS(1, 3, mf=mf)
    with pytest.warns(RuntimeWarning, match="skipped 15 of 15") as rec:
        h = a.fit(SIN_X, SIN_Y, epochs=15, method="gradient", lr=1e300)
    assert len([w for w in rec if issubclass(w.category, RuntimeWarning)]) == 1
    assert len(set(h)) == 1
    _check_guarded(a, h, 15)


def test_non_finite_gradient_skips_with_warning(monkeypatch):
    a = ANFIS(1, 2, ranges=[(0, 1)])
    n = len(a.get_params())
    monkeypatch.setattr(a, "_gradient", lambda *args, **kwargs: [math.nan] * n)
    start = a.get_params()
    with pytest.warns(RuntimeWarning, match="skipped 2 of 2"):
        a.fit([[0.0], [1.0]], [0.0, 1.0], epochs=2, method="gradient")
    assert a.get_params() == start


def test_normal_learning_rate_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ANFIS(1, 3).fit(SIN_X, SIN_Y, epochs=5)


def _lse_rmse(X, y):
    # lr 1e-12: the premise step is negligible, so the RMSE is the LSE fit's
    return ANFIS(1, 3).fit(X, y, epochs=1, lr=1e-12)[-1]


def test_lse_is_invariant_to_input_offset():
    base = _lse_rmse(SIN_X, SIN_Y)  # about 0.0041
    for shift in (1e4, 1e5):
        shifted = [[x[0] + shift] for x in SIN_X]
        assert _lse_rmse(shifted, SIN_Y) == approx(base, rel=1e-6, abs=1e-9)


def test_lse_is_invariant_to_input_scale():
    base = _lse_rmse(SIN_X, SIN_Y)
    for factor in (1e-3, 1e3):
        scaled = [[x[0] * factor] for x in SIN_X]
        assert _lse_rmse(scaled, SIN_Y) == approx(base, rel=1e-6, abs=1e-9)


def test_linear_target_on_tiny_range_is_exact():
    # y = 2x + 1 on [0, 1e-3]: a first-order Sugeno with p_r = 2, r_r = 1 is exact.
    # With the default ridge 1e-8 the residual is the ridge bias, the same size as on [0, 1]
    # (the brief's 1e-6 bar); without a ridge it is rounding only.
    X = [[i * 1e-4] for i in range(11)]
    y = [2 * x[0] + 1 for x in X]
    assert ANFIS(1, 2).fit(X, y, epochs=1)[0] < 1e-6
    assert ANFIS(1, 2).fit(X, y, epochs=1, ridge=0)[0] < 1e-12


def test_constant_input_without_ridge_does_not_crash():
    # every row is the same: the LSE design has rank 1, so ridge=0 makes A^T A singular;
    # any solution of the consistent system predicts y = 1 exactly
    a = ANFIS(1, 2, ranges=[(0, 1)])
    h = a.fit([[0.5]] * 5, [1.0] * 5, epochs=2, ridge=0)
    assert h[-1] == approx(0.0, abs=1e-6)
    assert a.predict([[0.5]]) == approx([1.0], abs=1e-6)


def test_degenerate_input_range_needs_ranges():
    with pytest.raises(ValueError, match="ranges"):
        ANFIS(1, 2).fit([[1.0], [1.0]], [0.0, 1.0])


def test_no_rule_fires_raises():
    # a centre 1e300 away with sigma 1e-3: z^2 overflows, every log-membership is -inf
    a = ANFIS(1, 1, ranges=[(0, 1)])
    a.set_params([1e300, 1e-3, 0, 0])
    with pytest.raises(ValueError, match="no rule fires"):
        a.predict([[0.0]])
    with pytest.raises(ValueError, match="no rule fires"):
        a.rmse([[0.0]], [0.0])


def test_construction_errors():
    with pytest.raises(ValueError, match="n_inputs"):
        ANFIS(0)
    with pytest.raises(ValueError, match="n_mfs"):
        ANFIS(2, [2])
    with pytest.raises(ValueError, match="n_mfs"):
        ANFIS(1, 0)
    with pytest.raises(ValueError, match="ranges"):
        ANFIS(1, 2, ranges=[(1, 1)])
    with pytest.raises(ValueError, match="ranges"):
        ANFIS(2, 2, ranges=[(0, 1)])


def test_fit_and_data_errors():
    a = ANFIS(1, 2, ranges=[(0, 1)])
    with pytest.raises(ValueError, match="valid: hybrid, gradient"):
        a.fit([[0.0]], [0.0], method="newton")
    with pytest.raises(ValueError, match="epochs"):
        a.fit([[0.0]], [0.0], epochs=0)
    with pytest.raises(ValueError, match="lr"):
        a.fit([[0.0]], [0.0], lr=0)
    with pytest.raises(ValueError, match="ridge"):
        a.fit([[0.0]], [0.0], ridge=-1)
    with pytest.raises(ValueError, match="at least one sample"):
        a.predict([])
    with pytest.raises(ValueError, match="1 input"):
        a.predict([[0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        a.predict([[float("nan")]])
    with pytest.raises(ValueError, match="targets"):
        a.rmse([[0.0]], [1.0, 2.0])


def test_set_params_errors():
    a = ANFIS(1, 2, ranges=[(0, 1)])
    with pytest.raises(ValueError, match="expected 8"):
        a.set_params([0.0] * 7)
    with pytest.raises(ValueError, match="sigma"):
        a.set_params([0, 0, 1, 0.5, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        a.set_params([0, 0.5, 1, 0.5, 0, 0, 0, float("inf")])
    b = ANFIS(1, 1, mf="bell", ranges=[(0, 1)])
    with pytest.raises(ValueError, match="b"):
        b.set_params([1, -1, 0, 0, 0])


def test_set_params_rejects_bools_and_enforces_floors():
    a = ANFIS(1, 2, ranges=[(0, 10)])  # span 10: width floor MIN_WIDTH * 10 = 0.01
    good = a.get_params()
    with pytest.raises(ValueError, match="bool"):
        a.set_params([True] + good[1:])
    low = list(good)
    low[1] = 0.005  # sigma of input 0 MF 0 below 0.01
    with pytest.raises(ValueError, match="sigma of input 0 MF 0"):
        a.set_params(low)
    at = list(good)
    at[1] = MIN_WIDTH * 10  # exactly at the floor is allowed
    a.set_params(at)
    b = ANFIS(1, 1, mf="bell", ranges=[(0, 1)])
    with pytest.raises(ValueError, match="b of input 0 MF 0"):
        b.set_params([1, MIN_B / 2, 0, 0, 0])
    with pytest.raises(ValueError, match="a of input 0 MF 0"):
        b.set_params([MIN_WIDTH / 2, 2, 0, 0, 0])


def test_typed_properties():
    a = ANFIS(2, [2, 3], mf="bell")
    assert (a.n_inputs, a.n_rules, a.mf) == (2, 6, "bell")
    assert ANFIS.n_rules.__doc__ and ANFIS.n_inputs.__doc__ and ANFIS.mf.__doc__


def test_rmse_raises_on_overflow():
    # constant 1e300 in both rules: the error is finite but its square overflows
    a = ANFIS(1, 2, ranges=[(0, 1)])
    a.set_params([0, 0.5, 1, 0.5, 0, 1e300, 0, 1e300])
    assert a.predict([[0.5]]) == approx([1e300])
    with pytest.raises(ValueError, match="not finite"):
        a.rmse([[0.5]], [0.0])


def test_private_rmse_helper_returns_inf_as_penalty():
    # the internal (and Task 13 penalty) variant returns inf where rmse() raises
    a = ANFIS(1, 1, ranges=[(0, 1)])
    a.set_params([1e300, 1e-3, 0, 0])  # no rule fires
    assert a._rmse_or_inf([[0.0]], [0.0]) == math.inf
    b = ANFIS(1, 2, ranges=[(0, 1)])
    b.set_params([0, 0.5, 1, 0.5, 0, 1e300, 0, 1e300])  # squared error overflows
    assert b._rmse_or_inf([[0.5]], [0.0]) == math.inf
    # 1e154^2 = 1e308 twice: finite terms whose fsum overflows
    b.set_params([0, 0.5, 1, 0.5, 0, 1e154, 0, 1e154])
    assert b._rmse_or_inf([[0.5], [0.5]], [0.0, 0.0]) == math.inf


def test_predict_warns_once_outside_the_trained_ranges():
    a = ANFIS(2, 2, ranges=[(0, 1), (0, 10)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning for in-range inputs, boundary included
        inside = a.predict([[0.0, 0.0], [1.0, 10.0], [0.5, 5.0]])
        a.predict([[0.2, 3.0]])
    assert len(inside) == 3
    with pytest.warns(ExtrapolationWarning, match=r"sample 1 has input 1 = 12 .*\[0, 10\]") as rec:
        out = a.predict([[0.5, 5.0], [0.5, 12.0], [2.0, 5.0]])
    assert len(rec) == 1 and issubclass(ExtrapolationWarning, RuntimeWarning)
    assert len(out) == 3  # still predicts
    with pytest.warns(ExtrapolationWarning, match="input 0 = -0.1"):
        a.predict([[-0.1, 5.0]])


def test_predict_after_fit_warns_outside_data_range_only():
    a = ANFIS(1, 2)
    a.fit(SIN_X, SIN_Y, epochs=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a.predict([[SIN_X[0][0]], [SIN_X[-1][0]]])
    with pytest.warns(ExtrapolationWarning):
        a.predict([[SIN_X[-1][0] + 1.0]])


def test_predict_ignores_one_ulp_noise_at_the_range_edge():
    a = ANFIS(1, 2, ranges=[(0, 0.3)])
    assert 0.1 * 3 > 0.3  # 0.30000000000000004: one ulp past the edge
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a.predict([[0.1 * 3]])
    with pytest.warns(ExtrapolationWarning, match=r"0\.31 .*\[0, 0\.3\]"):
        a.predict([[0.31]])  # a real excursion still warns, in round-trip format
    with pytest.warns(ExtrapolationWarning, match=r"\[0, 0\.29999999899999996\]"):
        ANFIS(1, 2, ranges=[(0, 0.3 - 1e-9)]).predict([[0.3]])


def test_rmse_warns_on_extrapolation_but_private_helper_is_silent():
    a = ANFIS(1, 2, ranges=[(0, 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a.rmse([[0.5]], [0.0])
        a._rmse_or_inf([[5.0]], [0.0])  # sezgi's DE objective stays silent
    with pytest.warns(ExtrapolationWarning, match="sample 0 has input 0 = 5"):
        a.rmse([[5.0]], [0.0])


# ------------------------------------------------------- training never gets worse


def _offset_sine():
    """A target far larger than its variation: the gradient is huge at the default lr."""
    X = [[0.05 * i / 199] for i in range(200)]
    return X, [125 + 5 * math.sin(120 * x[0]) for x in X]


def _rises(history, slack=1e-9):
    return [(a, b) for a, b in zip(history, history[1:]) if b > a * (1 + slack)]


@pytest.mark.parametrize("method", ["hybrid", "gradient"])
@pytest.mark.parametrize("mf", ["gaussian", "bell"])
def test_history_never_rises_on_a_badly_scaled_target(method, mf):
    X, y = _offset_sine()
    history = ANFIS(1, 2, mf=mf).fit(X, y, epochs=10, method=method)
    assert not _rises(history)


def test_default_lr_does_not_lose_to_consequents_alone():
    X, y = _offset_sine()
    premise_kept = ANFIS(1, 2).fit(X, y, epochs=1, lr=1e-12)[-1]
    assert ANFIS(1, 2).fit(X, y, epochs=10)[-1] <= premise_kept


def test_hybrid_ends_with_least_squares_consequents():
    X, y = _offset_sine()
    model = ANFIS(1, 2)
    model.fit(X, y, epochs=3)
    fitted = model.rmse(X, y)
    model.fit(X, y, epochs=1, lr=1e-300)  # re-solves the consequents, premise stays put
    assert model.rmse(X, y) == pytest.approx(fitted, rel=1e-9)


@pytest.mark.parametrize("method", ["hybrid", "gradient"])
def test_history_ends_at_the_rmse_of_the_returned_model(method):
    X, y = _offset_sine()
    model = ANFIS(1, 2)
    history = model.fit(X, y, epochs=5, method=method)
    assert history[-1] == pytest.approx(model.rmse(X, y), rel=1e-12)


@pytest.mark.parametrize("method", ["hybrid", "gradient"])
def test_continuing_a_fit_does_not_undo_it(method):
    X, y = _offset_sine()
    model = ANFIS(1, 2)
    before = model.fit(X, y, epochs=5, method=method)[-1]
    assert model.fit(X, y, epochs=5, method=method)[-1] <= before * (1 + 1e-9)


def test_converged_fit_does_not_warn(recwarn):
    X = [[i / 10] for i in range(21)]
    y = [2 * x[0] + 1 for x in X]  # exactly representable: the gradient is about zero
    ANFIS(1, 2, ranges=[(0, 2)]).fit(X, y, epochs=5)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_an_accepted_step_size_carries_to_the_next_epoch(monkeypatch):
    X, y = _offset_sine()
    model = ANFIS(1, 2)
    evals = []
    real = model._rmse_or_inf
    monkeypatch.setattr(model, "_rmse_or_inf", lambda *a: evals.append(1) or real(*a))
    model.fit(X, y, epochs=40)
    assert len(evals) < 40 * 5  # not one search down from lr per epoch


def test_a_step_too_large_to_descend_warns():
    X = [[i / 10] for i in range(21)]
    y = [x[0] ** 2 for x in X]
    model = ANFIS(1, 3, ranges=[(0, 2)])
    with pytest.warns(RuntimeWarning, match="skipped 4 of 4"):
        history = model.fit(X, y, epochs=4, method="gradient", lr=1e300)
    assert len(set(history)) == 1
