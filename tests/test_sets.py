import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pytest import approx

from bulanik.sets import (
    SIGMOID_EPS,
    LinguisticVariable,
    MembershipFunction,
    bell,
    gaussian,
    linear_decreasing,
    linear_increasing,
    sigmoid,
    t_conorm,
    t_norm,
    trapezoidal,
    triangular,
)


def test_membership_golden():
    # triangular(0,5,10) at 2.5: (2.5-0)/(5-0) = 0.5; left shoulder mu(0) = 1
    assert triangular(0, 5, 10)(2.5) == approx(0.5) and triangular(0, 0, 5)(0) == 1.0
    # trapezoid plateau [2,4]; at 5 falling edge (6-5)/(6-4) = 0.5
    assert trapezoidal(0, 2, 4, 6)(3) == 1.0 and trapezoidal(0, 2, 4, 6)(5) == approx(0.5)
    # gaussian at one sigma-ish: exp(-0.5 * (2/2)^2)
    assert gaussian(0, 2)(2) == approx(math.exp(-0.5))
    assert bell(2, 1, 0)(2) == approx(0.5)  # 1/(1+|1|^2)
    assert sigmoid(1, 0)(0) == approx(0.5)
    # (3-2)/(6-2) = 0.25 and 1 - 0.25
    assert linear_increasing(2, 6)(3) == approx(0.25)
    assert linear_decreasing(2, 6)(3) == approx(0.75)


def test_shoulders_and_ends():
    assert triangular(0, 5, 5)(5) == 1.0 and triangular(0, 5, 5)(6) == 0.0
    assert trapezoidal(0, 0, 2, 4)(0) == 1.0 and trapezoidal(0, 2, 4, 4)(4) == 1.0
    assert linear_increasing(2, 6)(2) == 0.0 and linear_increasing(2, 6)(6) == 1.0
    assert linear_decreasing(2, 6)(2) == 1.0 and linear_decreasing(2, 6)(6) == 0.0
    assert triangular(0, 5, 10)(-1) == 0.0 and triangular(0, 5, 10)(11) == 0.0


def test_inverse():
    # 40 + 0.3*(100-40) = 58 ; 60 - 0.7*(60-20) = 32 ; sigmoid midpoint = center
    assert linear_increasing(40, 100).inverse(0.3) == approx(58)
    assert linear_decreasing(20, 60).inverse(0.7) == approx(32)
    assert sigmoid(2, 1).inverse(0.5) == approx(1.0)
    with pytest.raises(ValueError, match="monotonic"):
        triangular(0, 1, 2).inverse(0.5)


def test_inverse_exact_ends_and_range():
    assert linear_increasing(0.1, 0.7).inverse(0.0) == 0.1
    assert linear_increasing(0.1, 0.7).inverse(1.0) == 0.7
    assert linear_decreasing(0.1, 0.7).inverse(1.0) == 0.1
    assert linear_decreasing(0.1, 0.7).inverse(0.0) == 0.7
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            linear_increasing(0, 1).inverse(bad)
    assert SIGMOID_EPS == 1e-9
    s = sigmoid(1, 0)
    # alpha=0 clamps to eps: x = -ln(1/eps - 1)
    assert s.inverse(0.0) == approx(-math.log(1 / SIGMOID_EPS - 1))
    assert s.inverse(1.0) == approx(math.log(1 / SIGMOID_EPS - 1))
    assert math.isfinite(sigmoid(-3, 2).inverse(0.0))


def test_monotonic_flags():
    assert linear_increasing(0, 1).is_monotonic and linear_decreasing(0, 1).is_monotonic
    assert sigmoid(1, 0).is_monotonic
    for mf in (triangular(0, 1, 2), trapezoidal(0, 1, 2, 3), gaussian(0, 1), bell(1, 1, 0)):
        assert not mf.is_monotonic


def test_factory_validation():
    bad = (
        lambda: triangular(3, 2, 1),
        lambda: triangular(1, 1, 1),
        lambda: trapezoidal(0, 3, 2, 4),
        lambda: gaussian(0, 0),
        lambda: bell(0, 1, 0),
        lambda: bell(1, 0, 0),
        lambda: linear_increasing(2, 2),
        lambda: linear_decreasing(3, 2),
        lambda: sigmoid(0, 1),
        lambda: gaussian(float("nan"), 1),
        lambda: MembershipFunction("nope", (1.0,)),
    )
    for b in bad:
        with pytest.raises(ValueError):
            b()
    with pytest.raises(ValueError, match="nan"):
        triangular(0, 1, 2)(float("nan"))


def test_frozen_and_repr_fields():
    mf = triangular(0, 1, 2)
    assert mf.kind == "triangular" and mf.params == (0.0, 1.0, 2.0)
    with pytest.raises(Exception):
        mf.kind = "x"  # type: ignore[misc]


def test_extreme_arguments_stay_in_unit_interval():
    for mf in (gaussian(0, 1e-3), bell(1e-3, 50, 0), sigmoid(1e6, 0), sigmoid(-1e6, 0)):
        for x in (-1e300, -1e6, 1e6, 1e300, float("inf"), float("-inf"), 0.0):
            assert 0.0 <= mf(x) <= 1.0
    assert gaussian(0, 1)(1e300) == 0.0
    assert sigmoid(1, 0)(-1e6) == approx(0.0, abs=1e-300)
    assert sigmoid(1, 0)(1e6) == 1.0


@given(
    st.floats(-1e6, 1e6),
    st.floats(-1e3, 1e3),
    st.floats(1e-3, 1e3),
    st.floats(0.1, 20),
)
def test_all_kinds_in_unit_interval(x, c, w, b):
    for mf in (
        gaussian(c, w),
        bell(w, b, c),
        sigmoid(w, c),
        sigmoid(-w, c),
        triangular(c - w, c, c + w),
        trapezoidal(c - 2 * w, c - w, c + w, c + 2 * w),
        linear_increasing(c, c + w),
        linear_decreasing(c, c + w),
    ):
        assert 0.0 <= mf(x) <= 1.0


@given(st.floats(-100, 100), st.floats(0.01, 100), st.floats(0, 1))
def test_linear_inverse_roundtrip(a, w, alpha):
    # mu(inverse(alpha)) == alpha for the linear monotone kinds
    for mf in (linear_increasing(a, a + w), linear_decreasing(a, a + w)):
        assert mf(mf.inverse(alpha)) == approx(alpha, abs=1e-9)


@given(st.floats(-5, 5), st.floats(0.1, 5), st.floats(0.001, 0.999))
def test_sigmoid_inverse_roundtrip(c, k, alpha):
    for s in (sigmoid(k, c), sigmoid(-k, c)):
        assert s(s.inverse(alpha)) == approx(alpha, abs=1e-7)


def test_linguistic_variable():
    service = LinguisticVariable(
        "service",
        (0, 10),
        {"poor": triangular(0, 0, 5), "average": triangular(0, 5, 10), "good": triangular(5, 10, 10)},
    )
    # at 3: poor (5-3)/5 = 0.4, average 3/5 = 0.6, good 0
    assert service.fuzzify(3) == approx({"poor": 0.4, "average": 0.6, "good": 0.0})
    with pytest.raises(ValueError, match=r"service.*\[0(\.0)?, 10(\.0)?\]"):
        service.fuzzify(11)
    with pytest.raises(ValueError):
        service.fuzzify(-0.5)
    assert service.term_names == ("poor", "average", "good")
    assert service["poor"](0) == 1.0
    with pytest.raises(KeyError, match="poor"):
        service["bad"]
    assert service.fuzzify(0)["poor"] == 1.0 and service.fuzzify(10)["good"] == 1.0


def test_linguistic_variable_validation():
    t = {"a": triangular(0, 1, 2)}
    with pytest.raises(ValueError):
        LinguisticVariable("x", (5, 5), t)
    with pytest.raises(ValueError):
        LinguisticVariable("x", (0, 1), {})
    with pytest.raises(ValueError):
        LinguisticVariable("x", (0, float("inf")), t)
    with pytest.raises(ValueError, match="finite"):
        LinguisticVariable("x", (0, 1), t).fuzzify(float("nan"))


def test_norms():
    assert t_norm(0.4, 0.6) == 0.4 and t_norm(0.4, 0.6, "prod") == approx(0.24)
    assert t_conorm(0.4, 0.6) == 0.6 and t_conorm(0.4, 0.6, "probor") == approx(0.76)
    with pytest.raises(ValueError, match="valid: min, prod"):
        t_norm(0.1, 0.2, "lukasiewicz")
    with pytest.raises(ValueError, match="valid: max, probor"):
        t_conorm(0.1, 0.2, "x")


@given(st.floats(0, 1), st.floats(0, 1))
def test_norm_laws(a, b):
    for k in ("min", "prod"):
        assert t_norm(a, 1, k) == approx(a) and t_norm(a, b, k) == approx(t_norm(b, a, k))
        assert t_norm(a, 0, k) == 0.0
    for k in ("max", "probor"):
        assert t_conorm(a, 0, k) == approx(a) and t_conorm(a, b, k) == approx(t_conorm(b, a, k))
        assert 0.0 <= t_conorm(a, b, k) <= 1.0


def test_reprs():
    assert "service" in repr(LinguisticVariable("service", (0, 1), {"a": triangular(0, 0.5, 1)}))


def test_direct_construction_validates_domains():
    bad = [
        ("gaussian", (0, 0)),
        ("gaussian", (0, -1)),
        ("bell", (0, 1, 0)),
        ("bell", (-1, 1, 0)),
        ("bell", (1, 0, 0)),
        ("triangular", (3, 2, 1)),
        ("triangular", (1, 1, 1)),
        ("trapezoidal", (0, 3, 2, 4)),
        ("linear_increasing", (2, 2)),
        ("linear_decreasing", (3, 2)),
        ("sigmoid", (0, 1)),
    ]
    for kind, params in bad:
        with pytest.raises(ValueError, match=kind):
            MembershipFunction(kind, params)
    # valid direct construction still works, params normalized to floats
    assert MembershipFunction("gaussian", (0, 2))(2) == approx(math.exp(-0.5))


def test_mf_call_and_inverse_reject_str_and_bool():
    mf = triangular(0, 5, 10)
    with pytest.raises(TypeError):
        mf("5")
    with pytest.raises(TypeError):
        mf(True)
    assert mf(2) == approx(0.4) and mf(math.inf) == 0.0  # ints and infinities still fine
    lin = linear_increasing(0, 10)
    with pytest.raises(TypeError):
        lin.inverse("0.5")
    with pytest.raises(TypeError):
        lin.inverse(False)
    assert lin.inverse(0.5) == approx(5.0)


def test_fuzzify_message_keeps_full_precision():
    v = LinguisticVariable("t", (0.0, 1.0), {"a": triangular(0, 0.5, 1)})
    with pytest.raises(ValueError, match=r"1\.0000001 is outside its universe \[0, 1\]"):
        v.fuzzify(1.0000001)


def test_linear_inverse_at_huge_span_has_no_nan():
    big = 1.7e308
    inc, dec = linear_increasing(-big, big), linear_decreasing(-big, big)
    assert inc.inverse(0.0) == -big and inc.inverse(1.0) == big
    assert dec.inverse(0.0) == big and dec.inverse(1.0) == -big
    for alpha in (0.25, 0.5, 0.75):
        assert math.isfinite(inc.inverse(alpha)) and math.isfinite(dec.inverse(alpha))
    assert inc.inverse(0.5) == approx(0.0, abs=1e295)
    assert linear_increasing(2, 6).inverse(0.25) == approx(3.0)
    assert linear_decreasing(2, 6).inverse(0.25) == approx(5.0)


def test_linear_inverse_stays_inside_its_support_despite_rounding():
    # found by search: a(1-alpha) + b alpha rounds to one ulp below a (increasing)
    a, b, alpha = -0.027149378239326283, -0.02714937823931729, 7.323477044226232e-07
    assert a * (1.0 - alpha) + b * alpha < a  # the raw formula leaves [a, b]
    assert linear_increasing(a, b).inverse(alpha) == a
    assert linear_decreasing(a, b).inverse(1.0 - alpha) == a  # mirrored: b(1-alpha') + a alpha'
    c, d, beta = 0.00020972722915324748, 0.00020972722915325658, 2.4973329281585343e-07
    for f in (linear_increasing(c, d), linear_decreasing(c, d)):
        assert c <= f.inverse(beta) <= d


def test_sigmoid_inverse_pinned_at_clamp():
    # closed form: x = center - ln(1/alpha - 1) / slope at alpha = 1e-9 and 1 - 1e-9
    # (ln(1e9 - 1) / 2 = 10.361632917973205), hard-coded rather than recomputed
    s = sigmoid(2.0, 1.0)
    lo, hi = -9.361632917973205, 11.361632917973205
    assert s.inverse(0.0) == approx(lo, abs=1e-9) == s.inverse(1e-10)
    # 1 - 1e-9 is not exact in binary (error ~1e-7 relative to 1e-9), hence abs=1e-6 on the upper end
    assert s.inverse(1.0) == approx(hi, abs=1e-6) == s.inverse(1.0 - 1e-10)
    assert s(s.inverse(0.0)) == approx(1e-9, rel=1e-6)
    assert sigmoid(-2.0, 1.0).inverse(0.0) == approx(hi, abs=1e-6)


def test_call_accepts_huge_ints():
    huge = 10**400
    t = triangular(0, 1, 2)
    assert t(huge) == 0.0 and t(-huge) == 0.0
    assert linear_increasing(0, 1)(huge) == 1.0 and linear_increasing(0, 1)(-huge) == 0.0
    assert sigmoid(1.0, 0.0)(huge) == 1.0 and sigmoid(1.0, 0.0)(-huge) == 0.0


def test_norms_validate_inputs():
    for fn in (t_norm, t_conorm):
        for bad in (float("nan"), 1.5, -0.5, float("inf")):
            with pytest.raises(ValueError):
                fn(bad, 0.5)
            with pytest.raises(ValueError):
                fn(0.5, bad)
        with pytest.raises(TypeError):
            fn(True, 0.5)
    assert t_norm(1 + 1e-13, 0.5) == 0.5  # within TOL is clamped


def test_linguistic_variable_type_checks():
    mf = triangular(0, 1, 2)
    with pytest.raises(TypeError, match="term 'bad'.*MembershipFunction"):
        LinguisticVariable("v", (0, 2), {"ok": mf, "bad": 3.0})
    with pytest.raises(TypeError, match="universe of 'v'.*pair"):
        LinguisticVariable("v", 5, {"ok": mf})
    with pytest.raises(TypeError, match="universe of 'v'.*pair"):
        LinguisticVariable("v", (0, 1, 2), {"ok": mf})
    with pytest.raises(TypeError, match="universe of 'v'.*pair"):
        LinguisticVariable("v", None, {"ok": mf})


def test_membership_function_error_names_kind_and_parameter():
    with pytest.raises(ValueError, match="gaussian parameter 2 must be finite"):
        MembershipFunction("gaussian", (0.0, float("inf")))
    with pytest.raises(ValueError, match="triangular parameter 1"):
        MembershipFunction("triangular", (float("nan"), 1.0, 2.0))
    with pytest.raises(TypeError, match="gaussian params must be a sequence"):
        MembershipFunction("gaussian", None)
    with pytest.raises(TypeError, match="params must be a sequence"):
        MembershipFunction("gaussian", "ab")
