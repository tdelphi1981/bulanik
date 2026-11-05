import pytest
from hypothesis import given
from pytest import approx

from bulanik import IFN, TFN, TrFN
from strategies import tfns, trfns


def test_construct_validates():
    with pytest.raises(ValueError):
        TFN(2, 1, 3)
    with pytest.raises(ValueError):
        TFN(0, float("inf"), 1)
    with pytest.raises(ValueError):
        TrFN(1, 3, 2, 4)


def test_repr_and_float_storage():
    assert repr(TFN(1, 2, 3)) == "TFN(l=1.0, m=2.0, u=3.0)"
    assert repr(TrFN(1, 2, 3, 4)) == "TrFN(a=1.0, b=2.0, c=3.0, d=4.0)"


def test_arithmetic_golden():
    A, B = TFN(1, 2, 3), TFN(2, 3, 5)
    assert A + B == TFN(3, 5, 8)
    assert A - B == TFN(-4, -1, 1)  # (1-5, 2-3, 3-2)
    assert A * B == TFN(2, 6, 15)
    q = A / B  # (1/5, 2/3, 3/2)
    assert (q.l, q.m, q.u) == approx((0.2, 2 / 3, 1.5))
    assert 2 * A == TFN(2, 4, 6) and -1 * A == TFN(-3, -2, -1)
    assert A * 2 == TFN(2, 4, 6) and -A == TFN(-3, -2, -1)
    assert A**2 == TFN(1, 4, 9)
    assert A / 2 == TFN(0.5, 1, 1.5)


def test_trapezoid_arithmetic_golden():
    A, B = TrFN(1, 2, 3, 4), TrFN(2, 3, 4, 6)
    assert A + B == TrFN(3, 5, 7, 10)
    assert A - B == TrFN(-5, -2, 0, 2)  # (1-6, 2-4, 3-3, 4-2)
    assert A * B == TrFN(2, 6, 12, 24)  # corners 2,6,8,24; inner 2*3, 3*4
    assert A**2 == TrFN(1, 4, 9, 16)
    assert -A == TrFN(-4, -3, -2, -1)
    assert TrFN(1, 2, 4, 8).reciprocal() == TrFN(0.125, 0.25, 0.5, 1.0)


def test_trapezoid_product_keeps_inner_points_ordered():
    # inner products 3 and 2 would be out of order; they are sorted
    A, B = TrFN(-2, -1, 1, 2), TrFN(-4, -3, 1, 2)
    # corners 8,-4,-8,4 -> (-8, 8); inner (-1)(-3)=3 and 1*1=1, sorted to (1, 3)
    assert A * B == TrFN(-8, 1, 3, 8)


def test_mixed_sign_product():
    assert TFN(-1, 1, 2) * TFN(2, 3, 4) == TFN(-4, 3, 8)  # P={-2,-4,4,8}


def test_division_by_interval_with_zero_raises():
    with pytest.raises(ValueError, match="0"):
        TFN(1, 2, 3) / TFN(-1, 1, 2)


def test_alpha_cut_and_mu():
    A = TFN(1, 2, 3)
    assert A.alpha_cut(0.5) == approx((1.5, 2.5))
    assert A.alpha_cut(0) == (1.0, 3.0) and A.alpha_cut(1) == (2.0, 2.0)
    assert A.mu(1.5) == approx(0.5)
    assert A.mu(2) == 1 and A.mu(0) == 0 and A.mu(2.5) == approx(0.5)
    assert TFN(2, 2, 4).mu(2) == 1.0  # vertical left edge


def test_trapezoid_alpha_cut_and_mu():
    C = TrFN(1, 2, 4, 7)
    assert C.alpha_cut(0.5) == approx((1.5, 5.5))  # (1+.5*1, 7-.5*3)
    assert C.mu(3) == 1.0 and C.mu(1.5) == approx(0.5) and C.mu(5.5) == approx(0.5)
    assert C.mu(8) == 0.0
    with pytest.raises(ValueError):
        C.alpha_cut(1.5)


def test_defuzzify_golden():
    B = TFN(2, 3, 5)
    assert B.defuzzify() == approx(10 / 3)  # centroid (2+3+5)/3
    assert B.defuzzify("graded_mean") == approx(19 / 6)  # (2+12+5)/6
    assert B.defuzzify("mean") == approx(10 / 3)
    C = TrFN(1, 2, 4, 7)
    assert C.defuzzify() == approx(86 / 24)  # (49+16+28-1-4-2)/(3*(7+4-1-2))
    assert C.defuzzify("mean") == approx(3.5)
    assert C.defuzzify("graded_mean") == approx(20 / 6)  # (1+4+8+7)/6
    assert TrFN(1, 2, 2, 3).defuzzify() == approx(2.0)  # TFN-shaped trapezoid
    assert TrFN(4, 4, 4, 4).defuzzify() == 4.0


def test_score_is_defuzzify_and_names_checked():
    assert TFN(2, 3, 5).score("graded_mean") == approx(19 / 6)
    assert TFN(2, 3, 5).score() == approx(10 / 3)
    with pytest.raises(ValueError, match="valid: centroid, graded_mean, mean"):
        TFN(1, 2, 3).score("x")
    with pytest.raises(ValueError, match="valid: centroid, graded_mean, mean"):
        TFN(1, 2, 3).defuzzify("x")
    with pytest.raises(ValueError, match="valid: algebraic"):
        TFN(1, 2, 3).add(TFN(1, 2, 3), ops="einstein")
    with pytest.raises(ValueError, match="no accuracy"):
        TFN(1, 2, 3).accuracy()


def test_vertex_distance():
    assert TFN(1, 2, 3).distance(TFN(2, 3, 5)) == approx(2**0.5)  # sqrt((1+1+4)/3)
    assert TrFN(1, 2, 3, 4).distance(TrFN(1, 2, 3, 6)) == approx(1.0)  # sqrt(4/4)
    with pytest.raises(ValueError, match="valid: vertex"):
        TFN(1, 2, 3).distance(TFN(1, 2, 3), "hamming")


def test_reciprocal():
    assert TFN(1, 2, 4).reciprocal() == TFN(0.25, 0.5, 1.0)


def test_power_needs_nonnegative():
    with pytest.raises(ValueError):
        TFN(-1, 1, 2) ** 0.5
    with pytest.raises(ValueError):
        TFN(0, 1, 2) ** -1
    assert TFN(1, 2, 4) ** -1 == TFN(0.25, 0.5, 1.0)


def test_to_trfn():
    assert TFN(1, 2, 3).to_trfn() == TrFN(1, 2, 2, 3)


def test_mixed_types_type_error():
    with pytest.raises(TypeError):
        TFN(1, 2, 3) + TrFN(1, 2, 3, 4)
    with pytest.raises(TypeError):
        TFN(1, 2, 3).add(TrFN(1, 2, 3, 4))
    with pytest.raises(TypeError):
        TFN(1, 2, 3) * "x"


@given(tfns(), tfns())
def test_add_commutative_and_valid(a, b):
    assert (a + b) == (b + a)


def test_division_by_zero_scalar_raises_value_error():
    with pytest.raises(ValueError, match="zero"):
        TFN(1, 2, 3) / 0
    with pytest.raises(ValueError, match="zero"):
        TrFN(1, 2, 3, 4) / 0.0
    assert TFN(2, 4, 6) / 2 == TFN(1, 2, 3)


def test_same_kind_error_names_arbitrary_objects():
    with pytest.raises(TypeError, match="cannot combine TFN with str"):
        TFN(1, 2, 3).same_kind("x")
    with pytest.raises(TypeError, match=r"cannot combine IFN\(q=1\) with TFN"):
        IFN(0.5, 0.2).same_kind(TFN(1, 2, 3))
    with pytest.raises(TypeError, match="cannot combine TFN with NoneType"):
        TFN(1, 2, 3).add(None)


def _corner_products(a, b):
    return [x * y for x in (a[0], a[-1]) for y in (b[0], b[-1])]


@given(trfns(), trfns())
def test_trfn_mul_matches_corner_products_and_commutes(a, b):
    r = a * b
    corners = _corner_products(a._pts(), b._pts())
    assert (r.a, r.d) == (min(corners), max(corners))  # independent of the code path
    assert sorted((r.b, r.c)) == sorted(x * y for x, y in zip(a._pts()[1:3], b._pts()[1:3]))
    assert r == b * a


@given(tfns(), tfns())
def test_signed_tfn_mul_sub_closure_and_membership(a, b):
    # mixed-sign supports: results stay valid ordered TFNs, and every product of
    # members lies inside the result's support (interval arithmetic enclosure)
    for r in (a * b, a - b):
        assert r.l <= r.m <= r.u
    p, d = a * b, a - b
    for x in (a.l, a.m, a.u):
        for y in (b.l, b.m, b.u):
            assert p.l - 1e-9 <= x * y <= p.u + 1e-9
            assert d.l - 1e-9 <= x - y <= d.u + 1e-9
    assert (a - a).l <= 0.0 <= (a - a).u  # a - a contains 0 (no cancellation)


@given(tfns(), tfns(positive=True))
def test_signed_tfn_division_closure(a, b):
    q = a / b  # divisor strictly positive: always defined
    assert q.l <= q.m <= q.u
    for x in (a.l, a.m, a.u):
        for y in (b.l, b.m, b.u):
            assert q.l - 1e-9 <= x / y <= q.u + 1e-9


@given(tfns(positive=True))
def test_alpha_cut_nested(a):
    lo1, hi1 = a.alpha_cut(0.2)
    lo2, hi2 = a.alpha_cut(0.8)
    assert lo1 <= lo2 <= hi2 <= hi1
