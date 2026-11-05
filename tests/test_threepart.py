import math
from decimal import Decimal, localcontext

import pytest
from hypothesis import assume, given, strategies as st
from pytest import approx

from bulanik import NeutrosophicFN, PictureFN, SphericalFN
from strategies import neutrosophics, pictures, sphericals


def test_validation():
    with pytest.raises(ValueError):
        SphericalFN(0.8, 0.5, 0.4)  # .64+.25+.16 = 1.05
    with pytest.raises(ValueError):
        PictureFN(0.5, 0.3, 0.3)
    with pytest.raises(ValueError):
        NeutrosophicFN(1.2, 0, 0)
    with pytest.raises(ValueError):
        PictureFN(float("nan"), 0, 0)
    assert NeutrosophicFN(1, 1, 1).t == 1  # no sum constraint


def test_boundary_float_noise_accepted():
    assert SphericalFN(0.6, 0.8, 0.0).mu == 0.6  # 0.36+0.64 = 1.0000000000000002
    assert PictureFN(0.7, 0.2, 0.1).nu == 0.1  # 0.7+0.2+0.1 = 0.9999999999999999+


def test_repr_frozen_eq():
    assert repr(PictureFN(0.5, 0.2, 0.2)) == "PictureFN(mu=0.5, eta=0.2, nu=0.2)"
    assert repr(NeutrosophicFN(0.7, 0.2, 0.1)) == "NeutrosophicFN(t=0.7, i=0.2, f=0.1)"
    with pytest.raises(AttributeError):
        SphericalFN(0.6, 0.3, 0.4).mu = 0.1
    assert SphericalFN(0.6, 0.3, 0.4) == SphericalFN(0.6, 0.3, 0.4)
    assert len({PictureFN(0.5, 0.2, 0.2), PictureFN(0.5, 0.2, 0.2)}) == 1
    assert SphericalFN(0.1, 0.1, 0.1).pi == 0.1 and PictureFN(0.1, 0.2, 0.3).eta == 0.2
    n = NeutrosophicFN(0.7, 0.2, 0.1)
    assert (n.t, n.i, n.f) == n.parts


def test_spherical_golden():  # a=(.6,.3,.4) b=(.5,.5,.2)
    a, b = SphericalFN(0.6, 0.3, 0.4), SphericalFN(0.5, 0.5, 0.2)
    assert (a + b).parts == approx((0.7211102551, 0.15, 0.3730951621))
    # mu=sqrt(.36+.25-.09); pi=sqrt(.75*.16+.64*.04-.0064)=sqrt(.1392)
    assert (a * b).parts == approx((0.3, 0.5634713835, 0.3872983346))
    assert (2 * a).parts == approx((0.7683749085, 0.09, 0.4233202098))  # pi=sqrt(.64^2-.48^2)
    assert (a**2).parts == approx((0.36, 0.4146082488, 0.5153639491))
    assert a.score() == approx(0.03)  # (.6-.4)^2-(.3-.4)^2 = .04-.01
    assert a.score("normalized") == approx(0.515)
    assert a.accuracy() == approx(0.61)
    assert a.distance(b) == approx(0.23)  # (|.36-.25|+|.09-.25|+|.16-.04|+|.39-.46|)/2
    assert a.distance(b, "euclidean") == approx(0.1688194302)
    assert a.complement().parts == (0.3, 0.6, 0.4)


def test_picture_golden():  # a=(.5,.2,.2) b=(.3,.3,.3)
    a, b = PictureFN(0.5, 0.2, 0.2), PictureFN(0.3, 0.3, 0.3)
    assert (a + b).parts == approx((0.65, 0.06, 0.06))
    assert (a * b).parts == approx((0.36, 0.06, 0.44))  # .7*.6-.06 ; 1-.8*.7
    assert (2 * a).parts == approx((0.75, 0.04, 0.04))
    assert (a**2).parts == approx((0.45, 0.04, 0.36))  # .49-.04
    assert a.score() == approx(0.3) and a.accuracy() == approx(0.9)
    assert a.score("normalized") == approx(0.65)
    assert a.distance(b) == approx(0.2)  # (.2+.1+.1+0)/2, refusal .1 and .1
    assert a.distance(b, "euclidean") == approx(0.1732050808)
    assert a.complement().parts == (0.2, 0.2, 0.5)


def test_neutrosophic_golden():  # a=(.7,.2,.1) b=(.4,.5,.3)
    a, b = NeutrosophicFN(0.7, 0.2, 0.1), NeutrosophicFN(0.4, 0.5, 0.3)
    assert (a + b).parts == approx((0.82, 0.1, 0.03))
    assert (a * b).parts == approx((0.28, 0.6, 0.37))
    assert (2 * a).parts == approx((0.91, 0.04, 0.01))  # 1-.3^2 ; .2^2 ; .1^2
    assert (a**2).parts == approx((0.49, 0.36, 0.19))  # .7^2 ; 1-.8^2 ; 1-.9^2
    assert a.score() == approx(0.8) and a.accuracy() == approx(0.6)
    assert a.accuracy("certainty") == 0.7
    assert a.distance(b) == approx(0.8 / 3)
    assert a.distance(b, "euclidean") == approx(0.2708012802)
    assert a.complement().parts == approx((0.1, 0.8, 0.7))


def test_einstein_refused():
    p = PictureFN(0.5, 0.2, 0.2)
    with pytest.raises(ValueError, match="valid: algebraic"):
        p.add(PictureFN(0.3, 0.3, 0.3), ops="einstein")
    with pytest.raises(ValueError, match="valid: algebraic"):
        SphericalFN(0.1, 0.1, 0.1).scale(2, ops="einstein")
    with pytest.raises(ValueError, match="valid: algebraic"):
        NeutrosophicFN(0.1, 0.1, 0.1).power(2, ops="einstein")


def test_unknown_names_and_misuse():
    s = SphericalFN(0.6, 0.3, 0.4)
    for call in (
        lambda: s.score("foo"),
        lambda: s.accuracy("foo"),
        lambda: s.distance(s, "foo"),
        lambda: NeutrosophicFN(0.1, 0.1, 0.1).score("foo"),
        lambda: NeutrosophicFN(0.1, 0.1, 0.1).accuracy("foo"),
    ):
        with pytest.raises(ValueError, match="valid:"):
            call()
    with pytest.raises(TypeError):
        s + PictureFN(0.1, 0.1, 0.1)
    with pytest.raises(ValueError):
        s.scale(-1)
    with pytest.raises(ValueError):
        s.power(float("nan"))
    assert (s.score("pi_adjusted"), s.accuracy("sum")) == (s.score(), s.accuracy())
    assert PictureFN(0.5, 0.2, 0.2).accuracy("sum") == approx(0.9)


# -- picture scale is not closed for k < 1 -------------------------------------
def test_picture_scale_below_one():
    p = PictureFN(0.0, 0.5, 0.5)
    assert p.scale(1).parts == approx(p.parts)
    with pytest.raises(ValueError, match="violates"):
        p.scale(0.5)  # eta^k + nu^k = 1.41 > (1-mu)^k = 1
    with pytest.raises(ValueError, match="violates"):
        PictureFN(0.5, 0.2, 0.2).scale(0)  # identity (0, 1, 1) is not a valid number
    # eta = 0: closed form, sits on the boundary: (1-.5^.5, 0, .5^.5)
    r = PictureFN(0.5, 0.0, 0.5).scale(0.5)
    assert r.parts == approx((1 - math.sqrt(0.5), 0.0, math.sqrt(0.5)), rel=1e-12)


def test_zero_scale_identity_for_spherical_and_neutrosophic():
    s, t = SphericalFN(0.6, 0.3, 0.4), SphericalFN(0.5, 0.5, 0.2)
    assert s.scale(0).parts == (0.0, 1.0, 0.0)
    assert (s.scale(0) + t).parts == approx(t.parts, rel=1e-12)
    assert s.power(0).parts == (1.0, 0.0, 0.0)  # mu^0 ; 1-(1-nu^2)^0 ; 0
    n = NeutrosophicFN(0.7, 0.2, 0.1)
    assert n.scale(0).parts == (0.0, 1.0, 1.0)
    assert n.power(0).parts == (1.0, 0.0, 0.0)
    assert PictureFN(0.5, 0.2, 0.2).power(0).parts == (0.0, 1.0, 0.0)


# -- boundary closed forms ---------------------------------------------------
def test_spherical_boundary_closed_forms():
    # (0.6, 0.8, 0): a=.36, b=.64, c=0, r=0 (up to the 1-ulp noise of 0.36+0.64)
    x = SphericalFN(0.6, 0.8, 0.0)
    r = x.scale(0.5)  # a'=1-.64^.5=.2 ; b'=.64^.5=.8 ; c'=0
    assert r.parts == approx((math.sqrt(0.2), math.sqrt(0.8), 0.0), rel=1e-12)
    r = x.power(0.5)  # a'=.36^.5=.6 ; b'=1-.36^.5=.4 ; c'=0
    assert r.parts == approx((math.sqrt(0.6), math.sqrt(0.4), 0.0), rel=1e-12)
    # (0.3, 0.4, sqrt(.75)): a=.09, b=.16, c=.75, r=0
    y = SphericalFN(0.3, 0.4, math.sqrt(0.75))
    r = y.scale(2)  # a'=1-.91^2=.1719 ; b'=.16^2=.0256 ; c'=.91^2-.16^2=.8025
    assert [p * p for p in r.parts] == approx((0.1719, 0.0256, 0.8025), rel=1e-12)
    r = y.power(2)  # a'=.09^2 ; b'=1-.84^2=.2944 ; c'=.84^2-.09^2=.6975
    assert [p * p for p in r.parts] == approx((0.0081, 0.2944, 0.6975), rel=1e-12)
    # (0.6, 0, 0.8): scale 1/2 gives a'=1-.64^.5=.2, b'=0, c'=.64^.5-0=.8
    z = SphericalFN(0.6, 0.0, 0.8).scale(0.5)
    assert [p * p for p in z.parts] == approx((0.2, 0.0, 0.8), rel=1e-12, abs=1e-15)


def test_spherical_tiny_parts_survive_at_boundary():
    # nu = 1e-9 next to a boundary point: nu^2 = 1e-18 must come out as 1e-36
    x = SphericalFN(0.6, 1e-9, math.sqrt(0.64 - 1e-18))
    assert x.scale(2).nu == approx(1e-18, rel=1e-9)  # b' = b^2
    assert x.scale(0.5).nu == approx(math.sqrt(1e-9), rel=1e-9)  # b' = b^0.5
    # mirrored: mu tiny, power keeps mu^k
    y = SphericalFN(1e-9, 0.6, math.sqrt(0.64 - 1e-18))
    assert y.power(2).mu == approx(1e-18, rel=1e-9)
    # a' for tiny mu under scale: 1-(1-a)^k ~ k a, so mu' ~ sqrt(k) mu
    assert y.scale(3).mu == approx(math.sqrt(3) * 1e-9, rel=1e-8)


def test_picture_boundary_closed_forms():
    x = PictureFN(0.5, 0.3, 0.2)  # rho = 0
    r = x**2  # (.8^2-.3^2, .3^2, 1-.8^2) = (.55, .09, .36)
    assert r.parts == approx((0.55, 0.09, 0.36), rel=1e-12)
    r = x**0.5  # (sqrt(.8)-sqrt(.3), sqrt(.3), 1-sqrt(.8))
    assert r.parts == approx(
        (math.sqrt(0.8) - math.sqrt(0.3), math.sqrt(0.3), 1 - math.sqrt(0.8)), rel=1e-12
    )
    # mul with another boundary point: ((.8)(.6)-.3*.4, .12, 1-.8*.6) = (.36,.12,.52)
    y = PictureFN(0.2, 0.4, 0.4)
    assert (x * y).parts == approx((0.36, 0.12, 0.52), rel=1e-12)
    # tiny nu on the boundary: 1-(1-nu)^2 ~ 2 nu stays accurate
    t = PictureFN(0.5, 0.5 - 1e-12, 1e-12)
    assert (t**2).nu == approx(2e-12 - 1e-24, rel=1e-9)
    assert (2 * PictureFN(1e-12, 0.0, 0.5)).mu == approx(2e-12 - 1e-24, rel=1e-9)  # 1-(1-m)^2


def test_neutrosophic_tiny_parts():
    n = NeutrosophicFN(1e-12, 1.0 - 1e-12, 1e-12)
    assert (2 * n).t == approx(2e-12 - 1e-24, rel=1e-9)  # 1-(1-t)^2
    assert (n**2).t == approx(1e-24, rel=1e-9)  # t^2
    assert (n**2).f == approx(2e-12 - 1e-24, rel=1e-9)  # 1-(1-f)^2
    m = n * n
    assert m.t == approx(1e-24, rel=1e-9)  # t*t
    assert m.f == approx(2e-12 - 1e-24, rel=1e-9)  # f+f-f*f, no cancellation
    # i+i-i*i = 1-(1-i)^2 = 1 - 1e-24, which rounds to exactly 1.0 (not 1 - 1e-12)
    assert m.i == 1.0 - 1e-24 == 1.0


# -- exact-arithmetic references ---------------------------------------------
_PREC = 80


def _D(x):
    return Decimal(x)


def _dpow(base, k):
    """Decimal base**k with 0**0 == 1."""
    return Decimal(1) if k == 0 else base ** _D(k)


def _sph_ref(op, x, other):
    """Squares of the result parts, in 80-digit arithmetic (clamped refusal)."""

    def st_(s):
        a, b, c = _D(s.mu) ** 2, _D(s.nu) ** 2, _D(s.pi) ** 2
        return a, b, c, max(Decimal(0), 1 - a - b - c)

    with localcontext() as ctx:
        ctx.prec = _PREC
        a1, b1, c1, r1 = st_(x)
        if op in ("add", "mul"):
            a2, b2, c2, r2 = st_(other)
            if op == "add":
                return (
                    a1 + a2 - a1 * a2,
                    b1 * b2,
                    (b2 + c2 + r2) * c1 + (b1 + c1 + r1) * c2 - c1 * c2,
                )
            return (
                a1 * a2,
                b1 + b2 - b1 * b2,
                (a2 + c2 + r2) * c1 + (a1 + c1 + r1) * c2 - c1 * c2,
            )
        k = other
        if op == "scale":
            ca = b1 + c1 + r1
            return (
                1 - _dpow(ca, k),
                _dpow(b1, k),
                # ca - c1 written as b1 + r1: at 80 digits 1e-144 + 1 - 1 cancels to 0
                Decimal(0) if k == 0 else _dpow(ca, k) - _dpow(b1 + r1, k),
            )
        cb = a1 + c1 + r1
        return (
            _dpow(a1, k),
            1 - _dpow(cb, k),
            Decimal(0) if k == 0 else _dpow(cb, k) - _dpow(a1 + r1, k),
        )


def _pic_ref(op, x, other):
    def rho(p):
        return max(Decimal(0), 1 - _D(p.mu) - _D(p.eta) - _D(p.nu))

    with localcontext() as ctx:
        ctx.prec = _PREC
        m1, e1, n1 = map(_D, x.parts)
        cm1, cn1 = e1 + n1 + rho(x), m1 + e1 + rho(x)
        if op in ("add", "mul"):
            m2, e2, n2 = map(_D, other.parts)
            if op == "add":
                return m1 + m2 - m1 * m2 + (m2 * (cm1 - 1 + m1)), e1 * e2, n1 * n2
            return (m1 + e1) * (m2 + e2) - e1 * e2, e1 * e2, n1 + n2 - n1 * n2 + n2 * (cn1 - 1 + n1)
        k = other
        if op == "scale":
            return 1 - _dpow(cm1, k), _dpow(e1, k), _dpow(n1, k)
        return _dpow(m1 + e1, k) - _dpow(e1, k), _dpow(e1, k), 1 - _dpow(cn1, k)


def _over(*xs):
    """True when an input lies (by rounding) beyond its constraint boundary."""
    with localcontext() as ctx:
        ctx.prec = _PREC
        return any(
            sum((_D(p) ** 2 if sq else _D(p)) for p in x.parts) > 1
            for x, sq in xs
        )


def _check_vs_ref(parts, ref, over=False):
    # relative accuracy on every part, tiny ones included. An input a rounding
    # step beyond the boundary has no exact answer: the clamped refusal and the
    # literal 1-x differ by that step, so allow an absolute 1e-15 there.
    assert tuple(parts) == approx(tuple(max(0.0, float(v)) for v in ref), rel=1e-9, abs=1e-15 if over else 1e-18
    )


@given(x=sphericals(), y=sphericals(), k=st.floats(0, 5))
def test_spherical_matches_reference(x, y, k):
    for op, arg in (("add", y), ("mul", y), ("scale", k), ("power", k)):
        r = getattr(x, op)(arg)
        _check_vs_ref(
            [p * p for p in r.parts],
            _sph_ref(op, x, arg),
            over=_over((x, True), (y, True)),
        )
        assert r.accuracy() <= 1 + 1e-12


_EXACT_BOUNDARY = [  # exact squares sum to >= 1, so the refusal r is exactly 0
    (0.6, 0.8, 0.0),
    (0.8, 0.0, 0.6),
    (0.0, 0.6, 0.8),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.5, math.sqrt(0.5)),
]


@given(
    p=st.sampled_from(_EXACT_BOUNDARY),
    perm=st.permutations([0, 1, 2]),
    k=st.floats(0.01, 5),
)
def test_spherical_boundary_stays_on_boundary(p, perm, k):
    # r = 0 gives sum' = 1 + b^k - (b+0)^k = 1 for scale, and likewise for power.
    # (A tiny r > 0 would be amplified by k < 1: (r)^k >> r. That is correct
    # maths, and test_spherical_matches_reference covers it.)
    x = SphericalFN(*(p[i] for i in perm))
    for r in (x.scale(k), x.power(k)):
        assert r.accuracy() == approx(1.0, abs=1e-12)


@given(x=sphericals(), y=sphericals(), z=sphericals())
def test_spherical_chains_never_raise(x, y, z):
    for r in ((x + y) * z, (x * y) + z, (x + y).scale(0.3).power(0.7), (x**0.5) * (y.scale(1.5))):
        assert r.accuracy() <= 1 + 1e-12


@given(x=pictures(), y=pictures(), k=st.floats(0, 5))
def test_picture_matches_reference(x, y, k):
    for op, arg in (("add", y), ("mul", y), ("power", k)):
        r = getattr(x, op)(arg)
        _check_vs_ref(r.parts, _pic_ref(op, x, arg), over=_over((x, False), (y, False)))
        assert r.accuracy() <= 1 + 1e-12
    ref = _pic_ref("scale", x, k)
    total = sum(ref)
    # No gap: up to _ROUNDING (1e-12) of overshoot is absorbed (the 1e-13 margin
    # below keeps clear of the float noise on the boundary), beyond it is refused.
    if total <= 1 + _D("9e-13"):
        _check_vs_ref(x.scale(k).parts, ref, over=_over((x, False)) or total > 1)
    elif total > 1 + _D("1.1e-12"):
        with pytest.raises(ValueError, match="violates"):
            x.scale(k)


@pytest.mark.parametrize("delta,ok", [(0.0, True), (5e-13, True), (2e-12, False), (5e-11, False), (5e-10, False)])
def test_picture_scale_overshoot_band(delta, ok):
    # scale 1/2 of (0, 1/4, nu): result sum is 0 + 0.5 + sqrt(nu), nu = (0.5+delta)^2,
    # so the overshoot of the exact result over 1 is delta; (1e-13, 1e-9] is the band.
    x = PictureFN(0.0, 0.25, (0.5 + delta) ** 2)
    if ok:
        r = x.scale(0.5)
        assert r.accuracy() <= 1 + 1e-15
        assert r.parts == approx((0.0, 0.5, 0.5 + delta), rel=1e-11)
    else:
        with pytest.raises(ValueError, match="violates"):
            x.scale(0.5)


_PIC_BOUNDARY = [  # dyadic, so mu + eta + nu = 1 exactly (rho = 0)
    (0.5, 0.25, 0.25),
    (0.5, 0.5, 0.0),
    (0.0, 0.0, 1.0),
    (0.125, 0.375, 0.5),
    (1.0, 0.0, 0.0),
    (0.0, 0.75, 0.25),
]


@given(
    p=st.sampled_from(_PIC_BOUNDARY),
    q=st.sampled_from(_PIC_BOUNDARY),
    k=st.floats(0.01, 5),
)
def test_picture_boundary_stays_on_boundary(p, q, k):
    # rho = 0: mul gives (1-n1)(1-n2) + 1 - (1-n1)(1-n2) = 1, power gives
    # (m+e)^k + 1 - (m+e)^k = 1. (A tiny rho > 0 would be amplified by k < 1.)
    x, y = PictureFN(*p), PictureFN(*q)
    for r in (x * y, x**k):
        assert r.accuracy() == approx(1.0, abs=1e-12)


@given(x=pictures(), y=pictures(), z=pictures())
def test_picture_chains_never_raise(x, y, z):
    for r in ((x + y) * z, (x * y) + z, (x + y) ** 0.5, (x**0.5) * y, (x + y).scale(2.0)):
        assert r.accuracy() <= 1 + 1e-12


def _neu_ref(op, x, other):
    with localcontext() as ctx:
        ctx.prec = _PREC
        t1, i1, f1 = map(_D, x.parts)
        if op in ("add", "mul"):
            t2, i2, f2 = map(_D, other.parts)
            if op == "add":
                return t1 + t2 - t1 * t2, i1 * i2, f1 * f2
            return t1 * t2, i1 + i2 - i1 * i2, f1 + f2 - f1 * f2
        k = other
        if op == "scale":
            return 1 - _dpow(1 - t1, k), _dpow(i1, k), _dpow(f1, k)
        return _dpow(t1, k), 1 - _dpow(1 - i1, k), 1 - _dpow(1 - f1, k)


@given(x=neutrosophics(), y=neutrosophics(), k=st.floats(0, 5))
def test_neutrosophic_matches_reference(x, y, k):
    for op, arg in (("add", y), ("mul", y), ("scale", k), ("power", k)):
        _check_vs_ref(getattr(x, op)(arg).parts, _neu_ref(op, x, arg))


@pytest.mark.parametrize("strategy", [sphericals, pictures, neutrosophics])
@given(data=st.data())
def test_closure_commutativity_identity(strategy, data):
    a, b = data.draw(strategy()), data.draw(strategy())
    k = data.draw(st.floats(0, 5))
    # picture scale is closed only for k >= 1 (see test_picture_scale_below_one)
    ks = max(k, 1.0) if strategy is pictures else k
    for r in (a + b, a * b, ks * a, a**k):
        type(r)(*r.parts)  # re-validates
    assert (a + b).parts == approx((b + a).parts, abs=1e-9)
    assert (a * b).parts == approx((b * a).parts, abs=1e-9)
    if strategy is not pictures:  # (0, 1, 1) is not a valid picture number
        assert (0 * a + b).parts == approx(b.parts, abs=1e-9)
    assert 0 <= a.distance(b) <= 1 + 1e-12 and a.distance(a) == approx(0, abs=1e-12)
    assert 0 <= a.distance(b, "euclidean") <= 1 + 1e-12
    assert a.distance(b) == approx(b.distance(a), abs=1e-12)


def test_genuine_violation_still_raises():
    x = SphericalFN(0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="violates"):
        x._finish((0.7, 0.7, 0.0))
    with pytest.raises(ValueError):
        PictureFN(0.5, 0.5, 0.5)


# -- deferred minors (batch A) --------------------------------------------------
@pytest.mark.parametrize("excess", [5e-13, 9.9e-13])
@pytest.mark.parametrize("share", [0.0, 0.3, 0.5, 1.0])
def test_spherical_near_boundary_input_is_safe_under_ops(excess, share):
    # sum of squares = 1 + excess, just under TOL: accepted, then projected to 1
    a = 0.5 * share
    x = SphericalFN(math.sqrt(a), math.sqrt(0.5 - a), math.sqrt(0.5 + excess))
    assert x.accuracy() <= 1 + 1e-15
    for r in (x + x, x * x, x.scale(0.5), x.scale(7), x.power(0.5), x.power(7)):
        assert r.accuracy() <= 1 + 1e-12


@pytest.mark.parametrize("excess", [5e-13, 9.9e-13])
@pytest.mark.parametrize("m", [0.0, 0.3, 0.7])
def test_picture_near_boundary_input_is_safe_under_ops(excess, m):
    x = PictureFN(m, 0.2, 0.8 - m + excess)
    assert x.accuracy() <= 1 + 1e-15
    for r in (x + x, x * x, x.power(0.5), x.power(7), x.scale(2), x.scale(7)):
        assert r.accuracy() <= 1 + 1e-12


def test_near_boundary_previously_raised():
    # these raised "result violates ..." before accepted inputs were projected
    s = SphericalFN(math.sqrt(0.5 + 9e-13), math.sqrt(0.5), 0.0)
    assert (s + s).accuracy() <= 1 + 1e-12 and (s * s).accuracy() <= 1 + 1e-12
    p = PictureFN(0.5 + 9e-13, 0.5, 0.0)
    assert p.power(2).accuracy() <= 1 + 1e-12
    assert (p + p).accuracy() <= 1 + 1e-12 and (p * p).accuracy() <= 1 + 1e-12


def test_projection_keeps_interior_exact_and_rejects_beyond_tol():
    assert SphericalFN(0.6, 0.3, 0.4).parts == (0.6, 0.3, 0.4)
    assert PictureFN(0.5, 0.2, 0.2).parts == (0.5, 0.2, 0.2)
    with pytest.raises(ValueError):
        PictureFN(0.5, 0.5, 2e-12)
    with pytest.raises(ValueError):
        SphericalFN(1.0, 0.0, 2e-6)  # squares sum 1 + 4e-12


def test_spherical_underflow_ruling():
    # RULING: parts below ~1.5e-154 square to 0.0 and act as 0 (documented limit).
    tiny = SphericalFN(1e-200, 0.5, 0.5)
    zero = SphericalFN(0.0, 0.5, 0.5)
    other = SphericalFN(0.3, 0.4, 0.5)
    for op in ("add", "mul"):
        assert getattr(tiny, op)(other).parts == approx(
            getattr(zero, op)(other).parts, abs=1e-150
        )
    assert tiny.scale(2).parts == approx(zero.scale(2).parts, abs=1e-150)
    assert tiny.power(2).parts == approx(zero.power(2).parts, abs=1e-150)
    # just above the limit the part still survives: (1e-150)^2 = 1e-300 is normal
    assert SphericalFN(1e-150, 0.5, 0.5).power(2).mu == approx(1e-300, rel=1e-9)


def test_default_names_come_from_class_defaults():
    s, p, n = SphericalFN(0.6, 0.3, 0.4), PictureFN(0.5, 0.2, 0.2), NeutrosophicFN(0.7, 0.2, 0.1)
    assert s.score() == s.score(SphericalFN.DEFAULT_SCORE)
    assert s.accuracy() == s.accuracy(SphericalFN.DEFAULT_ACCURACY)
    assert p.accuracy() == p.accuracy(PictureFN.DEFAULT_ACCURACY)
    assert n.score() == n.score(NeutrosophicFN.DEFAULT_SCORE)
    assert s._simplex_parts() == s._state()


@given(st.floats(0, 1, allow_nan=False), st.floats(0, 1, allow_nan=False))
def test_normalised_boundary_round_trips_bit_identically(u, v):
    # random normalised boundary points are float-noise only: no 1-ulp rewrite
    m, e = u, v * (1 - u)
    n = 1.0 - m - e
    p = PictureFN(m, e, n)
    assert p.parts == (m, e, n)
    r = math.sqrt
    t = 1.0 - u * u
    mu, nu = u, r(t) * v
    pi = r(max(0.0, 1.0 - mu * mu - nu * nu))
    s = SphericalFN(mu, nu, pi)
    assert s.parts == (mu, nu, pi)
