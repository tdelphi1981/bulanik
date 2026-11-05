import dataclasses
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
from hypothesis import given, strategies as st
from pytest import approx

from bulanik import FFN, IFN, PFN, QROFN
from bulanik.numbers import orthopair
from strategies import qrofns


def test_constraint():
    with pytest.raises(ValueError, match=r"mu\^1 \+ nu\^1"):
        IFN(0.6, 0.5)
    with pytest.raises(ValueError):
        PFN(0.8, 0.7)  # 0.64+0.49 = 1.13
    assert FFN(0.8, 0.7).nu == 0.7  # 0.512+0.343 = 0.855
    with pytest.raises(ValueError):
        QROFN(0.5, 0.5, 0)
    with pytest.raises(ValueError):
        QROFN(0.5, 0.5, True)
    with pytest.raises(ValueError):
        IFN(float("nan"), 0.1)


def test_boundary_float_noise_accepted():
    assert PFN(0.6, 0.8).mu == 0.6  # 0.36+0.64 = 1.0000000000000002


def test_repr_frozen_eq():
    assert repr(IFN(0.6, 0.3)) == "IFN(mu=0.6, nu=0.3)"
    assert repr(QROFN(0.6, 0.3, 4)) == "QROFN(mu=0.6, nu=0.3, q=4)"
    with pytest.raises(AttributeError):
        IFN(0.6, 0.3).mu = 0.1
    assert IFN(0.6, 0.3) == IFN(0.6, 0.3) and IFN(0.6, 0.3) != PFN(0.6, 0.3)
    assert len({IFN(0.6, 0.3), IFN(0.6, 0.3)}) == 1


def test_ifn_algebraic_golden():
    a, b = IFN(0.6, 0.3), IFN(0.5, 0.4)
    s = a + b
    assert (s.mu, s.nu) == approx((0.8, 0.12))  # .6+.5-.3 ; .3*.4
    p = a * b
    assert (p.mu, p.nu) == approx((0.3, 0.58))  # .6*.5 ; .3+.4-.12
    k = 2 * a
    assert (k.mu, k.nu) == approx((0.84, 0.09))  # 1-.4^2 ; .3^2
    w = a**2
    assert (w.mu, w.nu) == approx((0.36, 0.51))  # .6^2 ; 1-.7^2
    assert isinstance(s, IFN) and isinstance(w, IFN)


def test_ifn_einstein_golden():
    a, b = IFN(0.6, 0.3), IFN(0.5, 0.4)
    s = a.add(b, ops="einstein")  # 1.1/1.3 ; .12/(1+.7*.6)=.12/1.42
    assert (s.mu, s.nu) == approx((0.8461538462, 0.0845070423))
    p = a.mul(b, ops="einstein")  # .3/(1+.4*.5) ; .7/(1+.12)
    assert (p.mu, p.nu) == approx((0.25, 0.625))
    k = a.scale(2, ops="einstein")  # 2.4/2.72 ; .18/2.98
    assert (k.mu, k.nu) == approx((0.8823529412, 0.0604026846))
    w = a.power(2, ops="einstein")  # .72/2.32 ; 1.2/2.18
    assert (w.mu, w.nu) == approx((0.3103448276, 0.5504587156))


def test_pfn_golden():
    a, b = PFN(0.6, 0.7), PFN(0.8, 0.5)
    s = a + b  # mu = sqrt(.36+.64-.2304)=sqrt(.7696) ; nu = .35
    assert (s.mu, s.nu) == approx((0.8772684880, 0.35))
    e = a.add(b, ops="einstein")  # sqrt(1/(1+.2304)) ; sqrt(.49*.25/(1+.51*.75))
    assert (e.mu, e.nu) == approx((0.9015230575, 0.2976702789))
    assert a.score() == approx(-0.13) and a.accuracy() == approx(0.85)
    assert a.score("normalized") == approx(0.435)  # (1-.13)/2


def test_ffn_score():
    assert FFN(0.8, 0.7).score() == approx(0.169)  # .512-.343


def test_distances_golden():
    a, b = IFN(0.6, 0.3), IFN(0.5, 0.4)  # pi = .1, .1
    assert a.distance(b) == approx(0.1)  # (.1+.1+0)/2
    assert a.distance(b, "euclidean") == approx(0.1)  # sqrt((.01+.01)/2)
    assert IFN(1, 0).distance(IFN(0, 1)) == approx(1.0)
    assert IFN(1, 0).distance(IFN(0, 1), "euclidean") == approx(1.0)


def test_complement_and_hesitancy():
    assert IFN(0.6, 0.3).complement() == IFN(0.3, 0.6)
    assert IFN(0.6, 0.3).hesitancy == approx(0.1)
    assert PFN(0.6, 0.7).hesitancy == approx(0.15**0.5)  # 1-.36-.49


def test_mixing_q_raises():
    with pytest.raises(TypeError):
        IFN(0.5, 0.2) + PFN(0.5, 0.2)
    with pytest.raises(TypeError):
        QROFN(0.5, 0.2, 2) + PFN(0.5, 0.2)
    with pytest.raises(TypeError):
        IFN(0.5, 0.2).distance(PFN(0.5, 0.2))


def test_unknown_names():
    with pytest.raises(ValueError, match="valid: algebraic, einstein"):
        IFN(0.5, 0.2).add(IFN(0.5, 0.2), ops="hamacher")
    with pytest.raises(ValueError, match="valid: difference, normalized"):
        IFN(0.5, 0.2).score("x")
    with pytest.raises(ValueError, match="valid: sum"):
        IFN(0.5, 0.2).accuracy("x")
    with pytest.raises(ValueError, match="valid: hamming, euclidean"):
        IFN(0.5, 0.2).distance(IFN(0.5, 0.2), "x")
    with pytest.raises(ValueError):
        IFN(0.5, 0.2).scale(-1)
    with pytest.raises(ValueError):
        IFN(0.5, 0.2).power(float("inf"))


def test_zero_lambda():
    a = IFN(0.6, 0.3)
    z = a.scale(0)  # 1-(.4)^0 = 0 ; .3^0 = 1
    assert (z.mu, z.nu) == (0.0, 1.0)
    p = a.power(0, ops="einstein")  # 2*1/(1+1)=1 ; 0
    assert (p.mu, p.nu) == (1.0, 0.0)


def _exact(q, ops, op, x, arg):
    """Result q-th powers (a, b) from exact rational arithmetic (integer k)."""
    a1, b1 = Fraction(x.mu) ** q, Fraction(x.nu) ** q
    if op in ("add", "mul"):
        a2, b2 = Fraction(arg.mu) ** q, Fraction(arg.nu) ** q
        if ops == "algebraic":
            return (
                (a1 + a2 - a1 * a2, b1 * b2)
                if op == "add"
                else (a1 * a2, b1 + b2 - b1 * b2)
            )
        if op == "add":
            return (a1 + a2) / (1 + a1 * a2), b1 * b2 / (1 + (1 - b1) * (1 - b2))
        return a1 * a2 / (1 + (1 - a1) * (1 - a2)), (b1 + b2) / (1 + b1 * b2)
    k = arg

    def tanh_part(z):
        p, m = (1 + z) ** k, (1 - z) ** k
        return (p - m) / (p + m) if p + m else Fraction(0)

    def two_over(z):
        return 2 * z**k / ((2 - z) ** k + z**k)

    if ops == "algebraic":
        return (
            (1 - (1 - a1) ** k, b1**k)
            if op == "scale"
            else (a1**k, 1 - (1 - b1) ** k)
        )
    return (
        (tanh_part(a1), two_over(b1))
        if op == "scale"
        else (two_over(a1), tanh_part(b1))
    )


@pytest.mark.parametrize("q", [1, 2, 3, 5])
@pytest.mark.parametrize("ops", ["algebraic", "einstein"])
@given(data=st.data())
def test_operations_match_exact_rational_reference(q, ops, data):
    # Independent of the stable formulas: Fractions, integer k so powers are exact.
    a, b = data.draw(qrofns(q)), data.draw(qrofns(q))
    k = data.draw(st.integers(0, 5))
    for op, arg in (("add", b), ("mul", b), ("scale", k), ("power", k)):
        r = getattr(a, op)(arg, ops)
        ea, eb = _exact(q, ops, op, a, arg)
        assert r.mu**q == approx(float(ea), abs=1e-9)
        assert r.nu**q == approx(float(eb), abs=1e-9)


@pytest.mark.parametrize("q", [1, 2, 3, 5])
@pytest.mark.parametrize("ops", ["algebraic", "einstein"])
@given(data=st.data())
def test_closure_and_laws(q, ops, data):
    a, b, c = (data.draw(qrofns(q)) for _ in range(3))
    k = data.draw(st.floats(0, 5))
    for r in (a.add(b, ops), a.mul(b, ops), a.scale(k, ops), a.power(k, ops)):
        assert isinstance(r, QROFN) and r.q == q  # none of the four raised
    # Compare in q-th power space: the q-th root amplifies rounding of a part
    # near 0 (a ~ 1e-16 moves mu by ~1e-16^(1/q)), so parts are not comparable
    # to 1e-9 for q > 1, while the powers are stable to ~1e-15.
    ab, ba = a.add(b, ops), b.add(a, ops)
    assert ab._powers() == approx(ba._powers(), abs=1e-12)
    l, r = a.add(b, ops).add(c, ops), a.add(b.add(c, ops), ops)
    assert l._powers() == approx(r._powers(), abs=1e-12)
    z = a.scale(0, ops).add(b, ops)
    assert z._powers() == approx(b._powers(), abs=1e-12)  # 0.A is the + identity


@given(qrofns(2), qrofns(2))
def test_score_bounded_and_monotone(a, b):
    assert -1 <= a.score() <= 1 and 0 <= a.score("normalized") <= 1
    if a.mu >= b.mu and a.nu <= b.nu:
        assert a.score() >= b.score() - 1e-12


# -- fix round 1: boundary inputs must not raise spuriously -------------------
_MU = 0.0016253414155257273


@pytest.mark.parametrize("ops", ["algebraic", "einstein"])
@pytest.mark.parametrize("method,k", [("power", 0.5), ("scale", 0.5)])
def test_near_boundary_fractional_exponent_stays_valid(ops, method, k):
    # nu^3 = 1 - mu^3 up to rounding; k < 1 amplifies that rounding past TOL
    x = QROFN(_MU, (1 - _MU**3) ** (1 / 3), 3)
    r = getattr(x, method)(k, ops)
    assert r.mu**3 + r.nu**3 <= 1 + 1e-12


def test_near_boundary_scale_other_end():
    # same defect from the mu side: swap roles via the complement
    x = QROFN((1 - _MU**3) ** (1 / 3), _MU, 3)
    for ops in ("algebraic", "einstein"):
        for r in (x.scale(0.5, ops), x.power(0.5, ops)):
            assert r.mu**3 + r.nu**3 <= 1 + 1e-12


def test_genuine_violation_still_raises():
    x = QROFN(0.5, 0.5, 3)
    with pytest.raises(ValueError, match="violates"):
        x._new(0.7, 0.7)


@pytest.mark.parametrize("q", [3, 5, 10])
@given(
    mu=st.floats(0, 0.05, allow_nan=False),
    k=st.floats(0.01, 0.99),
    ops=st.sampled_from(["algebraic", "einstein"]),
)
def test_boundary_fractional_exponents(q, mu, k, ops):
    x = QROFN(mu, (1 - mu**q) ** (1 / q), q)  # u = 1: nu on the boundary
    for r in (x.scale(k, ops), x.power(k, ops), x.complement().scale(k, ops)):
        assert r.mu**q + r.nu**q <= 1 + 1e-12


# -- fix round 2: boundary handling must not damage the small part ------------
@pytest.mark.parametrize("ops", ["algebraic", "einstein"])
def test_small_mu_survives_boundary_power(ops):
    x = QROFN(0.028, 1.0, 10)  # a = 0.028^10 ~ 2.9e-16, b = 1
    # algebraic: a^k, so mu -> mu^k.
    # einstein, a tiny: 2a^k/((2-a)^k+a^k) ~ 2^(1-k) a^k, so mu -> mu^k 2^((1-k)/q)
    f = (lambda k: 1.0) if ops == "algebraic" else (lambda k: 2 ** ((1 - k) / 10))
    for k in (2, 0.5):
        assert x.power(k, ops).mu == approx(0.028**k * f(k), rel=1e-7)


def test_small_mu_survives_boundary_scale_mirror():
    # nu small on the boundary: scale keeps nu^k exactly (algebraic)
    x = QROFN(1.0, 0.028, 10)
    assert x.scale(2).nu == approx(0.028**2, rel=1e-9)
    assert x.scale(0.5).nu == approx(0.028**0.5, rel=1e-9)


@pytest.mark.parametrize("q", [3, 5, 10])
@given(mu=st.floats(1e-6, 1, allow_nan=False), k=st.floats(0.05, 5))
def test_boundary_power_keeps_mu(q, mu, k):
    x = QROFN(mu, (max(0.0, 1 - mu**q)) ** (1 / q), q)
    assert x.power(k).mu == approx(mu**k, rel=1e-9)
    assert x.complement().scale(k).nu == approx(mu**k, rel=1e-9)


# -- fix round 3: interior points within TOL of the boundary ------------------
def _ref_one_minus_pow(x, k):  # 1 - (1 - x)^k, exact complement
    with localcontext() as ctx:
        ctx.prec = 60
        return float(1 - (1 - Decimal(x)) ** Decimal(k))


def _ref_tanh_k_atanh(x, k):  # tanh(k atanh x) = ((1+x)^k-(1-x)^k)/((1+x)^k+(1-x)^k)
    with localcontext() as ctx:
        ctx.prec = 60
        x, k = Decimal(x), Decimal(k)
        p, m = (1 + x) ** k, (1 - x) ** k
        return float((p - m) / (p + m))


def test_interior_near_boundary_complement_part_is_accurate():
    a = IFN(0.9999999999996776, 1.6560627545253775e-13)  # slack ~ 1e-13 < TOL
    assert a.scale(0.03).mu == approx(_ref_one_minus_pow(a.mu, 0.03), rel=1e-9)
    assert a.scale(0.03, "einstein").mu == approx(
        _ref_tanh_k_atanh(a.mu, 0.03), rel=1e-9
    )
    b = IFN(1e-14, 0.99999999999899)  # slack ~ 9.9e-13 < TOL
    assert b.power(0.5).nu == approx(_ref_one_minus_pow(b.nu, 0.5), rel=1e-9)
    assert b.power(0.5, "einstein").nu == approx(
        _ref_tanh_k_atanh(b.nu, 0.5), rel=1e-9
    )


@pytest.mark.parametrize("ops", ["algebraic", "einstein"])
@given(
    big=st.floats(0.5, 1),
    slack=st.floats(0, 1e-12),
    k=st.floats(0.01, 3),
)
def test_interior_within_tol_matches_reference(ops, big, slack, k):
    small = max(0.0, 1 - big - slack)
    x = IFN(big, small)
    ref = _ref_one_minus_pow if ops == "algebraic" else _ref_tanh_k_atanh
    assert x.scale(k, ops).mu == approx(ref(x.mu, k), rel=1e-9, abs=1e-15)
    y = x.complement()
    assert y.power(k, ops).nu == approx(ref(y.nu, k), rel=1e-9, abs=1e-15)


# -- deferred minors (batch A) --------------------------------------------------
def test_pfn_ffn_goldens_mul_scale_power():
    # references: exact Decimal evaluation of the q-rung formulas on (mu^q, nu^q)
    a, b = PFN(0.6, 0.7), PFN(0.8, 0.5)
    for got, want in [
        (a * b, (0.48, 0.7858116822750855)),
        (a.scale(2), (0.7683749084919418, 0.49)),
        (a.scale(0.5), (0.4472135954999579, 0.8366600265340756)),
        (a**2, (0.36, 0.8601744009211155)),
        (a**0.5, (0.7745966692414834, 0.534656111108547)),
        (a.mul(b, "einstein"), (0.4327310675847713, 0.8119376587476715)),
        (a.scale(2, "einstein"), (0.7983694431185242, 0.43650937366675696)),
        (a.scale(0.5, "einstein"), (0.4315595002904901, 0.8519578385512125)),
        (a.power(2, "einstein"), (0.30321770423814404, 0.8889650443593012)),
        (a.power(0.5, "einstein"), (0.7988027256580057, 0.5116551116621422)),
    ]:
        assert (got.mu, got.nu) == approx(want, rel=1e-12)
        assert isinstance(got, PFN)
    a, b = FFN(0.8, 0.7), FFN(0.6, 0.5)
    for got, want in [
        (a * b, (0.48, 0.7519210013965683)),
        (a.scale(2), (0.9133227957415779, 0.49)),
        (a.scale(0.5), (0.670494942919109, 0.8366600265340756)),
        (a**2, (0.64, 0.8283341047844872)),
        (a**0.5, (0.8944271909999159, 0.5743290319282938)),
        (a.mul(b, "einstein"), (0.4308669913839056, 0.7656045964785675)),
        (a.scale(2, "einstein"), (0.9326750301395315, 0.4347609829162632)),
        (a.scale(0.5, "einstein"), (0.6506258640333833, 0.8551722574758323)),
        (a.power(2, "einstein"), (0.5960137168449828, 0.8498446810234126)),
        (a.power(0.5, "einstein"), (0.9042733665241913, 0.5613241552218052)),
    ]:
        assert (got.mu, got.nu) == approx(want, rel=1e-12)
        assert isinstance(got, FFN)


@pytest.mark.parametrize("cls", [IFN, PFN, FFN])
def test_dataclasses_replace_on_fixed_q(cls):
    x = cls(0.5, 0.2)
    y = dataclasses.replace(x, mu=0.4)
    assert type(y) is cls and (y.mu, y.nu, y.q) == (0.4, 0.2, cls.Q)
    assert dataclasses.replace(x) == x
    with pytest.raises(ValueError, match="q"):
        cls(0.5, 0.2, cls.Q + 1)
    with pytest.raises(ValueError, match="q"):
        cls(0.5, 0.2, True)
    assert cls(0.5, 0.2, cls.Q) == x  # an explicit, matching q is fine


_OPS_CALLS = [
    ("add", lambda x: x),
    ("mul", lambda x: x),
    ("scale", lambda x: 0.5),
    ("scale", lambda x: 7),
    ("power", lambda x: 0.5),
    ("power", lambda x: 7),
]


@pytest.mark.parametrize("q", [1, 2, 3, 5])
@pytest.mark.parametrize("ops", ["algebraic", "einstein"])
@pytest.mark.parametrize("a_share", [0.0, 0.05, 0.3, 0.5, 0.7, 0.95, 1.0])
@pytest.mark.parametrize("excess", [5e-13, 9.9e-13])
def test_accepted_near_boundary_input_is_safe_under_every_op(q, ops, a_share, excess):
    # mu^q + nu^q = 1 + excess, just under TOL: accepted by the constructor, and
    # (before projection on entry) amplifying ops magnified it past the result check.
    a = a_share
    x = QROFN(a ** (1 / q), (1 - a + excess) ** (1 / q), q)
    assert x.mu**q + x.nu**q <= 1 + 1e-15  # projected back onto the boundary
    for name, arg in _OPS_CALLS:
        r = getattr(x, name)(arg(x), ops)
        assert r.mu**q + r.nu**q <= 1 + 1e-12


def test_previously_raising_near_boundary_input():
    # found by search: raised "result violates mu^2 + nu^2 <= 1" before projection
    a = 0.4408
    x = PFN(math.sqrt(a), math.sqrt(1 - a + 9.9e-13))
    for ops in ("algebraic", "einstein"):
        for r in (x.add(x, ops), x.mul(x, ops)):
            assert r.mu**2 + r.nu**2 <= 1 + 1e-12


def test_projection_leaves_interior_and_exact_boundary_untouched():
    assert (IFN(0.6, 0.3).mu, IFN(0.6, 0.3).nu) == (0.6, 0.3)
    assert PFN(0.6, 0.8).mu == 0.6  # 1 + 2 ulp noise: projected only if > 1
    assert IFN(0.5, 0.5) == IFN(0.5, 0.5)
    assert IFN(1.0, 1e-13).mu < 1.0  # excess 1e-13 -> projected
    with pytest.raises(ValueError):
        IFN(1.0, 2e-12)  # beyond TOL still rejected


@pytest.mark.parametrize("q", [1, 2, 3, 5, 7, 10, 20, 50, 200])
@pytest.mark.parametrize("ops", ["algebraic", "einstein"])
@given(
    a=st.one_of(st.floats(0, 1, allow_nan=False), st.sampled_from([0.0, 1.0, 1e-9, 1e-5])),
    k=st.one_of(st.floats(0, 1e6), st.sampled_from([0.0, 1.0, 944.0, 1000.0, 1e6])),
    swap=st.booleans(),
)
def test_rounding_bound_pinned(q, ops, a, k, swap):
    # Derived bound (see orthopair._overshoot_ulps): input noise d <= 4 + 2 (q + 2)
    # ulp is magnified by at most k, plus a few ulps of rounding, so the raw
    # overshoot of a' + b' over 1 is at most d eps (1 + k) + 8 eps. The constants
    # here are written out, not imported, so the module cannot grade itself.
    x = QROFN(a ** (1 / q), (1 - a) ** (1 / q), q)
    if swap:
        x = x.complement()
    for name in ("scale", "power"):
        # capture the raw formula output through _new's argument
        seen = []
        orig = QROFN._new
        try:
            QROFN._new = lambda self, ra, rb, slack=0.0: seen.append(ra + rb - 1.0) or orig(self, ra, rb, slack)
            getattr(x, name)(k, ops)
        finally:
            QROFN._new = orig
        eps = 2.0**-52
        assert seen[0] <= (4 + 2 * (q + 2)) * eps * (1.0 + k) + 8 * eps
        assert x.mu is not None  # no ValueError raised by the operation above


@pytest.mark.parametrize("q", [1, 2, 3])
@given(a=st.floats(0, 1, allow_nan=False), data=st.data())
def test_normalised_boundary_round_trips_bit_identically(q, a, data):
    # a deliberate boundary value (float noise only) must not be altered by the
    # constructor's projection
    mu = a ** (1 / q)
    nu = (1 - mu**q) ** (1 / q)
    x = QROFN(mu, nu, q)
    assert (x.mu, x.nu) == (mu, nu)


def test_rounding_constant_is_documented_tolerance():
    assert orthopair._ROUNDING == 1e-12


@pytest.mark.parametrize("q,a,k", [(5, 1e-5, 944.0), (2, 1e-5, 29526.0), (3, 0.3, 1e5)])
@pytest.mark.parametrize("ops", ["algebraic", "einstein"])
def test_large_multiples_of_boundary_values_do_not_raise(q, a, k, ops):
    # found by hypothesis: input rounding d is magnified to k * d (up to 1e-12 here)
    x = QROFN(a ** (1 / q), (1 - a) ** (1 / q), q)
    for r in (x.scale(k, ops), x.power(k, ops), x.complement().scale(k, ops)):
        assert r.mu**q + r.nu**q <= 1 + 1e-15


def test_einstein_huge_k_does_not_overflow():
    x = IFN(0.0, 0.5)  # (1 + cx)^k used to overflow for k >= 1024
    r = x.scale(1024, "einstein")
    assert (r.mu, r.nu) == (0.0, 0.0)
    assert IFN(0.5, 0.0).power(5000, "einstein").mu == 0.0
    assert IFN(1.0, 0.0).power(5000, "einstein").mu == 1.0


@pytest.mark.parametrize(
    "q, a",
    [(200, 7.38668800442932e-05), (500, 0.00024739000144791135)],
)
def test_large_q_boundary_scale_does_not_raise(q, a):
    # found by search: a deliberate boundary value at large q carries ~q/4 ulps of
    # noise, which the old q-blind slack (32 ulps) rejected under scale by 1000
    x = QROFN(a ** (1 / q), (1 - a) ** (1 / q), q)
    r = x.scale(1000.0, "algebraic")
    assert r.mu**q + r.nu**q <= 1 + 1e-12


@pytest.mark.parametrize("q", [10, 20, 50])
@pytest.mark.parametrize("name", ["scale", "power"])
@pytest.mark.parametrize("ops", ["algebraic", "einstein"])
def test_large_q_boundary_inputs_survive_every_k(q, name, ops):
    for i in range(1, 200):
        a = i / 200
        x = QROFN(a ** (1 / q), (1 - a) ** (1 / q), q)
        for k in (0.5, 1.0, 10.0, 1e3, 1e6):
            r = getattr(x, name)(k, ops)
            assert r.mu**q + r.nu**q <= 1 + 1e-12
