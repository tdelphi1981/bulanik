import dataclasses
import math

import pytest
from hypothesis import given, strategies as st
from pytest import approx

from bulanik import IVNeutrosophicFN, NeutrosophicFN
from bulanik.aggregate import wa, wg
from bulanik.mcdm import weighting
from strategies import ivneutrosophics, neutrosophics

N = IVNeutrosophicFN
A, B = N(0.6, 0.7, 0.1, 0.2, 0.2, 0.3), N(0.5, 0.6, 0.2, 0.3, 0.1, 0.2)


def b(x):
    return x.bounds


def test_add_mul_golden():
    # Hand-derived (Zhang, Wang and Chen 2014 laws, per bound):
    # add T: [.6+.5-.30, .7+.6-.42] = [.8, .88]; I: [.1*.2, .2*.3] = [.02, .06];
    #     F: [.2*.1, .3*.2] = [.02, .06]
    # mul T: [.6*.5, .7*.6] = [.3, .42]; I: [.1+.2-.02, .2+.3-.06] = [.28, .44];
    #     F: [.2+.1-.02, .3+.2-.06] = [.28, .44]
    # 2A  T: [1-.4^2, 1-.3^2] = [.84, .91]; I: [.01, .04]; F: [.04, .09]
    # A^2 T: [.36, .49]; I: [1-.81, 1-.64] = [.19, .36]; F: [1-.64, 1-.49] = [.36, .51]
    assert b(A + B) == approx((0.8, 0.88, 0.02, 0.06, 0.02, 0.06))
    assert b(A * B) == approx((0.3, 0.42, 0.28, 0.44, 0.28, 0.44))
    assert b(A.scale(2)) == approx((0.84, 0.91, 0.01, 0.04, 0.04, 0.09))
    assert b(2 * A) == b(A * 2) == b(A.scale(2))
    assert b(A**2) == approx((0.36, 0.49, 0.19, 0.36, 0.36, 0.51))


def test_score_accuracy_golden():
    assert A.score() == approx(0.75)  # (4 + 1.3 - .3 - .5)/6
    assert A.accuracy() == approx(0.4) and A.accuracy("certainty") == approx(0.65)


def test_complement_and_from_neutrosophic():
    assert b(A.complement()) == approx((0.2, 0.3, 0.8, 0.9, 0.6, 0.7))
    x = N.from_neutrosophic(NeutrosophicFN(0.7, 0.2, 0.1))
    assert b(x) == (0.7, 0.7, 0.2, 0.2, 0.1, 0.1)
    with pytest.raises(TypeError):
        N.from_neutrosophic((0.7, 0.2, 0.1))


@given(x=neutrosophics(), y=neutrosophics(), k=st.floats(0, 5))
def test_degenerate_matches_neutrosophic(x, y, k):
    ix, iy = N.from_neutrosophic(x), N.from_neutrosophic(y)

    def deg(r):
        t, i, f = r.parts
        return (t, t, i, i, f, f)

    assert b(ix + iy) == approx(deg(x + y), abs=1e-12)
    assert b(ix * iy) == approx(deg(x * y), abs=1e-12)
    assert b(ix.scale(k)) == approx(deg(x.scale(k)), abs=1e-12)
    assert b(ix**k) == approx(deg(x**k), abs=1e-12)
    assert ix.score() == approx(x.score(), abs=1e-12)
    for m in N.ACCURACIES:
        assert ix.accuracy(m) == approx(x.accuracy(m), abs=1e-12)
    for m in N.DISTANCES:
        assert ix.distance(iy, m) == approx(x.distance(y, m), abs=1e-12)
    assert b(ix.complement()) == approx(deg(x.complement()), abs=1e-12)


@given(x=ivneutrosophics(), y=ivneutrosophics(), k=st.floats(0, 5))
def test_closure_commutativity_identity(x, y, k):
    for r in (x + y, x * y, x.scale(k), x**k):
        N(*r.bounds)  # re-validates
    assert b(x + y) == approx(b(y + x), abs=1e-9)
    assert b(x * y) == approx(b(y * x), abs=1e-9)
    assert b(0 * x + y) == approx(b(y), abs=1e-9)
    for m in N.DISTANCES:
        assert x.distance(x, m) == 0
        assert x.distance(y, m) == approx(y.distance(x, m), abs=1e-12)
        assert 0 <= x.distance(y, m) <= 1 + 1e-12
    assert 0 <= x.score() <= 1


# Zhang, Wang and Chen (2014), Section 5 numerical example.
W = [0.35, 0.25, 0.4]
RAW = [
    ((.4, .5), (.2, .3), (.3, .4)), ((.4, .6), (.1, .3), (.2, .4)), ((.7, .9), (.2, .3), (.4, .5)),
    ((.6, .7), (.1, .2), (.2, .3)), ((.6, .7), (.1, .2), (.2, .3)), ((.3, .6), (.3, .5), (.8, .9)),
    ((.3, .6), (.2, .3), (.3, .4)), ((.5, .6), (.2, .3), (.3, .4)), ((.4, .5), (.2, .4), (.7, .9)),
    ((.7, .8), (0, .1), (.1, .2)), ((.6, .7), (.1, .2), (.1, .3)), ((.6, .7), (.3, .4), (.8, .9)),
]
# Eq. 37 (INNWA) and Eq. 38 (INNWG), printed to 4 decimals, as (T, I, F) intervals.
INNWA = [
    (.5453, .7516, .1682, .3000, .3041, .4373),
    (.4996, .6634, .1552, .2885, .3482, .4656),
    (.3950, .5627, .2000, .3366, .4210, .5533),
    (.6383, .7397, .0000, .2071, .2297, .4040),
]
INNWG = [
    (.5004, .6620, .1761, .3000, .3195, .4422),
    (.4547, .6581, .1861, .3371, .5405, .6786),
    (.3824, .5578, .2000, .3419, .5012, .7070),
    (.6333, .7335, .1555, .2570, .5069, .6632),
]
# Eq. 39 / 40 score intervals s = [t_L + 2 - i_U - f_U, t_U + 2 - i_L - f_L].
S_WA = [(1.8080, 2.2793), (1.7455, 2.1600), (1.5051, 1.9417), (2.0272, 2.5100)]
S_WG = [(1.7582, 2.1664), (1.4390, 1.9315), (1.3335, 1.8566), (1.7131, 2.0711)]


def alternatives():
    nums = [N(*[x for p in r for x in p]) for r in RAW]
    return [nums[3 * a : 3 * a + 3] for a in range(4)]


def test_published_example():
    # Zhang et al. (2014) Section 5, Eqs. 37-40, tolerance 1e-4 (printed precision).
    rows = alternatives()
    for row, ref in zip(rows, INNWA):
        assert b(wa(row, W)) == approx(ref, abs=1.5e-4)
    for row, ref in zip(rows, INNWG):
        assert b(wg(row, W)) == approx(ref, abs=1.5e-4)
    # `score()` is the midpoint of the printed score interval, divided by 3.
    for agg, ref in ((wa, S_WA), (wg, S_WG)):
        for row, (lo, hi) in zip(rows, ref):
            assert agg(row, W).score() == approx((lo + hi) / 6.0, abs=1e-4)
    # Step 4 rankings
    order = lambda agg: sorted(range(4), key=lambda a: -agg(rows[a], W).score())
    assert order(wa) == [3, 0, 1, 2]  # A4 > A1 > A2 > A3
    assert order(wg) == [0, 3, 1, 2]  # A1 > A4 > A2 > A3


def test_aggregation_matches_innwaa_closed_form():
    w = [0.3, 0.7]
    r, g = wa([A, B], w), wg([A, B], w)
    lo = lambda x: (x.t_l, x.i_l, x.f_l)
    up = lambda x: (x.t_u, x.i_u, x.f_u)

    def wa_cf(f):
        t1, i1, f1 = f(A)
        t2, i2, f2 = f(B)
        return (1 - (1 - t1) ** 0.3 * (1 - t2) ** 0.7, i1**0.3 * i2**0.7, f1**0.3 * f2**0.7)

    def wg_cf(f):
        t1, i1, f1 = f(A)
        t2, i2, f2 = f(B)
        return (t1**0.3 * t2**0.7, 1 - (1 - i1) ** 0.3 * (1 - i2) ** 0.7,
                1 - (1 - f1) ** 0.3 * (1 - f2) ** 0.7)

    for res, cf in ((r, wa_cf), (g, wg_cf)):
        (tl, il, fl), (tu, iu, fu) = cf(lo), cf(up)
        assert b(res) == approx((tl, tu, il, iu, fl, fu))


def test_bounds_inversion_and_errors():
    with pytest.raises(ValueError, match="t_l <= t_u"):
        N(0.7 + 1e-9, 0.7, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="i_l <= i_u"):
        N(0, 0, 0.5, 0.2, 0, 0)
    with pytest.raises(ValueError, match="f_l <= f_u"):
        N(0, 0, 0, 0, 0.5, 0.2)
    x = N(0.7 + 1e-13, 0.7, 0.2, 0.2, 0.1, 0.1)  # within TOL: clamped silently
    assert x.t_l == x.t_u == 0.7
    with pytest.raises(TypeError):
        N(True, 1, 0, 0, 0, 0)
    with pytest.raises(TypeError):
        N("0.5", 1, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        N(0, 1.5, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        N(0, math.nan, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="valid: algebraic"):
        A.add(B, ops="einstein")
    with pytest.raises(ValueError, match="valid: average"):
        A.score("normalized")
    with pytest.raises(ValueError, match="valid: difference, certainty"):
        A.accuracy("sum")
    with pytest.raises(ValueError, match="valid: hamming, euclidean"):
        A.distance(B, "chebyshev")
    with pytest.raises(ValueError):
        A.scale(-1)
    with pytest.raises(TypeError):
        A + NeutrosophicFN(0.5, 0.2, 0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        A.t_l = 0.1  # type: ignore[misc]


def test_distance_golden():
    d = [0.1, 0.1, -0.1, -0.1, 0.1, 0.1]
    assert A.distance(B) == approx(sum(map(abs, d)) / 6)
    assert A.distance(B, "euclidean") == approx(math.sqrt(sum(x * x for x in d) / 6))


def test_registered_for_score_names():
    assert "average" in weighting._ALL_SCORES
