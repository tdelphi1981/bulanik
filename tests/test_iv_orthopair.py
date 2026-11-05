import dataclasses
import math

import pytest
from hypothesis import given, strategies as st
from pytest import approx

from bulanik import FFN, IFN, IVFFN, IVIFN, IVPFN, IVQROFN, PFN, QROFN
from bulanik.aggregate import wa, wg
from strategies import ivqrofns, qrofns

A, B = IVIFN(0.4, 0.5, 0.3, 0.4), IVIFN(0.5, 0.6, 0.2, 0.3)


def parts(x):
    return (x.mu_l, x.mu_u, x.nu_l, x.nu_u)


def test_ivifn_algebraic_golden():
    # Xu (2007) IIFWA arithmetic, hand-derived:
    # sum  mu: [.4+.5-.2, .5+.6-.3] = [.7, .8], nu: [.3*.2, .4*.3] = [.06, .12]
    # prod mu: [.4*.5, .5*.6] = [.2, .3], nu: [.3+.2-.06, .4+.3-.12] = [.44, .58]
    # 2A   mu: [1-.6^2, 1-.5^2] = [.64, .75], nu: [.3^2, .4^2] = [.09, .16]
    s, p, k = A + B, A * B, A.scale(2)
    assert parts(s) == approx((0.7, 0.8, 0.06, 0.12))
    assert parts(p) == approx((0.2, 0.3, 0.44, 0.58))
    assert parts(k) == approx((0.64, 0.75, 0.09, 0.16))
    assert parts(2 * A) == parts(k) == parts(A * 2)
    assert parts(A**2) == approx((0.16, 0.25, 0.51, 0.64))


def test_scores_accuracy_distances_golden():
    assert A.score() == approx(0.1) and A.score("normalized") == approx(0.55)
    assert A.accuracy() == approx(0.8)
    assert A.distance(B) == approx(0.1) and A.distance(B, "euclidean") == approx(0.1)
    # pi_L, pi_U of A: (.1, .3); of B: (.1, .3); an unequal-hesitancy pair
    C = IVIFN(0.2, 0.3, 0.1, 0.2)  # pi in [.5, .7]
    d = [0.4 - 0.2, 0.5 - 0.3, 0.3 - 0.1, 0.4 - 0.2, 0.1 - 0.5, 0.3 - 0.7]
    assert A.distance(C) == approx(sum(map(abs, d)) / 4)
    assert A.distance(C, "euclidean") == approx(math.sqrt(sum(x * x for x in d) / 4))
    assert tuple(A.hesitancy) == approx((0.1, 0.3))


def test_higher_rung_scores():
    x = IVPFN(0.6, 0.8, 0.3, 0.5)
    assert x.score() == approx((0.36 + 0.64 - 0.09 - 0.25) / 2)  # Peng and Yang 2016
    assert x.accuracy() == approx((0.36 + 0.64 + 0.09 + 0.25) / 2)
    y = IVFFN(0.5, 0.7, 0.2, 0.6)
    assert y.score() == approx((0.125 + 0.343 - 0.008 - 0.216) / 2)


def test_xu_2007_score_accuracy_ranking():
    # Xu (2007) ranking rule: larger score first, ties broken by larger accuracy.
    # The values below are hand-derived, not taken from Xu's paper.
    # (The published numerical example was not accessible; values are hand-derived.)
    x, y = IVIFN(0.3, 0.5, 0.2, 0.4), IVIFN(0.2, 0.6, 0.3, 0.3)
    assert x.score() == approx(0.1) and y.score() == approx(0.1)
    assert x.accuracy() == approx(0.7) and y.accuracy() == approx(0.7)
    z = IVIFN(0.4, 0.5, 0.1, 0.4)
    assert z.score() == approx(0.2) and z.score() > x.score()
    w = IVIFN(0.1, 0.5, 0.1, 0.3)  # score 0.1, accuracy 0.5
    assert w.score() == approx(x.score()) and w.accuracy() < x.accuracy()


@given(data=st.data(), q=st.sampled_from([1, 2, 3, 5]), ops=st.sampled_from(["algebraic", "einstein"]))
def test_degenerate_interval_matches_qrofn(data, q, ops):
    x, y = data.draw(qrofns(q)), data.draw(qrofns(q))
    k = data.draw(st.floats(0, 20, allow_nan=False))
    X, Y = IVQROFN.from_qrofn(x), IVQROFN.from_qrofn(y)
    for name in ("add", "mul"):
        r, e = getattr(X, name)(Y, ops=ops), getattr(x, name)(y, ops=ops)
        assert parts(r) == approx((e.mu, e.mu, e.nu, e.nu), abs=1e-12)
    for name in ("scale", "power"):
        r, e = getattr(X, name)(k, ops=ops), getattr(x, name)(k, ops=ops)
        assert parts(r) == approx((e.mu, e.mu, e.nu, e.nu), abs=1e-12)
    for s in QROFN.SCORES:
        assert X.score(s) == approx(x.score(s), abs=1e-12)
    assert X.accuracy() == approx(x.accuracy(), abs=1e-12)
    for m in QROFN.DISTANCES:
        assert X.distance(Y, m) == approx(x.distance(y, m), abs=1e-12)
    assert X.hesitancy == approx((x.hesitancy, x.hesitancy), abs=1e-12)
    c = X.complement()
    assert parts(c) == (x.nu, x.nu, x.mu, x.mu)


def _valid(r, q):
    assert r.mu_l <= r.mu_u and r.nu_l <= r.nu_u
    assert r.mu_u**q + r.nu_u**q <= 1 + 1e-12


@given(data=st.data(), q=st.sampled_from([1, 2, 3, 5]), ops=st.sampled_from(["algebraic", "einstein"]))
def test_closure_and_bound_order(data, q, ops):
    x, y = data.draw(ivqrofns(q)), data.draw(ivqrofns(q))
    k = data.draw(st.floats(0, 20, allow_nan=False))
    for r in (x.add(y, ops), x.mul(y, ops), x.scale(k, ops), x.power(k, ops), x.complement()):
        _valid(r, q)
    assert 0.0 <= x.hesitancy[0] <= x.hesitancy[1] <= 1.0


@given(data=st.data(), q=st.sampled_from([1, 2, 3]))
def test_distance_is_a_metric_like(data, q):
    x, y = data.draw(ivqrofns(q)), data.draw(ivqrofns(q))
    for m in IVQROFN.DISTANCES:
        assert x.distance(x, m) == 0
        assert x.distance(y, m) == approx(y.distance(x, m))
        assert 0 <= x.distance(y, m) <= 1 + 1e-12


def test_aggregation_matches_xu_2007_closed_form():
    w = [0.3, 0.7]
    r = wa([A, B], w)
    cf = (
        1 - (1 - A.mu_l) ** 0.3 * (1 - B.mu_l) ** 0.7,
        1 - (1 - A.mu_u) ** 0.3 * (1 - B.mu_u) ** 0.7,
        A.nu_l**0.3 * B.nu_l**0.7,
        A.nu_u**0.3 * B.nu_u**0.7,
    )
    assert parts(r) == approx(cf)
    g = wg([A, B], w)
    dual = (
        A.mu_l**0.3 * B.mu_l**0.7,
        A.mu_u**0.3 * B.mu_u**0.7,
        1 - (1 - A.nu_l) ** 0.3 * (1 - B.nu_l) ** 0.7,
        1 - (1 - A.nu_u) ** 0.3 * (1 - B.nu_u) ** 0.7,
    )
    assert parts(g) == approx(dual)
    for ops in ("algebraic", "einstein"):
        _valid(wa([A, B], w, ops=ops), 1)
        _valid(wg([A, B], w, ops=ops), 1)
        _valid(wa([IVPFN(0.6, 0.8, 0.3, 0.5), IVPFN(0.4, 0.5, 0.5, 0.7)], w, ops=ops), 2)


def test_einstein_sum_golden():
    # Einstein sum of A and B at q = 1, hand-derived from Wang and Liu (2013):
    # mu: (a1+a2)/(1+a1 a2): L .9/1.2 = .75, U 1.1/1.3 = .846154;
    # nu: b1 b2/(1+(1-b1)(1-b2)): L .06/(1+.7*.8)=.038462, U .12/(1+.6*.7)=.084507
    r = A.add(B, ops="einstein")
    assert parts(r) == approx((0.75, 11 / 13, 0.06 / 1.56, 0.12 / 1.42))


def test_einstein_product_and_scale_golden():
    # Einstein product, per end: mu = a1 a2 / (1 + (1-a1)(1-a2)), nu = (b1+b2) / (1 + b1 b2)
    # mu: L .2/1.3 = .153846, U .3/1.2 = .25; nu: L .5/1.06 = .471698, U .7/1.12 = .625
    p = A.mul(B, ops="einstein")
    assert parts(p) == approx((0.2 / 1.3, 0.25, 0.5 / 1.06, 0.625))
    assert parts(p) == approx((0.1538, 0.25, 0.4717, 0.625), abs=5e-5)
    # Einstein 2A: mu = 2a/(1+a^2), nu = 2b^2/((2-b)^2 + b^2)
    # mu: L .8/1.16 = .689655, U 1/1.25 = .8; nu: L .18/2.98 = .060403, U .32/2.72 = .117647
    k = A.scale(2, ops="einstein")
    assert parts(k) == approx((0.8 / 1.16, 0.8, 0.18 / 2.98, 0.32 / 2.72))
    assert parts(k) == approx((0.6897, 0.8, 0.0604, 0.1176), abs=5e-5)


def _prod(xs, ws):
    out = 1.0
    for x, w in zip(xs, ws):
        out *= x**w
    return out


def _ein_a(xs, ws):  # tanh-form part of the Einstein weighted average
    up = _prod([1 + x for x in xs], ws)
    dn = _prod([1 - x for x in xs], ws)
    return (up - dn) / (up + dn)


def _ein_b(xs, ws):  # 2 prod x^w / (prod (2-x)^w + prod x^w)
    num = _prod(xs, ws)
    return 2 * num / (_prod([2 - x for x in xs], ws) + num)


@pytest.mark.parametrize("cls,q", [(IVIFN, 1), (IVPFN, 2)])
def test_einstein_aggregation_closed_form(cls, q):
    w = [0.3, 0.7]
    xs = [cls(0.4, 0.5, 0.3, 0.4), cls(0.5, 0.6, 0.2, 0.3)]
    pw = lambda get: [get(x) ** q for x in xs]  # noqa: E731
    root = 1.0 / q
    a = wa(xs, w, ops="einstein")
    assert parts(a) == approx(tuple(
        v**root for v in (
            _ein_a(pw(lambda x: x.mu_l), w), _ein_a(pw(lambda x: x.mu_u), w),
            _ein_b(pw(lambda x: x.nu_l), w), _ein_b(pw(lambda x: x.nu_u), w),
        )
    ))
    g = wg(xs, w, ops="einstein")
    assert parts(g) == approx(tuple(
        v**root for v in (
            _ein_b(pw(lambda x: x.mu_l), w), _ein_b(pw(lambda x: x.mu_u), w),
            _ein_a(pw(lambda x: x.nu_l), w), _ein_a(pw(lambda x: x.nu_u), w),
        )
    ))


def test_bounds_inversion_and_constraint():
    with pytest.raises(ValueError, match="mu_l <= mu_u"):
        IVIFN(0.5, 0.4, 0.1, 0.2)
    with pytest.raises(ValueError, match="nu_l <= nu_u"):
        IVIFN(0.2, 0.4, 0.3, 0.2)
    with pytest.raises(ValueError, match=r"mu_u\^1 \+ nu_u\^1"):
        IVIFN(0.5, 0.7, 0.1, 0.4)
    assert IVIFN(0.5 + 1e-13, 0.5, 0.1, 0.2).mu_l == 0.5  # within TOL: clamped
    assert IVIFN(0.5, 0.6, 0.2 + 1e-13, 0.2).nu_l == 0.2
    assert IVPFN(0.6, 0.8, 0.5, 0.6).mu_u == 0.8  # .64 + .36 boundary accepted
    with pytest.raises(ValueError):
        IVIFN(0.1, 1.2, 0.1, 0.2)
    with pytest.raises(ValueError):
        IVIFN(float("nan"), 0.5, 0.1, 0.2)
    with pytest.raises(ValueError):
        IVQROFN(0.1, 0.2, 0.1, 0.2, 0)


def test_projection_clamps_lower_to_upper():
    # an upper bound projected a hair below an equal lower bound drags the lower along
    x = IVIFN(0.5, 0.5, 0.5 + 5e-13, 0.5 + 5e-13)
    assert x.mu_u**1 + x.nu_u <= 1 + 4 * 2.3e-16
    assert x.mu_l <= x.mu_u and x.nu_l <= x.nu_u


def test_negative_and_bad_k():
    with pytest.raises(ValueError, match="scale needs k >= 0"):
        A.scale(-1)
    with pytest.raises(ValueError, match="power needs k >= 0"):
        A.power(-1)
    with pytest.raises(ValueError):
        A.scale(float("inf"))
    assert parts(A.scale(0)) == (0, 0, 1, 1)
    assert parts(A.power(0)) == (1, 1, 0, 0)


def test_kind_mixing_names_and_types():
    with pytest.raises(TypeError, match=r"cannot combine IVIFN\(q=1\) with IVPFN\(q=2\)"):
        A.add(IVPFN(0.5, 0.6, 0.2, 0.3))
    with pytest.raises(TypeError):
        A + IVPFN(0.5, 0.6, 0.2, 0.3)
    with pytest.raises(TypeError):
        A + IFN(0.5, 0.4)
    with pytest.raises(TypeError):
        A.add(IVQROFN(0.4, 0.5, 0.3, 0.4, 1))  # subclass vs base
    with pytest.raises(ValueError, match="valid: difference, normalized"):
        A.score("foo")
    with pytest.raises(ValueError, match="valid: sum"):
        A.accuracy("foo")
    with pytest.raises(ValueError, match="valid: hamming, euclidean"):
        A.distance(B, "foo")
    with pytest.raises(ValueError, match="valid: algebraic, einstein"):
        A.add(B, ops="foo")
    with pytest.raises(TypeError):
        IVIFN(True, 0.5, 0.1, 0.2)
    with pytest.raises(TypeError):
        IVIFN("0.1", 0.5, 0.1, 0.2)


def test_from_qrofn_and_replace_repr():
    assert parts(IVIFN.from_qrofn(IFN(0.5, 0.5))) == (0.5, 0.5, 0.5, 0.5)
    assert type(IVIFN.from_qrofn(IFN(0.5, 0.5))) is IVIFN
    with pytest.raises(ValueError, match="q = 1"):
        IVIFN.from_qrofn(PFN(0.5, 0.5))
    with pytest.raises(TypeError):
        IVIFN.from_qrofn((0.5, 0.5))
    assert IVQROFN.from_qrofn(FFN(0.5, 0.5)).q == 3
    assert parts(dataclasses.replace(A, mu_u=0.6)) == (0.4, 0.6, 0.3, 0.4)
    assert repr(A) == "IVIFN(mu_l=0.4, mu_u=0.5, nu_l=0.3, nu_u=0.4)"
    assert repr(IVQROFN(0.4, 0.5, 0.3, 0.4, 4)) == "IVQROFN(mu_l=0.4, mu_u=0.5, nu_l=0.3, nu_u=0.4, q=4)"
    with pytest.raises(ValueError, match="q = 1"):
        IVIFN(0.4, 0.5, 0.3, 0.4, q=2)
    assert IVIFN(0.4, 0.5, 0.3, 0.4, q=1) == A
    with pytest.raises(AttributeError):
        A.mu_l = 0.1
    assert A == IVIFN(0.4, 0.5, 0.3, 0.4) and len({A, IVIFN(0.4, 0.5, 0.3, 0.4)}) == 1


def test_result_violation_raises():
    # an einstein/algebraic result cannot violate; force the check directly
    with pytest.raises(ValueError, match="result violates"):
        A._new_iv((0.9, 0.9), (0.9, 0.9))


def test_weighting_knows_the_score_names():
    from bulanik.mcdm import weighting

    assert "normalized" in weighting._ALL_SCORES


def test_power_of_upper_pair_a_hair_past_the_boundary():
    # found by hypothesis: the upper pair (1e-9, 1.0) sits 1e-18 past mu^2 + nu^2 = 1; the
    # complement clamp of the point law lifts its nu below the lower pair's 1.0 for small k
    x = IVPFN(0.0, 1e-9, 1.0, 1.0)
    r = x.power(0.5)
    assert r.nu_l <= r.nu_u and r.mu_l <= r.mu_u
    assert r.nu_u == approx(1.0, abs=1e-8)


def test_defuzzified_score_ivifn():
    # D = (mu_l + mu_u + (1 - nu_l) + (1 - nu_u) + mu_l mu_u - sqrt((1 - nu_l)(1 - nu_u))) / 4
    # (0.4, 0.5, 0.3, 0.4): (0.4 + 0.5 + 0.7 + 0.6 + 0.2 - sqrt(0.42)) / 4 = 0.4379814825
    assert IVIFN(0.4, 0.5, 0.3, 0.4).score("defuzzified") == approx(0.4379814825398036, abs=1e-12)
    # (1, 1, 0, 0) -> (1 + 1 + 1 + 1 + 1 - 1)/4 = 1; (0, 0, 1, 1) -> 0
    assert IVIFN(1, 1, 0, 0).score("defuzzified") == approx(1.0)
    assert IVIFN(0, 0, 1, 1).score("defuzzified") == approx(0.0)
    assert IVIFN(0.4, 0.5, 0.3, 0.4).score() == approx(0.1)  # default unchanged
    assert IVIFN.DEFAULT_SCORE == "difference"
    for bad in (IVPFN(0.4, 0.5, 0.3, 0.4), IVFFN(0.4, 0.5, 0.3, 0.4), IVQROFN(0.4, 0.5, 0.3, 0.4, 4)):
        with pytest.raises(ValueError, match='"defuzzified" is defined for IVIFN'):
            bad.score("defuzzified")
