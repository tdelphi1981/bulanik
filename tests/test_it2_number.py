import dataclasses
import math

import pytest
from hypothesis import given, settings, strategies as st
from pytest import approx

from bulanik import IT2TrFN, TFN, TrFN
from bulanik.aggregate import wa, wg
from bulanik.mcdm import swara
from strategies import it2trfns

# Chen and Lee (2010b), Table 1 (their IT2 linguistic scale), "Low".
LOW = IT2TrFN((0, 0.1, 0.1, 0.3, 1, 1), (0.05, 0.1, 0.1, 0.2, 0.9, 0.9))


def test_dtrat_golden():
    # Hand-derived from the DTraT formula of Kahraman et al. (2014), not read from a table:
    # upper: ((.3-0) + (1*.1-0) + (1*.1-0))/4 + 0 = .125
    # lower: ((.2-.05) + (.9*.1-.05) + (.9*.1-.05))/4 + .05 = (.15+.04+.04)/4 + .05 = .1075
    # mean .11625
    assert LOW.score() == approx(0.11625)
    assert LOW.score("dtrat") == LOW.score()


def test_add_sub_mul_golden():
    # Chen and Lee (2010a) definitions 2.4/2.5 (add, mul), checked against the
    # definitions as quoted in Ghaemi/Kahraman-style papers; sub is the interval rule.
    s = LOW + LOW
    assert s.upper == approx((0, 0.2, 0.2, 0.6, 1, 1)) and s.lower == approx((0.1, 0.2, 0.2, 0.4, 0.9, 0.9))
    d = LOW - LOW
    # upper: (0-.3, .1-.1, .1-.1, .3-0) ; lower: (.05-.2, 0, 0, .2-.05)
    assert d.upper == approx((-0.3, 0, 0, 0.3, 1, 1)) and d == LOW.sub(LOW)
    assert d.lower == approx((-0.15, 0, 0, 0.15, 0.9, 0.9))
    p = LOW * LOW
    assert p.upper == approx((0, 0.01, 0.01, 0.09, 1, 1))
    assert p.lower == approx((0.0025, 0.01, 0.01, 0.04, 0.9, 0.9))
    # mixed heights take the minimum
    a = IT2TrFN((0, 1, 2, 3, 1, 0.5), (0.5, 1, 2, 2.5, 0.5, 0.4))
    b = IT2TrFN((0, 1, 2, 3, 0.7, 1), (0.5, 1, 2, 2.5, 0.6, 0.4))
    assert (a + b).upper[4:] == (0.7, 0.5) and (a + b).lower[4:] == (0.5, 0.4)
    assert (a * b).upper[4:] == (0.7, 0.5) and (a - b).lower[4:] == (0.5, 0.4)


def test_scale_and_power():
    s = LOW.scale(2)
    assert s.upper == approx((0, 0.2, 0.2, 0.6, 1, 1)) and s.lower == approx((0.1, 0.2, 0.2, 0.4, 0.9, 0.9))
    assert 2 * LOW == LOW * 2 == s
    a = IT2TrFN((0, 1, 2, 3, 1, 0.5), (0.5, 1, 2, 2.5, 0.5, 0.4))
    n = a.scale(-2)  # points reverse and negate, heights swap
    assert n.upper == approx((-6, -4, -2, 0, 0.5, 1)) and n.lower == approx((-5, -4, -2, -1, 0.4, 0.5))
    z = a.scale(0)
    assert z.upper == (0, 0, 0, 0, 1, 0.5)
    q = LOW**2
    assert q.upper == approx((0, 0.01, 0.01, 0.09, 1, 1)) and q.lower == approx((0.0025, 0.01, 0.01, 0.04, 0.9, 0.9))
    r = IT2TrFN((1, 2, 4, 8, 1, 0.5), (2, 3, 4, 6, 0.5, 0.5)).power(-1)
    assert r.upper == approx((1 / 8, 1 / 4, 1 / 2, 1, 0.5, 1)) and r.lower == approx((1 / 6, 1 / 4, 1 / 3, 1 / 2, 0.5, 0.5))
    with pytest.raises(ValueError, match="lowest point"):
        IT2TrFN((0, 1, 2, 3, 1, 1), (0, 1, 2, 3, 1, 1)).power(-1)
    with pytest.raises(ValueError, match="lowest point"):
        IT2TrFN((-1, 1, 2, 3, 1, 1), (-1, 1, 2, 3, 1, 1)).power(0.5)


def test_degenerate_matches_trfn():
    t = TrFN(1, 2, 3, 5)
    x = IT2TrFN.from_trfn(t)
    assert x.upper == x.lower == (1, 2, 3, 5, 1, 1)
    assert x.score("dtrat") == approx(t.score("mean"))
    assert x.distance(IT2TrFN.from_trfn(TrFN(2, 3, 4, 6))) == approx(1.0)
    assert IT2TrFN.from_trfn(TFN(1, 2, 4)).upper == (1, 2, 2, 4, 1, 1)
    assert x.score("centroid") == approx(t.score("centroid"))
    with pytest.raises(TypeError):
        IT2TrFN.from_trfn((1, 2, 3, 4))


def test_centroid():
    # symmetric trapezoids have their centroid at the axis
    x = IT2TrFN((0, 1, 3, 4, 1, 1), (1, 1.5, 2.5, 3, 1, 1))
    assert x.score("centroid") == approx(2.0)
    # triangle (0,1,1,3): x-centroid (0+1+3)/3
    t = IT2TrFN((0, 1, 1, 3, 1, 1), (0, 1, 1, 3, 1, 1))
    assert t.score("centroid") == approx(4 / 3)
    # zero-area polygon: mean of the points (all heights 0 in the lower trapezoid)
    z = IT2TrFN((0, 1, 1, 3, 1, 1), (1, 1, 1, 2, 0, 0))
    assert z.score("centroid") == approx((4 / 3 + (1 + 1 + 1 + 2) / 4) / 2)


def test_chen_lee_hand_derived():
    # Hand-derived from the ranking value of Chen and Lee (2010a), population standard
    # deviations (the paper's Eqs. for S_q divide by 2 and by 4), NOT a printed table value.
    # upper (0,.1,.1,.3;1,1): M = .05, .1, .2 -> .35 ; S1 = .05, S2 = 0, S3 = .1,
    #   S4 = sqrt(((-.125)^2 + (-.025)^2 + (-.025)^2 + .175^2)/4) = sqrt(.011875)
    # lower (.05,.1,.1,.2;.9,.9): M = .075, .1, .15 -> .325 ; S1 = .025, S2 = 0, S3 = .05,
    #   S4 = sqrt(((-.0625)^2 + (-.0125)^2 + (-.0125)^2 + .0875^2)/4) = sqrt(.00296875)
    up = 0.35 - 0.25 * (0.05 + 0 + 0.1 + math.sqrt(0.011875)) + 2
    lo = 0.325 - 0.25 * (0.025 + 0 + 0.05 + math.sqrt(0.00296875)) + 1.8
    assert LOW.score("chen_lee") == approx(up + lo, abs=1e-12)
    assert LOW.score("chen_lee") == approx(4.377885, abs=1e-6)
    # ordering: a larger number ranks higher
    assert (LOW + LOW).score("chen_lee") > LOW.score("chen_lee")


def test_kahraman_ahp_geometric_mean():
    # Linguistic scale of Kahraman et al. (2014), Table 1 (as quoted in the IT2FAHP literature):
    # SS = ((1,2,4,5;1,1),(1.2,2.2,3.8,4.8;.8,.8)), VS = ((5,6,8,9;1,1),(5.2,6.2,7.8,8.8;.8,.8)).
    # The paper prints no value for this row; geometric mean (SS (x) VS)^(1/2) hand-derived:
    # upper sqrt(1*5, 2*6, 4*8, 5*9), lower sqrt(1.2*5.2, 2.2*6.2, 3.8*7.8, 4.8*8.8).
    ss = IT2TrFN((1, 2, 4, 5, 1, 1), (1.2, 2.2, 3.8, 4.8, 0.8, 0.8))
    vs = IT2TrFN((5, 6, 8, 9, 1, 1), (5.2, 6.2, 7.8, 8.8, 0.8, 0.8))
    g = (ss * vs) ** 0.5
    assert g.upper == approx((math.sqrt(5), math.sqrt(12), math.sqrt(32), math.sqrt(45), 1, 1))
    assert g.lower == approx(
        (math.sqrt(6.24), math.sqrt(13.64), math.sqrt(29.64), math.sqrt(42.24), 0.8, 0.8)
    )
    assert g == wg([ss, vs], [0.5, 0.5]) or g.upper == approx(wg([ss, vs], [0.5, 0.5]).upper)
    ee = IT2TrFN((1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1))
    g3 = (ee * ss * vs).power(1 / 3)
    assert g3.upper == approx(tuple(x ** (1 / 3) for x in (5, 12, 32, 45)) + (1, 1))


def test_aggregation_wa_wg():
    a = IT2TrFN((1, 2, 3, 4, 1, 0.8), (1.5, 2, 3, 3.5, 0.5, 0.5))
    b = IT2TrFN((2, 3, 4, 6, 0.7, 1), (2.5, 3, 4, 5, 0.6, 0.7))
    w = [0.3, 0.7]
    r = wa([a, b], w)
    assert r.upper == approx(tuple(0.3 * x + 0.7 * y for x, y in zip(a.upper[:4], b.upper[:4])) + (0.7, 0.8))
    assert r.lower == approx(tuple(0.3 * x + 0.7 * y for x, y in zip(a.lower[:4], b.lower[:4])) + (0.5, 0.5))
    g = wg([a, b], w)
    assert g.upper == approx(tuple(x**0.3 * y**0.7 for x, y in zip(a.upper[:4], b.upper[:4])) + (0.7, 0.8))
    assert g.lower == approx(tuple(x**0.3 * y**0.7 for x, y in zip(a.lower[:4], b.lower[:4])) + (0.5, 0.5))


def test_bounds_inversion_and_fou():
    with pytest.raises(ValueError, match=r"a1U <= a1L"):
        IT2TrFN((0, 1, 2, 3, 1, 1), (-0.5, 1, 2, 3, 0.9, 0.9))  # lower support outside
    with pytest.raises(ValueError, match=r"h1L <= min\(h1U, h2U\)"):
        IT2TrFN((0, 1, 2, 3, 0.5, 0.5), (0.5, 1, 2, 2.5, 0.9, 0.9))  # lower above upper
    with pytest.raises(ValueError, match=r"a1U <= a1L"):  # rises left of the vertical upper edge
        IT2TrFN((1, 1, 2, 3, 1, 1), (0, 1, 2, 3, 0.5, 0.5))
    with pytest.raises(ValueError, match="needs"):
        IT2TrFN((0, 2, 1, 3, 1, 1), (0, 2, 1, 3, 1, 1))  # unordered
    with pytest.raises(ValueError, match="height"):
        IT2TrFN((0, 1, 2, 3, 1.5, 1), (0, 1, 2, 3, 1, 1))
    with pytest.raises(ValueError, match="upper"):
        IT2TrFN((0, 1, 2, 3, 0, 0), (0, 1, 2, 3, 0, 0))
    with pytest.raises(ValueError):
        IT2TrFN((0, 1, 2, math.nan, 1, 1), (0, 1, 2, 3, 1, 1))
    with pytest.raises(ValueError):
        IT2TrFN((0, 1, 2), (0, 1, 2))  # wrong length
    with pytest.raises(TypeError, match="sequence"):
        IT2TrFN(5, (0, 1, 2, 3, 1, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        IT2TrFN((0, 1, 2, 3, True, 1), (0, 1, 2, 3, 1, 1))
    with pytest.raises(ValueError, match="mul"):
        LOW * IT2TrFN.from_trfn(TrFN(-1, 0, 1, 2))  # mul needs >= 0
    # within TOL: projected silently
    x = IT2TrFN((0, 1, 2, 3, 1, 1), (0, 1, 2, 3, 1 + 1e-13, 1))
    assert x.lower[4] == 1.0
    y = IT2TrFN((0, 1, 2, 3, 0.5, 0.5), (0, 1, 2, 3, 0.5 + 1e-13, 0.5))
    assert y.lower[4] <= 0.5 + 1e-12
    with pytest.raises(dataclasses.FrozenInstanceError):
        LOW.upper = (0,) * 6  # type: ignore[misc]


@given(it2trfns(positive=True), it2trfns(positive=True), st.floats(0, 5))
@settings(max_examples=500)
def test_ops_on_nested_numbers_are_closed(a, b, k):
    for f in (lambda: a + b, lambda: a * b, lambda: a - b, lambda: a.scale(k), lambda: a.scale(-k),
              lambda: a**k, lambda: a ** -k, lambda: a * k):
        r = f()  # nested inputs give nested results: never a ValueError
        IT2TrFN(r.upper, r.lower)  # re-validates
    assert a.distance(a) == 0
    assert a.distance(b) == approx(b.distance(a))
    assert a.score("centroid") >= a.upper[0] - 1e-9


def test_subnormal_scale_keeps_containment():
    # Found by hypothesis: scaling by a subnormal collapses widths and the interpolation
    # in the containment check overshot by rounding.
    a = IT2TrFN(
        (4.710437816179904, 7.232480889528254, 15.314172950808322, 72.98880034733983,
         0.9229846359117198, 0.9229846359117198),
        (7.641948478710152, 7.829700292680182, 10.595310455141366, 10.595317893024971,
         9.229846359117198e-10, 1.9132618952765397e-128),
    )
    assert a.scale(5e-324).upper[3] < 1e-300


@given(it2trfns())
def test_strategy_numbers_have_finite_scores(a):
    for m in IT2TrFN.SCORES:
        assert math.isfinite(a.score(m))


def test_names_and_types():
    assert IT2TrFN.OPS == ("algebraic",) and IT2TrFN.ACCURACIES == () and IT2TrFN.DISTANCES == ("vertex",)
    assert IT2TrFN.DEFAULT_SCORE == "dtrat"
    with pytest.raises(ValueError, match="IT2TrFN has no accuracy function"):
        LOW.accuracy()
    with pytest.raises(ValueError, match=r"valid: dtrat, centroid, chen_lee"):
        LOW.score("foo")
    with pytest.raises(ValueError, match="valid: algebraic"):
        LOW.add(LOW, ops="einstein")
    with pytest.raises(ValueError, match="valid: vertex"):
        LOW.distance(LOW, "hamming")
    with pytest.raises(TypeError):
        LOW + TrFN(1, 2, 3, 4)
    with pytest.raises(TypeError):
        LOW - TrFN(1, 2, 3, 4)
    with pytest.raises(TypeError):
        LOW.distance(TrFN(1, 2, 3, 4))
    with pytest.raises(TypeError):
        LOW.scale(True)
    with pytest.raises(ValueError):
        LOW.scale(math.inf)
    with pytest.raises(ValueError):  # overflow becomes ValueError
        IT2TrFN((1e200, 1e200, 1e200, 1e200, 1, 1), (1e200, 1e200, 1e200, 1e200, 1, 1)).power(5)
    w = swara([0.3, 0.2], ["a", "b", "c"], score="dtrat")  # crisp inputs accept the name
    assert w is not None
    assert "IT2TrFN" in repr(LOW)


def test_non_nested_reference_points_are_rejected():
    # Contained (the lower curve never exceeds the upper one) but not nested: a2L < a2U.
    # The nesting a1U <= a1L <= a2L <= a3L <= a4L <= a4U, a2U <= a2L, a3L <= a3U and
    # lower heights <= min(h1U, h2U) is required so that the arithmetic stays closed.
    with pytest.raises(ValueError, match=r"a2U <= a2L"):
        IT2TrFN((0, 2, 3, 4, 1, 1), (0.5, 1, 3, 3.5, 0.5, 0.5))
    with pytest.raises(ValueError, match=r"a3L <= a3U"):
        IT2TrFN((0, 1, 2, 4, 1, 1), (0.5, 1, 3, 3.5, 0.5, 0.5))
    with pytest.raises(ValueError, match=r"a4L <= a4U"):
        IT2TrFN((0, 1, 2, 3, 1, 1), (0.5, 1, 2, 3.5, 0.5, 0.5))
    with pytest.raises(ValueError, match=r"h2L <= min\(h1U, h2U\)"):  # h2U = 0.2 is the cap
        IT2TrFN((0, 1, 2, 3, 1, 0.2), (0.5, 1, 2, 2.5, 0.2, 0.5))
    IT2TrFN((0, 1, 2, 3, 1, 0.2), (0.5, 1, 2, 2.5, 0.2, 0.2))  # on the boundary: accepted
