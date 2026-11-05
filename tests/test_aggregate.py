import math

import pytest
from hypothesis import given, strategies as st
from pytest import approx

from bulanik import IFN, PFN, QROFN, TFN, NeutrosophicFN, PictureFN, SphericalFN
from bulanik._validate import check_weights
from bulanik.aggregate import aggregate_experts, wa, wg
from strategies import neutrosophics, pictures, qrofns, sphericals


def test_ifn_wa_wg_golden():
    a, b = IFN(0.6, 0.3), IFN(0.5, 0.4)
    r = wa([a, b], [0.5, 0.5])
    assert (r.mu, r.nu) == approx((0.5527864045, 0.3464101615))  # 1-sqrt(.4*.5) ; sqrt(.12)
    g = wg([a, b], [0.5, 0.5])
    assert (g.mu, g.nu) == approx((0.5477225575, 0.3519259302))  # sqrt(.3) ; 1-sqrt(.7*.6)
    e = wa([a, b], [0.5, 0.5], ops="einstein")
    # P=sqrt(1.6*1.5), M=sqrt(.4*.5): (P-M)/(P+M) ; Q=sqrt(.12), R=sqrt(1.7*1.6): 2Q/(R+Q)
    assert (e.mu, e.nu) == approx((0.5519815245, 0.3471648264))


def test_tfn_wa_wg_golden():
    A, B = TFN(1, 2, 3), TFN(3, 4, 5)
    assert wa([A, B], [0.25, 0.75]) == TFN(2.5, 3.5, 4.5)
    g = wg([A, B], [0.25, 0.75])
    assert (g.l, g.m, g.u) == approx((2.2795070570, 3.3635856610, 4.4005586840))  # 1^.25*3^.75 ...


def test_three_part_wa_golden():
    s = wa([SphericalFN(.6, .3, .4), SphericalFN(.5, .5, .2)], [.5, .5])
    # A=sqrt(.64*.75), B=sqrt(.48*.71): mu=sqrt(1-A), nu=sqrt(.15), pi=sqrt(A-B)
    assert s.parts == approx((0.5542379245, 0.3872983346, 0.3302113600))
    p = wa([PictureFN(.5, .2, .2), PictureFN(.3, .3, .3)], [.5, .5])
    assert p.parts == approx((0.4083920217, 0.2449489743, 0.2449489743))  # 1-sqrt(.5*.7), sqrt(.06), sqrt(.06)
    pg = wg([PictureFN(.5, .2, .2), PictureFN(.3, .3, .3)], [.5, .5])
    assert pg.parts == approx((0.4031250956, 0.2449489743, 0.2516685226))  # sqrt(.7*.6)-sqrt(.06) ; 1-sqrt(.8*.7)
    n = wa([NeutrosophicFN(.7, .2, .1), NeutrosophicFN(.4, .5, .3)], [.5, .5])
    assert n.parts == approx((0.5757359313, 0.3162277660, 0.1732050808))


def test_zero_weight_drops_input():
    a, b = IFN(0.6, 0.3), IFN(0.5, 0.4)
    assert (wa([a, b], [1, 0]).mu, wa([a, b], [1, 0]).nu) == approx((0.6, 0.3))
    assert (wg([a, b], [0, 1]).mu, wg([a, b], [0, 1]).nu) == approx((0.5, 0.4))


def test_zero_weight_drops_input_every_type():
    # a picture with a weight of 0 must not be scaled (0 * A is not an identity there)
    p, q = PictureFN(.5, .2, .2), PictureFN(.3, .3, .3)
    for fn in (wa, wg):
        assert fn([p, q], [1, 0]).parts == approx(p.parts)
        assert fn([q, p], [0, 1]).parts == approx(p.parts)
    s, t = SphericalFN(.6, .3, .4), SphericalFN(.5, .5, .2)
    assert wa([s, t], [0, 1]).parts == approx(t.parts)
    assert wg([s, t], [1, 0]).parts == approx(s.parts)
    A, B = TFN(1, 2, 3), TFN(3, 4, 5)
    assert wa([A, B], [0, 1]) == B and wg([A, B], [1, 0]) == A


def test_all_zero_weights_raise():
    with pytest.raises(ValueError):
        wa([IFN(.5, .2)] * 2, [0, 0], normalize=True)
    with pytest.raises(ValueError):
        wg([PictureFN(.5, .2, .2)] * 2, [0, 0], normalize=True)


def test_weight_validation_and_mixing():
    with pytest.raises(ValueError, match="sum to 1"):
        wa([IFN(.5, .2)] * 2, [0.6, 0.6])
    assert wa([IFN(.5, .2)] * 2, [3, 3], normalize=True).mu == approx(0.5)
    with pytest.raises(TypeError):
        wa([IFN(.5, .2), PFN(.5, .2)])
    with pytest.raises(TypeError):
        wa([1.0, 2.0])
    with pytest.raises(ValueError):
        wa([])
    with pytest.raises(ValueError):
        wg([])
    with pytest.raises(ValueError, match="valid: algebraic"):
        wa([TFN(1, 2, 3)], ops="einstein")
    with pytest.raises(ValueError, match="valid: algebraic"):
        wa([PictureFN(.1, .2, .3)], [1.0], ops="einstein")


def test_equal_weights_default():
    a, b = IFN(0.6, 0.3), IFN(0.5, 0.4)
    assert wa([a, b]).mu == approx(wa([a, b], [.5, .5]).mu)


def test_check_weights_empty_raises_value_error():
    with pytest.raises(ValueError):
        check_weights(None, 0)


def test_picture_closed_forms_survive_extremes():
    # mu = 1 and eta = 0 parts: no log(0) leak
    one = PictureFN(1, 0, 0)
    r = wa([one, PictureFN(.2, .3, .4)], [.5, .5])
    assert r.parts == approx((1.0, 0.0, 0.0))
    g = wg([PictureFN(0, 0, 1), PictureFN(.2, .3, .4)], [.5, .5])
    assert g.parts == approx((0.0, 0.0, 1.0))  # mu+eta = 0 -> product 0 ; nu: 1-(0)
    # tiny mu stays accurate (no 1 - larger cancellation)
    t = wg([PictureFN(1e-12, .5, .0), PictureFN(1e-12, .5, .0)], [.5, .5])
    # (0.5+1e-12)^1 - 0.5 = 1e-12
    assert t.mu == approx(1e-12, rel=1e-6)


def test_aggregate_experts_shape_and_values():
    e1 = [[IFN(.6, .3)], [IFN(.5, .4)]]
    e2 = [[IFN(.5, .4)], [IFN(.5, .4)]]
    out = aggregate_experts([e1, e2], [0.5, 0.5])
    assert len(out) == 2 and len(out[0]) == 1 and out[0][0].mu == approx(0.5527864045)
    g = aggregate_experts([e1, e2], [0.5, 0.5], method="wg")
    assert g[0][0].mu == approx(0.5477225575)
    with pytest.raises(ValueError):
        aggregate_experts([e1, [[IFN(.5, .4)]]])  # ragged
    with pytest.raises(ValueError):
        aggregate_experts([])
    with pytest.raises(ValueError, match="valid: wa, wg"):
        aggregate_experts([e1], method="owa")


@pytest.mark.parametrize("strategy", [lambda: qrofns(1), lambda: qrofns(3), sphericals, pictures, neutrosophics])
@pytest.mark.parametrize("fn", [wa, wg])
@given(data=st.data())
def test_idempotent_and_bounded(strategy, fn, data):
    a = data.draw(strategy())
    n = data.draw(st.integers(1, 5))
    w = data.draw(st.lists(st.floats(0.01, 1), min_size=n, max_size=n))
    parts = lambda x: getattr(x, "parts", None) or (x.mu, x.nu)
    assert parts(fn([a] * n, w, normalize=True)) == approx(parts(a), abs=1e-9)


@pytest.mark.parametrize("fn", [wa, wg])
@given(data=st.data())
def test_idempotent_with_zero_weights_and_einstein(fn, data):
    a = data.draw(qrofns(2))
    w = data.draw(st.lists(st.sampled_from([0.0, 0.3, 1.0]), min_size=3, max_size=3).filter(any))
    for ops in ("algebraic", "einstein"):
        r = fn([a] * 3, w, ops=ops, normalize=True)
        assert (r.mu, r.nu) == approx((a.mu, a.nu), abs=1e-9)


def test_picture_boundary_inputs_do_not_raise():
    # nu = 0.999999999 sits on the boundary with rounding noise; 1 - nu taken
    # naively carries relative error 1e-7, which exponents < 1 turned into a
    # spurious "result violates mu + eta + nu <= 1" (found by probing)
    a, b = PictureFN(0.0, 1e-9, 0.999999999), PictureFN(1.0, 0.0, 0.0)
    w = [0.2685565734213724, 0.7314434265786276]
    g = wg([a, b], w)
    assert sum(g.parts) <= 1.0 + 1e-12
    r = wa([a, b], w)
    assert sum(r.parts) <= 1.0 + 1e-12


def test_einstein_wg_golden():
    # mu = 2 prod(mu^w) / (prod((2-mu)^w) + prod(mu^w)),
    # nu = (prod((1+nu)^w) - prod((1-nu)^w)) / (prod((1+nu)^w) + prod((1-nu)^w))
    g = wg([IFN(0.6, 0.3), IFN(0.5, 0.4)], [0.5, 0.5], ops="einstein")
    assert (g.mu, g.nu) == approx((0.5485837704, 0.3510004003))


@pytest.mark.parametrize(
    "q, a, b, wa_gold, wg_gold",
    [
        (2, (.7, .5), (.6, .6), (0.6547409608, 0.5477225575), (0.6480740698, 0.5542379245)),
        (3, (.8, .6), (.7, .7), (0.7569838705, 0.6480740698), (0.7483314774, 0.6560025325)),
        (5, (.9, .7), (.85, .8), (0.8783178879, 0.7483314774), (0.8746427842, 0.7591398033)),
    ],
)
def test_qrofn_wa_wg_goldens(q, a, b, wa_gold, wg_gold):
    # WA: ((1 - prod(1-mu^q)^w)^(1/q), prod nu^w); WG: (prod mu^w, (1 - prod(1-nu^q)^w)^(1/q))
    x, y = QROFN(*a, q), QROFN(*b, q)
    r = wa([x, y], [0.5, 0.5])
    assert (r.mu, r.nu) == approx(wa_gold)
    g = wg([x, y], [0.5, 0.5])
    assert (g.mu, g.nu) == approx(wg_gold)


def test_aggregate_experts_validates_weights_even_without_cells():
    with pytest.raises(ValueError, match="sum to 1"):
        aggregate_experts([[], []], [0.6, 0.6])
    with pytest.raises(ValueError, match="expected 2 weights"):
        aggregate_experts([[[]], [[]]], [1.0])
    assert aggregate_experts([[], []], [0.5, 0.5]) == []


def test_weights_within_1e9_of_one_are_rescaled():
    # documented: accepted and rescaled to sum to exactly 1
    r = wa([IFN(.5, .2)] * 2, [0.5, 0.5 + 5e-10])
    assert (r.mu, r.nu) == approx((0.5, 0.2), abs=1e-12)
    assert "rescaled" in wa.__doc__


def test_wa_wg_return_fuzzy_numbers():
    from bulanik.numbers import FuzzyNumber

    assert isinstance(wa([IFN(.5, .2)]), FuzzyNumber)
    assert isinstance(wg([PictureFN(.5, .2, .2)]), FuzzyNumber)
