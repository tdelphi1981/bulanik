import math

import pytest
from conftest import OUT, X
from hypothesis import assume, given
from hypothesis import strategies as st
from pytest import approx

from bulanik.inference import Linear, NoRuleFiredError, SugenoOutput
from bulanik.sets import MembershipFunction, triangular
from bulanik.type2 import (
    IT2MF,
    IT2LinguisticVariable,
    IT2MamdaniFIS,
    IT2SugenoFIS,
    TypeReduced,
    gaussian_uncertain_mean,
    gaussian_uncertain_sigma,
    scaled_triangle,
    type_reduce,
)

YS, LO, HI = [0, 1, 3], [0.2, 0.5, 0.2], [0.6, 1.0, 0.6]


# --------------------------------------------------------------- brief goldens


def test_km_switch_points_golden():
    # y_l: L=0 all-lo (0+.5+.6)/.9=1.222 ; L=1 (0*.6+1*.5+3*.2)/1.3 = 1.1/1.3 ; L=2 (0+1+.6)/1.8=.889 -> min at L=1
    # y_r: R=0 all-hi 2.8/2.2=1.273 ; R=1 (0*.2+1+1.8)/1.8=1.556 ; R=2 (0+.5+1.8)/1.3 = 2.3/1.3 -> max at R=2
    t = type_reduce(YS, LO, HI)
    assert (t.left, t.right, t.output) == approx((11 / 13, 23 / 13, 17 / 13))
    assert (t.left_switch, t.right_switch) == (1, 2)


def test_ekm_equals_km():
    assert type_reduce(YS, LO, HI, "ekm") == type_reduce(YS, LO, HI, "km")


def test_nie_tan_and_bmm_golden():
    nt = type_reduce(YS, LO, HI, "nie_tan")  # (1*1.5+3*.8)/(.8+1.5+.8) = 3.9/3.1
    assert nt.output == approx(39 / 31) and nt.left == nt.right == nt.output and nt.left_switch is None
    b = type_reduce(YS, LO, HI, "bmm")  # .5*(1.1/.9) + .5*(2.8/2.2)
    assert b.output == approx(0.5 * 1.1 / 0.9 + 0.5 * 2.8 / 2.2) and (b.left, b.right) == approx(
        (1.1 / 0.9, 2.8 / 2.2)
    )


def test_unsorted_input_same_result():
    assert type_reduce([3, 0, 1], [0.2, 0.2, 0.5], [0.6, 0.6, 1]).output == approx(17 / 13)


def test_type_reduce_validation():
    with pytest.raises(ValueError):
        type_reduce([0, 1], [0.5, 0.2], [0.4, 0.3])  # lo > hi
    with pytest.raises(ValueError, match="valid: km, ekm, nie_tan, bmm"):
        type_reduce(YS, LO, HI, "cos")
    with pytest.raises(ValueError, match="km"):
        type_reduce(YS, [0, 0, 0], HI, "bmm")


@given(
    st.lists(st.tuples(st.floats(-10, 10), st.floats(0, 1), st.floats(0, 1)), min_size=1, max_size=8)
)
def test_km_is_exhaustive_optimum(rows):
    ys = [r[0] for r in rows]
    lo = [min(r[1], r[2]) for r in rows]
    hi = [max(r[1], r[2]) for r in rows]
    assume(sum(hi) > 1e-6 and sum(lo) > 1e-6)
    order = sorted(range(len(ys)), key=ys.__getitem__)

    def avg(ws):
        return sum(ys[i] * w for i, w in zip(order, ws)) / sum(ws)

    cands_l = [avg([hi[i] if k < L else lo[i] for k, i in enumerate(order)]) for L in range(len(ys) + 1)]
    cands_r = [avg([lo[i] if k < R else hi[i] for k, i in enumerate(order)]) for R in range(len(ys) + 1)]
    t = type_reduce(ys, lo, hi)
    e = type_reduce(ys, lo, hi, "ekm")
    assert t.left == approx(min(cands_l), abs=1e-9) and t.right == approx(max(cands_r), abs=1e-9)
    assert (e.left, e.right) == approx((t.left, t.right), abs=1e-9)


def test_it2_mf_golden():
    um = gaussian_uncertain_mean(4, 6, 1)
    assert um(5) == approx((math.exp(-0.5), 1.0))  # lower=min(g4,g6)
    assert um(3) == approx((math.exp(-4.5), math.exp(-0.5)))  # lower g6(3), upper g4(3)
    assert gaussian_uncertain_sigma(0, 1, 2)(2) == approx((math.exp(-2), math.exp(-0.5)))
    assert scaled_triangle(0, 5, 10, 0.5)(2.5) == approx((0.25, 0.5))
    with pytest.raises(ValueError):
        IT2MF(upper=triangular(0, 5, 10), lower=triangular(0, 5, 20))(8)


def test_collapse_to_type1_mamdani(tipper):
    t1 = tipper()
    it2 = IT2MamdaniFIS(
        [IT2LinguisticVariable.from_type1(v) for v in t1.inputs],
        [IT2LinguisticVariable.from_type1(v) for v in t1.outputs],
        t1.rules,
        resolution=3001,
    )
    r = it2.evaluate_interval({"service": 3, "food": 8})["tip"]
    assert r.left == approx(r.right, abs=1e-9) and r.output == approx(15.8620689655, abs=1e-6)


def test_collapse_to_type1_sugeno():
    rules = ["IF x IS low THEN y IS f_low", "IF x IS high THEN y IS f_high WITH 0.5"]  # Task 7 golden: 4.25
    r = IT2SugenoFIS([IT2LinguisticVariable.from_type1(X)], [OUT], rules).evaluate_interval({"x": 4})["y"]
    assert r.left == approx(r.right, abs=1e-12) and r.output == approx(4.25)


def test_fou_widens_interval(tipper):
    t1 = tipper()
    tip = IT2LinguisticVariable(
        "tip",
        (0, 30),
        {k: scaled_triangle(*t1.outputs[0][k].params, 0.5) for k in ("cheap", "average", "generous")},
    )
    it2 = IT2MamdaniFIS(
        [IT2LinguisticVariable.from_type1(v) for v in t1.inputs], [tip], t1.rules, resolution=3001
    )
    r = it2.evaluate_interval({"service": 3, "food": 8})["tip"]
    assert r.left < r.right and r.left <= r.output <= r.right


# --------------------------------------------------------------- type_reduce extras


def test_km_with_zero_lower_weights():
    # lo all 0, hi = (0, 1, 1) at y = (0, 1, 2).
    # y_l: hi for k <= L; L=0,1 have weight 0 (undefined), L=2 -> 1/1 = 1, L=3 -> 3/2 -> min 1
    # y_r: lo for k <= R; R=0,1 -> (1+2)/2 = 1.5, R=2 -> 2/1 = 2, R=3 undefined -> max 2
    for method in ("km", "ekm"):
        t = type_reduce([0, 1, 2], [0, 0, 0], [0, 1, 1], method)
        assert (t.left, t.right, t.output) == (1.0, 2.0, 1.5)
        assert (t.left_switch, t.right_switch) == (2, 2)


def test_km_rounded_mean_past_last_weighted_sample():
    # ys (.8, .8, 2), lo all 0, hi (.9, .5, 0): the mid-weight mean
    # (.8*.45 + .8*.25)/.7 rounds to .8000000000000002 > .8, so a plain
    # "count y < c" switch would put both .8 samples on lo weights and leave
    # 0/0; the true interval is [.8, .8].
    for method in ("km", "ekm"):
        t = type_reduce([0.8, 0.8, 2.0], [0.0, 0.0, 0.0], [0.9, 0.5, 0.0], method)
        assert (t.left, t.right) == approx((0.8, 0.8), abs=1e-15)


def test_lower_equals_upper_collapses_exactly():
    # every switch point gives the same weights, so left == right == weighted mean
    ys, w = [0.3, 2.0, 7.5, 9.0], [0.1, 0.7, 0.4, 0.05]
    mean = math.fsum(y * v for y, v in zip(ys, w)) / math.fsum(w)
    for method in ("km", "ekm", "nie_tan", "bmm"):
        t = type_reduce(ys, w, w, method)
        assert t.left == t.right == t.output == mean


def test_single_point_and_custom_bmm():
    t = type_reduce([4.0], [0.2], [0.9])
    assert (t.left, t.right, t.output) == (4.0, 4.0, 4.0)
    # bmm m=.2, n=.8: .2*(1.1/.9) + .8*(2.8/2.2) ; left/right sorted
    b = type_reduce(YS, LO, HI, "bmm", bmm_m=0.2, bmm_n=0.8)
    assert b.output == approx(0.2 * 1.1 / 0.9 + 0.8 * 2.8 / 2.2)
    assert (b.left, b.right) == approx((1.1 / 0.9, 2.8 / 2.2))
    assert b.left_switch is None and b.right_switch is None


def test_bmm_left_right_are_sorted():
    # lower average (1*.1 + 0*.1)/.2 = .5 ; upper average (1*.1 + 0*1)/1.1 = 1/11
    # the lower average is the larger one, so (left, right) = (1/11, .5)
    b = type_reduce([1, 0], [0.1, 0.1], [0.1, 1.0], "bmm")
    assert (b.left, b.right) == approx((1 / 11, 0.5))
    assert b.output == approx(0.5 * 0.5 + 0.5 / 11)


def test_type_reduce_more_validation():
    with pytest.raises(ValueError, match="same length"):
        type_reduce([0, 1], [0.1], [0.2, 0.3])
    with pytest.raises(ValueError, match="finite"):
        type_reduce([0, math.inf], [0.1, 0.1], [0.2, 0.3])
    with pytest.raises(ValueError, match="finite"):
        type_reduce([0, 1], [0.1, math.nan], [0.2, 0.3])
    with pytest.raises(ValueError, match="negative"):
        type_reduce([0, 1], [-0.1, 0.1], [0.2, 0.3])
    with pytest.raises(ValueError, match="all zero"):
        type_reduce([0, 1], [0, 0], [0, 0])
    with pytest.raises(ValueError, match="all zero"):
        type_reduce([], [], [])
    with pytest.raises(ValueError, match="finite"):
        type_reduce(YS, LO, HI, "bmm", bmm_m=math.nan)


def test_type_reduce_accepts_float_noise():
    # lo exceeds hi by 1e-13 and lo is -1e-13: both within TOL, clamped
    t = type_reduce([0, 1], [-1e-13, 0.5 + 1e-13], [0.5, 0.5])
    # weights effectively lo=(0, .5), hi=(.5, .5): y_l: L=1 -> (0*.5 + 1*.5)/1 = .5 ; y_r = 1
    assert (t.left, t.right) == approx((0.5, 1.0))


def test_type_reduced_is_frozen():
    t = TypeReduced(1.0, 2.0, 1.5, 0, 1)
    with pytest.raises(AttributeError):
        t.left = 0.0  # type: ignore[misc]


# --------------------------------------------------------------- sets and variables


def test_new_membership_kinds():
    flat = MembershipFunction("gaussian_flat_top", (4, 6, 1))
    assert flat(5) == 1.0 and flat(7) == approx(math.exp(-0.5)) and flat(3) == approx(math.exp(-0.5))
    pair = MembershipFunction("gaussian_min_pair", (4, 6, 1))
    assert pair(4) == approx(math.exp(-2))  # min(g4(4)=1, g6(4)=e^-2)
    st_ = MembershipFunction("scaled_triangular", (0, 5, 10, 0.4))
    assert st_(5) == approx(0.4) and st_(11) == 0.0 and st_(7.5) == approx(0.2)
    for mf in (flat, pair, st_):
        assert not mf.is_monotonic
    with pytest.raises(ValueError, match="scaled_triangular needs 4 parameters, got 3"):
        MembershipFunction("scaled_triangular", (0, 5, 10))
    with pytest.raises(ValueError, match="m1 <= m2"):
        MembershipFunction("gaussian_flat_top", (6, 4, 1))
    with pytest.raises(ValueError, match="sigma > 0"):
        MembershipFunction("gaussian_min_pair", (4, 6, 0))
    with pytest.raises(ValueError, match="0 < height <= 1"):
        MembershipFunction("scaled_triangular", (0, 5, 10, 0))
    with pytest.raises(ValueError, match="0 < height <= 1"):
        scaled_triangle(0, 5, 10, 1.5)
    with pytest.raises(ValueError, match="a <= b <= c"):
        scaled_triangle(5, 0, 10, 0.5)


def test_factory_validation():
    with pytest.raises(ValueError, match="sigma_low"):
        gaussian_uncertain_sigma(0, 2, 1)
    with pytest.raises(ValueError, match="sigma_low"):
        gaussian_uncertain_sigma(0, 0, 1)
    with pytest.raises(ValueError, match="m1 <= m2"):
        gaussian_uncertain_mean(6, 4, 1)
    with pytest.raises(TypeError, match="MembershipFunction"):
        IT2MF(upper=triangular(0, 5, 10), lower=0.5)  # type: ignore[arg-type]


def test_it2mf_bounds_property():
    # the lower never exceeds the upper for every factory, on a grid
    mfs = [gaussian_uncertain_mean(2, 3, 0.7), gaussian_uncertain_sigma(1, 0.5, 2), scaled_triangle(0, 1, 4, 0.3)]
    for mf in mfs:
        for i in range(-20, 61):
            lo, hi = mf(i / 10)
            assert 0.0 <= lo <= hi <= 1.0


def test_it2mf_clamps_float_noise():
    # at the peak the lower exceeds the upper by 1e-13: within TOL, clamped to the upper
    up = MembershipFunction("scaled_triangular", (0, 5, 10, 0.5))
    low = MembershipFunction("scaled_triangular", (0, 5, 10, 0.5 + 1e-13))
    lo, hi = IT2MF(upper=up, lower=low)(5)
    assert lo == hi == 0.5


def test_it2_linguistic_variable():
    v = IT2LinguisticVariable("x", (0, 10), {"mid": scaled_triangle(0, 5, 10, 0.5)})
    assert v.term_names == ("mid",)
    assert v.fuzzify(2.5) == {"mid": approx((0.25, 0.5))}
    with pytest.raises(ValueError, match=r"'x' input 11 is outside its universe \[0, 10\]"):
        v.fuzzify(11)
    with pytest.raises(KeyError, match="valid: mid"):
        v["low"]
    with pytest.raises(TypeError, match="IT2MF"):
        IT2LinguisticVariable("x", (0, 10), {"mid": triangular(0, 5, 10)})
    with pytest.raises(ValueError, match="lo < hi"):
        IT2LinguisticVariable("x", (10, 0), {"mid": scaled_triangle(0, 5, 10, 0.5)})
    with pytest.raises(ValueError, match="at least one term"):
        IT2LinguisticVariable("x", (0, 10), {})
    w = IT2LinguisticVariable.from_type1(X)
    assert w.name == "x" and w.universe == (0.0, 10.0) and w.fuzzify(4) == {"low": (0.6, 0.6), "high": (0.4, 0.4)}
    assert "IT2LinguisticVariable('x'" in repr(w)
    with pytest.raises(TypeError, match="LinguisticVariable"):
        IT2LinguisticVariable.from_type1(w)  # type: ignore[arg-type]


# --------------------------------------------------------------- FIS


def _x2():
    return IT2LinguisticVariable(
        "x",
        (0, 10),
        {"low": IT2MF(upper=triangular(0, 0, 10), lower=MembershipFunction("scaled_triangular", (0, 0, 10, 0.5))),
         "high": IT2MF(upper=triangular(0, 10, 10), lower=MembershipFunction("scaled_triangular", (0, 10, 10, 0.5)))},
    )


def test_sugeno_firing_intervals_not_and_weight():
    # x=4: low (.3, .6), high (.2, .4). NOT high -> (1-.4, 1-.2) = (.6, .8); WITH .5 halves both bounds
    fis = IT2SugenoFIS(
        [_x2()],
        [SugenoOutput("y", {"a": 0.0, "b": 10.0})],
        ["IF x IS NOT high THEN y IS a WITH 0.5", "IF x IS high THEN y IS b"],
    )
    ex = fis.explain({"x": 4})
    assert ex.degrees["x"]["low"] == approx((0.3, 0.6))
    assert ex.activations[0].strength == approx((0.3, 0.4))
    assert ex.activations[1].strength == approx((0.2, 0.4))
    # centre of sets, y = (0, 10): y_l uses hi at 0, lo at 10 -> 10*.2/(.4+.2) = 10/3
    # y_r uses lo at 0, hi at 10 -> 10*.4/(.3+.4) = 40/7
    r = fis.evaluate_interval({"x": 4})["y"]
    assert (r.left, r.right) == approx((10 / 3, 40 / 7))
    assert fis.evaluate({"x": 4})["y"] == approx((10 / 3 + 40 / 7) / 2) == ex.outputs["y"]


def test_sugeno_and_or_intervals_first_order():
    # x=4 low (.3,.6) high (.2,.4); AND min -> (.2,.4); OR max -> (.3,.6)
    # z for 'lin' = 1 + .5*4 = 3 ; 'c' = 8
    fis = IT2SugenoFIS(
        [_x2()],
        [SugenoOutput("y", {"lin": Linear({"x": 0.5}, 1.0), "c": 8.0})],
        ["IF x IS low AND x IS high THEN y IS c", "IF x IS low OR x IS high THEN y IS lin"],
        type_reduction="nie_tan",
    )
    ex = fis.explain({"x": 4})
    assert [a.strength for a in ex.activations] == [approx((0.2, 0.4)), approx((0.3, 0.6))]
    # nie-tan: (8*.6 + 3*.9)/(.6+.9) = 7.5/1.5 = 5
    assert ex.outputs["y"] == approx(5.0)


def test_mamdani_interval_and_methods():
    out = IT2LinguisticVariable(
        "y", (0, 10), {"m": scaled_triangle(0, 5, 10, 0.5), "r": scaled_triangle(5, 10, 10, 0.5)}
    )
    fis = IT2MamdaniFIS([_x2()], [out], ["IF x IS low THEN y IS m", "IF x IS high THEN y IS r"])
    a = fis.evaluate_interval({"x": 4})["y"]
    assert a.left < a.output < a.right
    # the four reducers agree to within the FOU interval, and ekm == km
    for method in ("ekm", "nie_tan", "bmm"):
        b = IT2MamdaniFIS([_x2()], [out], fis.rules, type_reduction=method).evaluate({"x": 4})["y"]
        assert a.left - 1e-9 <= b <= a.right + 1e-9
    e = IT2MamdaniFIS([_x2()], [out], fis.rules, type_reduction="ekm").evaluate_interval({"x": 4})["y"]
    assert (e.left, e.right) == approx((a.left, a.right), abs=1e-12)
    ex = fis.explain({"x": 4})
    assert ex.outputs["y"] == a.output and ex.activations[0].strength == approx((0.3, 0.6))


def test_mamdani_options_collapse_to_type1(tipper):
    # every option combination of the type-1 FIS is reproduced when lower = upper
    for kw in ({"and_method": "prod", "or_method": "probor"}, {"implication": "prod", "aggregation": "sum"},
               {"aggregation": "probor"}):
        t1 = tipper(**kw)
        it2 = IT2MamdaniFIS(
            [IT2LinguisticVariable.from_type1(v) for v in t1.inputs],
            [IT2LinguisticVariable.from_type1(v) for v in t1.outputs],
            t1.rules,
            resolution=3001,
            **kw,
        )
        r = it2.evaluate_interval({"service": 3, "food": 8})["tip"]
        assert r.left == r.right == t1.evaluate({"service": 3, "food": 8})["tip"]


def test_no_rule_fired_policy():
    x = IT2LinguisticVariable("x", (0, 10), {"low": scaled_triangle(0, 0, 5, 0.5)})
    y = IT2LinguisticVariable("y", (0, 10), {"m": scaled_triangle(0, 5, 10, 0.5)})
    fis = IT2MamdaniFIS([x], [y], ["IF x IS low THEN y IS m"])
    with pytest.raises(NoRuleFiredError, match="'y'"):
        fis.evaluate({"x": 8})
    d = IT2MamdaniFIS([x], [y], fis.rules, default={"y": 4.0})
    assert d.evaluate({"x": 8}) == {"y": 4.0}
    assert d.evaluate_interval({"x": 8})["y"] == TypeReduced(4.0, 4.0, 4.0, None, None)
    s = IT2SugenoFIS([x], [SugenoOutput("y", {"m": 3.0})], ["IF x IS low THEN y IS m"])
    with pytest.raises(NoRuleFiredError, match="'y'"):
        s.evaluate({"x": 8})
    s2 = IT2SugenoFIS([x], [SugenoOutput("y", {"m": 3.0})], s.rules, default={"y": -1.0})
    assert s2.evaluate({"x": 8}) == {"y": -1.0}
    with pytest.raises(ValueError, match="outside its universe"):
        IT2MamdaniFIS([x], [y], fis.rules, default={"y": 40.0})


def test_fis_construction_errors():
    x = _x2()
    y = IT2LinguisticVariable("y", (0, 10), {"m": scaled_triangle(0, 5, 10, 0.5)})
    with pytest.raises(TypeError, match="IT2MamdaniFIS inputs must be IT2LinguisticVariable"):
        IT2MamdaniFIS([X], [y], ["IF x IS low THEN y IS m"])
    with pytest.raises(TypeError, match="IT2MamdaniFIS outputs must be IT2LinguisticVariable"):
        IT2MamdaniFIS([x], [OUT], ["IF x IS low THEN y IS f_low"])
    with pytest.raises(TypeError, match="IT2SugenoFIS inputs must be IT2LinguisticVariable"):
        IT2SugenoFIS([X], [OUT], ["IF x IS low THEN y IS f_low"])
    with pytest.raises(TypeError, match="IT2SugenoFIS outputs must be SugenoOutput"):
        IT2SugenoFIS([x], [y], ["IF x IS low THEN y IS m"])
    with pytest.raises(ValueError, match="unknown type reduction method 'cos'"):
        IT2MamdaniFIS([x], [y], ["IF x IS low THEN y IS m"], type_reduction="cos")
    with pytest.raises(ValueError, match="unknown and_method"):
        IT2SugenoFIS([x], [OUT], ["IF x IS low THEN y IS f_low"], and_method="x")
    with pytest.raises(ValueError, match="resolution"):
        IT2MamdaniFIS([x], [y], ["IF x IS low THEN y IS m"], resolution=1)
    with pytest.raises(ValueError, match="unknown term 'big'"):
        IT2MamdaniFIS([x], [y], ["IF x IS big THEN y IS m"])
    with pytest.raises(ValueError, match="unknown input 'q'"):
        IT2SugenoFIS([x], [SugenoOutput("y", {"f": Linear({"q": 1.0})})], ["IF x IS low THEN y IS f"])
    bad = IT2LinguisticVariable("y", (0, 10), {"m": IT2MF(upper=triangular(0, 5, 10), lower=triangular(0, 5, 20))})
    with pytest.raises(ValueError, match="lower"):
        IT2MamdaniFIS([x], [bad], ["IF x IS low THEN y IS m"])


def test_fis_input_errors_and_properties():
    y = IT2LinguisticVariable("y", (0, 10), {"m": scaled_triangle(0, 5, 10, 0.5)})
    fis = IT2MamdaniFIS([_x2()], [y], ["IF x IS low THEN y IS m"], type_reduction="nie_tan", resolution=11)
    with pytest.raises(ValueError, match=r"missing input 'x'; inputs: 'x' on \[0, 10\]"):
        fis.evaluate({})
    with pytest.raises(ValueError, match="outside its universe"):
        fis.evaluate({"x": -1})
    assert fis.type_reduction == "nie_tan" and fis.resolution == 11 and fis.default is None
    assert (fis.and_method, fis.or_method, fis.implication, fis.aggregation) == ("min", "max", "min", "max")
    assert [v.name for v in fis.inputs] == ["x"] and len(fis.rules) == 1 and fis.outputs[0] is y
    with pytest.raises(AttributeError):
        fis.rules = ()  # type: ignore[misc]
    s = IT2SugenoFIS([_x2()], [OUT], ["IF x IS low THEN y IS f_low"], default={"y": 1.0})
    assert s.type_reduction == "km" and s.default == {"y": 1.0} and s.outputs[0] is OUT
    assert (s.and_method, s.or_method) == ("min", "max") and s.inputs[0].name == "x" and len(s.rules) == 1


def test_it2_consequent_term_without_positive_upper_sample_is_rejected():
    from bulanik.sets import LinguisticVariable

    tip = IT2LinguisticVariable.from_type1(
        LinguisticVariable("tip", (0, 30), {"cheap": triangular(0, 5, 10), "generous": triangular(30, 35, 40)})
    )
    with pytest.raises(ValueError, match=r"'tip'.*'generous'"):
        IT2MamdaniFIS([IT2LinguisticVariable.from_type1(X)], [tip], ["IF x IS high THEN tip IS generous"])


def test_type_reduce_rejects_str_and_bool():
    with pytest.raises(TypeError):
        type_reduce(["0", 1, 3], LO, HI)
    with pytest.raises(TypeError):
        type_reduce(YS, [True, 0.5, 0.2], HI)
    with pytest.raises(TypeError):
        type_reduce(YS, LO, [0.6, "1.0", 0.6])


# ------------------------------------------------------------ batch C minors


def test_ekm_and_km_on_plateaus_agree_in_value_not_in_switch_index():
    import random

    # equal ys / equal weights: several switch points are equally optimal
    ys, lo, hi = [1.0, 1.0], [0.5, 0.0], [0.5, 0.75]
    k, e = type_reduce(ys, lo, hi, "km"), type_reduce(ys, lo, hi, "ekm")
    assert (k.left, k.right) == (e.left, e.right) == (1.0, 1.0)
    assert (k.left_switch, k.right_switch) != (e.left_switch, e.right_switch)
    rng = random.Random(3)
    differing = 0
    for _ in range(600):
        n = rng.randint(2, 7)
        ys = [float(rng.randint(0, 2)) for _ in range(n)]
        lo = [rng.choice([0, 0.25, 0.5]) for _ in range(n)]
        hi = [a + rng.choice([0, 0.25, 0.5]) for a in lo]
        hi[rng.randrange(n)] += 0.25
        k, e = type_reduce(ys, lo, hi, "km"), type_reduce(ys, lo, hi, "ekm")
        assert e.left == approx(k.left, abs=1e-12) and e.right == approx(k.right, abs=1e-12)
        assert e.left <= e.output <= e.right
        differing += (k.left_switch, k.right_switch) != (e.left_switch, e.right_switch)
    assert differing > 0  # the plateau case is genuinely exercised


def test_ekm_start_switch_is_clamped_to_the_positive_weights():
    # all the mass sits at one end, so the n/2.4 and n/1.7 starting guesses are outside the valid range
    ys = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for lo, hi in (([0.0] * 7, [0.0] * 6 + [1.0]), ([0.0] * 7, [1.0] + [0.0] * 6)):
        k, e = type_reduce(ys, lo, hi, "km"), type_reduce(ys, lo, hi, "ekm")
        assert (e.left, e.right, e.output) == (k.left, k.right, k.output)


def test_bmm_weights_are_documented_as_unconstrained():
    assert "unconstrained" in (type_reduce.__doc__ or "")
    b = type_reduce(YS, LO, HI, "bmm", bmm_m=-1.0, bmm_n=3.0)  # only finiteness is required
    a_lo, a_hi = sum(y * w for y, w in zip(YS, LO)) / sum(LO), sum(y * w for y, w in zip(YS, HI)) / sum(HI)
    assert b.output == approx(-a_lo + 3.0 * a_hi)


def test_bmm_with_all_zero_lower_firing_has_a_clear_fis_level_error():
    from bulanik.type2 import NoLowerFiringError

    # at x=5 the lower MF is 0 but the upper is 0.5: the rule fires possibly, never certainly
    zero_lower = IT2MF(upper=triangular(0, 0, 10), lower=MembershipFunction("scaled_triangular", (0, 0, 2, 1.0)))
    x = IT2LinguisticVariable("x", (0, 10), {"low": zero_lower})
    y = IT2LinguisticVariable("y", (0, 10), {"m": zero_lower})
    for make in (
        lambda **kw: IT2MamdaniFIS([x], [y], ["IF x IS low THEN y IS m"], resolution=51, **kw),
        lambda **kw: IT2SugenoFIS([x], [SugenoOutput("y", {"m": 3.0})], ["IF x IS low THEN y IS m"], **kw),
    ):
        with pytest.raises(NoLowerFiringError, match=r"output 'y'.*bmm.*lower firing is 0.*km") as info:
            make(type_reduction="bmm").evaluate({"x": 5})
        assert isinstance(info.value, ValueError)
        assert make(type_reduction="km").evaluate({"x": 5})["y"] > 0  # the other reducers are fine
    with pytest.raises(NoLowerFiringError, match="every lower weight is 0"):
        type_reduce(YS, [0, 0, 0], HI, "bmm")


def test_type2_default_may_be_an_interval():
    x = IT2LinguisticVariable("x", (0, 10), {"low": scaled_triangle(0, 0, 5, 0.5)})
    y = IT2LinguisticVariable("y", (0, 10), {"m": scaled_triangle(0, 5, 10, 0.5)})
    fis = IT2MamdaniFIS([x], [y], ["IF x IS low THEN y IS m"], default={"y": (2.0, 6.0)})
    assert fis.evaluate_interval({"x": 8})["y"] == TypeReduced(2.0, 6.0, 4.0, None, None)
    assert fis.evaluate({"x": 8}) == {"y": 4.0}
    assert fis.explain({"x": 8}).outputs == {"y": 4.0}
    s = IT2SugenoFIS([x], [SugenoOutput("y", {"m": 3.0})], ["IF x IS low THEN y IS m"], default={"y": [-1.0, 1.0]})
    assert s.evaluate_interval({"x": 8})["y"].output == 0.0
    for bad, msg in (((1.0, 2.0, 3.0), "pair"), ((6.0, 2.0), "above right"), ((2.0, 40.0), "outside its universe"),
                     ((float("nan"), 1.0), "finite")):
        with pytest.raises(ValueError, match=msg):
            IT2MamdaniFIS([x], [y], fis.rules, default={"y": bad})
    # type-1 systems keep scalar-only defaults
    from conftest import TIP
    from bulanik.inference import MamdaniFIS

    with pytest.raises(TypeError):
        MamdaniFIS([_x2_t1()], [TIP], ["IF x IS low THEN tip IS cheap"], default={"tip": (1.0, 2.0)})


def _x2_t1():
    from conftest import X

    return X


def test_type2_no_rule_fired_carries_partial_explanation():
    x = IT2LinguisticVariable("x", (0, 10), {"low": scaled_triangle(0, 0, 5, 0.5)})
    y = IT2LinguisticVariable("y", (0, 10), {"m": scaled_triangle(0, 5, 10, 0.5)})
    for fis in (
        IT2MamdaniFIS([x], [y], ["IF x IS low THEN y IS m"]),
        IT2SugenoFIS([x], [SugenoOutput("y", {"m": 3.0})], ["IF x IS low THEN y IS m"]),
    ):
        with pytest.raises(NoRuleFiredError) as info:
            fis.explain({"x": 8})
        e = info.value
        assert e.output == "y" and e.explanation.degrees["x"]["low"] == (0.0, 0.0)
        assert e.explanation.activations[0].strength == (0.0, 0.0)


def test_type2_helpers_come_from_inference_common():
    from bulanik.inference import _common
    import bulanik.type2 as t2

    assert t2._imply is _common.imply and t2._aggregate is _common.aggregate
    assert t2._AND is _common.AND and t2.check_linear_inputs is _common.check_linear_inputs
