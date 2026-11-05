import math

import pytest
from conftest import X, Z
from pytest import approx

from bulanik.inference import NoRuleFiredError, TsukamotoFIS
from bulanik.sets import LinguisticVariable, linear_increasing, sigmoid, trapezoidal, triangular

RULES = ["IF x IS low THEN z IS small", "IF x IS high THEN z IS large"]


def test_tsukamoto_golden():
    fis = TsukamotoFIS([X], [Z], RULES)
    # x=3: a=(.7,.3); z_small = 60-40*.7 = 32 ; z_large = 40+60*.3 = 58 ; (.7*32+.3*58)/1
    assert fis.evaluate({"x": 3})["z"] == approx(39.8)
    # x=8: a=(.2,.8); 52 and 88 ; .2*52+.8*88
    assert fis.evaluate({"x": 8})["z"] == approx(80.8)


def test_tsukamoto_refuses_non_monotonic():
    Zm = LinguisticVariable("z", (0, 1), {"mid": triangular(0, 0.5, 1)})
    with pytest.raises(ValueError, match="mid.*monotonic"):
        TsukamotoFIS([X], [Zm], ["IF x IS low THEN z IS mid"])
    Zt = LinguisticVariable("z", (0, 1), {"ok": sigmoid(1, 0), "plat": trapezoidal(0, 0.2, 0.4, 1)})
    with pytest.raises(ValueError, match="plat"):
        TsukamotoFIS([X], [Zt], ["IF x IS low THEN z IS ok"])


def test_tsukamoto_sigmoid_consequent():
    Zs = LinguisticVariable("z", (-10, 10), {"up": sigmoid(1, 0)})
    out = TsukamotoFIS([X], [Zs], ["IF x IS low THEN z IS up"]).evaluate({"x": 5})["z"]
    assert out == approx(0.0, abs=1e-12)  # a=.5 -> inverse 0


def test_tsukamoto_skips_zero_strength_and_no_fire_policy():
    Xn = LinguisticVariable("x", (0, 10), {"low": triangular(0, 0, 4), "high": triangular(6, 10, 10)})
    rules = ["IF x IS low THEN z IS small", "IF x IS high THEN z IS large"]
    # x=2: low .5, high 0 -> z_small = 60 - 40*.5 = 40 ; the zero rule is skipped
    assert TsukamotoFIS([Xn], [Z], rules).evaluate({"x": 2})["z"] == approx(40.0)
    with pytest.raises(NoRuleFiredError, match="z"):
        TsukamotoFIS([Xn], [Z], rules).evaluate({"x": 5})
    assert TsukamotoFIS([Xn], [Z], rules, default={"z": 50}).evaluate({"x": 5}) == {"z": 50.0}
    with pytest.raises(ValueError, match="outside its universe"):
        TsukamotoFIS([Xn], [Z], rules, default={"z": 500})


def test_tsukamoto_explain_properties_and_errors():
    fis = TsukamotoFIS([X], [Z], RULES, and_method="prod")
    ex = fis.explain({"x": 3})
    assert [a.strength for a in ex.activations] == approx([0.7, 0.3])
    assert fis.and_method == "prod" and fis.or_method == "max" and fis.default is None
    for attr in ("inputs", "outputs", "rules"):
        with pytest.raises(AttributeError):
            setattr(fis, attr, None)
    with pytest.raises(ValueError, match="'x'"):
        fis.evaluate({"x": -1})
    with pytest.raises(ValueError, match="valid: min, prod"):
        TsukamotoFIS([X], [Z], RULES, and_method="foo")
    with pytest.raises(TypeError, match="LinguisticVariable"):
        TsukamotoFIS([X], [object()], RULES)


def test_tsukamoto_consequent_is_clamped_into_output_universe():
    Zs = LinguisticVariable("z", (-3, 3), {"up": sigmoid(1, 0)})
    fis = TsukamotoFIS([X], [Zs], ["IF x IS low THEN z IS up"])
    # alpha = 1: the sigmoid inverse is +20.7, clamped to the upper edge
    assert fis.evaluate({"x": 0})["z"] == approx(3.0)
    # alpha = 1e-7: inverse is about -16.1, clamped to the lower edge
    assert fis.evaluate({"x": 9.999999})["z"] == approx(-3.0)  # (a*z)/a rounding
    # linear_increasing(40, 100) on (0, 50): alpha = 1 gives 100, clamped to 50
    Zl = LinguisticVariable("z", (0, 50), {"up": linear_increasing(40, 100)})
    assert TsukamotoFIS([X], [Zl], ["IF x IS low THEN z IS up"]).evaluate({"x": 0})["z"] == approx(50.0)


def test_tsukamoto_in_universe_result_is_not_clamped():
    Zs = LinguisticVariable("z", (-3, 3), {"up": sigmoid(1, 0)})
    # x=5: alpha .5 -> 0 ; x=4 : alpha .6 -> -ln(1/.6 - 1) = ln 1.5
    fis = TsukamotoFIS([X], [Zs], ["IF x IS low THEN z IS up"])
    assert fis.evaluate({"x": 4})["z"] == approx(math.log(1.5))


def test_tsukamoto_no_rule_fired_carries_partial_explanation():
    Xn = LinguisticVariable("x", (0, 10), {"low": triangular(0, 0, 4)})
    fis = TsukamotoFIS([Xn], [Z], ["IF x IS low THEN z IS small"])
    with pytest.raises(NoRuleFiredError) as info:
        fis.explain({"x": 8})
    assert info.value.output == "z" and info.value.explanation.degrees == {"x": {"low": 0.0}}


def test_tsukamoto_final_average_is_clamped_into_the_universe():
    from bulanik.sets import linear_increasing

    # both z_k clamp to the universe top 0.3, yet fsum(a_k * 0.3) / sum(a_k) rounds to 0.3 + 1 ulp
    zv = LinguisticVariable("z", (0, 0.3), {"t": linear_increasing(0, 1.2)})
    xv = LinguisticVariable("x", (0, 1), {"a": triangular(0, 0, 1), "b": triangular(0, 1, 1)})
    rules = [
        "IF x IS a THEN z IS t WITH 0.8922024329763275",
        "IF x IS b THEN z IS t WITH 0.9135410140840489",
    ]
    x = 0.5840660878383858
    fis = TsukamotoFIS([xv], [zv], rules)
    al = [a.strength for a in fis.explain({"x": x}).activations]
    assert math.fsum(a * 0.3 for a in al) / sum(al) > 0.3  # the raw mean really overshoots
    assert fis.evaluate({"x": x})["z"] == 0.3
