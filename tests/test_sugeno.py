import pytest
from conftest import OUT, X
from pytest import approx

from bulanik.inference import Linear, NoRuleFiredError, SugenoFIS, SugenoOutput
from bulanik.sets import LinguisticVariable, triangular

RULES = ["IF x IS low THEN y IS f_low", "IF x IS high THEN y IS f_high WITH 0.5"]


def test_sugeno_first_and_zero_order():
    fis = SugenoFIS(
        [X],
        [SugenoOutput("y", {"f_low": Linear({"x": 0.5}, 1.0), "f_high": 8.0})],
        ["IF x IS low THEN y IS f_low", "IF x IS high THEN y IS f_high"],
    )
    # x=4: low .6, high .4 ; f_low = 1 + .5*4 = 3 ; (.6*3 + .4*8)/1 = 5
    assert fis.evaluate({"x": 4})["y"] == approx(5.0)


def test_sugeno_weighted_sum_and_rule_weight():
    # strengths .6 and .4*.5 = .2 ; z = 3 and 8 ; (1.8 + 1.6)/.8 = 4.25 ; sum 3.4
    assert SugenoFIS([X], [OUT], RULES).evaluate({"x": 4})["y"] == approx(4.25)
    tot = SugenoFIS([X], [OUT], RULES, output_method="weighted_sum").evaluate({"x": 4})["y"]
    assert tot == approx(3.4)


def test_sugeno_prod_and_probor_methods():
    y = SugenoOutput("y", {"c": 10.0, "d": 0.0})
    fis = SugenoFIS(
        [X, LinguisticVariable("w", (0, 10), {"on": triangular(0, 10, 10)})],
        [y],
        ["IF x IS low AND w IS on THEN y IS c", "IF x IS high THEN y IS d"],
        and_method="prod",
    )
    # x=4,w=5: low .6 * on .5 = .3 ; high .4 -> (.3*10 + 0)/.7
    assert fis.evaluate({"x": 4, "w": 5})["y"] == approx(3 / 0.7)


def test_sugeno_no_fire_and_validation():
    Xn = LinguisticVariable("x", (0, 10), {"low": triangular(0, 0, 4)})
    out = SugenoOutput("y", {"c": 2.0})
    rules = ["IF x IS low THEN y IS c"]
    with pytest.raises(NoRuleFiredError, match="y"):
        SugenoFIS([Xn], [out], rules).evaluate({"x": 8})
    with pytest.raises(NoRuleFiredError, match="y"):
        SugenoFIS([Xn], [out], rules, output_method="weighted_sum").evaluate({"x": 8})
    assert SugenoFIS([Xn], [out], rules, default={"y": -1.0}).evaluate({"x": 8}) == {"y": -1.0}
    with pytest.raises(ValueError, match="'z'"):
        SugenoFIS([Xn], [SugenoOutput("y", {"c": Linear({"z": 1})})], rules)
    with pytest.raises(ValueError, match="valid: weighted_average, weighted_sum"):
        SugenoFIS([Xn], [out], rules, output_method="mean")
    with pytest.raises(ValueError, match="default must cover"):
        SugenoFIS([Xn], [out], rules, default={})


def test_sugeno_explain_and_input_errors():
    fis = SugenoFIS([X], [OUT], RULES)
    ex = fis.explain({"x": 4})
    assert [a.strength for a in ex.activations] == approx([0.6, 0.2])
    assert ex.degrees["x"] == approx({"low": 0.6, "high": 0.4})
    assert ex.outputs == fis.evaluate({"x": 4})
    with pytest.raises(ValueError, match="'x'.*\\[0, 10\\]"):
        fis.evaluate({"x": 11})
    with pytest.raises(ValueError, match="missing input 'x'"):
        fis.evaluate({})


def test_sugeno_properties_are_read_only():
    fis = SugenoFIS([X], [OUT], RULES)
    assert fis.output_method == "weighted_average" and fis.and_method == "min"
    assert fis.default is None
    for attr in ("inputs", "outputs", "rules", "output_method"):
        with pytest.raises(AttributeError):
            setattr(fis, attr, None)


def test_sugeno_rejects_wrong_variable_kinds():
    with pytest.raises(TypeError, match="SugenoOutput"):
        SugenoFIS([X], [X], ["IF x IS low THEN x IS low"])
    with pytest.raises(TypeError, match="LinguisticVariable"):
        SugenoFIS([OUT], [OUT], RULES)


def test_sugeno_multi_output_skips_rules_without_that_output():
    o2 = SugenoOutput("u", {"c": 7.0})
    fis = SugenoFIS(
        [X], [OUT, o2], ["IF x IS low THEN y IS f_low AND u IS c", "IF x IS high THEN y IS f_high"]
    )
    # u only has rule 1 (strength .6) -> 7 ; y = (.6*3 + .4*8)/1
    assert fis.evaluate({"x": 4}) == approx({"y": 5.0, "u": 7.0})


def test_linear_and_output_validation():
    lin = Linear({"a": 2, "b": -1}, 3)
    assert lin({"a": 1.0, "b": 4.0}) == 1.0  # 3 + 2 - 4
    assert lin.coefficients == {"a": 2.0, "b": -1.0} and lin.constant == 3.0
    assert lin == Linear({"b": -1, "a": 2}, 3) and hash(lin) == hash(Linear({"b": -1, "a": 2}, 3))
    assert lin != 3 and "Linear" in repr(lin)
    with pytest.raises(ValueError, match="missing input 'b'"):
        lin({"a": 1})
    with pytest.raises(ValueError):
        Linear({"a": float("nan")})
    with pytest.raises(ValueError):
        Linear({}, float("inf"))
    with pytest.raises(ValueError, match="non-empty"):
        SugenoOutput("", {"c": 1.0})
    with pytest.raises(ValueError, match="at least one term"):
        SugenoOutput("y", {})
    with pytest.raises(TypeError, match="strings"):
        SugenoOutput("y", {1: 1.0})
    with pytest.raises(ValueError):
        SugenoOutput("y", {"c": float("nan")})
    o = SugenoOutput("y", {"c": 1})
    assert o.term_names == ("c",) and o.functions == {"c": 1.0} and "SugenoOutput" in repr(o)


def test_sugeno_weighted_sum_default_and_zero_weight_rule():
    Xn = LinguisticVariable("x", (0, 10), {"low": triangular(0, 0, 4)})
    out = SugenoOutput("y", {"c": 2.0})
    rules = ["IF x IS low THEN y IS c"]
    fis = SugenoFIS([Xn], [out], rules, output_method="weighted_sum", default={"y": -1.0})
    assert fis.evaluate({"x": 8}) == {"y": -1.0}
    # WITH 0 makes the only strength 0 -> sum of strengths 0 -> no-fire policy
    zero = ["IF x IS low THEN y IS c WITH 0"]
    with pytest.raises(NoRuleFiredError, match="y"):
        SugenoFIS([Xn], [out], zero).evaluate({"x": 1})
    assert SugenoFIS([Xn], [out], zero, default={"y": 7}).evaluate({"x": 1}) == {"y": 7.0}


def test_sugeno_no_rule_fired_carries_partial_explanation():
    Xn = LinguisticVariable("x", (0, 10), {"low": triangular(0, 0, 4)})
    fis = SugenoFIS([Xn], [SugenoOutput("y", {"c": 2.0})], ["IF x IS low THEN y IS c"])
    with pytest.raises(NoRuleFiredError) as info:
        fis.explain({"x": 8})
    assert info.value.output == "y"
    assert info.value.explanation.degrees == {"x": {"low": 0.0}}
    assert info.value.explanation.activations[0].strength == 0.0
