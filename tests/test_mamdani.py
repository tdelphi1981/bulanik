import pytest
from conftest import TIPPER_INPUTS, TIPPER_OUTPUTS, TIPPER_RULES
from pytest import approx

from bulanik.inference import MamdaniFIS, NoRuleFiredError, defuzzify_samples
from bulanik.sets import LinguisticVariable, triangular


def test_tipper_centroid_point1(tipper):
    # service=3: poor .4, average .6, good 0 ; food=8: rancid 0, delicious .6
    # R1=.4, R2=.6, R3=.6 ; areas 5h(2-h): 3.2, 4.2, 4.2 ; centres 5, 15, 25
    # (3.2*5 + 4.2*15 + 4.2*25)/11.6 = 184/11.6
    assert tipper().evaluate({"service": 3, "food": 8})["tip"] == approx(15.8620689655, abs=1e-6)


def test_tipper_centroid_point2(tipper):
    # service=8: average .4, good .6 ; food=2: rancid .6 -> R1=.6 R2=.4 R3=.6
    # (4.2*5 + 3.2*15 + 4.2*25)/11.6 = 174/11.6
    assert tipper().evaluate({"service": 8, "food": 2})["tip"] == approx(15.0, abs=1e-6)


def test_tipper_prod_implication(tipper):
    # scaled triangles: areas 5h = 2, 3, 3 -> (10+45+75)/8
    assert tipper(implication="prod").evaluate({"service": 3, "food": 8})["tip"] == approx(16.25, abs=1e-6)


def test_tipper_other_defuzzifiers(tipper):
    x = {"service": 3, "food": 8}
    # bisector: half area 5.8; cheap gives 3.2, average rising 10..13 gives .9, plateau .6 needs 1.7/.6 -> 13+2.8333
    assert tipper(defuzzification="bisector").evaluate(x)["tip"] == approx(15.8333, abs=0.01)
    # max .6 on plateaus [13,17] and [23,27]
    assert tipper(defuzzification="mom").evaluate(x)["tip"] == approx(20.0, abs=0.01)
    assert tipper(defuzzification="som").evaluate(x)["tip"] == approx(13.0, abs=0.01)
    assert tipper(defuzzification="lom").evaluate(x)["tip"] == approx(27.0, abs=0.01)


def test_defuzzify_samples_right_triangle():
    xs = [i / 100 for i in range(1001)]
    mus = [x / 10 for x in xs]
    assert defuzzify_samples(xs, mus, "centroid") == approx(20 / 3, abs=0.01)  # ∫x²/10 / ∫x/10
    assert defuzzify_samples(xs, mus, "bisector") == approx(50**0.5, abs=0.01)  # x²/20 = 2.5


def test_defuzzify_samples_errors():
    with pytest.raises(ValueError, match="valid: centroid, bisector, mom, som, lom"):
        defuzzify_samples([0, 1], [1, 1], "cog")
    with pytest.raises(ValueError, match="all zero"):
        defuzzify_samples([0, 1], [0, 0], "centroid")
    with pytest.raises(ValueError, match="same length"):
        defuzzify_samples([0, 1], [1], "centroid")


def test_aggregation_methods_on_overlap(overlap_fis):
    # x on [0,10]: a=tri(0,0,10), b=tri(0,10,10); y on [0,10]: m=tri(0,5,10), r=tri(5,10,10)
    # rules x IS a -> y IS m ; x IS b -> y IS r ; at x=4 strengths .6 and .4 ; resolution 1001 (step .01)
    # max:    m plateau .6 on [3,7] dominates r (<= .4)            -> mom 5.0
    # sum:    on [5,7] .6 + (x-5)/5 peaks at x=7 with 1.0           -> mom 7.0
    # probor: on [5,7] .6 + .4*(x-5)/5 peaks at x=7 with .76        -> mom 7.0
    for agg, expected in (("max", 5.0), ("sum", 7.0), ("probor", 7.0)):
        got = overlap_fis(aggregation=agg, defuzzification="mom").evaluate({"x": 4})["y"]
        assert got == approx(expected, abs=0.01)


def test_sum_aggregation_is_not_clipped(tipper):
    # Two identical rules double the aggregate (peak 1.2 > 1); the sum is not clipped
    # and a uniform scale leaves the centroid unchanged, so it equals the single-rule max result.
    rule = "IF service IS average THEN tip IS average"
    single = tipper(rules=[rule]).evaluate({"service": 3, "food": 8})["tip"]
    doubled = tipper(rules=[rule, rule], aggregation="sum").evaluate({"service": 3, "food": 8})["tip"]
    assert single == approx(15.0, abs=1e-9)  # symmetric triangle clipped at .6, centre 15
    assert doubled == approx(single, abs=1e-9)
    # mom picks the unclipped peak plateau [13, 17]: mean 15
    assert tipper(rules=[rule, rule], aggregation="sum", defuzzification="mom").evaluate(
        {"service": 3, "food": 8}
    )["tip"] == approx(15.0, abs=0.01)


def test_explain(tipper):
    ex = tipper().explain({"service": 3, "food": 8})
    assert [a.strength for a in ex.activations] == approx([0.4, 0.6, 0.6])
    assert ex.activations[0].rule == TIPPER_RULES[0]
    assert ex.degrees["service"] == approx({"poor": 0.4, "average": 0.6, "good": 0.0})
    assert ex.outputs["tip"] == approx(15.8620689655, abs=1e-6)


def test_rule_weight_scales_strength(tipper):
    fis = tipper(rules=[TIPPER_RULES[0], TIPPER_RULES[1] + " WITH 0.5", TIPPER_RULES[2]])
    assert fis.explain({"service": 3, "food": 8}).activations[1].strength == approx(0.3)


def _small():
    return MamdaniFIS(
        [LinguisticVariable("x", (0, 10), {"low": triangular(0, 0, 4)})],
        [LinguisticVariable("y", (0, 1), {"a": triangular(0, 0.5, 1)})],
        ["IF x IS low THEN y IS a"],
    )


def test_no_rule_fired():
    fis = _small()
    with pytest.raises(NoRuleFiredError, match="y"):
        fis.evaluate({"x": 8})
    assert MamdaniFIS(fis.inputs, fis.outputs, fis.rules, default={"y": 0.25}).evaluate({"x": 8}) == {"y": 0.25}


def test_input_errors_are_clear(tipper):
    with pytest.raises(ValueError, match="food"):
        tipper().evaluate({"service": 3})
    with pytest.raises(ValueError, match="price"):
        tipper().evaluate({"service": 3, "food": 8, "price": 1})
    with pytest.raises(ValueError, match=r"service.*\[0"):
        tipper().evaluate({"service": 11, "food": 8})
    with pytest.raises(ValueError, match=r"food.*\[0"):
        tipper().evaluate({"service": 3, "food": -0.5})  # never clamped
    with pytest.raises(ValueError, match="service"):
        tipper().evaluate({"service": float("nan"), "food": 8})


def test_construction_errors(tipper):
    with pytest.raises(ValueError, match="valid: poor, average, good"):
        tipper(rules=["IF service IS great THEN tip IS cheap"])
    with pytest.raises(ValueError, match="valid: centroid, bisector, mom, som, lom"):
        tipper(defuzzification="cog")
    with pytest.raises(ValueError, match="tip"):
        tipper(default={"other": 1.0})
    with pytest.raises(ValueError, match="valid: service, food"):
        tipper(rules=["IF price IS high THEN tip IS cheap"])
    with pytest.raises(ValueError, match="valid: tip"):
        tipper(rules=["IF service IS poor THEN mood IS happy"])
    with pytest.raises(ValueError, match="valid: cheap, average, generous"):
        tipper(rules=["IF service IS poor THEN tip IS lavish"])
    with pytest.raises(ValueError, match="valid: min, prod"):
        tipper(implication="max")
    with pytest.raises(ValueError, match="valid: max, sum, probor"):
        tipper(aggregation="min")
    with pytest.raises(ValueError, match="valid: min, prod"):
        tipper(and_method="max")
    with pytest.raises(ValueError, match="valid: max, probor"):
        tipper(or_method="min")
    with pytest.raises(ValueError, match="resolution"):
        tipper(resolution=1)
    with pytest.raises(ValueError, match="rule"):
        tipper(rules=[])
    with pytest.raises(ValueError, match="tip"):
        tipper(default={"tip": 99.0})  # outside the universe
    with pytest.raises(ValueError, match="duplicate"):
        MamdaniFIS(TIPPER_INPUTS + [TIPPER_INPUTS[0]], TIPPER_OUTPUTS, TIPPER_RULES)
    with pytest.raises(ValueError, match="duplicate"):
        MamdaniFIS(TIPPER_INPUTS, [TIPPER_INPUTS[0]], ["IF food IS rancid THEN service IS poor"])
    with pytest.raises(ValueError, match="at least one"):
        MamdaniFIS([], TIPPER_OUTPUTS, TIPPER_RULES)


def test_rule_objects_and_attributes(tipper):
    from bulanik.inference import Rule, Term

    fis = tipper(rules=[Rule(Term("service", "poor"), {"tip": "cheap"}), TIPPER_RULES[1]])
    assert isinstance(fis.inputs, tuple) and isinstance(fis.outputs, tuple) and isinstance(fis.rules, tuple)
    assert len(fis.rules) == 2 and fis.rules[1].consequents == {"tip": "average"}
    with pytest.raises(TypeError):
        tipper(rules=[3])


def test_multiple_outputs(tipper):
    # second output speed on [0,1] {slow: tri(0,0,1), fast: tri(0,1,1)}; R3 becomes
    # "IF service IS good OR food IS delicious THEN tip IS generous AND speed IS fast"; R1 adds "AND speed IS slow"
    # at service=8, food=8:
    #   poor 0, average .4, good .6; rancid 0, delicious .6 -> R1=0, R3=.6 -> speed = fast clipped at .6 only
    #   μ=min(x,.6): area = .18 + .6*.4 = .42 ; moment = .6³/3 + .6*(1-.36)/2 = .072+.192 = .264 -> .264/.42
    out = tipper(with_speed=True).evaluate({"service": 8, "food": 8})
    assert set(out) == {"tip", "speed"} and out["speed"] == approx(0.6285714286, abs=0.01)


def test_prod_and_probor_methods(tipper):
    # service=3, food=8 : R1 = probor(.4, 0) = .4 ; R3 = probor(0, .6) = .6 ; unchanged vs max
    ex = tipper(and_method="prod", or_method="probor").explain({"service": 3, "food": 8})
    assert [a.strength for a in ex.activations] == approx([0.4, 0.6, 0.6])


@pytest.mark.parametrize(
    "xs, mus, match",
    [
        ([0, 1], [1, float("nan")], "finite"),
        ([0, 1], [float("nan"), 1], "finite"),
        ([0, 1], [1, -1], "negative"),
        ([0, float("inf")], [1, 1], "finite"),
        ([0, 1], [1, float("inf")], "finite"),
    ],
)
def test_defuzzify_samples_rejects_invalid_numbers(xs, mus, match):
    with pytest.raises(ValueError, match=match):
        defuzzify_samples(xs, mus, "centroid")


def test_defuzzify_samples_clamps_float_noise_below_zero():
    # -1e-13 is within TOL of 0: treated as 0, so the centroid is that of the single sample at x=1
    assert defuzzify_samples([0, 1], [-1e-13, 1.0], "centroid") == 1.0


def test_fis_attributes_are_read_only(tipper):
    fis = tipper()
    for name in ("inputs", "outputs", "rules", "and_method", "or_method", "implication",
                 "aggregation", "defuzzification", "resolution", "default"):
        with pytest.raises(AttributeError):
            setattr(fis, name, getattr(fis, name))
    assert fis.implication == "min" and fis.resolution == 3001


class _Duck:
    """Minimal variable kind (as Sugeno outputs or IT2 variables will be)."""

    def __init__(self, name, universe, functions):
        self.name, self.universe, self.functions = name, universe, functions


def test_shared_helpers_accept_other_variable_kinds():
    from bulanik.inference._common import check_variables, parse_rules

    out = _Duck("y", (0, 1), {"lo": 0.0, "hi": 1.0})
    ins, outs = check_variables(TIPPER_INPUTS, [out])
    assert outs == (out,)
    rules = parse_rules(["IF service IS poor THEN y IS lo"], ins, outs)
    assert rules[0].consequents == {"y": "lo"}
    with pytest.raises(ValueError, match="valid: lo, hi"):
        parse_rules(["IF service IS poor THEN y IS mid"], ins, outs)
    # explicit lookup overrides the default
    with pytest.raises(ValueError, match="valid: only"):
        parse_rules(["IF service IS poor THEN y IS lo"], ins, outs, term_names=lambda v: ["only"])
    with pytest.raises(TypeError, match="name"):
        check_variables(TIPPER_INPUTS, [object()])


def test_consequent_term_without_positive_sample_is_rejected():
    # generous = tri(30, 35, 40) on [0, 30] is 0 at every sample (mu(30) = 0): a rule
    # firing on it used to look like "no rule fired" or return the default silently
    tip = LinguisticVariable(
        "tip", (0, 30), {"cheap": triangular(0, 5, 10), "generous": triangular(30, 35, 40)}
    )
    rules = ["IF service IS poor THEN tip IS cheap", "IF service IS good THEN tip IS generous"]
    with pytest.raises(ValueError, match=r"'tip'.*'generous'"):
        MamdaniFIS([TIPPER_INPUTS[0]], [tip], rules)
    # a narrow term between grid samples (step .03 at resolution 1001) is caught too
    narrow = LinguisticVariable(
        "tip", (0, 30), {"cheap": triangular(0, 5, 10), "spike": triangular(10.001, 10.002, 10.003)}
    )
    with pytest.raises(ValueError, match=r"'tip'.*'spike'.*resolution"):
        MamdaniFIS([TIPPER_INPUTS[0]], [narrow], ["IF service IS good THEN tip IS spike"])
    # a term no rule uses as a consequent is not checked
    fis = MamdaniFIS([TIPPER_INPUTS[0]], [tip], ["IF service IS poor THEN tip IS cheap"], resolution=3001)
    assert fis.evaluate({"service": 0})["tip"] == approx(5.0, abs=1e-6)  # symmetric tri(0,5,10), step .01


def test_mamdani_rejects_non_linguistic_variables():
    from conftest import OUT

    fis = _small()
    with pytest.raises(TypeError, match="MamdaniFIS inputs must be LinguisticVariable"):
        MamdaniFIS([OUT], fis.outputs, fis.rules)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="MamdaniFIS outputs must be LinguisticVariable"):
        MamdaniFIS(fis.inputs, [OUT], fis.rules)  # type: ignore[list-item]


def test_no_rule_fired_carries_a_partial_explanation(tipper):
    # y is a second output no rule fires for; the first output already succeeded
    from conftest import SPEED

    fis = tipper(with_speed=True)
    bad = MamdaniFIS(
        fis.inputs, [*fis.outputs[:1], SPEED], ["IF service IS good THEN tip IS generous"], resolution=301
    )
    with pytest.raises(NoRuleFiredError, match="speed") as info:
        bad.evaluate({"service": 9, "food": 8})
    e = info.value
    assert e.output == "speed" and "tip" in e.explanation.outputs and "speed" not in e.explanation.outputs
    assert e.explanation.degrees["service"]["good"] == pytest.approx(0.8)
    assert [a.rule for a in e.explanation.activations] == ["IF service IS good THEN tip IS generous"]
    assert e.explanation.activations[0].strength == pytest.approx(0.8)


def test_no_rule_fired_explanation_when_nothing_fires():
    fis = _small()
    with pytest.raises(NoRuleFiredError) as info:
        fis.explain({"x": 8})
    ex = info.value.explanation
    assert ex.outputs == {} and ex.activations[0].strength == 0.0 and ex.degrees["x"]["low"] == 0.0


def test_fuzzify_rejects_bool_and_str_inputs():
    fis = _small()
    with pytest.raises(TypeError, match="'x'"):
        fis.evaluate({"x": True})
    with pytest.raises(TypeError, match="'x'"):
        fis.evaluate({"x": "4"})  # type: ignore[dict-item]


def test_old_private_helper_names_still_import():
    from bulanik.inference import _common
    from bulanik.inference.mamdani import _AGGREGATION, _AND, _IMPLICATION, _OR, _aggregate, _imply
    from bulanik.inference.sugeno import _check_linear_inputs

    assert _AND is _common.AND and _OR is _common.OR
    assert _IMPLICATION is _common.IMPLICATION and _AGGREGATION is _common.AGGREGATION
    assert _imply is _common.imply and _aggregate is _common.aggregate
    assert _check_linear_inputs is _common.check_linear_inputs


def test_interval_default_on_a_type1_fis_is_a_clear_type_error(tipper):
    for bad in ((0.0, 1.0), [0.0, 1.0]):
        with pytest.raises(TypeError, match=r"default of 'tip'.*interval type-2"):
            tipper(default={"tip": bad})
