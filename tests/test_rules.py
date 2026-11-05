import pytest
from pytest import approx

from bulanik.inference.rules import (
    And,
    Explanation,
    NoRuleFiredError,
    Or,
    Rule,
    RuleActivation,
    Term,
    firing_strength,
    parse_rule,
    rule_terms,
)


def test_parse_precedence_not_weight():
    r = parse_rule("IF service IS poor OR food IS NOT rancid AND service IS good THEN tip IS cheap WITH 0.5")
    assert r.weight == 0.5 and r.consequents == {"tip": "cheap"}
    assert isinstance(r.antecedent, Or) and isinstance(r.antecedent.parts[1], And)
    assert r.antecedent.parts[1].parts[0] == Term("food", "rancid", negated=True)
    assert parse_rule(str(r)) == r


def test_object_form_equals_string_form():
    r = Rule(Term("service", "poor") | Term("food", "rancid"), {"tip": "cheap"})
    assert r == parse_rule("if service is poor or food is rancid then tip is cheap")


def test_parse_errors():
    with pytest.raises(ValueError, match="THEN"):
        parse_rule("IF a IS b tip IS c")
    with pytest.raises(ValueError, match="NOT"):
        parse_rule("IF a IS b THEN tip IS NOT c")
    with pytest.raises(ValueError):
        parse_rule("IF a IS b THEN t IS c WITH 1.5")


def test_parse_errors_quote_token_and_offset():
    with pytest.raises(ValueError, match=r"'tip' at offset 10"):
        parse_rule("IF a IS b tip IS c")
    with pytest.raises(ValueError, match=r"'#' at offset 5"):
        parse_rule("IF a # b THEN t IS c")
    with pytest.raises(ValueError, match="end of input"):
        parse_rule("IF a IS b THEN t IS")
    with pytest.raises(ValueError, match="IF"):
        parse_rule("a IS b THEN t IS c")
    with pytest.raises(ValueError, match=r"'\)'"):
        parse_rule("IF (a IS b THEN t IS c)")
    with pytest.raises(ValueError, match="WITH"):
        parse_rule("IF a IS b THEN t IS c WITH")
    with pytest.raises(ValueError, match="'x' at offset"):
        parse_rule("IF a IS b THEN t IS c WITH 0.5 x")
    with pytest.raises(ValueError, match="keyword"):
        parse_rule("IF a IS then THEN t IS c")
    with pytest.raises(ValueError, match="twice"):
        parse_rule("IF a IS b THEN t IS c AND t IS d")


def test_parentheses_and_flattening():
    r = parse_rule("IF (a IS x OR b IS y) AND (c IS z AND d IS w) THEN t IS c")
    assert isinstance(r.antecedent, And) and len(r.antecedent.parts) == 3
    assert isinstance(r.antecedent.parts[0], Or)
    assert "(a IS x OR b IS y) AND c IS z AND d IS w" in str(r)
    assert parse_rule(str(r)) == r
    chain = Term("a", "x") & Term("b", "y") & Term("c", "z")
    assert chain == parse_rule("IF a IS x AND b IS y AND c IS z THEN t IS c").antecedent
    either = (Term("a", "x") | Term("b", "y")) | Term("c", "z")
    assert len(either.parts) == 3


def test_multiple_consequents_roundtrip():
    r = parse_rule("IF a IS x THEN t IS c AND u IS d WITH 0.25")
    assert r.consequents == {"t": "c", "u": "d"} and r.weight == 0.25
    assert str(r) == "IF a IS x THEN t IS c AND u IS d WITH 0.25"
    assert parse_rule(str(r)) == r
    assert hash(r) == hash(parse_rule(str(r)))


def test_term_invert_and_rule_validation():
    t = Term("a", "x")
    assert (~t).negated and ~~t == t
    with pytest.raises(ValueError, match="weight"):
        Rule(t, {"o": "p"}, weight=-0.1)
    with pytest.raises(ValueError, match="weight"):
        Rule(t, {"o": "p"}, weight=float("nan"))
    with pytest.raises(ValueError, match="consequent"):
        Rule(t, {})
    with pytest.raises(ValueError, match="two"):
        And((t,))
    with pytest.raises(ValueError, match="two"):
        Or((t,))
    with pytest.raises(TypeError):
        Rule("a IS x", {"o": "p"})  # type: ignore[arg-type]


def test_firing_strength():
    deg = {"s": {"poor": 0.4, "good": 0.0}, "f": {"rancid": 0.6}}
    ant = Term("s", "poor") & ~Term("f", "rancid")  # min(.4, 1-.6)
    assert firing_strength(ant, deg, "min", "max") == approx(0.4)
    assert firing_strength(ant, deg, "prod", "max") == approx(0.16)
    assert firing_strength(Term("s", "poor") | Term("f", "rancid"), deg, "min", "probor") == approx(0.76)
    assert firing_strength(Term("s", "poor") | Term("f", "rancid"), deg, "min", "max") == approx(0.6)
    with pytest.raises(ValueError, match="valid: min, prod"):
        firing_strength(ant, deg, "foo", "max")
    with pytest.raises(ValueError, match="valid: max, probor"):
        firing_strength(Term("s", "poor") | Term("f", "rancid"), deg, "min", "foo")


def test_result_types():
    a = RuleActivation("r", 0.5)
    e = Explanation({"v": {"t": 1.0}}, (a,), {"o": 1.0})
    assert e.activations[0].strength == 0.5
    assert issubclass(NoRuleFiredError, ValueError)


def test_rule_hash_ignores_consequent_order():
    a = parse_rule("IF a IS x THEN t IS c AND u IS d")
    b = parse_rule("IF a IS x THEN u IS d AND t IS c")
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1


def test_explanation_degrees_may_hold_intervals():
    e = Explanation({"v": {"t": (0.2, 0.4)}}, (RuleActivation("r", (0.1, 0.3)),), {})
    assert e.degrees["v"]["t"] == (0.2, 0.4)


def test_rule_weight_rejects_str_and_bool():
    with pytest.raises(TypeError):
        Rule(Term("x", "low"), {"y": "m"}, weight="0.5")
    with pytest.raises(TypeError):
        Rule(Term("x", "low"), {"y": "m"}, weight=True)
    assert Rule(Term("x", "low"), {"y": "m"}, weight=1).weight == 1.0


# ------------------------------------------------------------ batch C minors


def test_dotless_i_is_not_a_keyword():
    # Turkish 'ı'.upper() is 'I': 'ıf' must stay a plain name, never the IF keyword
    assert "ıf".upper() == "IF"
    with pytest.raises(ValueError, match="expected 'IF'"):
        parse_rule("ıf a IS x THEN t IS c")
    r = parse_rule("IF ıf IS ıs THEN çıktı IS yüksek")
    assert str(r.antecedent) == "ıf IS ıs"
    assert parse_rule(str(r)) == r
    # a non-ASCII spelling of a keyword is a name, so the ASCII keyword check still fires on real ones
    with pytest.raises(ValueError, match="keyword"):
        parse_rule("IF a IS then THEN t IS c")


def test_non_ascii_names_still_work():
    r = parse_rule("IF sıcaklık IS soğuk AND nem IS yüksek THEN fan IS yavaş")
    assert [t.variable for t in rule_terms(r.antecedent)] == ["sıcaklık", "nem"]
    assert parse_rule(str(r)) == r


@pytest.mark.parametrize("bad", ["two words", "", "1abc", "a-b", "if", "AND", "Then"])
def test_object_form_names_must_round_trip(bad):
    with pytest.raises(ValueError, match="name"):
        Term(bad, "ok")
    with pytest.raises(ValueError, match="name"):
        Term("ok", bad)
    with pytest.raises(ValueError, match="name"):
        Rule(Term("a", "b"), {bad: "ok"})
    with pytest.raises(ValueError, match="name"):
        Rule(Term("a", "b"), {"ok": bad})


def test_object_form_name_types():
    with pytest.raises(TypeError, match="string"):
        Term(1, "ok")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        Term("a", "b", negated=1)  # type: ignore[arg-type]
    t = Term("sıcaklık", "soğuk")
    assert parse_rule(f"IF {t} THEN y IS z").antecedent == t


def test_rule_consequents_are_read_only():
    import copy
    import pickle

    r = parse_rule("IF a IS x THEN t IS c")
    with pytest.raises(TypeError):
        r.consequents["u"] = "d"  # type: ignore[index]
    with pytest.raises(TypeError):
        del r.consequents["t"]  # type: ignore[attr-defined]
    source = {"t": "c"}
    r2 = Rule(Term("a", "x"), source)
    source["u"] = "d"  # the rule took a copy
    assert dict(r2.consequents) == {"t": "c"} and r2 == r and hash(r2) == hash(r)
    assert repr(r) == "Rule(antecedent=Term(variable='a', term='x', negated=False), consequents={'t': 'c'}, weight=1.0)"
    for clone in (copy.copy(r), copy.deepcopy(r), pickle.loads(pickle.dumps(r))):
        assert clone == r and hash(clone) == hash(r)


def test_firing_strength_missing_degree_names_the_term():
    with pytest.raises(ValueError, match="'service IS poor'"):
        firing_strength(Term("service", "poor"), {"service": {"good": 1.0}})
    with pytest.raises(ValueError, match="'food IS rancid'"):
        firing_strength(Term("food", "rancid"), {})


def test_firing_strength_validates_methods_even_for_a_single_term():
    deg = {"s": {"poor": 0.5}}
    with pytest.raises(ValueError, match="valid: min, prod"):
        firing_strength(Term("s", "poor"), deg, "foo")
    with pytest.raises(ValueError, match="valid: max, probor"):
        firing_strength(Term("s", "poor"), deg, "min", "foo")


def test_no_rule_fired_error_defaults_to_no_context():
    e = NoRuleFiredError("boom")
    assert str(e) == "boom" and e.output is None and e.explanation is None


def test_rule_consequents_work_with_asdict_json_and_every_mutator_is_refused():
    import copy
    import dataclasses
    import json
    import pickle

    r = parse_rule("IF a IS x THEN t IS c AND u IS d")
    assert dataclasses.asdict(r)["consequents"] == {"t": "c", "u": "d"}
    assert json.dumps(r.consequents) == '{"t": "c", "u": "d"}'
    assert r.consequents == {"t": "c", "u": "d"} and {"t": "c", "u": "d"} == r.consequents
    assert isinstance(r.consequents, dict) and repr(r.consequents) == "{'t': 'c', 'u': 'd'}"
    for op in (
        lambda d: d.__setitem__("v", "e"),
        lambda d: d.__delitem__("t"),
        lambda d: d.clear(),
        lambda d: d.pop("t"),
        lambda d: d.popitem(),
        lambda d: d.setdefault("v", "e"),
        lambda d: d.update({"v": "e"}),
        lambda d: d.__ior__({"v": "e"}),
    ):
        with pytest.raises(TypeError, match="read-only"):
            op(r.consequents)
    c = r.consequents
    for clone in (copy.copy(c), copy.deepcopy(c), pickle.loads(pickle.dumps(c))):
        assert clone == c and hash(clone) == hash(c) and type(clone) is type(c)
    assert len({c, copy.copy(c)}) == 1
    assert dataclasses.replace(r, weight=0.5).consequents == c


def test_term_negated_message_shows_the_qualified_type():
    with pytest.raises(TypeError, match="got int$"):
        Term("a", "x", 1)  # type: ignore[arg-type]
    np = pytest.importorskip("numpy")
    with pytest.raises(TypeError, match=r"got numpy\.bool_?$"):
        Term("a", "x", np.True_)  # type: ignore[arg-type]
