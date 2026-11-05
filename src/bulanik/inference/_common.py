"""Construction and input checks shared by every fuzzy inference system.

Private helpers for the Mamdani, Sugeno, Tsukamoto and interval type-2 classes:
the option tuples (`AND`, `OR`, `IMPLICATION`, `AGGREGATION`), the pairwise
`imply` / `aggregate` operators, variable and rule validation, input
fuzzification and the no-rule-fired error. Nothing here is public API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from bulanik._validate import finite

from .rules import Explanation, NoRuleFiredError, Rule, RuleActivation, firing_strength, parse_rule, rule_terms

__all__ = [
    "AGGREGATION",
    "AND",
    "IMPLICATION",
    "OR",
    "aggregate",
    "check_default",
    "check_linear_inputs",
    "check_variables",
    "default_term_names",
    "empty_term_message",
    "fuzzify_inputs",
    "imply",
    "no_rule_fired",
    "parse_rules",
    "rule_strengths",
]

AND = ("min", "prod")
OR = ("max", "probor")
IMPLICATION = ("min", "prod")
AGGREGATION = ("max", "sum", "probor")


def imply(alpha: float, mu: float, kind: str) -> float:
    """Clip (``min``) or scale (``prod``) a membership ``mu`` by firing strength ``alpha``."""
    return min(alpha, mu) if kind == "min" else alpha * mu


def aggregate(acc: float, v: float, kind: str) -> float:
    """Combine two implied memberships: ``max``, ``sum`` (unclipped, as MATLAB) or ``probor``."""
    if kind == "max":
        return max(acc, v)
    if kind == "sum":
        return acc + v
    return acc + v - acc * v


def check_linear_inputs(inputs: Sequence[Any], outputs: Sequence[Any]) -> None:
    """Every ``Linear`` term of a Sugeno output may only name input variables."""
    known = {v.name for v in inputs}
    for o in outputs:
        for term, f in o.functions.items():
            coefficients = getattr(f, "coefficients", None)  # only Linear has them
            bad = [k for k in coefficients if k not in known] if coefficients else []
            if bad:
                raise ValueError(
                    f"term '{term}' of '{o.name}' uses unknown input {', '.join(map(repr, bad))}; "
                    f"valid: {', '.join(sorted(known))}"
                )


def empty_term_message(owner: str, out: str, term: str, xs: Sequence[float], n: int) -> str:
    return (
        f"{owner}: output '{out}' term '{term}' (a rule consequent) has membership 0 at "
        f"all {n} samples of [{xs[0]:g}, {xs[-1]:g}] (resolution={n}); it lies outside the "
        "universe or between samples, so a rule firing on it would be lost. Move the term "
        "inside the universe or raise resolution"
    )


def no_rule_fired(
    output: str,
    degrees: Mapping[str, Mapping[str, Any]],
    rule_text: Sequence[str],
    strengths: Sequence[Any],
    outputs: Mapping[str, float],
) -> NoRuleFiredError:
    """The error for an output no rule fired for, carrying the partial ``Explanation``."""
    activations = tuple(RuleActivation(t, s) for t, s in zip(rule_text, strengths))
    partial = Explanation(
        {k: dict(v) for k, v in degrees.items()}, activations, dict(outputs)
    )
    return NoRuleFiredError(
        f"no rule fired for output '{output}' and no default was given; "
        "the error's .explanation holds the input degrees and rule activations",
        output=output,
        explanation=partial,
    )


def default_term_names(variable: Any) -> Sequence[str]:
    """Term names of a variable: ``term_names``, else the keys of ``functions``."""
    names = getattr(variable, "term_names", None)
    return tuple(variable.functions) if names is None else names


def check_variables(
    inputs: Sequence[Any], outputs: Sequence[Any]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Freeze the variable lists; need at least one of each, with distinct names.

    Any object with a string ``name`` qualifies (``LinguisticVariable``, Sugeno
    outputs, type-2 variables), so each FIS class checks its own kinds on top.
    """
    ins, outs = tuple(inputs), tuple(outputs)
    if not ins or not outs:
        raise ValueError("a fuzzy system needs at least one input and one output variable")
    seen: set[str] = set()
    for v in ins + outs:
        if not isinstance(getattr(v, "name", None), str):
            raise TypeError(f"variables need a string 'name' attribute, got {type(v).__name__}")
        if v.name in seen:
            raise ValueError(f"duplicate variable name '{v.name}'")
        seen.add(v.name)
    return ins, outs


def parse_rules(
    rules: Sequence[Rule | str],
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    term_names: Callable[[Any], Sequence[str]] = default_term_names,
) -> tuple[Rule, ...]:
    """Parse string rules and check every name against the variables.

    ``term_names(variable)`` lists a variable's valid terms.
    """
    if not rules:
        raise ValueError("a fuzzy system needs at least one rule")
    parsed: list[Rule] = []
    for r in rules:
        if isinstance(r, str):
            r = parse_rule(r)
        elif not isinstance(r, Rule):
            raise TypeError(f"rules must be Rule or str, got {type(r).__name__}")
        parsed.append(r)
    by_in = {v.name: v for v in inputs}
    by_out = {v.name: v for v in outputs}
    for r in parsed:
        for t in rule_terms(r.antecedent):
            _check_term(r, by_in, t.variable, t.term, "input", term_names)
        for name, term in r.consequents.items():
            _check_term(r, by_out, name, term, "output", term_names)
    return tuple(parsed)


def _check_term(
    rule: Rule,
    variables: Mapping[str, Any],
    name: str,
    term: str,
    role: str,
    term_names: Callable[[Any], Sequence[str]],
) -> None:
    other = "an output" if role == "input" else "an input"
    if name not in variables:
        raise ValueError(
            f"unknown {role} variable '{name}' in rule '{rule}'; valid: {', '.join(variables)}"
            f" (a name that is {other} cannot be used here)"
        )
    valid = term_names(variables[name])
    if term not in valid:
        raise ValueError(
            f"unknown term '{term}' of '{name}' in rule '{rule}'; valid: {', '.join(valid)}"
        )


def fuzzify_inputs(
    variables: Sequence[Any], values: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    """Membership degrees of crisp inputs.

    A bool or str value raises TypeError. The keys must be exactly the input names; a missing or extra key, or a
    value outside its universe, raises ValueError naming the variable (and its
    universe). Values are never clamped.
    """
    names = [v.name for v in variables]
    missing = [n for n in names if n not in values]
    if missing:
        spans = ", ".join(f"'{v.name}' on [{v.universe[0]:g}, {v.universe[1]:g}]" for v in variables)
        raise ValueError(f"missing input {', '.join(repr(m) for m in missing)}; inputs: {spans}")
    extra = [k for k in values if k not in names]
    if extra:
        raise ValueError(
            f"unexpected input {', '.join(repr(e) for e in extra)}; expected: {', '.join(names)}"
        )
    return {v.name: v.fuzzify(values[v.name]) for v in variables}


def rule_strengths(
    rules: Sequence[Rule],
    degrees: Mapping[str, Mapping[str, float]],
    and_method: str,
    or_method: str,
) -> list[float]:
    """Weight x antecedent degree for every rule."""
    return [r.weight * firing_strength(r.antecedent, degrees, and_method, or_method) for r in rules]


def check_default(
    outputs: Sequence[Any], default: Mapping[str, Any] | None, *, allow_interval: bool = False
) -> dict[str, Any] | None:
    """Validate a no-fire ``default``: finite, exactly one value per output.

    A variable that has a ``universe`` also gets a range check. With
    ``allow_interval`` (interval type-2 systems) a value may also be a
    ``(left, right)`` pair with ``left <= right``, the type-reduced interval to
    report; the crisp output is then its midpoint.
    """
    if default is None:
        return None
    names = [v.name for v in outputs]
    missing = [n for n in names if n not in default]
    extra = [k for k in default if k not in names]
    if missing or extra:
        raise ValueError(
            f"default must cover exactly the outputs ({', '.join(names)}); "
            f"missing: {', '.join(missing) or 'none'}; unknown: {', '.join(map(str, extra)) or 'none'}"
        )
    out: dict[str, Any] = {}
    for v in outputs:
        raw = default[v.name]
        label = f"default of '{v.name}'"
        universe = getattr(v, "universe", None)
        if allow_interval and isinstance(raw, (tuple, list)):
            if len(raw) != 2:
                raise ValueError(f"{label} must be a number or a (left, right) pair, got {raw!r}")
            vals = (finite(raw[0], f"{label} (left)"), finite(raw[1], f"{label} (right)"))
            if vals[0] > vals[1]:
                raise ValueError(f"{label} has left {vals[0]:g} above right {vals[1]:g}")
            probes: tuple[float, ...] = vals
            out[v.name] = vals
        else:
            if isinstance(raw, (tuple, list)):
                raise TypeError(
                    f"{label} is {raw!r}: interval defaults are only for interval type-2 "
                    "systems; give a single number here"
                )
            val = finite(raw, label)
            probes = (val,)
            out[v.name] = val
        if universe is not None and not all(universe[0] <= p <= universe[1] for p in probes):
            shown = ", ".join(f"{p:g}" for p in probes)
            raise ValueError(
                f"{label} is {shown}, outside its universe [{universe[0]:g}, {universe[1]:g}]"
            )
    return out
