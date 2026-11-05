r"""Takagi-Sugeno-Kang fuzzy inference: constant or linear rule consequents.

A Sugeno system has no output membership functions. Each output variable
([`SugenoOutput`][bulanik.inference.sugeno.SugenoOutput]) maps its term names to
a constant (zero-order) or a [`Linear`][bulanik.inference.sugeno.Linear] function
of the inputs (first-order). [`SugenoFIS`][bulanik.inference.sugeno.SugenoFIS]
combines the fired rules' values by a weighted average.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from bulanik._validate import choose, finite
from bulanik.sets import LinguisticVariable

from ._common import (
    AND as _AND,
    OR as _OR,
    check_default,
    check_linear_inputs as _check_linear_inputs,
    check_variables,
    fuzzify_inputs,
    no_rule_fired,
    parse_rules,
    rule_strengths,
)
from .rules import Explanation, Rule, RuleActivation

__all__ = ["Linear", "SugenoFIS", "SugenoOutput"]

_OUTPUT = ("weighted_average", "weighted_sum")


class Linear:
    r"""First-order Sugeno consequent $c_0 + \sum_i c_i x_i$ over named inputs.

    Calling the object evaluates it on a mapping of crisp inputs; extra inputs
    are ignored.

    Args:
        coefficients: Mapping from input variable name to its coefficient $c_i$.
        constant: The constant term $c_0$.

    Raises:
        TypeError: If a coefficient or `constant` is a `bool` or a string.
        ValueError: If a coefficient or `constant` is not finite.

    Attributes:
        coefficients: A copy of the input name to coefficient mapping.
        constant: The constant term.

    Examples:
        >>> from bulanik.inference import Linear
        >>> f = Linear({"x": 2.0, "y": -1.0}, 0.5)
        >>> f({"x": 3, "y": 1})
        5.5
        >>> f
        Linear({'x': 2.0, 'y': -1.0}, 0.5)
        >>> f({"x": 3})
        Traceback (most recent call last):
        ...
        ValueError: missing input 'y' for Linear
    """

    def __init__(self, coefficients: Mapping[str, float], constant: float = 0.0) -> None:
        self._coefficients = {
            str(k): finite(v, f"coefficient of '{k}'") for k, v in dict(coefficients).items()
        }
        self._constant = finite(constant, "constant")

    coefficients = property(lambda self: dict(self._coefficients), doc="Input name -> coefficient.")
    constant = property(lambda self: self._constant, doc="The constant term $c_0$.")

    def __call__(self, inputs: Mapping[str, float]) -> float:
        """Evaluate $c_0 + \\sum_i c_i x_i$ at `inputs`; same as `linear(inputs)`.

        Raises:
            ValueError: If a coefficient's input is missing from `inputs`.
        """
        missing = [k for k in self._coefficients if k not in inputs]
        if missing:
            raise ValueError(f"missing input {', '.join(repr(m) for m in missing)} for Linear")
        return self._constant + math.fsum(c * inputs[k] for k, c in self._coefficients.items())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Linear):
            return NotImplemented
        return self._coefficients == other._coefficients and self._constant == other._constant

    def __hash__(self) -> int:
        return hash((tuple(sorted(self._coefficients.items())), self._constant))

    def __repr__(self) -> str:
        return f"Linear({self._coefficients!r}, {self._constant!r})"


class SugenoOutput:
    """Output variable whose terms are constants (zero-order) or `Linear` (first-order).

    Args:
        name: The output name, used in rule consequents.
        functions: Mapping from term name to a number or a
            [`Linear`][bulanik.inference.sugeno.Linear] (at least one term).

    Raises:
        ValueError: If `name` is not a non-empty string, `functions` is empty,
            or a constant is not finite.
        TypeError: If a term name is not a string, or a constant is a `bool`
            or a string.

    Attributes:
        name: The output name.
        functions: A copy of the term to constant-or-`Linear` mapping.
        term_names: The term names, in insertion order.

    Examples:
        >>> from bulanik.inference import Linear, SugenoOutput
        >>> speed = SugenoOutput("speed", {"slow": 10, "fast": Linear({"load": -2.0}, 80)})
        >>> speed.term_names
        ('slow', 'fast')
        >>> speed
        SugenoOutput('speed', {'slow': 10.0, 'fast': Linear({'load': -2.0}, 80.0)})
    """

    def __init__(self, name: str, functions: Mapping[str, float | Linear]) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"output name must be a non-empty string, got {name!r}")
        funcs: dict[str, float | Linear] = {}
        for term, f in dict(functions).items():
            if not isinstance(term, str):
                raise TypeError(f"term names must be strings, got {type(term).__name__}")
            funcs[term] = f if isinstance(f, Linear) else finite(f, f"term '{term}' of '{name}'")
        if not funcs:
            raise ValueError(f"output '{name}' needs at least one term")
        self._name = name
        self._functions = funcs

    name = property(lambda self: self._name, doc="The output variable name.")
    functions = property(lambda self: dict(self._functions), doc="Term -> float or Linear.")

    @property
    def term_names(self) -> tuple[str, ...]:
        """The term names, in insertion order."""
        return tuple(self._functions)

    def __repr__(self) -> str:
        return f"SugenoOutput({self._name!r}, {self._functions!r})"


class SugenoFIS:
    r"""Sugeno fuzzy inference system (Takagi and Sugeno 1985).

    Each fired rule $k$ has firing strength $\alpha_k$ (rule weight times the
    antecedent degree) and a crisp value $z_k$: the consequent constant, or the
    [`Linear`][bulanik.inference.sugeno.Linear] function evaluated at the crisp
    inputs. The output is

    $$y = \frac{\sum_k \alpha_k z_k}{\sum_k \alpha_k}$$

    for `output_method="weighted_average"` and $y = \sum_k \alpha_k z_k$ for
    `"weighted_sum"`.

    When no rule fires for an output ($\sum_k \alpha_k = 0$) the value from
    `default` is returned, or
    [`NoRuleFiredError`][bulanik.inference.rules.NoRuleFiredError] is raised.

    Args:
        inputs: Input [`LinguisticVariable`][bulanik.sets.LinguisticVariable]s.
        outputs: [`SugenoOutput`][bulanik.inference.sugeno.SugenoOutput]s (at
            least one; names distinct from the inputs).
        rules: [`Rule`][bulanik.inference.rules.Rule] objects and/or rule text.
        and_method: `"min"` or `"prod"`.
        or_method: `"max"` or `"probor"`.
        output_method: `"weighted_average"` or `"weighted_sum"`.
        default: Output value per output variable, used when no rule fires.

    Raises:
        TypeError: If a variable has the wrong class or a rule is neither a
            `Rule` nor a string.
        ValueError: If an option name is unknown, there are no inputs, outputs
            or rules, a variable name is duplicated, a rule names an unknown
            variable or term, a `Linear` names an unknown input, or `default`
            is not exactly one finite value per output.

    Attributes:
        inputs: The input variables (read-only).
        outputs: The output variables (read-only).
        rules: The parsed rules (read-only).
        and_method: The t-norm name.
        or_method: The t-conorm name.
        output_method: The output combination name.
        default: A copy of the no-fire defaults, or `None`.

    Examples:
        >>> from bulanik.sets import LinguisticVariable, linear_decreasing, linear_increasing
        >>> from bulanik.inference import Linear, SugenoFIS, SugenoOutput
        >>> x = LinguisticVariable("x", (0, 10), {
        ...     "lo": linear_decreasing(0, 10), "hi": linear_increasing(0, 10)})
        >>> y = SugenoOutput("y", {"a": 1.0, "b": Linear({"x": 2.0}, 1.0)})
        >>> rules = ["IF x IS lo THEN y IS a", "IF x IS hi THEN y IS b"]
        >>> SugenoFIS([x], [y], rules).evaluate({"x": 2.5})
        {'y': 2.25}
        >>> weighted = ["IF x IS lo THEN y IS a WITH 0.5", "IF x IS hi THEN y IS b"]
        >>> SugenoFIS([x], [y], weighted).evaluate({"x": 2.5})
        {'y': 3.0}
        >>> SugenoFIS([x], [y], weighted, output_method="weighted_sum").evaluate({"x": 2.5})
        {'y': 1.875}

    References:
        Takagi and Sugeno (1985): "Fuzzy identification of systems and its
        applications to modeling and control", IEEE Transactions on Systems,
        Man, and Cybernetics.
    """

    def __init__(
        self,
        inputs: Sequence[LinguisticVariable],
        outputs: Sequence[SugenoOutput],
        rules: Sequence[Rule | str],
        *,
        and_method: str = "min",
        or_method: str = "max",
        output_method: str = "weighted_average",
        default: Mapping[str, float] | None = None,
    ) -> None:
        owner = "SugenoFIS"
        self._and_method = choose(and_method, _AND, "and_method", owner)
        self._or_method = choose(or_method, _OR, "or_method", owner)
        self._output_method = choose(output_method, _OUTPUT, "output_method", owner)
        for v in inputs:
            if not isinstance(v, LinguisticVariable):
                raise TypeError(f"SugenoFIS inputs must be LinguisticVariable, got {type(v).__name__}")
        for o in outputs:
            if not isinstance(o, SugenoOutput):
                raise TypeError(f"SugenoFIS outputs must be SugenoOutput, got {type(o).__name__}")
        self._inputs, self._outputs = check_variables(inputs, outputs)
        _check_linear_inputs(self._inputs, self._outputs)
        self._rules = parse_rules(rules, self._inputs, self._outputs)
        self._default = check_default(self._outputs, default)
        self._rule_text = tuple(str(r) for r in self._rules)

    inputs = property(lambda self: self._inputs, doc="Input variables (read-only).")
    outputs = property(lambda self: self._outputs, doc="Output variables (read-only).")
    rules = property(lambda self: self._rules, doc="Parsed rules (read-only).")
    and_method = property(lambda self: self._and_method, doc="The AND t-norm, `min` or `prod`.")
    or_method = property(lambda self: self._or_method, doc="The OR t-conorm, `max` or `probor`.")
    output_method = property(
        lambda self: self._output_method, doc="`weighted_average` or `weighted_sum`."
    )
    default = property(
        lambda self: None if self._default is None else dict(self._default),
        doc="A copy of the value returned per output when no rule fires, or `None`.",
    )

    def explain(self, inputs: Mapping[str, float]) -> Explanation:
        """Evaluate and report degrees, rule activations and crisp outputs.

        Args:
            inputs: Crisp value of every input variable, by name. The keys must
                be exactly the input names and each value must lie inside its
                variable's universe.

        Returns:
            An [`Explanation`][bulanik.inference.rules.Explanation] with the
            input membership degrees, one activation per rule and the crisp
            outputs.

        Raises:
            TypeError: If a value is a `bool` or a string.
            ValueError: If an input is missing, unexpected, not finite or
                outside its universe.
            NoRuleFiredError: If no rule fires for an output and there is no
                `default`.

        Examples:
            >>> from bulanik.sets import LinguisticVariable, linear_decreasing, linear_increasing
            >>> from bulanik.inference import SugenoFIS, SugenoOutput
            >>> x = LinguisticVariable("x", (0, 10), {
            ...     "lo": linear_decreasing(0, 10), "hi": linear_increasing(0, 10)})
            >>> y = SugenoOutput("y", {"a": 0.0, "b": 100.0})
            >>> fis = SugenoFIS([x], [y], ["IF x IS lo THEN y IS a", "IF x IS hi THEN y IS b"])
            >>> info = fis.explain({"x": 2.5})
            >>> [a.strength for a in info.activations], info.outputs
            ([0.75, 0.25], {'y': 25.0})
        """
        degrees = fuzzify_inputs(self._inputs, inputs)
        crisp = {v.name: float(inputs[v.name]) for v in self._inputs}
        strengths = rule_strengths(self._rules, degrees, self._and_method, self._or_method)
        outputs: dict[str, float] = {}
        for o in self._outputs:
            total = 0.0
            parts: list[float] = []
            for rule, alpha in zip(self._rules, strengths):
                term = rule.consequents.get(o.name)
                if term is None or alpha == 0.0:
                    continue
                f = o.functions[term]
                z = f(crisp) if isinstance(f, Linear) else f
                parts.append(alpha * z)
                total += alpha
            weighted = math.fsum(parts)
            if not total > 0.0:
                if self._default is None:
                    raise no_rule_fired(o.name, degrees, self._rule_text, strengths, outputs)
                outputs[o.name] = self._default[o.name]
            elif self._output_method == "weighted_sum":
                outputs[o.name] = weighted
            else:
                outputs[o.name] = weighted / total
        activations = tuple(RuleActivation(t, s) for t, s in zip(self._rule_text, strengths))
        return Explanation(degrees, activations, outputs)

    def evaluate(self, inputs: Mapping[str, float]) -> dict[str, float]:
        """Crisp outputs for crisp inputs.

        Args:
            inputs: Crisp value of every input variable, by name.

        Returns:
            A dict from output variable name to its crisp value.

        Raises:
            TypeError: If a value is a `bool` or a string.
            ValueError: If an input is missing, unexpected, not finite or
                outside its universe.
            NoRuleFiredError: If no rule fires for an output and there is no
                `default`.

        Examples:
            >>> from bulanik.sets import LinguisticVariable, triangular
            >>> from bulanik.inference import SugenoFIS, SugenoOutput
            >>> x = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 3)})
            >>> y = SugenoOutput("y", {"a": 5})
            >>> SugenoFIS([x], [y], ["IF x IS lo THEN y IS a"], default={"y": -1}).evaluate({"x": 8})
            {'y': -1.0}
        """
        return self.explain(inputs).outputs
