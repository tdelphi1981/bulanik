"""Tsukamoto fuzzy inference: monotonic consequents inverted at the firing strength.

[`TsukamotoFIS`][bulanik.inference.tsukamoto.TsukamotoFIS] needs output terms
with a monotonic membership function; each fired rule reads its crisp value off
the term at the rule's firing strength, and the output is the weighted average
of those values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from bulanik._validate import choose
from bulanik.sets import LinguisticVariable

from ._common import (
    AND as _AND,
    OR as _OR,
    check_default,
    check_variables,
    fuzzify_inputs,
    no_rule_fired,
    parse_rules,
    rule_strengths,
)
from .rules import Explanation, Rule, RuleActivation

__all__ = ["TsukamotoFIS"]



class TsukamotoFIS:
    r"""Tsukamoto fuzzy inference system (Tsukamoto 1979).

    Every output term must be monotonic (`linear_increasing`,
    `linear_decreasing` or `sigmoid`). A fired rule $k$ with firing strength
    $\alpha_k$ (rule weight times the antecedent degree) gives the crisp value
    $z_k = \mu_{B_k}^{-1}(\alpha_k)$, the point where its consequent term
    reaches $\alpha_k$, and the output is

    $$y = \frac{\sum_k \alpha_k z_k}{\sum_k \alpha_k}$$

    over the rules with $\alpha_k > 0$. Each $z_k$ is clamped into the output's
    universe (the inverse of a sigmoid or of a linear ramp can fall outside
    it), and so is the final average (rounding can put it one ulp outside);
    inputs are never clamped. If no rule fires for an output the value from
    `default` is returned, or
    [`NoRuleFiredError`][bulanik.inference.rules.NoRuleFiredError] is raised.

    Args:
        inputs: Input [`LinguisticVariable`][bulanik.sets.LinguisticVariable]s.
        outputs: Output linguistic variables whose terms are all monotonic.
        rules: [`Rule`][bulanik.inference.rules.Rule] objects and/or rule text.
        and_method: `"min"` or `"prod"`.
        or_method: `"max"` or `"probor"`.
        default: Output value per output variable, used when no rule fires.

    Raises:
        TypeError: If a variable is not a `LinguisticVariable` or a rule is
            neither a `Rule` nor a string.
        ValueError: If an option name is unknown, an output term is not
            monotonic, there are no inputs, outputs or rules, a variable name
            is duplicated, a rule names an unknown variable or term, or
            `default` is not exactly one in-universe value per output.

    Attributes:
        inputs: The input variables (read-only).
        outputs: The output variables (read-only).
        rules: The parsed rules (read-only).
        and_method: The t-norm name.
        or_method: The t-conorm name.
        default: A copy of the no-fire defaults, or `None`.

    Examples:
        >>> from bulanik.sets import LinguisticVariable, linear_decreasing, linear_increasing
        >>> from bulanik.inference import TsukamotoFIS
        >>> x = LinguisticVariable("x", (0, 10), {
        ...     "lo": linear_decreasing(0, 10), "hi": linear_increasing(0, 10)})
        >>> z = LinguisticVariable("z", (0, 20), {
        ...     "low": linear_decreasing(0, 10), "high": linear_increasing(0, 20)})
        >>> fis = TsukamotoFIS([x], [z], ["IF x IS lo THEN z IS low", "IF x IS hi THEN z IS high"])
        >>> fis.evaluate({"x": 2.5})
        {'z': 3.125}

    References:
        Tsukamoto (1979): "An approach to fuzzy reasoning method", in Advances
        in Fuzzy Set Theory and Applications (North-Holland, Amsterdam).
    """

    def __init__(
        self,
        inputs: Sequence[LinguisticVariable],
        outputs: Sequence[LinguisticVariable],
        rules: Sequence[Rule | str],
        *,
        and_method: str = "min",
        or_method: str = "max",
        default: Mapping[str, float] | None = None,
    ) -> None:
        owner = "TsukamotoFIS"
        self._and_method = choose(and_method, _AND, "and_method", owner)
        self._or_method = choose(or_method, _OR, "or_method", owner)
        for v in (*inputs, *outputs):
            if not isinstance(v, LinguisticVariable):
                raise TypeError(f"TsukamotoFIS variables must be LinguisticVariable, got {type(v).__name__}")
        self._inputs, self._outputs = check_variables(inputs, outputs)
        for o in self._outputs:
            for t in o.term_names:
                if not o[t].is_monotonic:
                    raise ValueError(
                        f"output term '{t}' of '{o.name}' is not monotonic ({o[t].kind}); "
                        "Tsukamoto needs linear_increasing, linear_decreasing or sigmoid"
                    )
        self._rules = parse_rules(rules, self._inputs, self._outputs)
        self._default = check_default(self._outputs, default)
        self._rule_text = tuple(str(r) for r in self._rules)

    inputs = property(lambda self: self._inputs, doc="Input variables (read-only).")
    outputs = property(lambda self: self._outputs, doc="Output variables (read-only).")
    rules = property(lambda self: self._rules, doc="Parsed rules (read-only).")
    and_method = property(lambda self: self._and_method, doc="The AND t-norm, `min` or `prod`.")
    or_method = property(lambda self: self._or_method, doc="The OR t-conorm, `max` or `probor`.")
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
            >>> from bulanik.inference import TsukamotoFIS
            >>> x = LinguisticVariable("x", (0, 10), {"hi": linear_increasing(0, 10)})
            >>> z = LinguisticVariable("z", (0, 20), {"high": linear_increasing(0, 20)})
            >>> info = TsukamotoFIS([x], [z], ["IF x IS hi THEN z IS high"]).explain({"x": 5})
            >>> info.activations[0].strength, info.outputs
            (0.5, {'z': 10.0})
        """
        degrees = fuzzify_inputs(self._inputs, inputs)
        strengths = rule_strengths(self._rules, degrees, self._and_method, self._or_method)
        outputs: dict[str, float] = {}
        for o in self._outputs:
            total = 0.0
            parts: list[float] = []
            for rule, alpha in zip(self._rules, strengths):
                term = rule.consequents.get(o.name)
                if term is None or alpha == 0.0:
                    continue
                lo, hi = o.universe
                z = min(max(o[term].inverse(alpha), lo), hi)
                parts.append(alpha * z)
                total += alpha
            if not total > 0.0:
                if self._default is None:
                    raise no_rule_fired(o.name, degrees, self._rule_text, strengths, outputs)
                outputs[o.name] = self._default[o.name]
            else:
                lo, hi = o.universe
                # the weighted mean of values in [lo, hi] can round one ulp outside it
                outputs[o.name] = min(max(math.fsum(parts) / total, lo), hi)
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
            >>> from bulanik.sets import LinguisticVariable, linear_increasing
            >>> from bulanik.inference import TsukamotoFIS
            >>> x = LinguisticVariable("x", (0, 10), {"hi": linear_increasing(0, 10)})
            >>> z = LinguisticVariable("z", (0, 20), {"high": linear_increasing(0, 20)})
            >>> TsukamotoFIS([x], [z], ["IF x IS hi THEN z IS high"]).evaluate({"x": 10})
            {'z': 20.0}
        """
        return self.explain(inputs).outputs
