r"""Mamdani fuzzy inference over sampled output universes.

[`MamdaniFIS`][bulanik.inference.mamdani.MamdaniFIS] fuzzifies crisp inputs,
fires the rules (see [`parse_rule`][bulanik.inference.rules.parse_rule] for
the rule syntax), clips or scales each consequent set by its rule's firing
strength, aggregates the clipped sets over a sampled output universe and
defuzzifies the result with
[`defuzzify_samples`][bulanik.inference.mamdani.defuzzify_samples].
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from bulanik._validate import TOL, choose
from bulanik.sets import LinguisticVariable

from ._common import (
    AGGREGATION as _AGGREGATION,
    AND as _AND,
    IMPLICATION as _IMPLICATION,
    OR as _OR,
    aggregate as _aggregate,
    check_default,
    check_variables,
    empty_term_message as _empty_term_message,
    fuzzify_inputs,
    imply as _imply,
    no_rule_fired,
    parse_rules,
    rule_strengths,
)
from .rules import Explanation, Rule, RuleActivation

__all__ = ["MamdaniFIS", "defuzzify_samples"]

_DEFUZZ = ("centroid", "bisector", "mom", "som", "lom")
_PLATEAU_TOL = 1e-12


def defuzzify_samples(xs: Sequence[float], mus: Sequence[float], method: str) -> float:
    r"""Crisp value of a sampled fuzzy set.

    With samples $(x_i, \mu_i)$ the methods are:

    - `"centroid"`: $\sum_i x_i \mu_i / \sum_i \mu_i$.
    - `"bisector"`: the first $x_i$ where the running sum of $\mu$ reaches half
      of the total.
    - `"mom"`, `"som"`, `"lom"`: the mean, smallest and largest $x_i$ among the
      samples whose $\mu_i$ is within $10^{-12}$ of the maximum.

    Memberships may exceed 1 (`"sum"` aggregation). Tiny negative values within
    the library tolerance are treated as 0.

    Args:
        xs: Sample points (finite).
        mus: Membership at each sample, same length as `xs`.
        method: `"centroid"`, `"bisector"`, `"mom"`, `"som"` or `"lom"`.

    Returns:
        The crisp value.

    Raises:
        ValueError: If `method` is unknown, the lengths differ, a value is not
            finite, a membership is negative, or every membership is zero (the
            set has no crisp value).

    Examples:
        >>> from bulanik.inference import defuzzify_samples
        >>> xs, mus = [0, 1, 2, 3], [0, 1, 1, 0]
        >>> [defuzzify_samples(xs, mus, m) for m in ("centroid", "bisector", "mom", "som", "lom")]
        [1.5, 1, 1.5, 1, 2]
        >>> defuzzify_samples(xs, [0, 0, 0, 0], "centroid")
        Traceback (most recent call last):
        ...
        ValueError: cannot defuzzify a set whose memberships are all zero
    """
    method = choose(method, _DEFUZZ, "defuzzification method", "defuzzify_samples")
    if len(xs) != len(mus):
        raise ValueError(f"xs and mus must have the same length, got {len(xs)} and {len(mus)}")
    if not all(math.isfinite(x) for x in xs):
        raise ValueError("xs must all be finite")
    if not all(math.isfinite(m) for m in mus):
        raise ValueError("memberships must all be finite")
    if min(mus, default=0.0) < -TOL:
        raise ValueError(f"memberships must not be negative, got {min(mus)}")
    mus = [max(m, 0.0) for m in mus]  # float noise within TOL of 0 is clamped
    top = max(mus, default=0.0)
    if not top > 0.0:
        raise ValueError("cannot defuzzify a set whose memberships are all zero")
    if method == "centroid":
        return math.fsum(x * m for x, m in zip(xs, mus)) / math.fsum(mus)
    if method == "bisector":
        half = math.fsum(mus) / 2.0
        running = 0.0
        for x, m in zip(xs, mus):
            running += m
            if running >= half:
                return x
        return xs[-1]  # pragma: no cover - rounding guard
    peak = [x for x, m in zip(xs, mus) if m >= top - _PLATEAU_TOL]
    if method == "mom":
        return math.fsum(peak) / len(peak)
    return min(peak) if method == "som" else max(peak)


class MamdaniFIS:
    r"""Mamdani fuzzy inference system (Mamdani and Assilian 1975).

    For crisp inputs the system computes each rule's firing strength
    $\alpha_k$ (rule weight times the antecedent degree, with the t-norm
    `and_method` and t-conorm `or_method`). Every output universe is sampled at
    `resolution` evenly spaced points $x_i$, endpoints included, and the
    aggregated set is

    $$\mu(x_i) = \operatorname{agg}_k \; \operatorname{imp}\bigl(\alpha_k,\; \mu_{B_k}(x_i)\bigr)$$

    where $B_k$ is the consequent term of rule $k$. `implication` is `"min"`
    (clip, $\min(\alpha, \mu)$) or `"prod"` (scale, $\alpha\mu$); `aggregation`
    is `"max"`, `"sum"` (not clipped to 1, as in MATLAB) or `"probor"`
    ($a + b - ab$). The crisp output is the `defuzzification` of the aggregated
    samples, see [`defuzzify_samples`][bulanik.inference.mamdani.defuzzify_samples].
    If no rule fires for an output (all aggregated samples are 0) the value from
    `default` is returned, or [`NoRuleFiredError`][bulanik.inference.rules.NoRuleFiredError]
    is raised.

    Args:
        inputs: Input [`LinguisticVariable`][bulanik.sets.LinguisticVariable]s.
        outputs: Output linguistic variables (at least one; names distinct from
            the inputs).
        rules: [`Rule`][bulanik.inference.rules.Rule] objects and/or rule text.
        and_method: `"min"` or `"prod"`.
        or_method: `"max"` or `"probor"`.
        implication: `"min"` or `"prod"`.
        aggregation: `"max"`, `"sum"` or `"probor"`.
        defuzzification: `"centroid"`, `"bisector"`, `"mom"`, `"som"` or `"lom"`.
        resolution: Samples per output universe, an integer of at least 2.
        default: Output value per output variable, used when no rule fires
            (inside the universe). `None` raises instead.

    Raises:
        TypeError: If a variable is not a `LinguisticVariable` or a rule is
            neither a `Rule` nor a string.
        ValueError: If an option name is unknown, `resolution` is not an
            integer of at least 2, there are no inputs, outputs or rules, a
            variable name is duplicated, a rule names an unknown variable or
            term, an output term used by a rule has membership 0 at every
            sample, or `default` is not exactly one in-universe value per output.

    Attributes:
        inputs: The input variables (read-only).
        outputs: The output variables (read-only).
        rules: The parsed rules (read-only).
        and_method: The t-norm name.
        or_method: The t-conorm name.
        implication: The implication name.
        aggregation: The aggregation name.
        defuzzification: The defuzzification name.
        resolution: Samples per output universe.
        default: A copy of the no-fire defaults, or `None`.

    Examples:
        >>> from bulanik.sets import LinguisticVariable, triangular
        >>> from bulanik.inference import MamdaniFIS
        >>> service = LinguisticVariable("service", (0, 10), {
        ...     "poor": triangular(0, 0, 5), "good": triangular(5, 10, 10)})
        >>> tip = LinguisticVariable("tip", (0, 20), {
        ...     "low": triangular(0, 0, 10), "high": triangular(10, 20, 20)})
        >>> fis = MamdaniFIS([service], [tip], [
        ...     "IF service IS poor THEN tip IS low",
        ...     "IF service IS good THEN tip IS high"])
        >>> fis.evaluate({"service": 7.5})
        {'tip': 16.116...}
        >>> fis.evaluate({"service": 2})
        {'tip': 3.708...}

    References:
        Mamdani and Assilian (1975): "An experiment in linguistic synthesis with a fuzzy
        logic controller", International Journal of Man-Machine Studies.
    """

    def __init__(
        self,
        inputs: Sequence[LinguisticVariable],
        outputs: Sequence[LinguisticVariable],
        rules: Sequence[Rule | str],
        *,
        and_method: str = "min",
        or_method: str = "max",
        implication: str = "min",
        aggregation: str = "max",
        defuzzification: str = "centroid",
        resolution: int = 1001,
        default: Mapping[str, float] | None = None,
    ) -> None:
        owner = "MamdaniFIS"
        self._and_method = choose(and_method, _AND, "and_method", owner)
        self._or_method = choose(or_method, _OR, "or_method", owner)
        self._implication = choose(implication, _IMPLICATION, "implication", owner)
        self._aggregation = choose(aggregation, _AGGREGATION, "aggregation", owner)
        self._defuzzification = choose(defuzzification, _DEFUZZ, "defuzzification method", owner)
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2:
            raise ValueError(f"resolution must be an integer >= 2, got {resolution!r}")
        self._resolution = resolution
        for role, vs in (("inputs", inputs), ("outputs", outputs)):
            for v in vs:
                if not isinstance(v, LinguisticVariable):
                    raise TypeError(
                        f"MamdaniFIS {role} must be LinguisticVariable, got {type(v).__name__}"
                    )
        self._inputs, self._outputs = check_variables(inputs, outputs)
        self._rules = parse_rules(rules, self._inputs, self._outputs)
        self._default = check_default(self._outputs, default)
        self._rule_text = tuple(str(r) for r in self._rules)
        n = resolution
        self._grid: dict[str, list[float]] = {}
        self._terms: dict[tuple[str, str], list[float]] = {}
        for v in self.outputs:
            lo, hi = v.universe
            step = (hi - lo) / (n - 1)
            xs = [lo + i * step for i in range(n - 1)] + [hi]
            self._grid[v.name] = xs
            for t in v.term_names:
                mf = v[t]
                self._terms[(v.name, t)] = [mf(x) for x in xs]
        for rule in self._rules:
            for out, term in rule.consequents.items():
                if not any(mu > 0.0 for mu in self._terms[(out, term)]):
                    raise ValueError(_empty_term_message(owner, out, term, self._grid[out], n))

    inputs = property(lambda self: self._inputs, doc="Input variables (read-only).")
    outputs = property(lambda self: self._outputs, doc="Output variables (read-only).")
    rules = property(lambda self: self._rules, doc="Parsed rules (read-only).")
    and_method = property(lambda self: self._and_method, doc="The AND t-norm, `min` or `prod`.")
    or_method = property(lambda self: self._or_method, doc="The OR t-conorm, `max` or `probor`.")
    implication = property(lambda self: self._implication, doc="The implication, `min` or `prod`.")
    aggregation = property(
        lambda self: self._aggregation, doc="The aggregation, `max`, `sum` or `probor`."
    )
    defuzzification = property(
        lambda self: self._defuzzification, doc="The defuzzification method name."
    )
    resolution = property(
        lambda self: self._resolution, doc="Samples per output universe (endpoints included)."
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
                variable's universe (values are never clamped).

        Returns:
            An [`Explanation`][bulanik.inference.rules.Explanation] with the
            input membership degrees, one activation per rule and the crisp
            outputs.

        Raises:
            TypeError: If a value is a `bool` or a string.
            ValueError: If an input is missing, unexpected, not finite or
                outside its universe.
            NoRuleFiredError: If no rule fires for an output and there is no
                `default`; its `explanation` holds the partial result.

        Examples:
            >>> from bulanik.sets import LinguisticVariable, triangular
            >>> from bulanik.inference import MamdaniFIS
            >>> x = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 10), "hi": triangular(0, 10, 10)})
            >>> y = LinguisticVariable("y", (0, 10), {"lo": triangular(0, 0, 10), "hi": triangular(0, 10, 10)})
            >>> fis = MamdaniFIS([x], [y], ["IF x IS lo THEN y IS lo", "IF x IS hi THEN y IS hi"])
            >>> info = fis.explain({"x": 2.5})
            >>> info.degrees
            {'x': {'lo': 0.75, 'hi': 0.25}}
            >>> info.activations[0]
            RuleActivation(rule='IF x IS lo THEN y IS lo', strength=0.75)
            >>> fis.explain({"x": 11})
            Traceback (most recent call last):
            ...
            ValueError: 'x' input 11 is outside its universe [0, 10]
        """
        degrees = fuzzify_inputs(self._inputs, inputs)
        strengths = rule_strengths(self._rules, degrees, self._and_method, self._or_method)
        outputs: dict[str, float] = {}
        for v in self._outputs:
            agg = [0.0] * self._resolution
            for rule, alpha in zip(self._rules, strengths):
                term = rule.consequents.get(v.name)
                if term is None or alpha == 0.0:  # an alpha of 0 implies mu 0 everywhere
                    continue
                mus = self._terms[(v.name, term)]
                for i, mu in enumerate(mus):
                    agg[i] = _aggregate(agg[i], _imply(alpha, mu, self._implication), self._aggregation)
            if not any(m > 0.0 for m in agg):
                if self._default is None:
                    raise no_rule_fired(v.name, degrees, self._rule_text, strengths, outputs)
                outputs[v.name] = self._default[v.name]
            else:
                outputs[v.name] = defuzzify_samples(self._grid[v.name], agg, self._defuzzification)
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
            >>> from bulanik.inference import MamdaniFIS
            >>> x = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 3), "hi": triangular(7, 10, 10)})
            >>> y = LinguisticVariable("y", (0, 20), {"low": triangular(0, 0, 10), "high": triangular(10, 20, 20)})
            >>> rules = ["IF x IS lo THEN y IS low", "IF x IS hi THEN y IS high"]
            >>> MamdaniFIS([x], [y], rules, default={"y": 10}).evaluate({"x": 5})
            {'y': 10.0}
            >>> MamdaniFIS([x], [y], rules).evaluate({"x": 5})
            Traceback (most recent call last):
            ...
            bulanik.inference.rules.NoRuleFiredError: no rule fired for output 'y' and no default was given; the error's .explanation holds the input degrees and rule activations
        """
        return self.explain(inputs).outputs
