r"""Fuzzy rules: antecedent objects, the rule grammar and firing strength.

Everything here is independent of the inference style, so Mamdani, Sugeno,
Tsukamoto and interval type-2 systems share it. A rule is written as text and
parsed by [`parse_rule`][bulanik.inference.rules.parse_rule] (the inference
systems do this for you when you pass strings), or built from the objects
[`Term`][bulanik.inference.rules.Term], [`And`][bulanik.inference.rules.And],
[`Or`][bulanik.inference.rules.Or] and [`Rule`][bulanik.inference.rules.Rule].

## Rule syntax

A rule has an antecedent (the condition), one or more consequents and an
optional weight:

```text
IF <antecedent> THEN <var> IS <term> [AND <var> IS <term> ...] [WITH <weight>]
```

- **Keywords** are `IF`, `THEN`, `IS`, `NOT`, `AND`, `OR` and `WITH`. They are
  case-insensitive (`if`, `If` and `IF` are the same) and ASCII only. A keyword
  cannot be used as a variable or term name.
- **Names** (variables and terms) start with a letter or an underscore and
  continue with letters, digits or underscores; there are no spaces, so
  `very_high` and `x2` are names but `very high` is two. Letters outside ASCII
  are allowed.
- **Atoms** are `variable IS term` and `variable IS NOT term`. `NOT` negates
  the membership degree, $1 - \mu$, and is only allowed directly after `IS`
  in an antecedent; it cannot appear in a consequent.
- **Operators**: `AND` binds tighter than `OR`, and parentheses group, so
  `a IS x OR b IS y AND c IS z` means `a IS x OR (b IS y AND c IS z)`. Chains
  such as `p AND q AND r` are n-ary and are combined from left to right.
- **Consequents** follow `THEN` and are joined with `AND`, one per output
  variable (naming an output twice in one rule is an error). A rule may leave
  an output variable out; it then does not contribute to that output.
- **Weight**: `WITH w` multiplies the rule's firing strength by a number
  $w \in [0, 1]$ and defaults to 1. A weight outside $[0, 1]$ raises
  `ValueError`.

The firing strength of an antecedent is computed from the membership degrees
of the crisp inputs. For atoms $\mu$ is the degree (or $1 - \mu$ if negated);
`AND` is a t-norm ($\min$ or the product, `and_method` of the system) and `OR`
a t-conorm ($\max$ or the probabilistic sum $a + b - ab$, `or_method`); see
[`t_norm`][bulanik.sets.t_norm] and [`t_conorm`][bulanik.sets.t_conorm]. The
activation of a rule is its weight times that strength.

Parsing and printing round-trip: `str(rule)` is canonical upper-case rule text
that `parse_rule` reads back to an equal rule.

Examples:
    >>> from bulanik.inference import parse_rule
    >>> rule = parse_rule("if food is not rancid and (service is good or service is ok) then tip is high with 0.8")
    >>> print(rule)
    IF food IS NOT rancid AND (service IS good OR service IS ok) THEN tip IS high WITH 0.8
    >>> rule.weight, dict(rule.consequents)
    (0.8, {'tip': 'high'})
    >>> parse_rule(str(rule)) == rule
    True
    >>> parse_rule("IF a IS b THEN c IS NOT d")
    Traceback (most recent call last):
    ...
    ValueError: NOT is not allowed in a consequent (offset 20) in rule 'IF a IS b THEN c IS NOT d'
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NoReturn, Union

from bulanik._validate import choose, finite
from bulanik.sets import t_conorm, t_norm

__all__ = [
    "And",
    "Antecedent",
    "Explanation",
    "NoRuleFiredError",
    "Or",
    "Rule",
    "RuleActivation",
    "Term",
    "firing_strength",
    "parse_rule",
    "rule_terms",
]


def _is_keyword(word: str) -> bool:
    """Rule keyword, ASCII only: Turkish dotless 'ı' makes 'ıf'.upper() == 'IF'."""
    return word.isascii() and word.upper() in _KEYWORDS


def _check_name(name: object, what: str) -> None:
    """A name that ``str(rule)`` can round-trip through the parser."""
    if not isinstance(name, str):
        raise TypeError(f"{what} must be a string, got {type(name).__name__}")
    if _IDENT.fullmatch(name) is None:
        raise ValueError(
            f"{what} {name!r} is not usable in a rule: it must be a letter or underscore "
            "followed by letters, digits or underscores (no spaces)"
        )
    if _is_keyword(name):
        raise ValueError(f"{what} {name!r} is a rule keyword ({', '.join(sorted(_KEYWORDS))})")


class NoRuleFiredError(ValueError):
    """No rule fired for an output and no default value was given.

    A `ValueError` raised by `evaluate` / `explain` of the inference systems.
    Give the system a `default` to return a fixed value instead.

    Args:
        message: The error message.
        output: Name of the output variable no rule fired for.
        explanation: The partial [`Explanation`][bulanik.inference.rules.Explanation]
            of the failed evaluation.

    Attributes:
        output: The output variable name, or `None` when the error is raised
            without that context.
        explanation: The input `degrees`, every rule's `activations` and the
            `outputs` that did succeed before this one; `None` without context.

    Examples:
        >>> from bulanik.sets import LinguisticVariable, triangular
        >>> from bulanik.inference import MamdaniFIS, NoRuleFiredError
        >>> x = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 3)})
        >>> y = LinguisticVariable("y", (0, 10), {"low": triangular(0, 0, 10)})
        >>> fis = MamdaniFIS([x], [y], ["IF x IS lo THEN y IS low"])
        >>> try:
        ...     fis.evaluate({"x": 8})
        ... except NoRuleFiredError as err:
        ...     print(err.output, err.explanation.activations[0].strength)
        y 0.0
    """

    def __init__(
        self, message: str, *, output: str | None = None, explanation: Explanation | None = None
    ) -> None:
        super().__init__(message)
        self.output = output
        self.explanation = explanation


@dataclass(frozen=True)
class Term:
    r"""The atom `variable IS term`, or `variable IS NOT term` when negated.

    The operators `&`, `|` and `~` build antecedents: `a & b` is
    [`And`][bulanik.inference.rules.And], `a | b` is
    [`Or`][bulanik.inference.rules.Or] and `~a` is the negated term.

    Args:
        variable: Name of an input variable.
        term: Name of one of its terms.
        negated: Whether the membership degree is negated, $1 - \mu$.

    Raises:
        TypeError: If a name is not a string, or `negated` is not a `bool`.
        ValueError: If a name is not a valid rule name or is a rule keyword
            (see the module description for the syntax).

    Attributes:
        variable: The variable name.
        term: The term name.
        negated: Whether the term is negated.

    Examples:
        >>> from bulanik.inference import Term
        >>> food, service = Term("food", "good"), Term("service", "poor")
        >>> print(food & ~service)
        food IS good AND service IS NOT poor
        >>> print((food | service) & Term("price", "low"))
        (food IS good OR service IS poor) AND price IS low
    """

    variable: str
    term: str
    negated: bool = False

    def __post_init__(self) -> None:
        _check_name(self.variable, "variable name")
        _check_name(self.term, "term name")
        if not isinstance(self.negated, bool):
            t = type(self.negated)
            kind = t.__qualname__ if t.__module__ == "builtins" else f"{t.__module__}.{t.__qualname__}"
            raise TypeError(f"negated must be a bool, got {kind}")

    def __and__(self, other: Antecedent) -> And:
        return And((self, other))

    def __or__(self, other: Antecedent) -> Or:
        return Or((self, other))

    def __invert__(self) -> Term:
        return Term(self.variable, self.term, not self.negated)

    def __str__(self) -> str:
        return f"{self.variable} IS {'NOT ' if self.negated else ''}{self.term}"


@dataclass(frozen=True)
class And:
    """Conjunction of antecedents (`AND`); nested conjunctions are flattened.

    Evaluated with the system's t-norm, folded from left to right.

    Args:
        parts: Two or more antecedents.

    Raises:
        ValueError: If fewer than two parts remain after flattening.

    Attributes:
        parts: The flattened parts.

    Examples:
        >>> from bulanik.inference import And, Term
        >>> a, b, c = Term("x", "a"), Term("x", "b"), Term("y", "c")
        >>> And((And((a, b)), c)).parts == (a, b, c)
        True
        >>> print(a & b & c)
        x IS a AND x IS b AND y IS c
    """

    parts: tuple[Antecedent, ...]

    def __post_init__(self) -> None:
        parts: list[Antecedent] = []
        for p in self.parts:
            parts.extend(p.parts if isinstance(p, And) else (p,))
        if len(parts) < 2:
            raise ValueError("And needs at least two parts")
        object.__setattr__(self, "parts", tuple(parts))

    def __and__(self, other: Antecedent) -> And:
        return And((self, other))

    def __or__(self, other: Antecedent) -> Or:
        return Or((self, other))

    def __str__(self) -> str:
        return " AND ".join(f"({p})" if isinstance(p, Or) else str(p) for p in self.parts)


@dataclass(frozen=True)
class Or:
    """Disjunction of antecedents (`OR`); nested disjunctions are flattened.

    Evaluated with the system's t-conorm, folded from left to right.

    Args:
        parts: Two or more antecedents.

    Raises:
        ValueError: If fewer than two parts remain after flattening.

    Attributes:
        parts: The flattened parts.

    Examples:
        >>> from bulanik.inference import Or, Term
        >>> a, b = Term("x", "a"), Term("y", "b")
        >>> print(Or((a, b)))
        x IS a OR y IS b
        >>> Or((a,))
        Traceback (most recent call last):
        ...
        ValueError: Or needs at least two parts
    """

    parts: tuple[Antecedent, ...]

    def __post_init__(self) -> None:
        parts: list[Antecedent] = []
        for p in self.parts:
            parts.extend(p.parts if isinstance(p, Or) else (p,))
        if len(parts) < 2:
            raise ValueError("Or needs at least two parts")
        object.__setattr__(self, "parts", tuple(parts))

    def __and__(self, other: Antecedent) -> And:
        return And((self, other))

    def __or__(self, other: Antecedent) -> Or:
        return Or((self, other))

    def __str__(self) -> str:
        return " OR ".join(str(p) for p in self.parts)


Antecedent = Union[Term, And, Or]
"""An antecedent expression: a [`Term`][bulanik.inference.Term], or an [`And`][bulanik.inference.And] / [`Or`][bulanik.inference.Or] of antecedents."""


class _FrozenDict(dict):  # type: ignore[type-arg]
    """A dict that refuses mutation, but still json-dumps, copies, pickles and compares like one."""

    __slots__ = ()

    def _refuse(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("rule consequents are read-only")

    __setitem__ = __delitem__ = __ior__ = _refuse  # type: ignore[assignment]
    clear = pop = popitem = setdefault = update = _refuse  # type: ignore[assignment]

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[object, ...]:
        # the default dict-subclass protocol refills through __setitem__
        return (_FrozenDict, (dict(self),))

    def __repr__(self) -> str:
        return dict.__repr__(self)


@dataclass(frozen=True)
class Rule:
    """`IF antecedent THEN consequents [WITH weight]`.

    A rule is immutable and hashable; `str(rule)` is canonical rule text that
    [`parse_rule`][bulanik.inference.rules.parse_rule] reads back. Rules given
    to an inference system are checked against its variables and terms.

    Args:
        antecedent: A [`Term`][bulanik.inference.rules.Term],
            [`And`][bulanik.inference.rules.And] or
            [`Or`][bulanik.inference.rules.Or].
        consequents: Mapping from output variable name to term name (at least one).
        weight: Rule weight in $[0, 1]$; it multiplies the firing strength.

    Raises:
        TypeError: If `antecedent` has the wrong type, or a name is not a string.
        ValueError: If there are no consequents, a name is invalid or a rule
            keyword, or `weight` is not finite or outside $[0, 1]$.

    Attributes:
        antecedent: The condition.
        consequents: A read-only mapping of output variable to term; mutating
            it raises `TypeError`.
        weight: The rule weight as a float.

    Examples:
        >>> from bulanik.inference import Rule, Term
        >>> rule = Rule(Term("service", "good") | Term("food", "good"), {"tip": "high"}, 0.5)
        >>> print(rule)
        IF service IS good OR food IS good THEN tip IS high WITH 0.5
        >>> Rule(Term("a", "b"), {"c": "d"}, 1.5)
        Traceback (most recent call last):
        ...
        ValueError: rule weight 1.5 must be in [0, 1]
    """

    antecedent: Antecedent
    consequents: Mapping[str, str]  # stored as a read-only mapping
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.antecedent, (Term, And, Or)):
            raise TypeError(
                f"rule antecedent must be a Term, And or Or, got {type(self.antecedent).__name__}"
            )
        if not self.consequents:
            raise ValueError("a rule needs at least one consequent")
        weight = finite(self.weight, "rule weight")
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"rule weight {self.weight} must be in [0, 1]")
        object.__setattr__(self, "weight", weight)
        consequents = dict(self.consequents)
        for var, term in consequents.items():
            _check_name(var, "consequent variable name")
            _check_name(term, "consequent term name")
        # read-only dict (still a dict for json / asdict): a rule is hashable and FIS objects cache what they derive from it
        object.__setattr__(self, "consequents", _FrozenDict(consequents))

    def __hash__(self) -> int:
        return hash((self.antecedent, hash(self.consequents), self.weight))

    def __repr__(self) -> str:
        return (
            f"Rule(antecedent={self.antecedent!r}, consequents={dict(self.consequents)!r}, "
            f"weight={self.weight!r})"
        )

    def __reduce__(self) -> tuple[object, ...]:
        return (Rule, (self.antecedent, dict(self.consequents), self.weight))

    def __str__(self) -> str:
        then = " AND ".join(f"{v} IS {t}" for v, t in self.consequents.items())
        text = f"IF {self.antecedent} THEN {then}"
        return text if self.weight == 1.0 else f"{text} WITH {self.weight!r}"


@dataclass(frozen=True)
class RuleActivation:
    """A rule's firing level: weight times antecedent degree.

    Interval type-2 systems store a `(lower, upper)` pair instead of a float.

    Args:
        rule: The rule text, as `str(rule)`.
        strength: The activation, a float in $[0, 1]$ (or a pair for interval
            type-2 systems).

    Attributes:
        rule: The rule text.
        strength: The activation.

    Examples:
        >>> from bulanik.inference import RuleActivation
        >>> RuleActivation("IF x IS a THEN y IS b", 0.25).strength
        0.25
    """

    rule: str
    strength: float | tuple[float, float]


@dataclass(frozen=True)
class Explanation:
    """Everything behind one evaluation: degrees, rule activations and outputs.

    Returned by the `explain` method of every inference system; its `outputs`
    are what `evaluate` returns.

    Args:
        degrees: Membership degree of each input in each of its terms, as
            `degrees[variable][term]` (a `(lower, upper)` pair for interval
            type-2 systems).
        activations: One [`RuleActivation`][bulanik.inference.rules.RuleActivation]
            per rule, in rule order.
        outputs: The crisp value of each output variable.

    Attributes:
        degrees: The input membership degrees.
        activations: The rule activations.
        outputs: The crisp outputs.

    Examples:
        >>> from bulanik.sets import LinguisticVariable, triangular
        >>> from bulanik.inference import MamdaniFIS
        >>> x = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 10), "hi": triangular(0, 10, 10)})
        >>> y = LinguisticVariable("y", (0, 10), {"lo": triangular(0, 0, 10), "hi": triangular(0, 10, 10)})
        >>> fis = MamdaniFIS([x], [y], ["IF x IS lo THEN y IS lo", "IF x IS hi THEN y IS hi WITH 0.5"])
        >>> info = fis.explain({"x": 2.5})
        >>> info.degrees
        {'x': {'lo': 0.75, 'hi': 0.25}}
        >>> [a.strength for a in info.activations]
        [0.75, 0.125]
        >>> sorted(info.outputs)
        ['y']
    """

    degrees: dict[str, dict[str, float | tuple[float, float]]]
    activations: tuple[RuleActivation, ...]
    outputs: dict[str, float] = field(default_factory=dict)


def rule_terms(antecedent: Antecedent) -> list[Term]:
    """All [`Term`][bulanik.inference.rules.Term] leaves of an antecedent, left to right.

    Args:
        antecedent: A `Term`, `And` or `Or`.

    Returns:
        The terms in the order they appear in the rule text.

    Examples:
        >>> from bulanik.inference import Term
        >>> from bulanik.inference.rules import rule_terms
        >>> [str(t) for t in rule_terms((Term("a", "x") | Term("b", "y")) & ~Term("c", "z"))]
        ['a IS x', 'b IS y', 'c IS NOT z']
    """
    if isinstance(antecedent, Term):
        return [antecedent]
    return [t for p in antecedent.parts for t in rule_terms(p)]


_AND_METHODS = ("min", "prod")
_OR_METHODS = ("max", "probor")


def firing_strength(
    antecedent: Antecedent,
    degrees: Mapping[str, Mapping[str, float]],
    and_method: str = "min",
    or_method: str = "max",
) -> float:
    r"""Degree to which an antecedent holds.

    A term has its membership degree $\mu$, or $1 - \mu$ when negated.
    `And` folds the parts with the t-norm `and_method` and `Or` with the
    t-conorm `or_method`, from left to right (see
    [`t_norm`][bulanik.sets.t_norm] and [`t_conorm`][bulanik.sets.t_conorm]).
    The rule weight is not applied here.

    Args:
        antecedent: A `Term`, `And` or `Or`.
        degrees: Membership degrees as `degrees[variable][term]`.
        and_method: `"min"` or `"prod"`.
        or_method: `"max"` or `"probor"`.

    Returns:
        The firing strength in $[0, 1]$.

    Raises:
        ValueError: If a method name is unknown, or a term of the antecedent
            is missing from `degrees`.

    Examples:
        >>> from bulanik.inference import firing_strength, parse_rule
        >>> rule = parse_rule("IF x IS a AND y IS b THEN z IS c")
        >>> degrees = {"x": {"a": 0.6}, "y": {"b": 0.5}}
        >>> firing_strength(rule.antecedent, degrees)
        0.5
        >>> firing_strength(rule.antecedent, degrees, and_method="prod")
        0.3
    """
    and_method = choose(and_method, _AND_METHODS, "and_method", "firing_strength")
    or_method = choose(or_method, _OR_METHODS, "or_method", "firing_strength")
    return _strength(antecedent, degrees, and_method, or_method)


def _strength(
    antecedent: Antecedent,
    degrees: Mapping[str, Mapping[str, float]],
    and_method: str,
    or_method: str,
) -> float:
    if isinstance(antecedent, Term):
        try:
            mu = degrees[antecedent.variable][antecedent.term]
        except KeyError:
            raise ValueError(
                f"no membership degree for '{antecedent.variable} IS {antecedent.term}' in degrees"
            ) from None
        return 1.0 - mu if antecedent.negated else mu
    parts = [_strength(p, degrees, and_method, or_method) for p in antecedent.parts]
    combine = t_norm if isinstance(antecedent, And) else t_conorm
    kind = and_method if isinstance(antecedent, And) else or_method
    acc = parts[0]
    for p in parts[1:]:
        acc = combine(acc, p, kind)
    return acc


# --------------------------------------------------------------------------- parser

_TOKEN = re.compile(
    r"\s*(?:(?P<num>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<ident>[^\W\d]\w*)|(?P<lp>\()|(?P<rp>\)))"
)
_IDENT = re.compile(r"[^\W\d]\w*")
_KEYWORDS = frozenset({"IF", "THEN", "IS", "NOT", "AND", "OR", "WITH"})


def _tokenize(text: str) -> list[tuple[str, str, int]]:
    tokens: list[tuple[str, str, int]] = []
    pos = 0
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        m = _TOKEN.match(text, pos)
        if m is None or m.lastgroup is None:
            raise ValueError(f"unexpected character {text[pos]!r} at offset {pos} in rule {text!r}")
        kind = m.lastgroup
        start = m.start(kind)
        tokens.append((kind, m.group(kind), start))
        pos = m.end()
    tokens.append(("end", "", len(text)))
    return tokens


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.i = 0

    # -- helpers
    def _peek(self) -> tuple[str, str, int]:
        return self.tokens[self.i]

    def _is_kw(self, word: str) -> bool:
        kind, text, _ = self._peek()
        return kind == "ident" and text.isascii() and text.upper() == word

    def _fail(self, expected: str) -> ValueError:
        kind, text, pos = self._peek()
        got = "end of input" if kind == "end" else repr(text)
        return ValueError(f"expected {expected}, got {got} at offset {pos} in rule {self.text!r}")

    def _expect_kw(self, word: str) -> None:
        if not self._is_kw(word):
            raise self._fail(repr(word))
        self.i += 1

    def _name(self) -> str:
        kind, text, pos = self._peek()
        if kind != "ident":
            raise self._fail("a name")
        if _is_keyword(text):
            raise ValueError(
                f"keyword {text!r} used as a name at offset {pos} in rule {self.text!r}"
            )
        self.i += 1
        return text

    # -- grammar
    def rule(self) -> Rule:
        self._expect_kw("IF")
        antecedent = self.or_expr()
        self._expect_kw("THEN")
        consequents: dict[str, str] = {}
        while True:
            var = self._name()
            self._expect_kw("IS")
            if self._is_kw("NOT"):
                raise ValueError(
                    f"NOT is not allowed in a consequent (offset {self._peek()[2]}) in rule {self.text!r}"
                )
            term = self._name()
            if var in consequents:
                raise ValueError(f"consequent for {var!r} given twice in rule {self.text!r}")
            consequents[var] = term
            if self._is_kw("AND"):
                self.i += 1
                continue
            break
        weight = 1.0
        if self._is_kw("WITH"):
            self.i += 1
            kind, text, _ = self._peek()
            if kind != "num":
                raise self._fail("a number after WITH")
            self.i += 1
            weight = float(text)
        if self._peek()[0] != "end":
            raise self._fail("end of rule")
        return Rule(antecedent, consequents, weight)

    def or_expr(self) -> Antecedent:
        parts = [self.and_expr()]
        while self._is_kw("OR"):
            self.i += 1
            parts.append(self.and_expr())
        return parts[0] if len(parts) == 1 else Or(tuple(parts))

    def and_expr(self) -> Antecedent:
        parts = [self.atom()]
        while self._is_kw("AND"):
            self.i += 1
            parts.append(self.atom())
        return parts[0] if len(parts) == 1 else And(tuple(parts))

    def atom(self) -> Antecedent:
        if self._peek()[0] == "lp":
            self.i += 1
            inner = self.or_expr()
            if self._peek()[0] != "rp":
                raise self._fail("')'")
            self.i += 1
            return inner
        var = self._name()
        self._expect_kw("IS")
        negated = False
        if self._is_kw("NOT"):
            self.i += 1
            negated = True
        return Term(var, self._name(), negated)


def parse_rule(text: str) -> Rule:
    """Parse rule text `IF ... THEN ... [WITH w]` into a [`Rule`][bulanik.inference.rules.Rule].

    The grammar is described in the module documentation: keywords are
    case-insensitive, `AND` binds tighter than `OR`, parentheses group, and
    `NOT` only follows `IS`.

    Args:
        text: The rule text.

    Returns:
        The parsed rule.

    Raises:
        TypeError: If `text` is not a string.
        ValueError: If the text is not a valid rule (the message gives the
            offset), a keyword is used as a name, a consequent is negated or
            repeated, or the weight is outside $[0, 1]$.

    Examples:
        >>> from bulanik.inference import parse_rule
        >>> rule = parse_rule("IF temp IS hot OR NOT humidity IS low THEN fan IS fast")
        Traceback (most recent call last):
        ...
        ValueError: keyword 'NOT' used as a name at offset 18 in rule 'IF temp IS hot OR NOT humidity IS low THEN fan IS fast'
        >>> rule = parse_rule("IF temp IS hot OR humidity IS NOT low THEN fan IS fast WITH 0.7")
        >>> rule.weight, str(rule.antecedent)
        (0.7, 'temp IS hot OR humidity IS NOT low')
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_rule needs a string, got {type(text).__name__}")
    return _Parser(text).rule()
