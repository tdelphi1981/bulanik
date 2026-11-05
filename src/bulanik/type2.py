r"""Interval type-2 fuzzy sets, type reduction and inference.

An interval type-2 membership function ([`IT2MF`][bulanik.type2.IT2MF]) maps
$x$ to an interval $[\underline{\mu}(x), \overline{\mu}(x)]$ of membership
degrees, written `(lower, upper)`: the area between the two bounds is the
footprint of uncertainty (FOU). The module has

- set builders: [`IT2MF`][bulanik.type2.IT2MF] from two type-1 functions, and the
  helpers [`gaussian_uncertain_mean`][bulanik.type2.gaussian_uncertain_mean],
  [`gaussian_uncertain_sigma`][bulanik.type2.gaussian_uncertain_sigma] and
  [`scaled_triangle`][bulanik.type2.scaled_triangle];
- [`IT2LinguisticVariable`][bulanik.type2.IT2LinguisticVariable], a variable
  with interval type-2 terms;
- type reduction: [`type_reduce`][bulanik.type2.type_reduce] turns points with
  interval weights into an interval $[y_l, y_r]$ of crisp outputs (result type
  [`TypeReduced`][bulanik.type2.TypeReduced]), with the Karnik-Mendel (`km`),
  enhanced KM (`ekm`), Nie-Tan (`nie_tan`) and Begian-Melek-Mendel (`bmm`) methods;
- systems: [`IT2MamdaniFIS`][bulanik.type2.IT2MamdaniFIS] and
  [`IT2SugenoFIS`][bulanik.type2.IT2SugenoFIS]. Rules (same syntax as type-1, see
  [`parse_rule`][bulanik.inference.rules.parse_rule]) fire with an interval
  $[\underline{f}, \overline{f}]$, and the output is the midpoint of the
  type-reduced interval.

Rules use `bulanik.inference` for parsing and share its options; a system built
from [`IT2LinguisticVariable.from_type1`][bulanik.type2.IT2LinguisticVariable.from_type1]
variables (no uncertainty) reproduces the type-1 result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bulanik._validate import TOL, choose, finite
from bulanik.inference._common import (
    AGGREGATION as _AGGREGATION,
    AND as _AND,
    IMPLICATION as _IMPLICATION,
    OR as _OR,
    aggregate as _aggregate,
    check_default,
    check_linear_inputs,
    check_variables,
    empty_term_message,
    fuzzify_inputs,
    imply as _imply,
    no_rule_fired,
    parse_rules,
)
from bulanik.inference.rules import (
    And,
    Antecedent,
    Explanation,
    Rule,
    RuleActivation,
    Term,
)
from bulanik.inference.sugeno import Linear, SugenoOutput
from bulanik.sets import LinguisticVariable, MembershipFunction, t_conorm, t_norm

__all__ = [
    "IT2LinguisticVariable",
    "IT2MF",
    "IT2MamdaniFIS",
    "IT2SugenoFIS",
    "NoLowerFiringError",
    "TypeReduced",
    "gaussian_uncertain_mean",
    "gaussian_uncertain_sigma",
    "scaled_triangle",
    "type_reduce",
]

_METHODS = ("km", "ekm", "nie_tan", "bmm")

Interval = tuple[float, float]


class NoLowerFiringError(ValueError):
    """`bmm` type reduction needs a positive lower weight and every one is 0.

    A `ValueError` raised by [`type_reduce`][bulanik.type2.type_reduce] and by
    the interval type-2 systems' `evaluate` / `evaluate_interval` / `explain`
    when `type_reduction="bmm"` and no rule fires with certainty (every lower
    firing is 0). Use `"km"`, `"ekm"` or `"nie_tan"` instead.

    Examples:
        >>> from bulanik.type2 import type_reduce
        >>> type_reduce([1, 2], [0, 0], [1, 1], "bmm")
        Traceback (most recent call last):
        ...
        bulanik.type2.NoLowerFiringError: bmm needs a positive lower weight somewhere (every lower weight is 0); use 'km', 'ekm' or 'nie_tan' instead
    """


# --------------------------------------------------------------------------- sets


@dataclass(frozen=True)
class IT2MF:
    r"""Interval type-2 membership function bounded by two type-1 functions.

    Calling `mf(x)` returns the pair `(lower, upper)` of membership degrees at
    `x`, so $\underline{\mu}(x) \le \overline{\mu}(x)$. The region between
    the two bounds is the footprint of uncertainty. A lower above the upper by
    more than $10^{-12}$ at some `x` raises `ValueError` when that `x` is
    evaluated; float noise within that tolerance is clamped.

    Args:
        upper: The upper membership function $\overline{\mu}$.
        lower: The lower membership function $\underline{\mu}$.

    Raises:
        TypeError: If `upper` or `lower` is not a
            [`MembershipFunction`][bulanik.sets.MembershipFunction].

    Attributes:
        upper: The upper membership function.
        lower: The lower membership function.

    Examples:
        >>> from bulanik.sets import triangular
        >>> from bulanik.type2 import IT2MF
        >>> mf = IT2MF(upper=triangular(0, 5, 10), lower=triangular(2, 5, 8))
        >>> mf(5), mf(3)
        ((1.0, 1.0), (0.3333333333333333, 0.6))
        >>> IT2MF(upper=triangular(2, 5, 8), lower=triangular(0, 5, 10))(1)
        Traceback (most recent call last):
        ...
        ValueError: lower membership 0.2 exceeds upper membership 0 at x=1
    """

    upper: MembershipFunction
    lower: MembershipFunction

    def __post_init__(self) -> None:
        for role in ("upper", "lower"):
            if not isinstance(getattr(self, role), MembershipFunction):
                raise TypeError(
                    f"IT2MF {role} must be a MembershipFunction, "
                    f"got {type(getattr(self, role)).__name__}"
                )

    def __call__(self, x: float) -> Interval:
        """The membership interval `(lower(x), upper(x))` at `x`; same as `mf(x)`.

        Raises:
            ValueError: If the lower membership exceeds the upper by more than
                the library tolerance.
        """
        lo, hi = self.lower(x), self.upper(x)
        if lo > hi:
            if lo > hi + TOL:
                raise ValueError(
                    f"lower membership {lo:g} exceeds upper membership {hi:g} at x={float(x):g}"
                )
            lo = hi
        return lo, hi


def gaussian_uncertain_mean(mean_low: float, mean_high: float, sigma: float) -> IT2MF:
    r"""Gaussian with a mean anywhere in $[m_1, m_2]$ (Mendel 2001).

    With $g(x; m) = \exp\bigl(-(x - m)^2 / (2\sigma^2)\bigr)$, the upper bound is
    1 on $[m_1, m_2]$ and the Gaussian of the nearer mean outside it; the lower
    bound is $\min\bigl(g(x; m_1), g(x; m_2)\bigr)$.

    Args:
        mean_low: Left end $m_1$ of the mean interval.
        mean_high: Right end $m_2$, with $m_1 \le m_2$.
        sigma: Common standard deviation, positive.

    Returns:
        The interval type-2 membership function.

    Raises:
        ValueError: If `mean_low > mean_high`, `sigma <= 0` or a value is not
            finite.

    Examples:
        >>> from bulanik.type2 import gaussian_uncertain_mean
        >>> mf = gaussian_uncertain_mean(4, 6, 1.5)
        >>> mf(5)
        (0.800..., 1.0)
        >>> mf(8)
        (0.0285..., 0.411...)

    References:
        Mendel (2001): Uncertain Rule-Based Fuzzy Logic Systems: Introduction
        and New Directions.
    """
    params = (mean_low, mean_high, sigma)
    return IT2MF(
        upper=MembershipFunction("gaussian_flat_top", params),
        lower=MembershipFunction("gaussian_min_pair", params),
    )


def gaussian_uncertain_sigma(mean: float, sigma_low: float, sigma_high: float) -> IT2MF:
    r"""Gaussian with a width anywhere in $[\sigma_1, \sigma_2]$.

    The upper bound is the Gaussian with $\sigma_2$ and the lower bound the
    Gaussian with $\sigma_1$, both with the same mean.

    Args:
        mean: The centre.
        sigma_low: Smallest width $\sigma_1$, positive.
        sigma_high: Largest width $\sigma_2$, with $\sigma_1 \le \sigma_2$.

    Returns:
        The interval type-2 membership function.

    Raises:
        ValueError: If `0 < sigma_low <= sigma_high` does not hold, or a value
            is not finite.

    Examples:
        >>> from bulanik.type2 import gaussian_uncertain_sigma
        >>> mf = gaussian_uncertain_sigma(5, 1, 2)
        >>> mf(6)
        (0.606..., 0.882...)
        >>> gaussian_uncertain_sigma(5, 2, 1)
        Traceback (most recent call last):
        ...
        ValueError: gaussian_uncertain_sigma needs 0 < sigma_low <= sigma_high, got 2, 1
    """
    if not 0.0 < sigma_low <= sigma_high:  # also rejects NaN
        raise ValueError(
            f"gaussian_uncertain_sigma needs 0 < sigma_low <= sigma_high, got {sigma_low}, {sigma_high}"
        )
    return IT2MF(
        upper=MembershipFunction("gaussian", (mean, sigma_high)),
        lower=MembershipFunction("gaussian", (mean, sigma_low)),
    )


def scaled_triangle(a: float, b: float, c: float, height: float) -> IT2MF:
    r"""Triangle $\operatorname{tri}(a, b, c)$ over a lower triangle $h \cdot \operatorname{tri}(a, b, c)$.

    The upper bound is the triangular function with feet `a`, `c` and peak `b`;
    the lower bound is the same triangle scaled to height $h$, $0 < h \le 1$.

    Args:
        a: Left foot.
        b: Peak.
        c: Right foot, with `a <= b <= c` and `a < c`.
        height: Height $h$ of the lower triangle, in $(0, 1]$.

    Returns:
        The interval type-2 membership function.

    Raises:
        ValueError: If the points are not ordered or `height` is outside
            $(0, 1]$.

    Examples:
        >>> from bulanik.type2 import scaled_triangle
        >>> mf = scaled_triangle(0, 5, 10, 0.6)
        >>> mf(5), mf(2.5)
        ((0.6, 1.0), (0.3, 0.5))
    """
    return IT2MF(
        upper=MembershipFunction("triangular", (a, b, c)),
        lower=MembershipFunction("scaled_triangular", (a, b, c, height)),
    )


class IT2LinguisticVariable:
    """A named variable on a universe `(lo, hi)` with interval type-2 terms.

    The type-2 counterpart of [`LinguisticVariable`][bulanik.sets.LinguisticVariable];
    `variable["term"]` returns the [`IT2MF`][bulanik.type2.IT2MF].

    Args:
        name: The variable name, used in rule text.
        universe: The pair `(lo, hi)` with `lo < hi`.
        terms: Mapping from term name to [`IT2MF`][bulanik.type2.IT2MF] (at
            least one).

    Raises:
        ValueError: If `lo >= hi`, an end is not finite, or `terms` is empty.
        TypeError: If a term is not an `IT2MF`.

    Attributes:
        name: The variable name.
        universe: The domain as a tuple of floats.
        term_names: The term names, in insertion order.

    Examples:
        >>> from bulanik.type2 import IT2LinguisticVariable, scaled_triangle
        >>> x = IT2LinguisticVariable("x", (0, 10), {
        ...     "lo": scaled_triangle(0, 0, 10, 0.5), "hi": scaled_triangle(0, 10, 10, 0.5)})
        >>> x
        IT2LinguisticVariable('x', (0.0, 10.0), terms=('lo', 'hi'))
        >>> x.fuzzify(2)
        {'lo': (0.4, 0.8), 'hi': (0.1, 0.2)}
    """

    def __init__(self, name: str, universe: tuple[float, float], terms: Mapping[str, IT2MF]) -> None:
        lo, hi = finite(universe[0], "universe lo"), finite(universe[1], "universe hi")
        if not lo < hi:
            raise ValueError(f"universe of '{name}' needs lo < hi, got ({lo}, {hi})")
        if not terms:
            raise ValueError(f"variable '{name}' needs at least one term")
        for t, mf in terms.items():
            if not isinstance(mf, IT2MF):
                raise TypeError(f"term '{t}' of '{name}' must be an IT2MF, got {type(mf).__name__}")
        self._name = name
        self._universe = (lo, hi)
        self._terms = dict(terms)

    name = property(lambda self: self._name, doc="The variable name.")
    universe = property(lambda self: self._universe, doc="The domain `(lo, hi)`.")

    @property
    def term_names(self) -> tuple[str, ...]:
        """The term names, in insertion order."""
        return tuple(self._terms)

    def __getitem__(self, term: str) -> IT2MF:
        """The [`IT2MF`][bulanik.type2.IT2MF] of `term`.

        Args:
            term: A term name.

        Returns:
            The interval type-2 membership function of the term.

        Raises:
            KeyError: If the term does not exist (the message lists the valid terms).
        """
        try:
            return self._terms[term]
        except KeyError:
            raise KeyError(
                f"unknown term '{term}' of '{self._name}'; valid: {', '.join(self._terms)}"
            ) from None

    def fuzzify(self, x: float) -> dict[str, Interval]:
        """`(lower, upper)` membership of `x` in every term.

        Args:
            x: A crisp input inside the universe.

        Returns:
            A dict from term name to the pair `(lower, upper)`, in term order.

        Raises:
            ValueError: If `x` is not finite or lies outside the universe.

        Examples:
            >>> from bulanik.type2 import IT2LinguisticVariable, scaled_triangle
            >>> x = IT2LinguisticVariable("x", (0, 10), {"mid": scaled_triangle(0, 5, 10, 0.8)})
            >>> x.fuzzify(5)
            {'mid': (0.8, 1.0)}
            >>> x.fuzzify(11)
            Traceback (most recent call last):
            ...
            ValueError: 'x' input 11 is outside its universe [0, 10]
        """
        x = finite(x, f"input of '{self._name}'")
        lo, hi = self._universe
        if not lo <= x <= hi:
            raise ValueError(
                f"'{self._name}' input {x:g} is outside its universe [{lo:g}, {hi:g}]"
            )
        return {t: mf(x) for t, mf in self._terms.items()}

    @classmethod
    def from_type1(cls, lv: LinguisticVariable) -> IT2LinguisticVariable:
        """The type-1 variable as a type-2 one with no uncertainty (lower = upper).

        Args:
            lv: A [`LinguisticVariable`][bulanik.sets.LinguisticVariable].

        Returns:
            An `IT2LinguisticVariable` with the same name, universe and terms,
            each term's lower and upper bound being the type-1 function.

        Raises:
            TypeError: If `lv` is not a `LinguisticVariable`.

        Examples:
            >>> from bulanik.sets import LinguisticVariable, linear_decreasing
            >>> from bulanik.type2 import IT2LinguisticVariable
            >>> lv = LinguisticVariable("x", (0, 10), {"lo": linear_decreasing(0, 10)})
            >>> IT2LinguisticVariable.from_type1(lv).fuzzify(2.5)
            {'lo': (0.75, 0.75)}
        """
        if not isinstance(lv, LinguisticVariable):
            raise TypeError(f"from_type1 needs a LinguisticVariable, got {type(lv).__name__}")
        return cls(lv.name, lv.universe, {t: IT2MF(upper=lv[t], lower=lv[t]) for t in lv.term_names})

    def __repr__(self) -> str:
        return f"IT2LinguisticVariable({self._name!r}, {self._universe}, terms={self.term_names})"


# --------------------------------------------------------------------------- type reduction


@dataclass(frozen=True)
class TypeReduced:
    """Type-reduced interval `[left, right]` and the crisp `output`.

    Returned by [`type_reduce`][bulanik.type2.type_reduce] and, per output
    variable, by `evaluate_interval` of the interval type-2 systems.

    Args:
        left: Left end $y_l$ of the interval.
        right: Right end $y_r$.
        output: The crisp output, the interval midpoint (for `bmm`, the
            weighted combination of its two averages).
        left_switch: KM / EKM switch point of the left end, or `None`.
        right_switch: KM / EKM switch point of the right end, or `None`.

    Attributes:
        left: Left end of the interval.
        right: Right end of the interval.
        output: The crisp output.
        left_switch: The 1-based position in the `y`-sorted samples where the
            left end switches from upper to lower weights, or `None` for the
            closed-form reducers (`nie_tan`, `bmm`).
        right_switch: The same for the right end, where the weights switch
            from lower to upper instead.

    Examples:
        >>> from bulanik.type2 import type_reduce
        >>> r = type_reduce([1, 2, 3], [0.2, 0.3, 0.1], [0.6, 0.8, 0.5])
        >>> (r.left, r.right, r.output, r.left_switch, r.right_switch)
        (1.5, 2.3, 1.9, 1, 2)
    """

    left: float
    right: float
    output: float
    left_switch: int | None
    right_switch: int | None


def _number(v: Any, what: str) -> float:
    if isinstance(v, (bool, str)):
        raise TypeError(f"{what} must be numbers, got {type(v).__name__} {v!r}")
    return float(v)


def _check_weights(
    ys: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> tuple[list[float], list[float], list[float]]:
    if not len(ys) == len(lower) == len(upper):
        raise ValueError(
            f"ys, lower and upper must have the same length, got {len(ys)}, {len(lower)}, {len(upper)}"
        )
    ys = [_number(y, "ys") for y in ys]
    lo = [_number(v, "lower weight") for v in lower]
    hi = [_number(v, "upper weight") for v in upper]
    if not all(math.isfinite(v) for v in ys):
        raise ValueError("ys must all be finite")
    if not all(math.isfinite(v) for v in lo + hi):
        raise ValueError("lower and upper weights must all be finite")
    for k, (a, b) in enumerate(zip(lo, hi)):
        if a < -TOL:
            raise ValueError(f"weights must not be negative, got lower {a} at position {k}")
        if a > b + TOL:
            raise ValueError(f"lower weight {a} exceeds upper weight {b} at position {k}")
        lo[k] = min(max(a, 0.0), b)  # float noise within TOL is clamped
        hi[k] = max(b, lo[k])
    if not math.fsum(hi) > 0.0:
        raise ValueError("cannot type-reduce: the upper weights are all zero")
    return ys, lo, hi


def _wavg(ys: Sequence[float], ws: Sequence[float]) -> float:
    return math.fsum(y * w for y, w in zip(ys, ws)) / math.fsum(ws)


def _switch_avg(ys: list[float], first: list[float], second: list[float], s: int) -> float:
    """Weighted mean with ``first`` weights for positions <= s and ``second`` after."""
    return _wavg(ys, first[:s] + second[s:])


def _endpoint(ys: list[float], lo: list[float], hi: list[float], left: bool, ekm: bool) -> tuple[float, int]:
    """One end of the centroid interval by KM or EKM iteration over sorted ``ys``.

    Left end: upper weights up to the switch, lower after; the switch is the
    number of samples with ``y <= c``. Right end: the mirror image with
    ``y < c``. The running c is a mean of samples with positive upper weight,
    so the switch keeps the first (left) or last (right) such sample on its
    upper weight and every denominator is positive; the clamp below only
    absorbs a c rounded one ulp past that sample. The loop stops when a switch
    repeats, so it ends after at most N + 1 steps.
    """
    n = len(ys)
    first, second = (hi, lo) if left else (lo, hi)
    positive = [k for k, h in enumerate(hi) if h > 0.0]
    first_pos, last_pos = positive[0] + 1, positive[-1]

    def switch(c: float) -> int:
        if left:
            return max(sum(1 for y in ys if y <= c), first_pos)
        return min(sum(1 for y in ys if y < c), last_pos)

    visited: dict[int, float] = {}
    if ekm:
        # Wu & Mendel (2009): start near the usual optimum and update the sums incrementally.
        # clamped like every later switch, so the denominator always has a positive upper weight
        s = round(n / 2.4 if left else n / 1.7)
        s = max(s, first_pos) if left else min(s, last_pos)
        ws = first[:s] + second[s:]
        num = math.fsum(y * w for y, w in zip(ys, ws))
        den = math.fsum(ws)
        visited[s] = num / den
        while True:
            s_new = switch(visited[s])
            if s_new in visited:
                break
            lo_k, hi_k = min(s, s_new), max(s, s_new)
            sign = 1.0 if s_new > s else -1.0
            # positions lo_k+1 .. hi_k move from `second` to `first` (or back)
            diff = [first[k] - second[k] for k in range(lo_k, hi_k)]
            num += sign * math.fsum(ys[k] * d for k, d in zip(range(lo_k, hi_k), diff))
            den += sign * math.fsum(diff)
            visited[s_new] = num / den
            s = s_new
        pick = min if left else max
        best = pick(visited, key=visited.__getitem__)
        return _switch_avg(ys, first, second, best), best  # exact recompute, as KM does
    # Karnik & Mendel (2001): start from the mid weights.
    c = _wavg(ys, [(a + b) / 2.0 for a, b in zip(lo, hi)])
    while True:
        s = switch(c)
        if s in visited:
            break
        c = visited[s] = _switch_avg(ys, first, second, s)
    pick = min if left else max
    best = pick(visited, key=visited.__getitem__)
    return visited[best], best


def type_reduce(
    ys: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    method: str = "km",
    *,
    bmm_m: float = 0.5,
    bmm_n: float = 0.5,
) -> TypeReduced:
    r"""Reduce points `ys` with interval weights `[lower, upper]` to a crisp value.

    Each point $y_i$ may carry any weight $w_i \in [\underline{w}_i, \overline{w}_i]$.
    The centroid interval is

    $$[y_l, y_r], \qquad y_l = \min_{w} \frac{\sum_i y_i w_i}{\sum_i w_i}, \quad
    y_r = \max_{w} \frac{\sum_i y_i w_i}{\sum_i w_i},$$

    and the methods are

    - `"km"` (Karnik and Mendel 2001) and `"ekm"` (enhanced KM, Wu and Mendel
      2009) find the exact $[y_l, y_r]$; the output is its midpoint. Both
      converge to the same endpoints, but only up to rounding (about 1 ulp),
      and on plateaus (equal `ys` or equal weights) they may stop at different,
      equally optimal switch indices.
    - `"nie_tan"` (Nie and Tan 2008) is the closed form
      $\sum_i y_i (\underline{w}_i + \overline{w}_i) / \sum_i (\underline{w}_i + \overline{w}_i)$,
      with `left == right == output`.
    - `"bmm"` (Begian, Melek and Mendel 2008) is $m\,\bar y_{\text{lower}} + n\,\bar y_{\text{upper}}$,
      the combination of the weighted averages with the lower and with the upper
      weights; `left` / `right` are the two averages, sorted. `bmm_m` and
      `bmm_n` only have to be finite: they are deliberately unconstrained (the
      usual choice is $m = n = 0.5$, but neither sign nor $m + n = 1$ is
      enforced, so the output may fall outside `[left, right]`). It needs
      some positive lower weight.

    Args:
        ys: The points.
        lower: Lower weight of each point.
        upper: Upper weight of each point.
        method: `"km"`, `"ekm"`, `"nie_tan"` or `"bmm"`.
        bmm_m: Factor $m$ of the lower average (`"bmm"` only).
        bmm_n: Factor $n$ of the upper average (`"bmm"` only).

    Returns:
        A [`TypeReduced`][bulanik.type2.TypeReduced] with the interval and the
        crisp output.

    Raises:
        TypeError: If a point or weight is a `bool` or a string.
        ValueError: If `method` is unknown, the lengths differ, a value is not
            finite, a weight is negative, a lower weight exceeds its upper
            weight (by more than $10^{-12}$), or all upper weights are zero.
        NoLowerFiringError: If `method="bmm"` and every lower weight is zero
            (a `ValueError`).

    Examples:
        >>> from bulanik.type2 import type_reduce
        >>> ys, lo, hi = [1, 2, 3], [0.2, 0.3, 0.1], [0.6, 0.8, 0.5]
        >>> r = type_reduce(ys, lo, hi)
        >>> (r.left, r.right, r.output)
        (1.5, 2.3, 1.9)
        >>> type_reduce(ys, lo, hi, "nie_tan").output
        1.92
        >>> type_reduce(ys, [0.5] * 3, [0.5] * 3).output
        2.0

    References:
        Karnik and Mendel (2001): "Centroid of a type-2 fuzzy set", Information
        Sciences.
        Wu and Mendel (2009): "Enhanced Karnik-Mendel algorithms", IEEE
        Transactions on Fuzzy Systems.
        Nie and Tan (2008): "Towards an efficient type-reduction method for
        interval type-2 fuzzy logic systems", FUZZ-IEEE.
        Begian, Melek and Mendel (2008): "Stability analysis of type-2 fuzzy systems",
        IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), 947-953 (the
        BMM reducer).
    """
    method = choose(method, _METHODS, "type reduction method", "type_reduce")
    ys, lo, hi = _check_weights(ys, lower, upper)
    if method == "nie_tan":
        y = _wavg(ys, [a + b for a, b in zip(lo, hi)])
        return TypeReduced(y, y, y, None, None)
    if method == "bmm":
        m, n = finite(bmm_m, "bmm_m"), finite(bmm_n, "bmm_n")
        if not math.fsum(lo) > 0.0:
            raise NoLowerFiringError(
                "bmm needs a positive lower weight somewhere (every lower weight is 0); "
                "use 'km', 'ekm' or 'nie_tan' instead"
            )
        a, b = _wavg(ys, lo), _wavg(ys, hi)
        return TypeReduced(min(a, b), max(a, b), m * a + n * b, None, None)
    order = sorted(range(len(ys)), key=ys.__getitem__)
    ys = [ys[i] for i in order]
    lo = [lo[i] for i in order]
    hi = [hi[i] for i in order]
    ekm = method == "ekm"
    left, ls = _endpoint(ys, lo, hi, True, ekm)
    right, rs = _endpoint(ys, lo, hi, False, ekm)
    return TypeReduced(left, right, (left + right) / 2.0, ls, rs)


# --------------------------------------------------------------------------- inference


def _interval_strength(
    antecedent: Antecedent, degrees: Mapping[str, Mapping[str, Interval]], and_method: str, or_method: str
) -> Interval:
    """Firing interval: each bound over the matching bound degrees; NOT is ``(1 - hi, 1 - lo)``."""
    if isinstance(antecedent, Term):
        lo, hi = degrees[antecedent.variable][antecedent.term]
        return (1.0 - hi, 1.0 - lo) if antecedent.negated else (lo, hi)
    parts = [_interval_strength(p, degrees, and_method, or_method) for p in antecedent.parts]
    combine = t_norm if isinstance(antecedent, And) else t_conorm
    kind = and_method if isinstance(antecedent, And) else or_method
    lo, hi = parts[0]
    for plo, phi in parts[1:]:
        lo, hi = combine(lo, plo, kind), combine(hi, phi, kind)
    return lo, hi


def _interval_strengths(
    rules: Sequence[Rule], degrees: Mapping[str, Mapping[str, Interval]], and_method: str, or_method: str
) -> list[Interval]:
    """Rule weight x firing interval, for every rule (the weight scales both bounds)."""
    out: list[Interval] = []
    for r in rules:
        lo, hi = _interval_strength(r.antecedent, degrees, and_method, or_method)
        out.append((r.weight * lo, r.weight * hi))
    return out


class _IT2Base:
    """Shared evaluation flow; subclasses implement ``_reduce`` for one output."""

    _owner = "IT2FIS"

    def _setup(self, and_method: str, or_method: str, type_reduction: str) -> None:
        self._and_method = choose(and_method, _AND, "and_method", self._owner)
        self._or_method = choose(or_method, _OR, "or_method", self._owner)
        self._type_reduction = choose(type_reduction, _METHODS, "type reduction method", self._owner)

    inputs = property(lambda self: self._inputs, doc="Input variables (read-only).")
    outputs = property(lambda self: self._outputs, doc="Output variables (read-only).")
    rules = property(lambda self: self._rules, doc="Parsed rules (read-only).")
    and_method = property(lambda self: self._and_method, doc="The AND t-norm, `min` or `prod`.")
    or_method = property(lambda self: self._or_method, doc="The OR t-conorm, `max` or `probor`.")
    type_reduction = property(
        lambda self: self._type_reduction, doc="The type reduction method: `km`, `ekm`, `nie_tan` or `bmm`."
    )
    default = property(
        lambda self: None if self._default is None else dict(self._default),
        doc="A copy of the value (or `(left, right)` pair) returned per output when no rule fires, or `None`.",
    )

    def _run(
        self, inputs: Mapping[str, float]
    ) -> tuple[dict[str, dict[str, Interval]], list[Interval], dict[str, TypeReduced]]:
        degrees = fuzzify_inputs(self._inputs, inputs)
        strengths = _interval_strengths(self._rules, degrees, self._and_method, self._or_method)
        results: dict[str, TypeReduced] = {}
        for o in self._outputs:
            try:
                r = self._reduce(o, strengths, inputs)
            except NoLowerFiringError as exc:
                raise NoLowerFiringError(
                    f"{self._owner}: output '{o.name}' cannot be type-reduced with "
                    "type_reduction='bmm': every rule's lower firing is 0 (no rule fires with "
                    "certainty, only possibly); use 'km', 'ekm' or 'nie_tan' instead"
                ) from exc
            if r is None:
                if self._default is None:
                    raise no_rule_fired(
                        o.name,
                        degrees,
                        self._rule_text,
                        strengths,
                        {k: x.output for k, x in results.items()},
                    )
                d = self._default[o.name]
                r = TypeReduced(d, d, d, None, None) if isinstance(d, float) else TypeReduced(
                    d[0], d[1], (d[0] + d[1]) / 2.0, None, None
                )
            results[o.name] = r
        return degrees, strengths, results

    def evaluate_interval(self, inputs: Mapping[str, float]) -> dict[str, TypeReduced]:
        """Type-reduced interval and crisp output of every output variable.

        Args:
            inputs: Crisp value of every input variable, by name. The keys must
                be exactly the input names and each value inside its universe.

        Returns:
            A dict from output variable name to its
            [`TypeReduced`][bulanik.type2.TypeReduced].

        Raises:
            TypeError: If a value is a `bool` or a string.
            ValueError: If an input is missing, unexpected, not finite or
                outside its universe.
            NoRuleFiredError: If no rule fires for an output and there is no
                `default`.
            NoLowerFiringError: If `type_reduction="bmm"` and no rule has a
                positive lower firing.

        Examples:
            >>> from bulanik.type2 import IT2LinguisticVariable, IT2SugenoFIS, scaled_triangle
            >>> from bulanik.inference import SugenoOutput
            >>> x = IT2LinguisticVariable("x", (0, 10), {
            ...     "lo": scaled_triangle(0, 0, 10, 0.5), "hi": scaled_triangle(0, 10, 10, 0.5)})
            >>> y = SugenoOutput("y", {"a": 0.0, "b": 10.0})
            >>> fis = IT2SugenoFIS([x], [y], ["IF x IS lo THEN y IS a", "IF x IS hi THEN y IS b"])
            >>> r = fis.evaluate_interval({"x": 7})["y"]
            >>> round(r.left, 3), round(r.right, 3), round(r.output, 3)
            (5.385, 8.235, 6.81)
        """
        return self._run(inputs)[2]

    def evaluate(self, inputs: Mapping[str, float]) -> dict[str, float]:
        """Crisp outputs (the type-reduced `output`) for crisp inputs.

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
            NoLowerFiringError: If `type_reduction="bmm"` and no rule has a
                positive lower firing.

        Examples:
            >>> from bulanik.type2 import IT2LinguisticVariable, IT2SugenoFIS, scaled_triangle
            >>> from bulanik.inference import SugenoOutput
            >>> x = IT2LinguisticVariable("x", (0, 10), {
            ...     "lo": scaled_triangle(0, 0, 10, 0.5), "hi": scaled_triangle(0, 10, 10, 0.5)})
            >>> y = SugenoOutput("y", {"a": 0.0, "b": 10.0})
            >>> rules = ["IF x IS lo THEN y IS a", "IF x IS hi THEN y IS b"]
            >>> IT2SugenoFIS([x], [y], rules, type_reduction="nie_tan").evaluate({"x": 7})
            {'y': 7.0}
        """
        return {k: r.output for k, r in self._run(inputs)[2].items()}

    def explain(self, inputs: Mapping[str, float]) -> Explanation:
        """Degrees `(lower, upper)`, rule firing intervals and crisp outputs.

        Args:
            inputs: Crisp value of every input variable, by name.

        Returns:
            An [`Explanation`][bulanik.inference.rules.Explanation] whose
            `degrees` and rule `activations` hold `(lower, upper)` pairs and
            whose `outputs` are the crisp (type-reduced) outputs.

        Raises:
            TypeError: If a value is a `bool` or a string.
            ValueError: If an input is missing, unexpected, not finite or
                outside its universe.
            NoRuleFiredError: If no rule fires for an output and there is no
                `default`.
            NoLowerFiringError: If `type_reduction="bmm"` and no rule has a
                positive lower firing.

        Examples:
            >>> from bulanik.type2 import IT2LinguisticVariable, IT2SugenoFIS, scaled_triangle
            >>> from bulanik.inference import SugenoOutput
            >>> x = IT2LinguisticVariable("x", (0, 10), {
            ...     "lo": scaled_triangle(0, 0, 10, 0.5), "hi": scaled_triangle(0, 10, 10, 0.5)})
            >>> y = SugenoOutput("y", {"a": 0.0, "b": 10.0})
            >>> fis = IT2SugenoFIS([x], [y], ["IF x IS lo THEN y IS a", "IF x IS hi THEN y IS b"])
            >>> info = fis.explain({"x": 7})
            >>> info.degrees["x"]
            {'lo': (0.15, 0.3), 'hi': (0.35, 0.7)}
            >>> info.activations[1].strength
            (0.35, 0.7)
        """
        degrees, strengths, results = self._run(inputs)
        activations = tuple(RuleActivation(t, s) for t, s in zip(self._rule_text, strengths))
        return Explanation(
            dict(degrees), activations, {k: r.output for k, r in results.items()}
        )


class IT2MamdaniFIS(_IT2Base):
    r"""Interval type-2 Mamdani system with the centroid type reducer.

    Each rule fires with an interval $[\underline{f}_k, \overline{f}_k]$ (the
    rule weight times the antecedent firing, computed with the lower and upper
    membership bounds separately; `NOT` maps $[a, b]$ to $[1 - b, 1 - a]$).
    Each output universe is sampled at `resolution` points $x_i$. At each
    sample the upper and lower aggregated sets are

    $$\overline{\mu}(x_i) = \operatorname{agg}_k \operatorname{imp}\bigl(\overline{f}_k,
    \overline{\mu}_{B_k}(x_i)\bigr), \qquad
    \underline{\mu}(x_i) = \operatorname{agg}_k \operatorname{imp}\bigl(\underline{f}_k,
    \underline{\mu}_{B_k}(x_i)\bigr),$$

    and [`type_reduce`][bulanik.type2.type_reduce] turns the pair into the
    interval $[y_l, y_r]$ whose midpoint is the output. With lower = upper
    everywhere it gives exactly the type-1 centroid. If the upper set is zero
    everywhere no rule fired, and the `default` is returned or
    [`NoRuleFiredError`][bulanik.inference.rules.NoRuleFiredError] is raised. A
    `default` value is a number or a `(left, right)` pair (the type-reduced
    interval to report; the output is its midpoint). `type_reduction="bmm"`
    additionally needs some rule with a positive lower firing; without one,
    evaluation raises [`NoLowerFiringError`][bulanik.type2.NoLowerFiringError]
    naming the output.

    Args:
        inputs: Input [`IT2LinguisticVariable`][bulanik.type2.IT2LinguisticVariable]s.
        outputs: Output interval type-2 variables (at least one; names distinct
            from the inputs).
        rules: [`Rule`][bulanik.inference.rules.Rule] objects and/or rule text.
        and_method: `"min"` or `"prod"`.
        or_method: `"max"` or `"probor"`.
        implication: `"min"` or `"prod"`.
        aggregation: `"max"`, `"sum"` or `"probor"`.
        type_reduction: `"km"`, `"ekm"`, `"nie_tan"` or `"bmm"`.
        resolution: Samples per output universe, an integer of at least 2.
        default: Output value, or `(left, right)` interval, per output
            variable, used when no rule fires.

    Raises:
        TypeError: If a variable is not an `IT2LinguisticVariable` or a rule is
            neither a `Rule` nor a string.
        ValueError: If an option name is unknown, `resolution` is not an
            integer of at least 2, there are no inputs, outputs or rules, a
            variable name is duplicated, a rule names an unknown variable or
            term, an output term used by a rule has upper membership 0 at every
            sample, or `default` is malformed.

    Attributes:
        inputs: The input variables (read-only).
        outputs: The output variables (read-only).
        rules: The parsed rules (read-only).
        and_method: The t-norm name.
        or_method: The t-conorm name.
        implication: The implication name.
        aggregation: The aggregation name.
        type_reduction: The type reduction method.
        resolution: Samples per output universe.
        default: A copy of the no-fire defaults, or `None`.

    Examples:
        >>> from bulanik.type2 import IT2LinguisticVariable, IT2MamdaniFIS, scaled_triangle
        >>> x = IT2LinguisticVariable("x", (0, 10), {
        ...     "lo": scaled_triangle(0, 0, 10, 0.5), "hi": scaled_triangle(0, 10, 10, 0.5)})
        >>> tip = IT2LinguisticVariable("tip", (0, 20), {
        ...     "low": scaled_triangle(0, 0, 10, 0.6), "high": scaled_triangle(10, 20, 20, 0.6)})
        >>> fis = IT2MamdaniFIS([x], [tip], [
        ...     "IF x IS lo THEN tip IS low", "IF x IS hi THEN tip IS high"], resolution=201)
        >>> fis.evaluate({"x": 7})
        {'tip': 12.01...}
        >>> r = fis.evaluate_interval({"x": 7})["tip"]
        >>> round(r.left, 2), round(r.right, 2)
        (10.2, 13.83)

        Without uncertainty the system gives the type-1 answer:

        >>> from bulanik.sets import LinguisticVariable, triangular
        >>> from bulanik.inference import MamdaniFIS
        >>> t1x = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 10), "hi": triangular(0, 10, 10)})
        >>> t1y = LinguisticVariable("tip", (0, 20), {"low": triangular(0, 0, 10), "high": triangular(10, 20, 20)})
        >>> rules = ["IF x IS lo THEN tip IS low", "IF x IS hi THEN tip IS high"]
        >>> type1 = MamdaniFIS([t1x], [t1y], rules, resolution=201).evaluate({"x": 7})
        >>> type2 = IT2MamdaniFIS([IT2LinguisticVariable.from_type1(t1x)],
        ...                       [IT2LinguisticVariable.from_type1(t1y)], rules,
        ...                       resolution=201).evaluate({"x": 7})
        >>> abs(type1["tip"] - type2["tip"]) < 1e-9
        True

    References:
        Mendel (2001): Uncertain Rule-Based Fuzzy Logic Systems: Introduction
        and New Directions.
    """

    _owner = "IT2MamdaniFIS"

    def __init__(
        self,
        inputs: Sequence[IT2LinguisticVariable],
        outputs: Sequence[IT2LinguisticVariable],
        rules: Sequence[Rule | str],
        *,
        and_method: str = "min",
        or_method: str = "max",
        implication: str = "min",
        aggregation: str = "max",
        type_reduction: str = "km",
        resolution: int = 1001,
        default: Mapping[str, float | tuple[float, float]] | None = None,
    ) -> None:
        self._setup(and_method, or_method, type_reduction)
        self._implication = choose(implication, _IMPLICATION, "implication", self._owner)
        self._aggregation = choose(aggregation, _AGGREGATION, "aggregation", self._owner)
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2:
            raise ValueError(f"resolution must be an integer >= 2, got {resolution!r}")
        for role, vs in (("inputs", inputs), ("outputs", outputs)):
            for v in vs:
                if not isinstance(v, IT2LinguisticVariable):
                    raise TypeError(
                        f"IT2MamdaniFIS {role} must be IT2LinguisticVariable, got {type(v).__name__}"
                    )
        self._resolution = resolution
        self._inputs, self._outputs = check_variables(inputs, outputs)
        self._rules = parse_rules(rules, self._inputs, self._outputs)
        self._default = check_default(self._outputs, default, allow_interval=True)
        self._rule_text = tuple(str(r) for r in self._rules)
        n = resolution
        self._grid: dict[str, list[float]] = {}
        self._terms: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
        for v in self._outputs:
            lo, hi = v.universe
            step = (hi - lo) / (n - 1)
            xs = [lo + i * step for i in range(n - 1)] + [hi]
            self._grid[v.name] = xs
            for t in v.term_names:
                pairs = [v[t](x) for x in xs]
                self._terms[(v.name, t)] = ([p[0] for p in pairs], [p[1] for p in pairs])
        for rule in self._rules:
            for out, term in rule.consequents.items():
                if not any(u > 0.0 for u in self._terms[(out, term)][1]):
                    raise ValueError(empty_term_message(self._owner, out, term, self._grid[out], n))

    implication = property(lambda self: self._implication, doc="The implication name, `min` or `prod`.")
    aggregation = property(
        lambda self: self._aggregation, doc="The aggregation name, `max`, `sum` or `probor`."
    )
    resolution = property(lambda self: self._resolution, doc="Samples per output universe.")

    def _reduce(
        self, v: IT2LinguisticVariable, strengths: list[Interval], inputs: Mapping[str, float]
    ) -> TypeReduced | None:
        n, imp, agg = self._resolution, self._implication, self._aggregation
        lower, upper = [0.0] * n, [0.0] * n
        for rule, (f_lo, f_hi) in zip(self._rules, strengths):
            term = rule.consequents.get(v.name)
            if term is None or f_hi == 0.0:  # f_lo <= f_hi, so both sets are 0 everywhere
                continue
            mu_lo, mu_hi = self._terms[(v.name, term)]
            for i in range(n):
                lower[i] = _aggregate(lower[i], _imply(f_lo, mu_lo[i], imp), agg)
                upper[i] = _aggregate(upper[i], _imply(f_hi, mu_hi[i], imp), agg)
        if not any(u > 0.0 for u in upper):
            return None
        return type_reduce(self._grid[v.name], lower, upper, self._type_reduction)


class IT2SugenoFIS(_IT2Base):
    r"""Interval type-2 Sugeno system with the centre-of-sets type reducer.

    Each fired rule $k$ gives $z_k$ (a constant or a
    [`Linear`][bulanik.inference.sugeno.Linear] of the inputs) with firing
    interval $[\underline{f}_k, \overline{f}_k]$; the type-reduced interval is
    `type_reduce([z_k], [f_lower_k], [f_upper_k])` and the output its midpoint.
    When no rule fires for an output (all upper firings are 0) the `default`
    (a number or a `(left, right)` pair) is returned or
    [`NoRuleFiredError`][bulanik.inference.rules.NoRuleFiredError] is raised.
    `type_reduction="bmm"` needs some positive lower firing, else
    [`NoLowerFiringError`][bulanik.type2.NoLowerFiringError] is raised.

    Args:
        inputs: Input [`IT2LinguisticVariable`][bulanik.type2.IT2LinguisticVariable]s.
        outputs: [`SugenoOutput`][bulanik.inference.sugeno.SugenoOutput]s (at
            least one; names distinct from the inputs).
        rules: [`Rule`][bulanik.inference.rules.Rule] objects and/or rule text.
        and_method: `"min"` or `"prod"`.
        or_method: `"max"` or `"probor"`.
        type_reduction: `"km"`, `"ekm"`, `"nie_tan"` or `"bmm"`.
        default: Output value, or `(left, right)` interval, per output
            variable, used when no rule fires.

    Raises:
        TypeError: If a variable has the wrong class or a rule is neither a
            `Rule` nor a string.
        ValueError: If an option name is unknown, there are no inputs, outputs
            or rules, a variable name is duplicated, a rule names an unknown
            variable or term, a `Linear` names an unknown input, or `default`
            is malformed.

    Attributes:
        inputs: The input variables (read-only).
        outputs: The output variables (read-only).
        rules: The parsed rules (read-only).
        and_method: The t-norm name.
        or_method: The t-conorm name.
        type_reduction: The type reduction method.
        default: A copy of the no-fire defaults, or `None`.

    Examples:
        >>> from bulanik.type2 import IT2LinguisticVariable, IT2SugenoFIS, scaled_triangle
        >>> from bulanik.inference import SugenoOutput
        >>> x = IT2LinguisticVariable("x", (0, 10), {
        ...     "lo": scaled_triangle(0, 0, 10, 0.5), "hi": scaled_triangle(0, 10, 10, 0.5)})
        >>> y = SugenoOutput("y", {"a": 0.0, "b": 10.0})
        >>> fis = IT2SugenoFIS([x], [y], ["IF x IS lo THEN y IS a", "IF x IS hi THEN y IS b"])
        >>> fis.evaluate({"x": 7})
        {'y': 6.809...}
        >>> fis.evaluate_interval({"x": 7})["y"].left_switch
        1
    """

    _owner = "IT2SugenoFIS"

    def __init__(
        self,
        inputs: Sequence[IT2LinguisticVariable],
        outputs: Sequence[SugenoOutput],
        rules: Sequence[Rule | str],
        *,
        and_method: str = "min",
        or_method: str = "max",
        type_reduction: str = "km",
        default: Mapping[str, float | tuple[float, float]] | None = None,
    ) -> None:
        self._setup(and_method, or_method, type_reduction)
        for v in inputs:
            if not isinstance(v, IT2LinguisticVariable):
                raise TypeError(
                    f"IT2SugenoFIS inputs must be IT2LinguisticVariable, got {type(v).__name__}"
                )
        for o in outputs:
            if not isinstance(o, SugenoOutput):
                raise TypeError(f"IT2SugenoFIS outputs must be SugenoOutput, got {type(o).__name__}")
        self._inputs, self._outputs = check_variables(inputs, outputs)
        check_linear_inputs(self._inputs, self._outputs)
        self._rules = parse_rules(rules, self._inputs, self._outputs)
        self._default = check_default(self._outputs, default, allow_interval=True)
        self._rule_text = tuple(str(r) for r in self._rules)

    def _reduce(
        self, o: SugenoOutput, strengths: list[Interval], inputs: Mapping[str, float]
    ) -> TypeReduced | None:
        crisp = {v.name: float(inputs[v.name]) for v in self._inputs}
        zs: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for rule, (f_lo, f_hi) in zip(self._rules, strengths):
            term = rule.consequents.get(o.name)
            if term is None or f_hi == 0.0:
                continue
            f = o.functions[term]
            zs.append(f(crisp) if isinstance(f, Linear) else f)
            lows.append(f_lo)
            highs.append(f_hi)
        if not zs:
            return None
        return type_reduce(zs, lows, highs, self._type_reduction)
