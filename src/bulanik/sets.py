r"""Membership functions, linguistic variables and t-norms / t-conorms.

The building blocks of fuzzy inference:

- [`MembershipFunction`][bulanik.sets.MembershipFunction] and the factories
  [`triangular`][bulanik.sets.triangular], [`trapezoidal`][bulanik.sets.trapezoidal],
  [`gaussian`][bulanik.sets.gaussian], [`bell`][bulanik.sets.bell],
  [`sigmoid`][bulanik.sets.sigmoid], [`linear_increasing`][bulanik.sets.linear_increasing]
  and [`linear_decreasing`][bulanik.sets.linear_decreasing] build functions
  $\mu: \mathbb{R} \to [0, 1]$.
- [`LinguisticVariable`][bulanik.sets.LinguisticVariable] names a variable on a
  universe and its fuzzy terms (for example `cold`, `warm`, `hot`).
- [`t_norm`][bulanik.sets.t_norm] and [`t_conorm`][bulanik.sets.t_conorm] are the
  fuzzy AND and OR used to combine rule antecedents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ._validate import _num, choose, finite, unit

__all__ = [
    "SIGMOID_EPS",
    "MembershipFunction",
    "LinguisticVariable",
    "triangular",
    "trapezoidal",
    "gaussian",
    "bell",
    "sigmoid",
    "linear_increasing",
    "linear_decreasing",
    "t_norm",
    "t_conorm",
]

SIGMOID_EPS = 1e-9
"""Sigmoid memberships never reach exactly 0 or 1, so `inverse` of a sigmoid clamps
alpha into $[\\varepsilon, 1 - \\varepsilon]$ with $\\varepsilon = 10^{-9}$ instead of returning
$\\pm\\infty$."""

_KINDS = (
    "triangular",
    "trapezoidal",
    "gaussian",
    "bell",
    "sigmoid",
    "linear_increasing",
    "linear_decreasing",
    "gaussian_flat_top",
    "gaussian_min_pair",
    "scaled_triangular",
)
_MONOTONIC = ("linear_increasing", "linear_decreasing", "sigmoid")
_ARITY = {
    "triangular": 3,
    "trapezoidal": 4,
    "gaussian": 2,
    "bell": 3,
    "sigmoid": 2,
    "linear_increasing": 2,
    "linear_decreasing": 2,
    "gaussian_flat_top": 3,
    "gaussian_min_pair": 3,
    "scaled_triangular": 4,
}


def _need(ok: bool, kind: str, rule: str, params: tuple[float, ...]) -> None:
    if not ok:
        raise ValueError(f"{kind} needs {rule}, got {params}")


# Per-kind domain checks, run by MembershipFunction.__post_init__ (so direct
# construction is as safe as the factories). Add new kinds here and in _ARITY.
_VALIDATORS = {
    "triangular": lambda p: _need(
        p[0] <= p[1] <= p[2] and p[0] < p[2], "triangular", "a <= b <= c and a < c", p
    ),
    "trapezoidal": lambda p: _need(
        p[0] <= p[1] <= p[2] <= p[3] and p[0] < p[3],
        "trapezoidal",
        "a <= b <= c <= d and a < d",
        p,
    ),
    "gaussian": lambda p: _need(p[1] > 0, "gaussian", "sigma > 0", p),
    "bell": lambda p: _need(p[0] > 0 and p[1] > 0, "bell", "a > 0 and b > 0", p),
    "sigmoid": lambda p: _need(p[0] != 0, "sigmoid", "slope != 0", p),
    "linear_increasing": lambda p: _need(
        p[0] < p[1], "linear_increasing", "a < b", p
    ),
    "linear_decreasing": lambda p: _need(
        p[0] < p[1], "linear_decreasing", "a < b", p
    ),
    # The interval type-2 building blocks (see bulanik.type2).
    "gaussian_flat_top": lambda p: _need(
        p[0] <= p[1] and p[2] > 0, "gaussian_flat_top", "m1 <= m2 and sigma > 0", p
    ),
    "gaussian_min_pair": lambda p: _need(
        p[0] <= p[1] and p[2] > 0, "gaussian_min_pair", "m1 <= m2 and sigma > 0", p
    ),
    "scaled_triangular": lambda p: _need(
        p[0] <= p[1] <= p[2] and p[0] < p[2] and 0 < p[3] <= 1,
        "scaled_triangular",
        "a <= b <= c and a < c and 0 < height <= 1",
        p,
    ),
}


def _unit_clip(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _tri(a: float, b: float, c: float, x: float) -> float:
    if x < a or x > c:
        return 0.0
    if x == b:
        return 1.0
    return _unit_clip((x - a) / (b - a) if x < b else (c - x) / (c - b))


def _gauss(mean: float, sigma: float, x: float) -> float:
    z = (x - mean) / sigma
    return _unit_clip(math.exp(-0.5 * (z * z)))


@dataclass(frozen=True)
class MembershipFunction:
    r"""A membership function $\mu: \mathbb{R} \to [0, 1]$ of a named `kind`.

    Build one with the factories ([`triangular`][bulanik.sets.triangular],
    [`gaussian`][bulanik.sets.gaussian], ...). Calling the object evaluates it:
    `mf(x)` returns the membership degree, always inside $[0, 1]$, also for huge
    or infinite arguments. The supported kinds and their parameters are:

    | kind | parameters | $\mu(x)$ |
    |---|---|---|
    | `triangular` | `a, b, c` | rises linearly from 0 at $a$ to 1 at $b$, falls to 0 at $c$ |
    | `trapezoidal` | `a, b, c, d` | rises from $a$ to 1 at $b$, 1 on $[b, c]$, falls to 0 at $d$ |
    | `gaussian` | `mean, sigma` | $\exp(-(x - \text{mean})^2 / (2\sigma^2))$ |
    | `bell` | `a, b, c` | $1 / (1 + \lvert (x - c)/a \rvert^{2b})$ |
    | `sigmoid` | `slope, center` | $1 / (1 + e^{-\text{slope}\,(x - \text{center})})$ |
    | `linear_increasing` | `a, b` | 0 for $x \le a$, 1 for $x \ge b$, linear between |
    | `linear_decreasing` | `a, b` | 1 for $x \le a$, 0 for $x \ge b$, linear between |

    Three more kinds are the building blocks of interval type-2 sets
    (see `bulanik.type2`): `gaussian_flat_top` (`m1, m2, sigma`:
    1 on $[m_1, m_2]$, else the Gaussian of the nearer mean), `gaussian_min_pair`
    (`m1, m2, sigma`: the minimum of the two Gaussians) and `scaled_triangular`
    (`a, b, c, height`: `height` times the triangle).

    Direct construction validates the parameters as the factories do.

    Args:
        kind: One of the kinds above.
        params: The numeric parameters of the kind, in the order shown.

    Raises:
        ValueError: If `kind` is unknown, the number of parameters is wrong, a
            parameter is not finite, or the parameters violate the rule of the
            kind (for example `a <= b <= c` for a triangle).
        TypeError: If `params` is not a sequence of numbers.

    Attributes:
        kind: The kind name.
        params: The validated parameters, as a tuple of floats.

    Examples:
        >>> from bulanik.sets import triangular, gaussian
        >>> mf = triangular(0, 5, 10)
        >>> mf(2.5)
        0.5
        >>> round(gaussian(5, 1)(6), 4)
        0.6065
        >>> mf.kind, mf.params
        ('triangular', (0.0, 5.0, 10.0))
    """

    kind: str
    params: tuple[float, ...]

    def __post_init__(self) -> None:
        kind = choose(self.kind, _KINDS, "membership kind", "MembershipFunction")
        if isinstance(self.params, (str, bytes)) or not hasattr(self.params, "__len__"):
            raise TypeError(
                f"{kind} params must be a sequence of numbers, "
                f"got {type(self.params).__name__} {self.params!r}"
            )
        if len(self.params) != _ARITY[kind]:
            raise ValueError(f"{kind} needs {_ARITY[kind]} parameters, got {len(self.params)}")
        params = tuple(
            finite(p, f"{kind} parameter {i + 1}") for i, p in enumerate(self.params)
        )
        _VALIDATORS[kind](params)
        object.__setattr__(self, "params", params)

    @property
    def is_monotonic(self) -> bool:
        """True for the kinds `linear_increasing`, `linear_decreasing` and `sigmoid`.

        Only monotonic functions have an [`inverse`][bulanik.sets.MembershipFunction.inverse].

        Examples:
            >>> from bulanik.sets import sigmoid, triangular
            >>> sigmoid(1, 0).is_monotonic, triangular(0, 1, 2).is_monotonic
            (True, False)
        """
        return self.kind in _MONOTONIC

    def __call__(self, x: float) -> float:
        """Membership degree of `x`.

        Args:
            x: A real number (infinite values are allowed).

        Returns:
            The degree in $[0, 1]$.

        Raises:
            ValueError: If `x` is NaN.
            TypeError: If `x` is a `bool` or a string.

        Examples:
            >>> from bulanik.sets import linear_increasing
            >>> linear_increasing(0, 10)(2.5)
            0.25
        """
        if x.__class__ is not float:  # fast path: inference calls this in loops with floats
            if isinstance(x, (bool, str)):
                raise TypeError(f"x must be a number, got {type(x).__name__} {x!r}")
            try:
                x = float(x)
            except OverflowError:  # an int beyond the float range
                x = math.inf if x > 0 else -math.inf
        if math.isnan(x):
            raise ValueError("x must be a number, got nan")
        k, p = self.kind, self.params
        if k == "triangular":
            return _tri(p[0], p[1], p[2], x)
        if k == "trapezoidal":
            a, b, c, d = p
            if x < a or x > d:
                return 0.0
            if b <= x <= c:
                return 1.0
            return _unit_clip((x - a) / (b - a) if x < b else (d - x) / (d - c))
        if k == "gaussian":
            return _gauss(p[0], p[1], x)
        if k == "gaussian_flat_top":  # 1 on [m1, m2], else the gaussian of the nearer mean
            m1, m2, sigma = p
            return _gauss(m1, sigma, x) if x < m1 else _gauss(m2, sigma, x) if x > m2 else 1.0
        if k == "gaussian_min_pair":  # min(g_m1, g_m2)
            m1, m2, sigma = p
            return min(_gauss(m1, sigma, x), _gauss(m2, sigma, x))
        if k == "scaled_triangular":  # height * tri(a, b, c)
            return p[3] * _tri(p[0], p[1], p[2], x)
        if k == "bell":
            a, b, c = p
            try:
                t = abs((x - c) / a) ** (2.0 * b)
            except OverflowError:
                return 0.0
            return _unit_clip(1.0 / (1.0 + t))
        if k == "sigmoid":
            slope, center = p
            t = slope * (x - center)
            if t >= 0.0:
                return _unit_clip(1.0 / (1.0 + math.exp(-t)))
            e = math.exp(t)
            return _unit_clip(e / (1.0 + e))
        a, b = p
        if k == "linear_increasing":
            return 0.0 if x <= a else 1.0 if x >= b else _unit_clip((x - a) / (b - a))
        return 1.0 if x <= a else 0.0 if x >= b else _unit_clip((b - x) / (b - a))

    def inverse(self, alpha: float) -> float:
        r"""The $x$ with $\mu(x) = \alpha$ (monotonic kinds only).

        For `linear_increasing` this is $a (1 - \alpha) + b\alpha$ and for
        `linear_decreasing` $b (1 - \alpha) + a\alpha$, exact at the ends. For
        `sigmoid` it is $\text{center} - \ln(1/\alpha - 1)/\text{slope}$ with
        $\alpha$ clamped into $[\varepsilon, 1 - \varepsilon]$, where
        $\varepsilon$ is [`SIGMOID_EPS`][bulanik.sets.SIGMOID_EPS].

        Args:
            alpha: Membership level in $[0, 1]$.

        Returns:
            The point `x`.

        Raises:
            ValueError: If the function is not monotonic, or `alpha` is outside
                $[0, 1]$ or not finite.

        Examples:
            >>> from bulanik.sets import linear_increasing, sigmoid, triangular
            >>> linear_increasing(0, 10).inverse(0.25)
            2.5
            >>> sigmoid(1, 0).inverse(0.5)
            0.0
            >>> triangular(0, 1, 2).inverse(0.5)
            Traceback (most recent call last):
            ...
            ValueError: inverse needs a monotonic membership function; triangular is not monotonic
        """
        if not self.is_monotonic:
            raise ValueError(
                f"inverse needs a monotonic membership function; "
                f"{self.kind} is not monotonic"
            )
        alpha = finite(alpha, "alpha")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        p = self.params
        if self.kind == "linear_increasing":
            a, b = p
            # a(1-alpha) + b alpha: b - a may overflow, this cannot; rounding can leave
            # [a, b] by an ulp, so clamp
            return b if alpha == 1.0 else min(max(a * (1.0 - alpha) + b * alpha, a), b)
        if self.kind == "linear_decreasing":
            a, b = p
            return a if alpha == 1.0 else min(max(b * (1.0 - alpha) + a * alpha, a), b)
        slope, center = p
        alpha = min(max(alpha, SIGMOID_EPS), 1.0 - SIGMOID_EPS)
        return center - math.log(1.0 / alpha - 1.0) / slope


def triangular(a: float, b: float, c: float) -> MembershipFunction:
    """Triangle with feet `a`, `c` and peak `b` (`a <= b <= c`, `a < c`).

    Args:
        a: Left foot (membership 0).
        b: Peak (membership 1).
        c: Right foot (membership 0).

    Returns:
        A `MembershipFunction` of kind `"triangular"`.

    Raises:
        ValueError: If the parameters are not finite or violate `a <= b <= c` and `a < c`.

    Examples:
        >>> from bulanik.sets import triangular
        >>> tri = triangular(0, 5, 10)
        >>> tri(5), tri(7.5), tri(11)
        (1.0, 0.5, 0.0)
        >>> triangular(3, 2, 1)
        Traceback (most recent call last):
        ...
        ValueError: triangular needs a <= b <= c and a < c, got (3.0, 2.0, 1.0)
    """
    return MembershipFunction("triangular", (a, b, c))


def trapezoidal(a: float, b: float, c: float, d: float) -> MembershipFunction:
    """Trapezoid with plateau `[b, c]` (`a <= b <= c <= d`, `a < d`).

    Args:
        a: Left foot (membership 0).
        b: Start of the plateau.
        c: End of the plateau.
        d: Right foot (membership 0).

    Returns:
        A `MembershipFunction` of kind `"trapezoidal"`.

    Raises:
        ValueError: If the parameters are not finite or violate `a <= b <= c <= d` and `a < d`.

    Examples:
        >>> from bulanik.sets import trapezoidal
        >>> trapezoidal(0, 1, 2, 3)(0.5), trapezoidal(0, 1, 2, 3)(1.5)
        (0.5, 1.0)
    """
    return MembershipFunction("trapezoidal", (a, b, c, d))


def gaussian(mean: float, sigma: float) -> MembershipFunction:
    r"""Gaussian $\exp\!\left(-\frac{(x - \text{mean})^2}{2\sigma^2}\right)$ with $\sigma > 0$.

    Args:
        mean: Centre of the bell (membership 1).
        sigma: Standard deviation, strictly positive.

    Returns:
        A `MembershipFunction` of kind `"gaussian"`.

    Raises:
        ValueError: If a parameter is not finite or `sigma <= 0`.

    Examples:
        >>> from bulanik.sets import gaussian
        >>> g = gaussian(5, 1)
        >>> g(5), round(g(6), 4)
        (1.0, 0.6065)
    """
    return MembershipFunction("gaussian", (mean, sigma))


def bell(a: float, b: float, c: float) -> MembershipFunction:
    r"""Generalized bell $\dfrac{1}{1 + \lvert (x - c)/a \rvert^{2b}}$ with $a > 0$, $b > 0$.

    Args:
        a: Width, strictly positive.
        b: Slope of the shoulders, strictly positive.
        c: Centre (membership 1).

    Returns:
        A `MembershipFunction` of kind `"bell"`.

    Raises:
        ValueError: If a parameter is not finite or `a <= 0` or `b <= 0`.

    Examples:
        >>> from bulanik.sets import bell
        >>> bell(2, 4, 6)(6), round(bell(2, 4, 6)(7), 4)
        (1.0, 0.9961)
    """
    return MembershipFunction("bell", (a, b, c))


def sigmoid(slope: float, center: float) -> MembershipFunction:
    r"""Sigmoid $\dfrac{1}{1 + e^{-\text{slope}\,(x - \text{center})}}$ with $\text{slope} \ne 0$.

    Increasing for a positive slope and decreasing for a negative one;
    membership 0.5 at `center`. It is monotonic, so it has an
    [`inverse`][bulanik.sets.MembershipFunction.inverse].

    Args:
        slope: Steepness and direction, non-zero.
        center: The point where the membership is 0.5.

    Returns:
        A `MembershipFunction` of kind `"sigmoid"`.

    Raises:
        ValueError: If a parameter is not finite or `slope == 0`.

    Examples:
        >>> from bulanik.sets import sigmoid
        >>> sigmoid(1, 0)(0)
        0.5
    """
    return MembershipFunction("sigmoid", (slope, center))


def linear_increasing(a: float, b: float) -> MembershipFunction:
    """Ramp up: 0 for `x <= a`, 1 for `x >= b`, linear in between (`a < b`).

    Args:
        a: Last point with membership 0.
        b: First point with membership 1.

    Returns:
        A `MembershipFunction` of kind `"linear_increasing"`.

    Raises:
        ValueError: If a parameter is not finite or `a >= b`.

    Examples:
        >>> from bulanik.sets import linear_increasing
        >>> [linear_increasing(0, 10)(x) for x in (-1, 2.5, 12)]
        [0.0, 0.25, 1.0]
    """
    return MembershipFunction("linear_increasing", (a, b))


def linear_decreasing(a: float, b: float) -> MembershipFunction:
    """Ramp down: 1 for `x <= a`, 0 for `x >= b`, linear in between (`a < b`).

    Args:
        a: Last point with membership 1.
        b: First point with membership 0.

    Returns:
        A `MembershipFunction` of kind `"linear_decreasing"`.

    Raises:
        ValueError: If a parameter is not finite or `a >= b`.

    Examples:
        >>> from bulanik.sets import linear_decreasing
        >>> [linear_decreasing(0, 10)(x) for x in (-1, 2.5, 12)]
        [1.0, 0.75, 0.0]
    """
    return MembershipFunction("linear_decreasing", (a, b))


class LinguisticVariable:
    """A named variable on a universe ``(lo, hi)`` with named fuzzy terms.

    Terms are looked up with `variable["term"]`, which returns the
    [`MembershipFunction`][bulanik.sets.MembershipFunction]; the term order is
    the insertion order of `terms`.

    Args:
        name: The variable name, used in error messages and rule text.
        universe: The pair `(lo, hi)` of the domain, with `lo < hi`.
        terms: Mapping from term name to membership function (at least one).

    Raises:
        TypeError: If `universe` is not a pair, or a term is not a
            `MembershipFunction`.
        ValueError: If `lo >= hi`, an end is not finite, or `terms` is empty.

    Attributes:
        name: The variable name.
        universe: The domain as a tuple of floats `(lo, hi)`.

    Examples:
        >>> from bulanik.sets import LinguisticVariable, linear_decreasing, triangular, linear_increasing
        >>> temp = LinguisticVariable("temp", (0, 40), {
        ...     "cold": linear_decreasing(10, 20),
        ...     "warm": triangular(10, 20, 30),
        ...     "hot": linear_increasing(20, 30),
        ... })
        >>> temp
        LinguisticVariable('temp', (0.0, 40.0), terms=('cold', 'warm', 'hot'))
        >>> temp.fuzzify(15)
        {'cold': 0.5, 'warm': 0.5, 'hot': 0.0}
    """

    def __init__(
        self,
        name: str,
        universe: tuple[float, float],
        terms: Mapping[str, MembershipFunction],
    ) -> None:
        try:
            lo_raw, hi_raw = universe
        except (TypeError, ValueError):
            raise TypeError(
                f"universe of '{name}' must be a (lo, hi) pair, got {universe!r}"
            ) from None
        lo, hi = finite(lo_raw, "universe lo"), finite(hi_raw, "universe hi")
        if not lo < hi:
            raise ValueError(f"universe of '{name}' needs lo < hi, got ({lo}, {hi})")
        if not terms:
            raise ValueError(f"variable '{name}' needs at least one term")
        for term, mf in terms.items():
            if not isinstance(mf, MembershipFunction):
                raise TypeError(
                    f"term '{term}' of '{name}' must be a MembershipFunction, "
                    f"got {type(mf).__name__}"
                )
        self.name = name
        self.universe = (lo, hi)
        self._terms = dict(terms)

    @property
    def term_names(self) -> tuple[str, ...]:
        """The term names, in insertion order.

        Examples:
            >>> from bulanik.sets import LinguisticVariable, triangular
            >>> LinguisticVariable("x", (0, 1), {"lo": triangular(0, 0, 1)}).term_names
            ('lo',)
        """
        return tuple(self._terms)

    def __getitem__(self, term: str) -> MembershipFunction:
        """The membership function of `term`.

        Args:
            term: A term name.

        Returns:
            The `MembershipFunction` of the term.

        Raises:
            KeyError: If the term does not exist (the message lists the valid terms).
        """
        try:
            return self._terms[term]
        except KeyError:
            raise KeyError(
                f"unknown term '{term}' of '{self.name}'; valid: {', '.join(self._terms)}"
            ) from None

    def fuzzify(self, x: float) -> dict[str, float]:
        """Membership of `x` in every term.

        Args:
            x: A crisp input inside the universe.

        Returns:
            A dict from term name to membership degree, in term order.

        Raises:
            ValueError: If `x` is not finite or lies outside the universe.

        Examples:
            >>> from bulanik.sets import LinguisticVariable, triangular
            >>> v = LinguisticVariable("service", (0, 10), {"good": triangular(5, 10, 10)})
            >>> v.fuzzify(7.5)
            {'good': 0.5}
            >>> v.fuzzify(11)
            Traceback (most recent call last):
            ...
            ValueError: 'service' input 11 is outside its universe [0, 10]
        """
        x = finite(x, f"input of '{self.name}'")
        lo, hi = self.universe
        if not lo <= x <= hi:
            raise ValueError(
                f"'{self.name}' input {_num(x)} is outside its universe "
                f"[{_num(lo)}, {_num(hi)}]"
            )
        return {t: mf(x) for t, mf in self._terms.items()}

    def __repr__(self) -> str:
        return f"LinguisticVariable({self.name!r}, {self.universe}, terms={self.term_names})"


_T_NORMS = ("min", "prod")
_T_CONORMS = ("max", "probor")


def t_norm(a: float, b: float, kind: str = "min") -> float:
    r"""Fuzzy AND (t-norm): `"min"` $\min(a, b)$ or `"prod"` $a\,b$.

    Args:
        a: Membership degree in $[0, 1]$.
        b: Membership degree in $[0, 1]$.
        kind: `"min"` (default) or `"prod"`.

    Returns:
        The combined degree.

    Raises:
        ValueError: If `kind` is unknown or `a` or `b` is outside $[0, 1]$.

    Examples:
        >>> from bulanik.sets import t_norm
        >>> t_norm(0.3, 0.6), round(t_norm(0.3, 0.6, "prod"), 4)
        (0.3, 0.18)
    """
    kind = choose(kind, _T_NORMS, "t-norm", "t_norm")
    a, b = unit(a, "t_norm a"), unit(b, "t_norm b")
    return min(a, b) if kind == "min" else a * b


def t_conorm(a: float, b: float, kind: str = "max") -> float:
    r"""Fuzzy OR (t-conorm): `"max"` $\max(a, b)$ or `"probor"` $a + b - ab$.

    Args:
        a: Membership degree in $[0, 1]$.
        b: Membership degree in $[0, 1]$.
        kind: `"max"` (default) or `"probor"`.

    Returns:
        The combined degree.

    Raises:
        ValueError: If `kind` is unknown or `a` or `b` is outside $[0, 1]$.

    Examples:
        >>> from bulanik.sets import t_conorm
        >>> t_conorm(0.3, 0.6), round(t_conorm(0.3, 0.6, "probor"), 4)
        (0.6, 0.72)
    """
    kind = choose(kind, _T_CONORMS, "t-conorm", "t_conorm")
    a, b = unit(a, "t_conorm a"), unit(b, "t_conorm b")
    return max(a, b) if kind == "max" else a + b - a * b
