r"""q-rung orthopair fuzzy numbers: QROFN and its fixed-q subclasses IFN, PFN, FFN.

A q-rung orthopair fuzzy number (Yager 2017) is a pair $(\mu, \nu)$ of a
membership and a non-membership degree with $\mu^q + \nu^q \le 1$. The rung
$q = 1$ gives intuitionistic numbers ([`IFN`][bulanik.numbers.IFN]), $q = 2$
Pythagorean ones ([`PFN`][bulanik.numbers.PFN]) and $q = 3$ Fermatean ones
([`FFN`][bulanik.numbers.FFN]); [`QROFN`][bulanik.numbers.QROFN] takes any
integer $q \ge 1$.

Every operation is the intuitionistic one applied to the q-th powers
$(a, b) = (\mu^q, \nu^q)$, followed by the q-th root of each result part. Two
operator sets exist, `"algebraic"` and `"einstein"`.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any

from .._validate import TOL, choose, finite, unit
from ._base import FuzzyNumber


def _alg_add(a1: float, b1: float, a2: float, b2: float) -> tuple[float, float]:
    return a1 + a2 - a1 * a2, b1 * b2


def _alg_mul(a1: float, b1: float, a2: float, b2: float) -> tuple[float, float]:
    return a1 * a2, b1 + b2 - b1 * b2


def _ein_add(a1: float, b1: float, a2: float, b2: float) -> tuple[float, float]:
    return (a1 + a2) / (1.0 + a1 * a2), b1 * b2 / (1.0 + (1.0 - b1) * (1.0 - b2))


def _ein_mul(a1: float, b1: float, a2: float, b2: float) -> tuple[float, float]:
    return a1 * a2 / (1.0 + (1.0 - a1) * (1.0 - a2)), (b1 + b2) / (1.0 + b1 * b2)


def _log_one_minus(x: float, cx: float) -> float:
    """log(1 - x), where ``cx`` is an accurate value of 1 - x (cx > 0)."""
    return math.log1p(-x) if x < 0.5 else math.log(cx)


def _one_minus_pow(x: float, cx: float, k: float) -> float:
    """1 - (1 - x)^k without cancellation; ``cx`` is an accurate 1 - x."""
    if cx <= 0.0:
        return 1.0 if k > 0 else 0.0
    return -math.expm1(k * _log_one_minus(x, cx))


def _pow(x: float, k: float) -> float:
    return x**k  # 0.0 ** 0 == 1.0, which makes zero weights work


def _tanh_k_atanh(x: float, cx: float, k: float) -> float:
    """((1+x)^k - (1-x)^k) / ((1+x)^k + (1-x)^k) = tanh(k atanh x), stably."""
    if x <= 0.0 or k == 0:
        return 0.0
    if cx <= 0.0:
        return 1.0
    return math.tanh(0.5 * k * (math.log1p(x) - _log_one_minus(x, cx)))


def _two_pow_over(x: float, cx: float, k: float) -> float:
    """2 x^k / ((2 - x)^k + x^k) = 2 / (1 + e^t), t = k log((2 - x) / x); 2 - x = 1 + cx.

    Evaluated through t so a large k cannot overflow (1 + cx)^k.
    """
    if k == 0:
        return 1.0
    if x <= 0.0:
        return 0.0
    t = k * (math.log1p(cx) - math.log(x))
    if t > 0.0:
        e = math.exp(-t)
        return 2.0 * e / (1.0 + e)
    return 2.0 / (1.0 + math.exp(t))


def _alg_scale(a, ca, b, cb, k):
    return _one_minus_pow(a, ca, k), _pow(b, k)


def _alg_power(a, ca, b, cb, k):
    return _pow(a, k), _one_minus_pow(b, cb, k)


def _ein_scale(a, ca, b, cb, k):
    return _tanh_k_atanh(a, ca, k), _two_pow_over(b, cb, k)


def _ein_power(a, ca, b, cb, k):
    return _two_pow_over(a, ca, k), _tanh_k_atanh(b, cb, k)


_BINARY = {
    ("algebraic", "add"): _alg_add,
    ("algebraic", "mul"): _alg_mul,
    ("einstein", "add"): _ein_add,
    ("einstein", "mul"): _ein_mul,
}
_UNARY = {
    ("algebraic", "scale"): _alg_scale,
    ("algebraic", "power"): _alg_power,
    ("einstein", "scale"): _ein_scale,
    ("einstein", "power"): _ein_power,
}

_EPS = sys.float_info.epsilon
# The constructor projects an input only when it is more than _PROJECT_ULPS past
# the boundary, so float noise of a deliberate boundary value stays bit-identical.
_PROJECT_ULPS = 4
# Every input to an operation satisfies a + b <= 1 + d. The constructor leaves at
# most _PROJECT_ULPS. Beyond that, a boundary value built as (a^(1/q), (1-a)^(1/q))
# has each part's q-th power wrong by about q * (rounding of the root and of the
# exponent 1/q) + 1 relative ulps; two parts, each at most 1, and a factor 2 for
# pow() being faithful rather than correctly rounded give 2 * (q + 2) ulps. So
# d <= (_PROJECT_ULPS + 2 * (q + 2)) ulps, which grows with q (measured: ~q/4).


def _overshoot_ulps(q: int) -> int:
    """Input noise bound d (in ulps) for a point of rung ``q`` that the constructor accepted."""
    return _PROJECT_ULPS + 2 * (q + 2)


# Largest overshoot of a + b in an operation result treated as rounding rather
# than a violation. Derivation: a result sum is S(a, b) = a' + b' with S = 1 on
# the boundary, and |dS/d(input sum)| <= k for scale/power by k (the only
# amplified term is c^k against b^k, k * c^k <= k) and <= 1 for add/mul. An input
# d past the boundary therefore yields an overshoot of at most k * d plus a few
# ulps of expm1/log1p/pow rounding, which is the slack passed by _unary:
# _overshoot_ulps(q) * eps * (1 + k). Below that the result is put back on the
# boundary; the floor is _ROUNDING. test_rounding_bound_pinned measures it.
# Genuinely invalid results are never rescaled.
_ROUNDING = TOL


@dataclass(frozen=True)
class QROFN(FuzzyNumber):
    r"""q-rung orthopair fuzzy number $(\mu, \nu)$ with $\mu^q + \nu^q \le 1$.

    Every operation is the intuitionistic one applied to $(a, b) = (\mu^q, \nu^q)$,
    followed by the q-th root of each part. For two numbers with powers
    $(a_1, b_1)$ and $(a_2, b_2)$ the result powers are:

    | operation | `"algebraic"` | `"einstein"` |
    |---|---|---|
    | `add` | $(a_1 + a_2 - a_1 a_2,\; b_1 b_2)$ | $\left(\frac{a_1 + a_2}{1 + a_1 a_2},\; \frac{b_1 b_2}{1 + (1 - b_1)(1 - b_2)}\right)$ |
    | `mul` | $(a_1 a_2,\; b_1 + b_2 - b_1 b_2)$ | $\left(\frac{a_1 a_2}{1 + (1 - a_1)(1 - a_2)},\; \frac{b_1 + b_2}{1 + b_1 b_2}\right)$ |
    | `scale` by $k$ | $(1 - (1 - a)^k,\; b^k)$ | $\left(\frac{(1+a)^k - (1-a)^k}{(1+a)^k + (1-a)^k},\; \frac{2 b^k}{(2-b)^k + b^k}\right)$ |
    | `power` by $k$ | $(a^k,\; 1 - (1 - b)^k)$ | $\left(\frac{2 a^k}{(2-a)^k + a^k},\; \frac{(1+b)^k - (1-b)^k}{(1+b)^k + (1-b)^k}\right)$ |

    Scores: `"difference"` (default) $= \mu^q - \nu^q$ and `"normalized"`
    $= (1 + \mu^q - \nu^q)/2$. Accuracy: `"sum"` $= \mu^q + \nu^q$. Distances
    (default `"hamming"`) work on $(a, b, \pi)$ with the hesitancy power
    $\pi = 1 - a - b$: Hamming $= (|\Delta a| + |\Delta b| + |\Delta \pi|)/2$ and
    Euclidean $= \sqrt{(\Delta a^2 + \Delta b^2 + \Delta \pi^2)/2}$.

    Inputs that exceed the constraint by no more than $10^{-12}$ are accepted
    as rounding and put back on the boundary $\mu^q + \nu^q = 1$; results that
    violate it by more than rounding raise `ValueError`.

    Args:
        mu: Membership degree in $[0, 1]$.
        nu: Non-membership degree in $[0, 1]$.
        q: The rung, an integer $\ge 1$.

    Raises:
        ValueError: If `q` is not an integer $\ge 1$, a degree is outside
            $[0, 1]$ or not finite, or $\mu^q + \nu^q > 1$ (beyond rounding).
        TypeError: If a degree is a `bool` or a string.

    Attributes:
        mu: Membership degree.
        nu: Non-membership degree.
        q: The rung.

    References:
        Yager (2017), IEEE Transactions on Fuzzy Systems 25(5): q-rung orthopair
        fuzzy sets. Einstein operations: Wang and Liu (2011), International Journal
        of Intelligent Systems 26(11), 1049-1075.

    Examples:
        >>> from bulanik import QROFN
        >>> QROFN(0.9, 0.5, 3)
        QROFN(mu=0.9, nu=0.5, q=3)
        >>> QROFN(0.9, 0.8, 3)
        Traceback (most recent call last):
        ...
        ValueError: need mu^3 + nu^3 <= 1, got mu=0.9, nu=0.8 (sum 1.241)
    """

    mu: float
    nu: float
    q: int

    OPS = ("algebraic", "einstein")
    SCORES = ("difference", "normalized")
    DEFAULT_SCORE = "difference"
    ACCURACIES = ("sum",)
    DEFAULT_ACCURACY = "sum"
    DISTANCES = ("hamming", "euclidean")
    DEFAULT_DISTANCE = "hamming"

    def __post_init__(self) -> None:
        q = self.q
        if isinstance(q, bool) or not isinstance(q, int) or q < 1:
            raise ValueError(f"q must be an integer >= 1, got {q!r}")
        mu = unit(self.mu, "mu")
        nu = unit(self.nu, "nu")
        total = mu**q + nu**q
        if total > 1.0 + TOL:
            raise ValueError(
                f"need mu^{q} + nu^{q} <= 1, got mu={mu}, nu={nu} (sum {total})"
            )
        if total > 1.0 + _PROJECT_ULPS * _EPS:
            # An input up to TOL past the boundary is accepted as rounding. Put it
            # back on mu^q + nu^q = 1 now: otherwise an amplifying operation
            # (power, scale by k > 1, ...) magnifies the excess past the result
            # check and raises on a value the constructor accepted.
            mu = (mu**q / total) ** (1.0 / q)
            nu = (nu**q / total) ** (1.0 / q)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "nu", nu)

    # -- construction helpers ------------------------------------------------
    def _new(self, a: float, b: float, slack: float = 0.0) -> Any:
        """Build a result of the same class from the q-th powers ``(a, b)``.

        ``slack`` widens the accepted rounding overshoot beyond ``_ROUNDING``.
        """
        a, b = unit(a, "mu^q"), unit(b, "nu^q")
        total = a + b
        if total > 1.0 + max(_ROUNDING, slack):
            raise ValueError(
                f"result violates mu^{self.q} + nu^{self.q} <= 1 (sum {a + b})"
            )
        if total > 1.0:
            # fractional exponents amplify input rounding (1-b)^k; this is
            # rounding only, so put the result back on the boundary a + b = 1
            a, b = a / total, b / total
        mu, nu = a ** (1.0 / self.q), b ** (1.0 / self.q)
        return self._rebuild(mu, nu)

    def _rebuild(self, mu: float, nu: float) -> Any:
        return type(self)(mu, nu, self.q)

    def _powers(self) -> tuple[float, float]:
        return self.mu**self.q, self.nu**self.q

    # -- derived quantities --------------------------------------------------
    def complement(self) -> Any:
        """Complement: membership and non-membership swapped.

        Returns:
            The number $(\\nu, \\mu)$ of the same type.

        Examples:
            >>> from bulanik import IFN
            >>> IFN(0.6, 0.3).complement()
            IFN(mu=0.3, nu=0.6)
        """
        return self._rebuild(self.nu, self.mu)

    @property
    def hesitancy(self) -> float:
        r"""Hesitancy degree $\pi = (1 - \mu^q - \nu^q)^{1/q}$.

        Examples:
            >>> from bulanik import PFN
            >>> round(PFN(0.7, 0.5).hesitancy, 4)
            0.5099
        """
        a, b = self._powers()
        return max(0.0, 1.0 - a - b) ** (1.0 / self.q)

    # -- arithmetic ----------------------------------------------------------
    def _binary(self, name: str, other: Any, ops: str) -> Any:
        ops = self._check_ops(ops)
        self.same_kind(other)
        a1, b1 = self._powers()
        a2, b2 = other._powers()
        return self._new(*_BINARY[(ops, name)](a1, b1, a2, b2))

    def _unary(self, name: str, k: float, ops: str) -> Any:
        ops = self._check_ops(ops)
        k = finite(k, "k")
        if k < 0:
            raise ValueError(f"{name} needs k >= 0, got {k}")
        a, b = self._powers()
        ca, cb = 1.0 - a, 1.0 - b
        # A point may sit up to TOL past the boundary; there the larger part's
        # complement must not fall below the smaller part. Clamp only: interior
        # points keep their exact complements (exact for the larger part >= 0.5).
        if a >= b:
            ca = max(ca, b)
        else:
            cb = max(cb, a)
        # The result may overshoot a + b = 1 by k times the input's own noise
        # (see _overshoot_ulps), which is rounding, not a violation.
        slack = _overshoot_ulps(self.q) * _EPS * (1.0 + k)
        return self._new(*_UNARY[(ops, name)](a, ca, b, cb, k), slack=slack)

    def add(self, other: Any, ops: str = "algebraic") -> Any:
        """Sum of two numbers of the same type and rung (see the table of operators above).

        `A + B` calls this method with the algebraic operators.

        Args:
            other: A number of the same class and the same `q`.
            ops: `"algebraic"` (default) or `"einstein"`.

        Returns:
            The sum, of the same class.

        Raises:
            TypeError: If `other` differs in class or rung.
            ValueError: If `ops` is unknown.

        References:
            Wang and Liu (2011), International Journal of Intelligent Systems 26 (Einstein operations, intuitionistic case).

        Examples:
            >>> from bulanik import IFN
            >>> a, b = IFN(0.6, 0.3), IFN(0.5, 0.4)
            >>> a + b
            IFN(mu=0.8, nu=0.12)
            >>> e = a.add(b, ops="einstein")
            >>> round(e.mu, 4), round(e.nu, 4)
            (0.8462, 0.0845)
        """
        return self._binary("add", other, ops)

    def mul(self, other: Any, ops: str = "algebraic") -> Any:
        """Product of two numbers of the same type and rung (see the table of operators above).

        `A * B` calls this method with the algebraic operators.

        Args:
            other: A number of the same class and the same `q`.
            ops: `"algebraic"` (default) or `"einstein"`.

        Returns:
            The product, of the same class.

        Raises:
            TypeError: If `other` differs in class or rung.
            ValueError: If `ops` is unknown.

        Examples:
            >>> from bulanik import IFN
            >>> IFN(0.6, 0.3) * IFN(0.5, 0.4)
            IFN(mu=0.3, nu=0.58)
        """
        return self._binary("mul", other, ops)

    def scale(self, k: float, ops: str = "algebraic") -> Any:
        """Scalar multiple ``k * A`` (see the table of operators above).

        `k * A` and `A * k` call this method with the algebraic operators.

        Args:
            k: A finite scalar $\\ge 0$.
            ops: `"algebraic"` (default) or `"einstein"`.

        Returns:
            The scaled number, of the same class.

        Raises:
            ValueError: If `k` is negative or not finite, `ops` is unknown, or
                the result violates the constraint beyond rounding.

        Examples:
            >>> from bulanik import IFN
            >>> IFN(0.6, 0.3).scale(2)
            IFN(mu=0.84, nu=0.09)
            >>> IFN(0.6, 0.3).scale(-1)
            Traceback (most recent call last):
            ...
            ValueError: scale needs k >= 0, got -1.0
        """
        return self._unary("scale", k, ops)

    def power(self, k: float, ops: str = "algebraic") -> Any:
        """Power ``A ** k`` (see the table of operators above).

        `A ** k` calls this method with the algebraic operators.

        Args:
            k: A finite exponent $\\ge 0$.
            ops: `"algebraic"` (default) or `"einstein"`.

        Returns:
            The power, of the same class.

        Raises:
            ValueError: If `k` is negative or not finite, `ops` is unknown, or
                the result violates the constraint beyond rounding.

        Examples:
            >>> from bulanik import IFN
            >>> IFN(0.6, 0.3) ** 2
            IFN(mu=0.36, nu=0.51)
        """
        return self._unary("power", k, ops)

    # -- scores, accuracy, distance -------------------------------------------
    def score(self, method: str | None = None) -> float:
        r"""Score for ranking: $\mu^q - \nu^q$ (`"difference"`, default) or $(1 + \mu^q - \nu^q)/2$ (`"normalized"`).

        Args:
            method: `"difference"` or `"normalized"`; `None` means `"difference"`.

        Returns:
            The score.

        Raises:
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import IFN
            >>> round(IFN(0.6, 0.3).score(), 4)
            0.3
            >>> round(IFN(0.6, 0.3).score("normalized"), 4)
            0.65
        """
        method = choose(
            method, self.SCORES, "score", type(self).__name__, self.DEFAULT_SCORE
        )
        a, b = self._powers()
        if method == "normalized":
            return (1.0 + a - b) / 2.0
        return a - b

    def accuracy(self, method: str | None = None) -> float:
        r"""Accuracy $\mu^q + \nu^q$ (`"sum"`, the only method); the complement of the hesitancy power.

        Args:
            method: `"sum"` or `None`.

        Returns:
            The accuracy in $[0, 1]$.

        Raises:
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import IFN
            >>> round(IFN(0.6, 0.3).accuracy(), 4)
            0.9
        """
        choose(
            method, self.ACCURACIES, "accuracy", type(self).__name__, self.DEFAULT_ACCURACY
        )
        a, b = self._powers()
        return a + b

    def distance(self, other: Any, method: str | None = None) -> float:
        """Normalized Hamming (default) or Euclidean distance on ``(mu^q, nu^q, pi^q)``.

        Args:
            other: A number of the same class and the same `q`.
            method: `"hamming"` or `"euclidean"`; `None` means `"hamming"`.

        Returns:
            The distance, in $[0, 1]$.

        Raises:
            TypeError: If `other` differs in class or rung.
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import IFN
            >>> round(IFN(0.6, 0.2).distance(IFN(0.3, 0.3)), 4)
            0.3
            >>> round(IFN(0.6, 0.2).distance(IFN(0.3, 0.3), "euclidean"), 4)
            0.2646
        """
        method = choose(
            method, self.DISTANCES, "distance", type(self).__name__, self.DEFAULT_DISTANCE
        )
        self.same_kind(other)
        a1, b1 = self._powers()
        a2, b2 = other._powers()
        da, db = a1 - a2, b1 - b2
        dp = (1.0 - a1 - b1) - (1.0 - a2 - b2)
        if method == "euclidean":
            return math.sqrt((da * da + db * db + dp * dp) / 2.0)
        return (abs(da) + abs(db) + abs(dp)) / 2.0


class _Fixed(QROFN):
    """A QROFN with a fixed q; constructed as ``Cls(mu, nu)`` (private base of IFN, PFN, FFN)."""

    Q = 1

    def __init__(self, mu: float, nu: float, q: int | None = None) -> None:
        # ``q`` exists so dataclasses.replace() works; it must match the class.
        if q is not None and (isinstance(q, bool) or q != self.Q):
            raise ValueError(f"{type(self).__name__} has q = {self.Q}, got q={q!r}")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "q", self.Q)
        self.__post_init__()

    def _rebuild(self, mu: float, nu: float) -> Any:
        return type(self)(mu, nu)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu!r}, nu={self.nu!r})"


class IFN(_Fixed):
    """Intuitionistic fuzzy number $(\\mu, \\nu)$ with $\\mu + \\nu \\le 1$ (q = 1, Atanassov 1986).

    A [`QROFN`][bulanik.numbers.QROFN] with `q = 1`, constructed as `IFN(mu, nu)`.
    All methods are those of `QROFN`.

    Args:
        mu: Membership degree in $[0, 1]$.
        nu: Non-membership degree in $[0, 1]$.

    Raises:
        ValueError: If a degree is outside $[0, 1]$ or $\\mu + \\nu > 1$.

    References:
        Atanassov (1986), Fuzzy Sets and Systems 20(1): intuitionistic fuzzy sets.

    Examples:
        >>> from bulanik import IFN
        >>> IFN(0.6, 0.3)
        IFN(mu=0.6, nu=0.3)
        >>> IFN(0.8, 0.5)
        Traceback (most recent call last):
        ...
        ValueError: need mu^1 + nu^1 <= 1, got mu=0.8, nu=0.5 (sum 1.3)
    """

    Q = 1


class PFN(_Fixed):
    """Pythagorean fuzzy number $(\\mu, \\nu)$ with $\\mu^2 + \\nu^2 \\le 1$ (q = 2, Yager 2013).

    A [`QROFN`][bulanik.numbers.QROFN] with `q = 2`, constructed as `PFN(mu, nu)`.

    Args:
        mu: Membership degree in $[0, 1]$.
        nu: Non-membership degree in $[0, 1]$.

    Raises:
        ValueError: If a degree is outside $[0, 1]$ or $\\mu^2 + \\nu^2 > 1$.

    References:
        Yager (2013), Pythagorean fuzzy subsets, Joint IFSA World Congress and NAFIPS
        Annual Meeting, 57-61.

    Examples:
        >>> from bulanik import PFN
        >>> PFN(0.8, 0.5)
        PFN(mu=0.8, nu=0.5)
        >>> PFN(0.8, 0.5).q
        2
    """

    Q = 2


class FFN(_Fixed):
    """Fermatean fuzzy number $(\\mu, \\nu)$ with $\\mu^3 + \\nu^3 \\le 1$ (q = 3, Senapati and Yager 2020).

    A [`QROFN`][bulanik.numbers.QROFN] with `q = 3`, constructed as `FFN(mu, nu)`.

    Args:
        mu: Membership degree in $[0, 1]$.
        nu: Non-membership degree in $[0, 1]$.

    Raises:
        ValueError: If a degree is outside $[0, 1]$ or $\\mu^3 + \\nu^3 > 1$.

    References:
        Senapati and Yager (2020), Fermatean fuzzy sets, Journal of Ambient
        Intelligence and Humanized Computing 11(2), 663-674.

    Examples:
        >>> from bulanik import FFN
        >>> FFN(0.8, 0.6)
        FFN(mu=0.8, nu=0.6)
    """

    Q = 3
