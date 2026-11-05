r"""Three-part fuzzy numbers: SphericalFN, PictureFN and NeutrosophicFN.

Each number carries three parts in $[0, 1]$ with a type-specific constraint:

- [`SphericalFN`][bulanik.numbers.SphericalFN] $(\mu, \nu, \pi)$ with
  $\mu^2 + \nu^2 + \pi^2 \le 1$ (membership, non-membership, hesitancy).
- [`PictureFN`][bulanik.numbers.PictureFN] $(\mu, \eta, \nu)$ with
  $\mu + \eta + \nu \le 1$ (positive, neutral, negative).
- [`NeutrosophicFN`][bulanik.numbers.NeutrosophicFN] $(t, i, f)$ with no joint
  constraint (truth, indeterminacy, falsity).

All three have the single operator set `"algebraic"`, the distances `"hamming"`
(default) and `"euclidean"`, and the accuracy `"sum"` (neutrosophic numbers
define `"difference"` and `"certainty"` instead). `scale` and `power` need
$k \ge 0$. An input that exceeds its constraint by at most $10^{-12}$ is accepted
as rounding and put back on the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .._validate import TOL, choose, finite, unit
from ._base import FuzzyNumber
from .orthopair import _EPS, _PROJECT_ULPS, _ROUNDING, _log_one_minus, _one_minus_pow, _pow

# Magnitude limits (RULING, documented in SphericalFN): the spherical operations
# work on the squares of the parts, so a part below ~1.5e-154 squares to 0.0 in
# a double and is treated as 0; the absolute error this causes is below 1.5e-154.


def _pow_diff(x: float, y: float, k: float) -> float:
    """(x + y)^k - y^k for x, y >= 0 without cancellation.

    Equals ``t^k (1 - (y/t)^k)`` with ``t = x + y``; the bracket is evaluated
    with expm1 on log1p(-x/t) (x small) or log(y/t) (x large).
    """
    if k == 0 or x <= 0.0:
        return 0.0
    if y <= 0.0:
        return x**k
    t = x + y
    log_ratio = math.log1p(-x / t) if x <= 0.5 * t else math.log(y / t)
    return t**k * -math.expm1(k * log_ratio)


def _two_prod(x: float) -> tuple[float, float]:
    """Exact square of ``x`` as an unevaluated sum ``hi + lo`` (Dekker)."""
    hi = x * x
    c = 134217729.0 * x
    xh = c - (c - x)
    xl = x - xh
    return hi, ((xh * xh - hi) + 2.0 * xh * xl) + xl * xl


def _refusal(*xs: float) -> float:
    """1 - sum(x^2), computed from the exact squares and clamped at 0."""
    terms = [1.0]
    for x in xs:
        hi, lo = _two_prod(x)
        terms += [-hi, -lo]
    return max(0.0, math.fsum(terms))


def _fit(vals: tuple[float, ...], what: str, label: str) -> tuple[float, ...]:
    """Clamp each part to [0, 1], check their sum, absorb rounding overshoot."""
    vals = tuple(unit(v, f"{label} part") for v in vals)
    total = math.fsum(vals)
    if total > 1.0 + _ROUNDING:
        raise ValueError(f"result violates {what} (sum {total})")
    if total > 1.0:
        vals = tuple(v / total for v in vals)
    return vals


class _ThreePart(FuzzyNumber):
    """Shared base: three parts, algebraic operations, simplex distances.

    The methods documented here are inherited by [`SphericalFN`][bulanik.numbers.SphericalFN],
    [`PictureFN`][bulanik.numbers.PictureFN] and [`NeutrosophicFN`][bulanik.numbers.NeutrosophicFN].
    """

    _FIELDS: tuple[str, ...] = ()
    _DIST_DIV = 2.0
    OPS = ("algebraic",)
    DISTANCES = ("hamming", "euclidean")
    DEFAULT_DISTANCE = "hamming"
    ACCURACIES = ("sum",)
    DEFAULT_ACCURACY = "sum"

    @property
    def parts(self) -> tuple[float, float, float]:
        """The three parts as a tuple, in the order of the constructor arguments.

        Examples:
            >>> from bulanik import PictureFN
            >>> PictureFN(0.5, 0.2, 0.1).parts
            (0.5, 0.2, 0.1)
        """
        return tuple(getattr(self, n) for n in self._FIELDS)  # type: ignore[return-value]

    def __post_init__(self) -> None:
        for n in self._FIELDS:
            object.__setattr__(self, n, unit(getattr(self, n), n))
        self._check()

    def _project(self, scale: float) -> None:
        """Rescale the parts by ``scale`` (an input up to TOL past the boundary).

        Accepted near-boundary inputs are put back on the boundary on entry, so
        amplifying operations never magnify the excess past the result check.
        """
        for n in self._FIELDS:
            object.__setattr__(self, n, getattr(self, n) * scale)

    def _check(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _simplex_parts(self) -> tuple[float, ...]:  # pragma: no cover
        raise NotImplementedError

    def _finish(self, raw: tuple[float, float, float]) -> Any:  # pragma: no cover
        raise NotImplementedError

    def _binary_raw(self, name: str, other: Any) -> tuple[float, float, float]:
        raise NotImplementedError  # pragma: no cover

    def _unary_raw(self, name: str, k: float) -> tuple[float, float, float]:
        raise NotImplementedError  # pragma: no cover

    # -- arithmetic ----------------------------------------------------------
    def _binary(self, name: str, other: Any, ops: str) -> Any:
        self._check_ops(ops)
        self.same_kind(other)
        return self._finish(self._binary_raw(name, other))

    def _unary(self, name: str, k: float, ops: str) -> Any:
        self._check_ops(ops)
        k = finite(k, "k")
        if k < 0:
            raise ValueError(f"{name} needs k >= 0, got {k}")
        return self._finish(self._unary_raw(name, k))

    def add(self, other: Any, ops: str = "algebraic") -> Any:
        """Sum of two numbers of the same type; the formulas are on each class.

        `A + B` calls this method.

        Args:
            other: A number of the same class.
            ops: Only `"algebraic"` exists.

        Returns:
            The sum, of the same class.

        Raises:
            TypeError: If `other` is of a different class.
            ValueError: If `ops` is unknown, or the result violates the constraint
                of the type beyond rounding.

        Examples:
            >>> from bulanik import NeutrosophicFN
            >>> s = NeutrosophicFN(0.7, 0.2, 0.1) + NeutrosophicFN(0.5, 0.3, 0.2)
            >>> [round(p, 4) for p in s.parts]
            [0.85, 0.06, 0.02]
        """
        return self._binary("add", other, ops)

    def mul(self, other: Any, ops: str = "algebraic") -> Any:
        """Product of two numbers of the same type; the formulas are on each class.

        `A * B` calls this method.

        Args:
            other: A number of the same class.
            ops: Only `"algebraic"` exists.

        Returns:
            The product, of the same class.

        Raises:
            TypeError: If `other` is of a different class.
            ValueError: If `ops` is unknown, or the result violates the constraint
                of the type beyond rounding.

        Examples:
            >>> from bulanik import PictureFN
            >>> p = PictureFN(0.5, 0.2, 0.1) * PictureFN(0.4, 0.3, 0.2)
            >>> [round(x, 4) for x in p.parts]
            [0.43, 0.06, 0.28]
        """
        return self._binary("mul", other, ops)

    def scale(self, k: float, ops: str = "algebraic") -> Any:
        """Scalar multiple ``k * A``; the formulas are on each class.

        Args:
            k: A finite scalar $\\ge 0$.
            ops: Only `"algebraic"` exists.

        Returns:
            The scaled number, of the same class.

        Raises:
            ValueError: If `k` is negative or not finite, `ops` is unknown, or the
                result violates the constraint of the type (see
                [`PictureFN`][bulanik.numbers.PictureFN] for `k < 1`).

        Examples:
            >>> from bulanik import NeutrosophicFN
            >>> [round(p, 4) for p in NeutrosophicFN(0.7, 0.2, 0.1).scale(2).parts]
            [0.91, 0.04, 0.01]
        """
        return self._unary("scale", k, ops)

    def power(self, k: float, ops: str = "algebraic") -> Any:
        """Power ``A ** k``; the formulas are on each class.

        Args:
            k: A finite exponent $\\ge 0$.
            ops: Only `"algebraic"` exists.

        Returns:
            The power, of the same class.

        Raises:
            ValueError: If `k` is negative or not finite, `ops` is unknown, or the
                result violates the constraint of the type beyond rounding.

        Examples:
            >>> from bulanik import PictureFN
            >>> [round(p, 4) for p in (PictureFN(0.5, 0.2, 0.1) ** 2).parts]
            [0.45, 0.04, 0.19]
        """
        return self._unary("power", k, ops)

    # -- distance ------------------------------------------------------------
    def distance(self, other: Any, method: str | None = None) -> float:
        r"""Normalized Hamming (default) or Euclidean distance between two numbers.

        With $d_i$ the differences of the "simplex parts" of the two numbers
        (spherical: the squares plus the refusal $1 - \mu^2 - \nu^2 - \pi^2$;
        picture: $\mu, \eta, \nu$ and the refusal $1 - \mu - \eta - \nu$;
        neutrosophic: $t, i, f$) the Hamming distance is $\sum |d_i| / D$ and the
        Euclidean one $\sqrt{\sum d_i^2 / D}$, with $D = 2$ (spherical, picture)
        or $D = 3$ (neutrosophic).

        Args:
            other: A number of the same class.
            method: `"hamming"` or `"euclidean"`; `None` means `"hamming"`.

        Returns:
            The distance.

        Raises:
            TypeError: If `other` is of a different class.
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import NeutrosophicFN
            >>> a, b = NeutrosophicFN(0.7, 0.2, 0.1), NeutrosophicFN(0.5, 0.3, 0.2)
            >>> round(a.distance(b), 4)
            0.1333
        """
        method = choose(
            method, self.DISTANCES, "distance", type(self).__name__, self.DEFAULT_DISTANCE
        )
        self.same_kind(other)
        d = [x - y for x, y in zip(self._simplex_parts(), other._simplex_parts())]
        if method == "euclidean":
            return math.sqrt(sum(x * x for x in d) / self._DIST_DIV)
        return sum(abs(x) for x in d) / self._DIST_DIV


# -- spherical ----------------------------------------------------------------
@dataclass(frozen=True)
class SphericalFN(_ThreePart):
    r"""Spherical fuzzy number $(\mu, \nu, \pi)$ with $\mu^2 + \nu^2 + \pi^2 \le 1$.

    Operations of Kutlu Gundogdu and Kahraman (2019), evaluated on the squares
    $(a, b, c) = (\mu^2, \nu^2, \pi^2)$ and the exact refusal
    $r = 1 - a - b - c$ (squares of the results are taken back with a square root):

    | operation | $\mu^2$ | $\nu^2$ | $\pi^2$ |
    |---|---|---|---|
    | `add` | $1 - (1 - a_1)(1 - a_2)$ | $b_1 b_2$ | $(1 - a_1)(1 - a_2) - (b_1 + r_1)(b_2 + r_2)$ |
    | `mul` | $a_1 a_2$ | $1 - (1 - b_1)(1 - b_2)$ | $(1 - b_1)(1 - b_2) - (a_1 + r_1)(a_2 + r_2)$ |
    | `scale` by $k$ | $1 - (1 - a)^k$ | $b^k$ | $(1 - a)^k - (1 - a - c)^k$ |
    | `power` by $k$ | $a^k$ | $1 - (1 - b)^k$ | $(1 - b)^k - (1 - b - c)^k$ |

    Scores: `"pi_adjusted"` (default) $= (\mu - \pi)^2 - (\nu - \pi)^2$ and
    `"normalized"` $= (1 + \text{pi_adjusted})/2$. Accuracy `"sum"`
    $= \mu^2 + \nu^2 + \pi^2$.

    Limit: a part below about 1.5e-154 squares to 0.0 in a double, so it behaves
    as 0 in the operations (absolute error < 1.5e-154; the sqrt of a result part
    is likewise exact only down to that scale). Parts that small carry no
    information at double precision, so no log-domain arithmetic is used.

    Args:
        mu: Membership degree in $[0, 1]$.
        nu: Non-membership degree in $[0, 1]$.
        pi: Hesitancy degree in $[0, 1]$.

    Raises:
        ValueError: If a part is outside $[0, 1]$ or not finite, or
            $\mu^2 + \nu^2 + \pi^2 > 1$ beyond rounding.
        TypeError: If a part is a `bool` or a string.

    Attributes:
        mu: Membership degree.
        nu: Non-membership degree.
        pi: Hesitancy degree.

    References:
        Kutlu Gundogdu and Kahraman (2019), Journal of Intelligent & Fuzzy
        Systems 36(1), 337-352: "Spherical fuzzy sets and spherical fuzzy TOPSIS
        method".

    Examples:
        >>> from bulanik import SphericalFN
        >>> a = SphericalFN(0.5, 0.4, 0.3)
        >>> round(a.score(), 4), round(a.accuracy(), 4)
        (0.03, 0.5)
        >>> s = a + SphericalFN(0.4, 0.3, 0.2)
        >>> [round(p, 4) for p in s.parts]
        [0.6083, 0.12, 0.3194]
        >>> SphericalFN(0.8, 0.5, 0.5)
        Traceback (most recent call last):
        ...
        ValueError: need mu^2 + nu^2 + pi^2 <= 1, got mu=0.8, nu=0.5, pi=0.5 (sum 1.14)
    """

    mu: float
    nu: float
    pi: float

    _FIELDS = ("mu", "nu", "pi")
    SCORES = ("pi_adjusted", "normalized")
    DEFAULT_SCORE = "pi_adjusted"

    def _check(self) -> None:
        total = self.mu**2 + self.nu**2 + self.pi**2
        if total > 1.0 + TOL:
            raise ValueError(
                f"need mu^2 + nu^2 + pi^2 <= 1, got mu={self.mu}, nu={self.nu}, "
                f"pi={self.pi} (sum {total})"
            )
        if total > 1.0 + _PROJECT_ULPS * _EPS:
            self._project(1.0 / math.sqrt(total))

    def _state(self) -> tuple[float, float, float, float]:
        """Squares ``(a, b, c)`` and the accurate refusal ``r``."""
        return (
            self.mu * self.mu,
            self.nu * self.nu,
            self.pi * self.pi,
            _refusal(self.mu, self.nu, self.pi),
        )

    def _finish(self, raw: tuple[float, float, float]) -> Any:
        a, b, c = _fit(raw, "mu^2 + nu^2 + pi^2 <= 1", "spherical")
        return type(self)(math.sqrt(a), math.sqrt(b), math.sqrt(c))

    def _binary_raw(self, name: str, other: Any) -> tuple[float, float, float]:
        a1, b1, c1, r1 = self._state()
        a2, b2, c2, r2 = other._state()
        if name == "add":
            # 1-(1-a1)(1-a2) = a1 + a2 (1-a1); every term is non-negative
            return (
                a1 + a2 * (b1 + c1 + r1),
                b1 * b2,
                c1 * (b2 + c2 + r2) + c2 * (b1 + r1),
            )
        return (
            a1 * a2,
            b1 + b2 * (a1 + c1 + r1),
            c1 * (a2 + c2 + r2) + c2 * (a1 + r1),
        )

    def _unary_raw(self, name: str, k: float) -> tuple[float, float, float]:
        a, b, c, r = self._state()
        if name == "scale":
            # (1-a)^k - (1-a-c)^k = (c + s)^k - s^k with s = b + r
            return (
                _one_minus_pow(a, b + c + r, k),
                _pow(b, k),
                _pow_diff(c, b + r, k),
            )
        return (
            _pow(a, k),
            _one_minus_pow(b, a + c + r, k),
            _pow_diff(c, a + r, k),
        )

    def complement(self) -> "SphericalFN":
        r"""Complement $(\nu, \mu, \pi)$: membership and non-membership swapped.

        Examples:
            >>> from bulanik import SphericalFN
            >>> SphericalFN(0.5, 0.4, 0.3).complement()
            SphericalFN(mu=0.4, nu=0.5, pi=0.3)
        """
        return SphericalFN(self.nu, self.mu, self.pi)

    def score(self, method: str | None = None) -> float:
        r"""Score $(\mu - \pi)^2 - (\nu - \pi)^2$ (`"pi_adjusted"`, default) or its `"normalized"` form $(1 + s)/2$.

        Args:
            method: `"pi_adjusted"` or `"normalized"`; `None` means `"pi_adjusted"`.

        Returns:
            The score.

        Raises:
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import SphericalFN
            >>> round(SphericalFN(0.5, 0.4, 0.3).score("normalized"), 4)
            0.515
        """
        method = choose(
            method, self.SCORES, "score", type(self).__name__, self.DEFAULT_SCORE
        )
        s = (self.mu - self.pi) ** 2 - (self.nu - self.pi) ** 2
        return (1.0 + s) / 2.0 if method == "normalized" else s

    def accuracy(self, method: str | None = None) -> float:
        r"""Accuracy $\mu^2 + \nu^2 + \pi^2$ (`"sum"`, the only method).

        Args:
            method: `"sum"` or `None`.

        Returns:
            The accuracy in $[0, 1]$.

        Raises:
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import SphericalFN
            >>> round(SphericalFN(0.5, 0.4, 0.3).accuracy(), 4)
            0.5
        """
        choose(
            method, self.ACCURACIES, "accuracy", type(self).__name__, self.DEFAULT_ACCURACY
        )
        return self.mu**2 + self.nu**2 + self.pi**2

    def _simplex_parts(self) -> tuple[float, ...]:
        return self._state()


# -- picture ------------------------------------------------------------------
@dataclass(frozen=True)
class PictureFN(_ThreePart):
    r"""Picture fuzzy number $(\mu, \eta, \nu)$ with $\mu + \eta + \nu \le 1$.

    The parts are the degrees of positive membership, neutral membership and
    negative membership. Operations of Wei (2017), as the code evaluates them:

    | operation | $\mu$ | $\eta$ | $\nu$ |
    |---|---|---|---|
    | `add` | $\mu_1 + \mu_2 - \mu_1 \mu_2$ | $\eta_1 \eta_2$ | $\nu_1 \nu_2$ |
    | `mul` | $(\mu_1 + \eta_1)(\mu_2 + \eta_2) - \eta_1 \eta_2$ | $\eta_1 \eta_2$ | $\nu_1 + \nu_2 - \nu_1 \nu_2$ |
    | `scale` by $k$ | $1 - (1 - \mu)^k$ | $\eta^k$ | $\nu^k$ |
    | `power` by $k$ | $(\mu + \eta)^k - \eta^k$ | $\eta^k$ | $1 - (1 - \nu)^k$ |

    Multiplication and power are closed; addition is too. Scaling by $k < 1$ can
    leave the constraint set ($\eta^k + \nu^k$ may exceed $(1 - \mu)^k$) and then
    raises `ValueError`, as does $k = 0$, because the additive identity
    $(0, 1, 1)$ is not itself a valid number. The weighted aggregations
    [`wa`][bulanik.aggregate.wa] and [`wg`][bulanik.aggregate.wg] use closed
    forms that avoid this.

    Scores: `"difference"` (default) $= \mu - \nu$ and `"normalized"`
    $= (1 + \mu - \nu)/2$. Accuracy `"sum"` $= \mu + \eta + \nu$.

    Args:
        mu: Positive membership degree in $[0, 1]$.
        eta: Neutral membership degree in $[0, 1]$.
        nu: Negative membership degree in $[0, 1]$.

    Raises:
        ValueError: If a part is outside $[0, 1]$ or not finite, or
            $\mu + \eta + \nu > 1$ beyond rounding.
        TypeError: If a part is a `bool` or a string.

    Attributes:
        mu: Positive membership degree.
        eta: Neutral membership degree.
        nu: Negative membership degree.

    References:
        Wei (2017), Journal of Intelligent & Fuzzy Systems 33(2): "Picture fuzzy aggregation operators and their application to multiple attribute decision making".
        Cuong (2014), Journal of Computer Science and Cybernetics 30(4), 409-420: picture fuzzy sets.
        Note: Wei (2017) was retracted by the publisher in 2025; the closed forms
        are kept as documented and are tested against the code, not the paper.

    Examples:
        >>> from bulanik import PictureFN
        >>> a = PictureFN(0.5, 0.2, 0.1)
        >>> round(a.score(), 4), round(a.accuracy(), 4)
        (0.4, 0.8)
        >>> a.scale(0.5)
        Traceback (most recent call last):
        ...
        ValueError: result violates mu + eta + nu <= 1 (sum 1.0...)
    """

    mu: float
    eta: float
    nu: float

    _FIELDS = ("mu", "eta", "nu")
    SCORES = ("difference", "normalized")
    DEFAULT_SCORE = "difference"

    def _check(self) -> None:
        total = self.mu + self.eta + self.nu
        if total > 1.0 + TOL:
            raise ValueError(
                f"need mu + eta + nu <= 1, got mu={self.mu}, eta={self.eta}, "
                f"nu={self.nu} (sum {total})"
            )
        if total > 1.0 + _PROJECT_ULPS * _EPS:
            self._project(1.0 / total)

    def _rho(self) -> float:
        return max(0.0, math.fsum([1.0, -self.mu, -self.eta, -self.nu]))

    def _finish(self, raw: tuple[float, float, float]) -> Any:
        return type(self)(*_fit(raw, "mu + eta + nu <= 1", "picture"))

    def _binary_raw(self, name: str, other: Any) -> tuple[float, float, float]:
        m1, e1, n1, m2, e2, n2 = *self.parts, *other.parts
        if name == "add":  # 1-(1-m1)(1-m2) = m1 + m2 (1-m1), 1-m1 = e1+n1+rho1
            return m1 + m2 * (e1 + n1 + self._rho()), e1 * e2, n1 * n2
        # (m1+e1)(m2+e2) - e1 e2 expanded; 1-(1-n1)(1-n2) = n1 + n2 (1-n1)
        return (
            m1 * m2 + m1 * e2 + e1 * m2,
            e1 * e2,
            n1 + n2 * (m1 + e1 + self._rho()),
        )

    def _unary_raw(self, name: str, k: float) -> tuple[float, float, float]:
        m, e, n, rho = self.mu, self.eta, self.nu, self._rho()
        if name == "scale":
            return _one_minus_pow(m, e + n + rho, k), _pow(e, k), _pow(n, k)
        return _pow_diff(m, e, k), _pow(e, k), _one_minus_pow(n, m + e + rho, k)

    def complement(self) -> "PictureFN":
        r"""Complement $(\nu, \eta, \mu)$: positive and negative degrees swapped.

        Examples:
            >>> from bulanik import PictureFN
            >>> PictureFN(0.5, 0.2, 0.1).complement()
            PictureFN(mu=0.1, eta=0.2, nu=0.5)
        """
        return PictureFN(self.nu, self.eta, self.mu)

    def score(self, method: str | None = None) -> float:
        r"""Score $\mu - \nu$ (`"difference"`, default) or $(1 + \mu - \nu)/2$ (`"normalized"`).

        Args:
            method: `"difference"` or `"normalized"`; `None` means `"difference"`.

        Returns:
            The score.

        Raises:
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import PictureFN
            >>> round(PictureFN(0.5, 0.2, 0.1).score("normalized"), 4)
            0.7
        """
        method = choose(
            method, self.SCORES, "score", type(self).__name__, self.DEFAULT_SCORE
        )
        s = self.mu - self.nu
        return (1.0 + s) / 2.0 if method == "normalized" else s

    def accuracy(self, method: str | None = None) -> float:
        r"""Accuracy $\mu + \eta + \nu$ (`"sum"`, the only method).

        Args:
            method: `"sum"` or `None`.

        Returns:
            The accuracy in $[0, 1]$.

        Raises:
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import PictureFN
            >>> round(PictureFN(0.5, 0.2, 0.1).accuracy(), 4)
            0.8
        """
        choose(
            method, self.ACCURACIES, "accuracy", type(self).__name__, self.DEFAULT_ACCURACY
        )
        return self.mu + self.eta + self.nu

    def _simplex_parts(self) -> tuple[float, ...]:
        return self.mu, self.eta, self.nu, self._rho()


# -- picture weighted aggregation (used by bulanik.aggregate) ----
def _wsum_log(terms: Sequence[tuple[float, float]]) -> float:
    """sum(w * log x) for x >= 0; ``-inf`` when any x is 0 (all w > 0)."""
    if any(x <= 0.0 for _, x in terms):
        return -math.inf
    return math.fsum(w * math.log(x) for w, x in terms)


def _picture_aggregate(
    values: list[PictureFN], w: list[float], kind: str
) -> PictureFN:
    """Closed forms of Wei (2017); valid for sum(w) = 1 by Holder's inequality."""

    def one_minus_prod_one_minus(xs: list[float], cxs: list[float]) -> float:
        # cxs are the accurate complements 1 - x (the other two parts plus rho):
        # 1 - x itself would carry the rounding of x, which exponents < 1 amplify
        if any(c <= 0.0 for c in cxs):
            return 1.0
        return -math.expm1(
            math.fsum(wi * _log_one_minus(x, c) for wi, x, c in zip(w, xs, cxs))
        )

    def prod(xs: list[float]) -> float:
        return math.exp(_wsum_log(list(zip(w, xs))))

    mus = [v.mu for v in values]
    etas = [v.eta for v in values]
    nus = [v.nu for v in values]
    rhos = [v._rho() for v in values]
    if kind == "wa":
        c_mu = [math.fsum([e, n, r]) for e, n, r in zip(etas, nus, rhos)]
        raw = (one_minus_prod_one_minus(mus, c_mu), prod(etas), prod(nus))
    else:
        ts = [m + e for m, e in zip(mus, etas)]
        log_p = _wsum_log(list(zip(w, ts)))
        if log_p == -math.inf:
            mu = 0.0
        else:
            # prod(t^w) - prod(eta^w) = P (1 - prod((eta/t)^w)), no cancellation
            if any(e <= 0.0 for e in etas):
                mu = math.exp(log_p)
            else:
                lr = math.fsum(
                    wi * (math.log1p(-m / t) if m <= 0.5 * t else math.log(e / t))
                    for wi, m, e, t in zip(w, mus, etas, ts)
                )
                mu = math.exp(log_p) * -math.expm1(lr)
        c_nu = [math.fsum([m, e, r]) for m, e, r in zip(mus, etas, rhos)]
        raw = (mu, prod(etas), one_minus_prod_one_minus(nus, c_nu))
    return values[0]._finish(tuple(x + 0.0 for x in raw))  # + 0.0 turns -0.0 into 0.0


# -- neutrosophic ---------------------------------------------------------------
@dataclass(frozen=True)
class NeutrosophicFN(_ThreePart):
    r"""Single-valued neutrosophic number $(t, i, f)$, each part in $[0, 1]$.

    The parts are truth, indeterminacy and falsity; they are independent (no
    joint constraint). Operations as the code evaluates them:

    | operation | $t$ | $i$ | $f$ |
    |---|---|---|---|
    | `add` | $t_1 + t_2 - t_1 t_2$ | $i_1 i_2$ | $f_1 f_2$ |
    | `mul` | $t_1 t_2$ | $i_1 + i_2 - i_1 i_2$ | $f_1 + f_2 - f_1 f_2$ |
    | `scale` by $k$ | $1 - (1 - t)^k$ | $i^k$ | $f^k$ |
    | `power` by $k$ | $t^k$ | $1 - (1 - i)^k$ | $1 - (1 - f)^k$ |

    Score (the only one, `"average"`): $(2 + t - i - f)/3$. Accuracy:
    `"difference"` (default) $= t - f$ or `"certainty"` $= t$. Distances divide
    by 3 instead of 2.

    Args:
        t: Truth degree in $[0, 1]$.
        i: Indeterminacy degree in $[0, 1]$.
        f: Falsity degree in $[0, 1]$.

    Raises:
        ValueError: If a part is outside $[0, 1]$ or not finite.
        TypeError: If a part is a `bool` or a string.

    Attributes:
        t: Truth degree.
        i: Indeterminacy degree.
        f: Falsity degree.

    References:
        Wang, Smarandache, Zhang and Sunderraman (2010), Single valued
        neutrosophic sets, Multispace and Multistructure 4.

    Examples:
        >>> from bulanik import NeutrosophicFN
        >>> a = NeutrosophicFN(0.7, 0.2, 0.1)
        >>> round(a.score(), 4), a.accuracy(), a.accuracy("certainty")
        (0.8, 0.6, 0.7)
        >>> a.complement()
        NeutrosophicFN(t=0.1, i=0.8, f=0.7)
    """

    t: float
    i: float
    f: float

    _FIELDS = ("t", "i", "f")
    _DIST_DIV = 3.0
    SCORES = ("average",)
    DEFAULT_SCORE = "average"
    ACCURACIES = ("difference", "certainty")
    DEFAULT_ACCURACY = "difference"

    def _check(self) -> None:
        pass  # each part already validated into [0, 1]

    def _finish(self, raw: tuple[float, float, float]) -> Any:
        return type(self)(*raw)

    def _binary_raw(self, name: str, other: Any) -> tuple[float, float, float]:
        t1, i1, f1, t2, i2, f2 = *self.parts, *other.parts
        if name == "add":  # t1+t2-t1t2 = t1 + t2 (1-t1)
            return t1 + t2 * (1.0 - t1), i1 * i2, f1 * f2
        return t1 * t2, i1 + i2 * (1.0 - i1), f1 + f2 * (1.0 - f1)

    def _unary_raw(self, name: str, k: float) -> tuple[float, float, float]:
        t, i, f = self.parts
        if name == "scale":
            return _one_minus_pow(t, 1.0 - t, k), _pow(i, k), _pow(f, k)
        return _pow(t, k), _one_minus_pow(i, 1.0 - i, k), _one_minus_pow(f, 1.0 - f, k)

    def complement(self) -> "NeutrosophicFN":
        r"""Complement $(f, 1 - i, t)$.

        Examples:
            >>> from bulanik import NeutrosophicFN
            >>> NeutrosophicFN(0.7, 0.2, 0.1).complement()
            NeutrosophicFN(t=0.1, i=0.8, f=0.7)
        """
        return NeutrosophicFN(self.f, 1.0 - self.i, self.t)

    def score(self, method: str | None = None) -> float:
        r"""Score $(2 + t - i - f)/3$ (`"average"`, the only method).

        Args:
            method: `"average"` or `None`.

        Returns:
            The score in $[0, 1]$.

        Raises:
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import NeutrosophicFN
            >>> round(NeutrosophicFN(0.7, 0.2, 0.1).score(), 4)
            0.8
        """
        choose(method, self.SCORES, "score", type(self).__name__, self.DEFAULT_SCORE)
        return (2.0 + self.t - self.i - self.f) / 3.0

    def accuracy(self, method: str | None = None) -> float:
        r"""Accuracy: $t - f$ (`"difference"`, default) or $t$ (`"certainty"`).

        Args:
            method: `"difference"`, `"certainty"` or `None`.

        Returns:
            The accuracy.

        Raises:
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import NeutrosophicFN
            >>> a = NeutrosophicFN(0.7, 0.2, 0.1)
            >>> round(a.accuracy(), 4), a.accuracy("certainty")
            (0.6, 0.7)
        """
        method = choose(
            method, self.ACCURACIES, "accuracy", type(self).__name__, self.DEFAULT_ACCURACY
        )
        return self.t if method == "certainty" else self.t - self.f

    def _simplex_parts(self) -> tuple[float, ...]:
        return self.t, self.i, self.f
