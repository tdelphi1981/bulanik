r"""Interval-valued q-rung orthopair fuzzy numbers: IVQROFN and IVIFN, IVPFN, IVFFN.

An interval-valued q-rung orthopair fuzzy number pairs a membership interval
$[\mu_L, \mu_U]$ with a non-membership interval $[\nu_L, \nu_U]$ such that
$\mu_U^q + \nu_U^q \le 1$ (Joshi et al. 2018). The rung $q = 1$ gives
interval-valued intuitionistic numbers ([`IVIFN`][bulanik.numbers.IVIFN],
Atanassov and Gargov 1989), $q = 2$ interval-valued Pythagorean ones
([`IVPFN`][bulanik.numbers.IVPFN], Peng and Yang 2016) and $q = 3$
interval-valued Fermatean ones ([`IVFFN`][bulanik.numbers.IVFFN], Jeevaraj 2021).

Every operation applies the point law of [`QROFN`][bulanik.numbers.QROFN] to the
lower pair $(\mu_L^q, \nu_L^q)$ and to the upper pair $(\mu_U^q, \nu_U^q)$
separately. Each point law is monotone in every argument, so the lower bounds
stay below the upper ones, and the upper pair carries the constraint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .._validate import TOL, choose, finite, unit
from ._base import FuzzyNumber
from .orthopair import (
    _BINARY,
    _EPS,
    _PROJECT_ULPS,
    _ROUNDING,
    _UNARY,
    QROFN,
    _overshoot_ulps,
)


def _project(a: float, b: float, slack: float, q: int, what: str) -> tuple[float, float]:
    """Check a result pair of q-th powers and put rounding overshoot back on a + b = 1."""
    a, b = unit(a, f"{what} mu^q"), unit(b, f"{what} nu^q")
    total = a + b
    if total > 1.0 + max(_ROUNDING, slack):
        raise ValueError(f"result violates mu_u^{q} + nu_u^{q} <= 1 (sum {total})")
    if total > 1.0:
        a, b = a / total, b / total
    return a, b


@dataclass(frozen=True)
class IVQROFN(FuzzyNumber):
    r"""Interval-valued q-rung orthopair fuzzy number $([\mu_L, \mu_U], [\nu_L, \nu_U])$.

    With $a = \mu^q$ and $b = \nu^q$, every operation is the point law of
    [`QROFN`][bulanik.numbers.QROFN] (see its table of operators, both `"algebraic"`
    and `"einstein"`) applied to $(a_L, b_L)$ and to $(a_U, b_U)$ separately; the
    result is $\mu$ = [first part of the lower result, first part of the upper
    result] to the power $1/q$, and likewise for $\nu$. For $q = 1$ and the
    algebraic operators this is Xu's (2007) arithmetic, e.g.
    $\tilde a \oplus \tilde b = ([a^L + b^L - a^L b^L,\, a^U + b^U - a^U b^U],
    [c^L d^L,\, c^U d^U])$.

    Scores: `"difference"` (default)
    $S = \tfrac12(\mu_L^q + \mu_U^q - \nu_L^q - \nu_U^q)$ (Xu 2007 for $q = 1$,
    Peng and Yang 2016 for $q = 2$), `"normalized"` $(1 + S)/2$ and, for
    $q = 1$ only, `"defuzzified"` (see [`score`][bulanik.numbers.IVQROFN.score]). Accuracy
    `"sum"` $H = \tfrac12(\mu_L^q + \mu_U^q + \nu_L^q + \nu_U^q)$. Distances
    (default `"hamming"`) work on $(a_L, a_U, b_L, b_U, \pi_L, \pi_U)$ with
    $\pi_L = 1 - a_U - b_U$ and $\pi_U = 1 - a_L - b_L$:
    Hamming $= \tfrac14 \sum |\Delta|$ and Euclidean $= \sqrt{\tfrac14 \sum \Delta^2}$.
    For degenerate intervals all of these equal the `QROFN` values, except
    `"defuzzified"`, which `QROFN` does not have.

    Inversions $\mu_L > \mu_U$ or $\nu_L > \nu_U$ of up to $10^{-12}$ are clamped
    (lower := upper); larger ones raise `ValueError`. The constraint is checked and
    projected on the upper pair exactly as `QROFN` does, and the lower bounds are
    then clamped to at most the upper ones.

    Args:
        mu_l: Lower membership bound in $[0, 1]$.
        mu_u: Upper membership bound in $[0, 1]$.
        nu_l: Lower non-membership bound in $[0, 1]$.
        nu_u: Upper non-membership bound in $[0, 1]$.
        q: The rung, an integer $\ge 1$.

    Raises:
        ValueError: If `q` is not an integer $\ge 1$, a bound is outside $[0, 1]$ or
            not finite, a lower bound exceeds its upper bound, or
            $\mu_U^q + \nu_U^q > 1$ (beyond rounding).
        TypeError: If a bound is a `bool` or a string.

    Attributes:
        mu_l: Lower membership bound.
        mu_u: Upper membership bound.
        nu_l: Lower non-membership bound.
        nu_u: Upper non-membership bound.
        q: The rung.

    References:
        Atanassov and Gargov (1989), Fuzzy Sets and Systems 31(3), 343-349
        (interval-valued intuitionistic fuzzy sets). Xu (2007), Control and
        Decision 22(2), 215-219 (arithmetic, score and accuracy). Wang and Liu
        (2013), Journal of Intelligent & Fuzzy Systems 25(2), 279-290 (Einstein
        operations). Peng and Yang (2016), International Journal of Intelligent
        Systems 31(5), 444-487. Joshi, Singh, Bhatt and Vaisla (2018), Journal of
        Intelligent & Fuzzy Systems 35(5), 5225-5230 (interval-valued q-rung
        orthopair sets). Garg (2016), Journal of Intelligent & Fuzzy Systems 31(1),
        529-540, doi:10.3233/IFS-162165, gives an accuracy function for $q = 2$
        that is not implemented; the scores above are the ones of Xu (2007) and
        Peng and Yang (2016).

    Examples:
        >>> from bulanik import IVQROFN
        >>> IVQROFN(0.6, 0.8, 0.3, 0.5, 3)
        IVQROFN(mu_l=0.6, mu_u=0.8, nu_l=0.3, nu_u=0.5, q=3)
        >>> IVQROFN(0.6, 0.9, 0.3, 0.7, 3)
        Traceback (most recent call last):
        ...
        ValueError: need mu_u^3 + nu_u^3 <= 1, got mu_u=0.9, nu_u=0.7 (sum 1.072)
    """

    mu_l: float
    mu_u: float
    nu_l: float
    nu_u: float
    q: int

    OPS = ("algebraic", "einstein")
    SCORES = ("difference", "normalized", "defuzzified")
    DEFAULT_SCORE = "difference"
    ACCURACIES = ("sum",)
    DEFAULT_ACCURACY = "sum"
    DISTANCES = ("hamming", "euclidean")
    DEFAULT_DISTANCE = "hamming"

    def __post_init__(self) -> None:
        q = self.q
        if isinstance(q, bool) or not isinstance(q, int) or q < 1:
            raise ValueError(f"q must be an integer >= 1, got {q!r}")
        mu_l, mu_u = unit(self.mu_l, "mu_l"), unit(self.mu_u, "mu_u")
        nu_l, nu_u = unit(self.nu_l, "nu_l"), unit(self.nu_u, "nu_u")
        mu_l = self._ordered(mu_l, mu_u, "mu")
        nu_l = self._ordered(nu_l, nu_u, "nu")
        total = mu_u**q + nu_u**q
        if total > 1.0 + TOL:
            raise ValueError(
                f"need mu_u^{q} + nu_u^{q} <= 1, got mu_u={mu_u}, nu_u={nu_u} (sum {total})"
            )
        if total > 1.0 + _PROJECT_ULPS * _EPS:
            # as QROFN: put an input within TOL of the boundary back on it
            mu_u = (mu_u**q / total) ** (1.0 / q)
            nu_u = (nu_u**q / total) ** (1.0 / q)
        object.__setattr__(self, "mu_l", min(mu_l, mu_u))
        object.__setattr__(self, "mu_u", mu_u)
        object.__setattr__(self, "nu_l", min(nu_l, nu_u))
        object.__setattr__(self, "nu_u", nu_u)

    @staticmethod
    def _ordered(lower: float, upper: float, part: str) -> float:
        if lower > upper:
            if lower - upper > TOL:
                raise ValueError(
                    f"need {part}_l <= {part}_u, got {part}_l={lower}, {part}_u={upper}"
                )
            return upper
        return lower

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_qrofn(cls, x: QROFN) -> Any:
        """Degenerate interval $([\\mu, \\mu], [\\nu, \\nu])$ of a `QROFN`.

        On a fixed-rung subclass the rung of `x` must equal the class rung.

        Args:
            x: A `QROFN`, `IFN`, `PFN` or `FFN`.

        Returns:
            The interval-valued number with equal lower and upper bounds.

        Raises:
            TypeError: If `x` is not a `QROFN`.
            ValueError: If the class has a fixed rung that differs from `x.q`.

        Examples:
            >>> from bulanik import IFN, IVIFN
            >>> IVIFN.from_qrofn(IFN(0.6, 0.3))
            IVIFN(mu_l=0.6, mu_u=0.6, nu_l=0.3, nu_u=0.3)
        """
        if not isinstance(x, QROFN):
            raise TypeError(f"from_qrofn needs a QROFN, got {type(x).__name__}")
        return cls._from_qrofn(x)

    @classmethod
    def _from_qrofn(cls, x: QROFN) -> Any:
        return cls(x.mu, x.mu, x.nu, x.nu, x.q)

    def _rebuild(self, mu_l: float, mu_u: float, nu_l: float, nu_u: float) -> Any:
        return type(self)(mu_l, mu_u, nu_l, nu_u, self.q)

    def _new_iv(
        self, lower: tuple[float, float], upper: tuple[float, float], slack: float = 0.0
    ) -> Any:
        """Build a result from the result pairs of q-th powers of both ends.

        Each pair gets the `QROFN._new` check (``slack`` widens the accepted
        rounding overshoot). The point laws are monotone, so in exact arithmetic
        lower <= upper; the only inversions come from the upper pair sitting a
        hair past the boundary, where ``_unary`` raises the smaller complement
        to the other part (the clamp below ``QROFN._unary``), which can be
        amplified by a tiny ``k``. Those are put right by lowering the lower
        bound, which keeps the constraint.
        """
        q = self.q
        al, bl = _project(*lower, slack, q, "lower")
        au, bu = _project(*upper, slack, q, "upper")
        r = 1.0 / q
        return self._rebuild(min(al, au) ** r, au**r, min(bl, bu) ** r, bu**r)

    def _powers(self) -> tuple[float, float, float, float]:
        q = self.q
        return self.mu_l**q, self.nu_l**q, self.mu_u**q, self.nu_u**q

    # -- derived quantities --------------------------------------------------
    def complement(self) -> Any:
        """Complement: membership and non-membership intervals swapped.

        Returns:
            The number $([\\nu_L, \\nu_U], [\\mu_L, \\mu_U])$ of the same type.

        Examples:
            >>> from bulanik import IVIFN
            >>> IVIFN(0.4, 0.5, 0.3, 0.4).complement()
            IVIFN(mu_l=0.3, mu_u=0.4, nu_l=0.4, nu_u=0.5)
        """
        return self._rebuild(self.nu_l, self.nu_u, self.mu_l, self.mu_u)

    @property
    def hesitancy(self) -> tuple[float, float]:
        r"""Hesitancy interval $\left((1 - \mu_U^q - \nu_U^q)^{1/q},\, (1 - \mu_L^q - \nu_L^q)^{1/q}\right)$.

        Examples:
            >>> from bulanik import IVIFN
            >>> tuple(round(x, 4) for x in IVIFN(0.4, 0.5, 0.3, 0.4).hesitancy)
            (0.1, 0.3)
        """
        al, bl, au, bu = self._powers()
        r = 1.0 / self.q
        return max(0.0, 1.0 - au - bu) ** r, max(0.0, 1.0 - al - bl) ** r

    # -- arithmetic ----------------------------------------------------------
    def _binary(self, name: str, other: Any, ops: str) -> Any:
        ops = self._check_ops(ops)
        self.same_kind(other)
        f = _BINARY[(ops, name)]
        al1, bl1, au1, bu1 = self._powers()
        al2, bl2, au2, bu2 = other._powers()
        return self._new_iv(f(al1, bl1, al2, bl2), f(au1, bu1, au2, bu2))

    def _unary(self, name: str, k: float, ops: str) -> Any:
        ops = self._check_ops(ops)
        k = finite(k, "k")
        if k < 0:
            raise ValueError(f"{name} needs k >= 0, got {k}")
        f = _UNARY[(ops, name)]
        al, bl, au, bu = self._powers()
        slack = _overshoot_ulps(self.q) * _EPS * (1.0 + k)

        def one(a: float, b: float) -> tuple[float, float]:
            ca, cb = 1.0 - a, 1.0 - b
            # as QROFN._unary: the larger part's complement must not fall below
            # the smaller part for a point up to TOL past the boundary
            if a >= b:
                ca = max(ca, b)
            else:
                cb = max(cb, a)
            return f(a, ca, b, cb, k)

        return self._new_iv(one(al, bl), one(au, bu), slack=slack)

    def add(self, other: Any, ops: str = "algebraic") -> Any:
        """Sum of two numbers of the same type and rung (point law on both ends).

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
            Xu (2007), Control and Decision 22(2), 215-219 (IIFWA, q = 1).
            Wang and Liu (2013), Journal of Intelligent & Fuzzy Systems 25(2),
            279-290 (Einstein operations).

        Examples:
            >>> from bulanik import IVIFN
            >>> a, b = IVIFN(0.4, 0.5, 0.3, 0.4), IVIFN(0.5, 0.6, 0.2, 0.3)
            >>> a + b
            IVIFN(mu_l=0.7, mu_u=0.8, nu_l=0.06, nu_u=0.12)
        """
        return self._binary("add", other, ops)

    def mul(self, other: Any, ops: str = "algebraic") -> Any:
        """Product of two numbers of the same type and rung (point law on both ends).

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
            >>> from bulanik import IVIFN
            >>> IVIFN(0.4, 0.5, 0.3, 0.4) * IVIFN(0.5, 0.6, 0.2, 0.3)
            IVIFN(mu_l=0.2, mu_u=0.3, nu_l=0.44, nu_u=0.58)
        """
        return self._binary("mul", other, ops)

    def scale(self, k: float, ops: str = "algebraic") -> Any:
        """Scalar multiple ``k * A`` (point law on both ends).

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
            >>> from bulanik import IVIFN
            >>> IVIFN(0.4, 0.5, 0.3, 0.4).scale(2)
            IVIFN(mu_l=0.64, mu_u=0.75, nu_l=0.09, nu_u=0.16)
        """
        return self._unary("scale", k, ops)

    def power(self, k: float, ops: str = "algebraic") -> Any:
        """Power ``A ** k`` (point law on both ends).

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
            >>> from bulanik import IVIFN
            >>> IVIFN(0.4, 0.5, 0.3, 0.4) ** 2
            IVIFN(mu_l=0.16, mu_u=0.25, nu_l=0.51, nu_u=0.64)
        """
        return self._unary("power", k, ops)

    # -- scores, accuracy, distance -------------------------------------------
    def score(self, method: str | None = None) -> float:
        r"""Score $\tfrac12(\mu_L^q + \mu_U^q - \nu_L^q - \nu_U^q)$ (`"difference"`, default), $(1 + S)/2$ (`"normalized"`) or a defuzzified value (`"defuzzified"`).

        `"defuzzified"` is defined for intuitionistic numbers ($q = 1$) only:

        $$D = \frac{\mu_L + \mu_U + (1 - \nu_L) + (1 - \nu_U) + \mu_L \mu_U
              - \sqrt{(1 - \nu_L)(1 - \nu_U)}}{4},$$

        which lies in $[0, 1]$.

        Args:
            method: `"difference"`, `"normalized"` or `"defuzzified"`; `None`
                means `"difference"`.

        Returns:
            The score.

        Raises:
            ValueError: If `method` is unknown, or it is `"defuzzified"` and
                $q \ne 1$.

        References:
            Xu (2007), Control and Decision 22(2), 215-219 (q = 1); Peng and Yang
            (2016), International Journal of Intelligent Systems 31(5), 444-487 (q = 2).
            `"defuzzified"`: Bolturk and Kahraman (2022), Informatica 33(4), 693-711,
            Definition 4, doi:10.15388/22-INFOR478.

        Examples:
            >>> from bulanik import IVIFN
            >>> round(IVIFN(0.4, 0.5, 0.3, 0.4).score(), 4)
            0.1
            >>> round(IVIFN(0.4, 0.5, 0.3, 0.4).score("normalized"), 4)
            0.55
            >>> round(IVIFN(0.4, 0.5, 0.3, 0.4).score("defuzzified"), 4)
            0.438
        """
        method = choose(
            method, self.SCORES, "score", type(self).__name__, self.DEFAULT_SCORE
        )
        if method == "defuzzified":
            if self.q != 1:
                raise ValueError(
                    f'score "defuzzified" is defined for IVIFN (q = 1), got q = {self.q}'
                )
            cl, cu = 1.0 - self.nu_l, 1.0 - self.nu_u
            return (self.mu_l + self.mu_u + cl + cu + self.mu_l * self.mu_u - math.sqrt(cl * cu)) / 4.0
        al, bl, au, bu = self._powers()
        s = (al + au - bl - bu) / 2.0
        return (1.0 + s) / 2.0 if method == "normalized" else s

    def accuracy(self, method: str | None = None) -> float:
        r"""Accuracy $\tfrac12(\mu_L^q + \mu_U^q + \nu_L^q + \nu_U^q)$ (`"sum"`, the only method).

        Args:
            method: `"sum"` or `None`.

        Returns:
            The accuracy in $[0, 1]$.

        Raises:
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import IVIFN
            >>> round(IVIFN(0.4, 0.5, 0.3, 0.4).accuracy(), 4)
            0.8
        """
        choose(
            method, self.ACCURACIES, "accuracy", type(self).__name__, self.DEFAULT_ACCURACY
        )
        al, bl, au, bu = self._powers()
        return (al + au + bl + bu) / 2.0

    def distance(self, other: Any, method: str | None = None) -> float:
        """Normalized Hamming (default) or Euclidean distance on the powers of both ends and the hesitancy.

        Args:
            other: A number of the same class and the same `q`.
            method: `"hamming"` or `"euclidean"`; `None` means `"hamming"`.

        Returns:
            The distance, in $[0, 1]$.

        Raises:
            TypeError: If `other` differs in class or rung.
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import IVIFN
            >>> a, b = IVIFN(0.4, 0.5, 0.3, 0.4), IVIFN(0.5, 0.6, 0.2, 0.3)
            >>> round(a.distance(b), 4), round(a.distance(b, "euclidean"), 4)
            (0.1, 0.1)
        """
        method = choose(
            method, self.DISTANCES, "distance", type(self).__name__, self.DEFAULT_DISTANCE
        )
        self.same_kind(other)
        al1, bl1, au1, bu1 = self._powers()
        al2, bl2, au2, bu2 = other._powers()
        d = (
            al1 - al2, au1 - au2, bl1 - bl2, bu1 - bu2,
            (1.0 - au1 - bu1) - (1.0 - au2 - bu2),
            (1.0 - al1 - bl1) - (1.0 - al2 - bl2),
        )
        if method == "euclidean":
            return math.sqrt(sum(x * x for x in d) / 4.0)
        return sum(abs(x) for x in d) / 4.0


class _FixedIV(IVQROFN):
    """An IVQROFN with a fixed q; constructed as ``Cls(mu_l, mu_u, nu_l, nu_u)`` (private base)."""

    Q = 1

    def __init__(
        self, mu_l: float, mu_u: float, nu_l: float, nu_u: float, q: int | None = None
    ) -> None:
        # ``q`` exists so dataclasses.replace() works; it must match the class.
        if q is not None and (isinstance(q, bool) or q != self.Q):
            raise ValueError(f"{type(self).__name__} has q = {self.Q}, got q={q!r}")
        object.__setattr__(self, "mu_l", mu_l)
        object.__setattr__(self, "mu_u", mu_u)
        object.__setattr__(self, "nu_l", nu_l)
        object.__setattr__(self, "nu_u", nu_u)
        object.__setattr__(self, "q", self.Q)
        self.__post_init__()

    @classmethod
    def _from_qrofn(cls, x: QROFN) -> Any:
        if x.q != cls.Q:
            raise ValueError(f"{cls.__name__} has q = {cls.Q}, got a number with q={x.q}")
        return cls(x.mu, x.mu, x.nu, x.nu)

    def _rebuild(self, mu_l: float, mu_u: float, nu_l: float, nu_u: float) -> Any:
        return type(self)(mu_l, mu_u, nu_l, nu_u)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(mu_l={self.mu_l!r}, mu_u={self.mu_u!r}, "
            f"nu_l={self.nu_l!r}, nu_u={self.nu_u!r})"
        )


class IVIFN(_FixedIV):
    """Interval-valued intuitionistic fuzzy number with $\\mu_U + \\nu_U \\le 1$ (q = 1, Atanassov and Gargov 1989).

    An [`IVQROFN`][bulanik.numbers.IVQROFN] with `q = 1`, constructed as
    `IVIFN(mu_l, mu_u, nu_l, nu_u)`. All methods are those of `IVQROFN`.

    Args:
        mu_l: Lower membership bound in $[0, 1]$.
        mu_u: Upper membership bound in $[0, 1]$.
        nu_l: Lower non-membership bound in $[0, 1]$.
        nu_u: Upper non-membership bound in $[0, 1]$.

    Raises:
        ValueError: If a bound is outside $[0, 1]$, a lower bound exceeds its upper
            bound, or $\\mu_U + \\nu_U > 1$.

    References:
        Atanassov and Gargov (1989), Fuzzy Sets and Systems 31(3), 343-349.

    Examples:
        >>> from bulanik import IVIFN
        >>> IVIFN(0.4, 0.5, 0.3, 0.4)
        IVIFN(mu_l=0.4, mu_u=0.5, nu_l=0.3, nu_u=0.4)
        >>> IVIFN(0.5, 0.4, 0.1, 0.2)
        Traceback (most recent call last):
        ...
        ValueError: need mu_l <= mu_u, got mu_l=0.5, mu_u=0.4
    """

    Q = 1


class IVPFN(_FixedIV):
    """Interval-valued Pythagorean fuzzy number with $\\mu_U^2 + \\nu_U^2 \\le 1$ (q = 2, Peng and Yang 2016).

    An [`IVQROFN`][bulanik.numbers.IVQROFN] with `q = 2`, constructed as
    `IVPFN(mu_l, mu_u, nu_l, nu_u)`.

    Args:
        mu_l: Lower membership bound in $[0, 1]$.
        mu_u: Upper membership bound in $[0, 1]$.
        nu_l: Lower non-membership bound in $[0, 1]$.
        nu_u: Upper non-membership bound in $[0, 1]$.

    Raises:
        ValueError: If a bound is outside $[0, 1]$, a lower bound exceeds its upper
            bound, or $\\mu_U^2 + \\nu_U^2 > 1$.

    References:
        Peng and Yang (2016), International Journal of Intelligent Systems 31(5), 444-487.

    Examples:
        >>> from bulanik import IVPFN
        >>> IVPFN(0.6, 0.8, 0.5, 0.6)
        IVPFN(mu_l=0.6, mu_u=0.8, nu_l=0.5, nu_u=0.6)
    """

    Q = 2


class IVFFN(_FixedIV):
    """Interval-valued Fermatean fuzzy number with $\\mu_U^3 + \\nu_U^3 \\le 1$ (q = 3, Jeevaraj 2021).

    An [`IVQROFN`][bulanik.numbers.IVQROFN] with `q = 3`, constructed as
    `IVFFN(mu_l, mu_u, nu_l, nu_u)`.

    Args:
        mu_l: Lower membership bound in $[0, 1]$.
        mu_u: Upper membership bound in $[0, 1]$.
        nu_l: Lower non-membership bound in $[0, 1]$.
        nu_u: Upper non-membership bound in $[0, 1]$.

    Raises:
        ValueError: If a bound is outside $[0, 1]$, a lower bound exceeds its upper
            bound, or $\\mu_U^3 + \\nu_U^3 > 1$.

    References:
        Jeevaraj (2021), Expert Systems with Applications 185, 115613.

    Examples:
        >>> from bulanik import IVFFN
        >>> IVFFN(0.6, 0.8, 0.5, 0.6)
        IVFFN(mu_l=0.6, mu_u=0.8, nu_l=0.5, nu_u=0.6)
    """

    Q = 3
