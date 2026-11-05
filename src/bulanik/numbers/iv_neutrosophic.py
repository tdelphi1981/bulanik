r"""Interval neutrosophic numbers: IVNeutrosophicFN.

An interval neutrosophic number $\langle [t_L, t_U], [i_L, i_U], [f_L, f_U] \rangle$
carries an interval of truth, of indeterminacy and of falsity, each inside
$[0, 1]$ (Wang et al. 2005). As for single-valued neutrosophic numbers there is
no joint constraint.

Every operation applies the law of [`NeutrosophicFN`][bulanik.numbers.NeutrosophicFN]
to the lower bounds $(t_L, i_L, f_L)$ and to the upper bounds $(t_U, i_U, f_U)$
separately; this is the algebraic arithmetic of Zhang, Wang and Chen (2014).
Each law is monotone in every argument, so the lower bounds stay below the upper
ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .._validate import TOL, choose, unit
from ._base import FuzzyNumber
from .threepart import NeutrosophicFN


@dataclass(frozen=True)
class IVNeutrosophicFN(FuzzyNumber):
    r"""Interval neutrosophic number $\langle [t_L, t_U], [i_L, i_U], [f_L, f_U] \rangle$.

    With $\oplus$, $\otimes$ the laws of
    [`NeutrosophicFN`][bulanik.numbers.NeutrosophicFN] applied to the lower and
    to the upper triple separately:

    | operation | $T$ | $I$ | $F$ |
    |---|---|---|---|
    | `add` | $[t^L_1 + t^L_2 - t^L_1 t^L_2,\ t^U_1 + t^U_2 - t^U_1 t^U_2]$ | $[i^L_1 i^L_2,\ i^U_1 i^U_2]$ | $[f^L_1 f^L_2,\ f^U_1 f^U_2]$ |
    | `mul` | $[t^L_1 t^L_2,\ t^U_1 t^U_2]$ | $[i^L_1 + i^L_2 - i^L_1 i^L_2,\ \dots]$ | $[f^L_1 + f^L_2 - f^L_1 f^L_2,\ \dots]$ |
    | `scale` by $k$ | $[1 - (1 - t_L)^k,\ 1 - (1 - t_U)^k]$ | $[i_L^k,\ i_U^k]$ | $[f_L^k,\ f_U^k]$ |
    | `power` by $k$ | $[t_L^k,\ t_U^k]$ | $[1 - (1 - i_L)^k,\ \dots]$ | $[1 - (1 - f_L)^k,\ \dots]$ |

    Score (`"average"`, the only one, default):
    $S = \tfrac16\left(4 + t_L + t_U - i_L - i_U - f_L - f_U\right)$, the
    `NeutrosophicFN` score $(2 + t - i - f)/3$ on the interval midpoints. Zhang
    et al. (2014, Definition 18) define an interval-valued score
    $[t_L + 2 - i_U - f_U,\ t_U + 2 - i_L - f_L]$; its midpoint is $3S$, so
    `"average"` ranks exactly as the midpoint of that interval does.

    Accuracy: `"difference"` (default) $\tfrac12(t_L + t_U - f_L - f_U)$ or
    `"certainty"` $\tfrac12(t_L + t_U)$ (midpoints of the accuracy and certainty
    intervals of Zhang et al. 2014). Distances (default `"hamming"`) work on the
    six bounds: Hamming $= \tfrac16 \sum |\Delta|$ and Euclidean
    $= \sqrt{\tfrac16 \sum \Delta^2}$. For degenerate intervals all of these
    equal the `NeutrosophicFN` values.

    Inversions $t_L > t_U$ (likewise $i$, $f$) of up to $10^{-12}$ are clamped
    (lower := upper); larger ones raise `ValueError`.

    Args:
        t_l: Lower truth bound in $[0, 1]$.
        t_u: Upper truth bound in $[0, 1]$.
        i_l: Lower indeterminacy bound in $[0, 1]$.
        i_u: Upper indeterminacy bound in $[0, 1]$.
        f_l: Lower falsity bound in $[0, 1]$.
        f_u: Upper falsity bound in $[0, 1]$.

    Raises:
        ValueError: If a bound is outside $[0, 1]$ or not finite, or a lower bound
            exceeds its upper bound beyond rounding.
        TypeError: If a bound is a `bool` or a string.

    Attributes:
        t_l: Lower truth bound.
        t_u: Upper truth bound.
        i_l: Lower indeterminacy bound.
        i_u: Upper indeterminacy bound.
        f_l: Lower falsity bound.
        f_u: Upper falsity bound.

    References:
        Wang, Smarandache, Zhang and Sunderraman (2005), Interval Neutrosophic
        Sets and Logic: Theory and Applications in Computing, Hexis, Phoenix
        (definition, complement). Zhang, Wang and Chen (2014), The Scientific
        World Journal 2014, article 645953, doi:10.1155/2014/645953 (operations,
        score, INNWA and INNWG operators). The paper prints the addition and
        product with two typos (a $T$ in place of an $I$ index); the code follows
        the general t-norm forms of the same paper.

    Examples:
        >>> from bulanik import IVNeutrosophicFN
        >>> a = IVNeutrosophicFN(0.6, 0.7, 0.1, 0.2, 0.2, 0.3)
        >>> round(a.score(), 4), round(a.accuracy(), 4), round(a.accuracy("certainty"), 4)
        (0.75, 0.4, 0.65)
        >>> a.complement()
        IVNeutrosophicFN(t_l=0.2, t_u=0.3, i_l=0.8, i_u=0.9, f_l=0.6, f_u=0.7)
        >>> IVNeutrosophicFN(0.8, 0.7, 0.1, 0.2, 0.2, 0.3)
        Traceback (most recent call last):
        ...
        ValueError: need t_l <= t_u, got t_l=0.8, t_u=0.7
    """

    t_l: float
    t_u: float
    i_l: float
    i_u: float
    f_l: float
    f_u: float

    OPS = ("algebraic",)
    SCORES = ("average",)
    DEFAULT_SCORE = "average"
    ACCURACIES = ("difference", "certainty")
    DEFAULT_ACCURACY = "difference"
    DISTANCES = ("hamming", "euclidean")
    DEFAULT_DISTANCE = "hamming"

    def __post_init__(self) -> None:
        vals = {n: unit(getattr(self, n), n) for n in self._FIELDS}
        for part in "tif":
            lo, hi = vals[f"{part}_l"], vals[f"{part}_u"]
            if lo > hi:
                if lo - hi > TOL:
                    raise ValueError(
                        f"need {part}_l <= {part}_u, got {part}_l={lo}, {part}_u={hi}"
                    )
                vals[f"{part}_l"] = hi
        for n, v in vals.items():
            object.__setattr__(self, n, v)

    _FIELDS = ("t_l", "t_u", "i_l", "i_u", "f_l", "f_u")

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_neutrosophic(cls, x: NeutrosophicFN) -> "IVNeutrosophicFN":
        """Degenerate interval $\\langle [t, t], [i, i], [f, f] \\rangle$ of a `NeutrosophicFN`.

        Args:
            x: A single-valued neutrosophic number.

        Returns:
            The interval number with equal lower and upper bounds.

        Raises:
            TypeError: If `x` is not a `NeutrosophicFN`.

        Examples:
            >>> from bulanik import IVNeutrosophicFN, NeutrosophicFN
            >>> IVNeutrosophicFN.from_neutrosophic(NeutrosophicFN(0.7, 0.2, 0.1))
            IVNeutrosophicFN(t_l=0.7, t_u=0.7, i_l=0.2, i_u=0.2, f_l=0.1, f_u=0.1)
        """
        if not isinstance(x, NeutrosophicFN):
            raise TypeError(f"from_neutrosophic needs a NeutrosophicFN, got {type(x).__name__}")
        return cls(x.t, x.t, x.i, x.i, x.f, x.f)

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """The six bounds $(t_L, t_U, i_L, i_U, f_L, f_U)$.

        Examples:
            >>> from bulanik import IVNeutrosophicFN
            >>> IVNeutrosophicFN(0.6, 0.7, 0.1, 0.2, 0.2, 0.3).bounds
            (0.6, 0.7, 0.1, 0.2, 0.2, 0.3)
        """
        return tuple(getattr(self, n) for n in self._FIELDS)  # type: ignore[return-value]

    def _ends(self) -> tuple[NeutrosophicFN, NeutrosophicFN]:
        return (
            NeutrosophicFN(self.t_l, self.i_l, self.f_l),
            NeutrosophicFN(self.t_u, self.i_u, self.f_u),
        )

    @staticmethod
    def _join(lo: NeutrosophicFN, hi: NeutrosophicFN) -> "IVNeutrosophicFN":
        return IVNeutrosophicFN(lo.t, hi.t, lo.i, hi.i, lo.f, hi.f)

    def complement(self) -> "IVNeutrosophicFN":
        r"""Complement $([f_L, f_U], [1 - i_U, 1 - i_L], [t_L, t_U])$.

        Returns:
            The complement (Wang et al. 2005).

        Examples:
            >>> from bulanik import IVNeutrosophicFN
            >>> IVNeutrosophicFN(0.6, 0.7, 0.1, 0.2, 0.2, 0.3).complement().i_l
            0.8
        """
        return IVNeutrosophicFN(
            self.f_l, self.f_u, 1.0 - self.i_u, 1.0 - self.i_l, self.t_l, self.t_u
        )

    # -- arithmetic ----------------------------------------------------------
    def _binary(self, name: str, other: Any, ops: str) -> "IVNeutrosophicFN":
        self._check_ops(ops)
        self.same_kind(other)
        (l1, u1), (l2, u2) = self._ends(), other._ends()
        f = NeutrosophicFN.add if name == "add" else NeutrosophicFN.mul
        return self._join(f(l1, l2), f(u1, u2))

    def _unary(self, name: str, k: float, ops: str) -> "IVNeutrosophicFN":
        self._check_ops(ops)
        lo, hi = self._ends()
        f = NeutrosophicFN.scale if name == "scale" else NeutrosophicFN.power
        return self._join(f(lo, k), f(hi, k))

    def add(self, other: Any, ops: str = "algebraic") -> "IVNeutrosophicFN":
        """Sum of two interval neutrosophic numbers (law of `NeutrosophicFN` on both ends).

        `A + B` calls this method.

        Args:
            other: An `IVNeutrosophicFN`.
            ops: Only `"algebraic"` exists.

        Returns:
            The sum.

        Raises:
            TypeError: If `other` is of a different class.
            ValueError: If `ops` is unknown.

        References:
            Zhang, Wang and Chen (2014), The Scientific World Journal 2014, 645953.

        Examples:
            >>> from bulanik import IVNeutrosophicFN as N
            >>> s = N(0.6, 0.7, 0.1, 0.2, 0.2, 0.3) + N(0.5, 0.6, 0.2, 0.3, 0.1, 0.2)
            >>> tuple(round(x, 4) for x in s.bounds)
            (0.8, 0.88, 0.02, 0.06, 0.02, 0.06)
        """
        return self._binary("add", other, ops)

    def mul(self, other: Any, ops: str = "algebraic") -> "IVNeutrosophicFN":
        """Product of two interval neutrosophic numbers (law of `NeutrosophicFN` on both ends).

        `A * B` calls this method.

        Args:
            other: An `IVNeutrosophicFN`.
            ops: Only `"algebraic"` exists.

        Returns:
            The product.

        Raises:
            TypeError: If `other` is of a different class.
            ValueError: If `ops` is unknown.

        Examples:
            >>> from bulanik import IVNeutrosophicFN as N
            >>> p = N(0.6, 0.7, 0.1, 0.2, 0.2, 0.3) * N(0.5, 0.6, 0.2, 0.3, 0.1, 0.2)
            >>> tuple(round(x, 4) for x in p.bounds)
            (0.3, 0.42, 0.28, 0.44, 0.28, 0.44)
        """
        return self._binary("mul", other, ops)

    def scale(self, k: float, ops: str = "algebraic") -> "IVNeutrosophicFN":
        """Scalar multiple ``k * A`` (law of `NeutrosophicFN` on both ends).

        Args:
            k: A finite scalar $\\ge 0$.
            ops: Only `"algebraic"` exists.

        Returns:
            The scaled number.

        Raises:
            ValueError: If `k` is negative or not finite, or `ops` is unknown.

        Examples:
            >>> from bulanik import IVNeutrosophicFN as N
            >>> tuple(round(x, 4) for x in N(0.6, 0.7, 0.1, 0.2, 0.2, 0.3).scale(2).bounds)
            (0.84, 0.91, 0.01, 0.04, 0.04, 0.09)
        """
        return self._unary("scale", k, ops)

    def power(self, k: float, ops: str = "algebraic") -> "IVNeutrosophicFN":
        """Power ``A ** k`` (law of `NeutrosophicFN` on both ends).

        Args:
            k: A finite exponent $\\ge 0$.
            ops: Only `"algebraic"` exists.

        Returns:
            The power.

        Raises:
            ValueError: If `k` is negative or not finite, or `ops` is unknown.

        Examples:
            >>> from bulanik import IVNeutrosophicFN as N
            >>> tuple(round(x, 4) for x in (N(0.6, 0.7, 0.1, 0.2, 0.2, 0.3) ** 2).bounds)
            (0.36, 0.49, 0.19, 0.36, 0.36, 0.51)
        """
        return self._unary("power", k, ops)

    # -- scores, accuracy, distance -------------------------------------------
    def score(self, method: str | None = None) -> float:
        r"""Score $\tfrac16(4 + t_L + t_U - i_L - i_U - f_L - f_U)$ (`"average"`, the only method).

        Args:
            method: `"average"` or `None`.

        Returns:
            The score in $[0, 1]$.

        Raises:
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import IVNeutrosophicFN
            >>> round(IVNeutrosophicFN(0.6, 0.7, 0.1, 0.2, 0.2, 0.3).score(), 4)
            0.75
        """
        choose(method, self.SCORES, "score", type(self).__name__, self.DEFAULT_SCORE)
        return (
            4.0 + self.t_l + self.t_u - self.i_l - self.i_u - self.f_l - self.f_u
        ) / 6.0

    def accuracy(self, method: str | None = None) -> float:
        r"""Accuracy $\tfrac12(t_L + t_U - f_L - f_U)$ (`"difference"`, default) or $\tfrac12(t_L + t_U)$ (`"certainty"`).

        Args:
            method: `"difference"`, `"certainty"` or `None`.

        Returns:
            The accuracy.

        Raises:
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import IVNeutrosophicFN
            >>> a = IVNeutrosophicFN(0.6, 0.7, 0.1, 0.2, 0.2, 0.3)
            >>> round(a.accuracy(), 4), round(a.accuracy("certainty"), 4)
            (0.4, 0.65)
        """
        method = choose(
            method, self.ACCURACIES, "accuracy", type(self).__name__, self.DEFAULT_ACCURACY
        )
        t = (self.t_l + self.t_u) / 2.0
        return t if method == "certainty" else t - (self.f_l + self.f_u) / 2.0

    def distance(self, other: Any, method: str | None = None) -> float:
        r"""Normalized Hamming (default) $\tfrac16 \sum |\Delta|$ or Euclidean $\sqrt{\tfrac16 \sum \Delta^2}$ over the six bounds.

        Args:
            other: An `IVNeutrosophicFN`.
            method: `"hamming"` or `"euclidean"`; `None` means `"hamming"`.

        Returns:
            The distance, in $[0, 1]$.

        Raises:
            TypeError: If `other` is of a different class.
            ValueError: If `method` is unknown.

        Examples:
            >>> from bulanik import IVNeutrosophicFN as N
            >>> a, b = N(0.6, 0.7, 0.1, 0.2, 0.2, 0.3), N(0.5, 0.6, 0.2, 0.3, 0.1, 0.2)
            >>> round(a.distance(b), 4)
            0.1
        """
        method = choose(
            method, self.DISTANCES, "distance", type(self).__name__, self.DEFAULT_DISTANCE
        )
        self.same_kind(other)
        d = [x - y for x, y in zip(self.bounds, other.bounds)]
        if method == "euclidean":
            return math.sqrt(sum(x * x for x in d) / 6.0)
        return sum(abs(x) for x in d) / 6.0
