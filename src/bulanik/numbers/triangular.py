"""Triangular ([`TFN`][bulanik.numbers.TFN]) and trapezoidal ([`TrFN`][bulanik.numbers.TrFN]) fuzzy numbers.

Both are defined by ordered points on the real line: three for a triangle
$(l, m, u)$, four for a trapezoid $(a, b, c, d)$. The arithmetic is the usual
one on the points (the standard approximate operators of fuzzy arithmetic), and
the only operator set is `"algebraic"`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .._validate import choose, finite, unit
from ._base import FuzzyNumber


def _is_real(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class _Piecewise(FuzzyNumber):
    r"""Shared arithmetic of numbers defined by ordered points (3 or 4).

    The methods documented here are inherited by [`TFN`][bulanik.numbers.TFN]
    and [`TrFN`][bulanik.numbers.TrFN].

    Scores: `"centroid"` (default), `"graded_mean"`, `"mean"`. Distance:
    `"vertex"` (default). Triangular and trapezoidal numbers have no accuracy
    function.
    """

    OPS = ("algebraic",)
    SCORES = ("centroid", "graded_mean", "mean")
    DEFAULT_SCORE = "centroid"
    ACCURACIES = ()
    DEFAULT_ACCURACY = None
    DISTANCES = ("vertex",)
    DEFAULT_DISTANCE = "vertex"

    # -- hooks implemented by the concrete classes -------------------------
    def _pts(self) -> tuple[float, ...]:
        raise NotImplementedError

    @classmethod
    def _make(cls, pts: tuple[float, ...]) -> Any:
        return cls(*pts)

    def _core(self) -> tuple[float, float]:
        raise NotImplementedError

    def _validate_points(self) -> None:
        names = self.__dataclass_fields__  # type: ignore[attr-defined]
        pts = [finite(getattr(self, n), n) for n in names]
        for n, v in zip(names, pts):
            object.__setattr__(self, n, v)
        if any(a > b for a, b in zip(pts, pts[1:])):
            raise ValueError(
                f"{type(self).__name__} needs {' <= '.join(names)}, got {tuple(pts)}"
            )

    def __post_init__(self) -> None:
        self._validate_points()

    # -- arithmetic ----------------------------------------------------------
    def add(self, other: Any, ops: str = "algebraic") -> Any:
        r"""Sum, pointwise: $(x_1, \dots, x_n) + (y_1, \dots, y_n) = (x_i + y_i)$.

        Args:
            other: A number of the same type.
            ops: Operator set; only `"algebraic"` exists.

        Returns:
            The sum, of the same type.

        Raises:
            TypeError: If `other` is not the same type.
            ValueError: If `ops` is not `"algebraic"`.

        Examples:
            >>> from bulanik import TFN
            >>> TFN(1, 2, 4).add(TFN(2, 3, 5))
            TFN(l=3.0, m=5.0, u=9.0)
        """
        self._check_ops(ops)
        self.same_kind(other)
        return self._make(tuple(x + y for x, y in zip(self._pts(), other._pts())))

    def sub(self, other: Any, ops: str = "algebraic") -> Any:
        r"""Difference $A - B$, pairing each point of $A$ with the opposite point of $B$.

        For a triangle, $(l_1, m_1, u_1) - (l_2, m_2, u_2) = (l_1 - u_2,\; m_1 - m_2,\; u_1 - l_2)$.
        The operator `A - B` calls this method.

        Args:
            other: A number of the same type.
            ops: Operator set; only `"algebraic"` exists.

        Returns:
            The difference, of the same type.

        Raises:
            TypeError: If `other` is not the same type.
            ValueError: If `ops` is not `"algebraic"`.

        Examples:
            >>> from bulanik import TFN
            >>> TFN(1, 2, 4) - TFN(2, 3, 5)
            TFN(l=-4.0, m=-1.0, u=2.0)
        """
        self._check_ops(ops)
        self.same_kind(other)
        return self._make(
            tuple(x - y for x, y in zip(self._pts(), reversed(other._pts())))
        )

    def mul(self, other: Any, ops: str = "algebraic") -> Any:
        r"""Product, the standard approximate (point-based) multiplication.

        The outer points are the smallest and largest of the four products of
        the two outer points of each operand, and the inner points are the
        products of the inner points (sorted). For non-negative numbers this is
        the componentwise product $(l_1 l_2,\; m_1 m_2,\; u_1 u_2)$.

        Args:
            other: A number of the same type.
            ops: Operator set; only `"algebraic"` exists.

        Returns:
            The product, of the same type.

        Raises:
            TypeError: If `other` is not the same type.
            ValueError: If `ops` is not `"algebraic"`.

        Examples:
            >>> from bulanik import TFN
            >>> TFN(1, 2, 4) * TFN(2, 3, 5)
            TFN(l=2.0, m=6.0, u=20.0)
        """
        self._check_ops(ops)
        self.same_kind(other)
        p, q = self._pts(), other._pts()
        corners = [p[0] * q[0], p[0] * q[-1], p[-1] * q[0], p[-1] * q[-1]]
        inner = sorted(x * y for x, y in zip(p[1:-1], q[1:-1]))
        return self._make((min(corners), *inner, max(corners)))

    def scale(self, k: float, ops: str = "algebraic") -> Any:
        r"""Multiply every point by the real number `k`.

        A negative `k` also reverses the order of the points, so the result
        stays ordered. `k * A`, `A * k` and `A / k` call this method.

        Args:
            k: The scalar.
            ops: Operator set; only `"algebraic"` exists.

        Returns:
            The scaled number, of the same type.

        Raises:
            TypeError: If `k` is not a real number (`bool` and `str` are rejected).
            ValueError: If `k` is not finite, or `ops` is not `"algebraic"`.

        Examples:
            >>> from bulanik import TFN
            >>> TFN(1, 2, 4).scale(2)
            TFN(l=2.0, m=4.0, u=8.0)
            >>> TFN(1, 2, 4).scale(-1)
            TFN(l=-4.0, m=-2.0, u=-1.0)
        """
        self._check_ops(ops)
        k = finite(k, "k")
        pts = self._pts()
        if k < 0:
            pts = pts[::-1]
        return self._make(tuple(k * x for x in pts))

    def power(self, k: float, ops: str = "algebraic") -> Any:
        r"""Raise every point to the real power `k`; `A ** k` calls this method.

        For a negative `k` the order of the points is reversed. The lowest point
        must be non-negative (strictly positive when `k < 0`).

        Args:
            k: The exponent.
            ops: Operator set; only `"algebraic"` exists.

        Returns:
            The power, of the same type.

        Raises:
            TypeError: If `k` is not a real number (`bool` and `str` are rejected).
            ValueError: If `k` is not finite, `ops` is not `"algebraic"`, or the
                lowest point is negative (or zero with `k < 0`).

        Examples:
            >>> from bulanik import TFN
            >>> TFN(1, 2, 4) ** 2
            TFN(l=1.0, m=4.0, u=16.0)
            >>> TFN(-1, 2, 4).power(2)
            Traceback (most recent call last):
            ...
            ValueError: power 2.0 needs the lowest point >= 0, got -1.0
        """
        self._check_ops(ops)
        k = finite(k, "k")
        pts = self._pts()
        low = pts[0]
        if low < 0 or (k < 0 and low == 0):
            need = "> 0" if k < 0 else ">= 0"
            raise ValueError(f"power {k} needs the lowest point {need}, got {low}")
        if k < 0:
            pts = pts[::-1]
        return self._make(tuple(x**k for x in pts))

    def reciprocal(self) -> Any:
        r"""Reciprocal $1 / A$, taking $1/x$ of every point in reversed order.

        `A / B` computes `A * B.reciprocal()`.

        Returns:
            The reciprocal, of the same type.

        Raises:
            ValueError: If the support $[\text{lowest}, \text{highest}]$ contains 0.

        Examples:
            >>> from bulanik import TFN
            >>> TFN(1, 2, 4).reciprocal()
            TFN(l=0.25, m=0.5, u=1.0)
            >>> TFN(-1, 0, 1).reciprocal()
            Traceback (most recent call last):
            ...
            ValueError: cannot take the reciprocal of TFN(l=-1.0, m=0.0, u=1.0): its support contains 0
        """
        pts = self._pts()
        if pts[0] <= 0 <= pts[-1]:
            raise ValueError(
                f"cannot take the reciprocal of {self}: its support contains 0"
            )
        return self._make(tuple(1.0 / x for x in reversed(pts)))

    def __sub__(self, other: Any) -> Any:
        if not self._is_same(other):
            return NotImplemented
        return self.sub(other)

    def __truediv__(self, other: Any) -> Any:
        if self._is_same(other):
            return self.mul(other.reciprocal())
        if _is_real(other):
            if other == 0:
                raise ValueError(f"cannot divide {self} by zero")
            return self.scale(1.0 / other)
        return NotImplemented

    def __neg__(self) -> Any:
        return self.scale(-1.0)

    # -- membership ----------------------------------------------------------
    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        r"""The alpha-cut $\{x : \mu(x) \ge \alpha\}$ as an interval.

        With support $[p_0, p_n]$ and core $[c_\text{lo}, c_\text{hi}]$ the cut is
        $[p_0 + \alpha (c_\text{lo} - p_0),\; p_n - \alpha (p_n - c_\text{hi})]$.

        Args:
            alpha: Membership level in $[0, 1]$.

        Returns:
            The pair `(lower, upper)`.

        Raises:
            ValueError: If `alpha` is outside [0, 1] or not finite.

        Examples:
            >>> from bulanik import TFN
            >>> TFN(1, 2, 4).alpha_cut(0.5)
            (1.5, 3.0)
            >>> TFN(1, 2, 4).alpha_cut(1)
            (2.0, 2.0)
        """
        a = unit(alpha, "alpha")
        pts = self._pts()
        core_lo, core_hi = self._core()
        lo = pts[0] + a * (core_lo - pts[0])
        hi = pts[-1] - a * (pts[-1] - core_hi)
        return (min(lo, core_lo), max(hi, core_hi))

    def mu(self, x: float) -> float:
        r"""Membership degree $\mu(x)$: piecewise linear, 1 on the core, 0 outside the support.

        Args:
            x: A real number.

        Returns:
            The membership degree in $[0, 1]$.

        Raises:
            ValueError: If `x` is NaN or infinite.

        Examples:
            >>> from bulanik import TFN, TrFN
            >>> TFN(1, 2, 4).mu(1.5)
            0.5
            >>> TrFN(0, 1, 2, 4).mu(1.5)
            1.0
        """
        x = finite(x, "x")
        pts = self._pts()
        core_lo, core_hi = self._core()
        if x < pts[0] or x > pts[-1]:
            return 0.0
        if core_lo <= x <= core_hi:
            return 1.0
        if x < core_lo:
            return (x - pts[0]) / (core_lo - pts[0])
        return (pts[-1] - x) / (pts[-1] - core_hi)

    # -- scores, accuracy, distance -------------------------------------------
    def _defuzzify(self, method: str) -> float:
        raise NotImplementedError

    def defuzzify(self, method: str = "centroid") -> float:
        r"""Crisp value of the number (the same functions as [`score`][bulanik.numbers.TFN.score]).

        Methods: `"centroid"` (centre of gravity), `"graded_mean"`
        (graded mean integration) and `"mean"` (arithmetic mean
        of the points; equal to the centroid for a triangle).

        Args:
            method: One of `"centroid"`, `"graded_mean"`, `"mean"`.

        Returns:
            The crisp value.

        Raises:
            ValueError: If `method` is not one of the three names.

        Examples:
            >>> from bulanik import TFN
            >>> round(TFN(1, 2, 4).defuzzify("graded_mean"), 4)
            2.1667
        """
        method = choose(method, self.SCORES, "score", type(self).__name__)
        return self._defuzzify(method)

    def score(self, method: str | None = None) -> float:
        r"""Crisp score for ranking; same as [`defuzzify`][bulanik.numbers.TFN.defuzzify], default `"centroid"`.

        For a triangle $(l, m, u)$: centroid $= (l + m + u)/3$ (also the mean),
        graded mean $= (l + 4m + u)/6$. For a trapezoid $(a, b, c, d)$:
        graded mean $= (a + 2b + 2c + d)/6$, mean $= (a + b + c + d)/4$ and
        centroid $= \dfrac{d^2 + c^2 + cd - a^2 - b^2 - ab}{3\,(d + c - a - b)}$
        (the common point `a` if all four points coincide).

        Args:
            method: One of `"centroid"`, `"graded_mean"`, `"mean"`; `None`
                means `"centroid"`.

        Returns:
            The score.

        Raises:
            ValueError: If `method` is not one of the three names.

        Examples:
            >>> from bulanik import TFN, TrFN
            >>> round(TFN(1, 2, 4).score(), 4)
            2.3333
            >>> TrFN(1, 2, 3, 5).score("mean")
            2.75
        """
        method = choose(
            method, self.SCORES, "score", type(self).__name__, self.DEFAULT_SCORE
        )
        return self._defuzzify(method)

    def accuracy(self, method: str | None = None) -> float:
        """Not defined for triangular and trapezoidal numbers.

        Args:
            method: Ignored.

        Raises:
            ValueError: Always.

        Examples:
            >>> from bulanik import TFN
            >>> TFN(1, 2, 4).accuracy()
            Traceback (most recent call last):
            ...
            ValueError: TFN has no accuracy function
        """
        raise ValueError(f"{type(self).__name__} has no accuracy function")

    def distance(self, other: Any, method: str | None = None) -> float:
        r"""Vertex distance (Chen 2000): $\sqrt{\tfrac{1}{n}\sum_i (x_i - y_i)^2}$ over the $n$ points.

        Args:
            other: A number of the same type.
            method: Only `"vertex"` exists (the default).

        Returns:
            The distance.

        Raises:
            TypeError: If `other` is not the same type.
            ValueError: If `method` is not `"vertex"`.

        References:
            Chen (2000), Fuzzy Sets and Systems 114: the vertex method.

        Examples:
            >>> from bulanik import TFN
            >>> TFN(1, 2, 4).distance(TFN(2, 3, 5))
            1.0
        """
        choose(method, self.DISTANCES, "distance", type(self).__name__, self.DEFAULT_DISTANCE)
        self.same_kind(other)
        p, q = self._pts(), other._pts()
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(p, q)) / len(p))


@dataclass(frozen=True)
class TFN(_Piecewise):
    r"""Triangular fuzzy number $(l, m, u)$ with $l \le m \le u$.

    The membership rises linearly from 0 at `l` to 1 at `m` and falls to 0 at
    `u`. Points are converted to floats. Arithmetic, scores and membership are
    documented on the shared methods (`add`, `sub`, `mul`, `scale`, `power`,
    `reciprocal`, `alpha_cut`, `mu`, `score`, `defuzzify`, `distance`); the
    operators `+ - * /` and `**` and unary `-` call them.

    Args:
        l: Lower foot.
        m: Peak (core).
        u: Upper foot.

    Raises:
        ValueError: If a point is NaN or infinite, or the points are not
            ordered `l <= m <= u`.
        TypeError: If a point is a `bool` or a string.

    Attributes:
        l: Lower foot.
        m: Peak.
        u: Upper foot.

    Examples:
        >>> from bulanik import TFN
        >>> a = TFN(1, 2, 4)
        >>> a
        TFN(l=1.0, m=2.0, u=4.0)
        >>> a / TFN(2, 3, 5)
        TFN(l=0.2, m=0.6666666666666666, u=2.0)
        >>> TFN(3, 2, 1)
        Traceback (most recent call last):
        ...
        ValueError: TFN needs l <= m <= u, got (3.0, 2.0, 1.0)
    """

    l: float
    m: float
    u: float

    def _pts(self) -> tuple[float, ...]:
        return (self.l, self.m, self.u)

    def _core(self) -> tuple[float, float]:
        return (self.m, self.m)

    def _defuzzify(self, method: str) -> float:
        if method == "graded_mean":
            return (self.l + 4 * self.m + self.u) / 6
        # centroid and mean coincide for a triangle
        return (self.l + self.m + self.u) / 3

    def to_trfn(self) -> "TrFN":
        """Convert to the equivalent trapezoid $(l, m, m, u)$.

        Returns:
            A [`TrFN`][bulanik.numbers.TrFN] with a one-point core.

        Examples:
            >>> from bulanik import TFN
            >>> TFN(1, 2, 4).to_trfn()
            TrFN(a=1.0, b=2.0, c=2.0, d=4.0)
        """
        return TrFN(self.l, self.m, self.m, self.u)


@dataclass(frozen=True)
class TrFN(_Piecewise):
    r"""Trapezoidal fuzzy number $(a, b, c, d)$ with $a \le b \le c \le d$.

    The membership rises from 0 at `a` to 1 at `b`, stays 1 up to `c` and falls
    to 0 at `d`. It offers the same methods and operators as
    [`TFN`][bulanik.numbers.TFN].

    Args:
        a: Lower foot.
        b: Start of the core.
        c: End of the core.
        d: Upper foot.

    Raises:
        ValueError: If a point is NaN or infinite, or the points are not ordered.
        TypeError: If a point is a `bool` or a string.

    Attributes:
        a: Lower foot.
        b: Start of the core.
        c: End of the core.
        d: Upper foot.

    Examples:
        >>> from bulanik import TrFN
        >>> t = TrFN(1, 2, 3, 5)
        >>> t.alpha_cut(0.5)
        (1.5, 4.0)
        >>> round(t.score(), 4)
        2.8
        >>> t.accuracy()
        Traceback (most recent call last):
        ...
        ValueError: TrFN has no accuracy function
    """

    a: float
    b: float
    c: float
    d: float

    def _pts(self) -> tuple[float, ...]:
        return (self.a, self.b, self.c, self.d)

    def _core(self) -> tuple[float, float]:
        return (self.b, self.c)

    def _defuzzify(self, method: str) -> float:
        a, b, c, d = self._pts()
        if method == "graded_mean":
            return (a + 2 * b + 2 * c + d) / 6
        if method == "mean":
            return (a + b + c + d) / 4
        den = 3 * (d + c - a - b)
        if den == 0:  # only when all four points coincide
            return a
        return (d * d + c * c + c * d - a * a - b * b - a * b) / den
