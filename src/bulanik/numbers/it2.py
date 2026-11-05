r"""Interval type-2 trapezoidal fuzzy numbers: IT2TrFN.

An interval type-2 trapezoidal fuzzy number (Chen and Lee 2010) is a pair of
generalised trapezoids, the upper and the lower membership function of the
footprint of uncertainty. Each trapezoid is $(a_1, a_2, a_3, a_4; h_1, h_2)$:
the membership rises from 0 at $a_1$ to $h_1$ at $a_2$, goes linearly to $h_2$
at $a_3$ and falls to 0 at $a_4$. The lower membership never exceeds the upper
one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .._validate import TOL, choose, finite, unit
from ._base import FuzzyNumber
from .triangular import TFN, TrFN

_Six = tuple[float, float, float, float, float, float]


def _stdev(xs: Sequence[float]) -> float:
    """Population standard deviation."""
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _clean(t: Any, which: str) -> _Six:
    try:
        n = len(t)
    except TypeError:
        raise TypeError(f"{which} must be a sequence of 6 numbers, got {type(t).__name__}") from None
    if n != 6:
        raise ValueError(f"{which} needs 6 values (a1, a2, a3, a4, h1, h2), got {n}")
    pts = [finite(v, f"{which}[{i}]") for i, v in enumerate(t[:4])]
    if any(p > q for p, q in zip(pts, pts[1:])):
        raise ValueError(f"IT2TrFN needs a1 <= a2 <= a3 <= a4 in the {which} trapezoid, got {tuple(pts)}")
    h = [unit(v, f"height h{i + 1} of the {which} trapezoid") for i, v in enumerate(t[4:])]
    return (*pts, *h)  # type: ignore[return-value]


@dataclass(frozen=True)
class IT2TrFN(FuzzyNumber):
    r"""Interval type-2 trapezoidal fuzzy number $(\tilde A^U, \tilde A^L)$.

    Each of `upper` and `lower` holds $(a_1, a_2, a_3, a_4, h_1, h_2)$ with
    $a_1 \le a_2 \le a_3 \le a_4$; $h_1$ is the membership at $a_2$ and $h_2$ the
    membership at $a_3$. Following Chen and Lee (2010a) every operation applies
    the type-1 rule to the upper and to the lower trapezoid, and the heights
    combine with `min`:

    - sum: points add; difference: $(a_1 - b_4, a_2 - b_3, a_3 - b_2, a_4 - b_1)$;
    - product: componentwise, for non-negative points;
    - scalar $k$: points times $k$, heights unchanged (for $k < 0$ the points
      reverse and $h_1, h_2$ swap);
    - power $k$: points to the power $k$, heights unchanged (for $k < 0$ the
      points reverse and the heights swap). The product and power give the
      geometric mean $(\prod \tilde a_i)^{1/n}$ of Kahraman et al. (2014).

    Scores (default `"dtrat"`):

    - `"dtrat"`, the defuzzification DTraT of Kahraman et al. (2014),
      $\tfrac12 \sum_{X \in \{U, L\}} \Big[\dfrac{(a^X_4 - a^X_1) + (h^X_1 a^X_2 - a^X_1) + (h^X_2 a^X_3 - a^X_1)}{4} + a^X_1\Big]$;
      for a type-1 trapezoid (heights 1, upper = lower) it is the mean of the
      four points.
    - `"centroid"`: the mean of the x-centroids of the upper and the lower
      polygon $(a_1, 0), (a_2, h_1), (a_3, h_2), (a_4, 0)$ (shoelace formula; the
      mean of the points if the polygon has zero area). This is bulanik's own
      choice and not the Karnik-Mendel centroid of the type-2 set.
    - `"chen_lee"`, the ranking value of Chen and Lee (2010a),
      $\sum_{X \in \{U, L\}} \Big[\sum_{p=1}^{3} M_p(X) - \tfrac14 \sum_{q=1}^{4} S_q(X) + h_1(X) + h_2(X)\Big]$
      with $M_p = (a_p + a_{p+1})/2$, $S_q$ ($q \le 3$) the population standard
      deviation of $\{a_q, a_{q+1}\}$ and $S_4$ that of $\{a_1, \dots, a_4\}$.

    There is no accuracy function. The only distance is `"vertex"`,
    $\sqrt{\tfrac18 \sum (\Delta a)^2}$ over the 8 points of the two trapezoids;
    heights are ignored.

    Validation: points are finite and ordered, heights lie in $[0, 1]$ (out by at
    most $10^{-12}$ they are projected) and the upper trapezoid has
    $\max(h_1, h_2) > 0$. The lower trapezoid must be *nested* in the upper one
    (up to $10^{-12}$):
    $a^U_1 \le a^L_1 \le a^L_2 \le a^L_3 \le a^L_4 \le a^U_4$,
    $a^U_2 \le a^L_2$, $a^L_3 \le a^U_3$ and
    $h^L_1, h^L_2 \le \min(h^U_1, h^U_2)$. The violated inequality is named in the
    `ValueError`. Nesting implies that the lower membership never exceeds the upper
    one (on each rising or falling edge both curves are linear up to the upper
    core, where the upper membership is at least $\min(h^U_1, h^U_2) \ge h^L$), and
    it is *preserved* by the sum, difference, non-negative product, scalar and
    power (each acts monotonically on the reference points, and the heights take
    `min`), so arithmetic never leaves the class. General containment is slightly
    weaker, but the sum of two merely contained numbers can leave the footprint
    constraint, so it is not enough for a closed arithmetic. The nesting is the
    usual assumption of the interval type-2 MCDM literature (the linguistic scales
    of Chen and Lee 2010b and Kahraman et al. 2014 satisfy it). The lower heights
    are compared with $\min(h^U_1, h^U_2)$ and not point by point, because the
    upper membership between the two inner points is only known to be at least
    that minimum.

    Args:
        upper: The six values of the upper trapezoid.
        lower: The six values of the lower trapezoid.

    Raises:
        ValueError: If a sequence does not have six values, a value is NaN or
            infinite, the points are unordered, a height is outside $[0, 1]$, the
            upper heights are both 0, or the lower membership exceeds the upper.
        TypeError: If a value is a `bool` or a string.

    Attributes:
        upper: The upper trapezoid $(a_1, a_2, a_3, a_4, h_1, h_2)$.
        lower: The lower trapezoid.

    References:
        Chen and Lee (2010a), Expert Systems with Applications 37(1), 824-833,
        doi:10.1016/j.eswa.2009.06.094 (arithmetic and ranking value). Chen and
        Lee (2010b), Expert Systems with Applications 37(4), 2790-2798,
        doi:10.1016/j.eswa.2009.09.012 (IT2 TOPSIS, linguistic scale). Kahraman,
        Oztaysi, Ucal Sari and Turanoglu (2014), Knowledge-Based Systems 59,
        48-57, doi:10.1016/j.knosys.2014.02.001 (DTraT, IT2 AHP). Mendel, John and
        Liu (2006), IEEE Transactions on Fuzzy Systems 14(6), 808-821
        (terminology).

    Examples:
        >>> from bulanik import IT2TrFN
        >>> low = IT2TrFN((0, 0.1, 0.1, 0.3, 1, 1), (0.05, 0.1, 0.1, 0.2, 0.9, 0.9))
        >>> (low + low).upper
        (0.0, 0.2, 0.2, 0.6, 1.0, 1.0)
        >>> round(low.score(), 5)
        0.11625
        >>> IT2TrFN((0, 1, 2, 3, 0.5, 0.5), (0.5, 1, 2, 2.5, 0.9, 0.9))
        Traceback (most recent call last):
        ...
        ValueError: IT2TrFN needs nested reference points: h1L <= min(h1U, h2U), got 0.9 > 0.5
    """

    upper: _Six
    lower: _Six

    OPS = ("algebraic",)
    SCORES = ("dtrat", "centroid", "chen_lee")
    DEFAULT_SCORE = "dtrat"
    ACCURACIES = ()
    DEFAULT_ACCURACY = None
    DISTANCES = ("vertex",)
    DEFAULT_DISTANCE = "vertex"

    def __post_init__(self) -> None:
        up, lo = _clean(self.upper, "upper"), _clean(self.lower, "lower")
        if max(up[4], up[5]) <= 0:
            raise ValueError("the upper trapezoid needs max(h1, h2) > 0")
        checks = (
            ("a1U <= a1L", up[0], lo[0]),
            ("a2U <= a2L", up[1], lo[1]),
            ("a1L <= a2L", lo[0], lo[1]),
            ("a3L <= a3U", lo[2], up[2]),
            ("a4L <= a4U", lo[3], up[3]),
        )
        for rule, small, big in checks:
            if small > big + TOL:
                raise ValueError(
                    f"IT2TrFN needs nested reference points: {rule}, got {small:g} > {big:g}"
                )
        cap = min(up[4], up[5])
        for name, h in (("h1L", lo[4]), ("h2L", lo[5])):
            if h > cap + TOL:
                raise ValueError(
                    f"IT2TrFN needs nested reference points: {name} <= min(h1U, h2U), "
                    f"got {h:g} > {cap:g}"
                )
        object.__setattr__(self, "upper", up)
        object.__setattr__(self, "lower", lo)

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_trfn(cls, x: "TrFN | TFN") -> "IT2TrFN":
        r"""The type-1 number $x$ as an IT2 number: upper = lower = $(\text{points}; 1, 1)$.

        Args:
            x: A [`TrFN`][bulanik.numbers.TrFN] or [`TFN`][bulanik.numbers.TFN].

        Returns:
            The IT2 number with a zero-width footprint.

        Raises:
            TypeError: If `x` is neither a `TrFN` nor a `TFN`.

        Examples:
            >>> from bulanik import IT2TrFN, TrFN
            >>> IT2TrFN.from_trfn(TrFN(1, 2, 3, 5)).upper
            (1.0, 2.0, 3.0, 5.0, 1.0, 1.0)
        """
        if isinstance(x, TFN):
            x = x.to_trfn()
        if not isinstance(x, TrFN):
            raise TypeError(f"from_trfn needs a TrFN or TFN, got {type(x).__name__}")
        t = (x.a, x.b, x.c, x.d, 1.0, 1.0)
        return cls(t, t)

    # -- helpers --------------------------------------------------------------
    def _both(self) -> tuple[_Six, _Six]:
        return self.upper, self.lower

    def _lowest(self) -> float:
        return min(self.upper[0], self.lower[0])

    # -- arithmetic -----------------------------------------------------------
    def add(self, other: Any, ops: str = "algebraic") -> "IT2TrFN":
        r"""Sum: points add, heights take the minimum (Chen and Lee 2010a).

        Args:
            other: An `IT2TrFN`.
            ops: Only `"algebraic"` exists.

        Returns:
            The sum.

        Raises:
            TypeError: If `other` is not an `IT2TrFN`.
            ValueError: If `ops` is unknown.

        Examples:
            >>> from bulanik import IT2TrFN
            >>> a = IT2TrFN((0, 1, 2, 3, 1, 1), (0.5, 1, 2, 2.5, 0.8, 0.8))
            >>> (a + a).lower
            (1.0, 2.0, 4.0, 5.0, 0.8, 0.8)
        """
        self._check_ops(ops)
        self.same_kind(other)
        return IT2TrFN(*(
            (*(x + y for x, y in zip(s[:4], o[:4])), min(s[4], o[4]), min(s[5], o[5]))
            for s, o in zip(self._both(), other._both())
        ))

    def sub(self, other: Any, ops: str = "algebraic") -> "IT2TrFN":
        r"""Difference $(a_1 - b_4, a_2 - b_3, a_3 - b_2, a_4 - b_1)$ per trapezoid; `A - B` calls it.

        Args:
            other: An `IT2TrFN`.
            ops: Only `"algebraic"` exists.

        Returns:
            The difference; heights take the minimum.

        Raises:
            TypeError: If `other` is not an `IT2TrFN`.
            ValueError: If `ops` is unknown.

        Examples:
            >>> from bulanik import IT2TrFN
            >>> a = IT2TrFN((1, 2, 3, 4, 1, 1), (1, 2, 3, 4, 1, 1))
            >>> (a - a).upper
            (-3.0, -1.0, 1.0, 3.0, 1.0, 1.0)
        """
        self._check_ops(ops)
        self.same_kind(other)
        return IT2TrFN(*(
            (*(x - y for x, y in zip(s[:4], o[3::-1])), min(s[4], o[4]), min(s[5], o[5]))
            for s, o in zip(self._both(), other._both())
        ))

    def mul(self, other: Any, ops: str = "algebraic") -> "IT2TrFN":
        r"""Product: componentwise on the points, heights take the minimum (Chen and Lee 2010a).

        Args:
            other: An `IT2TrFN` with all points $\ge 0$ (as `self`).
            ops: Only `"algebraic"` exists.

        Returns:
            The product.

        Raises:
            TypeError: If `other` is not an `IT2TrFN`.
            ValueError: If `ops` is unknown, a point of either operand is negative.

        Examples:
            >>> from bulanik import IT2TrFN
            >>> a = IT2TrFN((1, 2, 3, 4, 1, 1), (1.5, 2, 3, 3.5, 0.8, 0.8))
            >>> (a * a).upper
            (1.0, 4.0, 9.0, 16.0, 1.0, 1.0)
        """
        self._check_ops(ops)
        self.same_kind(other)
        if min(self._lowest(), other._lowest()) < 0:
            raise ValueError(
                "mul needs all points >= 0, got lowest points "
                f"{self._lowest():g} and {other._lowest():g}"
            )
        return IT2TrFN(*(
            (*(x * y for x, y in zip(s[:4], o[:4])), min(s[4], o[4]), min(s[5], o[5]))
            for s, o in zip(self._both(), other._both())
        ))

    def scale(self, k: float, ops: str = "algebraic") -> "IT2TrFN":
        r"""Multiply the points by the real `k`; heights stay, and swap for $k < 0$.

        Args:
            k: The scalar.
            ops: Only `"algebraic"` exists.

        Returns:
            The scaled number.

        Raises:
            TypeError: If `k` is a `bool` or a string.
            ValueError: If `k` is not finite or `ops` is unknown.

        Examples:
            >>> from bulanik import IT2TrFN
            >>> a = IT2TrFN((0, 1, 2, 3, 1, 0.5), (0.5, 1, 2, 2.5, 0.5, 0.4))
            >>> a.scale(-1).upper
            (-3.0, -2.0, -1.0, -0.0, 0.5, 1.0)
        """
        self._check_ops(ops)
        k = finite(k, "k")

        def one(t: _Six) -> _Six:
            pts, h1, h2 = t[:4], t[4], t[5]
            if k < 0:
                pts, h1, h2 = pts[::-1], h2, h1
            return (*(k * x for x in pts), h1, h2)  # type: ignore[return-value]

        return IT2TrFN(*(one(t) for t in self._both()))

    def power(self, k: float, ops: str = "algebraic") -> "IT2TrFN":
        r"""Raise the points to the real power `k`; heights stay (swap for $k < 0$); `A ** k` calls it.

        Args:
            k: The exponent.
            ops: Only `"algebraic"` exists.

        Returns:
            The power.

        Raises:
            TypeError: If `k` is a `bool` or a string.
            ValueError: If `k` is not finite, `ops` is unknown, the lowest point is
                negative (zero for $k < 0$) or the power overflows.

        Examples:
            >>> from bulanik import IT2TrFN
            >>> a = IT2TrFN((1, 2, 3, 4, 1, 1), (1.5, 2, 3, 3.5, 0.8, 0.8))
            >>> (a ** 2).upper
            (1.0, 4.0, 9.0, 16.0, 1.0, 1.0)
        """
        self._check_ops(ops)
        k = finite(k, "k")
        low = self._lowest()
        if low < 0 or (k < 0 and low == 0):
            need = "> 0" if k < 0 else ">= 0"
            raise ValueError(f"power {k} needs the lowest point {need}, got {low}")

        def one(t: _Six) -> _Six:
            pts, h1, h2 = t[:4], t[4], t[5]
            if k < 0:
                pts, h1, h2 = pts[::-1], h2, h1
            try:
                return (*(x**k for x in pts), h1, h2)  # type: ignore[return-value]
            except OverflowError:
                raise ValueError(f"power {k} overflows a float") from None

        return IT2TrFN(*(one(t) for t in self._both()))

    def __sub__(self, other: Any) -> Any:
        if not self._is_same(other):
            return NotImplemented
        return self.sub(other)

    # -- scores, accuracy, distance -------------------------------------------
    @staticmethod
    def _dtrat(t: _Six) -> float:
        a1, a2, a3, a4, h1, h2 = t
        return ((a4 - a1) + (h1 * a2 - a1) + (h2 * a3 - a1)) / 4 + a1

    @staticmethod
    def _centroid(t: _Six) -> float:
        a1, a2, a3, a4, h1, h2 = t
        poly = [(a1, 0.0), (a2, h1), (a3, h2), (a4, 0.0)]
        area2 = cx6 = 0.0
        for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
            cross = x0 * y1 - x1 * y0
            area2 += cross
            cx6 += (x0 + x1) * cross
        if area2 == 0:
            return (a1 + a2 + a3 + a4) / 4
        return min(max(cx6 / (3 * area2), a1), a4)

    @staticmethod
    def _chen_lee(t: _Six) -> float:
        a = t[:4]
        m = sum((a[p] + a[p + 1]) / 2 for p in range(3))
        s = sum(abs(a[q + 1] - a[q]) / 2 for q in range(3)) + _stdev(a)
        return m - s / 4 + t[4] + t[5]

    def score(self, method: str | None = None) -> float:
        r"""Crisp score: `"dtrat"` (default), `"centroid"` or `"chen_lee"`; see the class docstring.

        Args:
            method: One of `"dtrat"`, `"centroid"`, `"chen_lee"`; `None` means `"dtrat"`.

        Returns:
            The score. `"dtrat"` and `"centroid"` are on the scale of the points;
            `"chen_lee"` also adds the heights and is only meaningful for ranking.

        Raises:
            ValueError: If `method` is not one of the three names.

        Examples:
            >>> from bulanik import IT2TrFN
            >>> a = IT2TrFN((1, 2, 3, 5, 1, 1), (1, 2, 3, 5, 1, 1))
            >>> a.score()
            2.75
            >>> round(a.score("chen_lee"), 4)
            18.2605
        """
        method = choose(method, self.SCORES, "score", "IT2TrFN", self.DEFAULT_SCORE)
        fn = {"dtrat": self._dtrat, "centroid": self._centroid, "chen_lee": self._chen_lee}[method]
        vals = [fn(t) for t in self._both()]
        return sum(vals) / 2 if method != "chen_lee" else sum(vals)

    def accuracy(self, method: str | None = None) -> float:
        """Not defined for IT2TrFN.

        Args:
            method: Ignored.

        Raises:
            ValueError: Always.

        Examples:
            >>> from bulanik import IT2TrFN
            >>> IT2TrFN((0, 1, 2, 3, 1, 1), (0, 1, 2, 3, 1, 1)).accuracy()
            Traceback (most recent call last):
            ...
            ValueError: IT2TrFN has no accuracy function
        """
        raise ValueError("IT2TrFN has no accuracy function")

    def distance(self, other: Any, method: str | None = None) -> float:
        r"""Vertex distance $\sqrt{\tfrac18 \sum (\Delta a)^2}$ over the eight points; heights are ignored.

        Args:
            other: An `IT2TrFN`.
            method: Only `"vertex"` exists.

        Returns:
            The distance.

        Raises:
            TypeError: If `other` is not an `IT2TrFN`.
            ValueError: If `method` is not `"vertex"`.

        Examples:
            >>> from bulanik import IT2TrFN
            >>> a = IT2TrFN((0, 1, 2, 3, 1, 1), (0, 1, 2, 3, 1, 1))
            >>> b = IT2TrFN((1, 2, 3, 4, 1, 1), (1, 2, 3, 4, 1, 1))
            >>> a.distance(b)
            1.0
        """
        choose(method, self.DISTANCES, "distance", "IT2TrFN", self.DEFAULT_DISTANCE)
        self.same_kind(other)
        d = sum(
            (x - y) ** 2
            for s, o in zip(self._both(), other._both())
            for x, y in zip(s[:4], o[:4])
        )
        return math.sqrt(d / 8)
