"""Abstract base of every fuzzy number.

[`FuzzyNumber`][bulanik.numbers.FuzzyNumber] fixes the interface shared by all
number types, so that the aggregation, ranking and MCDM code can work with any
of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .._validate import choose


def _kind_of(obj: Any) -> str:
    """Class name (and q, when present) of any object, for error messages."""
    q = getattr(obj, "q", None)
    name = type(obj).__name__
    return name if q is None else f"{name}(q={q})"


class FuzzyNumber(ABC):
    """Common interface of fuzzy numbers.

    The base assumes nothing about how many parts a number has, so interval-valued
    and type-2 numbers can implement it too. Subclasses declare which operator
    sets, scores, accuracy functions and distances they support in the class
    attributes below; the `ops`, `method` and `score` arguments of the methods
    are validated against them.

    Operators `+`, `*` (number or scalar), `**` (scalar) and, for triangular
    numbers, `-`, `/` and unary `-`, call the default (`"algebraic"`) operator set.

    Attributes:
        OPS: Names of the supported operator sets; the first is the default.
        SCORES: Names of the supported score functions.
        DEFAULT_SCORE: Score used when `score()` gets no `method`.
        ACCURACIES: Names of the supported accuracy functions (may be empty).
        DEFAULT_ACCURACY: Accuracy used when `accuracy()` gets no `method`.
        DISTANCES: Names of the supported distance measures.
        DEFAULT_DISTANCE: Distance used when `distance()` gets no `method`.

    Examples:
        >>> from bulanik import TFN, IFN
        >>> TFN.OPS
        ('algebraic',)
        >>> IFN.OPS
        ('algebraic', 'einstein')
        >>> TFN(1, 2, 3) + IFN(0.5, 0.4)
        Traceback (most recent call last):
        ...
        TypeError: unsupported operand type(s) for +: 'TFN' and 'IFN'
    """

    OPS: tuple[str, ...] = ()
    SCORES: tuple[str, ...] = ()
    DEFAULT_SCORE: str | None = None
    ACCURACIES: tuple[str, ...] = ()
    DEFAULT_ACCURACY: str | None = None
    DISTANCES: tuple[str, ...] = ()
    DEFAULT_DISTANCE: str | None = None

    @abstractmethod
    def add(self, other: Any, ops: str = "algebraic") -> Any:
        """Sum of two numbers of the same type.

        Args:
            other: A number of the same type (and the same `q`, if any).
            ops: Operator set name, one of `OPS`.

        Returns:
            The sum, a number of the same type.

        Raises:
            TypeError: If `other` is of a different type or rung.
            ValueError: If `ops` is not a supported operator set.
        """

    @abstractmethod
    def mul(self, other: Any, ops: str = "algebraic") -> Any:
        """Product of two numbers of the same type.

        Args:
            other: A number of the same type (and the same `q`, if any).
            ops: Operator set name, one of `OPS`.

        Returns:
            The product, a number of the same type.

        Raises:
            TypeError: If `other` is of a different type or rung.
            ValueError: If `ops` is not a supported operator set.
        """

    @abstractmethod
    def scale(self, k: float, ops: str = "algebraic") -> Any:
        """Scalar multiple ``k * A`` (repeated addition, extended to real ``k``).

        Args:
            k: The scalar. Orthopair and three-part numbers need `k >= 0`.
            ops: Operator set name, one of `OPS`.

        Returns:
            The scaled number, of the same type.

        Raises:
            ValueError: If `k` is not finite, is negative for a type that needs
                `k >= 0`, or `ops` is unsupported.
        """

    @abstractmethod
    def power(self, k: float, ops: str = "algebraic") -> Any:
        """Power ``A ** k`` (repeated multiplication, extended to real ``k``).

        Args:
            k: The exponent. Orthopair and three-part numbers need `k >= 0`.
            ops: Operator set name, one of `OPS`.

        Returns:
            The power, of the same type.

        Raises:
            ValueError: If `k` is not finite, is out of range for the type, or
                `ops` is unsupported.
        """

    @abstractmethod
    def score(self, method: str | None = None) -> float:
        """Crisp score used for ranking.

        Args:
            method: One of `SCORES`; `None` selects `DEFAULT_SCORE`.

        Returns:
            The score as a float.

        Raises:
            ValueError: If `method` is not one of `SCORES`.
        """

    @abstractmethod
    def accuracy(self, method: str | None = None) -> float:
        """Accuracy (certainty) function, used to break ties between equal scores.

        Args:
            method: One of `ACCURACIES`; `None` selects `DEFAULT_ACCURACY`.

        Returns:
            The accuracy as a float.

        Raises:
            ValueError: If `method` is not one of `ACCURACIES`, or the type has
                no accuracy function (triangular and trapezoidal numbers).
        """

    @abstractmethod
    def distance(self, other: Any, method: str | None = None) -> float:
        """Distance to another number of the same type.

        Args:
            other: A number of the same type (and the same `q`, if any).
            method: One of `DISTANCES`; `None` selects `DEFAULT_DISTANCE`.

        Returns:
            A non-negative float.

        Raises:
            TypeError: If `other` is of a different type or rung.
            ValueError: If `method` is not one of `DISTANCES`.
        """

    def same_kind(self, other: Any) -> None:
        """Check that `other` has the same class (and `q`, when present) as `self`.

        Args:
            other: The object to compare against.

        Raises:
            TypeError: If the classes or the rungs differ.

        Examples:
            >>> from bulanik import PFN, FFN
            >>> PFN(0.5, 0.5).same_kind(PFN(0.1, 0.2))
            >>> PFN(0.5, 0.5).same_kind(FFN(0.1, 0.2))
            Traceback (most recent call last):
            ...
            TypeError: cannot combine PFN(q=2) with FFN(q=3)
        """
        if type(other) is not type(self) or getattr(other, "q", None) != getattr(
            self, "q", None
        ):
            raise TypeError(
                f"cannot combine {self._kind()} with {_kind_of(other)}"
            )

    def _kind(self) -> str:
        return _kind_of(self)

    def _check_ops(self, ops: str) -> str:
        return choose(ops, self.OPS, "ops", type(self).__name__, self.OPS[0] if self.OPS else None)

    def _is_same(self, other: Any) -> bool:
        try:
            self.same_kind(other)
        except TypeError:
            return False
        return True

    def __add__(self, other: Any) -> Any:
        if not self._is_same(other):
            return NotImplemented
        return self.add(other)

    def __mul__(self, other: Any) -> Any:
        if self._is_same(other):
            return self.mul(other)
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return self.scale(other)
        return NotImplemented

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return self.scale(other)
        return NotImplemented

    def __pow__(self, k: Any) -> Any:
        if isinstance(k, (int, float)) and not isinstance(k, bool):
            return self.power(k)
        return NotImplemented
