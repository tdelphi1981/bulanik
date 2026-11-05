"""Helpers shared by the MCDM methods.

Input handling used by the weighting and ranking functions: ordering by score,
default names, and turning the several accepted forms of criteria weights (a
sequence, a mapping by name, a
[`WeightingResult`][bulanik.mcdm.results.WeightingResult] or a
[`BWMResult`][bulanik.mcdm.results.BWMResult]) into a list in criteria order.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .._validate import check_weights, finite
from .results import BWMResult, WeightingResult

__all__ = [
    "rank_order",
    "coerce_weights",
    "weights_in_order",
    "check_matrix",
    "check_types",
    "default_names",
]


def rank_order(scores: Mapping[str, float], descending: bool) -> tuple[str, ...]:
    """Names sorted by score; ties keep input order (stable sort).

    Args:
        scores: Mapping from name to score.
        descending: `True` puts the largest score first, `False` the smallest.

    Returns:
        The names as a tuple, best first.

    Examples:
        >>> from bulanik.mcdm._common import rank_order
        >>> rank_order({"a": 1, "b": 2, "c": 2}, descending=True)
        ('b', 'c', 'a')
    """
    names = list(scores)
    names.sort(key=lambda k: scores[k], reverse=descending)
    return tuple(names)


def default_names(prefix: str, n: int) -> list[str]:
    """Default names `prefix1 .. prefixn`, e.g. `A1..An` or `C1..Cn`.

    Args:
        prefix: Text put before the 1-based index.
        n: Number of names.

    Returns:
        A list of `n` names.

    Examples:
        >>> from bulanik.mcdm._common import default_names
        >>> default_names("A", 3)
        ['A1', 'A2', 'A3']
    """
    return [f"{prefix}{i + 1}" for i in range(n)]


def infer_criteria(weights: Any, criteria: Sequence[str] | None) -> Sequence[str] | None:
    """Criteria names: as given, else taken from the weights.

    Args:
        weights: Criteria weights in any accepted form.
        criteria: Explicit names, which win when not `None`.

    Returns:
        `criteria` if given; otherwise the keys, in order, of a mapping or of a
        [`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
        [`BWMResult`][bulanik.mcdm.results.BWMResult]; otherwise `None` (a plain
        sequence carries no names).

    Examples:
        >>> from bulanik.mcdm._common import infer_criteria
        >>> infer_criteria({"cost": 0.4, "speed": 0.6}, None)
        ['cost', 'speed']
        >>> infer_criteria([0.4, 0.6], None) is None
        True
    """
    if criteria is not None:
        return criteria
    if isinstance(weights, (WeightingResult, BWMResult)):
        weights = weights.weights
    return list(weights) if isinstance(weights, Mapping) else None


def coerce_weights(
    weights: Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult,
    criteria: Sequence[str],
    normalize: bool = False,
) -> list[float]:
    """Validated weights in `criteria` order from any accepted input form.

    Args:
        weights: A sequence in criteria order, a mapping by criterion name, or a
            [`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
            [`BWMResult`][bulanik.mcdm.results.BWMResult]. A mapping or result
            must name every criterion and no other key.
        criteria: The criterion names, which fix the order.
        normalize: If `True`, rescale non-negative weights to sum to 1; if
            `False` they must already sum to 1.

    Returns:
        The weights as a list of floats.

    Raises:
        ValueError: If a mapping or result misses a criterion or names an
            unknown one, the count is wrong, a weight is negative or not
            finite, or (with `normalize=False`) they do not sum to 1.

    Examples:
        >>> from bulanik.mcdm._common import coerce_weights
        >>> coerce_weights({"b": 2, "a": 1}, ["a", "b"], normalize=True)
        [0.3333333333333333, 0.6666666666666666]
        >>> coerce_weights({"a": 1.0}, ["a", "b"])
        Traceback (most recent call last):
        ...
        ValueError: weights are missing criteria: b
    """
    return check_weights(weights_in_order(weights, criteria), len(criteria), normalize)


def weights_in_order(weights: Any, criteria: Sequence[str]) -> list[Any]:
    """Weights in `criteria` order, without validating the values.

    Used where the weights may be fuzzy numbers, so only the order is fixed.

    Args:
        weights: A sequence, a mapping by criterion name, or a
            [`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
            [`BWMResult`][bulanik.mcdm.results.BWMResult].
        criteria: The criterion names, which fix the order.

    Returns:
        The values as a list.

    Raises:
        ValueError: If a mapping or result misses a criterion or names an
            unknown one.

    Examples:
        >>> from bulanik.mcdm._common import weights_in_order
        >>> weights_in_order({"b": 2, "a": 1}, ["a", "b"])
        [1, 2]
    """
    if isinstance(weights, (WeightingResult, BWMResult)):
        weights = weights.weights
    if not isinstance(weights, Mapping):
        return list(weights)
    missing = [c for c in criteria if c not in weights]
    if missing:
        raise ValueError(f"weights are missing criteria: {', '.join(map(str, missing))}")
    known = set(criteria)
    unknown = [k for k in weights if k not in known]
    if unknown:
        raise ValueError(
            f"weights name unknown criteria: {', '.join(map(str, unknown))}; "
            f"valid: {', '.join(map(str, criteria))}"
        )
    return [weights[c] for c in criteria]


def check_matrix(
    matrix: Sequence[Sequence[float]],
    n_alt: int | None = None,
    n_crit: int | None = None,
) -> list[list[float]]:
    """Validate a rectangular, finite alternatives x criteria matrix.

    Args:
        matrix: Rows are alternatives, columns are criteria.
        n_alt: Expected number of rows, or `None` for any.
        n_crit: Expected number of columns, or `None` for any.

    Returns:
        The matrix as a list of lists.

    Raises:
        ValueError: If the matrix is empty, ragged, of the wrong shape, or has a
            non-finite entry.

    Examples:
        >>> from bulanik.mcdm._common import check_matrix
        >>> check_matrix([[1, 2], [3, 4]])
        [[1.0, 2.0], [3.0, 4.0]]
    """
    rows = [list(r) for r in matrix]
    if not rows or not rows[0]:
        raise ValueError("matrix must have at least one row and one column")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("matrix rows must all have the same length")
    if n_alt is not None and len(rows) != n_alt:
        raise ValueError(f"expected {n_alt} alternatives (rows), got {len(rows)}")
    if n_crit is not None and width != n_crit:
        raise ValueError(f"expected {n_crit} criteria (columns), got {width}")
    return [[finite(x, "matrix entry") for x in r] for r in rows]


def check_types(criteria_types: Sequence[str], n: int) -> list[str]:
    """Validate `n` criterion types, each `"benefit"` or `"cost"`.

    Args:
        criteria_types: One type per criterion.
        n: Expected number of criteria.

    Returns:
        The types as a list.

    Raises:
        ValueError: If the count is not `n` or a type is not `"benefit"` or
            `"cost"`.

    Examples:
        >>> from bulanik.mcdm._common import check_types
        >>> check_types(["benefit", "cost"], 2)
        ['benefit', 'cost']
    """
    types = list(criteria_types)
    if len(types) != n:
        raise ValueError(f"expected {n} criteria types, got {len(types)}")
    for t in types:
        if t not in ("benefit", "cost"):
            raise ValueError(f"unknown criterion type '{t}'; valid: benefit, cost")
    return types
