r"""Weighted aggregation: weighted average (WA), weighted geometric (WG), experts.

[`wa`][bulanik.aggregate.wa] and [`wg`][bulanik.aggregate.wg] combine fuzzy numbers
of one type with non-negative weights summing to 1, using the type's own
operators (`scale` / `add` for the average, `power` / `mul` for the geometric
mean). [`aggregate_experts`][bulanik.aggregate.aggregate_experts] applies one of
them to a stack of expert rating matrices, cell by cell.

All functions require the inputs to be of one class (and rung, for orthopair
numbers); mixing types raises `TypeError`.
"""

from __future__ import annotations

import math
from typing import Sequence

from ._validate import check_weights, choose
from .numbers import FuzzyNumber, PictureFN
from .numbers.threepart import _picture_aggregate


def _prepare(
    values: Sequence[FuzzyNumber],
    weights: Sequence[float] | None,
    ops: str,
    normalize: bool,
) -> tuple[list[FuzzyNumber], list[float]]:
    """Validate, then return the inputs with a positive weight and those weights."""
    values = list(values)
    if not values:
        raise ValueError("cannot aggregate an empty list of values")
    first = values[0]
    if not isinstance(first, FuzzyNumber):
        raise TypeError(f"cannot aggregate {type(first).__name__}; need fuzzy numbers")
    for v in values[1:]:
        first.same_kind(v)
    first._check_ops(ops)
    w = check_weights(weights, len(values), normalize)
    total = math.fsum(w)  # within 1e-9 of 1 already; make it exactly 1
    pairs = [(v, x / total) for v, x in zip(values, w) if x > 0.0]
    return [v for v, _ in pairs], [x for _, x in pairs]


def _fold(
    values: Sequence[FuzzyNumber],
    weights: Sequence[float] | None,
    ops: str,
    normalize: bool,
    kind: str,
) -> FuzzyNumber:
    vals, w = _prepare(values, weights, ops, normalize)
    if not vals:  # unreachable for valid weights; kept as a guard
        raise ValueError("weights must not all be zero")  # pragma: no cover
    if isinstance(vals[0], PictureFN):
        return _picture_aggregate(vals, w, kind)  # scale(k < 1) is not closed for pictures
    if kind == "wa":
        out = vals[0].scale(w[0], ops)
        for v, x in zip(vals[1:], w[1:]):
            out = out.add(v.scale(x, ops), ops)
    else:
        out = vals[0].power(w[0], ops)
        for v, x in zip(vals[1:], w[1:]):
            out = out.mul(v.power(x, ops), ops)
    return out


def wa(
    values: Sequence[FuzzyNumber],
    weights: Sequence[float] | None = None,
    *,
    ops: str = "algebraic",
    normalize: bool = False,
) -> FuzzyNumber:
    r"""Weighted average $\sum_i w_i A_i$ with the type's own operators.

    Computed as `scale` of each number by its weight, folded with `add`. For
    triangular numbers this is the componentwise weighted mean of the points;
    for intuitionistic numbers it is the classic IFWA operator
    $\left(1 - \prod_i (1 - \mu_i)^{w_i},\; \prod_i \nu_i^{w_i}\right)$.
    Picture numbers use the closed form
    $\left(1 - \prod_i (1 - \mu_i)^{w_i},\; \prod_i \eta_i^{w_i},\; \prod_i \nu_i^{w_i}\right)$,
    because scaling by a weight below 1 is not closed for them.

    `weights=None` means equal weights $1/n$. Weights must be non-negative and
    sum to 1 unless `normalize=True` (then they are divided by their sum);
    weights that sum to 1 only within $10^{-9}$ are accepted and silently
    rescaled to sum to exactly 1. Inputs with a weight of 0 are ignored.

    Args:
        values: Fuzzy numbers of one type (and rung).
        weights: One weight per value, or `None` for equal weights.
        ops: Operator set of the type, `"algebraic"` or (orthopair) `"einstein"`.
        normalize: Divide the weights by their sum instead of requiring sum 1.

    Returns:
        The aggregated number, of the type of the inputs.

    Raises:
        ValueError: If `values` is empty, the number of weights differs, a
            weight is negative or not finite, the weights do not sum to 1 (and
            `normalize` is false) or are all zero (with `normalize`), or `ops`
            is unsupported.
        TypeError: If the values are not fuzzy numbers or differ in type.

    References:
        Xu (2007), IEEE Transactions on Fuzzy Systems 15(6): intuitionistic
        fuzzy aggregation operators. Wei (2017) for the picture closed forms
        (retracted by the publisher in 2025; the closed forms are tested
        against the code, not the paper).

    Examples:
        >>> from bulanik import TFN, IFN
        >>> from bulanik.aggregate import wa
        >>> wa([TFN(1, 2, 4), TFN(2, 3, 5)])
        TFN(l=1.5, m=2.5, u=4.5)
        >>> r = wa([IFN(0.6, 0.3), IFN(0.5, 0.4)], [2, 1], normalize=True)
        >>> round(r.mu, 4), round(r.nu, 4)
        (0.5691, 0.3302)
        >>> wa([TFN(1, 2, 4), TFN(2, 3, 5)], [0.5, 0.2])
        Traceback (most recent call last):
        ...
        ValueError: weights must sum to 1, got 0.7
    """
    return _fold(values, weights, ops, normalize, "wa")


def wg(
    values: Sequence[FuzzyNumber],
    weights: Sequence[float] | None = None,
    *,
    ops: str = "algebraic",
    normalize: bool = False,
) -> FuzzyNumber:
    r"""Weighted geometric mean $\prod_i A_i^{w_i}$ with the type's own operators.

    Computed as `power` of each number by its weight, folded with `mul`. For
    positive triangular numbers this is the componentwise weighted geometric
    mean of the points. Weights, `normalize` and `ops` behave as in
    [`wa`][bulanik.aggregate.wa]. Picture numbers use closed forms
    $\left(\prod_i (\mu_i + \eta_i)^{w_i} - \prod_i \eta_i^{w_i},\;
    \prod_i \eta_i^{w_i},\; 1 - \prod_i (1 - \nu_i)^{w_i}\right)$.

    Args:
        values: Fuzzy numbers of one type (and rung).
        weights: One weight per value, or `None` for equal weights.
        ops: Operator set of the type, `"algebraic"` or (orthopair) `"einstein"`.
        normalize: Divide the weights by their sum instead of requiring sum 1.

    Returns:
        The aggregated number, of the type of the inputs.

    Raises:
        ValueError: As for [`wa`][bulanik.aggregate.wa]; additionally, for
            triangular numbers with a negative lowest point.
        TypeError: If the values are not fuzzy numbers or differ in type.

    References:
        Xu and Yager (2006), International Journal of General Systems 35(4):
        intuitionistic fuzzy geometric aggregation operators.

    Examples:
        >>> from bulanik import TFN
        >>> from bulanik.aggregate import wg
        >>> r = wg([TFN(1, 2, 4), TFN(2, 3, 5)])
        >>> round(r.l, 4), round(r.m, 4), round(r.u, 4)
        (1.4142, 2.4495, 4.4721)
    """
    return _fold(values, weights, ops, normalize, "wg")


def aggregate_experts(
    ratings: Sequence[Sequence[Sequence[FuzzyNumber]]],
    weights: Sequence[float] | None = None,
    *,
    method: str = "wa",
    ops: str = "algebraic",
    normalize: bool = False,
) -> list[list[FuzzyNumber]]:
    """Aggregate expert ratings indexed ``[expert][alternative][criterion]``.

    For every alternative-criterion cell the experts' numbers are combined with
    [`wa`][bulanik.aggregate.wa] or [`wg`][bulanik.aggregate.wg] using the
    expert weights.

    Args:
        ratings: One matrix per expert; each is a list (alternatives) of lists
            (criteria) of fuzzy numbers, all of the same shape and type.
        weights: One weight per expert, or `None` for equal weights.
        method: `"wa"` (default) or `"wg"`.
        ops: Operator set of the number type.
        normalize: Divide the expert weights by their sum instead of requiring sum 1.

    Returns:
        The `[alternative][criterion]` matrix of aggregated numbers.

    Raises:
        ValueError: If `method` is unknown, there are no experts, the matrices
            differ in shape, or the weights are invalid (see
            [`wa`][bulanik.aggregate.wa]).
        TypeError: If the numbers differ in type.

    Examples:
        >>> from bulanik import TFN
        >>> from bulanik.aggregate import aggregate_experts
        >>> a, b = TFN(1, 2, 4), TFN(2, 3, 5)
        >>> experts = [[[a, b]], [[b, a]]]  # 2 experts, 1 alternative, 2 criteria
        >>> aggregate_experts(experts)
        [[TFN(l=1.5, m=2.5, u=4.5), TFN(l=1.5, m=2.5, u=4.5)]]
    """
    method = choose(method, ("wa", "wg"), "method", "aggregate_experts")
    fn = wa if method == "wa" else wg
    if not ratings:
        raise ValueError("need at least one expert")
    check_weights(weights, len(ratings), normalize)  # also when there are no cells
    n_alt, n_crit = len(ratings[0]), len(ratings[0][0]) if ratings[0] else 0
    for r in ratings:
        if len(r) != n_alt or any(len(row) != n_crit for row in r):
            raise ValueError("expert rating matrices must all have the same shape")
    return [
        [
            fn([r[i][j] for r in ratings], weights, ops=ops, normalize=normalize)
            for j in range(n_crit)
        ]
        for i in range(n_alt)
    ]
