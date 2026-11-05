"""Crisp ranking methods: TOPSIS, VIKOR, WASPAS and EDAS.

Four classical methods that rank alternatives from a crisp decision matrix
(rows are alternatives, columns are criteria), a vector of criteria weights and
the type of each criterion (`"benefit"` or `"cost"`). All have the same
signature apart from one method parameter, accept the weights as a sequence, a
mapping by criterion name or a
[`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
[`BWMResult`][bulanik.mcdm.results.BWMResult], and return a
[`RankingResult`][bulanik.mcdm.results.RankingResult]. Higher scores are better
for TOPSIS, WASPAS and EDAS; for VIKOR **lower** is better, and the `ranking`
of every result is already ordered best first.

References:
    Hwang & Yoon (1981); Opricovic & Tzeng (2004), European Journal of
    Operational Research 156(2), 445-455; Zavadskas et al. (2012), Elektronika
    ir Elektrotechnika 122(6), 3-6; Keshavarz Ghorabaee et al. (2015),
    Informatica 26(3), 435-451.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .._validate import finite
from ._common import (
    check_matrix,
    check_types,
    coerce_weights,
    default_names,
    infer_criteria,
    rank_order,
)
from .results import BWMResult, RankingResult, WeightingResult

__all__ = ["topsis", "vikor", "waspas", "edas"]


def _names(given: Sequence[str] | None, prefix: str, n: int, what: str) -> list[str]:
    if given is None:
        return default_names(prefix, n)
    names = list(given)
    if len(names) != n:
        raise ValueError(f"expected {n} {what}, got {len(names)}")
    if len(set(names)) != n:
        raise ValueError(f"{what} names must be unique")
    return names


def _prepare(matrix, weights, criteria_types, alternatives, criteria, normalize_weights):
    x = check_matrix(matrix)
    return (x, *_prepare_names(len(x), len(x[0]), weights, criteria_types,
                               alternatives, criteria, normalize_weights))


def _prepare_names(m, n, weights, criteria_types, alternatives, criteria, normalize_weights):
    """Names, criterion types and weights for an ``m`` by ``n`` matrix (the non-matrix part of ``_prepare``)."""
    alts = _names(alternatives, "A", m, "alternatives")
    crit = _names(infer_criteria(weights, criteria), "C", n, "criteria")
    types = check_types(criteria_types, n)
    w = coerce_weights(weights, crit, normalize_weights)
    return alts, crit, types, w


def _column(x: list[list[float]], j: int) -> list[float]:
    return [row[j] for row in x]


def _unit_scale(col: Sequence[float]) -> list[float]:
    """``col`` times the power of two that puts its largest magnitude in [0.5, 1).

    The scaling is exact (``ldexp``), so ratio-based methods (VIKOR, EDAS) are
    unchanged by it, while differences can no longer overflow (entries near
    1.8e308) and subnormal columns (5e-324) keep their full precision.
    """
    top = max(abs(c) for c in col)
    if top == 0.0:
        return list(col)
    e = math.frexp(top)[1]
    return [math.ldexp(c, -e) for c in col]


def _finite_all(method: str, values: Sequence[float], what: str) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError(
            f"{method}: {what} is not finite; the matrix entries are too large to "
            "combine in floating point, so rescale them"
        )


def _result(method, alts, scores, descending, operators, **extras) -> RankingResult:
    _finite_all(method, scores, "a score")
    for k, v in extras.items():
        _finite_all(method, v, k)
    sc = dict(zip(alts, scores))
    return RankingResult(
        method=method,
        alternatives=tuple(alts),
        scores=sc,
        ranking=rank_order(sc, descending),
        operators=operators,
        extras={k: dict(zip(alts, v)) for k, v in extras.items()},
    )


def topsis(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
) -> RankingResult:
    r"""TOPSIS (Hwang & Yoon 1981) with vector normalisation, Euclidean distance.

    Each column is divided by its Euclidean norm and multiplied by the weight,
    $v_{ij} = w_j x_{ij} / \sqrt{\sum_i x_{ij}^2}$. With the ideal point
    $v_j^+$ (column maximum for a benefit, minimum for a cost) and the
    anti-ideal $v_j^-$, the distances $D_i^{\pm}$ are Euclidean and the score is
    the relative closeness

    $$C_i = \frac{D_i^-}{D_i^+ + D_i^-} \in [0, 1],$$

    larger is better (0.5 if both distances are 0, i.e. all alternatives are
    equal). A column of zeros contributes 0. `extras` holds `d_plus` and
    `d_minus`.

    Args:
        matrix: Alternatives by criteria (rows by columns), finite numbers.
        weights: Criteria weights: a sequence in criteria order, a mapping by
            criterion name (naming every criterion and no other), or a
            [`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
            [`BWMResult`][bulanik.mcdm.results.BWMResult]. They must sum to 1
            unless `normalize_weights` is set.
        criteria_types: `"benefit"` (larger is better) or `"cost"` (smaller is
            better) for each criterion.
        alternatives: Alternative names; `A1..Am` by default.
        criteria: Criterion names; if `None` they are the keys of a mapping or
            result `weights` (in order), else `C1..Cn`.
        normalize_weights: Rescale non-negative weights to sum to 1 instead of
            requiring it.

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"normalization": "vector", "distance": "euclidean"}`.

    Raises:
        TypeError: If a matrix entry or a weight is a `bool` or `str` instead of a number.
        ValueError: If the matrix is empty, ragged or non-finite, the counts of
            names, types or weights do not match, names repeat, a mapping or
            result misses a criterion or names an unknown one, a type is not
            `"benefit"` or `"cost"`, or the weights are negative or (without
            `normalize_weights`) do not sum to 1.

    Examples:
        >>> from bulanik.mcdm import topsis
        >>> r = topsis([[250, 16], [200, 12], [300, 32]], [0.5, 0.5],
        ...            ["cost", "benefit"], alternatives=["A", "B", "C"])
        >>> r.ranking
        ('C', 'B', 'A')
        >>> {k: round(v, 3) for k, v in r.scores.items()}
        {'A': 0.262, 'B': 0.301, 'C': 0.699}

    References:
        Hwang & Yoon (1981), *Multiple Attribute Decision Making*, Springer.
    """
    x, alts, _, types, w = _prepare(
        matrix, weights, criteria_types, alternatives, criteria, normalize_weights
    )
    m, n = len(x), len(x[0])
    v = [[0.0] * n for _ in range(m)]
    for j in range(n):
        col = _column(x, j)
        norm = math.hypot(*col)  # no overflow to inf before the square root
        _finite_all("topsis", [norm], f"the norm of criterion {j + 1}")
        for i in range(m):
            v[i][j] = w[j] * (col[i] / norm) if norm > 0 else 0.0
    best, worst = [], []
    for j in range(n):
        col = _column(v, j)
        hi, lo = max(col), min(col)
        best.append(hi if types[j] == "benefit" else lo)
        worst.append(lo if types[j] == "benefit" else hi)
    d_plus = [math.sqrt(sum((v[i][j] - best[j]) ** 2 for j in range(n))) for i in range(m)]
    d_minus = [math.sqrt(sum((v[i][j] - worst[j]) ** 2 for j in range(n))) for i in range(m)]
    scores = [
        dm / (dp + dm) if dp + dm > 0 else 0.5 for dp, dm in zip(d_plus, d_minus)
    ]
    return _result(
        "topsis",
        alts,
        scores,
        True,
        {"normalization": "vector", "distance": "euclidean"},
        d_plus=d_plus,
        d_minus=d_minus,
    )


def _unit_param(value: float, name: str) -> float:
    value = finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def vikor(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
    v: float = 0.5,
) -> RankingResult:
    r"""VIKOR (Opricovic & Tzeng 2004); `v` is the weight of group utility.

    For each criterion the best $f_j^*$ and worst $f_j^-$ values are the column
    maximum and minimum for a benefit (swapped for a cost). The group utility
    and individual regret are

    $$S_i = \sum_j w_j \frac{|f_j^* - x_{ij}|}{|f_j^* - f_j^-|}, \qquad
    R_i = \max_j w_j \frac{|f_j^* - x_{ij}|}{|f_j^* - f_j^-|},$$

    (criteria with $f_j^* = f_j^-$ are skipped) and the compromise index is

    $$Q_i = v\,\frac{S_i - S^*}{S^- - S^*} + (1 - v)\,\frac{R_i - R^*}{R^- - R^*},$$

    with $S^* = \min S$, $S^- = \max S$ (likewise $R$; a term is 0 when its
    range is 0). **Lower $Q$ is better**: `ranking` lists the smallest score
    first and the `scores` are $Q$, not a goodness measure. `extras` holds `S`
    and `R`. The compromise-acceptability conditions C1/C2 are not checked.

    Args:
        matrix: Alternatives by criteria (rows by columns), finite numbers.
        weights: Criteria weights: a sequence in criteria order, a mapping by
            criterion name (naming every criterion and no other), or a
            [`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
            [`BWMResult`][bulanik.mcdm.results.BWMResult]. They must sum to 1
            unless `normalize_weights` is set.
        criteria_types: `"benefit"` (larger is better) or `"cost"` (smaller is
            better) for each criterion.
        alternatives: Alternative names; `A1..Am` by default.
        criteria: Criterion names; if `None` they are the keys of a mapping or
            result `weights` (in order), else `C1..Cn`.
        normalize_weights: Rescale non-negative weights to sum to 1 instead of
            requiring it.
        v: Weight of the group utility strategy in [0, 1]; $1 - v$ weights the
            individual regret. 0.5 is the usual consensus.

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"normalization": "linear_range", "v": "0.5"}` (the
        value of `v` as text).

    Raises:
        TypeError: If a matrix entry or a weight is a `bool` or `str` instead of a number.
        ValueError: If the matrix is empty, ragged or non-finite, the counts of
            names, types or weights do not match, names repeat, a mapping or
            result misses a criterion or names an unknown one, a type is not
            `"benefit"` or `"cost"`, or the weights are negative or (without
            `normalize_weights`) do not sum to 1, or `v` is outside [0, 1] or
            not finite.

    Examples:
        >>> from bulanik.mcdm import vikor
        >>> prices = [[250, 16, 12, 5], [200, 16, 8, 3], [300, 32, 16, 4], [275, 32, 8, 4]]
        >>> r = vikor(prices, [0.3, 0.3, 0.2, 0.2], ["cost", "benefit", "benefit", "benefit"],
        ...           alternatives=["A", "B", "C", "D"])
        >>> r.ranking
        ('D', 'C', 'A', 'B')
        >>> {k: round(v, 3) for k, v in r.scores.items()}
        {'A': 0.75, 'B': 1.0, 'C': 0.5, 'D': 0.208}

    References:
        Opricovic & Tzeng (2004), European Journal of Operational Research
        156(2), 445-455.
    """
    v = _unit_param(v, "v")
    x, alts, _, types, w = _prepare(
        matrix, weights, criteria_types, alternatives, criteria, normalize_weights
    )
    m, n = len(x), len(x[0])
    s = [0.0] * m
    r = [0.0] * m
    for j in range(n):
        col = _unit_scale(_column(x, j))  # exact; keeps differences finite and subnormals precise
        hi, lo = max(col), min(col)
        best, worst = (hi, lo) if types[j] == "benefit" else (lo, hi)
        span = abs(best - worst)
        if span == 0:
            continue
        for i in range(m):
            d = w[j] * (abs(best - col[i]) / span)
            s[i] += d
            r[i] = max(r[i], d)

    def term(val: float, lo: float, hi: float) -> float:
        return (val - lo) / (hi - lo) if hi > lo else 0.0

    s_lo, s_hi, r_lo, r_hi = min(s), max(s), min(r), max(r)
    q = [
        v * term(s[i], s_lo, s_hi) + (1.0 - v) * term(r[i], r_lo, r_hi) for i in range(m)
    ]
    return _result(
        "vikor",
        alts,
        q,
        False,
        {"normalization": "linear_range", "v": str(v)},
        S=s,
        R=r,
    )


def waspas(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
    lam: float = 0.5,
) -> RankingResult:
    r"""WASPAS (Zavadskas et al. 2012): `lam` * WSM + (1 - `lam`) * WPM.

    Entries are normalised linearly, $\bar x_{ij} = x_{ij} / \max_i x_{ij}$ for
    a benefit and $\min_i x_{ij} / x_{ij}$ for a cost. The weighted-sum and
    weighted-product scores are combined with the parameter $\lambda$
    (`lam`):

    $$Q_i = \lambda \sum_j w_j \bar x_{ij} + (1 - \lambda) \prod_j \bar x_{ij}^{\,w_j},$$

    larger is better. $\lambda = 1$ is pure WSM, $\lambda = 0$ pure WPM.
    `extras` holds `wsm` and `wpm`.

    Args:
        matrix: Alternatives by criteria (rows by columns), finite numbers.
        weights: Criteria weights: a sequence in criteria order, a mapping by
            criterion name (naming every criterion and no other), or a
            [`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
            [`BWMResult`][bulanik.mcdm.results.BWMResult]. They must sum to 1
            unless `normalize_weights` is set.
        criteria_types: `"benefit"` (larger is better) or `"cost"` (smaller is
            better) for each criterion.
        alternatives: Alternative names; `A1..Am` by default.
        criteria: Criterion names; if `None` they are the keys of a mapping or
            result `weights` (in order), else `C1..Cn`.
        normalize_weights: Rescale non-negative weights to sum to 1 instead of
            requiring it.
        lam: The combination parameter $\lambda$ in [0, 1] (0.5 by default).

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"normalization": "linear_max", "lambda": "0.5"}` (the
        value of `lam` as text).

    Raises:
        TypeError: If a matrix entry or a weight is a `bool` or `str` instead of a number.
        ValueError: If the matrix is empty, ragged or non-finite, the counts of
            names, types or weights do not match, names repeat, a mapping or
            result misses a criterion or names an unknown one, a type is not
            `"benefit"` or `"cost"`, or the weights are negative or (without
            `normalize_weights`) do not sum to 1, an entry is not strictly
            positive (for fuzzy scores try `score="normalized"`), or `lam` is
            outside [0, 1].

    Examples:
        >>> from bulanik.mcdm import waspas
        >>> r = waspas([[250, 16], [200, 12], [300, 32]], [0.5, 0.5],
        ...            ["cost", "benefit"], alternatives=["A", "B", "C"], lam=0.5)
        >>> r.ranking
        ('C', 'B', 'A')
        >>> {k: round(v, 3) for k, v in r.scores.items()}
        {'A': 0.641, 'B': 0.65, 'C': 0.825}

    References:
        Zavadskas, Turskis, Antucheviciene & Zakarevicius (2012), Elektronika
        ir Elektrotechnika 122(6), 3-6.
    """
    lam = _unit_param(lam, "lam")
    x, alts, _, types, w = _prepare(
        matrix, weights, criteria_types, alternatives, criteria, normalize_weights
    )
    if any(c <= 0 for row in x for c in row):
        raise ValueError(
            "WASPAS needs strictly positive entries; for fuzzy scores try score=\"normalized\""
        )
    m, n = len(x), len(x[0])
    norm = [[0.0] * n for _ in range(m)]
    for j in range(n):
        col = _column(x, j)
        hi, lo = max(col), min(col)
        for i in range(m):
            norm[i][j] = col[i] / hi if types[j] == "benefit" else lo / col[i]
    wsm = [sum(w[j] * norm[i][j] for j in range(n)) for i in range(m)]
    wpm = [math.prod(norm[i][j] ** w[j] for j in range(n)) for i in range(m)]
    q = [lam * a + (1.0 - lam) * b for a, b in zip(wsm, wpm)]
    return _result(
        "waspas",
        alts,
        q,
        True,
        {"normalization": "linear_max", "lambda": str(lam)},
        wsm=wsm,
        wpm=wpm,
    )


def edas(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
) -> RankingResult:
    r"""EDAS (Keshavarz Ghorabaee et al. 2015): distance from the average solution.

    With the column average $\bar x_j$, the positive and negative distances
    from average (PDA and NDA) of a benefit criterion are

    $$PDA_{ij} = \frac{\max(0, x_{ij} - \bar x_j)}{\bar x_j}, \qquad
    NDA_{ij} = \frac{\max(0, \bar x_j - x_{ij})}{\bar x_j},$$

    swapped for a cost criterion. The weighted sums $SP_i = \sum_j w_j PDA_{ij}$
    and $SN_i = \sum_j w_j NDA_{ij}$ are normalised, $NSP_i = SP_i / \max SP$
    and $NSN_i = 1 - SN_i / \max SN$ (0 and 1 when the maximum is 0), and the
    appraisal score is $(NSP_i + NSN_i) / 2$ in [0, 1], larger is better.
    `extras` holds `SP`, `SN`, `NSP` and `NSN`.

    Args:
        matrix: Alternatives by criteria (rows by columns), finite numbers.
        weights: Criteria weights: a sequence in criteria order, a mapping by
            criterion name (naming every criterion and no other), or a
            [`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
            [`BWMResult`][bulanik.mcdm.results.BWMResult]. They must sum to 1
            unless `normalize_weights` is set.
        criteria_types: `"benefit"` (larger is better) or `"cost"` (smaller is
            better) for each criterion.
        alternatives: Alternative names; `A1..Am` by default.
        criteria: Criterion names; if `None` they are the keys of a mapping or
            result `weights` (in order), else `C1..Cn`.
        normalize_weights: Rescale non-negative weights to sum to 1 instead of
            requiring it.

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"normalization": "average_distance"}`.

    Raises:
        TypeError: If a matrix entry or a weight is a `bool` or `str` instead of a number.
        ValueError: If the matrix is empty, ragged or non-finite, the counts of
            names, types or weights do not match, names repeat, a mapping or
            result misses a criterion or names an unknown one, a type is not
            `"benefit"` or `"cost"`, or the weights are negative or (without
            `normalize_weights`) do not sum to 1, or a column average is not positive.

    Examples:
        >>> from bulanik.mcdm import edas
        >>> r = edas([[250, 16], [200, 12], [300, 32]], [0.5, 0.5],
        ...          ["cost", "benefit"], alternatives=["A", "B", "C"])
        >>> r.ranking
        ('C', 'A', 'B')
        >>> {k: round(v, 3) for k, v in r.scores.items()}
        {'A': 0.25, 'B': 0.167, 'C': 0.75}

    References:
        Keshavarz Ghorabaee, Zavadskas, Olfat & Turskis (2015), Informatica
        26(3), 435-451.
    """
    x, alts, _, types, w = _prepare(
        matrix, weights, criteria_types, alternatives, criteria, normalize_weights
    )
    m, n = len(x), len(x[0])
    sp = [0.0] * m
    sn = [0.0] * m
    for j in range(n):
        raw = _column(x, j)
        col = _unit_scale(raw)  # exact; the ratios below do not depend on it
        av = math.fsum(c / m for c in col)  # no overflow in the sum
        if av <= 0:
            true_av = math.fsum(c / m for c in raw)  # report the user's units, not the scaled ones
            raise ValueError(
                f"EDAS needs a positive average per criterion; column {j + 1} has {true_av}"
            )
        for i in range(m):
            above = max(0.0, col[i] - av) / av
            below = max(0.0, av - col[i]) / av
            pda, nda = (above, below) if types[j] == "benefit" else (below, above)
            sp[i] += w[j] * pda
            sn[i] += w[j] * nda
    max_sp, max_sn = max(sp), max(sn)
    nsp = [p / max_sp if max_sp > 0 else 0.0 for p in sp]
    nsn = [1.0 - q / max_sn if max_sn > 0 else 1.0 for q in sn]
    scores = [(a + b) / 2.0 for a, b in zip(nsp, nsn)]
    return _result(
        "edas",
        alts,
        scores,
        True,
        {"normalization": "average_distance"},
        SP=sp,
        SN=sn,
        NSP=nsp,
        NSN=nsn,
    )
