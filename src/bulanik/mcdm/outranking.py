"""Outranking ranking methods: ELECTRE I with net concordance and discordance.

Methods that compare alternatives pair by pair instead of scoring each one
alone. [`electre`][bulanik.mcdm.outranking.electre] takes a crisp decision
matrix (rows are alternatives, columns are criteria), a vector of criteria
weights and the type of each criterion (`"benefit"` or `"cost"`), shares the
signature and the weight forms of the methods in
[`bulanik.mcdm.ranking`][bulanik.mcdm.ranking] and returns a
[`RankingResult`][bulanik.mcdm.results.RankingResult] whose `ranking` is ordered
best first.

References:
    Roy (1968), Revue francaise d'informatique et de recherche operationnelle
    2(8), 57-75; Hwang & Yoon (1981), *Multiple Attribute Decision Making*,
    Springer; van Delft & Nijkamp (1977), *Multi-criteria Analysis and Regional
    Decision-making*, Martinus Nijhoff.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .._validate import TOL, choose
from .ranking import _column, _finite_all, _prepare, _unit_param
from ._common import rank_order
from .results import BWMResult, RankingResult, WeightingResult

__all__ = ["electre"]


def electre(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
    cost: str = "reverse",
    c_threshold: float | None = None,
    d_threshold: float | None = None,
) -> RankingResult:
    r"""ELECTRE I (Roy 1968) with a complete ranking from net concordance and discordance.

    The matrix is vector-normalised and weighted as in TOPSIS,
    $v_{ij} = w_j x_{ij} / \sqrt{\sum_i x_{ij}^2}$ (a zero column gives 0). For
    an ordered pair of alternatives $k \ne l$ the concordance set is
    $C_{kl} = \{j : x_{kj} \ge x_{lj}\}$ for a benefit criterion
    ($x_{kj} \le x_{lj}$ for a cost one) and the discordance set $D_{kl}$ is its
    complement. The concordance and discordance indices are

    $$c_{kl} = \sum_{j \in C_{kl}} w_j, \qquad
    d_{kl} = \frac{\max_{j \in D_{kl}} |v_{kj} - v_{lj}|}{\max_j |v_{kj} - v_{lj}|},$$

    with $d_{kl} = 0$ when $D_{kl}$ is empty or the denominator is 0 (identical
    alternatives). The denominator is the largest weighted difference of the
    pair over all criteria, as in Hwang & Yoon (1981), not a per-criterion
    range. The thresholds $\bar c$ and $\bar d$ are the means of $c_{kl}$ and
    $d_{kl}$ over all pairs $k \ne l$ unless given. Alternative $k$ outranks $l$ when

    $$e_{kl} = [c_{kl} \ge \bar c]\,[d_{kl} \le \bar d],$$

    Two entries of a criterion are tied, and concordant both ways, when they
    differ by at most $10^{-12}$ times the larger magnitude (so scaling a
    column cannot create or break a tie by one ulp). The thresholds are applied
    with a slack of $10^{-12}$ so that a pair sitting on a threshold in exact
    arithmetic is not lost to rounding.

    The outranking relation is partial. For a complete ranking the net
    indices of van Delft & Nijkamp (1977) are used,

    $$C_k = \sum_{l \ne k} c_{kl} - \sum_{l \ne k} c_{lk}, \qquad
    D_k = \sum_{l \ne k} d_{kl} - \sum_{l \ne k} d_{lk},$$

    where a large $C_k$ and a small $D_k$ are good. Sources rank by $C_k$
    (descending) and by $D_k$ (ascending) and pick the alternative that is best
    on both; here the two are combined into one score so that a full ranking
    always exists,

    $$S_k = C_k - D_k,$$

    larger is better. The net indices do not depend on the thresholds, so
    neither do `scores` and `ranking`; the thresholds only shape `outranks`.
    `extras` holds `net_concordance`, `net_discordance`, `concordance` and
    `discordance` (each `{alternative: {other: value}}`) and `outranks`
    (`{alternative: tuple of the alternatives it outranks}`).

    The criterion types are the user's call: whether a criterion is a benefit
    or a cost is not inferred from the data, and the published example in the
    tests assumes its types.

    **Cost criteria.** Two treatments are in use, selected by `cost`:

    - `"reverse"` (default, Hwang & Yoon): a cost column is kept and the
      comparison reversed ($\le$ instead of $\ge$).
    - `"inverse"` (e.g. Cakin & Ozdemir 2013): a cost column is replaced by
      $1/x_{ij}$ before the vector norm and then compared like a benefit. It
      needs $x_{ij} > 0$ in every cost column. The concordance indices are
      the same; the discordance indices (and so the scores) differ.

    The choice is recorded in `operators["cost"]`.

    Args:
        matrix: Alternatives by criteria (rows by columns), finite numbers; at
            least two rows.
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
        cost: `"reverse"` or `"inverse"`; see above.
        c_threshold: Concordance threshold in [0, 1]; the mean of $c_{kl}$ by default.
        d_threshold: Discordance threshold in [0, 1]; the mean of $d_{kl}$ by default.

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"normalization": "vector", "cost": "reverse", "score":
        "net_concordance_minus_net_discordance", "c_threshold": "mean",
        "d_threshold": "mean"}` (a given threshold appears as the text of its value).

    Raises:
        TypeError: If a matrix entry, a weight or a threshold is a `bool` or `str` instead of a number.
        ValueError: If the matrix is empty, ragged or non-finite or has fewer
            than two rows, the counts of names, types or weights do not match,
            names repeat, a mapping or result misses a criterion or names an
            unknown one, a type is not `"benefit"` or `"cost"`, the weights are
            negative or (without `normalize_weights`) do not sum to 1, a
            threshold is outside [0, 1] or not finite, `cost` is not a valid name,
            or (`cost="inverse"`) a cost column has an entry that is not strictly positive.

    Examples:
        >>> from bulanik.mcdm import electre
        >>> r = electre([[7, 9], [8, 7], [9, 8]], [0.6, 0.4], ["benefit", "cost"],
        ...             alternatives=["A", "B", "C"])
        >>> r.ranking
        ('C', 'B', 'A')
        >>> r.extras["outranks"]
        {'A': (), 'B': ('A',), 'C': ('A',)}
        >>> {k: round(v, 3) for k, v in r.extras["net_concordance"].items()}
        {'A': -2.0, 'B': 0.8, 'C': 1.2}
        >>> {k: round(v, 3) for k, v in r.extras["net_discordance"].items()}
        {'A': 2.0, 'B': -0.667, 'C': -1.333}

    References:
        Roy (1968), Classement et choix en presence de points de vue multiples
        (la methode ELECTRE), Revue francaise d'informatique et de recherche
        operationnelle 2(8), 57-75, doi:10.1051/ro/196802V100571. Hwang & Yoon
        (1981), *Multiple Attribute Decision Making: Methods and Applications*,
        Springer, pp. 115-127. van Delft & Nijkamp (1977), *Multi-criteria
        Analysis and Regional Decision-making*, Martinus Nijhoff. Net indices as
        applied by Cakin & Ozdemir (2013), Afyon Kocatepe Universitesi IIBF
        Dergisi 15(2), 339-361.
    """
    cost = choose(cost, ("reverse", "inverse"), "cost handling", "electre")
    cbar_given = None if c_threshold is None else _unit_param(c_threshold, "c_threshold")
    dbar_given = None if d_threshold is None else _unit_param(d_threshold, "d_threshold")
    x, alts, _, types, w = _prepare(
        matrix, weights, criteria_types, alternatives, criteria, normalize_weights
    )
    m, n = len(x), len(x[0])
    if m < 2:
        raise ValueError("ELECTRE needs at least two alternatives to compare")
    v = [[0.0] * n for _ in range(m)]
    for j in range(n):
        col = _column(x, j)
        if cost == "inverse" and types[j] == "cost":
            if any(c <= 0 for c in col):
                raise ValueError(
                    f"ELECTRE with cost=\"inverse\" needs strictly positive entries in a cost "
                    f"column; criterion {j + 1} has none or a nonpositive one (use cost=\"reverse\")"
                )
            col = [1.0 / c for c in col]
        norm = math.hypot(*col)  # no overflow to inf before the square root
        _finite_all("electre", [norm], f"the norm of criterion {j + 1}")
        for i in range(m):
            v[i][j] = w[j] * (col[i] / norm) if norm > 0 else 0.0
    c = [[0.0] * m for _ in range(m)]
    d = [[0.0] * m for _ in range(m)]
    for k in range(m):
        for l in range(m):
            if k == l:
                continue
            worst = overall = 0.0
            for j in range(n):
                a, b = x[k][j], x[l][j]
                tie = abs(a - b) <= TOL * max(abs(a), abs(b))  # relative: scaling a column keeps ties
                gap = 0.0 if tie else abs(v[k][j] - v[l][j])
                overall = max(overall, gap)
                concordant = tie or (a >= b if types[j] == "benefit" else a <= b)
                if concordant:
                    c[k][l] += w[j]
                else:
                    worst = max(worst, gap)
            d[k][l] = worst / overall if overall > 0 else 0.0
    pairs = [(k, l) for k in range(m) for l in range(m) if k != l]
    cbar = sum(c[k][l] for k, l in pairs) / len(pairs) if cbar_given is None else cbar_given
    dbar = sum(d[k][l] for k, l in pairs) / len(pairs) if dbar_given is None else dbar_given
    net_c = [sum(c[k][l] - c[l][k] for l in range(m) if l != k) for k in range(m)]
    net_d = [sum(d[k][l] - d[l][k] for l in range(m) if l != k) for k in range(m)]
    scores = [a - b for a, b in zip(net_c, net_d)]
    _finite_all("electre", scores, "a score")
    outranks = {
        alts[k]: tuple(
            alts[l] for l in range(m) if l != k and c[k][l] >= cbar - TOL and d[k][l] <= dbar + TOL
        )
        for k in range(m)
    }
    sc = dict(zip(alts, scores))
    return RankingResult(
        method="electre",
        alternatives=tuple(alts),
        scores=sc,
        ranking=rank_order(sc, True),
        operators={
            "normalization": "vector",
            "cost": cost,
            "score": "net_concordance_minus_net_discordance",
            "c_threshold": "mean" if cbar_given is None else str(cbar_given),
            "d_threshold": "mean" if dbar_given is None else str(dbar_given),
        },
        extras={
            "net_concordance": dict(zip(alts, net_c)),
            "net_discordance": dict(zip(alts, net_d)),
            "concordance": {alts[k]: {alts[l]: c[k][l] for l in range(m) if l != k} for k in range(m)},
            "discordance": {alts[k]: {alts[l]: d[k][l] for l in range(m) if l != k} for k in range(m)},
            "outranks": outranks,
        },
    )
