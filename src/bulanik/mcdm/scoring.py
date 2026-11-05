"""Scoring ranking methods: CODAS, COPRAS, MARCOS, CoCoSo, WISP.

Methods that score each alternative from a crisp decision matrix (rows are
alternatives, columns are criteria), a vector of criteria weights and the type
of each criterion (`"benefit"` or `"cost"`). They share the signature and the
weight forms of the classical methods in [`bulanik.mcdm.ranking`][bulanik.mcdm.ranking]
and return a [`RankingResult`][bulanik.mcdm.results.RankingResult] whose
`ranking` is ordered best first. Higher scores are better throughout.

The methods are [`codas`][bulanik.mcdm.scoring.codas],
[`copras`][bulanik.mcdm.scoring.copras], [`marcos`][bulanik.mcdm.scoring.marcos],
[`cocoso`][bulanik.mcdm.scoring.cocoso] and [`wisp`][bulanik.mcdm.scoring.wisp].

References:
    Keshavarz Ghorabaee, Zavadskas, Turskis & Antucheviciene (2016), Economic
    Computation and Economic Cybernetics Studies and Research 50(3), 25-44;
    Zavadskas, Kaklauskas & Sarka (1994), Technological and Economic
    Development of Economy 1(3), 131-139;
    Stevic, Pamucar, Puska & Chatterjee (2020), Computers & Industrial
    Engineering 140, 106231; Yazdani, Zarate, Zavadskas & Turskis (2019),
    Management Decision 57(9), 2501-2519; Stanujkic, Popovic, Karabasevic,
    Meidute-Kavaliauskiene & Ulutas (2023), IEEE Transactions on Engineering
    Management 70(5), 1933-1944.
"""

from __future__ import annotations

import math
import sys
from typing import Mapping, Sequence

from .._validate import choose
from .ranking import (
    _column,
    _finite_all,
    _prepare,
    _result,
    _unit_param,
    _unit_scale,
)
from .results import BWMResult, RankingResult, WeightingResult

__all__ = ["codas", "copras", "marcos", "cocoso", "wisp"]


def _scaled_positive(col: list[float], method: str, name: str) -> list[float]:
    """Unit-scale a positive column; raise if the spread makes an entry underflow to 0."""
    out = _unit_scale(col)
    if not all(v > 0 for v in out):
        raise ValueError(
            f"{method}: criterion '{name}' spans too many orders of magnitude (an entry "
            f"underflows to 0 next to the column maximum), so rescale the matrix"
        )
    return out


def _codas_assessment(e: list[float], t: list[float], tau: float, psi: str) -> list[float]:
    """CODAS scores $H_i = \\sum_k h_{ik}$ from the Euclidean ``e`` and taxicab ``t`` distances."""
    m = len(e)
    h = []
    for i in range(m):
        total = 0.0
        for k in range(m):
            de = e[i] - e[k]
            counts = (abs(de) >= tau) == (psi == "equation")
            total += de + (t[i] - t[k] if counts else 0.0)
        h.append(total)
    return h


def codas(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
    tau: float = 0.02,
    psi: str = "equation",
) -> RankingResult:
    r"""CODAS (Keshavarz Ghorabaee et al. 2016): distances from the negative-ideal point.

    Entries are normalised linearly, $n_{ij} = x_{ij} / \max_i x_{ij}$ for a
    benefit and $\min_i x_{ij} / x_{ij}$ for a cost, and weighted,
    $r_{ij} = w_j n_{ij}$. The negative-ideal point is $ns_j = \min_i r_{ij}$;
    the Euclidean (primary) and taxicab (secondary) distances from it are

    $$E_i = \sqrt{\sum_j (r_{ij} - ns_j)^2}, \qquad T_i = \sum_j |r_{ij} - ns_j|.$$

    The relative assessment is

    $$h_{ik} = (E_i - E_k) + \psi(E_i - E_k)\,(T_i - T_k), \qquad
    \psi(x) = \begin{cases} 1 & |x| \ge \tau \\ 0 & |x| < \tau \end{cases}$$

    and the score is $H_i = \sum_k h_{ik}$, larger is better (the scores sum to
    0, so they can be negative). `extras` holds `E` and `T`.

    **Two conventions for $\psi$.** The paper is internally inconsistent. Its
    Eq. (10), printed above, adds the taxicab term when the Euclidean
    distances differ by at least $\tau$, although the text says the taxicab
    distance is the tie-breaker "if the difference ... is less than $\tau$".
    The paper's own Tables 4 and 8 (Examples 1 and 2) follow the text, i.e. the
    opposite branch, and Chen & Gou (2021) criticise the printed form. Both
    are available:

    - `psi="equation"` (default): Eq. (10) as printed, the taxicab term counts
      when $|E_i - E_k| \ge \tau$. It is the form printed in the paper.
    - `psi="tables"`: the convention of the paper's tables, the taxicab term
      counts when $|E_i - E_k| < \tau$. Use it to reproduce the paper's printed
      numbers (Example 1 gives $H = 0.1865, 0.6304, 0.3357, \dots$).

    Use `"tables"` to match a paper that follows the original tables and the
    default to match one that follows the equation; the distances $E$, $T$ are
    the same either way. The choice is recorded in `operators["psi"]`.

    Args:
        matrix: Alternatives by criteria (rows by columns), finite numbers.
            A benefit column needs entries $\ge 0$ with a maximum $> 0$ (a zero
            normalises to 0); a cost column needs entries $> 0$ (since
            $\min_i x_{ij} / x_{ij}$ is undefined at 0).
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
        tau: Threshold of $\psi$ in [0, 1]; the paper suggests 0.01 to 0.05 and
            uses 0.02.
        psi: `"equation"` (taxicab term when $|\Delta E| \ge \tau$, as printed) or
            `"tables"` (when $|\Delta E| < \tau$, as the paper's tables); see above.

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"normalization": "linear_max", "tau": "0.02", "psi": "equation"}`
        (the value of `tau` as text).

    Raises:
        TypeError: If a matrix entry, a weight or `tau` is a `bool` or `str` instead of a number.
        ValueError: If the matrix is empty, ragged or non-finite, the counts of
            names, types or weights do not match, names repeat, a mapping or
            result misses a criterion or names an unknown one, a type is not
            `"benefit"` or `"cost"`, the weights are negative or (without
            `normalize_weights`) do not sum to 1, an entry is negative, a cost
            column has a zero, a benefit column is all zeros (for fuzzy scores
            try `score="normalized"`), a column spans so many orders of magnitude
            that an entry underflows, `tau` is outside [0, 1] or not finite, or
            `psi` is not a valid name.

    Examples:
        >>> from bulanik.mcdm import codas
        >>> r = codas([[250, 16], [200, 12], [300, 32]], [0.5, 0.5],
        ...           ["cost", "benefit"], alternatives=["A", "B", "C"])
        >>> r.ranking
        ('C', 'B', 'A')
        >>> {k: round(v, 3) for k, v in r.scores.items()}
        {'A': -0.517, 'B': -0.179, 'C': 0.696}

    References:
        Keshavarz Ghorabaee, Zavadskas, Turskis & Antucheviciene (2016), A new
        combinative distance-based assessment (CODAS) method for multi-criteria
        decision-making, Economic Computation and Economic Cybernetics Studies
        and Research 50(3), 25-44. Chen & Gou (2021), Computational and Applied
        Mathematics 41(1), 11, doi:10.1007/s40314-021-01568-6, on the threshold function.
    """
    tau = _unit_param(tau, "tau")
    psi = choose(psi, ("equation", "tables"), "psi", "codas")
    x, alts, crit, types, w = _prepare(
        matrix, weights, criteria_types, alternatives, criteria, normalize_weights
    )
    m, n = len(x), len(x[0])
    hint = 'for fuzzy scores try score="normalized"'
    for j in range(n):
        col = _column(x, j)
        if any(c < 0 for c in col):
            raise ValueError(
                f"CODAS: criterion '{crit[j]}' has a negative entry; entries must be "
                f"non-negative in a benefit column and strictly positive in a cost column; {hint}"
            )
        if types[j] == "cost" and any(c == 0 for c in col):
            raise ValueError(
                f"CODAS: cost criterion '{crit[j]}' has a zero entry, so min/x is undefined; "
                f"a cost column needs strictly positive entries; {hint}"
            )
        if types[j] == "benefit" and max(col) <= 0:
            raise ValueError(
                f"CODAS: benefit criterion '{crit[j]}' is all zeros, so x/max is undefined; "
                f"a benefit column needs a strictly positive maximum; {hint}"
            )
    r = [[0.0] * n for _ in range(m)]
    for j in range(n):
        raw = _column(x, j)
        col = _unit_scale(raw)  # exact; ratios do not depend on it
        if any(v == 0 and c != 0 for v, c in zip(col, raw)):
            raise ValueError(
                f"CODAS: criterion '{crit[j]}' spans too many orders of magnitude (an entry "
                f"underflows to 0 next to the column maximum), so rescale the matrix"
            )
        hi, lo = max(col), min(col)
        for i in range(m):
            ratio = col[i] / hi if types[j] == "benefit" else lo / col[i]
            r[i][j] = w[j] * ratio
    ns = [min(r[i][j] for i in range(m)) for j in range(n)]
    e = [math.sqrt(sum((r[i][j] - ns[j]) ** 2 for j in range(n))) for i in range(m)]
    t = [sum(abs(r[i][j] - ns[j]) for j in range(n)) for i in range(m)]
    h = _codas_assessment(e, t, tau, psi)
    return _result(
        "codas",
        alts,
        h,
        True,
        {"normalization": "linear_max", "tau": str(tau), "psi": psi},
        E=e,
        T=t,
    )


def copras(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
) -> RankingResult:
    r"""COPRAS (Zavadskas, Kaklauskas & Sarka 1994): complex proportional assessment.

    Each column is divided by its sum and weighted, $d_{ij} = w_j x_{ij} /
    \sum_i x_{ij}$. Benefit and cost criteria are summed separately,
    $S^+_i = \sum_{j \in \text{benefit}} d_{ij}$ and
    $S^-_i = \sum_{j \in \text{cost}} d_{ij}$, and the relative significance is

    $$Q_i = S^+_i + \frac{\sum_k S^-_k}{S^-_i \sum_k 1 / S^-_k},$$

    larger is better (just $S^+_i$ without a cost criterion). `scores` holds
    $Q$; `extras` holds `S_plus`, `S_minus` and the utility degree
    `utility` $= 100\,Q_i / \max Q$ in percent. A cost criterion with weight 0
    is ignored, so a model whose cost criteria all have weight 0 is scored as
    having none.

    Args:
        matrix: Alternatives by criteria (rows by columns), non-negative finite
            numbers with a positive sum in every column.
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
        `operators == {"normalization": "sum"}`.

    Raises:
        TypeError: If a matrix entry or a weight is a `bool` or `str` instead of a number.
        ValueError: If the matrix is empty, ragged or non-finite, the counts of
            names, types or weights do not match, names repeat, a mapping or
            result misses a criterion or names an unknown one, a type is not
            `"benefit"` or `"cost"`, the weights are negative or (without
            `normalize_weights`) do not sum to 1, an entry is negative, a
            column sum is not positive, or an alternative has $S^-_i = 0$
            while a cost criterion carries weight (the error names it).

    Examples:
        >>> from bulanik.mcdm import copras
        >>> r = copras([[250, 16], [200, 12], [300, 32]], [0.5, 0.5],
        ...            ["cost", "benefit"], alternatives=["A", "B", "C"])
        >>> r.ranking
        ('C', 'B', 'A')
        >>> {k: round(v, 3) for k, v in r.scores.items()}
        {'A': 0.295, 'B': 0.303, 'C': 0.402}

    References:
        Zavadskas & Kaklauskas (1996), Determination of an efficient contractor
        by using the new method of multicriteria assessment, in Langford &
        Retik (eds.), The Organization and Management of Construction: Shaping
        Theory and Practice, Vol. 2, 94-104; Zavadskas, Kaklauskas & Sarka
        (1994), The new method of multicriteria complex proportional
        assessment of projects, Technological and Economic Development of
        Economy 1(3), 131-139. The $Q$ formula is Eq. (8) of Razavi Hajiagha,
        Hashemi & Zavadskas (2013), Technological and Economic Development of
        Economy 19(1), 22-37, doi:10.3846/20294913.2012.762953.
    """
    x, alts, crit, types, w = _prepare(
        matrix, weights, criteria_types, alternatives, criteria, normalize_weights
    )
    if any(c < 0 for row in x for c in row):
        raise ValueError(
            'COPRAS needs non-negative entries; for fuzzy scores try score="normalized"'
        )
    m, n = len(x), len(x[0])
    s_plus = [0.0] * m
    s_minus = [0.0] * m
    for j in range(n):
        if w[j] == 0:
            continue  # a zero-weight criterion is ignored, whatever its column holds
        col = _unit_scale(_column(x, j))  # exact; the share x / sum does not depend on it
        total = math.fsum(col)
        if not total > 0:
            raise ValueError(
                f"COPRAS needs a positive sum in every column; criterion '{crit[j]}' sums to 0"
            )
        target = s_plus if types[j] == "benefit" else s_minus
        for i in range(m):
            target[i] += w[j] * (col[i] / total)
    has_cost = any(t == "cost" and wj > 0 for t, wj in zip(types, w))
    if has_cost:
        for i in range(m):
            if not s_minus[i] > 0:
                raise ValueError(
                    f"COPRAS needs a positive cost sum for every alternative; alternative "
                    f"'{alts[i]}' has none on the weighted cost criteria"
                )
        s_min = min(s_minus)
        inv = math.fsum(s_min / v for v in s_minus)  # sum(1/S-) scaled by s_min: no overflow
        top = math.fsum(s_minus)
        q = [s_plus[i] + top * (s_min / s_minus[i]) / inv for i in range(m)]
    else:
        s_minus = [0.0] * m
        q = list(s_plus)
    _finite_all("copras", q, "a score")
    q_max = max(q)
    utility = [100.0 * v / q_max if q_max > 0 else 100.0 for v in q]
    return _result(
        "copras",
        alts,
        q,
        True,
        {"normalization": "sum"},
        S_plus=s_plus,
        S_minus=s_minus,
        utility=utility,
    )


def _positive(x: list[list[float]], method: str) -> None:
    if any(c <= 0 for row in x for c in row):
        raise ValueError(
            f"{method} needs strictly positive entries; for fuzzy scores try score=\"normalized\""
        )


def marcos(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
) -> RankingResult:
    r"""MARCOS (Stevic et al. 2020): measurement of alternatives and ranking according to compromise solution.

    The matrix is extended with the ideal solution AI (best value per
    criterion: the maximum of a benefit, the minimum of a cost) and the
    anti-ideal AAI (the worst value). Every entry is divided by the ideal
    value, $n_{ij} = x_{ij} / x_{AI,j}$ for a benefit and
    $x_{AI,j} / x_{ij}$ for a cost, and weighted, $v_{ij} = w_j n_{ij}$. With
    $S_i = \sum_j v_{ij}$ (also for the AAI and AI rows) the utility degrees
    are $K^-_i = S_i / S_{AAI}$ and $K^+_i = S_i / S_{AI}$. The utility
    functions are

    $$f(K^-_i) = \frac{K^+_i}{K^+_i + K^-_i}, \qquad
      f(K^+_i) = \frac{K^-_i}{K^+_i + K^-_i},$$

    and the score, larger is better, is

    $$f(K_i) = \frac{K^+_i + K^-_i}{1 + \frac{1 - f(K^+_i)}{f(K^+_i)} +
      \frac{1 - f(K^-_i)}{f(K^-_i)}}.$$

    `extras` holds `K_minus`, `K_plus` and `S`. Because $f(K^-_i)$ and
    $f(K^+_i)$ are the same for every alternative (they depend only on
    $S_{AAI} / S_{AI}$), the ranking equals the ranking by $S_i$.
    The entries must be strictly positive: a non-positive entry has no
    meaningful ratio to the ideal.
    Args:
        matrix: Alternatives by criteria (rows by columns), strictly positive finite numbers.
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
        `operators == {"normalization": "ideal_ratio"}`.

    Raises:
        TypeError: If a matrix entry or a weight is a `bool` or `str` instead of a number.
        ValueError: If the matrix is empty, ragged or non-finite, the counts of
            names, types or weights do not match, names repeat, a mapping or
            result misses a criterion or names an unknown one, a type is not
            `"benefit"` or `"cost"`, the weights are negative or (without
            `normalize_weights`) do not sum to 1, an entry is not strictly
            positive (for fuzzy scores try `score="normalized"`), or the
            ratios are so extreme that $S_{AAI}$ underflows to 0.

    Examples:
        >>> from bulanik.mcdm import marcos
        >>> r = marcos([[250, 16], [200, 12], [300, 32]], [0.5, 0.5],
        ...            ["cost", "benefit"], alternatives=["A", "B", "C"])
        >>> r.ranking
        ('C', 'B', 'A')
        >>> {k: round(v, 3) for k, v in r.scores.items()}
        {'A': 0.552, 'B': 0.583, 'C': 0.707}

    References:
        Stevic, Pamucar, Puska & Chatterjee (2020), Sustainable supplier
        selection in healthcare industries using a new MCDM method:
        Measurement of alternatives and ranking according to COmpromise
        solution (MARCOS), Computers & Industrial Engineering 140, 106231,
        doi:10.1016/j.cie.2019.106231. The equations follow the restatement in
        Ulutas, Karabasevic, Popovic, Stanujkic, Nguyen & Karakoy (2020),
        Mathematics 8(10), 1672, Eqs. (18)-(28), doi:10.3390/math8101672.
    """
    x, alts, crit, types, w = _prepare(
        matrix, weights, criteria_types, alternatives, criteria, normalize_weights
    )
    _positive(x, "MARCOS")
    m, n = len(x), len(x[0])
    s = [0.0] * (m + 2)  # alternatives, then AAI, then AI
    for j in range(n):
        col = _scaled_positive(_column(x, j), "MARCOS", crit[j])  # exact; ratios do not depend on it
        hi, lo = max(col), min(col)
        ai, aai = (hi, lo) if types[j] == "benefit" else (lo, hi)
        for i, v in enumerate([*col, aai, ai]):
            ratio = v / ai if types[j] == "benefit" else ai / v
            s[i] += w[j] * ratio
    s_aai, s_ai = s[m], s[m + 1]
    if not (s_aai > 0 and s_ai > 0):
        raise ValueError(
            "MARCOS: the weighted ratios to the ideal underflow to 0; "
            "the matrix entries span too many orders of magnitude, so rescale them"
        )
    k_minus = [s[i] / s_aai for i in range(m)]
    k_plus = [s[i] / s_ai for i in range(m)]
    f = []
    for km, kp in zip(k_minus, k_plus):
        f_minus = kp / (kp + km)
        f_plus = km / (kp + km)
        f.append((kp + km) / (1 + (1 - f_plus) / f_plus + (1 - f_minus) / f_minus))
    return _result(
        "marcos",
        alts,
        f,
        True,
        {"normalization": "ideal_ratio"},
        K_minus=k_minus,
        K_plus=k_plus,
        S=s[:m],
    )


def _cocoso_parts(
    s: list[float],
    p: list[float],
    lam: float,
    names: Sequence[str] | None = None,
    reason: str | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """CoCoSo appraisal scores $(k_a, k_b, k_c, k)$ from the sequences ``s`` and ``p``.

    Raises ``ValueError`` (naming the alternative, ``A<i>`` without ``names``) if one is
    worst on every weighted criterion, since $k_b$ then divides by zero. ``reason``
    replaces the default explanation ("is worst on every criterion") in that message.
    """
    m = len(s)
    for i in range(m):
        if not (s[i] > 0 and p[i] > 0):
            name = names[i] if names is not None else f"A{i + 1}"
            raise ValueError(
                f"CoCoSo: alternative '{name}' "
                + (reason or "is worst on every criterion")
                + ", so k_b divides by zero"
            )
    s_min, p_min = min(s), min(p)
    total = math.fsum(pi + si for pi, si in zip(p, s))
    k_a = [(p[i] + s[i]) / total for i in range(m)]
    k_b = [s[i] / s_min + p[i] / p_min for i in range(m)]
    denom = lam * max(s) + (1 - lam) * max(p)
    k_c = [(lam * s[i] + (1 - lam) * p[i]) / denom for i in range(m)]
    k = [
        (k_a[i] * k_b[i] * k_c[i]) ** (1 / 3) + (k_a[i] + k_b[i] + k_c[i]) / 3
        for i in range(m)
    ]
    return k_a, k_b, k_c, k


def _cocoso_appraisal(
    s: list[float], p: list[float], lam: float, names: Sequence[str] | None = None
) -> list[float]:
    """Final CoCoSo scores $k_i$ from the weighted-sum ``s`` and weighted-power ``p`` sequences."""
    return _cocoso_parts(s, p, lam, names)[3]


def cocoso(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
    lam: float = 0.5,
) -> RankingResult:
    r"""CoCoSo (Yazdani et al. 2019): combined compromise solution.

    Entries are scaled to $[0, 1]$ by min-max normalisation,
    $r_{ij} = (x_{ij} - \min_i x_{ij}) / (\max_i x_{ij} - \min_i x_{ij})$ for a
    benefit and $(\max_i x_{ij} - x_{ij}) / (\max_i x_{ij} - \min_i x_{ij})$ for
    a cost; a **constant column** gets $r_{ij} = 0$ for every alternative,
    which adds nothing to either sum below. The weighted-sum and
    weighted-power comparability sequences are

    $$S_i = \sum_j w_j r_{ij}, \qquad P_i = \sum_{j:\, w_j > 0} r_{ij}^{\,w_j},$$

    (a criterion with weight 0 is left out of both). Three appraisal scores
    combine them,

    $$k_{ia} = \frac{P_i + S_i}{\sum_i (P_i + S_i)}, \qquad
      k_{ib} = \frac{S_i}{\min_i S_i} + \frac{P_i}{\min_i P_i}, \qquad
      k_{ic} = \frac{\lambda S_i + (1 - \lambda) P_i}{\lambda \max_i S_i + (1 - \lambda) \max_i P_i},$$

    and the score, larger is better, is

    $$k_i = (k_{ia}\, k_{ib}\, k_{ic})^{1/3} + \tfrac13 (k_{ia} + k_{ib} + k_{ic}).$$

    `extras` holds `S`, `P`, `k_a`, `k_b` and `k_c`. Entries may be any finite
    numbers. If some alternative is worst on every weighted criterion then
    $\min S = \min P = 0$ and $k_{ib}$ is undefined (Rasoanaivo et al. 2024
    report this failure of the method); a `ValueError` names that alternative.
    This also happens when all alternatives are identical, and it
    always happens when only one criterion carries weight (a single effective
    criterion): the worst alternative on it is worst on every effective one, so
    CoCoSo cannot rank such a model.

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
        lam: Balance $\lambda \in [0, 1]$ between the weighted sum and the
            weighted power in $k_{ic}$; the paper suggests 0.5.

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"normalization": "min_max", "lambda": "0.5"}` (the value
        of `lam` as text).

    Raises:
        TypeError: If a matrix entry, a weight or `lam` is a `bool` or `str` instead of a number.
        ValueError: If the matrix is empty, ragged or non-finite, the counts of
            names, types or weights do not match, names repeat, a mapping or
            result misses a criterion or names an unknown one, a type is not
            `"benefit"` or `"cost"`, the weights are negative or (without
            `normalize_weights`) do not sum to 1, `lam` is outside [0, 1] or
            not finite, or an alternative is worst on every weighted criterion
            (the error names it).

    Examples:
        >>> from bulanik.mcdm import cocoso
        >>> r = cocoso([[7, 8], [8, 7], [9, 9]], [0.6, 0.4], ["benefit", "cost"],
        ...            alternatives=["A", "B", "C"])
        >>> r.ranking
        ('B', 'C', 'A')
        >>> {k: round(v, 3) for k, v in r.scores.items()}
        {'A': 1.408, 'B': 3.788, 'C': 2.758}

    References:
        Yazdani, Zarate, Zavadskas & Turskis (2019), A combined compromise
        solution (CoCoSo) method for multi-criteria decision-making problems,
        Management Decision 57(9), 2501-2519, doi:10.1108/MD-05-2017-0458.
        On the zero-division failure: Rasoanaivo, Rakotonirainy & Zarate
        (2024), Combined compromise for ideal solution (CoCoFISo), Expert
        Systems with Applications 251, 124079, doi:10.1016/j.eswa.2024.124079.
    """
    lam = _unit_param(lam, "lam")
    x, alts, _, types, w = _prepare(
        matrix, weights, criteria_types, alternatives, criteria, normalize_weights
    )
    m, n = len(x), len(x[0])
    s = [0.0] * m
    p = [0.0] * m
    for j in range(n):
        if w[j] == 0:
            continue
        col = _unit_scale(_column(x, j))  # exact; r does not depend on it, differences cannot overflow
        hi, lo = max(col), min(col)
        if hi == lo:
            continue  # constant column: r = 0 for all, neutral for S and P
        for i in range(m):
            ref = lo if types[j] == "benefit" else hi
            gap = abs(col[i] - ref)
            # A gap of rounding size is 0: r ** w would blow it up (2e-17 ** 0.25 ~ 7e-5).
            if gap <= 4 * sys.float_info.epsilon * max(abs(col[i]), abs(ref)):
                gap = 0.0
            r = gap / (hi - lo)
            s[i] += w[j] * r
            p[i] += r ** w[j]
    k_a, k_b, k_c, k = _cocoso_parts(s, p, lam, alts)
    return _result(
        "cocoso",
        alts,
        k,
        True,
        {"normalization": "min_max", "lambda": str(lam)},
        S=s,
        P=p,
        k_a=k_a,
        k_b=k_b,
        k_c=k_c,
    )


def wisp(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
    product: str = "power",
) -> RankingResult:
    r"""WISP (Stanujkic et al. 2023): integrated simple weighted sum product method.

    Entries are divided by the column maximum for every criterion,
    $r_{ij} = x_{ij} / \max_i x_{ij}$. With $\Omega_{\max}$ the benefit and
    $\Omega_{\min}$ the cost criteria (those with positive weight), four
    utility measures combine the weighted sums and weighted products:

    $$u^{sd}_i = \sum_{\Omega_{\max}} w_j r_{ij} - \sum_{\Omega_{\min}} w_j r_{ij}, \qquad
      u^{pd}_i = \prod_{\Omega_{\max}} r_{ij}^{w_j} - \prod_{\Omega_{\min}} r_{ij}^{w_j},$$

    $$u^{sr}_i = \frac{\sum_{\Omega_{\max}} w_j r_{ij}}{\sum_{\Omega_{\min}} w_j r_{ij}}, \qquad
      u^{pr}_i = \frac{\prod_{\Omega_{\max}} r_{ij}^{w_j}}{\prod_{\Omega_{\min}} r_{ij}^{w_j}}.$$

    Each is recalculated, $\bar u_i = (1 + u_i) / (1 + \max_k u_k)$, and the
    score, larger is better, is $u_i = \tfrac14 (\bar u^{sd}_i + \bar u^{pd}_i +
    \bar u^{sr}_i + \bar u^{pr}_i)$. `extras` holds the four measures `u_sd`,
    `u_pd`, `u_sr`, `u_pr` and their recalculated forms `ubar_sd`, `ubar_pd`,
    `ubar_sr`, `ubar_pr`.

    **Only benefit or only cost criteria.** For benefit criteria only, the
    paper (Stanujkic 2023, Sec. 2.3.1) uses two measures, the weighted sum
    and product, and averages their recalculated values. The convention here
    reproduces exactly that: in a difference an empty group contributes 0
    (sum or product); in a ratio an empty denominator group is replaced by 1,
    and an empty numerator is 0 for sums and 1 for products. For benefit
    criteria only, $u^{sd} = u^{sr}$ and $u^{pd} = u^{pr}$, so the mean of the
    four is the paper's mean of two. For cost criteria only the paper is
    silent; the same convention gives $u^{sd} = -\sum w r$,
    $u^{pd} = -\prod r^{w}$, $u^{sr} = 0$ and $u^{pr} = 1 / \prod r^{w}$.
    A criterion with weight 0 is ignored.

    **Degenerate matrices.** A measure with $1 + \max_k u_k = 0$ (every
    alternative has $r = 1$ on cost criteria only, so $u^{sd} = u^{pd} = -1$
    for all) is $0/0$ and all alternatives are equal on it, so $\bar u = 1$ for
    every alternative. A constant matrix, or a single alternative, therefore
    gives equal scores.

    **Two conventions for the products.** The cited formulas write the weighted
    product of the normalised ratings; the worked example of the CC-BY
    restatement (Stanujkic 2023, Table 7) evaluates it as $\prod_j r_{ij} w_j$
    (multiplication, values around $10^{-7}$) rather than $\prod_j r_{ij}^{w_j}$,
    which is the weighted product model and equals the WASPAS product in
    the same article's Table 5. Both are available: `product="power"`
    (default) uses $r_{ij}^{w_j}$; `product="times"` uses $r_{ij} w_j$ and
    reproduces Table 7. The choice is recorded in `operators["product"]`.
    Args:
        matrix: Alternatives by criteria (rows by columns), strictly positive finite numbers.
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
        product: `"power"` ($r_{ij}^{w_j}$) or `"times"` ($r_{ij} w_j$); see above.

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"normalization": "linear_max", "product": "power"}`.

    Raises:
        TypeError: If a matrix entry or a weight is a `bool` or `str` instead of a number.
        ValueError: If the matrix is empty, ragged or non-finite, the counts of
            names, types or weights do not match, names repeat, a mapping or
            result misses a criterion or names an unknown one, a type is not
            `"benefit"` or `"cost"`, the weights are negative or (without
            `normalize_weights`) do not sum to 1, an entry is not strictly
            positive (for fuzzy scores try `score="normalized"`), `product` is
            not a valid name, or the product ratio overflows.

    Examples:
        >>> from bulanik.mcdm import wisp
        >>> r = wisp([[250, 16], [200, 12], [300, 32]], [0.5, 0.5],
        ...          ["cost", "benefit"], alternatives=["A", "B", "C"])
        >>> r.ranking
        ('C', 'A', 'B')
        >>> {k: round(v, 3) for k, v in r.scores.items()}
        {'A': 0.829, 'B': 0.827, 'C': 1.0}

    References:
        Stanujkic, Popovic, Karabasevic, Meidute-Kavaliauskiene & Ulutas
        (2023), An integrated simple weighted sum product method - WISP, IEEE
        Transactions on Engineering Management 70(5), 1933-1944 (online 2021),
        doi:10.1109/TEM.2021.3075783. Equations (16)-(31) as restated in M.
        Stanujkic (2023), Comparative analysis of MULTIMOORA, WASPAS and WISP
        methods: the case of candidate selection, Journal of Process Management
        and New Technologies 11(3-4), 79-88, doi:10.5937/jpmnt11-47703.
    """
    product = choose(product, ("power", "times"), "product", "wisp")
    x, alts, crit, types, w = _prepare(
        matrix, weights, criteria_types, alternatives, criteria, normalize_weights
    )
    _positive(x, "WISP")
    m, n = len(x), len(x[0])
    # log of each factor; sums and products over a group are kept as exp of a log sum
    sums = {"benefit": [0.0] * m, "cost": [0.0] * m}
    logs = {"benefit": [0.0] * m, "cost": [0.0] * m}
    used = {"benefit": False, "cost": False}  # does the group hold a criterion with weight?
    for j in range(n):
        if w[j] == 0:
            continue
        used[types[j]] = True
        col = _scaled_positive(_column(x, j), "WISP", crit[j])
        top = max(col)
        for i in range(m):
            r = col[i] / top
            sums[types[j]][i] += w[j] * r
            logs[types[j]][i] += w[j] * math.log(r) if product == "power" else math.log(w[j] * r)
    try:
        prod = {t: [math.exp(v) if used[t] else 0.0 for v in logs[t]] for t in logs}
        u_pr = [math.exp(logs["benefit"][i] - logs["cost"][i]) for i in range(m)]
    except OverflowError:
        raise ValueError(
            "WISP: the product ratio overflows; the weighted products of the cost "
            "criteria are too small, so rescale the matrix"
        ) from None
    sb, sc = sums["benefit"], sums["cost"]
    u_sd = [sb[i] - sc[i] for i in range(m)]
    u_pd = [prod["benefit"][i] - prod["cost"][i] for i in range(m)]
    u_sr = [sb[i] / (sc[i] if sc[i] > 0 else 1.0) for i in range(m)]
    measures = {"sd": u_sd, "pd": u_pd, "sr": u_sr, "pr": u_pr}
    _finite_all("wisp", [v for u in measures.values() for v in u], "a utility measure")
    bar = {}
    for name, u in measures.items():
        denom = 1 + max(u)
        if denom < 1e-12:
            # u >= -1 always (r <= 1, weights sum to 1), so 1 + max u = 0 means every
            # alternative sits at -1: the measure is 0/0 and all are equal
            bar[name] = [1.0] * m
            continue
        bar[name] = [(1 + v) / denom for v in u]
    score = [sum(bar[k][i] for k in bar) / 4 for i in range(m)]
    return _result(
        "wisp",
        alts,
        score,
        True,
        {"normalization": "linear_max", "product": product},
        u_sd=u_sd,
        u_pd=u_pd,
        u_sr=u_sr,
        u_pr=u_pr,
        ubar_sd=bar["sd"],
        ubar_pd=bar["pd"],
        ubar_sr=bar["sr"],
        ubar_pr=bar["pr"],
    )
