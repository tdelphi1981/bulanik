r"""Criteria weighting: AHP, Buckley fuzzy AHP, SWARA, PIPRECIA, entropy and CRITIC.

Six methods that turn judgements or data into criteria weights summing to 1,
each returning a [`WeightingResult`][bulanik.mcdm.results.WeightingResult]:

| Method | Input | Kind |
|---|---|---|
| [`ahp`][bulanik.mcdm.weighting.ahp] | crisp pairwise comparison matrix | subjective |
| [`fuzzy_ahp`][bulanik.mcdm.weighting.fuzzy_ahp] | matrix of triangular numbers | subjective |
| [`swara`][bulanik.mcdm.weighting.swara] | comparative importances of an ordered list | subjective |
| [`piprecia`][bulanik.mcdm.weighting.piprecia] | relative importances of a list in any order (crisp or TFN) | subjective |
| [`entropy`][bulanik.mcdm.weighting.entropy] | decision matrix | objective |
| [`critic`][bulanik.mcdm.weighting.critic] | decision matrix | objective |

The AHP-based methods report Saaty's consistency ratio $CR = CI / RI$ with
$CI = (\lambda_{\max} - n) / (n - 1)$ and the random index $RI$ of Saaty's table
for the matrix order $n$:

| $n$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| $RI$ | 0 | 0 | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 |

so matrices larger than 10 are rejected, and for $n \le 2$ the ratio is 0.
The best-worst method is in [`bulanik.lumix`][bulanik.lumix] and
[`bulanik.sezgi`][bulanik.sezgi].

References:
    Saaty (1980), *The Analytic Hierarchy Process*; Buckley (1985), Fuzzy Sets
    and Systems 17(3), 233-247; Kersuliene et al. (2010), Journal of Business
    Economics and Management 11(2), 243-258; Diakoulaki, Mavrotas &
    Papayannakis (1995), Computers & Operations Research 22(7), 763-770;
    Stanujkic et al. (2017), Romanian Journal of Economic Forecasting 20(4),
    116-133; Stevic et al. (2018), Symmetry 10(11), 586.
"""

from __future__ import annotations

import math
import sys
from typing import Sequence

from .._validate import choose, finite
from ..aggregate import wa, wg
from ..numbers import (
    FFN,
    IFN,
    IT2TrFN,
    IVFFN,
    IVIFN,
    IVNeutrosophicFN,
    IVPFN,
    IVQROFN,
    PFN,
    QROFN,
    TFN,
    FuzzyNumber,
    NeutrosophicFN,
    PictureFN,
    SphericalFN,
    TrFN,
)
from ._common import check_matrix, check_types, default_names, rank_order
from .results import WeightingResult

__all__ = ["ahp", "fuzzy_ahp", "ahp_fuzzy_numbers", "ahp_interval_valued", "swara", "entropy", "piprecia", "critic"]

# Saaty's random consistency index by matrix order.
_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
_RECIPROCAL_TOL = 1e-9  # relative, to the larger of the two entries' bounds
_DISCRIMINATION_EPS = 8 * sys.float_info.epsilon
# every score name some fuzzy number kind accepts, in first-seen order
_ALL_SCORES = tuple(dict.fromkeys(
    name
    for kind in (TFN, TrFN, QROFN, IFN, PFN, FFN, IVQROFN, IVIFN, IVPFN, IVFFN, SphericalFN, PictureFN, NeutrosophicFN, IVNeutrosophicFN, IT2TrFN)
    for name in kind.SCORES
))


def _names(criteria: Sequence[str] | None, n: int) -> list[str]:
    if criteria is None:
        return default_names("C", n)
    names = list(criteria)
    if len(names) != n:
        raise ValueError(f"expected {n} criteria names, got {len(names)}")
    if len(set(names)) != n:
        raise ValueError("criteria names must be unique")
    return names


def _result(method, names, w, operators, **kw) -> WeightingResult:
    weights = dict(zip(names, w))
    return WeightingResult(
        method=method,
        criteria=tuple(names),
        weights=weights,
        ranking=rank_order(weights, True),
        operators=operators,
        **kw,
    )


def _eigen(a: list[list[float]]) -> tuple[list[float], float]:
    """Principal eigenvector (sum 1) by power iteration, and Saaty's CR."""
    n = len(a)
    w = [1.0 / n] * n
    converged = False
    for _ in range(1000):
        aw = [sum(a[i][j] * w[j] for j in range(n)) for i in range(n)]
        s = sum(aw)
        nxt = [x / s for x in aw]
        done = max(abs(x - y) for x, y in zip(nxt, w)) < 1e-12
        w = nxt
        if done:
            converged = True
            break
    if not converged:
        raise ValueError(
            "power iteration did not converge in 1000 iterations; "
            "the matrix may be extremely inconsistent"
        )
    aw = [sum(a[i][j] * w[j] for j in range(n)) for i in range(n)]
    lmax = sum(aw[i] / w[i] for i in range(n)) / n
    if n <= 2:
        return w, 0.0
    ci = (lmax - n) / (n - 1)
    return w, max(ci, 0.0) / _RI[n]


def _square(matrix: Sequence[Sequence], what: str) -> list[list]:
    rows = [list(r) for r in matrix]
    n = len(rows)
    if n == 0 or any(len(r) != n for r in rows):
        raise ValueError(f"{what} matrix must be square")
    if n > 10:
        raise ValueError(f"{what} supports at most 10 criteria (no random index beyond 10), got {n}")
    return rows


def ahp(
    matrix: Sequence[Sequence[float]], criteria: Sequence[str] | None = None
) -> WeightingResult:
    r"""Saaty's AHP: principal-eigenvector weights and consistency ratio.

    The weights are the principal eigenvector of the reciprocal comparison
    matrix $A$ ($a_{ij}$ = importance of criterion $i$ over $j$, $a_{ji} =
    1/a_{ij}$), normalised to sum 1 (power iteration). The consistency ratio is
    $CR = \max(0, (\lambda_{\max} - n)/(n - 1)) / RI_n$ with Saaty's random index
    (see the [module][bulanik.mcdm.weighting] table); it is 0 for $n \le 2$.
    A ratio below about 0.1 is conventionally acceptable.

    Args:
        matrix: Square, positive, reciprocal matrix with ones on the diagonal,
            of order at most 10.
        criteria: Criterion names; `C1..Cn` by default.

    Returns:
        A [`WeightingResult`][bulanik.mcdm.results.WeightingResult] with
        `consistency_ratio` set.

    Raises:
        TypeError: If an entry is a `bool` or `str` instead of a number.
        ValueError: If the matrix is not square, is larger than 10, has an
            entry that is not finite and positive, has a diagonal other than 1
            or is not reciprocal (to relative tolerance 1e-9), `criteria` has
            the wrong length or duplicates, or power iteration does not
            converge.

    Examples:
        >>> from bulanik.mcdm import ahp
        >>> r = ahp([[1, 3, 5], [1/3, 1, 2], [1/5, 1/2, 1]], ["cost", "quality", "speed"])
        >>> {k: round(v, 3) for k, v in r.weights.items()}
        {'cost': 0.648, 'quality': 0.23, 'speed': 0.122}
        >>> round(r.consistency_ratio, 3)
        0.003
        >>> r.ranking
        ('cost', 'quality', 'speed')

    References:
        Saaty (1980), *The Analytic Hierarchy Process*, McGraw-Hill.
    """
    rows = _square(matrix, "AHP")
    n = len(rows)
    a = [[finite(x, "AHP entry") for x in r] for r in rows]
    if any(x <= 0 for r in a for x in r):
        raise ValueError("AHP entries must be positive")
    if any(abs(a[i][i] - 1.0) > _RECIPROCAL_TOL for i in range(n)):
        raise ValueError("AHP diagonal must be 1")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] * a[j][i] - 1.0) > _RECIPROCAL_TOL:
                raise ValueError(
                    f"AHP matrix must be reciprocal: a[{i}][{j}]*a[{j}][{i}] != 1"
                )
    names = _names(criteria, n)
    w, cr = _eigen(a)
    return _result(
        "ahp", names, w, {"weights": "eigenvector", "consistency": "saaty_ri"},
        consistency_ratio=cr,
    )


def fuzzy_ahp(
    matrix: Sequence[Sequence[TFN]],
    criteria: Sequence[str] | None = None,
    *,
    defuzzification: str = "centroid",
) -> WeightingResult:
    r"""Buckley's fuzzy AHP (geometric mean of rows) with TFN judgements.

    For each row $i$ the fuzzy geometric mean $\tilde r_i = (\prod_j
    \tilde a_{ij})^{1/n}$ is taken componentwise; the fuzzy weight is
    $\tilde w_i = \tilde r_i \otimes (\sum_k \tilde r_k)^{-1}$. Each $\tilde
    w_i$ is defuzzified with `defuzzification` and the crisp values are
    normalised to sum 1. The consistency ratio is Saaty's, computed on the
    defuzzified matrix (see the [module][bulanik.mcdm.weighting] table).

    Args:
        matrix: Square matrix of [`TFN`][bulanik.numbers.TFN] with $l > 0$,
            reciprocal ($\tilde a_{ji} = 1/\tilde a_{ij}$, to relative tolerance
            1e-9), of order at most 10.
        criteria: Criterion names; `C1..Cn` by default.
        defuzzification: Score name of [`TFN`][bulanik.numbers.TFN] used to
            defuzzify (`"centroid"`, `"graded_mean"` or `"mean"`).

    Returns:
        A [`WeightingResult`][bulanik.mcdm.results.WeightingResult] with
        `fuzzy_weights` and `consistency_ratio`.

    Raises:
        TypeError: If an entry is not a TFN.
        ValueError: If the matrix is not square or is larger than 10, an entry
            has $l \le 0$, the matrix is not reciprocal, the names are wrong or
            `defuzzification` is unknown.

    Examples:
        >>> from bulanik import TFN
        >>> from bulanik.mcdm import fuzzy_ahp
        >>> m = [[TFN(1, 1, 1), TFN(2, 3, 4)], [TFN(1/4, 1/3, 1/2), TFN(1, 1, 1)]]
        >>> r = fuzzy_ahp(m, ["a", "b"])
        >>> {k: round(v, 3) for k, v in r.weights.items()}
        {'a': 0.742, 'b': 0.258}
        >>> sorted(r.fuzzy_weights)
        ['a', 'b']

    References:
        Buckley (1985), Fuzzy Sets and Systems 17(3), 233-247.
    """
    rows = _square(matrix, "fuzzy AHP")
    n = len(rows)
    for r in rows:
        for x in r:
            if not isinstance(x, TFN):
                raise TypeError(f"fuzzy AHP needs TFN entries, got {type(x).__name__}")
            if x.l <= 0:
                raise ValueError(f"fuzzy AHP entries must be positive (l > 0), got {x}")
    method = choose(defuzzification, TFN.SCORES, "defuzzification", "fuzzy_ahp")
    for i in range(n):
        for j in range(i, n):
            r = rows[i][j].reciprocal()
            b = rows[j][i]
            if any(
                abs(p - q) > _RECIPROCAL_TOL * max(abs(p), abs(q))
                for p, q in ((r.l, b.l), (r.m, b.m), (r.u, b.u))
            ):
                raise ValueError(
                    f"fuzzy AHP matrix must be reciprocal: a[{j}][{i}] != 1/a[{i}][{j}]"
                )
    names = _names(criteria, n)
    geo = [
        TFN(*(math.prod(getattr(x, f) for x in r) ** (1.0 / n) for f in ("l", "m", "u")))
        for r in rows
    ]
    inv = TFN(
        1.0 / sum(g.u for g in geo), 1.0 / sum(g.m for g in geo), 1.0 / sum(g.l for g in geo)
    )
    fuzzy = [g.mul(inv) for g in geo]
    crisp = [f.defuzzify(method) for f in fuzzy]
    total = sum(crisp)
    w = [c / total for c in crisp]
    _, cr = _eigen([[x.defuzzify(method) for x in r] for r in rows])
    return _result(
        "fuzzy_ahp", names, w,
        {"weights": "buckley_geometric_mean", "defuzzification": method, "consistency": "saaty_ri"},
        consistency_ratio=cr,
        fuzzy_weights=dict(zip(names, fuzzy)),
    )


def ahp_fuzzy_numbers(
    matrix: Sequence[Sequence[FuzzyNumber]],
    criteria: Sequence[str] | None = None,
    *,
    aggregation: str = "wg",
    ops: str = "algebraic",
    score: str | None = None,
    consistency_matrix: Sequence[Sequence[float]] | None = None,
) -> WeightingResult:
    r"""AHP over a comparison matrix of any one fuzzy number kind.

    Row $i$ is aggregated with equal weights $1/n$ by
    [`wg`][bulanik.aggregate.wg] (the fuzzy geometric mean $\tilde r_i =
    (\bigotimes_j \tilde a_{ij})^{1/n}$, default) or
    [`wa`][bulanik.aggregate.wa], using the operator set `ops` of the number
    kind. The row number is then scored, $s_i = \mathrm{score}(\tilde r_i)$,
    and the weights are $w_i = s_i / \sum_k s_k$.

    This differs from [`fuzzy_ahp`][bulanik.mcdm.weighting.fuzzy_ahp] (Buckley,
    triangular numbers only): Buckley first normalises *fuzzily*,
    $\tilde w_i = \tilde r_i \otimes (\sum_k \tilde r_k)^{-1}$, and defuzzifies
    afterwards, while here the row number is defuzzified first and the crisp
    scores are normalised. The generic form needs no fuzzy division, so it works
    for interval-valued, orthopair, neutrosophic and interval type-2 numbers.

    The consistency ratio is Saaty's. With `consistency_matrix` (typically the
    crisp Saaty values of the linguistic terms) it is that matrix's CR
    (`"saaty_ri_on_given_matrix"`); otherwise, if every entry's score is
    positive, it is computed on the matrix of entry scores
    (`"saaty_ri_on_scores"`); otherwise it is `None` (`"none"`).

    Args:
        matrix: Square matrix of fuzzy numbers of one kind (and rung), of order
            at most 10.
        criteria: Criterion names; `C1..Cn` by default.
        aggregation: `"wg"` (default) or `"wa"`.
        ops: Operator set of the number kind.
        score: Score name of the kind; `None` is its default. Orthopair and
            interval-valued kinds default to the signed `"difference"` score,
            which is not a ratio scale: whenever every row score happens to be
            positive it is used as one, with no error, and the weights can then
            differ a lot from those of `score="normalized"`. Use `"normalized"`
            for these kinds.
        consistency_matrix: Optional crisp matrix valid for
            [`ahp`][bulanik.mcdm.weighting.ahp] whose CR is reported.

    Returns:
        A [`WeightingResult`][bulanik.mcdm.results.WeightingResult] whose
        `fuzzy_weights` hold the aggregated row numbers. `operators["consistency"]`
        names how the consistency ratio was obtained: `"saaty_ri_on_scores"`,
        `"saaty_ri_on_given_matrix"` (with `consistency_matrix`) or `"none"`.

    Raises:
        TypeError: If an entry is not a fuzzy number or the kinds are mixed.
        ValueError: If the matrix is not square or larger than 10, `aggregation`,
            `ops` or `score` is unknown, a row score is not positive, or
            `consistency_matrix` is invalid.

    Examples:
        >>> from bulanik import TFN
        >>> from bulanik.mcdm import ahp_fuzzy_numbers
        >>> m = [[TFN(1, 1, 1), TFN(2, 3, 4)], [TFN(1/4, 1/3, 1/2), TFN(1, 1, 1)]]
        >>> r = ahp_fuzzy_numbers(m, ["a", "b"])
        >>> {k: round(v, 3) for k, v in r.weights.items()}
        {'a': 0.743, 'b': 0.257}

    References:
        Saaty (1980), *The Analytic Hierarchy Process*, McGraw-Hill. Kahraman,
        Oztaysi, Ucal Sari and Turanoglu (2014), Knowledge-Based Systems 59,
        48-57, doi:10.1016/j.knosys.2014.02.001 (interval type-2 AHP with
        geometric-mean rows and the DTraT score).
    """
    rows = _square(matrix, "ahp_fuzzy_numbers")
    n = len(rows)
    first = rows[0][0]
    if not isinstance(first, FuzzyNumber):
        raise TypeError(f"ahp_fuzzy_numbers needs fuzzy numbers, got {type(first).__name__}")
    for r in rows:
        for x in r:
            first.same_kind(x)
    agg = {"wa": wa, "wg": wg}[choose(aggregation, ("wa", "wg"), "aggregation", "ahp_fuzzy_numbers")]
    first._check_ops(ops)
    resolved = choose(score, type(first).SCORES, "score", "ahp_fuzzy_numbers", type(first).DEFAULT_SCORE)
    names = _names(criteria, n)
    given = None
    if consistency_matrix is not None:
        given = ahp(consistency_matrix)
        if len(given.criteria) != n:
            raise ValueError("consistency_matrix must have the order of the matrix")
    fuzzy = [agg(r, ops=ops) for r in rows]
    s = [f.score(resolved) for f in fuzzy]
    if any(x <= 0 for x in s):
        hint = (
            'try score="normalized"'
            if "normalized" in type(first).SCORES
            else f"the score {resolved!r} is not positive for these rows; use a positive-valued score"
        )
        raise ValueError(f"ahp_fuzzy_numbers needs positive row scores, got {s}; {hint}")
    total = sum(s)
    w = [x / total for x in s]
    if given is not None:
        cr, cons = given.consistency_ratio, "saaty_ri_on_given_matrix"
    else:
        entry = [[x.score(resolved) for x in r] for r in rows]
        if all(x > 0 for r in entry for x in r):
            cr, cons = _eigen(entry)[1], "saaty_ri_on_scores"
        else:
            cr, cons = None, "none"
    return _result(
        "ahp_fuzzy_numbers", names, w,
        {"weights": "row_" + aggregation, "ops": ops, "score": resolved, "consistency": cons},
        consistency_ratio=cr,
        fuzzy_weights=dict(zip(names, fuzzy)),
    )


def ahp_interval_valued(
    matrix: Sequence[Sequence[IVQROFN]],
    criteria: Sequence[str] | None = None,
    *,
    method: str | None = None,
    consistency_matrix: Sequence[Sequence[float]] | None = None,
) -> WeightingResult:
    r"""Interval-valued AHP over a comparison matrix of one `IVQROFN` kind.

    The matrix holds linguistic judgments written as interval-valued orthopair
    numbers $([\mu^L, \mu^U], [\nu^L, \nu^U])$ of one rung $q$ (IVIFN,
    IVPFN, IVFFN or a general `IVQROFN`). Both triangles are filled by the
    decision maker; reciprocity is not required and the diagonal is taken as
    given, so the scale's "equal" term may differ between scales.

    **`method="multiplicative"`** (any $q$; default for $q \ge 2$). With
    $d^L_{ij} = (\mu^L_{ij})^q - (\nu^U_{ij})^q$ and
    $d^U_{ij} = (\mu^U_{ij})^q - (\nu^L_{ij})^q$ the interval multiplicative
    entries are $s = 1000^{d/q}$ (so $s^L$, $s^U$ from $d^L$, $d^U$), the
    hesitancy factor is
    $h_{ij} = 1 - ((\mu^U_{ij})^q - (\mu^L_{ij})^q) - ((\nu^U_{ij})^q - (\nu^L_{ij})^q)$,
    and $t_{ij} = \tfrac{s^L_{ij} + s^U_{ij}}{2}\,h_{ij}$. The weights are
    $w_i = \sum_j t_{ij} / \sum_{i,j} t_{ij}$.

    **`method="possibility"`** (IVIFN, $q = 1$ only; default for $q = 1$). The
    interval exponential matrix is $a^-_{ij} = e^{\mu^L_{ij} - \nu^U_{ij}}$,
    $a^+_{ij} = e^{\mu^U_{ij} - \nu^L_{ij}}$. The interval weight of criterion
    $i$ is built from the row sums,
    $\tilde w_i = [w^-_i, w^+_i] = [\sum_j a^-_{ij} / \sum_{i,j} a^+_{ij},\
    \sum_j a^+_{ij} / \sum_{i,j} a^-_{ij}]$. The possibility degree is
    $p_{ij} = \dfrac{\max(0, w^+_j - w^-_i) - \max(0, w^-_j - w^+_i)}
    {(w^+_i - w^-_i) + (w^+_j - w^-_j)}$. Here $p_{ij}$ is the degree to which
    $\tilde w_j \ge \tilde w_i$, which is why the weight of $j$ sums over $i$:
    $w_j = \sum_i p_{ij} + n/2 - 1$, normalised to sum 1. When both intervals
    are single points, $p_{ij}$ is 0.5 if they are equal and otherwise 1 or 0
    by their order.

    The consistency ratio is Saaty's. With `consistency_matrix` it is that
    matrix's CR; otherwise, if every entry's default score is positive, it is
    computed on the matrix of entry scores; otherwise it is `None`. The value is
    also in `extras["consistency_ratio"]`. Saaty's CR assumes a reciprocal
    matrix, so when the judgments are not reciprocal (both triangles filled
    independently) the ratio on the scored matrix is indicative only.

    Args:
        matrix: Square matrix of `IVQROFN` numbers of one kind and rung, of
            order 2 to 10.
        criteria: Criterion names; `C1..Cn` by default.
        method: `"multiplicative"` or `"possibility"`; `None` picks
            `"possibility"` for $q = 1$ and `"multiplicative"` otherwise.
        consistency_matrix: Optional crisp matrix valid for
            [`ahp`][bulanik.mcdm.weighting.ahp] whose CR is reported.

    Returns:
        A [`WeightingResult`][bulanik.mcdm.results.WeightingResult]. `extras`
        holds `scored_matrix`, `consistency_ratio` and `consistency` (how the
        ratio was obtained: `"saaty_ri_on_given_matrix"` with `consistency_matrix`,
        `"saaty_ri_on_scores"` when the scored matrix is positive, else `"none"`
        and a `consistency_ratio` of `None`), and for the
        multiplicative method the $n \times n$ lists `d_lower`, `d_upper`,
        `s_lower`, `s_upper`, `h`, `t`; for the possibility method
        `interval_weights` (a list of `(lower, upper)`) and `possibility`
        ($p_{ij}$).

    Raises:
        TypeError: If an entry is not an `IVQROFN` or the kinds or rungs are
            mixed.
        ValueError: If the matrix is not square, has fewer than 2 or more than
            10 rows, `method` is unknown, `"possibility"` is asked for $q \ne
            1$, or `consistency_matrix` is invalid.

    Examples:
        >>> from bulanik import IVIFN
        >>> from bulanik.mcdm import ahp_interval_valued
        >>> e, h, l = IVIFN(.5, .5, .5, .5), IVIFN(.5, .6, .25, .4), IVIFN(.25, .4, .5, .6)
        >>> r = ahp_interval_valued([[e, h], [l, e]], ["a", "b"])
        >>> {k: round(v, 3) for k, v in r.weights.items()}
        {'a': 0.724, 'b': 0.276}

    References:
        Karasan, Ilbahar and Kahraman (2019), Soft Computing 23(21)
        (Pythagorean fuzzy AHP with the multiplicative interval matrix),
        doi:10.1007/s00500-018-3649-0. Alkan and Kahraman (2023), Informatica
        34(1), 1-33 (Fermatean fuzzy AHP, $q = 3$), doi:10.15388/22-INFOR493.
        The form for a general $q$ is this library's generalisation.
        Cebi and Ilbahar (2018), International Journal of the Analytic Hierarchy
        Process 10(2), doi:10.13033/ijahp.v10i2.549 (interval-valued intuitionistic
        fuzzy AHP) and Bolturk and Kahraman (2018), same journal, 10(3),
        doi:10.13033/ijahp.v10i3.545 (the neutrosophic counterpart) give the
        interval weights, the possibility degree and $w = \sum p + n/2 - 1$. Both
        use powers of 10 where this function uses $e$, which changes the weights.
    """
    rows = _square(matrix, "ahp_interval_valued")
    n = len(rows)
    if n < 2:
        raise ValueError("ahp_interval_valued needs a matrix of order at least 2")
    first = rows[0][0]
    if not isinstance(first, IVQROFN):
        raise TypeError(f"ahp_interval_valued needs IVQROFN numbers, got {type(first).__name__}")
    for r in rows:
        for x in r:
            if not isinstance(x, IVQROFN):
                raise TypeError(
                    f"ahp_interval_valued needs IVQROFN numbers, got {type(x).__name__} "
                    f"in a matrix of {type(first).__name__}"
                )
            first.same_kind(x)
    q = first.q
    resolved = choose(
        method, ("multiplicative", "possibility"), "method", "ahp_interval_valued",
        "possibility" if q == 1 else "multiplicative",
    )
    if resolved == "possibility" and q != 1:
        raise ValueError(
            f"method 'possibility' is defined for IVIFN only (q = 1), got q = {q}; "
            "use method='multiplicative'"
        )
    names = _names(criteria, n)
    given = None
    if consistency_matrix is not None:
        given = ahp(consistency_matrix)
        if len(given.criteria) != n:
            raise ValueError("consistency_matrix must have the order of the matrix")

    extras: dict = {}
    if resolved == "multiplicative":
        dl = [[x.mu_l**q - x.nu_u**q for x in r] for r in rows]
        du = [[x.mu_u**q - x.nu_l**q for x in r] for r in rows]
        sl = [[1000.0 ** (d / q) for d in r] for r in dl]
        su = [[1000.0 ** (d / q) for d in r] for r in du]
        h = [
            [1 - (x.mu_u**q - x.mu_l**q) - (x.nu_u**q - x.nu_l**q) for x in r]
            for r in rows
        ]
        t = [[(sl[i][j] + su[i][j]) / 2 * h[i][j] for j in range(n)] for i in range(n)]
        total = sum(map(sum, t))
        if total <= 0:
            raise ValueError("ahp_interval_valued: the weight matrix t sums to a non-positive value")
        w = [sum(r) / total for r in t]
        extras.update(d_lower=dl, d_upper=du, s_lower=sl, s_upper=su, h=h, t=t)
    else:
        a_lo = [[math.exp(x.mu_l - x.nu_u) for x in r] for r in rows]
        a_hi = [[math.exp(x.mu_u - x.nu_l) for x in r] for r in rows]
        sum_lo, sum_hi = sum(map(sum, a_lo)), sum(map(sum, a_hi))
        iw = [(sum(a_lo[i]) / sum_hi, sum(a_hi[i]) / sum_lo) for i in range(n)]
        p = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                (wil, wiu), (wjl, wju) = iw[i], iw[j]
                den = (wiu - wil) + (wju - wjl)
                if den > 0:
                    p[i][j] = (max(0.0, wju - wil) - max(0.0, wjl - wiu)) / den
                else:  # both are points
                    p[i][j] = 0.5 if wil == wjl else (1.0 if wjl > wil else 0.0)
        raw = [sum(p[i][j] for i in range(n)) + n / 2 - 1 for j in range(n)]
        total = sum(raw)
        w = [x / total for x in raw]
        extras.update(interval_weights=iw, possibility=p)

    scored = [[x.score() for x in r] for r in rows]
    if given is not None:
        cr, cons = given.consistency_ratio, "saaty_ri_on_given_matrix"
    elif all(x > 0 for r in scored for x in r):
        cr, cons = _eigen(scored)[1], "saaty_ri_on_scores"
    else:
        cr, cons = None, "none"
    extras.update(scored_matrix=scored, consistency_ratio=cr, consistency=cons)
    return _result(
        "ahp_interval_valued", names, w,
        {"method": resolved, "q": str(q)},
        consistency_ratio=cr,
        extras=extras,
    )


def swara(
    comparative_importance: Sequence[float | FuzzyNumber],
    criteria: Sequence[str],
    *,
    score: str | None = None,
) -> WeightingResult:
    r"""SWARA (Kersuliene et al. 2010); criteria are listed most important first.

    With the comparative importance $s_j$ of criterion $j$ relative to the
    one before it, $q_1 = 1$ and $q_j = q_{j-1} / (s_j + 1)$; the weights are
    $w_j = q_j / \sum_k q_k$.

    Args:
        comparative_importance: $n - 1$ values $s_2, \dots, s_n$, all crisp
            (non-negative) or all fuzzy.
        criteria: The $n$ criterion names, most important first.
        score: How to score fuzzy importances: a score name of their kind, or
            `None` for the kind's default. A given name must be a known score
            name even when every input is crisp (it is then unused).

    Returns:
        A [`WeightingResult`][bulanik.mcdm.results.WeightingResult];
        `operators["score"]` is the resolved score, or `"crisp"`.

    Raises:
        TypeError: If a comparative importance is a `bool` or `str` instead of a number.
        ValueError: If there are no criteria or duplicate names, the number of
            importances is not $n - 1$, crisp and fuzzy are mixed, a score is
            negative or `score` is unknown (for fuzzy numbers a negative
            score can be avoided with `score="normalized"`).

    Examples:
        >>> from bulanik.mcdm import swara
        >>> r = swara([0.3, 0.2], ["a", "b", "c"])
        >>> {k: round(v, 3) for k, v in r.weights.items()}
        {'a': 0.415, 'b': 0.319, 'c': 0.266}

    References:
        Kersuliene, Zavadskas & Turskis (2010), Journal of Business Economics
        and Management 11(2), 243-258.
    """
    crit = list(criteria)
    if not crit:
        raise ValueError("SWARA needs at least one criterion")
    names = _names(crit, len(crit))
    n = len(names)
    s_in = list(comparative_importance)
    if len(s_in) != n - 1:
        raise ValueError(
            f"expected {n - 1} comparative importances for {n} criteria, got {len(s_in)}"
        )
    fuzzy = [isinstance(x, FuzzyNumber) for x in s_in]
    if any(fuzzy) and not all(fuzzy):
        raise ValueError(
            "SWARA comparative importances must be all crisp or all fuzzy, not a mix"
        )
    if score is not None and not any(fuzzy):
        # crisp inputs ignore score, but a misspelt name is still an error
        choose(score, _ALL_SCORES, "score", "swara")
    s: list[float] = []
    resolved = "crisp"
    for x in s_in:
        if isinstance(x, FuzzyNumber):
            resolved = choose(score, x.SCORES, "score", type(x).__name__, x.DEFAULT_SCORE)
            v = x.score(resolved)
        else:
            v = finite(x, "comparative importance")
        if v < 0:
            raise ValueError(
                f"comparative importance must be >= 0, got {v}; "
                'for fuzzy inputs try score="normalized"'
            )
        s.append(v)
    q = [1.0]
    for v in s:
        q.append(q[-1] / (v + 1.0))
    total = sum(q)
    return _result("swara", names, [x / total for x in q], {"score": resolved})


def entropy(
    matrix: Sequence[Sequence[float]], criteria: Sequence[str] | None = None
) -> WeightingResult:
    r"""Shannon-entropy objective weights from a decision matrix.

    With $p_{ij} = x_{ij} / \sum_i x_{ij}$ and $m$ alternatives, the entropy of
    criterion $j$ is $e_j = -\sum_i p_{ij}\ln p_{ij} / \ln m$ (with $0 \ln 0 =
    0$), the divergence is $d_j = 1 - e_j$ and the weight is $w_j = d_j /
    \sum_k d_k$. A criterion whose values differ more between alternatives gets
    more weight.

    Args:
        matrix: Alternatives by criteria, with non-negative entries.
        criteria: Criterion names; `C1..Cn` by default.

    Returns:
        A [`WeightingResult`][bulanik.mcdm.results.WeightingResult].

    Raises:
        TypeError: If a matrix entry is a `bool` or `str` instead of a number.
        ValueError: If the matrix is invalid, has fewer than 2 alternatives, a
            negative entry, a column that sums to zero, or no criterion
            discriminates between the alternatives.

    Examples:
        >>> from bulanik.mcdm import entropy
        >>> r = entropy([[1, 9], [2, 8], [3, 7]], ["a", "b"])
        >>> {k: round(v, 3) for k, v in r.weights.items()}
        {'a': 0.944, 'b': 0.056}

    References:
        Shannon (1948), Bell System Technical Journal 27, 379-423 and 623-656;
        the entropy procedure for attribute weights is described in Hwang &
        Yoon (1981); see also Zeleny (1982), *Multiple Criteria Decision Making*.
    """
    x = check_matrix(matrix)
    m, n = len(x), len(x[0])
    if m < 2:
        raise ValueError("entropy weighting needs at least 2 alternatives")
    if any(v < 0 for r in x for v in r):
        raise ValueError("entropy weighting needs non-negative entries")
    names = _names(criteria, n)
    d = []
    for j in range(n):
        col = [x[i][j] for i in range(m)]
        s = sum(col)
        if s <= 0:
            raise ValueError(f"column {names[j]} sums to zero")
        e = -sum((v / s) * math.log(v / s) for v in col if v > 0) / math.log(m)
        d.append(max(1.0 - e, 0.0))
    total = sum(d)
    # entropy deficits below the rounding noise of the m-term sums are not information
    if total <= _DISCRIMINATION_EPS * m * n:
        raise ValueError("no criterion discriminates between the alternatives")
    return _result("entropy", names, [v / total for v in d], {"weights": "shannon_entropy"})


def _pip_kind(values: list, what: str) -> bool:
    """True for all-TFN, False for all-crisp; reject other fuzzy kinds and mixes."""
    for x in values:
        if isinstance(x, FuzzyNumber) and not isinstance(x, TFN):
            raise TypeError(f"PIPRECIA fuzzy {what} must be TFN, got {type(x).__name__}")
    tfn = [isinstance(x, TFN) for x in values]
    if any(tfn) and not all(tfn):
        raise ValueError(f"PIPRECIA {what} must be all crisp or all TFN, not a mix")
    return any(tfn)


def _pip_forward(s: list, fuzzy: bool, prime: bool = False) -> list:
    """q_1..q_n from judgements s_2..s_n (forward) or q'_1..q'_n from s'_1..s'_{n-1}."""
    one = TFN(1.0, 1.0, 1.0) if fuzzy else 1.0
    two = TFN(2.0, 2.0, 2.0) if fuzzy else 2.0
    n = len(s) + 1
    tag = "k'" if prime else "k"
    order = range(n - 2, -1, -1) if prime else range(n - 1)  # index into s
    q = [one]
    for idx in order:
        v = s[idx]
        k = two - v
        j = idx + 1 if prime else idx + 2  # criterion the k belongs to, 1-based
        sname = f"s{chr(39) if prime else ''}"
        if (v.l if fuzzy else v) <= 0:
            raise ValueError(
                f"PIPRECIA needs every {sname}_j > 0 (a judgement is a ratio of "
                f"importances, 0 < s_j < 2), got {sname}_{j} = {v}"
            )
        if (k.l if fuzzy else k) <= 0:
            raise ValueError(
                f"PIPRECIA needs every {tag}_j = 2 - s{'_j' if not prime else chr(39) + '_j'} "
                f"to be positive, got s{chr(39) if prime else ''}_{j} = {v} ({tag}_{j} = {k})"
            )
        q.append(q[-1] / k)
    return q[::-1] if prime else q


def piprecia(
    relative_importance: Sequence[float | TFN],
    criteria: Sequence[str],
    *,
    score: str | None = None,
    s_inverse: Sequence[float | TFN] | None = None,
) -> WeightingResult:
    r"""PIPRECIA (Stanujkic et al. 2017) and its fully fuzzy form (Stevic et al. 2018).

    The criteria are listed in any order (unlike SWARA they need not be sorted).
    The judgement $s_j$ for $j \ge 2$ compares criterion $j$ with the one before
    it: $s_j > 1$ if $C_j$ is more important, $s_j = 1$ if equal, $s_j < 1$ if
    less. The valid domain is $0 < s_j < 2$: the sources' scales
    (Stanujkic et al. 2017; Dalic et al. 2020) give $s_j$ as a positive ratio of
    importances below 2 (a value near 2 makes $k_j$ near 0), and $k_j = 2 - s_j$
    must stay positive. Zero and negative $s_j$ are meaningless as a ratio and
    are rejected, although the formula would still produce numbers. Then $k_1 = 1$, $k_j = 2 - s_j$, $q_1 = 1$, $q_j = q_{j-1} / k_j$ and
    $w_j = q_j / \sum_k q_k$.

    With [`TFN`][bulanik.numbers.TFN] judgements the method is Stevic et al.'s
    fully fuzzy PIPRECIA with the usual triangular arithmetic: $\tilde k_j =
    (2, 2, 2) - \tilde s_j = (2 - s^u, 2 - s^m, 2 - s^l)$, $\tilde q_j =
    \tilde q_{j-1} \oslash \tilde k_j$ (so $q^l_j = q^l_{j-1} / k^u_j$ and $q^u_j
    = q^u_{j-1} / k^l_j$) and $\tilde w_j = \tilde q_j \oslash \sum_k \tilde q_k$.
    The crisp `weights` are the fuzzy weights scored with `score` and normalised to
    sum 1. The source defuzzifies with the graded mean $(l + 4m + u) / 6$, which is
    the default; its fuzzy scales are designed so that no $\tilde k_j$ contains 0.

    **Inverse pass.** With `s_inverse` the method follows Stevic et al. (2018) and
    Dalic et al. (2020): the same criteria are judged again from the penultimate one
    backwards. $s'_j$ compares criterion $j$ with criterion $j + 1$ ($j = 1, \dots,
    n - 1$; so `s_inverse` has $n - 1$ entries in the order of `criteria`, not
    reversed). Then $k'_n = 1$, $k'_j = 2 - s'_j$, $q'_n = 1$, $q'_j = q'_{j+1} /
    k'_j$ and $w'_j = q'_j / \sum_k q'_k$ (fuzzy for TFN as above). Each of $w_j$ and
    $w'_j$ is scored with `score` (columns I and II of Table 6 in Dalic et al.) and
    the final weight is their mean, $(w_j + w'_j) / 2$. The source does not
    renormalise, and for fuzzy input that mean need not sum to 1 (the wide inverse
    weights score high); `weights` are renormalised to sum 1, which leaves their
    ranking unchanged. `extras` then holds `forward` and `inverse` (the scored
    vectors before averaging, as printed in the source, not renormalised for TFN
    input) and, for TFN input, `fuzzy_inverse`; `fuzzy_weights` stays the forward
    pass.

    Args:
        relative_importance: $n - 1$ values $s_2, \dots, s_n$, all crisp numbers
            or all [`TFN`][bulanik.numbers.TFN].
        criteria: The $n$ criterion names, in the order that was judged.
        score: How to score fuzzy weights: a [`TFN`][bulanik.numbers.TFN] score
            name (`"graded_mean"` by default). A given name must be a known score
            name even when every input is crisp (it is then unused).
        s_inverse: Optional inverse judgements $s'_1, \dots, s'_{n-1}$ (see
            above), of the same kind (crisp or TFN) as `relative_importance`.

    Returns:
        A [`WeightingResult`][bulanik.mcdm.results.WeightingResult];
        `fuzzy_weights` is set for TFN input. `operators["score"]` is the
        resolved score name, or `"crisp"`; `operators["inverse"]` is `"none"` or
        `"mean_of_forward_and_inverse"`.

    Raises:
        TypeError: If a judgement is a `bool`, a `str` or a fuzzy number that is
            not a TFN.
        ValueError: If there are no criteria or duplicate names, the number of
            judgements (or inverse judgements) is not $n - 1$, crisp numbers and
            TFN are mixed (also between `relative_importance` and `s_inverse`), a
            judgement is not finite, some $s_j$ or $s'_j$ is not positive (a crisp
            $s_j \le 0$, or a TFN with $s^l \le 0$), some $k_j$ or $k'_j$ is not
            positive (a crisp $s_j \ge 2$, or a TFN with $s^u \ge 2$), or `score` is unknown.

    Examples:
        >>> from bulanik.mcdm import piprecia
        >>> r = piprecia([1.2, 0.9], ["a", "b", "c"])
        >>> {k: round(v, 3) for k, v in r.weights.items()}
        {'a': 0.295, 'b': 0.369, 'c': 0.336}
        >>> from bulanik import TFN
        >>> f = piprecia([TFN(1.1, 1.15, 1.2)], ["a", "b"])
        >>> w = f.fuzzy_weights["b"]
        >>> round(w.l, 3), round(w.m, 3), round(w.u, 3)
        (0.494, 0.541, 0.592)
        >>> r = piprecia([1.2, 0.9], ["a", "b", "c"], s_inverse=[1.1, 0.7])
        >>> {k: round(v, 3) for k, v in r.weights.items()}
        {'a': 0.31, 'b': 0.33, 'c': 0.36}

    References:
        Stanujkic, Zavadskas, Karabasevic, Smarandache & Turskis (2017), Romanian
        Journal of Economic Forecasting 20(4), 116-133 (no DOI). Stevic,
        Stjepanovic, Bozickovic, Das & Stanujkic (2018), Symmetry 10(11), 586,
        doi:10.3390/sym10110586; the inverse pass as restated with a worked
        example in Dalic, Stevic, Karamasa & Puska (2020), Decision Making:
        Applications in Management and Engineering 3(1), 80-95,
        doi:10.31181/dmame2003114d.
    """
    crit = list(criteria)
    if not crit:
        raise ValueError("PIPRECIA needs at least one criterion")
    names = _names(crit, len(crit))
    n = len(names)
    s_in = list(relative_importance)
    if len(s_in) != n - 1:
        raise ValueError(
            f"expected {n - 1} relative importances for {n} criteria, got {len(s_in)}"
        )
    fuzzy = _pip_kind(s_in, "relative importances")
    inv_in = None
    if s_inverse is not None:
        inv_in = list(s_inverse)
        if len(inv_in) != n - 1:
            raise ValueError(
                f"expected {n - 1} inverse relative importances for {n} criteria, "
                f"got {len(inv_in)}"
            )
        if inv_in and s_in and _pip_kind(inv_in, "inverse relative importances") != fuzzy:
            raise ValueError(
                "PIPRECIA relative importances and inverse relative importances must "
                "be both crisp or both TFN, not a mix"
            )
    if fuzzy:
        resolved = choose(score, TFN.SCORES, "score", "TFN", "graded_mean")
    else:
        if score is not None:
            choose(score, _ALL_SCORES, "score", "piprecia")
        resolved = "crisp"
        s_in = [finite(x, "relative importance") for x in s_in]
        if inv_in is not None:
            inv_in = [finite(x, "inverse relative importance") for x in inv_in]

    def weigh(q: list) -> tuple[list, list[float]]:
        if not fuzzy:
            total = sum(q)
            w = [x / total for x in q]
            return w, w
        total = q[0]
        for x in q[1:]:
            total = total + x
        fw = [x / total for x in q]
        return fw, [f.score(resolved) for f in fw]

    fw, forward = weigh(_pip_forward(s_in, fuzzy))
    operators = {"score": resolved, "inverse": "none"}
    extras: dict = {}
    kw: dict = {"fuzzy_weights": dict(zip(names, fw))} if fuzzy else {}
    if inv_in is None:
        final = forward
    else:
        operators["inverse"] = "mean_of_forward_and_inverse"
        fi, inverse = weigh(_pip_forward(inv_in, fuzzy, prime=True))
        final = [(a + b) / 2 for a, b in zip(forward, inverse)]
        extras = {"forward": dict(zip(names, forward)), "inverse": dict(zip(names, inverse))}
        if fuzzy:
            extras["fuzzy_inverse"] = dict(zip(names, fi))
    total = sum(final)
    return _result("piprecia", names, [c / total for c in final], operators, extras=extras, **kw)


def critic(
    matrix: Sequence[Sequence[float]],
    criteria: Sequence[str] | None = None,
    *,
    criteria_types: Sequence[str] | None = None,
) -> WeightingResult:
    r"""CRITIC objective weights (Diakoulaki, Mavrotas & Papayannakis 1995).

    Each column is scaled to $[0, 1]$ with the best value at 1,
    $r_{ij} = (x_{ij} - x^-_j) / (x^+_j - x^-_j)$, where $x^+_j$ is the largest
    value of a benefit criterion and the smallest of a cost criterion. With the
    population standard deviation $\sigma_j$ of $r_{\cdot j}$ and the Pearson
    correlation $\rho_{jk}$ of two scaled columns, the information carried by
    criterion $j$ is $C_j = \sigma_j \sum_k (1 - \rho_{jk})$ and
    $w_j = C_j / \sum_k C_k$. A criterion gets weight for varying a lot
    (large $\sigma_j$) and for conflicting with the others (low $\rho_{jk}$).
    The weights do not depend on whether $\sigma_j$ uses $m$ or $m - 1$ in the
    denominator, because the factor is the same for every column and cancels in
    $w_j$; population is used. They are also unchanged by a positive scaling
    and a shift of any column (for a cost column the same holds).

    A constant column has no scale and no correlation: it gets weight 0 and is
    left out of the other columns' correlation sums, so adding one changes no
    other weight.

    Args:
        matrix: Alternatives by criteria, finite numbers.
        criteria: Criterion names; `C1..Cn` by default.
        criteria_types: `"benefit"` or `"cost"` per criterion; all benefit by
            default.

    Returns:
        A [`WeightingResult`][bulanik.mcdm.results.WeightingResult] with
        `extras` `{"std": {c: sigma}, "correlation": {c: {c2: rho}},
        "information": {c: C}}`. `std` and `information` are 0 for a constant
        column, which is absent from `correlation`.

    Raises:
        TypeError: If a matrix entry is a `bool` or `str`.
        ValueError: If the matrix is invalid, has fewer than 2 alternatives, the
            names or types are wrong, no criterion discriminates between the
            alternatives (all columns constant), or two or more informative
            columns are all perfectly positively correlated, so every $C_j$ is 0
            (e.g. 2 alternatives and no conflict).

    Examples:
        >>> from bulanik.mcdm import critic
        >>> r = critic([[7, 9], [8, 7], [9, 8]], ["a", "b"], criteria_types=["benefit", "cost"])
        >>> {k: round(v, 3) for k, v in r.weights.items()}
        {'a': 0.5, 'b': 0.5}
        >>> round(r.extras["correlation"]["a"]["b"], 3)
        0.5

    References:
        Diakoulaki, Mavrotas & Papayannakis (1995), Computers & Operations
        Research 22(7), 763-770, doi:10.1016/0305-0548(94)00059-H.
    """
    x = check_matrix(matrix)
    m, n = len(x), len(x[0])
    if m < 2:
        raise ValueError("CRITIC needs at least 2 alternatives")
    names = _names(criteria, n)
    types = check_types(["benefit"] * n if criteria_types is None else criteria_types, n)
    cols: dict[int, list[float]] = {}
    for j in range(n):
        col = [x[i][j] for i in range(m)]
        lo, hi = min(col), max(col)
        if hi - lo <= _DISCRIMINATION_EPS * max(abs(lo), abs(hi)):
            continue  # constant: no scale, no correlation
        if types[j] == "benefit":
            cols[j] = [(v - lo) / (hi - lo) for v in col]
        else:
            cols[j] = [(hi - v) / (hi - lo) for v in col]
    if not cols:
        raise ValueError("no criterion discriminates between the alternatives (all columns constant)")
    dev = {j: [v - sum(c) / m for v in c] for j, c in cols.items()}
    ss = {j: sum(d * d for d in ds) for j, ds in dev.items()}
    sigma = {j: math.sqrt(ss[j] / m) for j in cols}

    def rho(j: int, k: int) -> float:
        if j == k:
            return 1.0
        r = sum(a * b for a, b in zip(dev[j], dev[k])) / math.sqrt(ss[j] * ss[k])
        return max(-1.0, min(1.0, r))

    corr = {j: {k: rho(j, k) for k in cols} for j in cols}
    info = {j: sigma[j] * sum(1.0 - corr[j][k] for k in cols) for j in cols}
    total = sum(info.values())
    if len(cols) == 1:
        (only,) = cols  # its C_j is 0 (rho = 1 with itself); it takes all the weight
        info[only] = total = sigma[only]
    elif total <= _DISCRIMINATION_EPS * m * n:
        raise ValueError(
            "the informative criteria are perfectly correlated, so CRITIC's conflict term "
            "is zero for every criterion"
        )
    w = [info[j] / total if j in cols else 0.0 for j in range(n)]
    return _result(
        "critic", names, w, {"weights": "critic"},
        extras={
            "std": {names[j]: sigma.get(j, 0.0) for j in range(n)},
            "correlation": {
                names[j]: {names[k]: corr[j][k] for k in cols} for j in cols
            },
            "information": {names[j]: info.get(j, 0.0) for j in range(n)},
        },
    )
