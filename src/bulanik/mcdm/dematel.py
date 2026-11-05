r"""DEMATEL: crisp and fuzzy (CFCS) influence analysis.

[`dematel`][bulanik.mcdm.dematel.dematel] takes a direct-relation matrix $A$
(how strongly factor $i$ influences factor $j$) and returns the total-relation
matrix $T = N(I - N)^{-1}$ with $N = A/s$, the prominence $D + R$ and relation
$D - R$ of each factor (row sums $D$ and column sums $R$ of $T$), the cause and
effect groups and factor weights.

Fuzzy entries are defuzzified first. For triangular numbers the default is the
CFCS (converting fuzzy data into crisp scores) procedure of Opricovic & Tzeng
(2003), see [`cfcs`][bulanik.mcdm.dematel.cfcs]; any other kind uses a score of
the kind.

References:
    Gabus & Fontela (1972), *World Problems, an Invitation to Further Thought
    within the Framework of DEMATEL*, Battelle Geneva Research Centre.
    Opricovic & Tzeng (2003), International Journal of Uncertainty, Fuzziness
    and Knowledge-Based Systems 11(5), 635-652 (CFCS). Wu & Lee (2007), Expert
    Systems with Applications 32(2), 499-507 (fuzzy DEMATEL with CFCS).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .._linalg import inverse
from .._validate import choose, finite
from ..numbers import TFN, FuzzyNumber
from ._common import default_names, rank_order
from .results import DEMATELResult

_ILL_CONDITIONED = 1e8  # max(T) / max(N) above this: I - N is numerically singular

__all__ = ["dematel", "cfcs"]


def cfcs(matrices: Sequence[Sequence[Sequence[TFN]]]) -> list[list[float]]:
    r"""Crisp values from one or more experts' triangular-number matrices (CFCS).

    For cell $(i, j)$ with expert ratings $(l^k, m^k, r^k)$, $k = 1..K$, let
    $\min = \min_k l^k$, $\Delta = \max_k r^k - \min$ and, for each expert,
    $x_l = (l^k - \min)/\Delta$ (likewise $x_m$, $x_r$),
    $x^{ls} = x_m/(1 + x_m - x_l)$, $x^{rs} = x_r/(1 + x_r - x_m)$,
    $x = [x^{ls}(1 - x^{ls}) + x^{rs}x^{rs}]/(1 - x^{ls} + x^{rs})$ and
    $z^k = \min + x\Delta$. The cell value is the mean of $z^k$.

    **Scope.** As in Opricovic & Tzeng (2003), the minimum and maximum are taken
    *per cell across the experts*, not over the whole matrix; each expert's
    crisp value is computed first and the experts are averaged *after* the
    CFCS step. With a single expert the cell's own $l$ and $r$ are used. If
    $\Delta = 0$ then $z = \min$.

    Args:
        matrices: One square matrix of `TFN` per expert, all of the same order.

    Returns:
        The averaged crisp matrix.

    Raises:
        TypeError: If an entry is not a `TFN`.
        ValueError: If there is no expert or the matrices are not square and equal.

    Examples:
        >>> from bulanik import TFN
        >>> z = TFN(0, 0, 0)
        >>> round(cfcs([[[z, TFN(1, 2, 4)], [z, z]]])[0][1], 4)
        2.2167
    """
    mats = [[list(r) for r in m] for m in matrices]
    if not mats:
        raise ValueError("cfcs needs at least one expert matrix")
    n = len(mats[0])
    if n == 0 or any(len(m) != n or any(len(r) != n for r in m) for m in mats):
        raise ValueError("cfcs needs square matrices of one common order")
    for m in mats:
        for r in m:
            for x in r:
                if type(x) is not TFN:
                    raise TypeError(f"cfcs needs TFN entries, got {type(x).__name__}")
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cells = [m[i][j] for m in mats]
            lo = min(c.l for c in cells)
            delta = max(c.u for c in cells) - lo
            if delta == 0:
                out[i][j] = lo
                continue
            total = 0.0
            for c in cells:
                xl, xm, xr = (c.l - lo) / delta, (c.m - lo) / delta, (c.u - lo) / delta
                ls = xm / (1 + xm - xl)
                rs = xr / (1 + xr - xm)
                x = (ls * (1 - ls) + rs * rs) / (1 - ls + rs)
                total += lo + x * delta
            out[i][j] = total / len(cells)
    return out


def _crisp(matrix, defuzzification):
    """Defuzzify to a list of float rows; return it with the operator name."""
    rows = [list(r) for r in matrix]
    n = len(rows)
    if n < 2:
        raise ValueError("dematel needs at least 2 factors")
    if any(len(r) != n for r in rows):
        raise ValueError("dematel needs a square matrix")
    fuzzy = [isinstance(x, FuzzyNumber) for r in rows for x in r]
    if any(fuzzy) and not all(fuzzy):
        raise TypeError("dematel matrix mixes fuzzy numbers and crisp values")
    if not any(fuzzy):
        choose(defuzzification, ("crisp",), "defuzzification", "dematel", "crisp")
        for r in rows:
            for x in r:
                if isinstance(x, bool) or not isinstance(x, (int, float)):
                    raise TypeError(f"dematel entries must be numbers, got {type(x).__name__}")
        return [[float(x) for x in r] for r in rows], "crisp", False
    first = rows[0][0]
    for r in rows:
        for x in r:
            first.same_kind(x)
    kind = type(first)
    valid = tuple(kind.SCORES) + (("cfcs",) if kind is TFN else ())
    default = "cfcs" if kind is TFN else kind.DEFAULT_SCORE
    name = choose(defuzzification, valid, "defuzzification", "dematel", default)
    if name == "cfcs":
        return cfcs([rows]), name, True
    return [[x.score(name) for x in r] for r in rows], name, True


def dematel(
    matrix: Sequence[Sequence[float | FuzzyNumber]],
    factors: Sequence[str] | None = None,
    *,
    defuzzification: str | None = None,
    normalization: str = "max_row_col",
    threshold: float | None = None,
    weighting: str = "sqrt",
) -> DEMATELResult:
    r"""DEMATEL on a crisp or fuzzy direct-relation matrix.

    $N = A/s$ with $s = \max(\max_i\sum_j a_{ij}, \max_j\sum_i a_{ij})$
    (`"max_row_col"`) or $s = \max_i\sum_j a_{ij}$ (`"max_row"`), then
    $T = N(I - N)^{-1}$; $D$ and $R$ are the row and column sums of $T$. The
    diagonal of $A$ is used as given.

    Weights: `"sqrt"` gives $w_i \propto \sqrt{(D_i + R_i)^2 + (D_i - R_i)^2}$,
    the length of the factor's vector in the cause-effect plane (a common
    convention in DEMATEL-based weighting, not a formula of the original
    method); `"prominence"` gives $w_i \propto D_i + R_i$.

    A factor is a **cause** when $D_i - R_i > 10^{-12}\max_k(D_k + R_k)$ and an
    **effect** otherwise, so rounding noise around an exact zero net relation
    counts as an effect (as an exact zero does).

    Args:
        matrix: Square matrix, order at least 2, of finite non-negative numbers
            or of fuzzy numbers of one kind.
        factors: Factor names; `F1..Fn` by default.
        defuzzification: For crisp input only `None` or `"crisp"` are valid
            (anything else raises). For fuzzy input, `"cfcs"` (triangular numbers only;
            the default for them) or a score name of the kind (the kind's
            default score otherwise).
        normalization: `"max_row_col"` or `"max_row"`.
        threshold: Cut-off for `relations`; the mean of all $n^2$ entries of
            $T$ when `None`.
        weighting: `"sqrt"` or `"prominence"`.

    Returns:
        A [`DEMATELResult`][bulanik.mcdm.results.DEMATELResult]; its
        `relations` are sorted row-major.

    Raises:
        TypeError: If entries are not numbers, or crisp and fuzzy are mixed.
        ValueError: If the matrix is not square or smaller than 2, has a
            negative or non-finite entry (after defuzzification; a signed
            default score suggests a positive-valued one), is all zero, an option
            is unknown, `factors` is wrong, or $I - N$ is singular or
            ill-conditioned (the normalised matrix has spectral radius 1, or
            numerically 1 so that the largest total relation exceeds $10^8$ times
            the largest normalised direct relation; e.g. equal or nearly equal
            row and column sums).

    Examples:
        >>> from bulanik.mcdm import dematel
        >>> r = dematel([[0, 2], [1, 0]])
        >>> r.causes, r.effects, r.relation
        (('F1',), ('F2',), {'F1': 1.0, 'F2': -1.0})
    """
    a, op, is_fuzzy = _crisp(matrix, defuzzification)
    n = len(a)
    norm = choose(normalization, ("max_row_col", "max_row"), "normalization", "dematel")
    wname = choose(weighting, ("sqrt", "prominence"), "weighting", "dematel")
    if threshold is not None:
        threshold = finite(threshold, "threshold")
    names = default_names("F", n) if factors is None else list(factors)
    if len(names) != n:
        raise ValueError(f"expected {n} factor names, got {len(names)}")
    if len(set(names)) != n:
        raise ValueError("factor names must be unique")
    if any(not math.isfinite(x) for r in a for x in r):
        raise ValueError("dematel entries must be finite")
    if any(x < 0 for r in a for x in r):
        hint = ""
        if is_fuzzy:
            kind = type(next(x for r in matrix for x in r))
            hint = (
                '; try defuzzification="normalized"' if "normalized" in kind.SCORES
                else "; use a positive-valued score"
            )
        raise ValueError(f"dematel entries must be >= 0 (negative after defuzzification){hint}")
    rows = [sum(r) for r in a]
    s = max(rows) if norm == "max_row" else max(max(rows), max(sum(a[i][j] for i in range(n)) for j in range(n)))
    if s == 0:
        raise ValueError("dematel matrix is all zero (normalising maximum is zero)")
    nm = [[x / s for x in r] for r in a]
    ima = [[(1.0 if i == j else 0.0) - nm[i][j] for j in range(n)] for i in range(n)]
    try:
        inv = inverse(ima)
    except ValueError as e:
        raise ValueError(
            "I - N is singular: the normalised matrix has spectral radius 1 "
            "(e.g. equal row and column sums), so the total relation is undefined"
        ) from e
    t = [[sum(nm[i][k] * inv[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    if any(not math.isfinite(x) or x < -1e-9 for r in t for x in r):
        raise ValueError("I - N is singular or ill-conditioned: total relation is not finite and non-negative")
    # T = (I - N)^-1 - I blows up as the spectral radius of N approaches 1: a total
    # relation more than 1e8 times the largest direct relation is rounding noise
    if max(x for r in t for x in r) > _ILL_CONDITIONED * max(x for r in nm for x in r):
        raise ValueError(
            "I - N is ill-conditioned: the normalised matrix has a spectral radius "
            "numerically 1 (e.g. nearly equal row and column sums), so the total "
            "relation is huge and meaningless"
        )
    d = [sum(r) for r in t]
    rr = [sum(t[i][j] for i in range(n)) for j in range(n)]
    prom = [x + y for x, y in zip(d, rr)]
    rel = [x - y for x, y in zip(d, rr)]
    tol = 1e-12 * max(prom)
    thr = sum(x for r in t for x in r) / (n * n) if threshold is None else float(threshold)
    if wname == "sqrt":
        raw = [math.hypot(p, q) for p, q in zip(prom, rel)]
    else:
        raw = prom
    total = sum(raw)
    w = [x / total for x in raw]
    weights = dict(zip(names, w))
    return DEMATELResult(
        method="dematel",
        factors=tuple(names),
        direct=tuple(tuple(r) for r in a),
        normalized=tuple(tuple(r) for r in nm),
        total=tuple(tuple(r) for r in t),
        d=dict(zip(names, d)),
        r=dict(zip(names, rr)),
        prominence=dict(zip(names, prom)),
        relation=dict(zip(names, rel)),
        causes=tuple(f for f, x in zip(names, rel) if x > tol),
        effects=tuple(f for f, x in zip(names, rel) if x <= tol),
        threshold=thr,
        relations=tuple(
            (names[i], names[j], t[i][j]) for i in range(n) for j in range(n) if i != j and t[i][j] > thr
        ),
        weights=weights,
        ranking=rank_order(weights, True),
        operators={
            "normalization": norm,
            "defuzzification": "crisp" if op == "crisp" else op,
            "threshold": "mean" if threshold is None else str(threshold),
            "weights": wname,
        },
    )
