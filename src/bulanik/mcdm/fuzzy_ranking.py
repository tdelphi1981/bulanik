r"""Ranking on fuzzy numbers: CODAS, WASPAS and CoCoSo computed on the numbers themselves.

[`rank_fuzzy`][bulanik.mcdm.fuzzy.rank_fuzzy] with a crisp method scores every
fuzzy rating first and then ranks the crisp matrix. The three methods here keep
the ratings fuzzy: the weighted matrix, the negative-ideal solution, the
weighted sums and the weighted products are built with the number kind's own
operators (`scale`, `add`, `mul`, `power`, through
[`wa`][bulanik.aggregate.wa] and [`wg`][bulanik.aggregate.wg] where they
apply), and the numbers are scored only at the end, as the published
intuitionistic, Pythagorean and neutrosophic variants of these methods do.

- [`fuzzy_codas`][bulanik.mcdm.fuzzy_ranking.fuzzy_codas]: distances of the
  weighted numbers from the negative-ideal numbers.
- [`fuzzy_waspas`][bulanik.mcdm.fuzzy_ranking.fuzzy_waspas]: the weighted sum
  and the weighted product combined as numbers, then scored (or scored, then
  combined).
- [`fuzzy_cocoso`][bulanik.mcdm.fuzzy_ranking.fuzzy_cocoso]: the weighted sum
  and the weighted product scored, then CoCoSo's three appraisal scores.

Each takes an already aggregated matrix (alternatives by criteria) of one
number kind, crisp criteria weights in the forms the crisp methods accept,
`criteria_types`, the operator set `ops` and a `score` name, and returns a
[`RankingResult`][bulanik.mcdm.results.RankingResult] whose `ranking` is best
first. A **cost criterion** is turned into a benefit one with the kind's
`complement()` (for an orthopair number $(\mu, \nu) \mapsto (\nu, \mu)$), which
exists for the orthopair, interval-valued orthopair, picture, spherical,
neutrosophic and interval neutrosophic kinds; triangular, trapezoidal and
interval type-2 numbers have none, so they can only be used with benefit
criteria here (or through [`rank_fuzzy`][bulanik.mcdm.fuzzy.rank_fuzzy]).

References:
    Rani, Ali,
    Krishankumar, Mishra, Cavallaro & Ravichandran (2021), Energies 14(15), 4594
    (SVN-CoCoSo).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .._validate import choose
from ..aggregate import wa, wg
from ..numbers import FuzzyNumber
from ._common import rank_order
from .ranking import _finite_all, _prepare_names, _unit_param
from .results import BWMResult, RankingResult, WeightingResult
from .scoring import _cocoso_parts, _codas_assessment

__all__ = ["fuzzy_codas", "fuzzy_waspas", "fuzzy_cocoso"]

_Weights = Sequence[float] | Mapping[str, float] | WeightingResult | BWMResult


def _fuzzy_matrix(matrix: Sequence[Sequence[Any]], owner: str) -> list[list[Any]]:
    """Rows of fuzzy numbers of one kind; rectangular and non-empty."""
    rows = [list(r) for r in matrix]
    if not rows or not rows[0]:
        raise ValueError("matrix must have at least one row and one column")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("matrix rows must all have the same length")
    first = rows[0][0]
    for r in rows:
        for x in r:
            if not isinstance(x, FuzzyNumber):
                raise TypeError(f"{owner} needs fuzzy number entries, got {type(x).__name__}")
            first.same_kind(x)  # TypeError naming both kinds
    return rows


def _setup(
    owner: str,
    matrix: Sequence[Sequence[Any]],
    weights: _Weights,
    criteria_types: Sequence[str],
    alternatives: Sequence[str] | None,
    criteria: Sequence[str] | None,
    normalize_weights: bool,
    ops: str,
    score: str | None,
    prefer_normalized: bool = False,
) -> tuple[list[list[Any]], list[str], list[str], list[float], str, str]:
    """Validate the input; return the benefit-oriented matrix, names, weights, ops and score."""
    x = _fuzzy_matrix(matrix, owner)
    alts, crit, types, w = _prepare_names(
        len(x), len(x[0]), weights, criteria_types, alternatives, criteria, normalize_weights
    )
    kind = type(x[0][0])
    ops = choose(ops, kind.OPS, "ops", kind.__name__)
    default = "normalized" if prefer_normalized and "normalized" in kind.SCORES else kind.DEFAULT_SCORE
    resolved = choose(score, kind.SCORES, "score", kind.__name__, default)
    for j, t in enumerate(types):
        if t != "cost":
            continue
        if getattr(kind, "complement", None) is None:
            raise TypeError(
                f"{owner}: criterion '{crit[j]}' is a cost criterion, but {kind.__name__} has no "
                f"complement() to turn it into a benefit one; use rank_fuzzy (which scores the "
                f"numbers first) or rate the criterion as a benefit"
            )
        for r in x:
            r[j] = r[j].complement()
    return x, alts, crit, w, ops, resolved


def _ranking(method: str, alts: list[str], scores: list[float], operators: dict[str, str],
             **extras: Any) -> RankingResult:
    _finite_all(method, scores, "a score")
    sc = dict(zip(alts, scores))
    return RankingResult(
        method=method,
        alternatives=tuple(alts),
        scores=sc,
        ranking=rank_order(sc, True),
        operators=operators,
        extras=extras,
    )


def fuzzy_codas(
    matrix: Sequence[Sequence[Any]],
    weights: _Weights,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
    ops: str = "algebraic",
    score: str | None = None,
    tau: float = 0.02,
    psi: str = "equation",
) -> RankingResult:
    r"""CODAS on fuzzy numbers: distances of the weighted numbers from the negative-ideal numbers.

    Cost criteria are first replaced by their complements. Each rating is
    weighted with the kind's scalar multiple, $\tilde u_{ij} = w_j \tilde
    x_{ij}$ (`scale`; for an intuitionistic number $(1 - (1 - \mu)^{w_j},
    \nu^{w_j})$). The negative-ideal number of criterion $j$, $\tilde{ns}_j$,
    is the $\tilde u_{ij}$ with the lowest `score` in its column (the first
    one on a tie). With the kind's normalised Euclidean and Hamming distances
    $d_E$ and $d_H$,

    $$E_i = \sqrt{\sum_j d_E(\tilde u_{ij}, \tilde{ns}_j)^2}, \qquad
      T_i = \sum_j d_H(\tilde u_{ij}, \tilde{ns}_j).$$

    For an intuitionistic number the method defines the distances as
    $d_E(a, b)^2 = \tfrac12[(\mu_a - \mu_b)^2 + (\nu_a - \nu_b)^2 +
    (\pi_a - \pi_b)^2]$ and $d_H(a, b) = \tfrac12(|\mu_a - \mu_b| + |\nu_a -
    \nu_b| + |\pi_a - \pi_b|)$ (the kind's `"euclidean"` and `"hamming"`
    distances), and $E_i$ and $T_i$ are the sums of Altın et al. (2026),
    Eqs. (15)-(16), with these distances (see also Szmidt & Kacprzyk 2000 on
    distances between intuitionistic fuzzy sets). The relative assessment and the score are those
    of crisp [`codas`][bulanik.mcdm.scoring.codas] (shared code),

    $$h_{ik} = (E_i - E_k) + \psi(E_i - E_k)\,(T_i - T_k), \qquad H_i = \sum_k h_{ik},$$

    larger is better; `psi` selects the convention of $\psi$ exactly as there
    (`"equation"`, the default, adds the taxicab term when
    $|E_i - E_k| \ge \tau$, as in Altın et al. (2026), Eq. (19)).

    `extras` holds `E` and `T` (by alternative) and `negative_ideal`, the
    numbers $\tilde{ns}_j$ **by criterion**. Picture numbers fail to scale
    by almost any weight below 1 (see [`PictureFN`][bulanik.numbers.PictureFN]),
    so `fuzzy_codas` raises a `ValueError` naming the criterion and the weight for
    them; use [`fuzzy_waspas`][bulanik.mcdm.fuzzy_ranking.fuzzy_waspas],
    [`fuzzy_cocoso`][bulanik.mcdm.fuzzy_ranking.fuzzy_cocoso] or
    `rank_fuzzy(method="codas", score="normalized")` instead.

    Args:
        matrix: Alternatives by criteria of fuzzy numbers of one kind (and
            rung) with `euclidean` and `hamming` distances.
        weights: Crisp criteria weights: a sequence in criteria order, a
            mapping by criterion name, or a
            [`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
            [`BWMResult`][bulanik.mcdm.results.BWMResult]. They must sum to 1
            unless `normalize_weights` is set.
        criteria_types: `"benefit"` or `"cost"` for each criterion.
        alternatives: Alternative names; `A1..Am` by default.
        criteria: Criterion names; if `None` they are the keys of a mapping or
            result `weights` (in order), else `C1..Cn`.
        normalize_weights: Rescale non-negative weights to sum to 1 instead of
            requiring it.
        ops: Operator set of the kind, e.g. `"algebraic"` or `"einstein"`.
        score: Score that picks the negative-ideal numbers; `None` for the
            kind's default.
        tau: Threshold of $\psi$ in [0, 1].
        psi: `"equation"` or `"tables"`, as for crisp CODAS.

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"ops": ..., "score": ..., "tau": "0.02", "psi": "equation"}`.

    Raises:
        TypeError: If an entry is not a fuzzy number, the kinds (or rungs)
            differ, a cost criterion meets a kind without `complement()`, the
            kind lacks a Euclidean or Hamming distance, or a weight or `tau` is
            a `bool` or `str`.
        ValueError: If the matrix is empty or ragged, names, types or weights
            do not match (as for the crisp methods), `ops`, `score` or `psi` is
            unknown, `tau` is outside [0, 1], or the kind cannot scale a rating.

    Examples:
        >>> from bulanik import IFN
        >>> from bulanik.mcdm import fuzzy_codas
        >>> m = [[IFN(0.6, 0.3), IFN(0.4, 0.5)], [IFN(0.5, 0.4), IFN(0.7, 0.2)]]
        >>> r = fuzzy_codas(m, [0.6, 0.4], ["benefit", "benefit"])
        >>> r.ranking
        ('A2', 'A1')
        >>> {k: round(v, 4) for k, v in r.scores.items()}
        {'A1': -0.2708, 'A2': 0.2708}

    References:
        Keshavarz Ghorabaee,
        Zavadskas, Turskis & Antucheviciene (2016), Economic Computation and
        Economic Cybernetics Studies and Research 50(3), 25-44 (crisp CODAS).
        See also: Szmidt & Kacprzyk (2000), Distances between intuitionistic
        fuzzy sets, Fuzzy Sets and Systems 114(3), 505-518,
        doi:10.1016/S0165-0114(98)00244-9; Bolturk (2018), Pythagorean fuzzy
        CODAS and its application to supplier selection in a manufacturing
        firm, Journal of Enterprise Information Management 31(4), 550-564,
        doi:10.1108/JEIM-01-2018-0020.
    """
    tau = _unit_param(tau, "tau")
    psi = choose(psi, ("equation", "tables"), "psi", "fuzzy_codas")
    x, alts, crit, w, ops, resolved = _setup(
        "fuzzy_codas", matrix, weights, criteria_types, alternatives, criteria,
        normalize_weights, ops, score,
    )
    kind = type(x[0][0])
    missing = [d for d in ("euclidean", "hamming") if d not in kind.DISTANCES]
    if missing:
        raise TypeError(
            f"fuzzy_codas needs the 'euclidean' and 'hamming' distances; "
            f"{kind.__name__} has: {', '.join(kind.DISTANCES)}; "
            f'use rank_fuzzy(..., method="codas", score="normalized") instead'
        )
    m, n = len(x), len(x[0])
    u: list[list[Any]] = [[None] * n for _ in range(m)]
    for j in range(n):
        for i in range(m):
            try:
                u[i][j] = x[i][j].scale(w[j], ops)
            except ValueError as exc:
                raise ValueError(
                    f"fuzzy_codas: {kind.__name__} cannot be weighted by {w[j]:g} for "
                    f"criterion '{crit[j]}' ({exc}); use fuzzy_waspas, fuzzy_cocoso or "
                    f'rank_fuzzy(..., method="codas", score="normalized")'
                ) from exc
    ns = []
    for j in range(n):
        col = [u[i][j] for i in range(m)]
        scores = [v.score(resolved) for v in col]
        ns.append(col[scores.index(min(scores))])
    e = [
        math.sqrt(math.fsum(u[i][j].distance(ns[j], "euclidean") ** 2 for j in range(n)))
        for i in range(m)
    ]
    t = [math.fsum(u[i][j].distance(ns[j], "hamming") for j in range(n)) for i in range(m)]
    h = _codas_assessment(e, t, tau, psi)
    return _ranking(
        "fuzzy_codas",
        alts,
        h,
        {"ops": ops, "score": resolved, "tau": str(tau), "psi": psi},
        E=dict(zip(alts, e)),
        T=dict(zip(alts, t)),
        negative_ideal=dict(zip(crit, ns)),
    )


def fuzzy_waspas(
    matrix: Sequence[Sequence[Any]],
    weights: _Weights,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
    ops: str = "algebraic",
    score: str | None = None,
    lam: float = 0.5,
    combine: str = "fuzzy",
) -> RankingResult:
    r"""WASPAS on fuzzy numbers: weighted sum and weighted product combined as numbers, then scored.

    Cost criteria are first replaced by their complements. With the kind's
    operators, the weighted sum (WSM) and the weighted product (WPM) of each
    alternative are the numbers

    $$\tilde Q^{(1)}_i = \bigoplus_j w_j \tilde x_{ij}, \qquad
      \tilde Q^{(2)}_i = \bigotimes_j \tilde x_{ij}^{\,w_j},$$

    i.e. [`wa`][bulanik.aggregate.wa] and [`wg`][bulanik.aggregate.wg] of the
    row (for an intuitionistic number $\tilde Q^{(1)} = (1 - \prod_j (1 -
    \mu_j)^{w_j}, \prod_j \nu_j^{w_j})$). `combine` selects how they meet
    $\lambda$; larger $Q_i$ is better either way.

    - `combine="fuzzy"` (default): they are combined as numbers, with the
      kind's `scale` and `add`, and the combination is scored once,

      $$\tilde Q_i = \lambda \tilde Q^{(1)}_i \oplus (1 - \lambda) \tilde Q^{(2)}_i,
        \qquad Q_i = S(\tilde Q_i).$$

      Interval-valued intuitionistic numbers can use
      `score="defuzzified"`. It is computed as [`wa`][bulanik.aggregate.wa] of
      $(\tilde Q^{(1)}_i, \tilde Q^{(2)}_i)$ with weights
      $(\lambda, 1 - \lambda)$, which is that `scale` and `add` (a weight of
      0 drops its term, so $\lambda = 1$ gives the WSM alone and
      $\lambda = 0$ the WPM alone).
    - `combine="scores"`: the two numbers are scored first and the scores
      combined, $Q_i = \lambda S(\tilde Q^{(1)}_i) + (1 - \lambda)
      S(\tilde Q^{(2)}_i)$, the form of crisp WASPAS.

    `extras` holds the numbers `Q1` and `Q2`, the scores `Q1_score` and
    `Q2_score`, and (for `combine="fuzzy"`) the combined number `Q`.

    Args:
        matrix: Alternatives by criteria of fuzzy numbers of one kind (and rung).
        weights: Crisp criteria weights: a sequence in criteria order, a
            mapping by criterion name, or a
            [`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
            [`BWMResult`][bulanik.mcdm.results.BWMResult]. They must sum to 1
            unless `normalize_weights` is set.
        criteria_types: `"benefit"` or `"cost"` for each criterion.
        alternatives: Alternative names; `A1..Am` by default.
        criteria: Criterion names; if `None` they are the keys of a mapping or
            result `weights` (in order), else `C1..Cn`.
        normalize_weights: Rescale non-negative weights to sum to 1 instead of
            requiring it.
        ops: Operator set of the kind, e.g. `"algebraic"` or `"einstein"`.
        score: Score of the final numbers; `None` for the kind's default.
        lam: Balance $\lambda \in [0, 1]$ between the WSM and the WPM.
        combine: `"fuzzy"` (combine the numbers, then score) or `"scores"`
            (score, then combine); see above.

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"ops": ..., "score": ..., "lambda": "0.5", "combine": "fuzzy"}`.

    Raises:
        TypeError: If an entry is not a fuzzy number, the kinds (or rungs)
            differ, a cost criterion meets a kind without `complement()`, or a
            weight or `lam` is a `bool` or `str`.
        ValueError: If the matrix is empty or ragged, names, types or weights
            do not match (as for the crisp methods), `ops`, `score` or
            `combine` is unknown, `lam` is outside [0, 1], or the kind rejects an operation
            (e.g. [`wg`][bulanik.aggregate.wg] of a triangular number with a
            negative point).

    Examples:
        >>> from bulanik import IFN
        >>> from bulanik.mcdm import fuzzy_waspas
        >>> m = [[IFN(0.6, 0.3), IFN(0.4, 0.5)], [IFN(0.5, 0.4), IFN(0.7, 0.2)]]
        >>> r = fuzzy_waspas(m, [0.6, 0.4], ["benefit", "benefit"])
        >>> r.ranking
        ('A2', 'A1')
        >>> {k: round(v, 4) for k, v in r.scores.items()}
        {'A1': 0.142, 'A2': 0.2676}
        >>> s = fuzzy_waspas(m, [0.6, 0.4], ["benefit", "benefit"], combine="scores")
        >>> {k: round(v, 4) for k, v in s.scores.items()}
        {'A1': 0.1418, 'A2': 0.2672}

    References:
        Zavadskas, Turskis, Antucheviciene &
        Zakarevicius (2012), Elektronika ir Elektrotechnika 122(6), 3-6
        (crisp WASPAS). See also: Ilbahar & Kahraman (2018), Retail store
        performance measurement using a novel interval-valued Pythagorean
        fuzzy WASPAS method, Journal of Intelligent & Fuzzy Systems 35(3),
        3835-3846, doi:10.3233/JIFS-18730.
    """
    lam = _unit_param(lam, "lam")
    combine = choose(combine, ("fuzzy", "scores"), "combine", "fuzzy_waspas")
    x, alts, _, w, ops, resolved = _setup(
        "fuzzy_waspas", matrix, weights, criteria_types, alternatives, criteria,
        normalize_weights, ops, score,
    )
    q1 = [wa(r, w, ops=ops) for r in x]
    q2 = [wg(r, w, ops=ops) for r in x]
    s1 = [v.score(resolved) for v in q1]
    s2 = [v.score(resolved) for v in q2]

    def by_alt(vals: Sequence[Any]) -> dict[str, Any]:
        return dict(zip(alts, vals))

    extras: dict[str, Any] = {"Q1": by_alt(q1), "Q2": by_alt(q2)}
    if combine == "fuzzy":
        q = [wa([a, b], [lam, 1.0 - lam], ops=ops) for a, b in zip(q1, q2)]
        scores = [v.score(resolved) for v in q]
        extras["Q"] = by_alt(q)
    else:
        scores = [lam * a + (1.0 - lam) * b for a, b in zip(s1, s2)]
    return _ranking(
        "fuzzy_waspas",
        alts,
        scores,
        {"ops": ops, "score": resolved, "lambda": str(lam), "combine": combine},
        **extras,
        Q1_score=by_alt(s1),
        Q2_score=by_alt(s2),
    )


def fuzzy_cocoso(
    matrix: Sequence[Sequence[Any]],
    weights: _Weights,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
    normalize_weights: bool = False,
    ops: str = "algebraic",
    score: str | None = None,
    lam: float = 0.5,
) -> RankingResult:
    r"""CoCoSo on fuzzy numbers: weighted sum and weighted product as numbers, scored, then compromised.

    Cost criteria are first replaced by their complements. With the kind's
    operators the weighted-sum and weighted-product sequences are the numbers

    $$\tilde S_i = \bigoplus_j w_j \tilde x_{ij}, \qquad
      \tilde P_i = \bigotimes_j \tilde x_{ij}^{\,w_j}$$

    ([`wa`][bulanik.aggregate.wa] and [`wg`][bulanik.aggregate.wg] of the row),
    with no normalisation step: $\tilde P_i$ is the weighted geometric
    product, not a fuzzy sum of powers. This follows the form of the
    neutrosophic CoCoSo of Rani et al. (2021), Eqs. (11)-(12). Their
    scores $S_i = S(\tilde S_i)$ and $P_i = S(\tilde P_i)$ enter crisp
    CoCoSo's appraisal scores (shared code with
    [`cocoso`][bulanik.mcdm.scoring.cocoso]),

    $$k_{ia} = \frac{P_i + S_i}{\sum_i (P_i + S_i)}, \qquad
      k_{ib} = \frac{S_i}{\min_i S_i} + \frac{P_i}{\min_i P_i}, \qquad
      k_{ic} = \frac{\lambda S_i + (1 - \lambda) P_i}{\lambda \max_i S_i + (1 - \lambda) \max_i P_i},$$

    $$k_i = (k_{ia}\, k_{ib}\, k_{ic})^{1/3} + \tfrac13 (k_{ia} + k_{ib} + k_{ic}),$$

    larger is better. The ratios need **non-negative scores**: the default
    score is the kind's `"normalized"` one where it has it (orthopair,
    interval-valued orthopair, picture, spherical: values in [0, 1]), and the
    kind's default otherwise; a negative score raises `ValueError`. If an
    alternative's $S_i$ or $P_i$ is 0, $k_{ib}$ divides by zero and a
    `ValueError` names the alternative. Because $\tilde P_i$ is a weighted
    product, a single intuitionistic $(0, 1)$ rating is enough to make $P_i$
    zero, even if the alternative is good on the other criteria. `extras` holds the numbers `S` and `P`, their
    scores `S_score` and `P_score`, and `k_a`, `k_b`, `k_c`.

    Args:
        matrix: Alternatives by criteria of fuzzy numbers of one kind (and rung).
        weights: Crisp criteria weights: a sequence in criteria order, a
            mapping by criterion name, or a
            [`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
            [`BWMResult`][bulanik.mcdm.results.BWMResult]. They must sum to 1
            unless `normalize_weights` is set.
        criteria_types: `"benefit"` or `"cost"` for each criterion.
        alternatives: Alternative names; `A1..Am` by default.
        criteria: Criterion names; if `None` they are the keys of a mapping or
            result `weights` (in order), else `C1..Cn`.
        normalize_weights: Rescale non-negative weights to sum to 1 instead of
            requiring it.
        ops: Operator set of the kind, e.g. `"algebraic"` or `"einstein"`.
        score: A non-negative score; `None` for `"normalized"` where the kind
            has it, else the kind's default.
        lam: Balance $\lambda \in [0, 1]$ in $k_{ic}$.

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"ops": ..., "score": ..., "lambda": "0.5"}`.

    Raises:
        TypeError: If an entry is not a fuzzy number, the kinds (or rungs)
            differ, a cost criterion meets a kind without `complement()`, or a
            weight or `lam` is a `bool` or `str`.
        ValueError: If the matrix is empty or ragged, names, types or weights
            do not match (as for the crisp methods), `ops` or `score` is
            unknown, `lam` is outside [0, 1], a score is negative, or an
            alternative's $S_i$ or $P_i$ is 0 (the error names it).

    Examples:
        >>> from bulanik import IFN
        >>> from bulanik.mcdm import fuzzy_cocoso
        >>> m = [[IFN(0.6, 0.3), IFN(0.4, 0.5)], [IFN(0.5, 0.4), IFN(0.7, 0.2)]]
        >>> r = fuzzy_cocoso(m, [0.6, 0.4], ["benefit", "benefit"])
        >>> r.ranking, r.operators["score"]
        (('A2', 'A1'), 'normalized')
        >>> {k: round(v, 4) for k, v in r.scores.items()}
        {'A1': 2.0738, 'A2': 2.3016}

    References:
        Rani, Ali, Krishankumar, Mishra, Cavallaro & Ravichandran (2021), An
        integrated single-valued neutrosophic combined compromise solution
        methodology for renewable energy resource selection problem, Energies
        14(15), 4594, doi:10.3390/en14154594, Eqs. (7) and (11)-(16).
        Yazdani, Zarate, Zavadskas & Turskis (2019), Management Decision
        57(9), 2501-2519, doi:10.1108/MD-05-2017-0458 (crisp CoCoSo).
        See also: Liu, Rani & Mishra (2021), A novel Pythagorean fuzzy
        combined compromise solution framework for the assessment of medical
        waste treatment technology, Journal of Cleaner Production 292,
        126047, doi:10.1016/j.jclepro.2021.126047.
    """
    lam = _unit_param(lam, "lam")
    x, alts, _, w, ops, resolved = _setup(
        "fuzzy_cocoso", matrix, weights, criteria_types, alternatives, criteria,
        normalize_weights, ops, score, prefer_normalized=True,
    )
    s_num = [wa(r, w, ops=ops) for r in x]
    p_num = [wg(r, w, ops=ops) for r in x]
    s = [v.score(resolved) for v in s_num]
    p = [v.score(resolved) for v in p_num]
    for label, vals in (("S", s), ("P", p)):
        for i, v in enumerate(vals):
            if v < 0:
                hint = (
                    'try score="normalized"' if "normalized" in type(x[0][0]).SCORES
                    else "use a non-negative score"
                )
                raise ValueError(
                    f"fuzzy_cocoso needs non-negative scores, but score '{resolved}' of {label} "
                    f"for alternative '{alts[i]}' is {v:.6g}; {hint}"
                )
    k_a, k_b, k_c, k = _cocoso_parts(
        s, p, lam, alts,
        reason="has a zero score: its score of S or P is 0 (one (0, 1) rating is enough for P)",
    )

    def by_alt(vals: Sequence[Any]) -> dict[str, Any]:
        return dict(zip(alts, vals))

    return _ranking(
        "fuzzy_cocoso",
        alts,
        k,
        {"ops": ops, "score": resolved, "lambda": str(lam)},
        S=by_alt(s_num),
        P=by_alt(p_num),
        S_score=by_alt(s),
        P_score=by_alt(p),
        k_a=by_alt(k_a),
        k_b=by_alt(k_b),
        k_c=by_alt(k_c),
    )
