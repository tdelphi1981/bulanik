"""Fuzzy ranking: expert decision matrices, rank_fuzzy and Chen's fuzzy TOPSIS.

Two routes from fuzzy ratings to a ranking.

- [`rank_fuzzy`][bulanik.mcdm.fuzzy.rank_fuzzy] takes a
  [`DecisionMatrix`][bulanik.mcdm.fuzzy.DecisionMatrix] of expert ratings (any
  fuzzy number kind), aggregates the experts, **scores** each fuzzy number to a
  crisp value and hands the crisp matrix to a classical method
  ([`topsis`][bulanik.mcdm.ranking.topsis], [`vikor`][bulanik.mcdm.ranking.vikor],
  [`waspas`][bulanik.mcdm.ranking.waspas], [`edas`][bulanik.mcdm.ranking.edas],
  [`codas`][bulanik.mcdm.scoring.codas], [`copras`][bulanik.mcdm.scoring.copras],
  [`marcos`][bulanik.mcdm.scoring.marcos], [`cocoso`][bulanik.mcdm.scoring.cocoso],
  [`wisp`][bulanik.mcdm.scoring.wisp] or [`electre`][bulanik.mcdm.outranking.electre]).
  With `method="fuzzy_codas"`, `"fuzzy_waspas"` or `"fuzzy_cocoso"` it skips
  the scoring and hands the aggregated fuzzy matrix to
  [`fuzzy_codas`][bulanik.mcdm.fuzzy_ranking.fuzzy_codas],
  [`fuzzy_waspas`][bulanik.mcdm.fuzzy_ranking.fuzzy_waspas] or
  [`fuzzy_cocoso`][bulanik.mcdm.fuzzy_ranking.fuzzy_cocoso].
- [`fuzzy_topsis`][bulanik.mcdm.fuzzy.fuzzy_topsis] keeps the numbers fuzzy
  throughout, following Chen (2000), for triangular numbers.

References:
    Chen (2000), Fuzzy Sets and Systems 114(1), 1-9.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .._validate import check_weights, choose, finite
from ..aggregate import aggregate_experts
from ..numbers import TFN, FuzzyNumber
from ..scales import LinguisticScale
from ._common import check_types, default_names, infer_criteria, rank_order, weights_in_order
from .fuzzy_ranking import fuzzy_cocoso, fuzzy_codas, fuzzy_waspas
from .outranking import electre
from .ranking import edas, topsis, vikor, waspas
from .results import BWMResult, RankingResult, WeightingResult
from .scoring import cocoso, codas, copras, marcos, wisp

__all__ = ["DecisionMatrix", "rank_fuzzy", "fuzzy_topsis"]

_METHODS = {
    "topsis": topsis,
    "vikor": vikor,
    "waspas": waspas,
    "edas": edas,
    "codas": codas,
    "copras": copras,
    "marcos": marcos,
    "cocoso": cocoso,
    "wisp": wisp,
    "electre": electre,
}
# methods that take the aggregated fuzzy matrix itself, with ops and score
_FUZZY_METHODS = {
    "fuzzy_codas": fuzzy_codas,
    "fuzzy_waspas": fuzzy_waspas,
    "fuzzy_cocoso": fuzzy_cocoso,
}


def _unique(names: Sequence[str], what: str) -> tuple[str, ...]:
    out = tuple(names)
    if not out:
        raise ValueError(f"need at least one of {what}")
    if len(set(out)) != len(out):
        raise ValueError(f"{what} names must be unique")
    return out


@dataclass(frozen=True)
class DecisionMatrix:
    """Expert ratings `[expert][alternative][criterion]` of one fuzzy number kind.

    The input of [`rank_fuzzy`][bulanik.mcdm.fuzzy.rank_fuzzy]. Construction
    validates the shape and that all ratings are fuzzy numbers of the same
    kind, and normalises the containers to tuples.

    Attributes:
        alternatives: Alternative names (unique, at least one).
        criteria: Criterion names (unique, at least one).
        experts: Ratings indexed `[expert][alternative][criterion]`.
        expert_weights: Expert weights summing to 1, or `None` for equal
            weights.

    Raises:
        ValueError: If there are no experts, alternatives or criteria, names
            repeat, an expert's table has the wrong shape, or the expert
            weights are invalid (negative, wrong count, not summing to 1).
        TypeError: If a rating is not a fuzzy number or the kinds differ.

    Examples:
        >>> from bulanik import TFN
        >>> from bulanik.mcdm import DecisionMatrix
        >>> dm = DecisionMatrix(
        ...     alternatives=["A", "B"], criteria=["price"],
        ...     experts=[[[TFN(1, 2, 3)], [TFN(2, 3, 4)]]])
        >>> dm.expert_weights is None
        True
        >>> dm.experts[0][1][0]
        TFN(l=2.0, m=3.0, u=4.0)
    """

    alternatives: tuple[str, ...]
    criteria: tuple[str, ...]
    experts: tuple[tuple[tuple[FuzzyNumber, ...], ...], ...]
    expert_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        alts = _unique(self.alternatives, "alternatives")
        crit = _unique(self.criteria, "criteria")
        experts = tuple(tuple(tuple(row) for row in e) for e in self.experts)
        if not experts:
            raise ValueError("need at least one expert")
        for e in experts:
            if len(e) != len(alts):
                raise ValueError(f"each expert must rate {len(alts)} alternatives")
            if any(len(row) != len(crit) for row in e):
                raise ValueError(f"each alternative must be rated on {len(crit)} criteria")
        first = experts[0][0][0]
        for e in experts:
            for row in e:
                for x in row:
                    if not isinstance(x, FuzzyNumber):
                        raise TypeError(f"ratings must be fuzzy numbers, got {type(x).__name__}")
                    first.same_kind(x)
        ew = None
        if self.expert_weights is not None:
            ew = tuple(check_weights(self.expert_weights, len(experts)))
        object.__setattr__(self, "alternatives", alts)
        object.__setattr__(self, "criteria", crit)
        object.__setattr__(self, "experts", experts)
        object.__setattr__(self, "expert_weights", ew)

    @classmethod
    def from_terms(
        cls,
        scale: LinguisticScale,
        alternatives: Sequence[str],
        criteria: Sequence[str],
        experts_terms: Sequence[Sequence[Sequence[str]]],
        expert_weights: Sequence[float] | None = None,
    ) -> "DecisionMatrix":
        """Build from linguistic terms `[expert][alternative][criterion]`.

        Args:
            scale: The [`LinguisticScale`][bulanik.scales.LinguisticScale] that
                maps each term to a fuzzy number.
            alternatives: Alternative names.
            criteria: Criterion names.
            experts_terms: Term names indexed `[expert][alternative][criterion]`.
            expert_weights: Expert weights summing to 1, or `None` for equal.

        Returns:
            The decision matrix of converted ratings.

        Raises:
            KeyError: If a term is not in the scale.
            ValueError: As for the constructor.

        Examples:
            >>> from bulanik import TFN
            >>> from bulanik.mcdm import DecisionMatrix
            >>> from bulanik.scales import LinguisticScale
            >>> scale = LinguisticScale("demo", {"L": TFN(0, 0, 5), "H": TFN(5, 10, 10)})
            >>> dm = DecisionMatrix.from_terms(scale, ["A", "B"], ["price"],
            ...                                [[["L"], ["H"]], [["H"], ["H"]]])
            >>> dm.experts[0][0][0]
            TFN(l=0.0, m=0.0, u=5.0)
        """
        return cls(
            tuple(alternatives),
            tuple(criteria),
            scale.convert(list(experts_terms)),
            None if expert_weights is None else tuple(expert_weights),
        )


def rank_fuzzy(
    matrix: DecisionMatrix,
    weights: Any,
    criteria_types: Sequence[str],
    *,
    method: str = "topsis",
    aggregation: str = "wa",
    ops: str = "algebraic",
    score: str | None = None,
    **method_kwargs: Any,
) -> RankingResult:
    """Aggregate experts, score the fuzzy numbers, then rank with a crisp method.

    The expert ratings are combined cell by cell with
    [`aggregate_experts`][bulanik.aggregate.aggregate_experts] (weighted
    average `"wa"` or weighted geometric mean `"wg"`), each aggregated number
    is converted to a crisp value with its kind's `score`, and the resulting
    matrix is ranked by `method` with the crisp function of the same name
    (so VIKOR scores are still lower-is-better). The returned `operators` are
    those of the crisp method plus `aggregation`, `ops` and `score`. ELECTRE
    already uses the key `score` for its own ranking score, which is kept as
    `ranking_score`.

    The methods `"fuzzy_codas"`, `"fuzzy_waspas"` and `"fuzzy_cocoso"` are
    different: the aggregated **fuzzy** matrix goes straight to
    [`fuzzy_codas`][bulanik.mcdm.fuzzy_ranking.fuzzy_codas],
    [`fuzzy_waspas`][bulanik.mcdm.fuzzy_ranking.fuzzy_waspas] or
    [`fuzzy_cocoso`][bulanik.mcdm.fuzzy_ranking.fuzzy_cocoso] with `ops` and
    `score` (no scoring first; `score=None` leaves the choice to the method),
    and `operators` are the method's plus `aggregation`.

    Args:
        matrix: The expert ratings.
        weights: Criteria weights in any form the crisp methods accept (a
            sequence, a mapping, a
            [`WeightingResult`][bulanik.mcdm.results.WeightingResult] or
            [`BWMResult`][bulanik.mcdm.results.BWMResult]).
        criteria_types: `"benefit"` or `"cost"` for each criterion.
        method: `"topsis"`, `"vikor"`, `"waspas"`, `"edas"`, `"codas"`,
            `"copras"`, `"marcos"`, `"cocoso"`, `"wisp"`, `"electre"`,
            `"fuzzy_codas"`, `"fuzzy_waspas"` or `"fuzzy_cocoso"`.
        aggregation: `"wa"` or `"wg"`.
        ops: Operator set of the fuzzy number kind (`"algebraic"` for TFN).
        score: Score name for defuzzification, or `None` for the kind's
            default. The ratio methods (`waspas`, `codas`, `copras`, `marcos`,
            `cocoso`, `wisp`) need strictly positive scores; the orthopair
            kinds default to a signed score, so pass `"normalized"` for them.
            Kinds without that score (for example `IVNeutrosophicFN`) get a
            "use a positive-valued score" hint instead. For the fuzzy
            methods the name goes to the method unchanged; with `None`,
            `fuzzy_cocoso` uses `"normalized"` when the kind has it (else the
            kind's default) and `fuzzy_codas` / `fuzzy_waspas` the kind's default.
        **method_kwargs: Passed to the crisp method, e.g. `v=` for VIKOR,
            `lam=` for WASPAS and CoCoSo, `tau=` and `psi=` for CODAS,
            `product=` for WISP, `c_threshold=` and `d_threshold=` for ELECTRE,
            `normalize_weights=` (and the same keywords for the fuzzy methods).

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult].

    Raises:
        ValueError: If `method`, `aggregation`, `ops` or `score` is unknown, or
            the crisp (or fuzzy) method rejects its input.
        TypeError: If a fuzzy method rejects the number kind (see its Raises).

    Examples:
        >>> from bulanik import TFN
        >>> from bulanik.mcdm import DecisionMatrix, rank_fuzzy
        >>> from bulanik.scales import LinguisticScale
        >>> scale = LinguisticScale("demo", {"L": TFN(0, 0, 5), "M": TFN(3, 5, 7), "H": TFN(5, 10, 10)})
        >>> dm = DecisionMatrix.from_terms(
        ...     scale, ["A", "B"], ["price", "quality"],
        ...     [[["L", "H"], ["M", "M"]], [["M", "H"], ["H", "L"]]])
        >>> r = rank_fuzzy(dm, {"price": 0.4, "quality": 0.6}, ["cost", "benefit"])
        >>> r.ranking
        ('A', 'B')
        >>> r.operators["aggregation"], r.operators["score"]
        ('wa', 'centroid')
        >>> rank_fuzzy(dm, [0.4, 0.6], ["cost", "benefit"], method="waspas", lam=0.7).ranking
        ('A', 'B')
        >>> from bulanik.scales import EXAMPLE_IFN_5
        >>> ifn = DecisionMatrix.from_terms(EXAMPLE_IFN_5, ["A", "B"], ["price", "quality"],
        ...                                 [[["L", "H"], ["M", "M"]]])
        >>> r = rank_fuzzy(ifn, [0.4, 0.6], ["cost", "benefit"], method="fuzzy_cocoso")
        >>> r.ranking, r.operators["score"]
        (('A', 'B'), 'normalized')
    """
    method = choose(method, (*_METHODS, *_FUZZY_METHODS), "method", "rank_fuzzy")
    aggregation = choose(aggregation, ("wa", "wg"), "aggregation", "rank_fuzzy")
    agg = aggregate_experts(
        matrix.experts, matrix.expert_weights, method=aggregation, ops=ops
    )
    if method in _FUZZY_METHODS:
        res = _FUZZY_METHODS[method](
            agg,
            weights,
            criteria_types,
            alternatives=list(matrix.alternatives),
            criteria=list(matrix.criteria),
            ops=ops,
            score=score,
            **method_kwargs,
        )
        return dataclasses.replace(res, operators=res.operators | {"aggregation": aggregation})
    sample = agg[0][0]
    resolved = choose(score, sample.SCORES, "score", type(sample).__name__, sample.DEFAULT_SCORE)
    crisp = [[x.score(resolved) for x in row] for row in agg]
    try:
        res = _METHODS[method](
            crisp,
            weights,
            criteria_types,
            alternatives=list(matrix.alternatives),
            criteria=list(matrix.criteria),
            **method_kwargs,
        )
    except ValueError as exc:
        hint = 'score="normalized"'
        if hint in str(exc) and "normalized" not in type(sample).SCORES:
            raise ValueError(
                str(exc).replace(f"try {hint}", "use a positive-valued score")
            ) from exc
        raise
    operators = dict(res.operators)
    if "score" in operators:  # ELECTRE names its own ranking score; keep it
        operators["ranking_score"] = operators.pop("score")
    operators |= {"aggregation": aggregation, "ops": ops, "score": resolved}
    return dataclasses.replace(res, operators=operators)


def _vertex(a: TFN, b: TFN) -> float:
    return math.sqrt(((a.l - b.l) ** 2 + (a.m - b.m) ** 2 + (a.u - b.u) ** 2) / 3.0)


def _unit_weight(x: TFN | float, name: str) -> TFN:
    """A TFN or crisp weight inside [0, 1], as Chen's FPIS (1,1,1) / FNIS (0,0,0) need."""
    if not isinstance(x, TFN):
        c = finite(x, f"weight of criterion '{name}'")
        x = TFN(c, c, c) if 0.0 <= c <= 1.0 else None  # type: ignore[assignment]
    if x is None or x.l < 0.0 or x.u > 1.0:
        raise ValueError(
            f"weight of criterion '{name}' must lie in [0, 1] for fuzzy TOPSIS "
            "(the ideal solutions are (1,1,1) and (0,0,0)); normalise it, e.g. divide "
            "a 0-10 scale weight by 10"
        )
    return x


def fuzzy_topsis(
    matrix: Sequence[Sequence[TFN]],
    weights: Sequence[TFN | float] | Mapping[str, TFN | float] | WeightingResult | BWMResult,
    criteria_types: Sequence[str],
    *,
    alternatives: Sequence[str] | None = None,
    criteria: Sequence[str] | None = None,
) -> RankingResult:
    r"""Fuzzy TOPSIS of Chen (2000) on triangular fuzzy numbers.

    Ratings are normalised per criterion to $[0, 1]$: for a benefit,
    $\tilde r_{ij} = (l_{ij}, m_{ij}, u_{ij}) / c_j^*$ with $c_j^* = \max_i
    u_{ij}$; for a cost, $\tilde r_{ij} = (a_j^-/u_{ij}, a_j^-/m_{ij},
    a_j^-/l_{ij})$ with $a_j^- = \min_i l_{ij}$. They are multiplied by the
    weights (componentwise, $\tilde v_{ij} = \tilde r_{ij}\otimes\tilde w_j$).
    The fuzzy positive and negative ideal solutions are fixed at
    $(1,1,1)$ and $(0,0,0)$, the distance between triangular numbers is the
    vertex distance $d(\tilde a, \tilde b) = \sqrt{\tfrac13 \sum (a_k -
    b_k)^2}$, and the score is the closeness coefficient
    $CC_i = d_i^- / (d_i^* + d_i^-)$ with $d_i^* = \sum_j d(\tilde v_{ij},
    (1,1,1))$ and $d_i^- = \sum_j d(\tilde v_{ij}, (0,0,0))$, larger is
    better. `extras` holds `d_star` and `d_minus`.

    **The weights must lie in [0, 1]** (TFN or crisp). Because the ideal
    solutions are (1,1,1) and (0,0,0), a weight above 1 would push the
    weighted ratings past the "ideal"; normalise a 0-10 weight by dividing by
    10. Weights need not sum to 1.

    Args:
        matrix: Alternatives by criteria of [`TFN`][bulanik.numbers.TFN]
            ratings. Benefit ratings need $l \ge 0$ and a positive upper
            bound somewhere in the column; cost ratings need $l > 0$.
        weights: Weights in $[0, 1]$: a sequence in criteria order, a mapping by
            criterion name (naming every criterion and no others, as for the
            crisp methods) or a
            [`WeightingResult`][bulanik.mcdm.results.WeightingResult] /
            [`BWMResult`][bulanik.mcdm.results.BWMResult].
        criteria_types: `"benefit"` or `"cost"` for each criterion.
        alternatives: Alternative names; `A1..Am` by default.
        criteria: Criterion names; if `None`, the keys (in order) of a mapping
            or result `weights`, else `C1..Cn`. They appear in error messages
            and select the mapping entries.

    Returns:
        A [`RankingResult`][bulanik.mcdm.results.RankingResult] with
        `operators == {"normalization": "linear_scale_chen2000",
        "distance": "vertex"}`.

    Raises:
        TypeError: If a rating is not a TFN.
        ValueError: If the matrix is empty or ragged, names or weights do not
            match, a weight lies outside $[0, 1]$, or a rating violates the
            sign requirements above.

    Examples:
        >>> from bulanik import TFN
        >>> from bulanik.mcdm import fuzzy_topsis
        >>> m = [[TFN(5, 7, 9), TFN(3, 5, 7)],
        ...      [TFN(7, 9, 9), TFN(1, 3, 5)],
        ...      [TFN(3, 5, 7), TFN(7, 9, 9)]]
        >>> r = fuzzy_topsis(m, {"quality": TFN(.5, .7, .9), "price": 0.3},
        ...                  ["benefit", "cost"], alternatives=["X", "Y", "Z"])
        >>> r.ranking
        ('Y', 'X', 'Z')
        >>> {k: round(v, 3) for k, v in r.scores.items()}
        {'X': 0.329, 'Y': 0.413, 'Z': 0.243}
        >>> fuzzy_topsis(m, [5, 0.3], ["benefit", "cost"])
        Traceback (most recent call last):
        ...
        ValueError: weight of criterion 'C1' must lie in [0, 1] for fuzzy TOPSIS ...

    References:
        Chen (2000), Fuzzy Sets and Systems 114(1), 1-9.
    """
    rows = [list(r) for r in matrix]
    if not rows or not rows[0]:
        raise ValueError("matrix must have at least one row and one column")
    m, n = len(rows), len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError("matrix rows must all have the same length")
    for r in rows:
        for x in r:
            if not isinstance(x, TFN):
                raise TypeError(f"fuzzy_topsis needs TFN entries, got {type(x).__name__}")
    types = check_types(criteria_types, n)
    criteria = infer_criteria(weights, criteria)
    crit = list(criteria) if criteria is not None else default_names("C", n)
    if len(crit) != n:
        raise ValueError(f"expected {n} criteria, got {len(crit)}")
    if len(set(crit)) != n:
        raise ValueError("criteria names must be unique")
    ws = weights_in_order(weights, crit)
    if len(ws) != n:
        raise ValueError(f"expected {n} weights, got {len(ws)}")
    alts = list(alternatives) if alternatives is not None else default_names("A", m)
    if len(alts) != m or len(set(alts)) != m:
        raise ValueError(f"alternatives must be {m} unique names")
    w = [_unit_weight(x, crit[j]) for j, x in enumerate(ws)]
    v: list[list[TFN]] = [[None] * n for _ in range(m)]  # type: ignore[list-item]
    for j in range(n):
        col = [rows[i][j] for i in range(m)]
        if types[j] == "benefit":
            l_min = min(x.l for x in col)
            if l_min < 0:
                raise ValueError(
                    f"benefit criterion '{crit[j]}' needs ratings with l >= 0 (min lower bound {l_min})"
                )
            c_star = max(x.u for x in col)
            if c_star <= 0:
                raise ValueError(f"benefit criterion '{crit[j]}' needs a positive upper bound")
            r = [TFN(x.l / c_star, x.m / c_star, x.u / c_star) for x in col]
        else:
            a_min = min(x.l for x in col)
            if a_min <= 0:
                raise ValueError(
                    f"cost criterion '{crit[j]}' needs positive values (min lower bound {a_min})"
                )
            r = [TFN(a_min / x.u, a_min / x.m, a_min / x.l) for x in col]
        for i in range(m):
            v[i][j] = TFN(r[i].l * w[j].l, r[i].m * w[j].m, r[i].u * w[j].u)
    one, zero = TFN(1, 1, 1), TFN(0, 0, 0)
    d_star = [sum(_vertex(x, one) for x in v[i]) for i in range(m)]
    d_minus = [sum(_vertex(x, zero) for x in v[i]) for i in range(m)]
    cc = [dm / (ds + dm) if ds + dm > 0 else 0.5 for ds, dm in zip(d_star, d_minus)]
    scores = dict(zip(alts, cc))
    return RankingResult(
        method="fuzzy_topsis",
        alternatives=tuple(alts),
        scores=scores,
        ranking=rank_order(scores, True),
        operators={"normalization": "linear_scale_chen2000", "distance": "vertex"},
        extras={"d_star": dict(zip(alts, d_star)), "d_minus": dict(zip(alts, d_minus))},
    )
