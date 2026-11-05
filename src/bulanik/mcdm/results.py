"""Result containers shared by the MCDM methods.

Three frozen dataclasses: [`WeightingResult`][bulanik.mcdm.results.WeightingResult]
(criteria weights), [`RankingResult`][bulanik.mcdm.results.RankingResult]
(alternative scores) and [`BWMResult`][bulanik.mcdm.results.BWMResult] (best-worst
weights). Each carries a `ranking` tuple (best first) and an `operators` mapping
that names the operator choices behind the numbers, so a result is
self-describing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..numbers import FuzzyNumber

__all__ = ["WeightingResult", "RankingResult", "BWMResult", "DEMATELResult"]


@dataclass(frozen=True)
class WeightingResult:
    r"""Criteria weights from a weighting method (`ranking` is best first).

    Attributes:
        method: Method name, e.g. `"ahp"`.
        criteria: Criterion names in input order.
        weights: Weight of each criterion; they sum to 1.
        ranking: Criterion names sorted by weight, largest first (ties keep
            input order).
        operators: Operator choices used, e.g. `{"weights": "eigenvector"}`.
        consistency_ratio: Saaty's consistency ratio $CR = CI / RI$ for
            [`ahp`][bulanik.mcdm.weighting.ahp] and
            [`fuzzy_ahp`][bulanik.mcdm.weighting.fuzzy_ahp]; `None` for methods
            without one.
        fuzzy_weights: The fuzzy weights before defuzzification
            ([`fuzzy_ahp`][bulanik.mcdm.weighting.fuzzy_ahp] only), else `None`.
        extras: Method-specific additional values.

    Examples:
        >>> from bulanik.mcdm import swara
        >>> r = swara([0.3], ["a", "b"])
        >>> r.ranking, round(r.weights["a"], 4)
        (('a', 'b'), 0.5652)
        >>> r.consistency_ratio is None
        True
    """

    method: str
    criteria: tuple[str, ...]
    weights: dict[str, float]
    ranking: tuple[str, ...]
    operators: dict[str, str]
    consistency_ratio: float | None = None
    fuzzy_weights: dict[str, FuzzyNumber] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingResult:
    """Alternative scores and their ranking (`ranking` is best first).

    "Best" depends on the method: a larger score is better for TOPSIS, WASPAS
    and EDAS, a smaller one for VIKOR (whose `ranking` is already ordered
    accordingly). Use `ranking` rather than sorting `scores` yourself.

    Attributes:
        method: Method name, e.g. `"topsis"`.
        alternatives: Alternative names in input order.
        scores: Score of each alternative.
        ranking: Alternative names, best first (ties keep input order).
        operators: Operator choices used, e.g. `{"normalization": "vector"}`.
        extras: Per-alternative intermediate values keyed by name, e.g.
            `d_plus` / `d_minus` for TOPSIS; each maps alternative to value.

    Examples:
        >>> from bulanik.mcdm import vikor
        >>> m = [[250, 16, 12, 5], [200, 16, 8, 3], [300, 32, 16, 4], [275, 32, 8, 4]]
        >>> r = vikor(m, [0.3, 0.3, 0.2, 0.2], ["cost", "benefit", "benefit", "benefit"])
        >>> r.ranking, sorted(r.extras)
        (('A4', 'A3', 'A1', 'A2'), ['R', 'S'])
    """

    method: str
    alternatives: tuple[str, ...]
    scores: dict[str, float]
    ranking: tuple[str, ...]
    operators: dict[str, str]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BWMResult:
    r"""Best-worst method weights (`ranking` is best first).

    Produced by [`bwm_linear`][bulanik.lumix.bwm_linear],
    [`BWMProblem`][bulanik.sezgi.BWMProblem] and
    [`FuzzyBWMProblem`][bulanik.sezgi.FuzzyBWMProblem].

    Attributes:
        method: `"bwm_linear"`, `"bwm_nonlinear"` or `"fuzzy_bwm"`.
        criteria: Criterion names in input order.
        weights: Weight of each criterion; they sum to 1.
        ranking: Criterion names sorted by weight, largest first.
        xi: Optimal objective of the BWM model, the largest deviation between
            the comparisons and the weight ratios.
        consistency_ratio: **Not the same quantity in every producer.**
            [`bwm_linear`][bulanik.lumix.bwm_linear] stores the linear model's
            $\xi^L$ itself; [`BWMProblem`][bulanik.sezgi.BWMProblem] and
            [`FuzzyBWMProblem`][bulanik.sezgi.FuzzyBWMProblem] store
            $\xi / CI$ (and `None` when $CI = 0$ and the comparisons are
            inconsistent). Check `operators["consistency"]` before comparing.
        best: The best (most important) reference criterion.
        worst: The worst (least important) reference criterion.
        operators: Operator choices used, including `"consistency"`.
        fuzzy_weights: Fuzzy weights ([`FuzzyBWMProblem`][bulanik.sezgi.FuzzyBWMProblem]
            only), else `None`.
        extras: Solver details (e.g. `evals_used`, `best_f`, or the LumiX model).
    """

    method: str
    criteria: tuple[str, ...]
    weights: dict[str, float]
    ranking: tuple[str, ...]
    xi: float
    consistency_ratio: float | None
    best: str
    worst: str
    operators: dict[str, str]
    fuzzy_weights: dict[str, FuzzyNumber] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DEMATELResult:
    r"""DEMATEL influence analysis: total-relation matrix, cause/effect groups and weights.

    Produced by [`dematel`][bulanik.mcdm.dematel.dematel]. `ranking` lists the
    factors by weight, largest first (ties keep input order).

    Attributes:
        method: `"dematel"`.
        factors: Factor names in input order.
        direct: Crisp direct-relation matrix (after defuzzification).
        normalized: Normalised matrix $N = A/s$.
        total: Total-relation matrix $T = N(I - N)^{-1}$.
        d: Row sums of $T$ (influence given), by factor.
        r: Column sums of $T$ (influence received), by factor.
        prominence: $D + R$, by factor.
        relation: $D - R$, by factor.
        causes: Factors with `relation > 1e-12 * max(prominence)`, in input order.
        effects: The other factors (a zero net relation counts as an effect).
        threshold: Cut-off used for `relations`.
        relations: `(from, to, t_ij)` for every off-diagonal $t_{ij}$ above the
            threshold, sorted row-major (by `from` then `to` in input order).
        weights: Factor weights summing to 1.
        ranking: Factor names by weight, best first.
        operators: Operator choices used.
        extras: Method-specific additional values.
    """

    method: str
    factors: tuple[str, ...]
    direct: tuple[tuple[float, ...], ...]
    normalized: tuple[tuple[float, ...], ...]
    total: tuple[tuple[float, ...], ...]
    d: dict[str, float]
    r: dict[str, float]
    prominence: dict[str, float]
    relation: dict[str, float]
    causes: tuple[str, ...]
    effects: tuple[str, ...]
    threshold: float
    relations: tuple[tuple[str, str, float], ...]
    weights: dict[str, float]
    ranking: tuple[str, ...]
    operators: dict[str, str]
    extras: dict[str, Any] = field(default_factory=dict)

    def as_weights(self) -> WeightingResult:
        """The factor weights as a [`WeightingResult`][bulanik.mcdm.results.WeightingResult].

        Usable as the criteria weights of every ranking method.
        """
        return WeightingResult(
            method="dematel",
            criteria=self.factors,
            weights=dict(self.weights),
            ranking=self.ranking,
            operators=dict(self.operators),
            extras={"prominence": dict(self.prominence), "relation": dict(self.relation)},
        )
