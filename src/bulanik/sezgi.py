r"""sezgi bridge: nonlinear and fuzzy BWM, and an ANFIS trainer, as sezgi problems.

Needs the `sezgi` extra (`pip install "bulanik[sezgi]"`). sezgi is a
metaheuristic optimisation library; here its problems are minimised without
derivatives.

- [`BWMProblem`][bulanik.sezgi.BWMProblem] - Rezaei's (2015) nonlinear
  best-worst method.
- [`FuzzyBWMProblem`][bulanik.sezgi.FuzzyBWMProblem] - Guo & Zhao's (2017) fuzzy
  best-worst method.
- [`ANFISProblem`][bulanik.sezgi.ANFISProblem] /
  [`train_anfis`][bulanik.sezgi.train_anfis] - derivative-free training of an
  [`ANFIS`][bulanik.anfis.ANFIS].

sezgi has no single-objective constraint API, so every constraint is a
penalty inside `evaluate` (minimised). The default algorithm everywhere is
`sezgi.DifferentialEvolution(pop_size=40, variant="jde")`; pass any other
sezgi algorithm as `algorithm=`. Runs are deterministic for a fixed `seed` and
`budget` (the number of objective evaluations).

!!! warning "The two `consistency_ratio` fields differ"
    In the [`BWMResult`][bulanik.mcdm.results.BWMResult] returned by
    `BWMProblem.solve` the `consistency_ratio` is Rezaei's (2015) ratio
    $\xi / CI(a_{BW})$, where $CI$ is
    [`bwm_consistency_index`][bulanik.mcdm.bwm.bwm_consistency_index] of the
    best-over-worst comparison $a_{BW}$. It lies in $[0, 1]$ at the
    optimum and is analogous to (not interchangeable with) AHP's
    consistency ratio. When $a_{BW} = 1$ the index is 0: the ratio is then
    `0.0` if $\xi \le 10^{-6}$ (consistent) and `None` otherwise, with an
    explanation in `extras["consistency_note"]`. `FuzzyBWMProblem.solve`
    reports $k / CI$ with Guo & Zhao's
    [`fuzzy_bwm_consistency_index`][bulanik.mcdm.bwm.fuzzy_bwm_consistency_index],
    which is never 0. By contrast, [`bwm_linear`][bulanik.lumix.bwm_linear] in
    `bulanik.lumix` stores the linear model's optimum $\xi^L$ itself under the
    same field name. Read `operators["consistency"]` (`"rezaei2015_ci"`,
    `"guo_zhao2017_ci"` or `"xi_L"`) before comparing results.

Examples:
    >>> from bulanik.sezgi import BWMProblem
    >>> problem = BWMProblem({"price": 1, "quality": 2, "speed": 8},
    ...                      {"price": 8, "quality": 4, "speed": 1})
    >>> r = problem.solve(budget=3000, seed=1)
    >>> {k: round(v, 2) for k, v in r.weights.items()}
    {'price': 0.62, 'quality': 0.31, 'speed': 0.08}
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    from sezgi import DifferentialEvolution, Float, Problem
except ImportError as exc:  # pragma: no cover - exercised with sezgi hidden
    raise ImportError(
        "bulanik.sezgi needs the 'sezgi' extra: install it with "
        'pip install "bulanik[sezgi]"'
    ) from exc

from ._validate import finite
from .anfis import ANFIS
from .mcdm._common import rank_order
from .mcdm.bwm import bwm_consistency_index, fuzzy_bwm_consistency_index, resolve_best_worst
from .mcdm.results import BWMResult
from .numbers import TFN

__all__ = [
    "BWMProblem",
    "FuzzyBWMProblem",
    "ANFISProblem",
    "ANFISTrainResult",
    "train_anfis",
]

_LOWER = 1e-6  # lower bound of every BWM weight component
_CONSISTENT_XI = 1e-6  # xi at or below this counts as consistent when the CI is 0 (a_BW = 1)


def _default_algorithm() -> Any:
    return DifferentialEvolution(pop_size=40, variant="jde")


def _penalty(p: float) -> float:
    p = finite(p, "penalty")
    if p <= 0.0:
        raise ValueError(f"penalty must be > 0, got {p}")
    return p


def _budget(budget: int) -> int:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"budget must be an integer >= 1, got {budget!r}")
    return budget


def _vector(x: Sequence[float], n: int, positive: bool) -> list[float]:
    vals = list(x)
    if len(vals) != n:
        raise ValueError(f"expected {n} values, got {len(vals)}")
    out = [finite(v, "x") for v in vals]
    if positive and any(v <= 0.0 for v in out):
        raise ValueError(f"every weight component must be > 0, got {out}")
    return out


def _run(problem: Any, algorithm: Any, budget: int, seed: int) -> Any:
    algo = _default_algorithm() if algorithm is None else algorithm
    return algo, algo.run(problem, budget=_budget(budget), seed=seed)


# --------------------------------------------------------------------- crisp BWM


class BWMProblem(Problem):
    r"""Rezaei's (2015, Omega 53:49-57) nonlinear best-worst method.

    With best criterion $B$, worst criterion $W$ and comparisons $a_{Bj}$,
    $a_{jW}$ the model is

    $$\min \xi \quad \text{s.t.} \quad \left|\frac{w_B}{w_j} - a_{Bj}\right| \le \xi\ (j \ne B),\;
    \left|\frac{w_j}{w_W} - a_{jW}\right| \le \xi\ (j \ne W),\; \textstyle\sum_j w_j = 1.$$

    `evaluate(x)` is the largest of those deviations plus
    `penalty * |sum(x) - 1|`; the ratios are scale-invariant, so the penalty only
    fixes the scale. Search space: `Float(1e-6, 1, n)`.

    !!! warning "`solve().consistency_ratio` is $\xi / CI$, not $\xi^L$"
        Here `consistency_ratio` is Rezaei's (2015) ratio $\xi / CI(a_{BW})$, a
        normalised ratio, analogous to (not interchangeable with) AHP's
        consistency ratio; Rezaei's acceptance thresholds differ. It is *not*
        the quantity [`bwm_linear`][bulanik.lumix.bwm_linear] reports under
        the same field name, which is the linear model's $\xi^L$ itself. For
        $a_{BW} = 1$ the index $CI$ is 0, so the ratio is `0.0` when
        $\xi \le 10^{-6}$ and `None` (with a note in
        `extras["consistency_note"]`) otherwise.

    Attributes:
        criteria: Criterion names in order.
        best: The best criterion.
        worst: The worst criterion.
        best_to_others: Validated best-to-others comparisons by criterion.
        others_to_worst: Validated others-to-worst comparisons by criterion.
        penalty: Weight of the sum-to-one penalty.

    Args:
        best_to_others: Comparison of the best criterion over each criterion.
        others_to_worst: Comparison of each criterion over the worst criterion.
        criteria: Criterion names, fixing the order. Defaults to the order of
            `best_to_others`.
        best: The best criterion. Inferred as the unique one with comparison 1
            when `None`; pass it when several criteria tie at 1.
        worst: The worst criterion, inferred likewise.
        penalty: Weight of the penalty for violating the sum-to-one constraint
            (positive).

    Raises:
        ValueError: For invalid comparison vectors (see
            [`resolve_best_worst`][bulanik.mcdm.bwm.resolve_best_worst]) or a
            non-positive `penalty`.

    Examples:
        >>> from bulanik.sezgi import BWMProblem
        >>> problem = BWMProblem({"a": 1, "b": 2, "c": 8}, {"a": 8, "b": 4, "c": 1})
        >>> problem.best, problem.worst
        ('a', 'c')
        >>> round(problem.deviation([8, 4, 1]), 6)   # perfectly consistent, any scale
        0.0
        >>> round(problem.deviation([0.6, 0.3, 0.1]), 6)
        2.0

    References:
        Rezaei (2015), Omega 53, 49-57.
    """

    def __init__(
        self,
        best_to_others: Mapping[str, float],
        others_to_worst: Mapping[str, float],
        criteria: Sequence[str] | None = None,
        *,
        best: str | None = None,
        worst: str | None = None,
        penalty: float = 10.0,
    ) -> None:
        crit, b, w = resolve_best_worst(best_to_others, others_to_worst, criteria, best, worst)
        self.criteria = crit
        self.best, self.worst = b, w
        self.best_to_others = {c: finite(best_to_others[c], f"best_to_others[{c}]") for c in crit}
        self.others_to_worst = {c: finite(others_to_worst[c], f"others_to_worst[{c}]") for c in crit}
        self.penalty = _penalty(penalty)

    def space(self) -> Any:
        return Float(_LOWER, 1.0, len(self.criteria))

    def deviation(self, x: Sequence[float]) -> float:
        """The BWM objective $\\xi$ at weights `x`, without the penalty.

        Args:
            x: One positive weight per criterion, in criteria order (need not
                sum to 1).

        Returns:
            The largest deviation $\\max(|w_B/w_j - a_{Bj}|, |w_j/w_W - a_{jW}|)$.

        Raises:
            ValueError: If `x` has the wrong length or a component is not a
                finite positive number.
        """
        vals = _vector(x, len(self.criteria), positive=True)
        wt = dict(zip(self.criteria, vals))
        wb, ww = wt[self.best], wt[self.worst]
        devs = [abs(wb / wt[j] - self.best_to_others[j]) for j in self.criteria if j != self.best]
        devs += [abs(wt[j] / ww - self.others_to_worst[j]) for j in self.criteria if j != self.worst]
        return max(devs)

    def evaluate(self, x: Sequence[float]) -> float:
        vals = _vector(x, len(self.criteria), positive=True)
        return self.deviation(vals) + self.penalty * abs(math.fsum(vals) - 1.0)

    def solve(self, algorithm: Any = None, *, budget: int = 20000, seed: int = 0) -> BWMResult:
        r"""Minimise with sezgi and return weights rescaled to sum to 1.

        `consistency_ratio` is $\xi /$
        [`bwm_consistency_index`][bulanik.mcdm.bwm.bwm_consistency_index]$(a_{BW})$,
        with $\xi$ recomputed at the rescaled weights. For $a_{BW} = 1$ that
        index is 0: the ratio is then `0.0` when $\xi \le 10^{-6}$
        (consistent) and `None` otherwise, with `extras["consistency_note"]`
        saying why. This differs from
        [`bwm_linear`][bulanik.lumix.bwm_linear], whose `consistency_ratio` is
        $\xi^L$ itself. `extras` also holds `evals_used` and `best_f`.

        Args:
            algorithm: A sezgi algorithm instance; `None` selects
                `DifferentialEvolution(pop_size=40, variant="jde")`.
            budget: Number of objective evaluations (integer, at least 1).
            seed: Random seed, making the run reproducible.

        Returns:
            A [`BWMResult`][bulanik.mcdm.results.BWMResult] with
            `method == "bwm_nonlinear"` and
            `operators["consistency"] == "rezaei2015_ci"`.

        Raises:
            ValueError: If `budget` is not a positive integer.

        Examples:
            >>> from bulanik.sezgi import BWMProblem
            >>> problem = BWMProblem({"a": 1, "b": 2, "c": 8}, {"a": 8, "b": 4, "c": 1})
            >>> r = problem.solve(budget=3000, seed=1)
            >>> r.ranking, r.xi < 0.01, r.consistency_ratio < 0.01
            (('a', 'b', 'c'), True, True)

            With $a_{BW} = 1$ the consistency index is 0, so an inconsistent
            set of comparisons has no ratio:

            >>> p = BWMProblem({"a": 1, "b": 1, "c": 3}, {"a": 1, "b": 1, "c": 2},
            ...                best="a", worst="b")
            >>> r = p.solve(budget=2000, seed=1)
            >>> r.consistency_ratio is None
            True
            >>> r.extras["consistency_note"]
            "the comparisons are inconsistent (xi = 1.00213) but Rezaei's consistency index is 0 for a_BW = 1, so no consistency ratio exists"
        """
        algo, res = _run(self, algorithm, budget, seed)
        x = _vector(res.best_x, len(self.criteria), positive=True)
        total = math.fsum(x)
        weights = {c: v / total for c, v in zip(self.criteria, x)}
        xi = self.deviation(list(weights.values()))
        ci = bwm_consistency_index(self.best_to_others[self.worst])
        extras: dict[str, Any] = {"evals_used": res.evals_used, "best_f": res.best_f}
        cr: float | None
        if ci > 0.0:
            cr = xi / ci
        elif xi <= _CONSISTENT_XI:
            cr = 0.0
        else:
            cr = None
            extras["consistency_note"] = (
                f"the comparisons are inconsistent (xi = {xi:.6g}) but Rezaei's "
                "consistency index is 0 for a_BW = 1, so no consistency ratio exists"
            )
        return BWMResult(
            method="bwm_nonlinear",
            criteria=tuple(self.criteria),
            weights=weights,
            ranking=rank_order(weights, descending=True),
            xi=xi,
            consistency_ratio=cr,
            best=self.best,
            worst=self.worst,
            operators={"model": "nonlinear", "consistency": "rezaei2015_ci",
                       "algorithm": type(algo).__name__},
            extras=extras,
        )


# --------------------------------------------------------------------- fuzzy BWM


def _graded_mean(l: float, m: float, u: float) -> float:  # noqa: E741
    return (l + 4.0 * m + u) / 6.0


def _check_tfn(v: Any, what: str) -> TFN:
    if not isinstance(v, TFN):
        raise TypeError(f"{what} must be a TFN, got {type(v).__name__}")
    if v.l <= 0.0:
        raise ValueError(f"{what} must have l > 0, got {v}")
    if v.m < 1.0:
        raise ValueError(f"{what} must be >= 1 in its middle value (m), got {v}")
    return v


class FuzzyBWMProblem(Problem):
    r"""Guo & Zhao's (2017, Knowledge-Based Systems 121:23-31) fuzzy best-worst method.

    The weights are triangular numbers $\tilde w_j = (l_j, m_j, u_j)$. With
    fuzzy comparisons $\tilde a_{Bj} = (l_{Bj}, m_{Bj}, u_{Bj})$, the model is
    $\min k$ subject to, for $j \ne B$,

    $$|l_B - l_{Bj} u_j| \le k u_j,\quad |m_B - m_{Bj} m_j| \le k m_j,\quad
    |u_B - u_{Bj} l_j| \le k l_j$$

    (since $\tilde w_B / \tilde w_j = (l_B/u_j, m_B/m_j, u_B/l_j)$), the same rows
    for $\tilde w_j / \tilde w_W$ with $\tilde a_{jW}$ for $j \ne W$, the
    normalisation $\sum_j R(\tilde w_j) = 1$ with the graded mean
    $R = (l + 4m + u)/6$, and $l \le m \le u$. `evaluate` is the largest
    |row| / denominator plus
    `penalty * (|sum_j R_j - 1| + sum_j [max(0, l_j - m_j) + max(0, m_j - u_j)])`,
    where both ordering terms are summed over the criteria. Search
    space: `Float(1e-6, 1, 3n)`, $(l, m, u)$ per criterion.

    Comparisons are [`TFN`][bulanik.numbers.TFN] with $l > 0$ and $m \ge 1$,
    e.g. from [`FUZZY_BWM_TFN`][bulanik.scales.FUZZY_BWM_TFN]; best and worst
    are found by their `TFN(1, 1, 1)` comparison.

    Attributes:
        criteria: Criterion names in order.
        best: The best criterion.
        worst: The worst criterion.
        best_to_others: Fuzzy best-to-others comparisons by criterion.
        others_to_worst: Fuzzy others-to-worst comparisons by criterion.
        penalty: Weight of the normalisation and ordering penalty.

    Args:
        best_to_others: Fuzzy comparison of the best criterion over each one.
        others_to_worst: Fuzzy comparison of each criterion over the worst.
        criteria: Criterion names, fixing the order. Defaults to the order of
            `best_to_others`.
        best: The best criterion; inferred from `TFN(1, 1, 1)` when `None`.
        worst: The worst criterion; inferred likewise.
        penalty: Penalty weight (positive).

    Raises:
        TypeError: If a comparison is not a TFN.
        ValueError: If a TFN has $l \le 0$ or $m < 1$, the vectors are invalid
            (see [`resolve_best_worst`][bulanik.mcdm.bwm.resolve_best_worst]),
            or `penalty` is not positive.

    Examples:
        >>> from bulanik.scales import FUZZY_BWM_TFN as F
        >>> from bulanik.sezgi import FuzzyBWMProblem
        >>> problem = FuzzyBWMProblem({"a": F["EI"], "b": F["WI"], "c": F["AI"]},
        ...                           {"a": F["AI"], "b": F["FI"], "c": F["EI"]})
        >>> problem.best, problem.worst, problem.space().__class__.__name__
        ('a', 'c', 'Float')

    References:
        Guo & Zhao (2017), Knowledge-Based Systems 121, 23-31.
    """

    def __init__(
        self,
        best_to_others: Mapping[str, TFN],
        others_to_worst: Mapping[str, TFN],
        criteria: Sequence[str] | None = None,
        *,
        best: str | None = None,
        worst: str | None = None,
        penalty: float = 10.0,
    ) -> None:
        for vec, name in ((best_to_others, "best_to_others"), (others_to_worst, "others_to_worst")):
            for c, v in vec.items():
                _check_tfn(v, f"{name}[{c}]")
        crit, b, w = resolve_best_worst(
            best_to_others, others_to_worst, criteria, best, worst, one=TFN(1, 1, 1)
        )
        self.criteria = crit
        self.best, self.worst = b, w
        self.best_to_others = {c: best_to_others[c] for c in crit}
        self.others_to_worst = {c: others_to_worst[c] for c in crit}
        self.penalty = _penalty(penalty)

    def space(self) -> Any:
        return Float(_LOWER, 1.0, 3 * len(self.criteria))

    def _triples(self, x: Sequence[float]) -> dict[str, tuple[float, float, float]]:
        vals = _vector(x, 3 * len(self.criteria), positive=True)
        return {c: (vals[3 * i], vals[3 * i + 1], vals[3 * i + 2]) for i, c in enumerate(self.criteria)}

    @staticmethod
    def _rows(num: tuple[float, float, float], den: tuple[float, float, float], a: TFN) -> list[float]:
        # num / den = (num.l / den.u, num.m / den.m, num.u / den.l) compared with a
        return [
            abs(num[0] - a.l * den[2]) / den[2],
            abs(num[1] - a.m * den[1]) / den[1],
            abs(num[2] - a.u * den[0]) / den[0],
        ]

    def deviation(self, x: Sequence[float]) -> float:
        """The fuzzy BWM objective $k$ at `x`, without the penalty.

        Args:
            x: The triples $(l, m, u)$ of all criteria, flattened in criteria
                order (length $3n$, positive components).

        Returns:
            The largest row deviation.

        Raises:
            ValueError: If `x` has the wrong length or a non-positive component.
        """
        t = self._triples(x)
        tb, tw = t[self.best], t[self.worst]
        devs: list[float] = []
        for j in self.criteria:
            if j != self.best:
                devs += self._rows(tb, t[j], self.best_to_others[j])
            if j != self.worst:
                devs += self._rows(t[j], tw, self.others_to_worst[j])
        return max(devs)

    def evaluate(self, x: Sequence[float]) -> float:
        t = self._triples(x)
        sum_r = math.fsum(_graded_mean(*v) for v in t.values())
        order = math.fsum(max(0.0, l - m) + max(0.0, m - u) for l, m, u in t.values())  # noqa: E741
        return self.deviation(x) + self.penalty * (abs(sum_r - 1.0) + order)

    def solve(self, algorithm: Any = None, *, budget: int = 100000, seed: int = 0) -> BWMResult:
        r"""Minimise with sezgi; return fuzzy weights rescaled so $\sum R = 1$.

        Each best triple is sorted so that $l \le m \le u$ (a no-op unless the
        search ended a hair inside the ordering penalty) and every triple is
        scaled by $1 / \sum R$, which leaves $k$ unchanged. `xi` is $k$
        recomputed at the returned weights, `weights` are the graded means
        $R(\tilde w_j)$, `fuzzy_weights` the TFNs and `consistency_ratio` is
        $k /$
        [`fuzzy_bwm_consistency_index`][bulanik.mcdm.bwm.fuzzy_bwm_consistency_index]$(\tilde a_{BW})$:
        a ratio to the index, like [`BWMProblem`][bulanik.sezgi.BWMProblem]'s and
        unlike [`bwm_linear`][bulanik.lumix.bwm_linear]'s, which is $\xi^L$
        itself. The fuzzy index is never 0, so the ratio is always a number.

        Args:
            algorithm: A sezgi algorithm instance; `None` selects
                `DifferentialEvolution(pop_size=40, variant="jde")`.
            budget: Number of objective evaluations (integer, at least 1).
            seed: Random seed, making the run reproducible.

        Returns:
            A [`BWMResult`][bulanik.mcdm.results.BWMResult] with
            `method == "fuzzy_bwm"` and `fuzzy_weights` set.

        Raises:
            ValueError: If `budget` is not a positive integer.

        Examples:
            >>> from bulanik.scales import FUZZY_BWM_TFN as F
            >>> from bulanik.sezgi import FuzzyBWMProblem
            >>> problem = FuzzyBWMProblem({"a": F["EI"], "b": F["WI"], "c": F["AI"]},
            ...                           {"a": F["AI"], "b": F["FI"], "c": F["EI"]})
            >>> r = problem.solve(budget=20000, seed=1)
            >>> r.ranking, {k: round(v, 2) for k, v in r.weights.items()}
            (('a', 'b', 'c'), {'a': 0.51, 'b': 0.35, 'c': 0.14})
            >>> round(sum(r.weights.values()), 6)
            1.0
            >>> r.operators["consistency"]
            'guo_zhao2017_ci'
        """
        algo, res = _run(self, algorithm, budget, seed)
        t = self._triples(res.best_x)
        t = {c: tuple(sorted(v)) for c, v in t.items()}
        total = math.fsum(_graded_mean(*v) for v in t.values())
        scaled = {c: tuple(p / total for p in v) for c, v in t.items()}
        k = self.deviation([p for c in self.criteria for p in scaled[c]])
        fuzzy = {c: TFN(*scaled[c]) for c in self.criteria}
        weights = {c: _graded_mean(*scaled[c]) for c in self.criteria}
        ci = fuzzy_bwm_consistency_index(self.best_to_others[self.worst])
        return BWMResult(
            method="fuzzy_bwm",
            criteria=tuple(self.criteria),
            weights=weights,
            ranking=rank_order(weights, descending=True),
            xi=k,
            consistency_ratio=k / ci,
            best=self.best,
            worst=self.worst,
            operators={"model": "fuzzy_nonlinear", "consistency": "guo_zhao2017_ci",
                       "defuzzification": "graded_mean", "algorithm": type(algo).__name__},
            fuzzy_weights=fuzzy,
            extras={"evals_used": res.evals_used, "best_f": res.best_f},
        )


# ------------------------------------------------------------------------- ANFIS


class ANFISProblem(Problem):
    """Training RMSE of an [`ANFIS`][bulanik.anfis.ANFIS] as a sezgi problem.

    The space is `Float(0, 1, D)` over all $D$ parameters;
    [`decode`][bulanik.sezgi.ANFISProblem.decode] maps each $z$ affinely to its
    parameter's bounds (span = hi - lo of its input): gaussian $c$ in
    [lo, hi], $\\sigma$ in [span/100, span]; bell $a$ in [span/100, span], $b$
    in [0.5, 5], $c$ in [lo, hi]; consequents in [-coef_bound, coef_bound].
    `evaluate` sets the decoded parameters on `anfis` (this instance is the
    working copy and is **mutated**) and returns the RMSE, or `inf` when no rule
    fires or the output overflows. The ANFIS needs its ranges (pass `ranges=`
    when building it; [`train_anfis`][bulanik.sezgi.train_anfis] infers them
    from the data when it has none).

    Attributes:
        anfis: The working ANFIS whose parameters `evaluate` overwrites.
        coef_bound: Bound of the consequent coefficients.

    Args:
        anfis: An [`ANFIS`][bulanik.anfis.ANFIS] with input ranges.
        X: Training inputs, one row per sample.
        y: Training targets.
        coef_bound: Consequent coefficients are searched in
            `[-coef_bound, coef_bound]` (positive).

    Raises:
        TypeError: If `anfis` is not an ANFIS.
        RuntimeError: If the ANFIS has no input ranges.
        ValueError: If `coef_bound` is not positive or `X`, `y` are invalid.

    Examples:
        >>> from bulanik.anfis import ANFIS
        >>> from bulanik.sezgi import ANFISProblem
        >>> X = [[i / 10] for i in range(11)]
        >>> y = [2 * x[0] + 1 for x in X]
        >>> problem = ANFISProblem(ANFIS(1, 2, ranges=[(0, 1)]), X, y)
        >>> problem.dim
        8
        >>> 0.0 <= problem.evaluate([0.5] * problem.dim) < 10
        True
    """

    def __init__(
        self,
        anfis: ANFIS,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        *,
        coef_bound: float = 10.0,
    ) -> None:
        if not isinstance(anfis, ANFIS):
            raise TypeError(f"anfis must be an ANFIS, got {type(anfis).__name__}")
        ranges = anfis.ranges
        if ranges is None:
            raise RuntimeError("ANFISProblem needs an ANFIS with input ranges: build it with ranges=...")
        cb = finite(coef_bound, "coef_bound")
        if cb <= 0.0:
            raise ValueError(f"coef_bound must be > 0, got {cb}")
        self.anfis = anfis
        self.coef_bound = cb
        self._rows, self._ys = anfis._check_xy(X, y)
        self._bounds: list[tuple[float, float]] = []
        for kind, i, _ in anfis.param_layout():
            if kind in ("p", "r"):
                self._bounds.append((-cb, cb))
                continue
            lo, hi = ranges[i]
            span = hi - lo
            if kind == "c":
                self._bounds.append((lo, hi))
            elif kind == "b":
                self._bounds.append((0.5, 5.0))
            elif kind in ("sigma", "a"):
                self._bounds.append((span / 100.0, span))
            else:  # pragma: no cover - param_layout only yields the kinds above
                raise ValueError(f"unknown parameter kind {kind!r} in the ANFIS layout")

    @property
    def dim(self) -> int:
        """Number of parameters $D$ (the length of `z` in `decode`)."""
        return len(self._bounds)

    def space(self) -> Any:
        return Float(0.0, 1.0, self.dim)

    def decode(self, z: Sequence[float]) -> list[float]:
        """Map `z` in $[0, 1]^D$ to the ANFIS parameter vector.

        Args:
            z: $D$ numbers in [0, 1].

        Returns:
            The parameters in the order of
            [`param_layout`][bulanik.anfis.ANFIS.param_layout], each between its
            bounds.

        Raises:
            ValueError: If `z` has the wrong length or a component outside [0, 1].

        Examples:
            >>> from bulanik.anfis import ANFIS
            >>> from bulanik.sezgi import ANFISProblem
            >>> X = [[0.0], [1.0]]
            >>> p = ANFISProblem(ANFIS(1, 1, ranges=[(0, 1)]), X, [0.0, 1.0], coef_bound=2)
            >>> p.decode([0.0] * p.dim)
            [0.0, 0.01, -2.0, -2.0]
            >>> p.decode([1.0] * p.dim)
            [1.0, 1.0, 2.0, 2.0]
        """
        vals = _vector(z, self.dim, positive=False)
        if any(not 0.0 <= v <= 1.0 for v in vals):
            raise ValueError("every z must lie in [0, 1]")
        # lo (1 - v) + hi v, not lo + v (hi - lo): hi - lo overflows for coef_bound > 8.9e307
        return [lo * (1.0 - v) + hi * v for v, (lo, hi) in zip(vals, self._bounds)]

    def evaluate(self, z: Sequence[float]) -> float:
        self.anfis.set_params(self.decode(z))
        return self.anfis._rmse_or_inf(self._rows, self._ys)


@dataclass(frozen=True)
class ANFISTrainResult:
    """Outcome of [`train_anfis`][bulanik.sezgi.train_anfis].

    Attributes:
        anfis: The trained copy of the ANFIS.
        rmse: Its training root-mean-square error.
        evals_used: Number of objective evaluations sezgi used.
    """

    anfis: ANFIS
    rmse: float
    evals_used: int


def train_anfis(
    anfis: ANFIS,
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    algorithm: Any = None,
    budget: int = 20000,
    seed: int = 0,
    coef_bound: float = 10.0,
) -> ANFISTrainResult:
    """Train a deep copy of `anfis` with sezgi; `anfis` itself is untouched.

    An ANFIS built without `ranges` gets them from the min and max of each
    column of `X` on the copy, as
    [`ANFIS.fit`][bulanik.anfis.ANFIS.fit] does (a constant column raises
    `ValueError`). Use this derivative-free trainer instead of `fit` when you
    want a global search over all membership and consequent parameters.

    Args:
        anfis: The ANFIS whose structure is trained (not modified).
        X: Training inputs, one row per sample.
        y: Training targets.
        algorithm: A sezgi algorithm instance; `None` selects
            `DifferentialEvolution(pop_size=40, variant="jde")`.
        budget: Number of objective evaluations (integer, at least 1).
        seed: Random seed, making the run reproducible.
        coef_bound: Consequent coefficients are searched in
            `[-coef_bound, coef_bound]` (positive).

    Returns:
        An [`ANFISTrainResult`][bulanik.sezgi.ANFISTrainResult].

    Raises:
        TypeError: If `anfis` is not an ANFIS.
        ValueError: If the data or `budget` or `coef_bound` is invalid, or a
            column of `X` is constant and ranges must be inferred.
        RuntimeError: If no finite RMSE is found.

    Examples:
        >>> from bulanik.anfis import ANFIS
        >>> from bulanik.sezgi import train_anfis
        >>> X = [[i / 10] for i in range(11)]
        >>> y = [2 * x[0] + 1 for x in X]
        >>> model = ANFIS(1, 2)
        >>> res = train_anfis(model, X, y, budget=2000, seed=0)
        >>> res.rmse < 0.1, res.evals_used, model.ranges is None
        (True, 2000, True)
    """
    if not isinstance(anfis, ANFIS):
        raise TypeError(f"anfis must be an ANFIS, got {type(anfis).__name__}")
    work = copy.deepcopy(anfis)
    work._infer_ranges(work._check_xy(X, y)[0])
    problem = ANFISProblem(work, X, y, coef_bound=coef_bound)
    _, res = _run(problem, algorithm, budget, seed)
    err = problem.evaluate(res.best_x)
    if not math.isfinite(err):
        raise RuntimeError("sezgi found no parameters with a finite RMSE; widen ranges or coef_bound")
    return ANFISTrainResult(anfis=problem.anfis, rmse=err, evals_used=res.evals_used)

