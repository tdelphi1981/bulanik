r"""LumiX bridge: linear BWM, Zimmermann fuzzy goals and scores as LumiX data.

Needs the `lumix` extra (`pip install "bulanik[lumix]"`, which installs
`lumix-opt`; a solver such as OR-Tools is used by default, so install it too
if your environment does not already have it).

- [`bwm_linear`][bulanik.lumix.bwm_linear] - Rezaei's (2016) linear
  best-worst method as a linear program.
- [`zimmermann`][bulanik.lumix.zimmermann] /
  [`solve_zimmermann`][bulanik.lumix.solve_zimmermann] - Zimmermann's (1978)
  max-min fuzzy linear programming on top of your own `lumix.LXModel`, with the
  goals described by [`FuzzyGoal`][bulanik.lumix.FuzzyGoal] and the outcome in a
  [`ZimmermannResult`][bulanik.lumix.ZimmermannResult].
- [`scores_as_data`][bulanik.lumix.scores_as_data] - MCDM results as plain
  records ([`ScoreRecord`][bulanik.lumix.ScoreRecord]), ready for
  `LXVariable(...).from_data(records)`.

!!! warning "The two `consistency_ratio` fields differ"
    The `consistency_ratio` of the [`BWMResult`][bulanik.mcdm.results.BWMResult]
    returned by [`bwm_linear`][bulanik.lumix.bwm_linear] is the optimum
    $\xi^L$ of the *linear* model itself (the same number as `xi`). The
    nonlinear and fuzzy models in [`bulanik.sezgi`][bulanik.sezgi] store the
    ratio $\xi / CI(a_{BW})$ under the same field name. $\xi^L$ is on the
    scale of the comparisons (0 to a few units) and has no fixed upper bound,
    so compare it only with other linear-model results; `operators["consistency"]`
    is `"xi_L"` here and `"rezaei2015_ci"` / `"guo_zhao2017_ci"` in the sezgi
    bridge.

Examples:
    >>> from bulanik.lumix import bwm_linear
    >>> r = bwm_linear({"price": 1, "quality": 2, "speed": 8},
    ...                {"price": 8, "quality": 4, "speed": 1})
    >>> {k: round(v, 3) for k, v in r.weights.items()}
    {'price': 0.615, 'quality': 0.308, 'speed': 0.077}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    from lumix import LXConstraint, LXModel, LXOptimizer, LXVariable
    from lumix.core import LXLinearExpression
except ImportError as exc:  # pragma: no cover - exercised with lumix hidden
    raise ImportError(
        "bulanik.lumix needs the 'lumix' extra: install it with "
        'pip install "bulanik[lumix]"'
    ) from exc

from ._validate import choose, finite
from .mcdm._common import rank_order
from .mcdm.bwm import resolve_best_worst
from .mcdm.results import BWMResult, RankingResult, WeightingResult
from .sets import linear_decreasing, linear_increasing, triangular

__all__ = [
    "bwm_linear",
    "FuzzyGoal",
    "zimmermann",
    "ZimmermannResult",
    "solve_zimmermann",
    "ScoreRecord",
    "scores_as_data",
]

_SENSES = ("ge", "le", "eq")


def _solve(model: Any, solver: str, what: str) -> Any:
    solution = LXOptimizer().use_solver(solver).solve(model)
    if not solution.is_optimal():
        raise RuntimeError(f"{what}: solver did not find an optimum (status {solution.status})")
    return solution


# --------------------------------------------------------------------------- BWM


def bwm_linear(
    best_to_others: Mapping[str, float],
    others_to_worst: Mapping[str, float],
    criteria: Sequence[str] | None = None,
    *,
    best: str | None = None,
    worst: str | None = None,
    solver: str = "ortools",
) -> BWMResult:
    r"""Linear best-worst method (Rezaei 2016, Omega 64:126-130).

    With best criterion $B$, worst criterion $W$ and comparison vectors
    $a_{Bj}$ (`best_to_others`) and $a_{jW}$ (`others_to_worst`), solve the
    linear program

    $$\min \xi \quad \text{s.t.} \quad |w_B - a_{Bj} w_j| \le \xi\ (j \ne B),\;
    |w_j - a_{jW} w_W| \le \xi\ (j \ne W),\; \textstyle\sum_j w_j = 1,\; w \ge 0.$$

    The optimal $\xi^L$ is the consistency indicator: 0 for perfectly
    consistent comparisons, larger when they conflict. Each absolute value is
    modelled by two rows. The LumiX model and solution are kept in
    `extras["model"]` and `extras["solution"]`.

    !!! warning "`consistency_ratio` here is $\xi^L$ itself"
        The returned `consistency_ratio` equals `xi`, the optimum $\xi^L$ of
        the linear model. It is **not** the ratio $\xi / CI(a_{BW})$ that
        [`BWMProblem`][bulanik.sezgi.BWMProblem] and
        [`FuzzyBWMProblem`][bulanik.sezgi.FuzzyBWMProblem] in `bulanik.sezgi`
        report under the same field name (there `None` when $CI = 0$ and the
        comparisons are inconsistent). Compare $\xi^L$ only with other
        linear-model results; `operators["consistency"]` is `"xi_L"`.

    Args:
        best_to_others: Comparison of the best criterion over each criterion
            (finite, at least 1; the best criterion has 1 unless `best` is
            given).
        others_to_worst: Comparison of each criterion over the worst criterion.
        criteria: Criterion names, fixing the order. Defaults to the order of
            `best_to_others`.
        best: The best criterion. Inferred as the unique one with comparison 1
            when `None`.
        worst: The worst criterion, inferred likewise from `others_to_worst`.
        solver: LumiX solver name, `"ortools"` by default.

    Returns:
        A [`BWMResult`][bulanik.mcdm.results.BWMResult] with
        `operators == {"model": "linear", "consistency": "xi_L"}`.

    Raises:
        TypeError: If a comparison value is a `bool` or `str` instead of a number.
        ValueError: For invalid comparison vectors (see
            [`resolve_best_worst`][bulanik.mcdm.bwm.resolve_best_worst]).
        RuntimeError: If the solver finds no optimum or returns a clearly
            negative weight.

    Examples:
        >>> from bulanik.lumix import bwm_linear
        >>> r = bwm_linear({"a": 1, "b": 2, "c": 8}, {"a": 8, "b": 3, "c": 1})
        >>> r.best, r.worst, r.ranking
        ('a', 'c', ('a', 'b', 'c'))
        >>> round(r.xi, 4), r.consistency_ratio == r.xi
        (0.0417, True)

    References:
        Rezaei (2016), Omega 64, 126-130.
    """
    crit, b, w_name = resolve_best_worst(best_to_others, others_to_worst, criteria, best, worst)
    bo = {c: finite(best_to_others[c], f"best_to_others[{c}]") for c in crit}
    ow = {c: finite(others_to_worst[c], f"others_to_worst[{c}]") for c in crit}

    w = LXVariable[str, float]("w").continuous().bounds(lower=0.0).indexed_by(lambda c: c).from_data(crit)
    xi = LXVariable[None, float]("xi").continuous().bounds(lower=0.0)
    model = LXModel("bwm_linear").add_variable(w).add_variable(xi)

    def add_abs_rows(prefix: str, j: str, coeff: Any) -> None:
        # |sum_c coeff(c) w_c| <= xi as two rows: ±(...) - xi <= 0
        for tag, s in (("pos", 1.0), ("neg", -1.0)):
            expr = (LXLinearExpression()
                    .add_term(w, coeff=lambda c, s=s: s * coeff(c))
                    .add_term(xi, coeff=-1.0))
            model.add_constraint(LXConstraint(f"{prefix}_{j}_{tag}").expression(expr).le().rhs(0.0))

    for j in crit:
        if j != b:
            add_abs_rows("bo", j, lambda c, j=j: float(c == b) - bo[j] * float(c == j))
    for j in crit:
        if j != w_name:
            add_abs_rows("ow", j, lambda c, j=j: float(c == j) - ow[j] * float(c == w_name))
    model.add_constraint(
        LXConstraint("sum_w").expression(LXLinearExpression().add_term(w, coeff=1.0)).eq().rhs(1.0)
    )
    model.minimize(LXLinearExpression().add_term(xi, coeff=1.0))

    solution = _solve(model, solver, "bwm_linear")
    raw = solution.variables["w"]
    weights = {c: _nonneg(raw[c], f"weight of {c}") for c in crit}
    xi_val = _nonneg(solution.variables["xi"], "xi")
    return BWMResult(
        method="bwm_linear",
        criteria=tuple(crit),
        weights=weights,
        ranking=rank_order(weights, descending=True),
        xi=xi_val,
        consistency_ratio=xi_val,
        best=b,
        worst=w_name,
        operators={"model": "linear", "consistency": "xi_L"},
        extras={"model": model, "solution": solution},
    )


def _nonneg(x: float, what: str) -> float:
    """Clamp solver noise of at most 1e-9 below zero; larger negatives are errors."""
    x = finite(x, what)
    if x < 0.0:
        if x < -1e-9:
            raise RuntimeError(f"{what} is negative ({x}); the solve is not trustworthy")
        return 0.0
    return x


# --------------------------------------------------------------------- Zimmermann


@dataclass(frozen=True)
class FuzzyGoal:
    r"""A fuzzy goal on a linear expression $z$ (Zimmermann 1978).

    With aspiration $g$ and tolerance $t > 0$ the membership of the goal is

    - `"ge"` ("at least about $g$"): rises linearly from 0 at $g - t$ to 1 at $g$;
    - `"le"` ("at most about $g$"): falls linearly from 1 at $g$ to 0 at $g + t$;
    - `"eq"` ("about $g$"): $\mu = 1 - |z - g| / t$ (triangular, 0 beyond $g \pm t$).

    Attributes:
        name: Goal name, used in the added variable and row names.
        expression: An `LXLinearExpression` in the model's variables.
        aspiration: The aspiration level $g$ (finite).
        tolerance: The tolerance $t$ (finite, positive).
        sense: `"ge"`, `"le"` or `"eq"`.

    Raises:
        TypeError: If `expression` is not an `LXLinearExpression`.
        ValueError: If `sense` is unknown, `aspiration` or `tolerance` is not
            finite, or `tolerance` is not positive.

    Examples:
        >>> from lumix.core import LXLinearExpression
        >>> from bulanik.lumix import FuzzyGoal
        >>> goal = FuzzyGoal("profit", LXLinearExpression(), aspiration=20,
        ...                  tolerance=8, sense="ge")
        >>> goal.membership(12), goal.membership(16), goal.membership(25)
        (0.0, 0.5, 1.0)
    """

    name: str
    expression: Any
    aspiration: float
    tolerance: float
    sense: str

    def __post_init__(self) -> None:
        if not isinstance(self.expression, LXLinearExpression):
            raise TypeError(
                f"FuzzyGoal '{self.name}': expression must be an LXLinearExpression, "
                f"got {type(self.expression).__name__}"
            )
        choose(self.sense, _SENSES, "sense", "FuzzyGoal")
        g = finite(self.aspiration, "aspiration")
        t = finite(self.tolerance, "tolerance")
        if t <= 0.0:
            raise ValueError(f"FuzzyGoal '{self.name}': tolerance must be > 0, got {t}")
        object.__setattr__(self, "aspiration", g)
        object.__setattr__(self, "tolerance", t)

    def membership(self, z: float) -> float:
        """Membership degree of the value `z` in the goal.

        Args:
            z: A value of the goal expression.

        Returns:
            The degree in [0, 1], using the membership functions of
            [`bulanik.sets`][bulanik.sets].
        """
        g, t = self.aspiration, self.tolerance
        if self.sense == "ge":
            return linear_increasing(g - t, g)(z)
        if self.sense == "le":
            return linear_decreasing(g, g + t)(z)
        return triangular(g - t, g, g + t)(z)


def zimmermann(
    model: Any, goals: Sequence[FuzzyGoal], *, lambda_name: str = "fuzzy_lambda"
) -> Any:
    r"""Add Zimmermann's max-min fuzzy goal rows to `model` and return it.

    Zimmermann's (1978) approach maximises the smallest goal membership
    $\lambda$. Each goal gets a free variable `goal_{name}` with the definition
    row `goal_{name}_def` (expression $- z = 0$) and the membership rows

    $$z - t\lambda \ge g - t \quad (\texttt{ge}), \qquad
    z + t\lambda \le g + t \quad (\texttt{le}),$$

    in rows `goal_{name}_ge` and `goal_{name}_le`; `"eq"` adds both. The
    variable $0 \le \lambda \le 1$ is added and the objective becomes
    $\max \lambda$ (replacing any previous objective). Your variables and
    constraints stay, so the model can still be extended before solving. The
    goal expressions are not mutated.

    Args:
        model: A `lumix.LXModel` holding the hard constraints.
        goals: At least one [`FuzzyGoal`][bulanik.lumix.FuzzyGoal] with unique
            names.
        lambda_name: Name of the added $\lambda$ variable.

    Returns:
        The same `model`, modified in place.

    Raises:
        ValueError: If there are no goals, goal names repeat, or an added
            variable name collides with a model variable.
        TypeError: If a goal is not a [`FuzzyGoal`][bulanik.lumix.FuzzyGoal].

    Examples:
        >>> from lumix import LXConstraint, LXModel, LXVariable
        >>> from lumix.core import LXLinearExpression
        >>> from bulanik.lumix import FuzzyGoal, zimmermann
        >>> x = LXVariable[None, float]("x").continuous().bounds(lower=0, upper=10)
        >>> model = LXModel("demo").add_variable(x)
        >>> goal = FuzzyGoal("big", LXLinearExpression().add_term(x, coeff=1.0),
        ...                  aspiration=8, tolerance=4, sense="ge")
        >>> model = zimmermann(model, [goal])
        >>> sorted(v.name for v in model.variables)
        ['fuzzy_lambda', 'goal_big', 'x']

    References:
        Zimmermann (1978), Fuzzy Sets and Systems 1(1), 45-55.
    """
    goals = list(goals)
    if not goals:
        raise ValueError("zimmermann needs at least one goal")
    for goal in goals:
        if not isinstance(goal, FuzzyGoal):
            raise TypeError(f"goals must be FuzzyGoal, got {type(goal).__name__}")
    names = [g.name for g in goals]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate goal names: {', '.join(dupes)}")
    new_vars = [lambda_name] + [f"goal_{n}" for n in names]
    existing = {v.name for v in model.variables}
    clashes = [n for n in new_vars if n in existing]
    if lambda_name in new_vars[1:]:
        clashes.append(lambda_name)
    if clashes:
        raise ValueError(
            f"zimmermann variable names collide with the model or each other: "
            f"{', '.join(clashes)}; rename the goal or pass another lambda_name="
        )

    lam = LXVariable[None, float](lambda_name).continuous().bounds(lower=0.0, upper=1.0)
    model.add_variable(lam)
    for goal in goals:
        g, t = goal.aspiration, goal.tolerance
        z = LXVariable[None, float](f"goal_{goal.name}").continuous()  # no bounds: free
        model.add_variable(z)
        definition = goal.expression.copy().add_term(z, coeff=-1.0)
        model.add_constraint(
            LXConstraint(f"goal_{goal.name}_def").expression(definition).eq().rhs(0.0)
        )
        if goal.sense in ("ge", "eq"):
            row = LXLinearExpression().add_term(z, coeff=1.0).add_term(lam, coeff=-t)
            model.add_constraint(LXConstraint(f"goal_{goal.name}_ge").expression(row).ge().rhs(g - t))
        if goal.sense in ("le", "eq"):
            row = LXLinearExpression().add_term(z, coeff=1.0).add_term(lam, coeff=t)
            model.add_constraint(LXConstraint(f"goal_{goal.name}_le").expression(row).le().rhs(g + t))
    model.maximize(LXLinearExpression().add_term(lam, coeff=1.0))
    return model


@dataclass(frozen=True)
class ZimmermannResult:
    r"""Outcome of [`solve_zimmermann`][bulanik.lumix.solve_zimmermann].

    Attributes:
        lambda_: The optimal $\lambda$, the smallest goal membership
            (clipped to at most 1).
        goal_values: The value $z$ of each goal expression, by goal name.
        memberships: The membership $\mu(z)$ of each goal at that value.
        solution: The LumiX `LXSolution`, for the decision variables.
    """

    lambda_: float
    goal_values: dict[str, float]
    memberships: dict[str, float]
    solution: Any


def solve_zimmermann(
    model: Any,
    goals: Sequence[FuzzyGoal],
    *,
    solver: str = "ortools",
    lambda_name: str = "fuzzy_lambda",
) -> ZimmermannResult:
    r"""Build the Zimmermann model on `model` and solve it.

    Calls [`zimmermann`][bulanik.lumix.zimmermann] (so `model` is modified in
    place) and optimises it, then reads the goal values and memberships.

    Args:
        model: A `lumix.LXModel` holding the hard constraints.
        goals: The [`FuzzyGoal`][bulanik.lumix.FuzzyGoal] list.
        solver: LumiX solver name, `"ortools"` by default.
        lambda_name: Name of the added $\lambda$ variable.

    Returns:
        A [`ZimmermannResult`][bulanik.lumix.ZimmermannResult].

    Raises:
        RuntimeError: With the solver status when no optimum is found, e.g.
            when a goal cannot even reach membership 0 under the hard
            constraints.
        ValueError: As for [`zimmermann`][bulanik.lumix.zimmermann].
        TypeError: As for [`zimmermann`][bulanik.lumix.zimmermann].

    Examples:
        Maximise the worst of two goals: profit at least about 20 (tolerance 8)
        and total use at most about 4 (tolerance 4), with at most 8 units.

        >>> from lumix import LXConstraint, LXModel, LXVariable
        >>> from lumix.core import LXLinearExpression
        >>> from bulanik.lumix import FuzzyGoal, solve_zimmermann
        >>> x = LXVariable[None, float]("x").continuous().bounds(lower=0, upper=10)
        >>> y = LXVariable[None, float]("y").continuous().bounds(lower=0, upper=10)
        >>> use = LXLinearExpression().add_term(x, coeff=1).add_term(y, coeff=1)
        >>> model = (LXModel("plan").add_variable(x).add_variable(y)
        ...          .add_constraint(LXConstraint("cap").expression(use).le().rhs(8)))
        >>> profit = LXLinearExpression().add_term(x, coeff=3).add_term(y, coeff=2)
        >>> goals = [FuzzyGoal("profit", profit, aspiration=20, tolerance=8, sense="ge"),
        ...          FuzzyGoal("use", use, aspiration=4, tolerance=4, sense="le")]
        >>> res = solve_zimmermann(model, goals)
        >>> round(res.lambda_, 3)
        0.6
        >>> {k: round(v, 2) for k, v in res.goal_values.items()}
        {'profit': 16.8, 'use': 5.6}
        >>> {k: round(v, 2) for k, v in res.memberships.items()}
        {'profit': 0.6, 'use': 0.6}

    References:
        Zimmermann (1978), Fuzzy Sets and Systems 1(1), 45-55.
    """
    goals = list(goals)
    zimmermann(model, goals, lambda_name=lambda_name)
    solution = _solve(model, solver, "zimmermann")
    values = {g.name: finite(solution.variables[f"goal_{g.name}"], f"goal_{g.name}") for g in goals}
    lam = min(1.0, _nonneg(solution.variables[lambda_name], lambda_name))
    return ZimmermannResult(
        lambda_=lam,
        goal_values=values,
        memberships={g.name: g.membership(values[g.name]) for g in goals},
        solution=solution,
    )


# ------------------------------------------------------------------ scores as data


@dataclass(frozen=True)
class ScoreRecord:
    """One scored item.

    Attributes:
        key: The criterion or alternative name.
        value: Its weight or score.
        rank: Its 1-based rank (1 is best).
    """

    key: str
    value: float
    rank: int


def scores_as_data(
    result: WeightingResult | RankingResult | BWMResult | Mapping[str, float],
) -> list[ScoreRecord]:
    """MCDM scores as records for `LXVariable(...).from_data(records)`.

    Records keep the order of the result's scores/weights; ranks are 1-based
    from `result.ranking`. A plain mapping is ranked by descending value
    (ties keep input order). `value` is the method's raw score, so for VIKOR,
    where lower is better, use `rank` or mind the direction.

    Args:
        result: A [`WeightingResult`][bulanik.mcdm.results.WeightingResult],
            [`RankingResult`][bulanik.mcdm.results.RankingResult],
            [`BWMResult`][bulanik.mcdm.results.BWMResult], or a mapping from
            name to a finite score.

    Returns:
        A list of [`ScoreRecord`][bulanik.lumix.ScoreRecord].

    Raises:
        TypeError: If `result` is none of the accepted kinds, or a score is a
            `bool` or `str` instead of a number.
        ValueError: If a score is not finite.

    Examples:
        >>> from bulanik.mcdm import ahp
        >>> from bulanik.lumix import scores_as_data
        >>> w = ahp([[1, 3], [1/3, 1]], ["cost", "quality"])
        >>> for rec in scores_as_data(w):
        ...     print(rec.key, round(rec.value, 2), rec.rank)
        cost 0.75 1
        quality 0.25 2
    """
    if isinstance(result, RankingResult):
        values, ranking = dict(result.scores), result.ranking
    elif isinstance(result, (WeightingResult, BWMResult)):
        values, ranking = dict(result.weights), result.ranking
    elif isinstance(result, Mapping):
        values = {k: finite(v, f"score of {k}") for k, v in result.items()}
        ranking = rank_order(values, descending=True)
    else:
        raise TypeError(
            "scores_as_data expects a WeightingResult, RankingResult, BWMResult or a "
            f"mapping of scores, got {type(result).__name__}"
        )
    rank = {k: i + 1 for i, k in enumerate(ranking)}
    return [ScoreRecord(key=k, value=finite(v, f"score of {k}"), rank=rank[k]) for k, v in values.items()]
