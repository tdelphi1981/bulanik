# LumiX bridge

[`bulanik.lumix`][bulanik.lumix] connects bulanik to
[LumiX](https://pypi.org/project/lumix-opt/), a Python modelling layer for linear and
integer programming. It adds three things: the **linear best-worst method**, **Zimmermann's
fuzzy goal programming** on top of your own LumiX model, and a converter that hands MCDM
scores to LumiX as plain data.

## Install

```bash
pip install "bulanik[lumix]"
```

The `lumix` extra installs `lumix-opt` and OR-Tools, the default solver. From a checkout of
the repository use `pip install -e ".[lumix]"` (or `uv sync`, which includes them).
Importing `bulanik.lumix` without the extra raises an `ImportError` that names it. Every
solving function takes `solver="ortools"`, the name of any LumiX solver.

## Linear best-worst method

[`bwm_linear`][bulanik.lumix.bwm_linear] implements Rezaei's (2016) **linear** BWM. With the
best criterion $B$, the worst $W$ and the comparison vectors $a_{Bj}$ (`best_to_others`) and
$a_{jW}$ (`others_to_worst`), it solves the linear program

$$
\min \xi \quad \text{s.t.} \quad |w_B - a_{Bj} w_j| \le \xi\ (j \ne B),\;\;
|w_j - a_{jW} w_W| \le \xi\ (j \ne W),\;\; \textstyle\sum_j w_j = 1,\; w \ge 0.
$$

Each absolute value is modelled by two rows. The optimum $\xi^L$ is 0 when the comparisons
are perfectly consistent and grows as they conflict. The best and worst criteria are
inferred as the unique criterion with comparison 1 in each vector, or given with `best=` and
`worst=` (needed when several criteria tie at 1). The comparison of best over worst must
agree in both vectors, $a_{BW}$ = `best_to_others[worst]` = `others_to_worst[best]`.

The result is a [`BWMResult`][bulanik.mcdm.results.BWMResult]; the LumiX model and solution
are kept in `extras["model"]` and `extras["solution"]`, and a
[ranking method](ranking.md) accepts it directly as `weights`.

```python exec="on" source="above" result="text"
from bulanik.lumix import bwm_linear

consistent = bwm_linear({"price": 1, "quality": 2, "speed": 8},
                        {"price": 8, "quality": 4, "speed": 1})
print({k: round(v, 3) for k, v in consistent.weights.items()})
print(consistent.best, consistent.worst, round(consistent.xi, 4))
print(consistent.operators)

# a_Bb x a_bW = 2 x 3 = 6 but a_BW = 8: a small conflict
r = bwm_linear({"a": 1, "b": 2, "c": 8}, {"a": 8, "b": 3, "c": 1})
print(r.ranking, {k: round(v, 3) for k, v in r.weights.items()})
print(round(r.xi, 4), r.consistency_ratio == r.xi, sorted(r.extras))
```

!!! warning "`consistency_ratio` is not the same number in every bridge"
    `bwm_linear` stores $\xi^L$ **itself** in `consistency_ratio` (it equals `xi`, and
    `operators["consistency"]` is `"xi_L"`). It is on the scale of the comparisons: 0 for
    perfect consistency, then a few tenths to a few units, with **no fixed upper bound**. The
    sezgi bridge stores a *ratio* under the same field name:
    [`BWMProblem`][bulanik.sezgi.BWMProblem] gives $\xi / CI(a_{BW})$ and
    [`FuzzyBWMProblem`][bulanik.sezgi.FuzzyBWMProblem] gives $k / CI$, where $CI$ is a
    consistency index of the best-over-worst comparison (see
    [Weighting](weighting.md#best-worst-method)). Compare a linear $\xi^L$ only with other
    linear results, and read `operators["consistency"]` before comparing anything. The
    [sezgi bridge page](bridge-sezgi.md#consistency_ratio-is-not-one-quantity) has the
    other side of this box.

## Zimmermann's fuzzy goal programming

Sometimes a linear program has goals that are *soft*: "profit of about 20 or more, use of
about 4 or less". Zimmermann's (1978) **max-min** approach gives each goal a membership
function and maximises the smallest membership $\lambda$ over all goals.

A [`FuzzyGoal`][bulanik.lumix.FuzzyGoal] describes a goal on a linear expression $z$ with an
**aspiration** $g$ and a **tolerance** $t > 0$:

| `sense` | Reads | Membership of $z$ |
|---|---|---|
| `"ge"` | at least about $g$ | 0 at $g - t$, rising linearly to 1 at $g$ |
| `"le"` | at most about $g$ | 1 at $g$, falling linearly to 0 at $g + t$ |
| `"eq"` | about $g$ | $1 - |z - g|/t$: triangular, 0 beyond $g \pm t$ |

[`zimmermann(model, goals)`][bulanik.lumix.zimmermann] adds the rows to your model **in
place** and returns it. Each goal gets a free variable `goal_<name>`, a definition row
`goal_<name>_def` ($z$ minus that variable is 0) and the membership row (`_ge` and/or
`_le`):

$$
z - t\lambda \ge g - t \quad (\texttt{ge}), \qquad z + t\lambda \le g + t \quad (\texttt{le}),
$$

plus a variable $0 \le \lambda \le 1$ (`fuzzy_lambda`, renamable) and the objective
$\max \lambda$, which replaces any earlier objective. Your own variables and constraints
stay, so you can keep extending the model before solving.
[`solve_zimmermann`][bulanik.lumix.solve_zimmermann] does that call and solves, returning a
[`ZimmermannResult`][bulanik.lumix.ZimmermannResult] with `lambda_`, the `goal_values`, the
`memberships` at those values and the LumiX `solution` for your decision variables.

```python exec="on" source="above" result="text"
from lumix import LXConstraint, LXModel, LXVariable
from lumix.core import LXLinearExpression
from bulanik.lumix import FuzzyGoal, solve_zimmermann, zimmermann

def build():
    x = LXVariable[None, float]("x").continuous().bounds(lower=0, upper=10)
    y = LXVariable[None, float]("y").continuous().bounds(lower=0, upper=10)
    use = LXLinearExpression().add_term(x, coeff=1).add_term(y, coeff=1)
    model = (LXModel("plan").add_variable(x).add_variable(y)
             .add_constraint(LXConstraint("cap").expression(use).le().rhs(8)))   # at most 8 units
    profit = LXLinearExpression().add_term(x, coeff=3).add_term(y, coeff=2)
    goals = [FuzzyGoal("profit", profit, aspiration=20, tolerance=8, sense="ge"),
             FuzzyGoal("use", use, aspiration=4, tolerance=4, sense="le")]
    return model, goals

goal = build()[1][0]
print([goal.membership(z) for z in (12, 16, 20, 25)])      # 0 at g - t, 0.5, 1 at g

model, goals = build()
zimmermann(model, goals)                                    # inspect what was added
print([v.name for v in model.variables])
print([c.name for c in model.constraints])

model, goals = build()
res = solve_zimmermann(model, goals)
print("lambda:", round(res.lambda_, 3))
print({k: round(v, 2) for k, v in res.goal_values.items()})
print({k: round(v, 2) for k, v in res.memberships.items()})
print("plan:", {n: round(res.solution.variables[n], 2) for n in ("x", "y")})
```

Here $\lambda = 0.6$: no plan does better than 60% satisfaction of the *worse* goal, and
the optimum balances the two goals at exactly that level. Because the model is changed in
place, build a fresh one if you want to solve a second time.

## MCDM scores as LumiX data

[`scores_as_data`][bulanik.lumix.scores_as_data] converts a weighting, ranking or BWM result
(or a plain `{name: score}` mapping) into a list of
[`ScoreRecord`][bulanik.lumix.ScoreRecord] with `key`, `value` (the raw weight or score) and
`rank` (1 is best). The records are what `LXVariable(...).from_data(records)` expects, so an
MCDM score can drive an optimisation model: score the suppliers, then choose the best
*combination* under a budget.

```python exec="on" source="above" result="text"
from lumix import LXConstraint, LXModel, LXOptimizer, LXVariable
from lumix.core import LXLinearExpression
from bulanik.lumix import scores_as_data
from bulanik.mcdm import swara, topsis

criteria = ["price", "quality", "delivery"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
data = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]
ranking = topsis(data, swara([0.2, 0.5], criteria), ["cost", "benefit", "benefit"],
                 alternatives=suppliers)
records = scores_as_data(ranking)
for rec in records:
    print(rec.key, round(rec.value, 4), rec.rank)

setup_cost = {"Atlas": 50, "Borealis": 25, "Cedar": 45, "Delta": 30}
pick = LXVariable[object, int]("pick").binary().indexed_by(lambda r: r.key).from_data(records)
model = (
    LXModel("supplier_pair")
    .add_variable(pick)
    .add_constraint(LXConstraint("budget")
                    .expression(LXLinearExpression().add_term(pick, lambda r: setup_cost[r.key]))
                    .le().rhs(75))
    .add_constraint(LXConstraint("at_most_two")
                    .expression(LXLinearExpression().add_term(pick, 1.0)).le().rhs(2))
    .maximize(LXLinearExpression().add_term(pick, lambda r: r.value))
)
solution = LXOptimizer().use_solver("ortools").solve(model)
print(solution.is_optimal(), sorted(k for k, v in solution.variables["pick"].items() if v > 0.5))
```

The two best suppliers (Cedar and Atlas) cost 95 together, over the budget of 75, so the
optimum is Cedar with Delta: the ranking alone would not have found it.

Records keep the order of the result's scores and take `rank` from `result.ranking`. A
plain mapping is ranked by descending value (ties keep input order). **The `value` is the
method's raw score**: for VIKOR, where lower is better, a model that maximises `value`
would prefer the *worst* alternative, so use `rank` or negate it.

## Pitfalls

**Two different `consistency_ratio`s.** See the box above: $\xi^L$ here, a ratio in the
sezgi bridge.

**`zimmermann` mutates your model.** It adds variables and rows and *replaces the
objective*. Variable names `fuzzy_lambda` and `goal_<name>` must be free, goal names unique,
and a goal expression must be an `LXLinearExpression`.

**A goal that cannot reach membership 0 is infeasible.** If the hard constraints keep a
goal's expression beyond $g - t$ (or $g + t$), `solve_zimmermann` raises `RuntimeError` with
the solver status. Widen the tolerance or relax the constraint.

**Solver failures are `RuntimeError`s**, never silent: `bwm_linear` and
`solve_zimmermann` raise if the solver finds no optimum.

**BWM inputs.** Comparisons are finite numbers at least 1; a missing best or worst (or an
ambiguous one) is a `ValueError` that tells you to pass `best=` or `worst=`.

```python exec="on" source="above" result="text"
from lumix import LXConstraint, LXModel, LXVariable
from lumix.core import LXLinearExpression
from bulanik.lumix import FuzzyGoal, bwm_linear, scores_as_data, solve_zimmermann

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: bwm_linear({"a": 1, "b": 2, "c": 8}, {"a": 8, "b": 4, "c": 1}, best="b"))
show(lambda: bwm_linear({"a": 1, "b": 1, "c": 8}, {"a": 8, "b": 8, "c": 1}))            # best is ambiguous
show(lambda: bwm_linear({"a": 1, "b": 2, "c": 8}, {"a": 8, "b": 4, "c": 2}))            # nobody has 1
show(lambda: bwm_linear({"a": 1, "b": 0.5, "c": 8}, {"a": 8, "b": 4, "c": 1}))          # below 1
show(lambda: bwm_linear({"a": 1, "b": 2, "c": 8}, {"a": 8, "b": 4, "c": 1}, solver="nosuchsolver"))
show(lambda: scores_as_data([1, 2, 3]))
show(lambda: scores_as_data({"a": float("nan")}))

x = LXVariable[None, float]("x").continuous().bounds(lower=0, upper=10)
z = LXLinearExpression().add_term(x, coeff=1)
show(lambda: FuzzyGoal("g", z, aspiration=5, tolerance=0, sense="ge"))                  # tolerance > 0
show(lambda: FuzzyGoal("g", z, aspiration=5, tolerance=1, sense="about"))
show(lambda: FuzzyGoal("g", "x", aspiration=5, tolerance=1, sense="ge"))

model = LXModel("m").add_variable(x).add_constraint(
    LXConstraint("low").expression(z).le().rhs(2))
far = FuzzyGoal("far", z, aspiration=20, tolerance=4, sense="ge")                       # needs x >= 16
show(lambda: solve_zimmermann(model, [far]))
```

## See also

- [Weighting](weighting.md#best-worst-method) for the BWM input and its consistency indices,
  [Ranking](ranking.md) to use a `BWMResult` as weights, and the
  [sezgi bridge](bridge-sezgi.md) for the nonlinear and fuzzy BWM.
- API: [`bulanik.lumix`][bulanik.lumix].
- Examples: [06 Zimmermann](../examples/06-zimmermann-lumix.md),
  [07 BWM with LumiX](../examples/07-bwm-lumix.md),
  [09 weights to LumiX](../examples/09-weights-to-lumix.md).
- Rezaei (2016), "Best-worst multi-criteria decision-making method: some properties and a
  linear model", *Omega* 64; Zimmermann (1978), "Fuzzy programming and linear programming
  with several objective functions", *Fuzzy Sets and Systems* 1(1).
