# sezgi bridge

[`bulanik.sezgi`][bulanik.sezgi] connects bulanik to **sezgi**, a library of metaheuristic
optimisers that minimise a function without derivatives. The bridge expresses three
bulanik problems as sezgi problems: the **nonlinear best-worst method**, the **fuzzy
best-worst method** and **ANFIS training**.

## Install

```bash
pip install "bulanik[sezgi]"
```

From a checkout use `pip install -e ".[sezgi]"` (or `uv sync`). Importing `bulanik.sezgi`
without the extra raises an `ImportError` that names it.

sezgi has no constraint API for a single objective, so every constraint is a **penalty**
inside the objective. The default algorithm everywhere is
`sezgi.DifferentialEvolution(pop_size=40, variant="jde")`; pass any other sezgi algorithm
instance as `algorithm=`. A run is **deterministic for a fixed `seed` and `budget`** (the
number of objective evaluations), which is what the examples on this page rely on.

## Nonlinear best-worst method

[`BWMProblem`][bulanik.sezgi.BWMProblem] is Rezaei's (2015) original, nonlinear BWM. With
best criterion $B$, worst $W$ and comparisons $a_{Bj}$, $a_{jW}$,

$$
\min \xi \quad \text{s.t.} \quad \left|\frac{w_B}{w_j} - a_{Bj}\right| \le \xi\ (j \ne B),\;\;
\left|\frac{w_j}{w_W} - a_{jW}\right| \le \xi\ (j \ne W),\;\; \textstyle\sum_j w_j = 1.
$$

`evaluate(x)` is the largest deviation plus `penalty * |sum(x) - 1|` (default penalty 10).
`deviation(x)` gives the objective *without* the penalty, which is handy to check a
hand-made weight vector. [`solve(algorithm=None, *, budget=20000, seed=0)`][bulanik.sezgi.BWMProblem.solve]
runs the search and returns a [`BWMResult`][bulanik.mcdm.results.BWMResult] with the weights
rescaled to sum to 1, `xi` recomputed at those weights, `extras["evals_used"]` and
`extras["best_f"]`. Best and worst are inferred as in the
[LumiX bridge](bridge-lumix.md#linear-best-worst-method), or pass `best=` and `worst=`.

```python exec="on" source="above" result="text"
from bulanik.sezgi import BWMProblem

problem = BWMProblem({"price": 1, "quality": 2, "speed": 8},
                     {"price": 8, "quality": 4, "speed": 1})
print(problem.best, problem.worst)
print(round(problem.deviation([8, 4, 1]), 6), "(perfectly consistent, any scale)")
print(round(problem.deviation([0.6, 0.3, 0.1]), 6))

r = problem.solve(budget=3000, seed=1)
print({k: round(v, 3) for k, v in r.weights.items()}, r.ranking)
print(r.method, r.operators)
print(r.xi < 1e-3, r.consistency_ratio < 1e-3, r.extras["evals_used"])
again = problem.solve(budget=3000, seed=1)
print(again.weights == r.weights)               # same seed and budget: same answer
```

## Fuzzy best-worst method

[`FuzzyBWMProblem`][bulanik.sezgi.FuzzyBWMProblem] is Guo and Zhao's (2017) version: the
comparisons are triangular numbers, for example the words of
[`FUZZY_BWM_TFN`][bulanik.scales.FUZZY_BWM_TFN] (EI, WI, FI, VI, AI for equally to
absolutely important), and the weights are triangular numbers
$\tilde w_j = (l_j, m_j, u_j)$. With the fuzzy ratio
$\tilde w_B / \tilde w_j = (l_B/u_j,\; m_B/m_j,\; u_B/l_j)$ each comparison gives three rows,
$|l_B - l_{Bj} u_j| \le k u_j$, $|m_B - m_{Bj} m_j| \le k m_j$ and
$|u_B - u_{Bj} l_j| \le k l_j$, the same for $\tilde w_j / \tilde w_W$, and the weights are
normalised so that the **graded means** $R(\tilde w) = (l + 4m + u)/6$ sum to 1 and ordered
$l \le m \le u$. The search space has $3n$ components.

`solve` returns the fuzzy weights in `fuzzy_weights`, their graded means as `weights`, and
the optimum $k$ in `xi`. Comparisons must be `TFN` with $l > 0$ and $m \ge 1$; the best and
worst are the criteria whose comparison is `TFN(1, 1, 1)`. The default budget is larger
(100 000) because the space is larger.

```python exec="on" source="above" result="text"
from bulanik.scales import FUZZY_BWM_TFN as F
from bulanik.sezgi import FuzzyBWMProblem

problem = FuzzyBWMProblem({"a": F["EI"], "b": F["WI"], "c": F["AI"]},
                          {"a": F["AI"], "b": F["FI"], "c": F["EI"]})
r = problem.solve(budget=20000, seed=1)
print(r.best, r.worst, r.ranking)
print({k: round(v, 2) for k, v in r.weights.items()}, round(sum(r.weights.values()), 6))
print(r.fuzzy_weights["a"])
print(r.method, r.operators["consistency"], r.operators["defuzzification"])
```

## `consistency_ratio` is not one quantity

All three BWM producers fill the `consistency_ratio` field of the same
[`BWMResult`][bulanik.mcdm.results.BWMResult], but they put **different quantities** in it.
`operators["consistency"]` says which.

| Producer | `consistency_ratio` | `operators["consistency"]` | Scale |
|---|---|---|---|
| [`bwm_linear`][bulanik.lumix.bwm_linear] (LumiX) | $\xi^L$ itself, equal to `xi` | `"xi_L"` | the comparisons' scale: 0 to a few units, no fixed upper bound |
| [`BWMProblem.solve`][bulanik.sezgi.BWMProblem.solve] | $\xi / CI(a_{BW})$ with Rezaei's index | `"rezaei2015_ci"` | about 0 to 1, analogous to AHP's CR; **`None`** when $CI = 0$ and $\xi > 0$ |
| [`FuzzyBWMProblem.solve`][bulanik.sezgi.FuzzyBWMProblem.solve] | $k / CI$ with Guo and Zhao's index | `"guo_zhao2017_ci"` | a ratio; the fuzzy index is never 0, so always a number |

The crisp index is $CI(a) = ((1 + 2a) - \sqrt{8a + 1})/2$ and is **0 for $a_{BW} = 1$**
(best and worst equally important). Then the ratio is defined only if the comparisons are
consistent: `0.0` when $\xi \le 10^{-6}$, and otherwise `None` with the reason in
`extras["consistency_note"]`. The fuzzy index $((1 + 2u) + \sqrt{8u + 1})/2$ is at least 3.

The same inconsistent comparisons give different numbers, partly because the models differ:
the linear model reports its own optimum $\xi^L$ (about 0.04), the nonlinear one a larger
$\xi$ (about 0.32, deviations of *ratios* rather than of linear rows), and the ratio then
divides that by $CI$. The two `consistency_ratio` values (0.04 and 0.07) only look close by
chance:

```python exec="on" source="above" result="text"
from bulanik.lumix import bwm_linear
from bulanik.sezgi import BWMProblem

best_to_others = {"a": 1, "b": 2, "c": 8}
others_to_worst = {"a": 8, "b": 3, "c": 1}             # b over c should be 4 to be consistent

linear = bwm_linear(best_to_others, others_to_worst)
nonlinear = BWMProblem(best_to_others, others_to_worst).solve(budget=3000, seed=1)
for r in (linear, nonlinear):
    print(f"{r.operators['consistency']:14} consistency_ratio = {r.consistency_ratio:.2f}  xi = {r.xi:.2f}")

# a_BW = 1: the index is 0, so an inconsistent set has no ratio
p = BWMProblem({"a": 1, "b": 1, "c": 3}, {"a": 1, "b": 1, "c": 2}, best="a", worst="b")
r = p.solve(budget=2000, seed=1)
print(r.consistency_ratio)
print(r.extras["consistency_note"])
```

Never compare a linear $\xi^L$ with a sezgi ratio, and treat `None` as "no ratio exists",
not as an error. See also the box on the [LumiX bridge](bridge-lumix.md).

## Training an ANFIS with sezgi

[`ANFIS.fit`](anfis.md) is hybrid (least-squares plus gradient) and uses derivatives for the premise parameters. For a derivative-free *global* search over all
membership and consequent parameters at once,
[`train_anfis(anfis, X, y, *, algorithm=None, budget=20000, seed=0, coef_bound=10.0)`][bulanik.sezgi.train_anfis]
trains a **deep copy** of the model (yours is untouched) and returns an
[`ANFISTrainResult`][bulanik.sezgi.ANFISTrainResult] with the trained `anfis`, its `rmse` and
`evals_used`. An ANFIS built without `ranges` gets them from the data on the copy.

Underneath is [`ANFISProblem`][bulanik.sezgi.ANFISProblem]: the space is `Float(0, 1, D)`
over the $D$ parameters, and `decode(z)` maps each coordinate affinely to its bounds
(Gaussian centre in the input range, $\sigma$ in $[\text{span}/100, \text{span}]$, bell $b$ in
$[0.5, 5]$, consequents in $[-\texttt{coef\_bound}, \texttt{coef\_bound}]$). `evaluate` is the
training RMSE, or infinity if no rule fires or the output overflows. The problem *mutates* its
working ANFIS; that is why `train_anfis` hands it a copy.

```python exec="on" source="above" result="text"
import math
from bulanik.anfis import ANFIS
from bulanik.sezgi import ANFISProblem, train_anfis

X = [[2 * math.pi * i / 49] for i in range(50)]
y = [math.sin(x[0]) for x in X]

start = ANFIS(1, 3, ranges=[(0, 2 * math.pi)])           # 3 rules, 12 parameters
before = start.get_params()
result = train_anfis(start, X, y, budget=20000, seed=1)
print("sezgi RMSE < 0.1:", result.rmse < 0.1, "|", result.evals_used, "evaluations")
print("original untouched:", start.get_params() == before)

hybrid = ANFIS(1, 3, ranges=[(0, 2 * math.pi)])
print("hybrid RMSE < 0.1:", hybrid.fit(X, y, epochs=10)[-1] < 0.1)

problem = ANFISProblem(ANFIS(1, 2, ranges=[(0, 1)]), [[0.0], [1.0]], [0.0, 1.0], coef_bound=2)
print(problem.dim, problem.decode([0.0] * problem.dim))
print(problem.decode([1.0] * problem.dim))
```

For a smooth one-dimensional target the hybrid rule is both faster and more accurate; a
metaheuristic earns its keep when the loss surface is rough, when you want a different
objective, or when you want to start the gradient method from a global solution. Example
[16 ANFIS with sezgi](../examples/16-anfis-with-sezgi.md) compares the two.

## Pitfalls

**Budget and seed are part of the answer.** A small `budget` gives a rough optimum, and a
different `seed` a slightly different one. Fix both when you report numbers, and raise the
budget until the result stops moving.

**The sezgi objective is penalised, not constrained.** Weights from `solve` are rescaled
to sum to 1 afterwards; `xi` is recomputed at the rescaled weights.

**Do not compare `consistency_ratio` across producers** (see the table above), and handle
`None` for crisp BWM with $a_{BW} = 1$.

**`FuzzyBWMProblem` wants `TFN` with $l > 0$ and $m \ge 1$.** Crisp numbers and other
number kinds are errors, and so is a missing `TFN(1, 1, 1)` (pass `best=` and `worst=`
only to resolve a tie, they must still have the "one" comparison).

**`ANFISProblem` needs ranges.** Build the ANFIS with `ranges=` (or use `train_anfis`,
which infers them from the data; a constant input column is an error then).

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.anfis import ANFIS
from bulanik.scales import FUZZY_BWM_TFN as F
from bulanik.sezgi import ANFISProblem, BWMProblem, FuzzyBWMProblem, train_anfis

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

ok = ({"a": 1, "b": 2, "c": 8}, {"a": 8, "b": 4, "c": 1})
show(lambda: BWMProblem(*ok).solve(budget=0))
show(lambda: BWMProblem(*ok, penalty=0))
show(lambda: BWMProblem({"a": 1, "b": 2}, {"a": 8, "b": 1}))                       # a_BW mismatch
show(lambda: BWMProblem(ok[0], {"a": 8, "b": 4}))                                  # keys differ
show(lambda: BWMProblem(*ok).deviation([1, 2]))                                    # wrong length
show(lambda: FuzzyBWMProblem({"a": 1, "b": 2}, {"a": 2, "b": 1}))                  # not TFN
show(lambda: FuzzyBWMProblem({"a": F["EI"], "b": TFN(0, 1, 2)}, {"a": F["AI"], "b": F["EI"]}))
show(lambda: ANFISProblem(ANFIS(1, 2), [[0.0], [1.0]], [0.0, 1.0]))                # no ranges
show(lambda: train_anfis(ANFIS(1, 2), [[1.0], [1.0]], [0.0, 1.0]))                 # constant column
show(lambda: train_anfis("not an anfis", [[0.0]], [0.0]))
```

## See also

- [Weighting](weighting.md#best-worst-method) for the BWM and its consistency indices, and
  the [LumiX bridge](bridge-lumix.md) for the linear BWM.
- [ANFIS](anfis.md) for the model and gradient training.
- API: [`bulanik.sezgi`][bulanik.sezgi].
- Examples: [08 fuzzy BWM with sezgi](../examples/08-fuzzy-bwm-sezgi.md),
  [12 tune Sugeno with sezgi](../examples/12-tune-sugeno-with-sezgi.md),
  [16 ANFIS with sezgi](../examples/16-anfis-with-sezgi.md).
- Rezaei (2015), "Best-worst multi-criteria decision-making method", *Omega* 53; Guo and
  Zhao (2017), "Fuzzy best-worst multi-criteria decision-making method and its
  applications", *Knowledge-Based Systems* 121.
