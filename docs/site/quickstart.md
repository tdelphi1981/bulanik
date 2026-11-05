# Quickstart

Five short recipes. Every code block runs when the site is built, so the output
under each one is real. Recipe 5 (the `lumix` bridge) and the plot in recipe 1
(`bulanik.plot`) need their [extras](install.md); everything else needs nothing
but bulanik.

## 1. Fuzzy numbers

Numbers are immutable, validated values. Operators call the default
(`"algebraic"`) operator family; `add`, `mul` and friends take an `ops=` name.

```python exec="on" source="above" result="text"
from bulanik import IFN, TFN

a, b = TFN(1, 2, 4), TFN(2, 3, 5)
print(a + b)                    # componentwise sum of the triangles
print(a * b)                    # min/max of the corner products, product of the peaks
print(a.alpha_cut(0.5))         # the interval where membership >= 0.5
print(round(a.score(), 4))      # crisp score for ranking (centroid by default)

x, y = IFN(0.6, 0.3), IFN(0.5, 0.4)   # intuitionistic: membership + non-membership <= 1
print(x + y)                          # algebraic operators (default)
print(x.add(y, ops="einstein"))       # Einstein operators
print(x.score(), x.score("normalized"))

try:
    IFN(0.8, 0.5)                     # 0.8 + 0.5 > 1
except ValueError as error:
    print("ValueError:", error)
```

A membership plot of two triangles, drawn with [`bulanik.plot`](api/plot.md) (the
`plot` extra) at build time in both themes; the one matching your colour scheme
is shown:

```python exec="on" html="1"
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bulanik import TFN, plot

for mode in ("light", "dark"):
    with plot.theme(mode):
        ax = plot.membership([TFN(1, 2, 4), TFN(2, 4, 5)], labels=["A", "B"], alpha_cut=0.5)
        buffer = io.StringIO()
        ax.figure.savefig(buffer, format="svg")
        plt.close(ax.figure)
    print(f'<div class="bulanik-fig bulanik-fig-{mode}">{buffer.getvalue()}</div>')
```

## 2. Aggregate expert opinions

Experts write words; a [`LinguisticScale`][bulanik.scales.LinguisticScale] turns
them into fuzzy numbers; [`aggregate_experts`][bulanik.aggregate.aggregate_experts]
combines the `[expert][alternative][criterion]` table cell by cell.

```python exec="on" source="above" result="text"
from bulanik.aggregate import aggregate_experts, wa, wg
from bulanik.scales import EXAMPLE_IFN_5

words = [
    [["H", "M"], ["L", "VL"]],      # expert 1: two alternatives, two criteria
    [["VH", "M"], ["M", "H"]],      # expert 2
]
ratings = EXAMPLE_IFN_5.convert(words)
combined = aggregate_experts(ratings, [0.6, 0.4], method="wa")
for row in combined:
    print([f"({n.mu:.3f}, {n.nu:.3f})" for n in row])

a, b = ratings[0][0][0], ratings[1][0][0]
for name, mean in [("wa", wa([a, b], [0.5, 0.5])), ("wg", wg([a, b], [0.5, 0.5]))]:
    print(f"{name}: ({mean.mu:.4f}, {mean.nu:.4f})")    # weighted average, weighted geometric mean
```

## 3. A Mamdani fuzzy controller

Rules are plain text. `evaluate` returns the defuzzified outputs; `explain` shows
why.

```python exec="on" source="above" result="text"
from bulanik.inference import MamdaniFIS
from bulanik.sets import LinguisticVariable, triangular

service = LinguisticVariable("service", (0, 10), {
    "poor": triangular(0, 0, 5), "average": triangular(0, 5, 10), "good": triangular(5, 10, 10)})
food = LinguisticVariable("food", (0, 10), {
    "rancid": triangular(0, 0, 5), "delicious": triangular(5, 10, 10)})
tip = LinguisticVariable("tip", (0, 30), {
    "cheap": triangular(0, 5, 10), "average": triangular(10, 15, 20),
    "generous": triangular(20, 25, 30)})

fis = MamdaniFIS([service, food], [tip], [
    "IF service IS poor OR food IS rancid THEN tip IS cheap",
    "IF service IS average THEN tip IS average",
    "IF service IS good OR food IS delicious THEN tip IS generous",
])
inputs = {"service": 3, "food": 8}
print(f"tip = {fis.evaluate(inputs)['tip']:.2f}")
for activation in fis.explain(inputs).activations:
    print(f"  fired {activation.strength:.2f}: {activation.rule}")
```

## 4. Fuzzy TOPSIS

Triangular ratings and triangular weights, ranked with Chen's (2000) fuzzy TOPSIS.

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.mcdm import fuzzy_topsis
from bulanik.scales import EXAMPLE_TFN_7 as scale

criteria = ["quality", "service", "price"]
ratings = scale.convert([
    ["G", "VG", "F"],               # alternative A on each criterion
    ["F", "G", "G"],                # B
    ["MP", "F", "VG"],              # C
])
weights = [TFN(0.7, 0.9, 1.0), TFN(0.5, 0.7, 0.9), TFN(0.3, 0.5, 0.7)]
result = fuzzy_topsis(ratings, weights, ["benefit", "benefit", "cost"],
                      alternatives=["A", "B", "C"], criteria=criteria)
print(result.ranking)
for name in result.ranking:
    print(f"  {name}: closeness {result.scores[name]:.4f}")
print(result.operators)             # the names of the methods that were used
```

## 5. Best-worst weights, then a ranking

The best-worst method (BWM) asks for two short comparison vectors. It needs a
solver, so it lives in the bridges: [`bwm_linear`][bulanik.lumix.bwm_linear]
(the `lumix` extra) solves it as a linear program. The result feeds straight
into [`topsis`][bulanik.mcdm.topsis].

```python exec="on" source="above" result="text"
from bulanik.lumix import bwm_linear
from bulanik.mcdm import topsis

# quality is the best criterion, price the worst
best_to_others = {"quality": 1, "service": 2, "price": 4}
others_to_worst = {"quality": 4, "service": 3, "price": 1}
bwm = bwm_linear(best_to_others, others_to_worst)
print({k: round(v, 4) for k, v in bwm.weights.items()}, f"xi = {bwm.xi:.4f}")

data = [[8, 7, 120], [7, 5, 100], [9, 6, 110]]       # rows: A, B, C
result = topsis(data, bwm, ["benefit", "benefit", "cost"],
                alternatives=["A", "B", "C"], criteria=list(bwm.weights))
print(result.ranking)
print({k: round(v, 4) for k, v in result.scores.items()})
```

## Next

- [Conventions](concepts/conventions.md): what the library promises about
  immutability, errors, tolerances and names.
- [Choosing a method](concepts/choosing.md): which number type, inference system
  and MCDM method fits your problem.
- The [user guide](guide/numbers-triangular.md) and the 24 [examples](examples/index.md).
