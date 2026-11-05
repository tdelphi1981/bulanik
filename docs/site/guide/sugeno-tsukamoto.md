# Sugeno and Tsukamoto inference

Mamdani systems defuzzify a whole output *set*. Two other families skip that step by
giving every rule a crisp number and averaging the numbers, weighted by how strongly
each rule fires. They are cheaper (no sampling grid), smooth, and the natural form for
control and for models fitted to data (see [ANFIS](anfis.md)).

- [`SugenoFIS`][bulanik.inference.sugeno.SugenoFIS] (Takagi-Sugeno-Kang): each rule's
  consequent is a **constant** (zero order) or a **linear function of the inputs** (first
  order).
- [`TsukamotoFIS`][bulanik.inference.tsukamoto.TsukamotoFIS]: each rule's consequent is a
  **monotonic** membership function, and the rule reads its crisp value off that function
  at its firing strength.

Both use the rule language of the [Mamdani page](mamdani.md): the same syntax, `and_method`
(`"min"` or `"prod"`), `or_method` (`"max"` or `"probor"`), `default`, `explain` and
`NoRuleFiredError`.

## Sugeno

Let rule $k$ fire with strength $\alpha_k$ (rule weight times the antecedent degree) and
have crisp value $z_k$. The output is

$$
y = \frac{\sum_k \alpha_k z_k}{\sum_k \alpha_k}
\qquad\text{(\texttt{output\_method="weighted\_average"}, the default)},
$$

or $y = \sum_k \alpha_k z_k$ for `"weighted_sum"`, over the rules that fire. A constant
consequent gives $z_k = c$; a [`Linear`][bulanik.inference.sugeno.Linear] consequent gives
$z_k = c_0 + \sum_i c_i x_i$ evaluated at the crisp inputs.

Output terms are not membership functions, so outputs are
[`SugenoOutput`][bulanik.inference.sugeno.SugenoOutput] objects that map each term name to
a number or a `Linear`. Inputs are ordinary linguistic variables.

```python exec="on" source="above" result="text"
from bulanik.inference import Linear, SugenoFIS, SugenoOutput
from bulanik.sets import LinguisticVariable, linear_decreasing, linear_increasing

x = LinguisticVariable("x", (0, 10), {
    "lo": linear_decreasing(0, 10), "hi": linear_increasing(0, 10)})
y = SugenoOutput("y", {"a": 1.0, "b": Linear({"x": 2.0}, 1.0)})   # constant and first order
rules = ["IF x IS lo THEN y IS a", "IF x IS hi THEN y IS b"]

fis = SugenoFIS([x], [y], rules)
print(fis.evaluate({"x": 2.5}))     # (0.75 * 1 + 0.25 * 6) / 1.0
weighted = ["IF x IS lo THEN y IS a WITH 0.5", "IF x IS hi THEN y IS b"]
print(SugenoFIS([x], [y], weighted).evaluate({"x": 2.5}))
print(SugenoFIS([x], [y], weighted, output_method="weighted_sum").evaluate({"x": 2.5}))
print(y)
```

A `Linear` may only name input variables of the system (checked at construction); extra
inputs are ignored when it is called, and a missing one is an error. When the consequents
are linear in the inputs and the rules partition the input space, the output is smooth:
the Sugeno system is a blend of local linear models. A two-input control surface is drawn
with [`plot.surface`][bulanik.plot.surface].

`explain` reports the degrees, the activations and the outputs, exactly as in Mamdani.

```python exec="on" source="above" result="text"
from bulanik.inference import SugenoFIS, SugenoOutput
from bulanik.sets import LinguisticVariable, linear_decreasing, linear_increasing

x = LinguisticVariable("x", (0, 10), {
    "lo": linear_decreasing(0, 10), "hi": linear_increasing(0, 10)})
y = SugenoOutput("y", {"a": 0.0, "b": 100.0})
fis = SugenoFIS([x], [y], ["IF x IS lo THEN y IS a", "IF x IS hi THEN y IS b"])
info = fis.explain({"x": 2.5})
print([a.strength for a in info.activations], info.outputs)
```

## Tsukamoto

Every output term must be monotonic: `linear_increasing`, `linear_decreasing` or
`sigmoid`. A fired rule with strength $\alpha_k$ and consequent term $B_k$ gives the crisp
value $z_k = \mu_{B_k}^{-1}(\alpha_k)$, the point where $B_k$ reaches the level
$\alpha_k$ (see [`inverse`][bulanik.sets.MembershipFunction.inverse]), and the output is the
same weighted average as for Sugeno,

$$
y = \frac{\sum_k \alpha_k z_k}{\sum_k \alpha_k}.
$$

A strongly fired "high" rule therefore pushes the output towards the top of its ramp, a
weakly fired one stays near the foot. Each $z_k$ is clamped into the output's universe
(the inverse of a sigmoid or a ramp can fall outside it), and so is the final average;
inputs are never clamped.

```python exec="on" source="above" result="text"
from bulanik.inference import TsukamotoFIS
from bulanik.sets import LinguisticVariable, linear_decreasing, linear_increasing

x = LinguisticVariable("x", (0, 10), {
    "lo": linear_decreasing(0, 10), "hi": linear_increasing(0, 10)})
z = LinguisticVariable("z", (0, 20), {
    "low": linear_decreasing(0, 10), "high": linear_increasing(0, 20)})
fis = TsukamotoFIS([x], [z], ["IF x IS lo THEN z IS low", "IF x IS hi THEN z IS high"])
for value in (0, 2.5, 5, 10):
    print(value, fis.evaluate({"x": value}))
info = fis.explain({"x": 2.5})
print([a.strength for a in info.activations])    # z = (0.75 * 2.5 + 0.25 * 5) / 1.0 = 3.125
```

## Choosing between the three

| | Mamdani | Sugeno | Tsukamoto |
|---|---|---|---|
| Consequent | fuzzy set | constant or linear function | monotonic fuzzy set |
| Defuzzification | on a sampled grid (`resolution`) | weighted average, closed form | weighted average of inverses |
| Output is | explainable by the picture of the aggregated set | a blend of local models | a blend of inverted ramps |
| Use it for | expert knowledge in words | control and data-fitted models | monotone effects, sigmoids |

Use Mamdani when people will read the rule base and the aggregated set; Sugeno when you
want speed, smoothness or a model you will tune (the sezgi example
[12 tune Sugeno with sezgi](../examples/12-tune-sugeno-with-sezgi.md) tunes the constants);
Tsukamoto for a monotone consequent where the *degree* of firing should map to a *position*.

## When no rule fires

If $\sum_k \alpha_k = 0$ for an output, there is nothing to average. The behaviour is the
same as in Mamdani: a [`NoRuleFiredError`][bulanik.inference.rules.NoRuleFiredError]
with `.output` and `.explanation`, or the `default` value (a finite number per output; for
Tsukamoto it must lie in the universe) if you gave one.

```python exec="on" source="above" result="text"
from bulanik.inference import NoRuleFiredError, SugenoFIS, SugenoOutput
from bulanik.sets import LinguisticVariable, triangular

x = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 3)})
y = SugenoOutput("y", {"a": 1.0})
rules = ["IF x IS lo THEN y IS a"]
try:
    SugenoFIS([x], [y], rules).evaluate({"x": 8})
except NoRuleFiredError as error:
    print(error.output, [a.strength for a in error.explanation.activations])
print(SugenoFIS([x], [y], rules, default={"y": -1.0}).evaluate({"x": 8}))
```

## Pitfalls

**A rule's weight scales its strength but also its pull.** With the default
`weighted_average` a lone fired rule with weight 0.1 still returns its own $z_k$ (the
weight cancels in the ratio); it only matters against other rules. With `weighted_sum` the
weight scales the output directly.

**`weighted_sum` is not an average.** The result is outside the range of the consequents
whenever the strengths sum to more than 1, which happens as soon as the terms overlap
beyond a partition of unity (memberships adding to 1 at every input).

```python exec="on" source="above" result="text"
from bulanik.inference import SugenoFIS, SugenoOutput
from bulanik.sets import LinguisticVariable, gaussian

x = LinguisticVariable("x", (0, 10), {"a": gaussian(5, 3), "b": gaussian(5, 3)})
y = SugenoOutput("y", {"p": 10.0, "q": 10.0})
rules = ["IF x IS a THEN y IS p", "IF x IS b THEN y IS q"]
print(SugenoFIS([x], [y], rules).evaluate({"x": 5}))                              # 10: an average
print(SugenoFIS([x], [y], rules, output_method="weighted_sum").evaluate({"x": 5}))  # 20: overshoots
```

**Tsukamoto needs monotonic outputs.** A triangle or a Gaussian output term raises
`ValueError` at construction (its inverse is not a function). A sigmoid's inverse is
clamped near 0 and 1 (`SIGMOID_EPS`), so a rule fired at exactly 1 reads a finite value.

**A `Linear` must name known inputs**, and must be called with every input it uses:
coefficients for a variable the system does not have raise `ValueError` when the system
is built.

**`NOT` and weights follow the Mamdani rules**: `NOT` only after `IS` in an antecedent,
weights in $[0, 1]$.

```python exec="on" source="above" result="text"
from bulanik.inference import Linear, SugenoFIS, SugenoOutput, TsukamotoFIS
from bulanik.sets import LinguisticVariable, linear_decreasing, linear_increasing, triangular

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

x = LinguisticVariable("x", (0, 10), {"lo": linear_decreasing(0, 10), "hi": linear_increasing(0, 10)})
tri = LinguisticVariable("z", (0, 10), {"mid": triangular(0, 5, 10)})
show(lambda: TsukamotoFIS([x], [tri], ["IF x IS lo THEN z IS mid"]))
y = SugenoOutput("y", {"a": Linear({"t": 1.0})})
show(lambda: SugenoFIS([x], [y], ["IF x IS lo THEN y IS a"]))
show(lambda: Linear({"x": 1.0})({}))
show(lambda: SugenoOutput("y", {}))
show(lambda: SugenoOutput("y", {"a": float("nan")}))
show(lambda: SugenoFIS([x], [SugenoOutput("y", {"a": 1.0})], ["IF x IS lo THEN y IS a"],
                       output_method="median"))
```

## See also

- [Mamdani inference](mamdani.md) for the rule language and options both systems share;
  [ANFIS](anfis.md) fits a first-order Sugeno system to data.
- [Interval type-2](type2.md) for `IT2SugenoFIS`.
- API: [`bulanik.inference`][bulanik.inference].
- Examples: [11 Sugeno controller](../examples/11-sugeno-controller.md),
  [13 Tsukamoto](../examples/13-tsukamoto.md).
- Takagi and Sugeno (1985), "Fuzzy identification of systems and its applications to
  modeling and control", *IEEE Transactions on Systems, Man, and Cybernetics*;
  Tsukamoto (1979), "An approach to fuzzy reasoning method", in *Advances in Fuzzy Set
  Theory and Applications*.
