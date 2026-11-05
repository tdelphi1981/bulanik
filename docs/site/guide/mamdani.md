# Mamdani inference

A **Mamdani system** ([`MamdaniFIS`][bulanik.inference.mamdani.MamdaniFIS]) turns crisp
inputs into a crisp output through rules written in words: *if service is poor or food
is rancid then tip is cheap*. Use it when the knowledge you have is linguistic and you
want an output you can explain rule by rule. The same rule language drives the
[Sugeno and Tsukamoto](sugeno-tsukamoto.md) systems and the [interval type-2](type2.md)
systems.

## The idea

Inference runs in four steps, and every step is a named option of the constructor.

1. **Fuzzify** each crisp input: its membership degree in every term of its variable.
2. **Fire** each rule. The antecedent is combined with `AND` (a t-norm) and `OR` (a
   t-conorm); multiplying by the rule weight gives the firing strength $\alpha_k$.
3. **Imply and aggregate.** Each rule's consequent set is clipped (or scaled) by
   $\alpha_k$, and the clipped sets are combined into one output set.
4. **Defuzzify** the aggregated set into one number.

<figure markdown="span">
  ![Three clipped consequent sets of the tipper, their aggregate as a dark outline and the centroid as a vertical marker at 15.86](../assets/figures/mamdani-aggregate-light.svg#only-light)
  ![Three clipped consequent sets of the tipper, their aggregate as a dark outline and the centroid as a vertical marker at 15.86](../assets/figures/mamdani-aggregate-dark.svg#only-dark)
  <figcaption markdown="span">The tipper at service 3, food 8. Each fired rule clips its
  consequent term (thin outline) at its firing strength (filled area); the heavy outline
  is the aggregate and the marker is its centroid. Drawn by
  [`plot.mamdani_output`][bulanik.plot.mamdani_output].</figcaption>
</figure>

## A first system

Variables are [`LinguisticVariable`][bulanik.sets.LinguisticVariable]s from
[Sets and scales](sets-and-scales.md); rules are plain strings. `evaluate` takes every
input by name (inside its universe) and returns a dict with one value per output.

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
print(fis.evaluate({"service": 3, "food": 8}))
print(fis.evaluate({"service": 9, "food": 9}))
```

## Rules

The full grammar, with the reserved words and the error messages, is in the
[`bulanik.inference.rules`][bulanik.inference.rules] module description. In short:

```text
IF <antecedent> THEN <var> IS <term> [AND <var> IS <term> ...] [WITH <weight>]
```

- Keywords (`IF THEN IS NOT AND OR WITH`) are case-insensitive and ASCII only, and
  cannot be used as variable or term names.
- Names are a letter or underscore followed by letters, digits or underscores, with no
  spaces: `very_high` and `x2` are fine. Letters outside ASCII are allowed, so Turkish
  names such as `hizmet` / `yemek` / `bahşiş` work.
- An atom is `var IS term` or `var IS NOT term` (degree $1 - \mu$). `NOT` appears only
  after `IS`, never in a consequent.
- `AND` binds tighter than `OR`; parentheses group.
- `WITH w` multiplies the strength by $w \in [0, 1]$ (default 1).
- A rule may name several outputs (`THEN a IS x AND b IS y`) or leave an output out.

You can also build rules from objects: `Term`, `And`, `Or` and `Rule`, with the Python
operators `&`, `|` and `~` on terms. `str(rule)` prints canonical text that
[`parse_rule`][bulanik.inference.rules.parse_rule] reads back.

```python exec="on" source="above" result="text"
from bulanik.inference import MamdaniFIS, Rule, Term, parse_rule
from bulanik.sets import LinguisticVariable, triangular

hizmet = LinguisticVariable("hizmet", (0, 10), {"kötü": triangular(0, 0, 5), "iyi": triangular(5, 10, 10)})
yemek = LinguisticVariable("yemek", (0, 10), {"bayat": triangular(0, 0, 5), "lezzetli": triangular(5, 10, 10)})
bahsis = LinguisticVariable("bahşiş", (0, 20), {"az": triangular(0, 0, 10), "çok": triangular(10, 20, 20)})

built = Rule(Term("hizmet", "iyi") | ~Term("yemek", "bayat"), {"bahşiş": "çok"}, 0.8)
print(built)
print(parse_rule(str(built)) == built)
fis = MamdaniFIS([hizmet, yemek], [bahsis], [built, "IF hizmet IS kötü THEN bahşiş IS az"])
print(fis.evaluate({"hizmet": 8, "yemek": 2}))
```

## Operators and defuzzifiers

Every option is a string, checked at construction, and the defaults are those of the
first row of each list.

| Option | Choices | Meaning |
|---|---|---|
| `and_method` | `"min"`, `"prod"` | t-norm for `AND`: $\min(a, b)$ or $ab$ |
| `or_method` | `"max"`, `"probor"` | t-conorm for `OR`: $\max(a, b)$ or $a + b - ab$ |
| `implication` | `"min"`, `"prod"` | clip $\min(\alpha, \mu)$ or scale $\alpha\mu$ the consequent |
| `aggregation` | `"max"`, `"sum"`, `"probor"` | combine implied sets: $\max$, $a + b$ (not clipped to 1, as in MATLAB), or $a + b - ab$ |
| `defuzzification` | `"centroid"`, `"bisector"`, `"mom"`, `"som"`, `"lom"` | see below |
| `resolution` | integer $\ge 2$, default 1001 | samples per output universe |

With each output universe sampled at the evenly spaced points $x_i$ (endpoints
included), the aggregated set is

$$
\mu(x_i) = \operatorname{agg}_k \; \operatorname{imp}\bigl(\alpha_k,\; \mu_{B_k}(x_i)\bigr),
$$

where $B_k$ is the consequent term of rule $k$. The crisp output is a function of the
samples $(x_i, \mu_i)$ (see
[`defuzzify_samples`][bulanik.inference.mamdani.defuzzify_samples]):

| Method | Result |
|---|---|
| `centroid` | $\sum_i x_i \mu_i \big/ \sum_i \mu_i$ |
| `bisector` | the first $x_i$ where the running sum of $\mu$ reaches half of the total |
| `mom`, `som`, `lom` | the mean, smallest and largest $x_i$ among the samples whose $\mu_i$ is within $10^{-12}$ of the maximum |

The centroid is the usual choice because it moves smoothly with the inputs;
`mom`/`som`/`lom` jump when the maximum moves from one plateau to another. You can call
`defuzzify_samples` directly on any samples.

```python exec="on" source="above" result="text"
from bulanik.inference import MamdaniFIS, defuzzify_samples
from bulanik.sets import LinguisticVariable, triangular

service = LinguisticVariable("service", (0, 10), {
    "poor": triangular(0, 0, 5), "good": triangular(5, 10, 10)})
tip = LinguisticVariable("tip", (0, 20), {
    "low": triangular(0, 0, 10), "high": triangular(10, 20, 20)})
rules = ["IF service IS poor THEN tip IS low", "IF service IS good THEN tip IS high"]

for method in ("centroid", "bisector", "mom", "som", "lom"):
    fis = MamdaniFIS([service], [tip], rules, defuzzification=method)
    print(f"{method:9s}", round(fis.evaluate({"service": 7.5})["tip"], 3))
print("prod / sum:", round(MamdaniFIS([service], [tip], rules, implication="prod",
                                      aggregation="sum").evaluate({"service": 7.5})["tip"], 3))
print(defuzzify_samples([0, 1, 2, 3], [0, 1, 1, 0], "centroid"))   # a trapezoid's centre
```

## Seeing the control surface

[`plot.surface`][bulanik.plot.surface] draws the whole input-output map of a system with
one or two inputs (a curve or a coloured surface); it works for every system in this
library. For the tipper it shows the flat corners where one rule dominates and the
ridges where the rules blend.

<figure markdown="span">
  ![Three-dimensional control surface of the tipper: tip as a function of service and food](../assets/figures/mamdani-surface-light.svg#only-light)
  ![Three-dimensional control surface of the tipper: tip as a function of service and food](../assets/figures/mamdani-surface-dark.svg#only-dark)
  <figcaption markdown="span">Control surface of the tipper over the whole input square.</figcaption>
</figure>

## Explaining a result

`explain` evaluates and returns an [`Explanation`][bulanik.inference.rules.Explanation]:
the membership `degrees` of every input in every term, one
[`RuleActivation`][bulanik.inference.rules.RuleActivation] per rule (the rule text and
its strength, weight included) and the crisp `outputs` that `evaluate` returns.

```python exec="on" source="above" result="text"
from bulanik.inference import MamdaniFIS
from bulanik.sets import LinguisticVariable, triangular

service = LinguisticVariable("service", (0, 10), {
    "poor": triangular(0, 0, 5), "good": triangular(5, 10, 10)})
food = LinguisticVariable("food", (0, 10), {
    "rancid": triangular(0, 0, 5), "delicious": triangular(5, 10, 10)})
tip = LinguisticVariable("tip", (0, 20), {
    "low": triangular(0, 0, 10), "high": triangular(10, 20, 20)})
fis = MamdaniFIS([service, food], [tip], [
    "IF service IS poor AND food IS rancid THEN tip IS low",
    "IF service IS good OR food IS delicious THEN tip IS high WITH 0.9",
])
info = fis.explain({"service": 3, "food": 8})
print(info.degrees)
for activation in info.activations:
    print(f"{activation.strength:.2f}  {activation.rule}")
print(info.outputs)
```

## When no rule fires

If every rule's strength is 0 for an output, the aggregate is empty and there is nothing
to defuzzify. By default a
[`NoRuleFiredError`][bulanik.inference.rules.NoRuleFiredError] (a `ValueError`) is
raised. It carries `output` (the variable name) and `explanation`, the same
`Explanation` that `explain` would have returned, with every strength 0, so you can see
which input fell through the gaps in your rule base. To return a fixed value instead,
pass `default={"tip": 10}`, one in-universe value per output.

```python exec="on" source="above" result="text"
from bulanik.inference import MamdaniFIS, NoRuleFiredError
from bulanik.sets import LinguisticVariable, triangular

x = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 3)})
y = LinguisticVariable("y", (0, 10), {"low": triangular(0, 0, 10)})
rules = ["IF x IS lo THEN y IS low"]

try:
    MamdaniFIS([x], [y], rules).evaluate({"x": 8})
except NoRuleFiredError as error:
    print(error)
    print(error.output, error.explanation.degrees, error.explanation.activations[0].strength)
print(MamdaniFIS([x], [y], rules, default={"y": 5}).evaluate({"x": 8}))
```

## Pitfalls

**Inputs must lie inside their universe**, must be given for every input variable and no
others, and must be finite numbers (a `bool` or a string is a `TypeError`). Inputs are
never clamped; a sensor value just past the end of the universe is an error, not a
silent 10.

**Rule text is checked at construction.** An unknown variable or term, a keyword used as
a name, `NOT` in a consequent, a weight outside $[0, 1]$, or an output named twice in one
consequent raises `ValueError`, usually naming the offset in the rule text.

**`resolution` trades accuracy for time.** The output is exact only up to the sampling
grid. For the two-rule example above (service 7.5) the centroid is 16.67 at resolution
11, 16.16 at 101, 16.116 at 1001 and 16.112 at 10001; the default 1001 is accurate to
about three decimals for ordinary shapes. `bisector`, `mom`, `som` and `lom` return grid
points, so they move in steps of one grid cell.

**`aggregation="sum"` can exceed 1**, which is intended (it is MATLAB's `sum`); `centroid`
and `bisector` then weight overlapping rules more heavily than `max` does.

**A consequent term must be visible on the grid.** A term that is 0 at every sample
(entirely outside the output universe, or narrower than one grid cell) is rejected at
construction, with a message that suggests moving it or raising `resolution`. If a rule
seems ignored, `explain` shows whether it fired and the figure shows what it contributed.

```python exec="on" source="above" result="text"
from bulanik.inference import MamdaniFIS
from bulanik.sets import LinguisticVariable, triangular

x = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 5), "hi": triangular(5, 10, 10)})
y = LinguisticVariable("y", (0, 10), {"low": triangular(0, 0, 5), "high": triangular(5, 10, 10)})
rules = ["IF x IS lo THEN y IS low", "IF x IS hi THEN y IS high"]

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

fis = MamdaniFIS([x], [y], rules)
show(lambda: fis.evaluate({"x": 11}))
show(lambda: fis.evaluate({}))
show(lambda: fis.evaluate({"x": True}))
show(lambda: MamdaniFIS([x], [y], ["IF x IS medium THEN y IS low"]))
show(lambda: MamdaniFIS([x], [y], ["IF x IS lo THEN y IS NOT low"]))
show(lambda: MamdaniFIS([x], [y], ["IF x IS lo THEN y IS low WITH 1.5"]))
show(lambda: MamdaniFIS([x], [y], rules, defuzzification="median"))
show(lambda: MamdaniFIS([x], [y], rules, resolution=1))
show(lambda: MamdaniFIS([x], [y], rules, default={"y": 50}))
gone = LinguisticVariable("z", (0, 10), {"gone": triangular(20, 25, 30)})
show(lambda: MamdaniFIS([x], [gone], ["IF x IS lo THEN z IS gone"]))
```

## See also

- [Sugeno and Tsukamoto](sugeno-tsukamoto.md) for crisp consequents, and
  [Interval type-2](type2.md) for uncertain membership functions.
- [Sets and scales](sets-and-scales.md) for the membership functions and variables.
- API: [`bulanik.inference`][bulanik.inference], [`bulanik.plot`][bulanik.plot].
- Example: [10 Mamdani tipper](../examples/10-mamdani-tipper.md).
- Mamdani and Assilian (1975), "An experiment in linguistic synthesis with a fuzzy logic controller",
  *International Journal of Man-Machine Studies*.
