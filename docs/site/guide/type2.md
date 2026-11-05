# Interval type-2 inference

A type-1 membership function says "the temperature 22 is *warm* to degree 0.7", with
the 0.7 exact. An **interval type-2** (IT2) set admits that the 0.7 is itself uncertain:
it gives each $x$ an *interval* of degrees $[\underline{\mu}(x), \overline{\mu}(x)]$. The
region between the lower and the upper curve is the **footprint of uncertainty** (FOU).
Use it when experts disagree about where a term starts and ends, or when you want a
controller that is less sensitive to noisy inputs than a single sharp curve.

## Interval sets

An [`IT2MF`][bulanik.type2.IT2MF] is built from two type-1 functions, `upper` and
`lower`; calling it returns the pair `(lower, upper)`. Three builders cover the common
cases:

| Builder | Upper bound | Lower bound |
|---|---|---|
| [`gaussian_uncertain_mean`][bulanik.type2.gaussian_uncertain_mean]`(m1, m2, sigma)` | 1 on $[m_1, m_2]$, the Gaussian of the nearer mean outside | $\min\bigl(g(x; m_1), g(x; m_2)\bigr)$ |
| [`gaussian_uncertain_sigma`][bulanik.type2.gaussian_uncertain_sigma]`(mean, s1, s2)` | Gaussian with the wider $\sigma_2$ | Gaussian with the narrower $\sigma_1$ |
| [`scaled_triangle`][bulanik.type2.scaled_triangle]`(a, b, c, h)` | triangle $(a, b, c)$ | the same triangle scaled to height $h$ |

Here $g(x; m) = \exp\bigl(-(x - m)^2 / (2\sigma^2)\bigr)$.

<figure markdown="span">
  ![Footprints of uncertainty of an uncertain-mean Gaussian, an uncertain-sigma Gaussian and a scaled triangle: a shaded band between a thick upper and a thin lower curve](../assets/figures/it2-fou-light.svg#only-light)
  ![Footprints of uncertainty of an uncertain-mean Gaussian, an uncertain-sigma Gaussian and a scaled triangle: a shaded band between a thick upper and a thin lower curve](../assets/figures/it2-fou-dark.svg#only-dark)
  <figcaption markdown="span">The shaded band is the footprint of uncertainty; the thick
  line is the upper bound, the thin line the lower bound. Drawn by
  [`plot.it2_set`][bulanik.plot.it2_set].</figcaption>
</figure>

A lower bound above the upper bound at any $x$ is an error (`ValueError`, raised when that
$x$ is evaluated; float noise within $10^{-12}$ is clamped). An
[`IT2LinguisticVariable`][bulanik.type2.IT2LinguisticVariable] groups IT2 terms on a
universe; [`from_type1`][bulanik.type2.IT2LinguisticVariable.from_type1] turns a type-1
variable into one with no uncertainty (lower equals upper).

```python exec="on" source="above" result="text"
from bulanik.type2 import (IT2LinguisticVariable, gaussian_uncertain_mean, scaled_triangle)

mf = gaussian_uncertain_mean(4, 6, 1.2)
print(mf(5), mf(8))                      # (lower, upper) at two points
x = IT2LinguisticVariable("x", (0, 10), {
    "lo": scaled_triangle(0, 0, 10, 0.5), "hi": scaled_triangle(0, 10, 10, 0.5)})
print(x.fuzzify(2))
print(x)
```

## Type reduction

Fuzzy inference with IT2 sets does not produce one number. Each rule fires with an
interval $[\underline{f}, \overline{f}]$: the antecedent is evaluated once with the lower
bounds and once with the upper bounds (`NOT` maps $[a, b]$ to $[1 - b, 1 - a]$), and the
rule weight multiplies both. The result is a set of points $y_i$ whose weights $w_i$ are
only known to lie in intervals. **Type reduction** turns that into an interval
$[y_l, y_r]$ of possible crisp outputs, and the system returns its midpoint (the
**output**).

[`type_reduce`][bulanik.type2.type_reduce] works directly on points with interval weights
$w_i \in [\underline{w}_i, \overline{w}_i]$. The centroid interval is

$$
y_l = \min_{w} \frac{\sum_i y_i w_i}{\sum_i w_i}, \qquad
y_r = \max_{w} \frac{\sum_i y_i w_i}{\sum_i w_i},
$$

and the four methods are:

| Method | What it computes | Use it when |
|---|---|---|
| `"km"` (Karnik and Mendel 2001) | the exact $[y_l, y_r]$ by iterating a switch point over the $y$-sorted samples; output $(y_l + y_r)/2$ | you want the reference answer; the default |
| `"ekm"` (Wu and Mendel 2009) | the same endpoints, with a better starting switch point and incremental sums | many samples; the endpoints agree with `km` up to rounding (about 1 ulp) |
| `"nie_tan"` (Nie and Tan 2008) | closed form $\sum_i y_i(\underline{w}_i + \overline{w}_i) / \sum_i(\underline{w}_i + \overline{w}_i)$; `left == right == output` | you need speed and no iteration, and do not need the interval |
| `"bmm"` (Begian, Melek and Mendel 2008) | $m\,\bar{y}_\text{lower} + n\,\bar{y}_\text{upper}$, the weighted averages with the lower and with the upper weights; $m = n = 0.5$ by default | you want a cheap, smooth approximation; needs some positive lower weight |

On plateaus (equal $y$ or equal weights) `km` and `ekm` may stop at different, equally
optimal switch indices; the endpoints are the same. `bmm_m` and `bmm_n` have only to be
finite; neither sign nor $m + n = 1$ is enforced, so the output of `bmm` can fall outside
$[y_l, y_r]$ for unusual factors.

```python exec="on" source="above" result="text"
from bulanik.type2 import type_reduce

ys, lower, upper = [1, 2, 3], [0.2, 0.3, 0.1], [0.6, 0.8, 0.5]
for method in ("km", "ekm", "nie_tan", "bmm"):
    r = type_reduce(ys, lower, upper, method)
    print(f"{method:8s} [{r.left:.4f}, {r.right:.4f}]  output {r.output:.4f}  "
          f"switches {r.left_switch}, {r.right_switch}")
print(type_reduce(ys, [0.5] * 3, [0.5] * 3).output)   # no uncertainty: the plain weighted mean
```

`left_switch` and `right_switch` are the 1-based positions in the sorted samples where
the left end changes from upper to lower weights (right end: lower to upper); they are
`None` for the closed-form methods.

## The IT2 systems

[`IT2MamdaniFIS`][bulanik.type2.IT2MamdaniFIS] and
[`IT2SugenoFIS`][bulanik.type2.IT2SugenoFIS] use the same rule language and options as
their type-1 counterparts, plus `type_reduction` (default `"km"`).

- **IT2 Mamdani.** Each output universe is sampled at `resolution` points (default 1001).
  The upper and the lower aggregated sets are
  $\overline{\mu}(x_i) = \operatorname{agg}_k \operatorname{imp}(\overline{f}_k, \overline{\mu}_{B_k}(x_i))$ and
  $\underline{\mu}(x_i) = \operatorname{agg}_k \operatorname{imp}(\underline{f}_k, \underline{\mu}_{B_k}(x_i))$,
  and `type_reduce(xs, lower, upper, method)` gives the interval.
- **IT2 Sugeno.** Outputs are ordinary
  [`SugenoOutput`][bulanik.inference.sugeno.SugenoOutput] terms (constants or `Linear`),
  so the points $y_i = z_k$ are crisp and the weights are the firing intervals:
  `type_reduce([z_k], [f_lower_k], [f_upper_k])`.

`evaluate` returns the crisp outputs; `evaluate_interval` returns a
[`TypeReduced`][bulanik.type2.TypeReduced] per output (with `left`, `right`, `output` and
the switch points); `explain` returns the usual `Explanation`, whose `degrees` and rule
`activations` hold `(lower, upper)` pairs.

```python exec="on" source="above" result="text"
from bulanik.inference import SugenoOutput
from bulanik.type2 import (IT2LinguisticVariable, IT2MamdaniFIS, IT2SugenoFIS,
                           scaled_triangle)

x = IT2LinguisticVariable("x", (0, 10), {
    "lo": scaled_triangle(0, 0, 10, 0.5), "hi": scaled_triangle(0, 10, 10, 0.5)})
tip = IT2LinguisticVariable("tip", (0, 20), {
    "low": scaled_triangle(0, 0, 10, 0.6), "high": scaled_triangle(10, 20, 20, 0.6)})
mamdani = IT2MamdaniFIS([x], [tip], [
    "IF x IS lo THEN tip IS low", "IF x IS hi THEN tip IS high"], resolution=201)
r = mamdani.evaluate_interval({"x": 7})["tip"]
print(f"Mamdani  [{r.left:.2f}, {r.right:.2f}]  output {r.output:.3f}")

y = SugenoOutput("y", {"a": 0.0, "b": 10.0})
sugeno = IT2SugenoFIS([x], [y], ["IF x IS lo THEN y IS a", "IF x IS hi THEN y IS b"])
info = sugeno.explain({"x": 7})
print(info.degrees["x"])
print([a.strength for a in info.activations], info.outputs)
```

With no uncertainty (lower equal to upper everywhere) the IT2 system reproduces its
type-1 version exactly; this is a good sanity check when you build a model.

```python exec="on" source="above" result="text"
from bulanik.inference import SugenoFIS, SugenoOutput
from bulanik.sets import LinguisticVariable, gaussian
from bulanik.type2 import IT2LinguisticVariable, IT2SugenoFIS

t1 = LinguisticVariable("x", (0, 10), {"lo": gaussian(0, 2), "hi": gaussian(10, 2)})
y = SugenoOutput("y", {"a": 0.0, "b": 10.0})
rules = ["IF x IS lo THEN y IS a", "IF x IS hi THEN y IS b"]
print(SugenoFIS([t1], [y], rules).evaluate({"x": 3}))
print(IT2SugenoFIS([IT2LinguisticVariable.from_type1(t1)], [y], rules).evaluate({"x": 3}))
```

For the whole input-output map use [`plot.surface`][bulanik.plot.surface], which accepts
the IT2 systems too (and shows a gap where `NoRuleFiredError` or `NoLowerFiringError`
would be raised). The example [14 IT2 vs T1 noise](../examples/14-it2-vs-t1-noise.md)
compares an IT2 and a type-1 Sugeno approximator on noisy inputs; read its caveat: a
wider type-1 term is also less noise-sensitive, so IT2 is not a free robustness win.

## Defaults and the missing-rule cases

- **No rule fires** (every upper firing is 0): `default` is returned, or
  [`NoRuleFiredError`][bulanik.inference.rules.NoRuleFiredError] is raised, with
  `.explanation` as in type-1. An IT2 `default` is a number or a `(left, right)` pair (the
  interval to report; the output is its midpoint).
- **`type_reduction="bmm"` with no lower firing.** BMM needs some rule with a positive
  *lower* firing: the rules fire possibly (upper $> 0$) but nowhere with certainty. Then
  [`NoLowerFiringError`][bulanik.type2.NoLowerFiringError] (a `ValueError`) is raised,
  naming the output. Use `"km"`, `"ekm"` or `"nie_tan"` instead, or widen the lower sets.

```python exec="on" source="above" result="text"
from bulanik.inference import NoRuleFiredError, SugenoOutput
from bulanik.sets import triangular
from bulanik.type2 import IT2MF, IT2LinguisticVariable, IT2SugenoFIS, NoLowerFiringError

# wide upper bound, narrow lower bound: at x = 3 the rule can fire (upper > 0) but not for sure
mid = IT2MF(upper=triangular(2, 5, 8), lower=triangular(4, 5, 6))
x = IT2LinguisticVariable("x", (0, 10), {"mid": mid})
y = SugenoOutput("y", {"a": 1.0})
rules = ["IF x IS mid THEN y IS a"]

for point in (5.0, 3.0, 9.5):
    for method in ("km", "bmm"):
        fis = IT2SugenoFIS([x], [y], rules, type_reduction=method)
        try:
            print(point, method, fis.evaluate({"x": point}))
        except (NoRuleFiredError, NoLowerFiringError) as error:
            print(point, method, type(error).__name__)
print(x.fuzzify(5.0), x.fuzzify(3.0), x.fuzzify(9.5))
print(IT2SugenoFIS([x], [y], rules, default={"y": (0, 2)}).evaluate_interval({"x": 9.5}))
```

## Pitfalls

**Lower must stay below upper.** A hand-built `IT2MF` whose lower exceeds its upper at
some $x$ raises `ValueError` when evaluated, not at construction. The three builders
cannot violate it; for your own pairs, check with `plot.it2_set` or evaluate a grid.

**Error messages use internal kind names.** `gaussian_flat_top` is the upper bound of
`gaussian_uncertain_mean`, `gaussian_min_pair` its lower bound, and `scaled_triangular` the
lower bound of `scaled_triangle`.

**Parameter order is checked.** `gaussian_uncertain_mean` needs $m_1 \le m_2$ and
$\sigma > 0$; `gaussian_uncertain_sigma` needs $0 < \sigma_1 \le \sigma_2$;
`scaled_triangle` needs ordered points and a height in $(0, 1]$.

**Weights are validated by `type_reduce`.** Negative weights, lower above upper, unequal
lengths, non-finite values and all-zero upper weights raise `ValueError`; a `bool` or a
string is a `TypeError`.

**The output is a midpoint**, not a probability or a most-likely value; report
`evaluate_interval` when the width of $[y_l, y_r]$ matters, since it is the system's own
measure of how uncertain the answer is.

```python exec="on" source="above" result="text"
from bulanik.type2 import (IT2MF, gaussian_uncertain_mean, gaussian_uncertain_sigma,
                           scaled_triangle, type_reduce)
from bulanik.sets import triangular

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: IT2MF(upper=triangular(2, 5, 8), lower=triangular(0, 5, 10))(1))
show(lambda: gaussian_uncertain_mean(6, 4, 1))
show(lambda: gaussian_uncertain_sigma(5, 2, 1))
show(lambda: scaled_triangle(0, 5, 10, 1.5))
show(lambda: type_reduce([1, 2], [0.5], [0.6, 0.7]))
show(lambda: type_reduce([1, 2], [0.9, 0.1], [0.5, 0.7]))
show(lambda: type_reduce([1, 2], [0, 0], [0, 0]))
show(lambda: type_reduce([1, 2], [0, 0], [1, 1], "bmm"))
show(lambda: type_reduce([1, 2], [0, 0], [1, 1], "karnik"))
```

## See also

- [Mamdani inference](mamdani.md) and [Sugeno and Tsukamoto](sugeno-tsukamoto.md) for the
  type-1 systems that these generalise.
- [Sets and scales](sets-and-scales.md) for type-1 variables.
- API: [`bulanik.type2`][bulanik.type2], [`bulanik.plot`][bulanik.plot].
- Example: [14 IT2 vs T1 noise](../examples/14-it2-vs-t1-noise.md).
- Mendel (2001), *Uncertain Rule-Based Fuzzy Logic Systems: Introduction and New
  Directions*; Karnik and Mendel (2001), "Centroid of a type-2 fuzzy set", *Information
  Sciences*; Wu and Mendel (2009), "Enhanced Karnik-Mendel algorithms", *IEEE Transactions
  on Fuzzy Systems*; Nie and Tan (2008), "Towards an efficient type-reduction method for
  interval type-2 fuzzy logic systems", *FUZZ-IEEE*; Begian, Melek and Mendel (2008),
  "Stability analysis of type-2 fuzzy systems", *FUZZ-IEEE* (the BMM type reducer).
