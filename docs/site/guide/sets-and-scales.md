# Sets and scales

Two modules turn words and curves into things you can compute with.

- [`bulanik.sets`][bulanik.sets] holds **membership functions** (a curve
  $\mu: \mathbb{R} \to [0, 1]$), **linguistic variables** (a named quantity such
  as `temperature` with terms `cold`, `warm`, `hot`), and the fuzzy AND and OR.
  They are the inputs of [fuzzy inference](mamdani.md).
- [`bulanik.scales`][bulanik.scales] holds **linguistic scales**: ordered terms
  mapped to fuzzy *numbers*, so ratings written as words ("good", "very high")
  become data for [aggregation](aggregation.md) and MCDM.

The difference: a membership function is a curve over a crisp universe, used when
you fuzzify a measurement; a scale term is a single fuzzy number, used when
somebody rates something.

## Membership functions

Seven factories build a [`MembershipFunction`][bulanik.sets.MembershipFunction].
Calling it evaluates the membership of a crisp value; the result is always in
$[0, 1]$, even for huge or infinite arguments.

<figure markdown="span">
  ![The seven membership-function shapes: triangular, trapezoidal, gaussian, bell, sigmoid and the two linear ramps](../assets/figures/membership-shapes-light.svg#only-light)
  ![The seven membership-function shapes: triangular, trapezoidal, gaussian, bell, sigmoid and the two linear ramps](../assets/figures/membership-shapes-dark.svg#only-dark)
  <figcaption markdown="span">One panel per factory, drawn with
  [`bulanik.plot.variable`][bulanik.plot.variable].</figcaption>
</figure>

| Factory | Parameters | $\mu(x)$ |
|---|---|---|
| [`triangular`][bulanik.sets.triangular] | `a, b, c` | rises from 0 at $a$ to 1 at $b$, falls to 0 at $c$ (needs $a \le b \le c$, $a < c$) |
| [`trapezoidal`][bulanik.sets.trapezoidal] | `a, b, c, d` | rises to 1 at $b$, 1 on $[b, c]$, falls to 0 at $d$ ($a \le b \le c \le d$, $a < d$) |
| [`gaussian`][bulanik.sets.gaussian] | `mean, sigma` | $\exp\!\bigl(-(x - \text{mean})^2 / (2\sigma^2)\bigr)$, $\sigma > 0$ |
| [`bell`][bulanik.sets.bell] | `a, b, c` | $1 / (1 + \lvert (x - c)/a \rvert^{2b})$, $a, b > 0$ |
| [`sigmoid`][bulanik.sets.sigmoid] | `slope, center` | $1 / (1 + e^{-\text{slope}\,(x - \text{center})})$, slope $\ne 0$ |
| [`linear_increasing`][bulanik.sets.linear_increasing] | `a, b` | 0 up to $a$, 1 from $b$, linear between ($a < b$) |
| [`linear_decreasing`][bulanik.sets.linear_decreasing] | `a, b` | 1 up to $a$, 0 from $b$, linear between ($a < b$) |

A triangle here is a *function* on a universe; `TFN(l, m, u)` is a *number*. They
describe the same shape, and `TFN.mu` gives the same values, but only the number
supports fuzzy arithmetic.

Three more kinds, `gaussian_flat_top`, `gaussian_min_pair` and `scaled_triangular`,
are building blocks of interval type-2 sets; see [Interval type-2](type2.md).

The monotonic shapes (`linear_increasing`, `linear_decreasing`, `sigmoid`) have an
[`inverse`][bulanik.sets.MembershipFunction.inverse] ($x$ for a given degree);
for a sigmoid the level is clamped to $[10^{-9}, 1 - 10^{-9}]$ because the curve
never reaches 0 or 1. Triangles, trapezoids, Gaussians and bells are not
monotonic and have none.

```python exec="on" source="above" result="text"
from bulanik.sets import (bell, gaussian, linear_increasing, sigmoid,
                          trapezoidal, triangular)

tri = triangular(0, 5, 10)
print(tri(0), tri(2.5), tri(5), tri(7.5), tri(12))
print(trapezoidal(0, 1, 2, 3)(0.5), trapezoidal(0, 1, 2, 3)(1.5))
print(round(gaussian(5, 1)(6), 4), round(bell(2, 4, 6)(7), 4), sigmoid(1, 0)(0))
print(tri.kind, tri.params, tri.is_monotonic)
ramp = linear_increasing(0, 10)
print(ramp.inverse(0.25), ramp(ramp.inverse(0.25)))
print(sigmoid(1, 0).inverse(0.5), sigmoid(1, 0).inverse(1.0) > 20)   # clamped, not infinite
```

## Linguistic variables

A [`LinguisticVariable`][bulanik.sets.LinguisticVariable] gives a name, a universe
`(lo, hi)` and named terms. `fuzzify(x)` returns the membership of a crisp input
in every term; the universe is checked.

```python exec="on" source="above" result="text"
from bulanik.sets import LinguisticVariable, linear_decreasing, linear_increasing, triangular

temp = LinguisticVariable("temp", (0, 40), {
    "cold": linear_decreasing(10, 20),
    "warm": triangular(10, 20, 30),
    "hot": linear_increasing(20, 30),
})
print(temp, temp.term_names)
print(temp.fuzzify(15))                 # cold and warm, half each
print(temp.fuzzify(25))
print(temp["warm"](25))                 # one term, as a membership function
```

## t-norms and t-conorms

[`t_norm`][bulanik.sets.t_norm] is the fuzzy AND and
[`t_conorm`][bulanik.sets.t_conorm] the fuzzy OR of two degrees. Two of each exist:

| Function | `kind` (default first) | Formula |
|---|---|---|
| `t_norm(a, b, kind)` | `"min"`, `"prod"` | $\min(a, b)$, $ab$ |
| `t_conorm(a, b, kind)` | `"max"`, `"probor"` | $\max(a, b)$, $a + b - ab$ |

The inference engines use them for `AND` and `OR` in rules (options `and_method`,
`or_method`; see [Mamdani inference](mamdani.md)).

```python exec="on" source="above" result="text"
from bulanik.sets import t_conorm, t_norm

print(t_norm(0.3, 0.6), round(t_norm(0.3, 0.6, "prod"), 4))
print(t_conorm(0.3, 0.6), round(t_conorm(0.3, 0.6, "probor"), 4))
```

## Linguistic scales

A [`LinguisticScale`][bulanik.scales.LinguisticScale] maps term names to fuzzy
numbers of **one** type. Look terms up with `scale["term"]`, test with `in`, list
with `.terms`, and convert whole rating tables with
[`convert`][bulanik.scales.LinguisticScale.convert], which keeps the nesting of
your lists.

Built-in scales:

```python exec="on"
from bulanik import scales

print("| Scale | Type | Terms | Source |")
print("|---|---|---|---|")
for name in ("SAATY_TFN", "FUZZY_BWM_TFN", "EXAMPLE_IFN_5", "EXAMPLE_IFN_7", "EXAMPLE_TFN_7"):
    s = getattr(scales, name)
    print(f"| `{name}` | `{s.number_type.__name__}` | {', '.join(f'`{t}`' for t in s.terms)} | {s.source or '*illustrative, no source*'} |")
```

The three `EXAMPLE_*` scales are illustrations with made-up values. Use them to
try the library, but define your own scale (or cite a real one) for results you
report. The values of `EXAMPLE_TFN_7` and of the 7-term IFN scale, drawn:

<figure markdown="span">
  ![The seven TFN terms of EXAMPLE_TFN_7 on 0 to 10, with a rating of 6.5 belonging to MG at 0.75 and to F at 0.25](../assets/figures/linguistic-scale-tfn-light.svg#only-light)
  ![The seven TFN terms of EXAMPLE_TFN_7 on 0 to 10, with a rating of 6.5 belonging to MG at 0.75 and to F at 0.25](../assets/figures/linguistic-scale-tfn-dark.svg#only-dark)
</figure>

<figure markdown="span">
  ![The seven IFN terms of EXAMPLE_IFN_7 in the μ–ν plane, joined in scale order](../assets/figures/linguistic-scale-ifn-light.svg#only-light)
  ![The seven IFN terms of EXAMPLE_IFN_7 in the μ–ν plane, joined in scale order](../assets/figures/linguistic-scale-ifn-dark.svg#only-dark)
</figure>

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.scales import EXAMPLE_IFN_5, FUZZY_BWM_TFN, SAATY_TFN, LinguisticScale

print(SAATY_TFN)
print(SAATY_TFN["3"], "3" in SAATY_TFN)
print(SAATY_TFN.reciprocal("3"))               # TFN scales only: 1 / (2, 3, 4)
print(FUZZY_BWM_TFN["WI"])
print(EXAMPLE_IFN_5.convert([["H", "VL"], "M"]))   # any nesting; tuples become lists

mine = LinguisticScale("mine", {"low": TFN(0, 0, 5), "high": TFN(5, 10, 10)},
                       source="my survey")
print(mine.terms, mine.number_type.__name__, mine.source)
```

`SAATY_TFN` is Saaty's 1 to 9 scale (1980) as triangles
$(k - 1, k, k + 1)$ with the ends crisp; the $\pm 1$ spread is a common
convention, not Saaty's. `FUZZY_BWM_TFN` is the five-term best-worst scale of Guo
and Zhao (2017).

## Pitfalls

**Parameters are validated when the function is built**, so a bad curve never
exists: wrong order, a non-positive `sigma`, a zero `slope`, a wrong parameter
count when you use `MembershipFunction` directly.

**`fuzzify` stays inside the universe**, and a membership function rejects NaN.

**`inverse` needs a monotonic shape.**

**A scale holds one number type.** Mixing `TFN` and `IFN` terms raises
`TypeError`; an empty scale raises `ValueError`. `reciprocal` exists for
`TFN` scales only.

**Terms are case-sensitive**, and `convert` rejects anything that is not a string
or a list of them. A missing term raises `KeyError` that lists the valid ones.

```python exec="on" source="above" result="text"
from bulanik import IFN, TFN
from bulanik.scales import EXAMPLE_IFN_5, EXAMPLE_TFN_7, LinguisticScale
from bulanik.sets import (LinguisticVariable, MembershipFunction, gaussian, linear_increasing,
                          sigmoid, t_norm, triangular)

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: triangular(3, 2, 1))
show(lambda: gaussian(0, 0))
show(lambda: sigmoid(0, 1))
show(lambda: linear_increasing(5, 5))
show(lambda: MembershipFunction("triangular", (0, 1)))
show(lambda: MembershipFunction("hexagonal", (0, 1)))
show(lambda: triangular(0, 1, 2)(float("nan")))
show(lambda: triangular(0, 1, 2).inverse(0.5))
v = LinguisticVariable("service", (0, 10), {"good": triangular(5, 10, 10)})
show(lambda: v.fuzzify(11))
show(lambda: v["bad"])
show(lambda: LinguisticVariable("x", (5, 5), {"a": triangular(0, 1, 2)}))
show(lambda: t_norm(0.3, 1.2))
show(lambda: t_norm(0.3, 0.6, "lukasiewicz"))
show(lambda: LinguisticScale("mixed", {"a": TFN(0, 1, 2), "b": IFN(0.5, 0.4)}))
show(lambda: LinguisticScale("empty", {}))
show(lambda: EXAMPLE_TFN_7["good"])
show(lambda: EXAMPLE_TFN_7.convert(["g"]))
show(lambda: EXAMPLE_TFN_7.convert([5]))
show(lambda: EXAMPLE_IFN_5.reciprocal("H"))
```

## See also

- [Mamdani inference](mamdani.md) and [Sugeno and Tsukamoto](sugeno-tsukamoto.md),
  which consume linguistic variables.
- [Aggregation](aggregation.md) for what to do with converted ratings, and
  [Triangular numbers](numbers-triangular.md) for the number types.
- API: [`bulanik.sets`][bulanik.sets], [`bulanik.scales`][bulanik.scales].
- Examples: [02 scale and experts](../examples/02-scale-and-experts.md),
  [10 Mamdani tipper](../examples/10-mamdani-tipper.md).
