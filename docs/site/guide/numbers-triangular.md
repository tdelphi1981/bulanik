# Triangular and trapezoidal numbers

A **triangular fuzzy number** ([`TFN`][bulanik.TFN]) is a vague quantity such as
"about 4, certainly between 2 and 7". It is three numbers $(l, m, u)$: the
lower foot, the peak and the upper foot. A **trapezoidal fuzzy number**
([`TrFN`][bulanik.TrFN]) has four, $(a, b, c, d)$: a plateau $[b, c]$ of fully
plausible values instead of a single peak. Use them when your input is a
measurement with a spread, an interval estimate, or a linguistic rating such as
"good" that you have mapped to a range.

## The idea

The membership degree $\mu(x) \in [0, 1]$ says how well the crisp value $x$
fits the number. For a triangle it rises linearly from 0 at $l$ to 1 at $m$ and
falls back to 0 at $u$. For a trapezoid it stays 1 on the whole plateau.

<figure markdown="span">
  ![Membership functions of TFN(2, 4, 7) and TrFN(1, 3, 5, 8), with support, core and the α-cut at 0.5 marked](../assets/figures/tfn-anatomy-light.svg#only-light)
  ![Membership functions of TFN(2, 4, 7) and TrFN(1, 3, 5, 8), with support, core and the α-cut at 0.5 marked](../assets/figures/tfn-anatomy-dark.svg#only-dark)
  <figcaption markdown="span">The **support** is where $\mu > 0$, the **core** is where
  $\mu = 1$, and the **α-cut** at level α is the interval where $\mu \ge \alpha$.</figcaption>
</figure>

Two rules govern the points, and the constructor checks both: every point is a
finite real number (a `bool` or a string is a `TypeError`), and the points are
ordered, $l \le m \le u$ or $a \le b \le c \le d$. Equal points are allowed, so
`TFN(5, 5, 5)` is the crisp number 5 and `TFN(0, 0, 5)` has a vertical left edge.

```python exec="on" source="above" result="text"
from bulanik import TFN, TrFN

a = TFN(1, 2, 4)
t = TrFN(1, 2, 3, 5)
print(a, t, sep="\n")                  # points are stored as floats
print(a.l, a.m, a.u)
print(a.mu(1.5), a.mu(3), a.mu(9))     # membership: rising edge, falling edge, outside
print(t.mu(2.5))                       # on the plateau
print(a.to_trfn())                     # a TFN is a trapezoid with a one-point core
print(TFN(5, 5, 5).score())            # a crisp number is a degenerate triangle
```

## α-cuts

The α-cut $\{x : \mu(x) \ge \alpha\}$ of a triangle or trapezoid is always an
interval. With support $[p_0, p_n]$ and core $[c_\text{lo}, c_\text{hi}]$ the code
computes

$$
\bigl[p_0 + \alpha\,(c_\text{lo} - p_0),\;\; p_n - \alpha\,(p_n - c_\text{hi})\bigr].
$$

At $\alpha = 0$ this is the support and at $\alpha = 1$ the core. α-cuts are the
bridge to interval arithmetic: fuzzy arithmetic is interval arithmetic carried out
at every level.

```python exec="on" source="above" result="text"
from bulanik import TFN, TrFN

a = TFN(1, 2, 4)
for alpha in (0, 0.5, 1):
    print(alpha, a.alpha_cut(alpha))
print(TrFN(1, 2, 3, 5).alpha_cut(1))   # the plateau
```

## Arithmetic

The operators `+ - * /`, `**` and unary `-` use the only operator family,
`"algebraic"`; the methods `add`, `sub`, `mul`, `scale`, `power` and
`reciprocal` do the same and take an `ops=` argument that only accepts that name.
Every operation returns a new number of the same type and combines `TFN` with
`TFN` and `TrFN` with `TrFN`. The code is the standard *approximate* arithmetic on
the defining points:

| Operation | Result, for a triangle $(l_1, m_1, u_1)$ and $(l_2, m_2, u_2)$ |
|---|---|
| `A + B` | $(l_1 + l_2,\; m_1 + m_2,\; u_1 + u_2)$ |
| `A - B` | $(l_1 - u_2,\; m_1 - m_2,\; u_1 - l_2)$: each point against the *opposite* point of $B$ |
| `A * B` | outer points are the smallest and largest of the four products $l_1 l_2,\, l_1 u_2,\, u_1 l_2,\, u_1 u_2$; the inner points are the products of the inner points, sorted. For non-negative numbers this is $(l_1 l_2,\; m_1 m_2,\; u_1 u_2)$ |
| `k * A`, `A * k`, `A / k` | every point times $k$ (or $1/k$); for $k < 0$ the points are also reversed so the order is kept |
| `A ** k` | every point to the power $k$; reversed for $k < 0$; needs the lowest point $\ge 0$ (and $> 0$ for $k < 0$) |
| `A.reciprocal()` | $(1/u,\; 1/m,\; 1/l)$; the support must not contain 0 |
| `A / B` | `A * B.reciprocal()` |

A trapezoid works the same way on its four points. The sum and the difference are
exact in the sense of α-cuts (the cut of $A + B$ is the sum of the cuts). The
product and the quotient are **approximations**: the exact product of two
triangles has curved sides, and `A * B` returns the triangle through its corner
products. The two agree at $\alpha = 0$ and $\alpha = 1$ but not in between, and
the triangle is not simply wider or narrower: for the example below, at
$\alpha = 0.5$ the exact cut is $[3.75, 12]$ and the triangle's is $[4, 13]$, so it
is inside at the lower end and outside at the upper end.

<figure markdown="span">
  ![α-cuts of A = TFN(1, 2, 4), B = TFN(2, 3, 5), A + B and A × B, stacked on one x axis](../assets/figures/alpha-cut-arithmetic-light.svg#only-light)
  ![α-cuts of A = TFN(1, 2, 4), B = TFN(2, 3, 5), A + B and A × B, stacked on one x axis](../assets/figures/alpha-cut-arithmetic-dark.svg#only-dark)
  <figcaption markdown="span">Each row shows five α-cuts. The sum is exact. The outline
  on the last row is the exact product; the returned triangle matches it at
  $\alpha = 0$ and $\alpha = 1$ and differs in between (at $\alpha = 0.5$: $[4, 13]$
  against the exact $[3.75, 12]$).</figcaption>
</figure>

```python exec="on" source="above" result="text"
from bulanik import TFN

a, b = TFN(1, 2, 4), TFN(2, 3, 5)
print("a + b  =", a + b)
print("a - b  =", a - b)
print("a * b  =", a * b)
print("a / b  =", a / b)
print("2 * a  =", 2 * a, "| -a =", -a)             # a negative factor reverses the points
print("a ** 2 =", a ** 2)
print("1 / a  =", a.reciprocal())
print(a.add(b, ops="algebraic"))                   # the only valid name
```

## Scores, defuzzification and distance

To rank numbers you need one crisp value per number. `score(method)` (and its
synonym `defuzzify(method)`) offers three, the default being the centroid:

| Method | Triangle $(l, m, u)$ | Trapezoid $(a, b, c, d)$ |
|---|---|---|
| `"centroid"` (default) | $(l + m + u)/3$ | $\dfrac{d^2 + c^2 + cd - a^2 - b^2 - ab}{3\,(d + c - a - b)}$ |
| `"graded_mean"` | $(l + 4m + u)/6$ | $(a + 2b + 2c + d)/6$ |
| `"mean"` | $(l + m + u)/3$, the same as the centroid | $(a + b + c + d)/4$ |

<figure markdown="span">
  ![TFN(1, 2, 7) with stems at its graded mean 2.667 and its centroid and mean 3.333](../assets/figures/defuzzification-light.svg#only-light)
  ![TFN(1, 2, 7) with stems at its graded mean 2.667 and its centroid and mean 3.333](../assets/figures/defuzzification-dark.svg#only-dark)
  <figcaption markdown="span">For a skewed triangle the methods disagree. The graded mean
  weights the peak four times, so it stays closer to $m$.</figcaption>
</figure>

There are three distances. The default, `"vertex"` (Chen 2000), is the root mean
square of the point differences,

$$
d(A, B) = \sqrt{\tfrac{1}{n} \sum_{i=1}^{n} (x_i - y_i)^2},
$$

with $n = 3$ for triangles and $n = 4$ for trapezoids. Fuzzy TOPSIS
([`fuzzy_topsis`][bulanik.mcdm.fuzzy_topsis]) uses it. `"euclidean"` and
`"hamming"` are the weighted distances of Li (2007) that fuzzy CODAS
([`fuzzy_codas`][bulanik.mcdm.fuzzy_ranking.fuzzy_codas]) uses. For trapezoids
$M = (m_1, m_2, m_3, m_4)$ and $N = (n_1, n_2, n_3, n_4)$ the core points count twice,

$$
d_E(M, N) = \sqrt{\tfrac16\left[(m_1 - n_1)^2 + 2(m_2 - n_2)^2 + 2(m_3 - n_3)^2 + (m_4 - n_4)^2\right]},
\qquad
d_H(M, N) = \tfrac16\left[|m_1 - n_1| + 2|m_2 - n_2| + 2|m_3 - n_3| + |m_4 - n_4|\right],
$$

and a triangle $(l, m, u)$ counts as the trapezoid $(l, m, m, u)$, so its peak
weighs $4/6$: $d_H = \tfrac16(|\Delta l| + 4|\Delta m| + |\Delta u|)$. A `TFN` and
its `to_trfn()` have the same Euclidean and Hamming distances. Triangular numbers have
**no accuracy function**: ties between equal scores are not broken by a second
quantity.

```python exec="on" source="above" result="text"
from bulanik import TFN, TrFN

a = TFN(1, 2, 7)
for method in TFN.SCORES:
    print(f"{method:12s} {a.score(method):.4f}")
print(a.score() == a.score("centroid"), a.defuzzify("graded_mean") == a.score("graded_mean"))

t = TrFN(1, 2, 3, 5)
print({m: round(t.score(m), 4) for m in TrFN.SCORES})
print(TFN(1, 2, 4).distance(TFN(2, 3, 5)))           # sqrt(((1-2)^2 + (2-3)^2 + (4-5)^2) / 3)
a, b = TFN(1, 2, 3), TFN(2, 3, 5)
print(TFN.DISTANCES, round(a.distance(b, "euclidean"), 4), round(a.distance(b, "hamming"), 4))
```

## Drawing them

[`bulanik.plot`][bulanik.plot] (the `plot` extra) draws membership functions,
α-cuts and defuzzification markers; the figures above come from it. The
[gallery](gallery.md) shows more.

```python exec="on" html="1"
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bulanik import TFN, TrFN, plot

for mode in ("light", "dark"):
    with plot.theme(mode):
        ax = plot.membership([TFN(1, 2, 4), TrFN(2, 3, 5, 7)], labels=["A", "B"], alpha_cut=0.5)
        buffer = io.StringIO()
        ax.figure.savefig(buffer, format="svg")
        plt.close(ax.figure)
    print(f'<div class="bulanik-fig bulanik-fig-{mode}">{buffer.getvalue()}</div>')
```

## Pitfalls

**Points must be ordered, finite and real numbers.** Swapped points, NaN, a `bool`
or a string are rejected when the number is built, so a bad `TFN` never exists.

**Numbers of different types do not mix.** A `TFN` is not a `TrFN`, and neither
is an `IFN`. Convert a triangle with `to_trfn()`. A plain number is not a fuzzy
number either: `TFN + 1` fails, so write `TFN + TFN(1, 1, 1)`. Only
multiplication, division and `**` accept a plain number.

**Reciprocal, division and powers need a safe support.** The support, endpoints
included, must not contain 0 for `reciprocal()` and `/`, and the lowest point must be non-negative
for `**`.

**Multiplication is an approximation.** `A * B` is not the exact product of the
fuzzy sets, only the standard point-based one (see the figure above). If you need
exact products, work on α-cuts with interval arithmetic yourself.

**Score names and accuracy.** `accuracy()` always raises, and an unknown score
name lists the valid ones.

```python exec="on" source="above" result="text"
from bulanik import IFN, TFN, TrFN

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: TFN(3, 2, 1))                      # points out of order
show(lambda: TFN(1, 2, float("inf")))           # not finite
show(lambda: TFN(True, 2, 3))                   # a bool is not a number
show(lambda: TFN(1, 2, 3) + TrFN(1, 2, 3, 4))   # different types
show(lambda: TFN(1, 2, 3) + IFN(0.5, 0.4))
show(lambda: TFN(1, 2, 3) + 1)                  # a plain number is not a fuzzy number
show(lambda: TFN(-1, 0, 1).reciprocal())        # the support contains 0
show(lambda: TFN(0, 1, 2).reciprocal())         # a zero endpoint counts too
show(lambda: TFN(1, 2, 3) / TFN(-1, 0, 1))
show(lambda: TFN(-1, 2, 4) ** 2)                # negative lowest point
show(lambda: TFN(1, 2, 3).score("median"))
show(lambda: TFN(1, 2, 3).accuracy())
show(lambda: TFN(1, 2, 3).add(TFN(1, 2, 3), ops="einstein"))
```

## See also

- [Conventions](../concepts/conventions.md): validation, immutability and the
  naming rule behind `ops=` and `score(method)`.
- [Choosing a method](../concepts/choosing.md): when triangles are enough and when
  an orthopair or three-part number fits better.
- [Aggregation](aggregation.md) to combine several triangles, and
  [Sets and scales](sets-and-scales.md) for linguistic terms.
- [Fuzzy TOPSIS](fuzzy-topsis.md), which ranks triangles with the vertex distance.
- API: [`TFN`][bulanik.TFN], [`TrFN`][bulanik.TrFN], [`bulanik.numbers`][bulanik.numbers].
- Examples: [01 fuzzy numbers](../examples/01-fuzzy-numbers.md) and
  [03 fuzzy AHP](../examples/03-fuzzy-ahp.md).
