# Interval-valued numbers: IVQROFN and IVNeutrosophicFN

An expert is rarely sure of a *degree* itself. "My support is somewhere between 0.4
and 0.5, and my opposition between 0.3 and 0.4." An **interval-valued** number keeps
that second layer of doubt: every degree of an
[orthopair](numbers-orthopair.md) or [three-part](numbers-threepart.md) number becomes
an interval. bulanik has two families:

| Class | Parts | Constraint |
|---|---|---|
| [`IVQROFN`][bulanik.numbers.IVQROFN] | membership $[\mu_L, \mu_U]$, non-membership $[\nu_L, \nu_U]$, rung $q$ | $\mu_U^q + \nu_U^q \le 1$ |
| [`IVIFN`][bulanik.numbers.IVIFN] | the same with $q = 1$ (Atanassov and Gargov 1989) | $\mu_U + \nu_U \le 1$ |
| [`IVPFN`][bulanik.numbers.IVPFN] | the same with $q = 2$ (Peng and Yang 2016) | $\mu_U^2 + \nu_U^2 \le 1$ |
| [`IVFFN`][bulanik.numbers.IVFFN] | the same with $q = 3$ (Jeevaraj 2021) | $\mu_U^3 + \nu_U^3 \le 1$ |
| [`IVNeutrosophicFN`][bulanik.numbers.IVNeutrosophicFN] | truth $[t_L, t_U]$, indeterminacy $[i_L, i_U]$, falsity $[f_L, f_U]$ (Wang et al. 2005) | none, each bound in $[0, 1]$ |

Every bound lies in $[0, 1]$, each lower bound is at most its upper bound, and for the
orthopair family only the **upper pair** carries the rung constraint. If the upper
corner is admissible, the whole rectangle is, because the lower pair is componentwise
smaller. `IVIFN`, `IVPFN` and `IVFFN` are `IVQROFN` with the rung fixed, so everything
below holds for all four. They are still separate classes: the same data in an alias and
in `IVQROFN` does not compare equal and does not mix in arithmetic. You build `IVIFN(mu_l, mu_u, nu_l, nu_u)` and
`IVQROFN(mu_l, mu_u, nu_l, nu_u, q)`.

<figure markdown="span">
  ![An interval-valued intuitionistic, Pythagorean and Fermatean number as rectangles over the q = 1, 2 and 3 feasible regions](../assets/figures/iv-orthopair-regions-light.svg#only-light)
  ![An interval-valued intuitionistic, Pythagorean and Fermatean number as rectangles over the q = 1, 2 and 3 feasible regions](../assets/figures/iv-orthopair-regions-dark.svg#only-dark)
  <figcaption markdown="span">A rectangle instead of a point: the pairs
  $(\mu, \nu)$ the number admits. Draw your own with
  [`plot.iv_orthopair`][bulanik.plot.iv_orthopair].</figcaption>
</figure>

```python exec="on" source="above" result="text"
from bulanik import IFN, IVFFN, IVIFN, IVPFN, IVQROFN

x = IVIFN(0.4, 0.5, 0.3, 0.4)
print(x, x.q)
print(tuple(round(h, 4) for h in x.hesitancy))        # (pi_L, pi_U): the interval of hesitancy
print(x.complement())                                 # membership and non-membership swapped
print(IVIFN.from_qrofn(IFN(0.6, 0.3)))                # a point is a degenerate interval
print(IVPFN(0.6, 0.8, 0.5, 0.6))                      # the rung q = 2 admits this pair
print(IVQROFN(0.6, 0.8, 0.3, 0.5, 3) == IVFFN(0.6, 0.8, 0.3, 0.5))  # False: same data, different class
```

## Operations: the point law at both ends

Every operation applies the point law of [`QROFN`][bulanik.numbers.QROFN] to the lower
pair $(a_L, b_L) = (\mu_L^q, \nu_L^q)$ and to the upper pair
$(a_U, b_U) = (\mu_U^q, \nu_U^q)$ **separately**, then takes the $q$-th root of each
result part. The two operator families, `ops="algebraic"` (the default, and what
`+ * **` use) and `ops="einstein"`, are the ones in the table of
[Orthopair numbers](numbers-orthopair.md#one-implementation-two-operator-families).
For $q = 1$ and the algebraic operators this is Xu's (2007) arithmetic, for example

$$
\tilde a \oplus \tilde b =
\bigl([a^L + b^L - a^L b^L,\; a^U + b^U - a^U b^U],\;
[c^L d^L,\; c^U d^U]\bigr).
$$

Each point law is monotone in every argument, so the lower bounds stay below the upper
ones and no operation can invert an interval. The upper pair carries the constraint and
is checked and projected exactly as `QROFN` does.

```python exec="on" source="above" result="text"
from bulanik import IVIFN, IVPFN

a, b = IVIFN(0.4, 0.5, 0.3, 0.4), IVIFN(0.5, 0.6, 0.2, 0.3)

def box(n):
    return f"mu [{n.mu_l:.4f}, {n.mu_u:.4f}]  nu [{n.nu_l:.4f}, {n.nu_u:.4f}]"

print("a + b             ", box(a + b))
print("a.add(b, einstein)", box(a.add(b, ops="einstein")))
print("a * b             ", box(a * b))
print("a.scale(2)        ", box(a.scale(2)))
print("a ** 2            ", box(a ** 2))
print("IVPFN ** 2        ", box(IVPFN(0.6, 0.8, 0.5, 0.6) ** 2))
print(IVIFN.OPS)
```

<figure markdown="span">
  ![Two interval-valued intuitionistic numbers A and B with their sum and product as rectangles](../assets/figures/iv-orthopair-operations-light.svg#only-light)
  ![Two interval-valued intuitionistic numbers A and B with their sum and product as rectangles](../assets/figures/iv-orthopair-operations-dark.svg#only-dark)
  <figcaption markdown="span">The algebraic sum moves toward high membership and low
  non-membership, the product the other way. See [`IVQROFN.add`][bulanik.numbers.IVQROFN.add]
  and [`IVQROFN.mul`][bulanik.numbers.IVQROFN.mul].</figcaption>
</figure>

### Scores, accuracy and distance

With $a = \mu^q$, $b = \nu^q$ and $\pi_L = 1 - a_U - b_U$, $\pi_U = 1 - a_L - b_L$
(powers, no root; the `hesitancy` property is the $q$-th root of these, $\pi^{1/q}$):

| Quantity | Names (default first) | Formula |
|---|---|---|
| score | `"difference"` | $S = \tfrac12(\mu_L^q + \mu_U^q - \nu_L^q - \nu_U^q)$ |
| | `"normalized"` | $(1 + S)/2$, a score in $[0, 1]$ |
| | `"defuzzified"` | $q = 1$ only, the defuzzified value of the interval-valued intuitionistic number (see [`score`][bulanik.numbers.IVQROFN.score]) |
| accuracy | `"sum"` | $H = \tfrac12(\mu_L^q + \mu_U^q + \nu_L^q + \nu_U^q)$ |
| distance | `"hamming"` | $\tfrac14 \sum \lvert\Delta\rvert$ over $(a_L, a_U, b_L, b_U, \pi_L, \pi_U)$ |
| | `"euclidean"` | $\sqrt{\tfrac14 \sum \Delta^2}$ over the same six |

The score is Xu's (2007) for $q = 1$ and Peng and Yang's (2016) for $q = 2$. Garg's
(2016) accuracy function for $q = 2$ is **not** implemented. For degenerate intervals
(lower equal to upper) every value equals the one the point number would give.

```python exec="on" source="above" result="text"
from bulanik import IVIFN

a, b = IVIFN(0.4, 0.5, 0.3, 0.4), IVIFN(0.5, 0.6, 0.2, 0.3)
print(round(a.score(), 4), round(a.score("normalized"), 4), round(a.accuracy(), 4))
print(round(a.distance(b), 4), round(a.distance(b, "euclidean"), 4))
print(IVIFN.SCORES, IVIFN.ACCURACIES, IVIFN.DISTANCES)
```

The default score is **signed** (it lies in $[-1, 1]$). A ratio method such as MARCOS
needs positive values, so rank orthopair data with `score="normalized"`; see
[More ranking methods](ranking-more.md).

## Interval neutrosophic numbers

An [`IVNeutrosophicFN`][bulanik.numbers.IVNeutrosophicFN]
$\langle [t_L, t_U], [i_L, i_U], [f_L, f_U] \rangle$ gives an interval of truth,
indeterminacy and falsity. As for the single-valued
[`NeutrosophicFN`][bulanik.numbers.NeutrosophicFN] there is **no joint constraint**.
Its only operator family is `"algebraic"`, the arithmetic of Zhang, Wang and Chen
(2014): the law of `NeutrosophicFN` on the lower triple and on the upper triple
separately.

| Operation | $T$ | $I$ | $F$ |
|---|---|---|---|
| `add` | $[t^L_1 + t^L_2 - t^L_1 t^L_2,\ t^U_1 + t^U_2 - t^U_1 t^U_2]$ | $[i^L_1 i^L_2,\ i^U_1 i^U_2]$ | $[f^L_1 f^L_2,\ f^U_1 f^U_2]$ |
| `mul` | $[t^L_1 t^L_2,\ t^U_1 t^U_2]$ | $[i^L_1 + i^L_2 - i^L_1 i^L_2,\ \dots]$ | $[f^L_1 + f^L_2 - f^L_1 f^L_2,\ \dots]$ |
| `scale` by $k$ | $[1 - (1 - t_L)^k,\ 1 - (1 - t_U)^k]$ | $[i_L^k,\ i_U^k]$ | $[f_L^k,\ f_U^k]$ |
| `power` by $k$ | $[t_L^k,\ t_U^k]$ | $[1 - (1 - i_L)^k,\ \dots]$ | $[1 - (1 - f_L)^k,\ \dots]$ |

The scoring is different from the orthopair family. The **only** score is `"average"`,
and there is no `"normalized"`:

$$
S = \tfrac16\left(4 + t_L + t_U - i_L - i_U - f_L - f_U\right),
$$

the `NeutrosophicFN` score $(2 + t - i - f)/3$ on the interval midpoints. Zhang et al.
(2014, Definition 18) define an interval-valued score
$[t_L + 2 - i_U - f_U,\ t_U + 2 - i_L - f_L]$; its midpoint is $3S$, so `"average"`
ranks exactly as that midpoint does. It lies in $[0, 1]$ and is $0$ only for the
number $\langle [0,0], [1,1], [1,1] \rangle$. The accuracy is `"difference"` (default)
$\tfrac12(t_L + t_U - f_L - f_U)$ or `"certainty"` $\tfrac12(t_L + t_U)$. The distances
`"hamming"` $\tfrac16 \sum \lvert\Delta\rvert$ and `"euclidean"`
$\sqrt{\tfrac16 \sum \Delta^2}$ run over the six bounds.

```python exec="on" source="above" result="text"
from bulanik import IVNeutrosophicFN

n = IVNeutrosophicFN(0.6, 0.7, 0.1, 0.2, 0.2, 0.3)
print(round(n.score(), 4), round(n.accuracy(), 4), round(n.accuracy("certainty"), 4))
print(n.complement())
s = n + n
print(tuple(round(v, 4) for v in (s.t_l, s.t_u, s.i_l, s.i_u, s.f_l, s.f_u)))
print(IVNeutrosophicFN.SCORES, IVNeutrosophicFN.ACCURACIES, IVNeutrosophicFN.OPS)
```

!!! note "A typo in the source"
    Zhang et al. (2014) print the addition and product with two typos (a $T$ in place of
    an $I$ index). The code follows the general t-norm forms of the same paper.

## Averaging opinions

`wa` and `wg` accept every kind on this page and apply the point law at both ends, so
an interval-valued rating of several experts aggregates like any other. See
[Aggregation](aggregation.md).

## Rounding and the boundary

A lower bound that exceeds its upper bound by up to $10^{-12}$ is clamped silently
(lower := upper); a larger inversion raises. The constraint on the upper pair is checked
and projected as for `QROFN`. The exact rules are in
[Conventions](../concepts/conventions.md#tolerance-and-boundary-projection).

## Pitfalls

**Bounds must be ordered.** $\mu_L \le \mu_U$ and $\nu_L \le \nu_U$ (and the same for
$t$, $i$, $f$). An inversion beyond $10^{-12}$ raises `ValueError` naming the fields.

**The rung constraint is on the upper pair.** `IVIFN(0.4, 0.7, 0.3, 0.5)` fails because
$0.7 + 0.5 > 1$, although its lower pair is fine. Pick the lowest rung that holds the
upper corner.

**Rungs do not mix.** `IVIFN + IVPFN` raises `TypeError`, as does a different `q`. The
same holds between an alias and `IVQROFN`: `IVQROFN(0.6, 0.8, 0.3, 0.5, 3) + IVFFN(0.6, 0.8, 0.3, 0.5)`
raises `TypeError` and the two compare unequal, although the data is identical. Stay in
one class.

**Scale and power need $k \ge 0$.**

**Use a positive-valued score for ratio methods.** The orthopair default `"difference"`
is signed, and `ahp_fuzzy_numbers` or the ratio ranking methods reject non-positive
values. For `IVIFN` and its relatives pass `score="normalized"`. `IVNeutrosophicFN` has
no `"normalized"` score; its single score `"average"` is already non-negative, and a
non-positive value gets the hint "use a positive-valued score".

```python exec="on" source="above" result="text"
from bulanik import IVIFN, IVNeutrosophicFN, IVPFN

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: IVIFN(0.5, 0.4, 0.1, 0.2))                      # mu_l > mu_u
show(lambda: IVIFN(0.4, 0.7, 0.3, 0.5))                      # upper pair: 0.7 + 0.5 > 1
show(lambda: IVIFN(0.4, 0.5, 0.3, 0.4) + IVPFN(0.4, 0.5, 0.3, 0.4))
show(lambda: IVIFN(0.4, 0.5, 0.3, 0.4).scale(-1))
show(lambda: IVIFN(0.4, 0.5, 0.3, 0.4).score("median"))
show(lambda: IVNeutrosophicFN(0.8, 0.7, 0.1, 0.2, 0.2, 0.3))
show(lambda: IVNeutrosophicFN(0.6, 0.7, 0.1, 0.2, 0.2, 0.3).score("normalized"))
```

## See also

- [Orthopair numbers](numbers-orthopair.md) and [Three-part numbers](numbers-threepart.md)
  for the point numbers these generalise.
- [Interval type-2 trapezoids](numbers-it2.md) for another way to hold uncertainty about
  a degree, [Aggregation](aggregation.md), and
  [Choosing a method](../concepts/choosing.md).
- [More ranking methods](ranking-more.md) and [More weighting methods](weighting-more.md)
  for what to do with the scores.
- API: [`IVQROFN`][bulanik.numbers.IVQROFN], [`IVIFN`][bulanik.numbers.IVIFN],
  [`IVPFN`][bulanik.numbers.IVPFN], [`IVFFN`][bulanik.numbers.IVFFN],
  [`IVNeutrosophicFN`][bulanik.numbers.IVNeutrosophicFN],
  [`plot.iv_orthopair`][bulanik.plot.iv_orthopair].
- Atanassov and Gargov (1989), "Interval valued intuitionistic fuzzy sets", *Fuzzy Sets
  and Systems* 31(3), 343-349; Xu (2007), *Control and Decision* 22(2), 215-219; Peng and
  Yang (2016), *International Journal of Intelligent Systems* 31(5), 444-487; Joshi,
  Singh, Bhatt and Vaisla (2018), *Journal of Intelligent & Fuzzy Systems* 35(5),
  5225-5230; Wang, Smarandache, Zhang and Sunderraman (2005), *Interval Neutrosophic Sets
  and Logic*, Hexis; Zhang, Wang and Chen (2014), *The Scientific World Journal* 2014,
  article 645953.
