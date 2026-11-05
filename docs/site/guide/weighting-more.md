# More weighting methods: CRITIC, PIPRECIA, generic and interval-valued AHP, DEMATEL

[Weighting](weighting.md) covers AHP, Buckley fuzzy AHP, SWARA and entropy. This page adds
five ways to get weights, all returning a
[`WeightingResult`][bulanik.mcdm.results.WeightingResult] (or a result that converts to
one), so they plug straight into any [ranking method](ranking.md):

| Method | Function | Input | Kind |
|---|---|---|---|
| CRITIC | [`critic`][bulanik.mcdm.weighting.critic] | decision matrix | objective |
| PIPRECIA | [`piprecia`][bulanik.mcdm.weighting.piprecia] | relative importances, crisp or `TFN` | subjective |
| Generic fuzzy AHP | [`ahp_fuzzy_numbers`][bulanik.mcdm.weighting.ahp_fuzzy_numbers] | comparison matrix of any one fuzzy number kind | subjective |
| Interval-valued AHP | [`ahp_interval_valued`][bulanik.mcdm.weighting.ahp_interval_valued] | comparison matrix of one `IVQROFN` kind | subjective |
| DEMATEL | [`dematel`][bulanik.mcdm.dematel.dematel] | direct-influence matrix, crisp or fuzzy | structural |

## CRITIC

**CRITIC** (Diakoulaki, Mavrotas and Papayannakis 1995) is objective like entropy, but
rewards criteria for *conflict* with the others as well as for variation. Each column is
scaled to $[0, 1]$ with the best value at 1,
$r_{ij} = (x_{ij} - x^-_j) / (x^+_j - x^-_j)$, where $x^+_j$ is the largest value of a
benefit criterion and the smallest of a cost criterion. With the population standard
deviation $\sigma_j$ of $r_{\cdot j}$ and the Pearson correlation $\rho_{jk}$ of two
scaled columns, the information carried by criterion $j$ is

$$
C_j = \sigma_j \sum_k (1 - \rho_{jk}), \qquad w_j = \frac{C_j}{\sum_k C_k}.
$$

A criterion gets weight for varying a lot (large $\sigma_j$) and for conflicting with the
others (low $\rho_{jk}$). The weights do not depend on whether $\sigma_j$ uses $m$ or
$m - 1$ in the denominator (the factor cancels), and they are unchanged by a positive
scaling and a shift of any column. A **constant column** has no scale and no correlation:
it gets weight 0 and is left out of the other columns' correlation sums, so adding one
changes no other weight. `extras` holds `std`, `correlation` and `information`.

!!! note "Sourcing"
    CRITIC has **no published golden** in the tests. The method is checked against
    hand-derived cases (the derivations are in `tests/test_critic_piprecia.py`), not against
    a worked example from the paper.

```python exec="on" source="above" result="text"
from bulanik.mcdm import critic

data = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]
types = ["cost", "benefit", "benefit"]
r = critic(data, ["price", "quality", "delivery"], criteria_types=types)
print({k: round(v, 3) for k, v in r.weights.items()}, r.ranking)
print({k: round(v, 3) for k, v in r.extras["std"].items()})
print({k: round(v, 2) for k, v in r.extras["correlation"]["price"].items()})   # delivery conflicts with price

c = critic([[1, 5, 3], [2, 4, 3], [3, 3, 3]], ["a", "b", "constant"])
print({k: round(v, 3) for k, v in c.weights.items()})                         # the constant column weighs 0
```

## PIPRECIA

**PIPRECIA** (Stanujkic et al. 2017) is a subjective method like SWARA, with two
differences: the criteria **need not be sorted**, and the judgements are *ratios*. The
judgement $s_j$ for $j \ge 2$ compares criterion $j$ with the one before it: $s_j > 1$ if
$C_j$ is more important, $s_j = 1$ if equal, $s_j < 1$ if less. Then

$$
k_1 = 1,\; k_j = 2 - s_j, \qquad q_1 = 1,\; q_j = \frac{q_{j-1}}{k_j}, \qquad
w_j = \frac{q_j}{\sum_k q_k}.
$$

With `TFN` judgements the method is the fully fuzzy PIPRECIA of Stevic et al. (2018) with
the usual triangular arithmetic: $\tilde k_j = (2, 2, 2) - \tilde s_j$,
$\tilde q_j = \tilde q_{j-1} \oslash \tilde k_j$ and
$\tilde w_j = \tilde q_j \oslash \sum_k \tilde q_k$. The crisp `weights` are the fuzzy
weights scored with `score` (the graded mean $(l + 4m + u)/6$ by default, as in the source)
and normalised to sum 1, and `fuzzy_weights` holds the fuzzy ones. The valid domain is $0 < s_j < 2$ (a ratio of importances; a `TFN` needs
$s^l > 0$ and $s^u < 2$): a crisp $s_j \le 0$ or $s_j \ge 2$ raises, naming the index.

**The inverse pass.** With `s_inverse` the method follows Stevic et al. (2018) and
Dalic et al. (2020): the same criteria are judged again from the penultimate one
backwards. $s'_j$ compares criterion $j$ with criterion $j + 1$ ($j = 1, \dots, n - 1$,
so `s_inverse` has $n - 1$ entries in the order of `criteria`, not reversed). Then
$k'_n = 1$, $k'_j = 2 - s'_j$, $q'_n = 1$, $q'_j = q'_{j+1} / k'_j$ and
$w'_j = q'_j / \sum_k q'_k$. Each of $w_j$ and $w'_j$ is scored, and the final weight is
their **mean** $(w_j + w'_j)/2$. The source does not renormalise, and for fuzzy input that
mean need not sum to 1; bulanik **renormalises** the weights to sum 1, which leaves their
ranking unchanged. `extras` then holds `forward` and `inverse` (the scored vectors before
averaging, as printed in the source) and, for `TFN` input, `fuzzy_inverse`;
`fuzzy_weights` stays the forward pass.

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.mcdm import piprecia

names = ["a", "b", "c"]
r = piprecia([1.2, 0.9], names)                       # b is more important than a, c less than b
print({k: round(v, 3) for k, v in r.weights.items()}, r.operators)

both = piprecia([1.2, 0.9], names, s_inverse=[1.1, 0.7])
print({k: round(v, 3) for k, v in both.weights.items()}, both.operators["inverse"])
print({k: round(v, 3) for k, v in both.extras["forward"].items()},
      {k: round(v, 3) for k, v in both.extras["inverse"].items()})

f = piprecia([TFN(1.1, 1.15, 1.2), TFN(0.8, 0.9, 1.0)], names)
print({k: round(v, 3) for k, v in f.weights.items()}, f.operators["score"])
w = f.fuzzy_weights["b"]
print(round(w.l, 3), round(w.m, 3), round(w.u, 3))
```

## Generic fuzzy AHP

[`fuzzy_ahp`][bulanik.mcdm.weighting.fuzzy_ahp] is Buckley's method for *triangular*
numbers. [`ahp_fuzzy_numbers`][bulanik.mcdm.weighting.ahp_fuzzy_numbers] runs AHP over a
comparison matrix of **any one number kind and rung**: interval-valued, orthopair,
neutrosophic or interval type-2. Row $i$ is aggregated with equal weights $1/n$ by
[`wg`][bulanik.aggregate.wg] (the fuzzy geometric mean
$\tilde r_i = (\bigotimes_j \tilde a_{ij})^{1/n}$, the default) or
[`wa`][bulanik.aggregate.wa], the row number is scored, $s_i = \mathrm{score}(\tilde r_i)$,
and the weights are

$$
w_i = \frac{s_i}{\sum_k s_k}.
$$

This differs from Buckley. He first normalises *fuzzily*,
$\tilde w_i = \tilde r_i \otimes (\sum_k \tilde r_k)^{-1}$, and defuzzifies afterwards;
here the row number is defuzzified first and the crisp scores are normalised. The generic
form needs no fuzzy division, which is why it works for every kind. The matrix has order
at most 10, and `fuzzy_weights` holds the aggregated row numbers.

The **consistency ratio** is Saaty's. With `consistency_matrix` (typically the crisp Saaty
values of the linguistic terms) it is that matrix's ratio, recorded as
`"saaty_ri_on_given_matrix"`; otherwise, if every entry's score is positive, it is computed
on the matrix of entry scores (`"saaty_ri_on_scores"`); otherwise it is `None` (`"none"`).
The row scores must be positive, so an orthopair kind needs `score="normalized"`.

!!! warning "Use `score=\"normalized\"` for orthopair and interval-valued kinds"
    Their default score is the signed difference $\mu - \nu$, which is not a ratio scale.
    A signed row score raises, but whenever every row score *happens* to be positive it is
    silently used as a ratio scale, and the weights can differ a lot from those of
    `score="normalized"`. Pass `score="normalized"` for `IFN`, `PFN`, `IVIFN` and the like,
    and do not rely on the default.

```python exec="on" source="above" result="text"
from bulanik import IVIFN, TFN
from bulanik.mcdm import ahp_fuzzy_numbers

names = ["cost", "quality", "speed"]
m = [[TFN(1, 1, 1), TFN(2, 3, 4), TFN(4, 5, 6)],
     [TFN(1/4, 1/3, 1/2), TFN(1, 1, 1), TFN(1, 2, 3)],
     [TFN(1/6, 1/5, 1/4), TFN(1/3, 1/2, 1), TFN(1, 1, 1)]]
r = ahp_fuzzy_numbers(m, names)
print({k: round(v, 3) for k, v in r.weights.items()}, round(r.consistency_ratio, 3))
print(r.operators)

one = IVIFN(0.5, 0.5, 0.5, 0.5)                     # "equally important"
iv = [[one, IVIFN(0.7, 0.8, 0.1, 0.2)], [IVIFN(0.1, 0.2, 0.7, 0.8), one]]
try:
    ahp_fuzzy_numbers(iv, ["a", "b"])               # the default score is signed
except ValueError as error:
    print(error)
r = ahp_fuzzy_numbers(iv, ["a", "b"], score="normalized")
print({k: round(v, 3) for k, v in r.weights.items()}, r.operators["score"])
```

## Interval-valued AHP

[`ahp_interval_valued`][bulanik.mcdm.weighting.ahp_interval_valued] takes a comparison
matrix of one interval-valued orthopair kind, $([\mu^L, \mu^U], [\nu^L, \nu^U])$ of rung
$q$: [`IVIFN`][bulanik.numbers.IVIFN], [`IVPFN`][bulanik.numbers.IVPFN],
[`IVFFN`][bulanik.numbers.IVFFN] or a general [`IVQROFN`][bulanik.numbers.IVQROFN]. It
works on the interval endpoints directly rather than on a row number. Both triangles are
filled by the decision maker, reciprocity is not required, and the diagonal is taken as
given, so a scale's "equal" term may be whatever that scale says.

**`method="multiplicative"`** takes any $q$ and is the default for $q \ge 2$ (Karasan,
Ilbahar and Kahraman 2019; Alkan and Kahraman 2023 for $q = 3$; the form for a general $q$ is
this library's generalisation). With

$$
d^L_{ij} = (\mu^L_{ij})^q - (\nu^U_{ij})^q, \qquad d^U_{ij} = (\mu^U_{ij})^q - (\nu^L_{ij})^q,
$$

the interval multiplicative entries are $s^L_{ij} = 1000^{d^L_{ij}/q}$ and
$s^U_{ij} = 1000^{d^U_{ij}/q}$, the hesitancy factor is
$h_{ij} = 1 - ((\mu^U_{ij})^q - (\mu^L_{ij})^q) - ((\nu^U_{ij})^q - (\nu^L_{ij})^q)$, and

$$
t_{ij} = \frac{s^L_{ij} + s^U_{ij}}{2}\, h_{ij}, \qquad w_i = \frac{\sum_j t_{ij}}{\sum_{i,j} t_{ij}}.
$$

**`method="possibility"`** is defined for `IVIFN` ($q = 1$) only and is the default there
(interval weights, possibility degree and $w = \sum p + n/2 - 1$ as in Cebi and Ilbahar 2018 and,
for neutrosophic numbers, Bolturk and Kahraman 2018; both use powers of 10 where this function
uses $e$, which changes the weights). The
interval exponential matrix is $a^-_{ij} = e^{\mu^L_{ij} - \nu^U_{ij}}$,
$a^+_{ij} = e^{\mu^U_{ij} - \nu^L_{ij}}$, the interval weights come from the row sums,

$$
\tilde w_i = [w^-_i, w^+_i] = \left[\frac{\sum_j a^-_{ij}}{\sum_{i,j} a^+_{ij}},\;
\frac{\sum_j a^+_{ij}}{\sum_{i,j} a^-_{ij}}\right],
$$

and the possibility degree that $\tilde w_j \ge \tilde w_i$ is

$$
p_{ij} = \frac{\max(0, w^+_j - w^-_i) - \max(0, w^-_j - w^+_i)}{(w^+_i - w^-_i) + (w^+_j - w^-_j)},
\qquad w_j \propto \sum_i p_{ij} + \frac{n}{2} - 1.
$$

**Which one?**

| You have | Use | Why |
|---|---|---|
| A rung of 2 or more | `"multiplicative"` | The only one of the two defined there. |
| `IVIFN` judgements, and you want clear separation between criteria | `"multiplicative"` | The factor $1000^{d/q}$ spreads the weights widely. |
| `IVIFN` judgements, and you want interval weights or a flatter result | `"possibility"` | `extras["interval_weights"]` reports the intervals, and the weights stay closer together. |
| Any other number kind | [`ahp_fuzzy_numbers`](#generic-fuzzy-ahp) | Works for every kind through its score. |

The consistency ratio is Saaty's. With `consistency_matrix` it is that matrix's ratio;
otherwise, if every entry's default score is positive, it is computed on the matrix of
entry scores; otherwise it is `None`. `extras` holds `scored_matrix` and
`consistency_ratio`, and for the multiplicative method `d_lower`, `d_upper`, `s_lower`,
`s_upper`, `h` and `t`, for the possibility method `interval_weights` and `possibility`.

```python exec="on" source="above" result="text"
from bulanik import IVIFN, IVPFN
from bulanik.mcdm import ahp_interval_valued

names = ["cost", "quality", "speed"]
e, a, b = IVPFN(.45, .55, .45, .55), IVPFN(.65, .80, .20, .35), IVPFN(.20, .35, .65, .80)
r = ahp_interval_valued([[e, a, a], [b, e, a], [b, b, e]], names)        # q = 2: multiplicative
print({k: round(v, 3) for k, v in r.weights.items()}, r.operators)

e, a, b = IVIFN(.50, .50, .50, .50), IVIFN(.55, .65, .20, .35), IVIFN(.20, .35, .55, .65)
m = [[e, a, a], [b, e, a], [b, b, e]]
r = ahp_interval_valued(m, names)                                        # q = 1: possibility
print({k: round(v, 3) for k, v in r.weights.items()}, r.operators)
print([(round(lo, 3), round(hi, 3)) for lo, hi in r.extras["interval_weights"]])
r = ahp_interval_valued(m, names, method="multiplicative")
print({k: round(v, 3) for k, v in r.weights.items()}, r.operators)
```

[Example 23](../examples/23-interval-valued-ahp.md) compares the methods on one set of
judgements.

## DEMATEL

**DEMATEL** (Gabus and Fontela 1972) does not weigh criteria by preference. It asks how
strongly each factor *influences* every other, and from the network of influences finds
which factors are **causes** (they push others) and which are **effects** (they are
pushed). Take a direct-relation matrix $A$, where $a_{ij}$ says how strongly factor $i$
influences factor $j$. The normalised matrix and the **total-relation** matrix are

$$
N = A / s, \qquad T = N (I - N)^{-1},
$$

with $s = \max(\max_i \sum_j a_{ij},\ \max_j \sum_i a_{ij})$ (`normalization="max_row_col"`,
the default) or $s = \max_i \sum_j a_{ij}$ (`"max_row"`). The diagonal of $A$ is used as
given. The row sums $D$ and column sums $R$ of $T$ give each factor's

- **prominence** $D + R$: how involved the factor is overall, and
- **relation** $D - R$: its net influence.

A factor is a **cause** when $D_i - R_i > 10^{-12} \max_k (D_k + R_k)$ and an **effect**
otherwise, so rounding noise around an exact zero net relation counts as an effect, as an
exact zero does. The result lists `relations`, the off-diagonal $t_{ij}$ above a
`threshold` (the mean of all $n^2$ entries of $T$ by default), sorted row-major.

**Weights.** `weighting="sqrt"` (the default) gives
$w_i \propto \sqrt{(D_i + R_i)^2 + (D_i - R_i)^2}$, the length of the factor's vector in the
cause-effect plane; `"prominence"` gives $w_i \propto D_i + R_i$. The square-root form is
**a common convention** in DEMATEL-based weighting, not a formula of the original method;
if you need the plain prominence, ask for it.
[`DEMATELResult.as_weights()`][bulanik.mcdm.results.DEMATELResult.as_weights] returns the
weights as a `WeightingResult`, with `prominence` and `relation` in its `extras`, usable
as the criteria weights of every ranking method.

The example is the 5 by 5 matrix of the **vignette of the R package `dematel`**, which the
test suite also uses as a cross-check against the printed values.

```python exec="on" source="above" result="text"
from bulanik.mcdm import dematel

A = [[0, 2, 4, 3, 4],
     [3, 0, 2, 0, 0],
     [3, 3, 0, 0, 0],
     [3, 0, 4, 0, 2],
     [4, 0, 2, 0, 0]]
r = dematel(A)
print(r.causes, r.effects)
print({k: round(v, 3) for k, v in r.prominence.items()})
print({k: round(v, 3) for k, v in r.relation.items()})
print({k: round(v, 3) for k, v in r.weights.items()}, r.ranking)
print(round(r.threshold, 3), len(r.relations), r.relations[0][:2])
print(r.operators)

p = dematel(A, weighting="prominence")
print({k: round(v, 3) for k, v in p.weights.items()})
w = r.as_weights()
print(w.method, w.ranking == r.ranking, sorted(w.extras))
```

<figure markdown="span">
  ![DEMATEL cause-effect diagram of five factors, with the strongest influences as arrows](../assets/figures/dematel-cause-effect-light.svg#only-light)
  ![DEMATEL cause-effect diagram of five factors, with the strongest influences as arrows](../assets/figures/dematel-cause-effect-dark.svg#only-dark)
  <figcaption markdown="span">The matrix above drawn by [`plot.dematel`][bulanik.plot.dematel]:
  each factor sits at its prominence $D + R$ and relation $D - R$ (above the line, a net
  giver of influence), and the arrows are the total influences above 0.45 (the default
  threshold, the mean of $T$ at 0.33, would draw 13).</figcaption>
</figure>

### Fuzzy and crisp input

The matrix may hold fuzzy numbers of one kind; they are defuzzified first
(`defuzzification=`). For **triangular numbers** the default is **CFCS** (converting fuzzy
data into crisp scores) of Opricovic and Tzeng (2003), also available as
[`cfcs`][bulanik.mcdm.dematel.cfcs]. For cell $(i, j)$ with expert ratings
$(l^k, m^k, r^k)$, $k = 1..K$, let $\min = \min_k l^k$, $\Delta = \max_k r^k - \min$ and,
for each expert, $x_l = (l^k - \min)/\Delta$ (likewise $x_m$, $x_r$),
$x^{ls} = x_m/(1 + x_m - x_l)$, $x^{rs} = x_r/(1 + x_r - x_m)$,

$$
x = \frac{x^{ls}(1 - x^{ls}) + x^{rs}x^{rs}}{1 - x^{ls} + x^{rs}}, \qquad z^k = \min + x\Delta,
$$

and the cell value is the mean of $z^k$.

!!! note "Scope of CFCS"
    As in Opricovic and Tzeng (2003), the minimum and maximum are taken **per cell across the
    experts**, not over the whole matrix. Each expert's crisp value is computed first, and
    the experts are averaged *after* the CFCS step. With a single expert the cell's own $l$
    and $r$ are used; if $\Delta = 0$ then $z = \min$.

Any other kind is defuzzified with a score of the kind (the kind's default unless you name
one). Crisp input accepts only `defuzzification=None` or `"crisp"`.

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.mcdm import dematel
from bulanik.mcdm.dematel import cfcs

z = TFN(0, 0, 0)
expert1 = [[z, TFN(1, 2, 4)], [z, z]]
expert2 = [[z, TFN(2, 3, 4)], [z, z]]
print(round(cfcs([expert1])[0][1], 4))                     # one expert: the cell's own l and r
print(round(cfcs([expert1, expert2])[0][1], 4))            # two experts: min and max per cell, then mean

fuzzy = [[z, TFN(1, 2, 4), TFN(2, 3, 4)], [TFN(0, 1, 2), z, TFN(1, 2, 3)], [TFN(3, 4, 4), TFN(0, 1, 2), z]]
r = dematel(fuzzy)
print(r.operators["defuzzification"], r.causes, {k: round(v, 3) for k, v in r.weights.items()})
print(dematel(fuzzy, defuzzification="centroid").operators["defuzzification"])
```

## Pitfalls

**DEMATEL needs `I - N` to be invertible.** If the normalised matrix has spectral radius
1, which happens when every row sum equals the normalising maximum (for example
`[[0, 1], [1, 0]]`), the total relation is undefined and the call raises `ValueError`
saying so, never a meaningless huge $T$. An all-zero matrix, a negative entry, a non-square
matrix or fewer than 2 factors also raise.

**Fuzzy scores can be negative.** After defuzzification every entry must be $\ge 0$.
Orthopair kinds default to a signed score, so pass `defuzzification="normalized"`; a kind
without that score gets the hint "use a positive-valued score".

**Influence is not preference.** DEMATEL weights say how *central* a factor is in the
network. Whether that is the right notion of importance for your decision is your call,
and the `"sqrt"` weighting is a convention.

**CRITIC needs variation and conflict.** At least two alternatives are required. If every
column is constant there is nothing to weigh, and if the informative columns are all
perfectly positively correlated (for example 2 alternatives and no conflict), every $C_j$
is 0; both raise `ValueError`.

**PIPRECIA judgements are $n - 1$ ratios.** The count must match, $s_j \ge 2$ is
rejected, and crisp and `TFN` judgements cannot be mixed, nor can the forward and inverse
sets.

**`ahp_fuzzy_numbers` needs positive row scores**, so give an orthopair kind
`score="normalized"`.

**The possibility method needs $q = 1$.** On any other rung it raises `ValueError`; the
multiplicative method takes any rung. The default also follows the rung, so state `method`
when you compare runs.

**The two interval-valued methods are on different scales of spread.** The weights of one
are not interchangeable with those of the other: the factor $1000^{d/q}$ of the
multiplicative method separates criteria far more than the possibility degrees do, and
the order can be the same while the weights are not.

**`ahp_interval_valued` takes the matrix as given.** It does not fill the lower triangle
and does not force a diagonal, so write the whole matrix. The ratio on the scored matrix
is indicative only when the judgements are not reciprocal; pass the crisp Saaty matrix as
`consistency_matrix` for the usual one. The order is limited to 2 to 10.

```python exec="on" source="above" result="text"
from bulanik import IFN, TFN
from bulanik.mcdm import critic, dematel, piprecia

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: dematel([[0, 1], [1, 0]]))                             # I - N is singular
show(lambda: dematel([[0, 0], [0, 0]]))
show(lambda: dematel([[0, -1], [1, 0]]))
show(lambda: dematel([[0, 1, 2], [1, 0, 1]]))
show(lambda: dematel([[0, 2], [1, 0]], weighting="sum"))
show(lambda: dematel([[0, 2], [1, 0]], defuzzification="cfcs"))    # crisp input: no defuzzification
ifn = [[IFN(0.5, 0.4), IFN(0.5, 0.4)], [IFN(0.1, 0.4), IFN(0.5, 0.4)]]
show(lambda: dematel(ifn))                                          # signed default score
show(lambda: dematel(ifn, defuzzification="normalized").causes)
show(lambda: critic([[1, 2], [1, 2]]))                              # every column constant
show(lambda: critic([[1, 2], [2, 4]]))                              # perfectly correlated
show(lambda: piprecia([2.0], ["a", "b"]))                           # k = 2 - s must be positive
show(lambda: piprecia([1.1], ["a", "b", "c"]))                      # needs n - 1 = 2 judgements
show(lambda: piprecia([1.1, TFN(0.8, 0.9, 1.0)], ["a", "b", "c"]))
```

## See also

- [Weighting](weighting.md) for AHP, fuzzy AHP, SWARA and entropy, and
  [Ranking](ranking.md) and [More ranking methods](ranking-more.md) for what to do with
  the weights.
- [Choosing a method](../concepts/choosing.md) for when to use which.
- [Interval-valued numbers](numbers-interval.md) and
  [Interval type-2 trapezoids](numbers-it2.md) for the kinds `ahp_fuzzy_numbers` and
  `dematel` accept; [Aggregation](aggregation.md).
- API: [`bulanik.mcdm.weighting`][bulanik.mcdm.weighting],
  [`bulanik.mcdm.dematel`][bulanik.mcdm.dematel],
  [`DEMATELResult`][bulanik.mcdm.results.DEMATELResult],
  [`plot.dematel`][bulanik.plot.dematel].
- Diakoulaki, Mavrotas and Papayannakis (1995), *Computers & Operations Research* 22(7),
  763-770 (CRITIC); Stanujkic, Zavadskas, Karabasevic, Smarandache and Turskis (2017),
  *Romanian Journal of Economic Forecasting* 20(4), 116-133, Stevic, Stjepanovic,
  Bozickovic, Das and Stanujkic (2018), *Symmetry* 10(11), 586, and Dalic, Stevic,
  Karamasa and Puska (2020), *Decision Making: Applications in Management and Engineering*
  3(1), 80-95 (PIPRECIA); Saaty (1980), *The Analytic Hierarchy Process*, McGraw-Hill, and
  Kahraman, Oztaysi, Ucal Sari and Turanoglu (2014), *Knowledge-Based Systems* 59, 48-57
  (generic and interval type-2 AHP); Karasan, Ilbahar and Kahraman (2019), *Soft Computing* 23(21),
  10953-10968, Alkan and Kahraman (2023), *Informatica* 34(1), 1-33, Cebi and Ilbahar (2018),
  *International Journal of the Analytic Hierarchy Process* 10(2), and Bolturk and Kahraman
  (2018), the same journal, 10(3) (interval-valued AHP); Gabus and Fontela (1972), *World Problems, an
  Invitation to Further Thought within the Framework of DEMATEL*, Battelle Geneva Research
  Centre, Opricovic and Tzeng (2003), *International Journal of Uncertainty, Fuzziness and
  Knowledge-Based Systems* 11(5), 635-652, and Wu and Lee (2007), *Expert Systems with
  Applications* 32(2), 499-507 (fuzzy DEMATEL with CFCS).
