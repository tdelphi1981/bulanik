# Interval type-2 trapezoids: IT2TrFN

A trapezoid says "roughly between 4 and 6, never below 2 or above 9". An **interval
type-2 trapezoidal number** admits that even this description is vague: instead of one
trapezoid it carries two, an **upper** and a **lower** membership function, and the
region between them is the *footprint of uncertainty*. Use it when you want a
linguistic term ("high") to be uncertain about its own shape, as in the interval type-2
TOPSIS and AHP literature.

This page is about the *number* [`IT2TrFN`][bulanik.numbers.IT2TrFN], which you can
add, multiply, score and aggregate. For interval type-2 *sets* and inference systems
(Mamdani and Sugeno rule bases, type reduction) see
[Interval type-2 inference](type2.md).

## The number

Each trapezoid is $(a_1, a_2, a_3, a_4; h_1, h_2)$: the membership rises from 0 at
$a_1$ to $h_1$ at $a_2$, goes linearly to $h_2$ at $a_3$ and falls to 0 at $a_4$
(Chen and Lee 2010). You build `IT2TrFN(upper, lower)` from two six-tuples
$(a_1, a_2, a_3, a_4, h_1, h_2)$.

<figure markdown="span">
  ![A seven-term interval type-2 trapezoidal scale from VL to VH with the DTraT score of each term](../assets/figures/it2-trapezoid-light.svg#only-light)
  ![A seven-term interval type-2 trapezoidal scale from VL to VH with the DTraT score of each term](../assets/figures/it2-trapezoid-dark.svg#only-dark)
  <figcaption markdown="span">A seven-term scale drawn by
  [`plot.it2_number`][bulanik.plot.it2_number]: upper trapezoid (bold), lower trapezoid
  (thin), footprint of uncertainty between them. The numbers under the axis are the DTraT
  scores. The scale is **illustrative**, in the style of the interval type-2 TOPSIS
  literature; the values are not a transcription of any published table.</figcaption>
</figure>

```python exec="on" source="above" result="text"
from bulanik import IT2TrFN, TFN, TrFN

low = IT2TrFN((0, 0.1, 0.1, 0.3, 1, 1), (0.05, 0.1, 0.1, 0.2, 0.9, 0.9))
print(low.upper)
print(low.lower)
print(IT2TrFN.from_trfn(TrFN(1, 2, 3, 5)))        # a type-1 number: upper = lower, zero-width footprint
print(IT2TrFN.from_trfn(TFN(1, 2, 3)).upper)
```

### Nesting

The lower trapezoid must be **nested** in the upper one (up to $10^{-12}$):

$$
a^U_1 \le a^L_1 \le a^L_2 \le a^L_3 \le a^L_4 \le a^U_4, \qquad
a^U_2 \le a^L_2, \quad a^L_3 \le a^U_3, \qquad
h^L_1, h^L_2 \le \min(h^U_1, h^U_2),
$$

and the upper trapezoid needs $\max(h_1, h_2) > 0$. The violated inequality is named in
the `ValueError`. Nesting guarantees that the lower membership never exceeds the upper
one, and it is *preserved* by the sum, difference, non-negative product, scalar and power,
so arithmetic never leaves the class. It is the usual assumption of the interval type-2
MCDM literature. General containment is slightly weaker, but the sum of two merely
contained numbers can leave the footprint constraint, so it is not enough for a closed
arithmetic.

## Arithmetic

Following Chen and Lee (2010a) every operation applies the type-1 rule to the upper and
to the lower trapezoid, and the heights combine with `min`:

- sum: points add; difference: $(a_1 - b_4, a_2 - b_3, a_3 - b_2, a_4 - b_1)$;
- product: componentwise, for non-negative points;
- scalar $k$: points times $k$, heights unchanged (for $k < 0$ the points reverse and
  $h_1, h_2$ swap);
- power $k$: points to the power $k$, heights unchanged (for $k < 0$ the points reverse
  and the heights swap). The product and power give the geometric mean
  $(\prod \tilde a_i)^{1/n}$ of Kahraman et al. (2014).

There is a single operator family, `"algebraic"`.

```python exec="on" source="above" result="text"
from bulanik import IT2TrFN
from bulanik.aggregate import wa, wg

low = IT2TrFN((0, 0.1, 0.1, 0.3, 1, 1), (0.05, 0.1, 0.1, 0.2, 0.9, 0.9))
high = IT2TrFN((0.5, 0.7, 0.7, 0.9, 1, 1), (0.55, 0.7, 0.7, 0.85, 0.9, 0.9))

def r(t):
    return tuple(round(v, 4) for v in t)

print("low + high   ", r((low + high).upper), r((low + high).lower))
print("high - low   ", r((high - low).upper))
print("low * high   ", r((low * high).upper))
print("2 * low      ", r((2 * low).upper))
print("low ** 2     ", r((low ** 2).upper))
print("wa           ", r(wa([low, high]).upper))
print("wg           ", r(wg([low, high]).upper))
```

## Scores and distance

The default score is `"dtrat"`. All three are defuzzifications of the whole number:

| Score | Formula |
|---|---|
| `"dtrat"` (default) | $\tfrac12 \sum_{X \in \{U, L\}} \Big[\dfrac{(a^X_4 - a^X_1) + (h^X_1 a^X_2 - a^X_1) + (h^X_2 a^X_3 - a^X_1)}{4} + a^X_1\Big]$ (the DTraT of Kahraman et al. 2014; for a type-1 trapezoid with heights 1 it is the mean of the four points) |
| `"centroid"` | the mean of the x-centroids of the upper and the lower polygon $(a_1, 0), (a_2, h_1), (a_3, h_2), (a_4, 0)$ by the shoelace formula (the mean of the points if the area is zero). This is **bulanik's own choice**, not the Karnik-Mendel centroid of the type-2 set |
| `"chen_lee"` | $\sum_{X \in \{U, L\}} \Big[\sum_{p=1}^{3} M_p(X) - \tfrac14 \sum_{q=1}^{4} S_q(X) + h_1(X) + h_2(X)\Big]$ with $M_p = (a_p + a_{p+1})/2$, $S_q$ ($q \le 3$) the population standard deviation of $\{a_q, a_{q+1}\}$ and $S_4$ that of $\{a_1, \dots, a_4\}$ (the ranking value of Chen and Lee 2010a) |

There is **no accuracy function**. The only distance is `"vertex"`,
$\sqrt{\tfrac18 \sum (\Delta a)^2}$ over the 8 points of the two trapezoids; heights are
ignored. `"chen_lee"` is a ranking value, not a value on the scale of the points: it adds
heights, so compare it only with other `"chen_lee"` values.

```python exec="on" source="above" result="text"
from bulanik import IT2TrFN

low = IT2TrFN((0, 0.1, 0.1, 0.3, 1, 1), (0.05, 0.1, 0.1, 0.2, 0.9, 0.9))
high = IT2TrFN((0.5, 0.7, 0.7, 0.9, 1, 1), (0.55, 0.7, 0.7, 0.85, 0.9, 0.9))
for name in IT2TrFN.SCORES:
    print(f"{name:9}", round(low.score(name), 5), round(high.score(name), 5))
print(round(low.distance(high), 4))
print(IT2TrFN.SCORES, IT2TrFN.ACCURACIES, IT2TrFN.DISTANCES)
```

## In a decision

A [`LinguisticScale`][bulanik.scales.LinguisticScale] of `IT2TrFN` terms turns words
into interval type-2 numbers, `DecisionMatrix` and
[`rank_fuzzy`][bulanik.mcdm.fuzzy.rank_fuzzy] aggregate the experts and rank with the
`"dtrat"` score, and [`ahp_fuzzy_numbers`][bulanik.mcdm.weighting.ahp_fuzzy_numbers]
turns a matrix of interval type-2 judgements into weights (Kahraman et al. 2014).

```python exec="on" source="above" result="text"
from bulanik import IT2TrFN
from bulanik.mcdm import DecisionMatrix, rank_fuzzy
from bulanik.scales import LinguisticScale

scale = LinguisticScale("demo", {
    "L": IT2TrFN((0, 0.1, 0.1, 0.3, 1, 1), (0.05, 0.1, 0.1, 0.2, 0.9, 0.9)),
    "H": IT2TrFN((0.5, 0.7, 0.7, 0.9, 1, 1), (0.55, 0.7, 0.7, 0.85, 0.9, 0.9)),
})
dm = DecisionMatrix.from_terms(
    scale, ["A", "B"], ["price", "quality"],
    [[["L", "H"], ["H", "L"]],            # expert 1: rows are alternatives A, B
     [["H", "H"], ["H", "L"]]],           # expert 2
)
r = rank_fuzzy(dm, [0.5, 0.5], ["benefit", "cost"], method="codas")
print(r.ranking, r.operators["score"])
```

## Pitfalls

**Nesting is a hard rule.** A lower height above $\min(h^U_1, h^U_2)$, or a lower point
outside the upper trapezoid, raises `ValueError` naming the inequality. A lower
trapezoid drawn "a bit wider" is an error, not a warning.

**Six values per trapezoid.** The order is $(a_1, a_2, a_3, a_4, h_1, h_2)$, points
first, heights last. Points must be ordered and finite, heights in $[0, 1]$ (out by at
most $10^{-12}$ they are projected).

**Products need non-negative points.** `mul` raises for a trapezoid with a negative
point; the geometric mean `wg` likewise.

**No accuracy, one distance.** `accuracy()` raises `ValueError`, because the class has
no accuracy function.

**It is not a type-1 number.** `IT2TrFN + TFN` raises `TypeError`; convert with
`IT2TrFN.from_trfn` first.

```python exec="on" source="above" result="text"
from bulanik import IT2TrFN, TFN

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

low = IT2TrFN((0, 0.1, 0.1, 0.3, 1, 1), (0.05, 0.1, 0.1, 0.2, 0.9, 0.9))
show(lambda: IT2TrFN((0, 1, 2, 3, 0.5, 0.5), (0.5, 1, 2, 2.5, 0.9, 0.9)))   # lower height above upper
show(lambda: IT2TrFN((0, 1, 2, 3, 1, 1), (-0.5, 1, 2, 2.5, 0.9, 0.9)))      # lower pokes out
show(lambda: IT2TrFN((0, 1, 2, 3, 1), (0, 1, 2, 3, 1, 1)))                  # five values
show(lambda: IT2TrFN((0, 2, 1, 3, 1, 1), (0, 2, 1, 3, 1, 1)))               # unordered points
show(lambda: IT2TrFN((0, 1, 2, 3, 0, 0), (0, 1, 2, 3, 0, 0)))               # no membership at all
show(lambda: low.accuracy())
show(lambda: low.score("median"))
show(lambda: low * IT2TrFN((-1, 0, 0, 1, 1, 1), (-1, 0, 0, 1, 1, 1)))
show(lambda: low + TFN(1, 2, 3))
```

## See also

- [Interval type-2 inference](type2.md) for interval type-2 sets and fuzzy systems.
- [Interval-valued numbers](numbers-interval.md) for the orthopair and neutrosophic
  interval kinds, [Triangular numbers](numbers-triangular.md) for the type-1 `TrFN`.
- [Aggregation](aggregation.md), [Sets and scales](sets-and-scales.md),
  [More ranking methods](ranking-more.md) and [More weighting methods](weighting-more.md).
- API: [`IT2TrFN`][bulanik.numbers.IT2TrFN], [`plot.it2_number`][bulanik.plot.it2_number].
- Example: [14 interval type-2 versus type-1 under noise](../examples/14-it2-vs-t1-noise.md).
- Chen and Lee (2010a), *Expert Systems with Applications* 37(1), 824-833,
  doi:10.1016/j.eswa.2009.06.094 (arithmetic and ranking value); Chen and Lee (2010b),
  *Expert Systems with Applications* 37(4), 2790-2798, doi:10.1016/j.eswa.2009.09.012
  (interval type-2 TOPSIS); Kahraman, Oztaysi, Ucal Sari and Turanoglu (2014),
  *Knowledge-Based Systems* 59, 48-57, doi:10.1016/j.knosys.2014.02.001 (DTraT, interval
  type-2 AHP).
