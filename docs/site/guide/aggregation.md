# Aggregation

Several experts rate the same thing, or one decision has several criteria, and
you want one fuzzy number back. bulanik offers three functions in
[`bulanik.aggregate`][bulanik.aggregate]:

- [`wa`][bulanik.aggregate.wa], the **weighted average** $\sum_i w_i A_i$;
- [`wg`][bulanik.aggregate.wg], the **weighted geometric mean** $\prod_i A_i^{w_i}$;
- [`aggregate_experts`][bulanik.aggregate.aggregate_experts], which applies one of
  them to a stack of expert rating tables, cell by cell.

They work for **every number type**, because they are built from the type's own
`scale`, `add`, `power` and `mul`. The result has the type of the inputs.

## Weighted average and weighted geometric mean

`wa` scales each number by its weight and folds the results with `add`; `wg` raises
each to its weight with `power` and folds with `mul`. What that means depends on
the type:

| Type | `wa` | `wg` |
|---|---|---|
| `TFN`, `TrFN` | componentwise weighted mean of the points | componentwise weighted geometric mean (points $\ge 0$) |
| `IFN` | IFWA: $\bigl(1 - \prod (1 - \mu_i)^{w_i},\; \prod \nu_i^{w_i}\bigr)$ | IFWG: $\bigl(\prod \mu_i^{w_i},\; 1 - \prod (1 - \nu_i)^{w_i}\bigr)$ |
| `PFN`, `FFN`, `QROFN` | the same, on the $q$-th powers, then the $q$-th root | the same |
| `PictureFN` | $\bigl(1 - \prod (1 - \mu_i)^{w_i},\; \prod \eta_i^{w_i},\; \prod \nu_i^{w_i}\bigr)$ (closed form) | $\bigl(\prod (\mu_i + \eta_i)^{w_i} - \prod \eta_i^{w_i},\; \prod \eta_i^{w_i},\; 1 - \prod (1 - \nu_i)^{w_i}\bigr)$ (closed form) |
| `SphericalFN`, `NeutrosophicFN` | built from that type's `scale` and `add` | built from `power` and `mul` |
| `IVQROFN`, `IVIFN`, `IVPFN`, `IVFFN` | the orthopair rule on the lower pair and on the upper pair separately | the same |
| `IVNeutrosophicFN` | the neutrosophic rule on the lower and on the upper bounds | the same |
| `IT2TrFN` | componentwise weighted mean of the four points of the upper and of the lower trapezoid; the heights take the `min` | componentwise weighted geometric mean (points $\ge 0$) |

The interval-valued kinds (see [Interval-valued numbers](numbers-interval.md) and
[Interval type-2 trapezoids](numbers-it2.md)) aggregate the same way, so `wa` and `wg` and
`aggregate_experts` accept them, and so does `rank_fuzzy`:

```python exec="on" source="above" result="text"
from bulanik import IT2TrFN, IVIFN
from bulanik.aggregate import wa, wg

x = [IVIFN(0.4, 0.5, 0.3, 0.4), IVIFN(0.5, 0.6, 0.2, 0.3)]
r = wa(x, [0.7, 0.3])
print(tuple(round(v, 4) for v in (r.mu_l, r.mu_u, r.nu_l, r.nu_u)))

low = IT2TrFN((0, 0.1, 0.1, 0.3, 1, 1), (0.05, 0.1, 0.1, 0.2, 0.9, 0.9))
high = IT2TrFN((0.5, 0.7, 0.7, 0.9, 1, 1), (0.55, 0.7, 0.7, 0.85, 0.9, 0.9))
print(tuple(round(v, 4) for v in wa([low, high]).upper))
print(tuple(round(v, 4) for v in wg([low, high]).upper))
```

The picture forms are closed because scaling a picture number by a weight below 1
can leave its constraint set (see [Three-part numbers](numbers-threepart.md)).
Weights must be non-negative; inputs with weight 0 are dropped.

`ops=` selects the operator family of the type (`"einstein"` for orthopair
numbers) and `weights=None` means equal weights.

```python exec="on" source="above" result="text"
from bulanik import IFN, PFN, TFN, PictureFN
from bulanik.aggregate import wa, wg

t = [TFN(1, 2, 4), TFN(2, 3, 5), TFN(0, 1, 2)]
print(wa(t))                                       # equal weights
print(wa(t, [0.5, 0.3, 0.2]))
print(wg(t[:2], [0.5, 0.5]))

x = [IFN(0.6, 0.3), IFN(0.5, 0.4)]
for name, fn in (("wa", wa), ("wg", wg)):
    r = fn(x, [0.7, 0.3])
    print(name, f"({r.mu:.4f}, {r.nu:.4f})")
r = wa(x, [0.7, 0.3], ops="einstein")
print("wa einstein", f"({r.mu:.4f}, {r.nu:.4f})")

p = wa([PFN(0.7, 0.5), PFN(0.6, 0.6)])             # same rung required
print(p)
print([round(v, 4) for v in wa([PictureFN(0.5, 0.2, 0.1), PictureFN(0.4, 0.3, 0.2)]).parts])
```

### Weights

Weights must sum to 1 (within $10^{-9}$, then silently rescaled to exactly 1).
Pass `normalize=True` to give raw importances instead, say `[3, 2, 1]`; the
weights are then divided by their sum.

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.aggregate import wa

a, b = TFN(0, 2, 4), TFN(2, 4, 6)
print(wa([a, b], [3, 1], normalize=True))          # same as [0.75, 0.25]
print(wa([a, b], [0.75, 0.25]))
print(wa([a, b], [1.0, 0.0]))                      # a weight of 0 drops the input
```

## Aggregating experts

`aggregate_experts(ratings, weights)` takes ratings indexed
`[expert][alternative][criterion]` (a list of equally shaped matrices of fuzzy
numbers) and returns the `[alternative][criterion]` matrix. `method="wa"` is the
default; `"wg"` gives the geometric version. The `weights` are one per expert.
The typical flow is *words → numbers → one table*:

```python exec="on" source="above" result="text"
from bulanik.aggregate import aggregate_experts
from bulanik.scales import EXAMPLE_TFN_7 as scale

words = [
    [["G", "F"], ["MG", "VG"]],       # expert 1: 2 alternatives x 2 criteria
    [["VG", "MP"], ["F", "G"]],       # expert 2
    [["G", "F"], ["G", "G"]],         # expert 3
]
ratings = scale.convert(words)        # same nesting, now TFNs
weights = [0.5, 0.3, 0.2]             # expert 1 counts the most
table = aggregate_experts(ratings, weights)
for row in table:
    print([f"({n.l:.2f}, {n.m:.2f}, {n.u:.2f})" for n in row])

geo = aggregate_experts(ratings, weights, method="wg")
print(geo[0][0])
```

The aggregated table is what [`fuzzy_topsis`][bulanik.mcdm.fuzzy_topsis] and the
other MCDM functions consume. See [Fuzzy TOPSIS](fuzzy-topsis.md).

## Pitfalls

**Types must match, rungs included.** Mixing `TFN` and `IFN`, or `IFN` and `PFN`,
raises `TypeError`. Convert your scale once so that every expert uses the same
type.

**Weights are validated.** They must be non-negative and finite, one per value,
and sum to 1 unless you pass `normalize=True`.

**Geometric means need non-negative triangles.** `wg` of a triangle with a
negative lowest point fails, because `power` does.

**Expert tables must have the same shape.** Every expert needs the same number of
alternatives and criteria, and the weights count experts, not criteria.

**Empty input.** There is nothing to average.

```python exec="on" source="above" result="text"
from bulanik import IFN, PFN, TFN
from bulanik.aggregate import aggregate_experts, wa, wg

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: wa([TFN(1, 2, 3), IFN(0.5, 0.4)]))
show(lambda: wa([IFN(0.5, 0.4), PFN(0.5, 0.4)]))
show(lambda: wa([TFN(1, 2, 3), TFN(2, 3, 4)], [0.5, 0.2]))              # sums to 0.7
show(lambda: wa([TFN(1, 2, 3), TFN(2, 3, 4)], [0.5, 0.25, 0.25]))       # three weights, two values
show(lambda: wa([TFN(1, 2, 3), TFN(2, 3, 4)], [1.5, -0.5]))
show(lambda: wa([TFN(1, 2, 3), TFN(2, 3, 4)], [0, 0], normalize=True))
show(lambda: wg([TFN(-1, 2, 3), TFN(2, 3, 4)]))
show(lambda: wa([]))
show(lambda: wa([1, 2]))
show(lambda: wa([IFN(0.5, 0.4)], ops="hamacher"))
show(lambda: aggregate_experts([[[TFN(1, 2, 3)]], [[TFN(1, 2, 3), TFN(1, 2, 3)]]]))
show(lambda: aggregate_experts([[[TFN(1, 2, 3)]]], method="median"))
show(lambda: aggregate_experts([]))
```

## See also

- [Orthopair numbers](numbers-orthopair.md) and [Three-part numbers](numbers-threepart.md)
  for what the operators do to each type; [Interval-valued numbers](numbers-interval.md)
  and [Interval type-2 trapezoids](numbers-it2.md) for the v0.2 kinds.
- [Sets and scales](sets-and-scales.md) for `LinguisticScale.convert`.
- [Weighting](weighting.md) for computing the criterion weights, and
  [Fuzzy TOPSIS](fuzzy-topsis.md) for ranking the aggregated table.
- API: [`wa`][bulanik.aggregate.wa], [`wg`][bulanik.aggregate.wg],
  [`aggregate_experts`][bulanik.aggregate.aggregate_experts].
- Examples: [02 scale and experts](../examples/02-scale-and-experts.md) and
  [05 intuitionistic pipeline](../examples/05-intuitionistic-pipeline.md).
