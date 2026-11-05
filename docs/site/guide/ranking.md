# Ranking alternatives

Given alternatives rated on several criteria, a **ranking method** scores each alternative
and orders them. bulanik has four classical methods on a crisp matrix,
[`topsis`][bulanik.mcdm.ranking.topsis], [`vikor`][bulanik.mcdm.ranking.vikor],
[`waspas`][bulanik.mcdm.ranking.waspas] and [`edas`][bulanik.mcdm.ranking.edas], and
[`rank_fuzzy`][bulanik.mcdm.fuzzy.rank_fuzzy], which first turns *fuzzy expert ratings*
into that crisp matrix. Chen's fuzzy TOPSIS, which stays fuzzy throughout, has
[its own page](fuzzy-topsis.md).

## The common signature

The four crisp methods differ only in one extra parameter:

```text
method(matrix, weights, criteria_types, *, alternatives=None, criteria=None,
       normalize_weights=False)
```

- **`matrix`**: rows are alternatives, columns are criteria, all finite numbers.
- **`criteria_types`**: `"benefit"` (larger is better) or `"cost"` (smaller is better), one
  per column.
- **`weights`**: in any of four forms: a *sequence* in column order, a *mapping* from
  criterion name to weight, or a [`WeightingResult`][bulanik.mcdm.results.WeightingResult] or
  [`BWMResult`][bulanik.mcdm.results.BWMResult] straight from a
  [weighting method](weighting.md). They must sum to 1 unless `normalize_weights=True`,
  which rescales non-negative weights instead.
- **`alternatives`** and **`criteria`**: names. Alternatives default to `A1..Am`. Criteria
  are **inferred**: from `criteria=` if given, else from the keys of a mapping or result,
  else `C1..Cn` (a plain sequence carries no names). A mapping or result must name every
  criterion and no other.

The result is a [`RankingResult`][bulanik.mcdm.results.RankingResult] with `scores`
(a dict), `ranking` (alternatives, best first; ties keep input order), `operators` (the
normalisation and distance or parameter used) and `extras` (per-alternative intermediate
values).

```python exec="on" source="above" result="text"
from bulanik.mcdm import swara, topsis

criteria = ["price", "quality", "delivery"]
types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
data = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]

w = swara([0.2, 0.5], criteria)             # a WeightingResult carries the criteria names
a = topsis(data, w, types, alternatives=suppliers)
b = topsis(data, w.weights, types, alternatives=suppliers)                       # mapping
c = topsis(data, list(w.weights.values()), types, alternatives=suppliers)        # sequence
print(a.ranking, a.scores == b.scores == c.scores)
print({k: round(v, 4) for k, v in a.scores.items()})
print(a.operators, sorted(a.extras))

# raw importances 5 : 3 : 2 instead of weights summing to 1
d = topsis(data, [5, 3, 2], types, alternatives=suppliers, normalize_weights=True)
print(d.ranking)
```

Every method here works on the numbers *as given*: the scales of the columns are
normalised away inside each method, so you can mix euros and ratings.

## TOPSIS

**TOPSIS** (Hwang and Yoon 1981) prefers the alternative closest to the ideal solution
and farthest from the anti-ideal. Each column is divided by its Euclidean norm and
multiplied by the weight, $v_{ij} = w_j x_{ij} / \sqrt{\sum_i x_{ij}^2}$. The ideal
$v_j^+$ is the column maximum for a benefit and the minimum for a cost (the anti-ideal
$v_j^-$ the reverse). With Euclidean distances $D_i^+$ and $D_i^-$ to them, the score is
the **relative closeness**

$$
C_i = \frac{D_i^-}{D_i^+ + D_i^-} \in [0, 1],
$$

and larger is better (0.5 when both distances are 0, that is, all alternatives are
equal). `extras` holds `d_plus` and `d_minus`. A column of zeros contributes 0.

## VIKOR

**VIKOR** (Opricovic and Tzeng 2004) looks for a *compromise*: it balances the overall
distance to the best values (group utility) against the worst single shortfall
(individual regret). With $f_j^*$ and $f_j^-$ the best and worst value of criterion $j$,

$$
S_i = \sum_j w_j \frac{|f_j^* - x_{ij}|}{|f_j^* - f_j^-|}, \qquad
R_i = \max_j w_j \frac{|f_j^* - x_{ij}|}{|f_j^* - f_j^-|},
$$

and the compromise index, with $S^* = \min S$, $S^- = \max S$ and likewise for $R$,

$$
Q_i = v\,\frac{S_i - S^*}{S^- - S^*} + (1 - v)\,\frac{R_i - R^*}{R^- - R^*}.
$$

Criteria with $f_j^* = f_j^-$ are skipped and a term whose range is 0 counts as 0. The
parameter `v` in $[0, 1]$ (default 0.5, the usual consensus) is the weight of the group
utility strategy: raise it to favour the overall average, lower it to punish a single bad
criterion. `extras` holds `S` and `R`.

!!! warning "VIKOR scores are lower-is-better"
    For VIKOR the `scores` are $Q$, and the **smallest** is best. The `ranking` tuple is
    already ordered that way, and so is [`plot.ranking`][bulanik.plot.ranking], which
    labels the axis "lower is better". Use `ranking`, not a sort of `scores`. The
    compromise-acceptability conditions C1 and C2 of the original method are *not*
    checked.

## WASPAS

**WASPAS** (Zavadskas et al. 2012) blends two aggregations. Entries are normalised
linearly, $\bar x_{ij} = x_{ij} / \max_i x_{ij}$ for a benefit and
$\min_i x_{ij} / x_{ij}$ for a cost, and

$$
Q_i = \lambda \sum_j w_j \bar x_{ij} + (1 - \lambda) \prod_j \bar x_{ij}^{\,w_j},
$$

the weighted sum model plus the weighted product model, larger is better. The parameter
is `lam` (default 0.5): $\lambda = 1$ is the pure weighted sum, $\lambda = 0$ the pure
product. `extras` holds `wsm` and `wpm`. Entries must be **strictly positive** (the
product and the ratios need it).

## EDAS

**EDAS** (Keshavarz Ghorabaee et al. 2015) scores each alternative by its distance from
the *average* solution rather than from an extreme. With the column average $\bar x_j$,
the positive and negative distances from average of a benefit criterion are

$$
PDA_{ij} = \frac{\max(0, x_{ij} - \bar x_j)}{\bar x_j}, \qquad
NDA_{ij} = \frac{\max(0, \bar x_j - x_{ij})}{\bar x_j},
$$

swapped for a cost criterion. The weighted sums are $SP_i = \sum_j w_j PDA_{ij}$ and
$SN_i = \sum_j w_j NDA_{ij}$, normalised to $NSP_i = SP_i / \max SP$ and
$NSN_i = 1 - SN_i / \max SN$ (0 and 1 when the maximum is 0), and the appraisal score is
$(NSP_i + NSN_i)/2 \in [0, 1]$, larger is better. `extras` holds `SP`, `SN`, `NSP`, `NSN`.
Each column average must be positive.

## Comparing the methods

The methods answer subtly different questions, so on a close call they can disagree. Run
several on the same data and weights; the alternative that wins everywhere is a safe
choice.

```python exec="on" source="above" result="text"
from bulanik.mcdm import edas, topsis, vikor, waspas

criteria = ["price", "quality", "delivery"]
types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
data = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]
w = {"price": 0.4186, "quality": 0.3488, "delivery": 0.2326}

for method in (topsis, vikor, waspas, edas):
    r = method(data, w, types, alternatives=suppliers)
    print(f"{r.method:7}", r.ranking, {k: round(v, 3) for k, v in r.scores.items()})

for v in (0.0, 0.5, 1.0):                       # VIKOR: regret only .. utility only (no change here)
    print("v =", v, vikor(data, w, types, alternatives=suppliers, v=v).ranking)
for lam in (0.0, 1.0):                          # WASPAS: pure product .. pure sum
    print("lam =", lam, waspas(data, w, types, alternatives=suppliers, lam=lam).ranking)
```

On this data neither `v` nor `lam` changes the order (Cedar is best on both group utility
and regret), which is itself a robustness check; on closer data they can.

<figure markdown="span">
  ![Four dot plots of the supplier scores. TOPSIS: Cedar 0.5978, Atlas 0.5231, Delta 0.4338, Borealis 0.4199, higher is better. VIKOR: Cedar 0, Borealis 0.2857, Atlas 0.4322, Delta 1, lower is better. WASPAS and EDAS rank Cedar first and Atlas second](../assets/figures/mcdm-ranking-methods-light.svg#only-light)
  ![Four dot plots of the supplier scores. TOPSIS: Cedar 0.5978, Atlas 0.5231, Delta 0.4338, Borealis 0.4199, higher is better. VIKOR: Cedar 0, Borealis 0.2857, Atlas 0.4322, Delta 1, lower is better. WASPAS and EDAS rank Cedar first and Atlas second](../assets/figures/mcdm-ranking-methods-dark.svg#only-dark)
  <figcaption markdown="span">The same four suppliers and SWARA weights through the four
  methods, drawn by [`plot.ranking`][bulanik.plot.ranking]: best at the top, the winner in
  the accent colour. All four pick Cedar; VIKOR's scale runs the other way, and
  it puts Borealis second where the others put Atlas.</figcaption>
</figure>

## Fuzzy ratings: `DecisionMatrix` and `rank_fuzzy`

Experts often rate in words. A [`DecisionMatrix`][bulanik.mcdm.fuzzy.DecisionMatrix] holds
ratings indexed `[expert][alternative][criterion]`, all fuzzy numbers of **one kind**, with
optional `expert_weights` summing to 1 (equal weights by default). Build one directly, or
from linguistic terms with
[`DecisionMatrix.from_terms(scale, alternatives, criteria, terms, expert_weights)`][bulanik.mcdm.fuzzy.DecisionMatrix.from_terms]
using a [`LinguisticScale`][bulanik.scales.LinguisticScale].

[`rank_fuzzy(matrix, weights, criteria_types, *, method="topsis", aggregation="wa",
ops="algebraic", score=None, **method_kwargs)`][bulanik.mcdm.fuzzy.rank_fuzzy] then does
three steps:

1. **Aggregate** the experts cell by cell with
   [`aggregate_experts`][bulanik.aggregate.aggregate_experts]: `"wa"` (weighted average)
   or `"wg"` (weighted geometric mean).
2. **Score** each aggregated fuzzy number to a crisp value with its kind's `score` (the
   kind's default, or the name you pass).
3. **Rank** the crisp matrix with `method`: `"topsis"`, `"vikor"`, `"waspas"` or `"edas"`,
   or one of the six methods of [More ranking methods](ranking-more.md) (`"codas"`,
   `"copras"`, `"marcos"`, `"cocoso"`, `"wisp"`, `"electre"`).
   Extra keywords go to that method (`v=`, `lam=`, `normalize_weights=`).

The `operators` of the result are those of the crisp method plus `aggregation`, `ops`
and `score`. Because step 2 *discards the fuzziness*, the method is a convenient way to
use any number kind (triangular, intuitionistic, and so on), not a fuzzy-arithmetic
ranking; for that see [Fuzzy TOPSIS](fuzzy-topsis.md).

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.mcdm import DecisionMatrix, rank_fuzzy
from bulanik.scales import LinguisticScale

scale = LinguisticScale("demo", {"L": TFN(0, 0, 5), "M": TFN(3, 5, 7), "H": TFN(5, 10, 10)})
dm = DecisionMatrix.from_terms(
    scale, ["A", "B"], ["price", "quality"],
    [[["L", "H"], ["M", "M"]],           # expert 1: rows are alternatives A, B
     [["M", "H"], ["H", "L"]]],          # expert 2
    expert_weights=[0.6, 0.4],
)
print(dm.experts[0][0][0], dm.expert_weights)

r = rank_fuzzy(dm, {"price": 0.4, "quality": 0.6}, ["cost", "benefit"])
print(r.ranking, {k: round(v, 3) for k, v in r.scores.items()})
print(r.operators)
g = rank_fuzzy(dm, [0.4, 0.6], ["cost", "benefit"], method="vikor", aggregation="wg",
               score="graded_mean", v=0.3)
print(g.ranking, g.operators)
```

## End to end: words to a ranking

A complete decision, from linguistic judgements to a chart:

1. three experts say, in words, how much less important each criterion is than the one
   before it (price first, then quality, then delivery) as triangular numbers;
2. [`aggregate_experts`][bulanik.aggregate.aggregate_experts] combines the three opinions
   into one fuzzy importance per step;
3. [`swara`][bulanik.mcdm.weighting.swara] turns those into criteria weights (scoring the
   fuzzy importances with the default centroid);
4. the same experts rate four suppliers in words, and
   [`rank_fuzzy`][bulanik.mcdm.fuzzy.rank_fuzzy] aggregates, scores and ranks them with the
   SWARA weights;
5. the results are plotted.

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.aggregate import aggregate_experts
from bulanik.mcdm import DecisionMatrix, rank_fuzzy, swara
from bulanik.scales import LinguisticScale

criteria = ["price", "quality", "delivery"]
types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
experts = [0.4, 0.35, 0.25]                       # how much each expert counts

# 1-2. comparative importance of quality vs price and delivery vs quality, in words
importance = LinguisticScale("importance", {
    "slightly": TFN(0.1, 0.2, 0.3), "clearly": TFN(0.3, 0.4, 0.5), "much": TFN(0.5, 0.6, 0.7)})
words = [[["slightly", "clearly"]],               # [expert][one row][step]
         [["clearly", "much"]],
         [["slightly", "much"]]]
steps = aggregate_experts(importance.convert(words), experts)[0]
print("aggregated importances:", [tuple(round(p, 2) for p in (s.l, s.m, s.u)) for s in steps])

# 3. SWARA weights from the fuzzy importances
weights = swara(steps, criteria)
print("weights:", {k: round(v, 3) for k, v in weights.weights.items()}, weights.operators)

# 4. experts rate the suppliers: [expert][supplier][criterion]; "L" price means cheap
rating = LinguisticScale("rating", {
    "VL": TFN(0, 0, 3), "L": TFN(1, 3, 5), "M": TFN(3, 5, 7), "H": TFN(5, 7, 9), "VH": TFN(7, 10, 10)})
terms = [
    [["H", "M", "H"], ["VH", "H", "L"], ["M", "VH", "M"], ["L", "L", "VH"]],
    [["M", "H", "M"], ["H", "M", "L"], ["H", "H", "M"], ["L", "VL", "H"]],
    [["H", "M", "VH"], ["VH", "H", "M"], ["M", "VH", "L"], ["VL", "L", "VH"]],
]
dm = DecisionMatrix.from_terms(rating, suppliers, criteria, terms, experts)
result = rank_fuzzy(dm, weights, types)           # topsis, weighted average, centroid
print(result.ranking)
print({k: round(v, 4) for k, v in result.scores.items()})

# 5. the two charts below come from the same objects
#    from bulanik import plot; plot.weights(weights); plot.ranking(result)
```

<figure markdown="span">
  ![Left: SWARA weights price 0.434, quality 0.342, delivery 0.225 as horizontal bars. Right: TOPSIS closeness of the suppliers, Cedar 0.5912, Delta 0.571, Atlas 0.4614, Borealis 0.327, higher is better](../assets/figures/mcdm-pipeline-light.svg#only-light)
  ![Left: SWARA weights price 0.434, quality 0.342, delivery 0.225 as horizontal bars. Right: TOPSIS closeness of the suppliers, Cedar 0.5912, Delta 0.571, Atlas 0.4614, Borealis 0.327, higher is better](../assets/figures/mcdm-pipeline-dark.svg#only-dark)
  <figcaption markdown="span">The result of the pipeline above: weights from
  [`plot.weights`][bulanik.plot.weights] (left) and the supplier ranking from
  [`plot.ranking`][bulanik.plot.ranking] (right). Cedar wins, with Delta close behind.</figcaption>
</figure>

If the decision matters, repeat the last step with `method="vikor"`, `"waspas"` and
`"edas"`, and with `aggregation="wg"`, and see whether Cedar still wins.

## Pitfalls

**VIKOR reads the other way.** Lower $Q$ is better. A chart or a sort that assumes
"larger is better" will invert the answer for VIKOR; `ranking` is always right.

**Weights must sum to 1.** Otherwise the method raises `ValueError`; pass
`normalize_weights=True` for raw importances. Negative weights are always an error.

**Mappings and results must match the criteria exactly.** A mapping that misses a
criterion or has an extra key is an error, so a typo in a name cannot silently drop a
criterion.

**Cost criteria are "smaller is better", not "negative".** Declare them with
`"cost"`; do not negate the column. WASPAS needs strictly positive entries, EDAS a
positive average per column.

**Ties keep input order.** Two alternatives with equal scores appear in the order you
gave them.

**`rank_fuzzy` throws the fuzziness away at step 2**, and different score names can give
different rankings. `waspas` needs positive scores (for orthopair numbers pass
`score="normalized"`).

**Experts must use one number kind.** A `DecisionMatrix` rejects a mix of, say, `TFN` and
`IFN`.

```python exec="on" source="above" result="text"
from bulanik import IFN, TFN
from bulanik.mcdm import DecisionMatrix, rank_fuzzy, topsis, waspas

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

data = [[250, 16], [200, 12], [300, 32]]
types = ["cost", "benefit"]
show(lambda: topsis(data, [0.5, 0.6], types))                              # sum is 1.1
show(lambda: topsis(data, {"price": 0.5}, types, criteria=["price", "mem"]))   # missing criterion
show(lambda: topsis(data, {"price": 0.5, "mem": 0.4, "x": 0.1}, types, criteria=["price", "mem"]))
show(lambda: topsis(data, [0.5, 0.5], ["cost", "profit"]))                 # unknown type
show(lambda: topsis(data, [0.5, 0.5], ["cost"]))                           # wrong count
show(lambda: topsis([[1, 2], [3]], [0.5, 0.5], types))                     # ragged
show(lambda: waspas([[0, 2], [3, 4]], [0.5, 0.5], types))                  # a zero entry
show(lambda: rank_fuzzy(DecisionMatrix(["A"], ["c"], [[[TFN(1, 2, 3)]]]), [1.0], ["benefit"], method="ahp"))
show(lambda: DecisionMatrix(["A"], ["c"], [[[TFN(1, 2, 3)]], [[IFN(0.5, 0.4)]]]))
```

## See also

- [More ranking methods](ranking-more.md) for CODAS, COPRAS, MARCOS, CoCoSo, WISP and
  ELECTRE, which `rank_fuzzy` also accepts.
- [Weighting](weighting.md) for the weights, and [Fuzzy TOPSIS](fuzzy-topsis.md) for
  ranking that keeps the numbers fuzzy.
- [Aggregation](aggregation.md) for `aggregate_experts`, [Sets and scales](sets-and-scales.md)
  for linguistic scales.
- [LumiX bridge](bridge-lumix.md): `scores_as_data` hands a ranking to an optimisation
  model.
- API: [`bulanik.mcdm`][bulanik.mcdm], [`bulanik.plot.ranking`][bulanik.plot.ranking].
- Examples: [04 SWARA and TOPSIS](../examples/04-swara-topsis-supplier.md),
  [02 scale and experts](../examples/02-scale-and-experts.md),
  [05 intuitionistic pipeline](../examples/05-intuitionistic-pipeline.md).
- Hwang and Yoon (1981), *Multiple Attribute Decision Making*, Springer; Opricovic and
  Tzeng (2004), "Compromise solution by MCDM methods: a comparative analysis of VIKOR and
  TOPSIS", *European Journal of Operational Research* 156(2); Zavadskas et al. (2012),
  "Optimization of weighted aggregated sum product assessment", *Elektronika ir
  Elektrotechnika* 122(6); Keshavarz Ghorabaee et al. (2015), "Multi-criteria inventory
  classification using a new method of evaluation based on distance from average solution
  (EDAS)", *Informatica* 26(3).
