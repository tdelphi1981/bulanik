# More ranking methods: CODAS, COPRAS, MARCOS, CoCoSo, WISP, ELECTRE and fuzzy variants

[Ranking](ranking.md) covers TOPSIS, VIKOR, WASPAS and EDAS. This page adds six more.
Five of them, [`codas`][bulanik.mcdm.scoring.codas], [`copras`][bulanik.mcdm.scoring.copras],
[`marcos`][bulanik.mcdm.scoring.marcos], [`cocoso`][bulanik.mcdm.scoring.cocoso] and
[`wisp`][bulanik.mcdm.scoring.wisp], *score* each alternative from a crisp matrix, and
[`electre`][bulanik.mcdm.outranking.electre] *compares alternatives pair by pair*.
Fuzzy versions of CODAS, WASPAS and CoCoSo are [further down](#ranking-on-fuzzy-numbers).

All six share the signature and the weight forms of the classical methods (a sequence, a
mapping, or a [`WeightingResult`][bulanik.mcdm.results.WeightingResult]; `matrix`,
`weights`, `criteria_types`, `alternatives=`, `criteria=`, `normalize_weights=`), return a
[`RankingResult`][bulanik.mcdm.results.RankingResult], and rank **larger as better**
(unlike VIKOR). A criterion with weight 0 is ignored in every additive and
multiplicative term, as in [`wa`][bulanik.aggregate.wa]. All of them are also available
in [`rank_fuzzy`][bulanik.mcdm.fuzzy.rank_fuzzy] (see
[below](#fuzzy-ratings-through-rank_fuzzy)).

| Method | Idea | Extra parameter | Entries must be |
|---|---|---|---|
| CODAS | distance from the negative-ideal point | `tau`, `psi` | non-negative in benefit columns, strictly positive in cost columns |
| COPRAS | proportional share of benefit and cost sums | none | non-negative, positive column sums |
| MARCOS | utility relative to the ideal and anti-ideal | none | strictly positive |
| CoCoSo | three combined compromise scores | `lam` | any finite numbers |
| WISP | four sum and product measures | `product` | strictly positive |
| ELECTRE | concordance and discordance between pairs | `cost`, `c_threshold`, `d_threshold` | any finite numbers, at least two rows |

The snippets use the supplier data of the [ranking page](ranking.md) and its SWARA
weights, so the answers can be compared with the four classical methods:

```python exec="on" source="above" result="text"
from bulanik.mcdm import (cocoso, codas, copras, edas, electre, marcos, topsis, vikor,
                          waspas, wisp)

types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
data = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]
w = {"price": 0.4186, "quality": 0.3488, "delivery": 0.2326}

for method in (topsis, vikor, waspas, edas, codas, copras, marcos, cocoso, wisp, electre):
    r = method(data, w, types, alternatives=suppliers)
    print(f"{r.method:8}", r.ranking, {k: round(v, 3) for k, v in r.scores.items()})
```

All ten pick Cedar. Atlas versus Borealis for second place is where the methods differ:
the second place of CODAS and VIKOR is Borealis, of the others Atlas. As elsewhere, rank
with two or three methods and compare.

## CODAS

**CODAS** (Keshavarz Ghorabaee et al. 2016) scores an alternative by how far it is from
the **negative-ideal point**. Entries are normalised linearly,
$n_{ij} = x_{ij} / \max_i x_{ij}$ for a benefit and $\min_i x_{ij} / x_{ij}$ for a cost,
and weighted, $r_{ij} = w_j n_{ij}$. With $ns_j = \min_i r_{ij}$, the Euclidean (primary)
and taxicab (secondary) distances from it are

$$
E_i = \sqrt{\sum_j (r_{ij} - ns_j)^2}, \qquad T_i = \sum_j |r_{ij} - ns_j|.
$$

The relative assessment is

$$
h_{ik} = (E_i - E_k) + \psi(E_i - E_k)\,(T_i - T_k), \qquad
\psi(x) = \begin{cases} 1 & |x| \ge \tau \\ 0 & |x| < \tau \end{cases}
$$

and the score is $H_i = \sum_k h_{ik}$. The scores sum to 0, so they can be negative.
`extras` holds `E` and `T`; `tau` (default 0.02) lies in $[0, 1]$.

**Two conventions for $\psi$.** The paper is internally inconsistent. Its Eq. (10), printed
above, adds the taxicab term when the Euclidean distances differ by *at least* $\tau$,
although the text says the taxicab distance is the tie-breaker "if the difference ... is
less than $\tau$". The paper's own Tables 4 and 8 follow the text, that is, the opposite
branch, and Chen and Gou (2021) criticise the printed form. Both are available:

- `psi="equation"` (default): Eq. (10) as printed, the taxicab term counts when
  $|E_i - E_k| \ge \tau$. This is what most re-implementations and textbooks use.
- `psi="tables"`: the taxicab term counts when $|E_i - E_k| < \tau$. Use it to reproduce the
  paper's printed numbers.

The distances $E$ and $T$ are the same either way, and the choice is recorded in
`operators["psi"]`. With a larger $\tau$ the two can disagree:

```python exec="on" source="above" result="text"
from bulanik.mcdm import codas

types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
data = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]
w = {"price": 0.4186, "quality": 0.3488, "delivery": 0.2326}

r = codas(data, w, types, alternatives=suppliers)
print({k: round(v, 3) for k, v in r.extras["E"].items()})
print({k: round(v, 3) for k, v in r.extras["T"].items()})
for psi in ("equation", "tables"):
    r = codas(data, w, types, alternatives=suppliers, tau=0.1, psi=psi)
    print(psi, r.ranking, {k: round(v, 3) for k, v in r.scores.items()})
print(r.operators)
```

## COPRAS

**COPRAS** (Zavadskas, Kaklauskas and Sarka 1994) divides each column by its sum and
weights it, $d_{ij} = w_j x_{ij} / \sum_i x_{ij}$. Benefit and cost criteria are summed
separately, $S^+_i = \sum_{j \in \text{benefit}} d_{ij}$ and
$S^-_i = \sum_{j \in \text{cost}} d_{ij}$, and the relative significance is

$$
Q_i = S^+_i + \frac{\sum_k S^-_k}{S^-_i \sum_k 1 / S^-_k}
$$

(just $S^+_i$ without a cost criterion). `scores` holds $Q$; `extras` holds `S_plus`,
`S_minus` and the **utility degree** $100\,Q_i / \max Q$ in percent. The $Q$ formula is
Eq. (8) of Razavi Hajiagha, Hashemi and Zavadskas (2013). A cost criterion with weight 0
is ignored, so a model whose cost criteria all have weight 0 is scored as having none.

```python exec="on" source="above" result="text"
from bulanik.mcdm import copras

types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
data = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]
w = {"price": 0.4186, "quality": 0.3488, "delivery": 0.2326}

r = copras(data, w, types, alternatives=suppliers)
print({k: round(v, 4) for k, v in r.scores.items()})
print({k: round(v, 2) for k, v in r.extras["utility"].items()})   # percent of the best
print(sorted(r.extras), r.operators)
```

## MARCOS

**MARCOS** (Stevic et al. 2020) extends the matrix with the **ideal** solution AI (the
best value per criterion: the maximum of a benefit, the minimum of a cost) and the
**anti-ideal** AAI (the worst). Every entry is divided by the ideal value,
$n_{ij} = x_{ij} / x_{AI,j}$ for a benefit and $x_{AI,j} / x_{ij}$ for a cost, and
weighted, $v_{ij} = w_j n_{ij}$. With $S_i = \sum_j v_{ij}$ (also for the AAI and AI rows)
the utility degrees are $K^-_i = S_i / S_{AAI}$ and $K^+_i = S_i / S_{AI}$, the utility
functions are

$$
f(K^-_i) = \frac{K^+_i}{K^+_i + K^-_i}, \qquad
f(K^+_i) = \frac{K^-_i}{K^+_i + K^-_i},
$$

and the score is

$$
f(K_i) = \frac{K^+_i + K^-_i}{1 + \frac{1 - f(K^+_i)}{f(K^+_i)} + \frac{1 - f(K^-_i)}{f(K^-_i)}}.
$$

`extras` holds `K_minus`, `K_plus` and `S`. Because $f(K^-_i)$ and $f(K^+_i)$ depend only
on $S_{AAI} / S_{AI}$, they are the same for every alternative, so the **ranking equals
the ranking by $S_i$**. The equations follow the restatement of Ulutas et al. (2020). The
entries must be strictly positive: a non-positive entry has no meaningful ratio to the
ideal.

```python exec="on" source="above" result="text"
from bulanik.mcdm import marcos

types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
data = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]
w = {"price": 0.4186, "quality": 0.3488, "delivery": 0.2326}

r = marcos(data, w, types, alternatives=suppliers)
print({k: round(v, 3) for k, v in r.scores.items()})
print({k: round(v, 3) for k, v in r.extras["K_plus"].items()})
print(r.ranking == tuple(sorted(r.extras["S"], key=r.extras["S"].get, reverse=True)))
```

## CoCoSo

**CoCoSo** (Yazdani et al. 2019) scales entries to $[0, 1]$ by min-max normalisation,
$r_{ij} = (x_{ij} - \min_i x_{ij}) / (\max_i x_{ij} - \min_i x_{ij})$ for a benefit and
$(\max_i x_{ij} - x_{ij}) / (\max_i x_{ij} - \min_i x_{ij})$ for a cost. A **constant
column** gets $r_{ij} = 0$ for every alternative, which adds nothing to either sum below.
The weighted-sum and weighted-power comparability sequences are

$$
S_i = \sum_j w_j r_{ij}, \qquad P_i = \sum_{j:\, w_j > 0} r_{ij}^{\,w_j}.
$$

Three appraisal scores combine them,

$$
k_{ia} = \frac{P_i + S_i}{\sum_i (P_i + S_i)}, \qquad
k_{ib} = \frac{S_i}{\min_i S_i} + \frac{P_i}{\min_i P_i}, \qquad
k_{ic} = \frac{\lambda S_i + (1 - \lambda) P_i}{\lambda \max_i S_i + (1 - \lambda) \max_i P_i},
$$

and the score is

$$
k_i = (k_{ia}\, k_{ib}\, k_{ic})^{1/3} + \tfrac13 (k_{ia} + k_{ib} + k_{ic}).
$$

`lam` $= \lambda \in [0, 1]$ (default 0.5, the paper's suggestion) balances sum and power
in $k_{ic}$. `extras` holds `S`, `P`, `k_a`, `k_b`, `k_c`. Entries may be any finite
numbers.

!!! warning "All-worst alternatives"
    If some alternative is **worst on every weighted criterion**, then
    $\min S = \min P = 0$ and $k_{ib}$ is undefined; Rasoanaivo et al. (2024) report this
    failure of the method. bulanik raises a `ValueError` naming that alternative rather
    than return garbage. The same happens when all alternatives are identical, and always
    when only one criterion carries weight (a single effective criterion: its worst alternative
    is worst on every effective criterion). The
    remedy is to drop the dominated alternative or to rank with another method.

```python exec="on" source="above" result="text"
from bulanik.mcdm import cocoso

types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
data = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]
w = {"price": 0.4186, "quality": 0.3488, "delivery": 0.2326}

for lam in (0.0, 0.5, 1.0):                      # pure power .. pure sum
    r = cocoso(data, w, types, alternatives=suppliers, lam=lam)
    print(lam, r.ranking, {k: round(v, 3) for k, v in r.scores.items()})
print(sorted(r.extras), r.operators)
```

## WISP

**WISP** (Stanujkic et al. 2023) divides entries by the column maximum,
$r_{ij} = x_{ij} / \max_i x_{ij}$. With $\Omega_{\max}$ the benefit and $\Omega_{\min}$ the
cost criteria (those with positive weight), four utility measures combine weighted sums
and weighted products:

$$
u^{sd}_i = \sum_{\Omega_{\max}} w_j r_{ij} - \sum_{\Omega_{\min}} w_j r_{ij}, \qquad
u^{pd}_i = \prod_{\Omega_{\max}} r_{ij}^{w_j} - \prod_{\Omega_{\min}} r_{ij}^{w_j},
$$

$$
u^{sr}_i = \frac{\sum_{\Omega_{\max}} w_j r_{ij}}{\sum_{\Omega_{\min}} w_j r_{ij}}, \qquad
u^{pr}_i = \frac{\prod_{\Omega_{\max}} r_{ij}^{w_j}}{\prod_{\Omega_{\min}} r_{ij}^{w_j}}.
$$

Each is recalculated, $\bar u_i = (1 + u_i) / (1 + \max_k u_k)$, and the score is
$u_i = \tfrac14 (\bar u^{sd}_i + \bar u^{pd}_i + \bar u^{sr}_i + \bar u^{pr}_i)$. `extras`
holds `u_sd`, `u_pd`, `u_sr`, `u_pr` and `ubar_sd`, `ubar_pd`, `ubar_sr`, `ubar_pr`.

**Only benefit or only cost criteria.** For benefit criteria only, the paper uses
two measures and averages their recalculated values; the convention here reproduces
exactly that (an empty group contributes 0 to a difference; in a ratio an empty
denominator group is replaced by 1, an empty numerator is 0 for sums and 1 for products).
For cost criteria only the paper is silent, and the same convention is used. A constant
matrix, or a single alternative, gives equal scores.

**Two conventions for the products.** The cited formulas write the weighted product of the
normalised ratings; the worked example of the restatement (Stanujkic 2023, Table 7)
evaluates it as $\prod_j r_{ij} w_j$ (multiplication) rather than
$\prod_j r_{ij}^{w_j}$, which is the weighted product model. Both are available:
`product="power"` (default) uses $r_{ij}^{w_j}$; `product="times"` uses $r_{ij} w_j$ and
reproduces Table 7. The choice is recorded in `operators["product"]`.

```python exec="on" source="above" result="text"
from bulanik.mcdm import wisp

types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
data = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]
w = {"price": 0.4186, "quality": 0.3488, "delivery": 0.2326}

for product in ("power", "times"):
    r = wisp(data, w, types, alternatives=suppliers, product=product)
    print(product, r.ranking, {k: round(v, 3) for k, v in r.scores.items()})
print(sorted(r.extras))
```

## ELECTRE

**ELECTRE I** (Roy 1968) compares alternatives **pair by pair**. The matrix is
vector-normalised and weighted as in TOPSIS, $v_{ij} = w_j x_{ij} / \sqrt{\sum_i x_{ij}^2}$
(a zero column gives 0). For an ordered pair $k \ne l$ the concordance set is
$C_{kl} = \{j : x_{kj} \ge x_{lj}\}$ for a benefit criterion ($x_{kj} \le x_{lj}$ for a
cost one) and the discordance set $D_{kl}$ is its complement. The indices are

$$
c_{kl} = \sum_{j \in C_{kl}} w_j, \qquad
d_{kl} = \frac{\max_{j \in D_{kl}} |v_{kj} - v_{lj}|}{\max_j |v_{kj} - v_{lj}|},
$$

with $d_{kl} = 0$ when $D_{kl}$ is empty or the denominator is 0 (identical alternatives).
The denominator is the largest weighted difference of the pair over *all* criteria, as in
Hwang and Yoon (1981), not a per-criterion range. The thresholds $\bar c$ and $\bar d$ are
the means of $c_{kl}$ and $d_{kl}$ over all pairs unless you give `c_threshold` and
`d_threshold`. Alternative $k$ **outranks** $l$ when

$$
e_{kl} = [c_{kl} \ge \bar c]\,[d_{kl} \le \bar d].
$$

Two entries of a criterion are tied, and concordant both ways, when they differ by at most
$10^{-12}$ times the larger magnitude, so scaling a column cannot create or break a tie by
one ulp.

**A complete ranking.** The outranking relation is partial: it can leave two alternatives
unrelated. For a complete ranking the net indices of van Delft and Nijkamp (1977) are used,

$$
C_k = \sum_{l \ne k} c_{kl} - \sum_{l \ne k} c_{lk}, \qquad
D_k = \sum_{l \ne k} d_{kl} - \sum_{l \ne k} d_{lk},
$$

where a large $C_k$ and a small $D_k$ are good. Sources rank by each and pick the
alternative that is best on both; here the two are combined into one score so that a full
ranking always exists,

$$
S_k = C_k - D_k,
$$

larger is better. The net indices do not depend on the thresholds, so neither do `scores`
and `ranking`: **the thresholds only shape `outranks`**. `extras` holds `net_concordance`,
`net_discordance`, `concordance` and `discordance` (each `{alternative: {other: value}}`)
and `outranks` (`{alternative: tuple of the alternatives it outranks}`).

**Cost criteria** (`cost`):

- `"reverse"` (default, Hwang and Yoon): a cost column is kept and the comparison reversed
  ($\le$ instead of $\ge$).
- `"inverse"` (e.g. Cakin and Ozdemir 2013): a cost column is replaced by $1/x_{ij}$ before
  the vector norm and then compared like a benefit. It needs $x_{ij} > 0$ in every cost
  column. The concordance indices are the same; the discordance indices, and so the scores,
  differ.

The choice is recorded in `operators["cost"]`. Whether a criterion is a benefit or a cost
is your call; it is not inferred from the data.

```python exec="on" source="above" result="text"
from bulanik.mcdm import electre

types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
data = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]
w = {"price": 0.4186, "quality": 0.3488, "delivery": 0.2326}

for cost in ("reverse", "inverse"):
    r = electre(data, w, types, alternatives=suppliers, cost=cost)
    print(cost, r.ranking, {k: round(v, 3) for k, v in r.scores.items()})
print(r.extras["outranks"])                         # who outranks whom (inverse)

strict = electre(data, w, types, alternatives=suppliers, c_threshold=0.7, d_threshold=0.5)
print(strict.extras["outranks"], strict.ranking == r.ranking)    # stricter thresholds, same ranking
print({k: round(v, 2) for k, v in strict.extras["concordance"]["Cedar"].items()})
print(strict.operators)
```

## Fuzzy ratings through `rank_fuzzy`

[`rank_fuzzy`][bulanik.mcdm.fuzzy.rank_fuzzy] accepts ten crisp methods: `"topsis"`, `"vikor"`,
`"waspas"`, `"edas"`, `"codas"`, `"copras"`, `"marcos"`, `"cocoso"`, `"wisp"` and
`"electre"` (and the three fuzzy methods of the [next section](#ranking-on-fuzzy-numbers)). Extra keywords go to the crisp method (`tau=` and `psi=` for CODAS, `lam=` for
WASPAS and CoCoSo, `product=` for WISP, `c_threshold=` and `d_threshold=` for ELECTRE).
The experts are aggregated and each fuzzy number is scored first, so the
[number kinds](numbers-interval.md) with a signed default score need a positive one for
the ratio methods.

!!! note "ELECTRE's own `score` key"
    ELECTRE already names its ranking score `operators["score"]`. Through `rank_fuzzy`
    the key `"score"` records the *defuzzification* score, so ELECTRE's own entry becomes
    `operators["ranking_score"]`.

```python exec="on" source="above" result="text"
from bulanik import IVIFN
from bulanik.mcdm import DecisionMatrix, rank_fuzzy

I = IVIFN
dm = DecisionMatrix(["A", "B", "C"], ["price", "quality"], [
    [[I(0.3, 0.4, 0.4, 0.5), I(0.6, 0.7, 0.2, 0.3)],        # expert 1
     [I(0.5, 0.6, 0.2, 0.3), I(0.4, 0.5, 0.3, 0.4)],
     [I(0.7, 0.8, 0.1, 0.2), I(0.2, 0.3, 0.5, 0.6)]],
    [[I(0.4, 0.5, 0.3, 0.4), I(0.5, 0.7, 0.2, 0.3)],        # expert 2
     [I(0.4, 0.6, 0.2, 0.4), I(0.5, 0.6, 0.2, 0.3)],
     [I(0.6, 0.8, 0.1, 0.2), I(0.3, 0.4, 0.4, 0.5)]],
])
for method in ("codas", "marcos", "wisp", "electre"):
    r = rank_fuzzy(dm, [0.5, 0.5], ["cost", "benefit"], method=method, score="normalized")
    print(f"{method:8}", r.ranking, {k: round(v, 3) for k, v in r.scores.items()})
print(r.operators)
```

## Ranking on fuzzy numbers

`rank_fuzzy` with a crisp method is **score-first**: it aggregates the experts, scores every
number, and ranks the crisp matrix. The fuzzy methods are **score-last**: the weighted
ratings, the sums and the products are built with the number kind's own operators, and the
numbers are scored only at the end, as the published intuitionistic, Pythagorean and
neutrosophic variants of these methods do.

| | Score-first | Score-last |
|---|---|---|
| Call | `rank_fuzzy(dm, w, t, method="codas")` | [`fuzzy_codas`][bulanik.mcdm.fuzzy_ranking.fuzzy_codas], [`fuzzy_waspas`][bulanik.mcdm.fuzzy_ranking.fuzzy_waspas], [`fuzzy_cocoso`][bulanik.mcdm.fuzzy_ranking.fuzzy_cocoso], or `rank_fuzzy(dm, w, t, method="fuzzy_codas")` |
| Input | a `DecisionMatrix` of experts | an aggregated matrix of one number kind, or a `DecisionMatrix` through `rank_fuzzy` |
| Cost criterion | the crisp method's own treatment of the scores | the kind's `complement()`, so $(\mu, \nu) \mapsto (\nu, \mu)$ for an orthopair number; for `fuzzy_codas` on `TFN` and `TrFN`, the normalisation $1 - \tilde x / \max D$ |
| Score | needs positive values for the ratio methods | picks the negative-ideal (CODAS) or scores the final numbers |
| Kinds | every kind with a suitable score | `fuzzy_waspas` and `fuzzy_cocoso`: a cost criterion needs `complement()`, which `TFN`, `TrFN` and `IT2TrFN` lack; `fuzzy_codas` takes `TFN` and `TrFN` with any criteria, but not `IT2TrFN` (no `euclidean` and `hamming` distances) |

Use score-first when the kind has no `complement()` and a cost criterion is involved, or
when you want one comparison across the ten crisp methods. Use score-last when you follow a
study stated on the numbers, or when you want the fuzzy sum and product kept in `extras`.
Weights are crisp in both.

**`fuzzy_codas`** weights each rating, $\tilde u_{ij} = w_j \tilde x_{ij}$, and takes as
negative-ideal number $\tilde{ns}_j$ the $\tilde u_{ij}$ with the lowest score in its column.
With the kind's normalised Euclidean and Hamming distances,

$$
E_i = \sqrt{\sum_j d_E(\tilde u_{ij}, \tilde{ns}_j)^2}, \qquad
T_i = \sum_j d_H(\tilde u_{ij}, \tilde{ns}_j),
$$

and the relative assessment is that of crisp [CODAS](#codas), with `tau` and `psi`. The
score only picks the negative-ideal numbers. This is the form of Altın et al. (2026).

**`fuzzy_codas` on triangular and trapezoidal numbers** follows the fuzzy CODAS of
Keshavarz Ghorabaee et al. (2017), with crisp weights. Each column is normalised by its
largest score $D$ (the centroid by default, the paper's defuzzification),
$\tilde n_{ij} = \tilde x_{ij} / \max_i D(\tilde x_{ij})$ for a benefit and
$1 - \tilde x_{ij} / \max_i D(\tilde x_{ij})$ for a cost, so no `complement()` is needed;
then weighted, $\tilde r_{ij} = w_j \tilde n_{ij}$, with the lowest-scored $\tilde r_{ij}$ of
each column as negative ideal. The distances are the kind's `"euclidean"` and `"hamming"`
(Li 2007, see [Triangular numbers](numbers-triangular.md)), **summed** over the criteria,

$$
E_i = \sum_j d_E(\tilde r_{ij}, \tilde{ns}_j), \qquad
T_i = \sum_j d_H(\tilde r_{ij}, \tilde{ns}_j),
$$

and the assessment is the same. `operators` also records `"normalization": "max_score"`.
The largest score of every column must be positive.

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.mcdm import DecisionMatrix, fuzzy_codas, rank_fuzzy

m = [[TFN(2, 3, 4), TFN(1, 2, 3)], [TFN(4, 5, 6), TFN(3, 4, 6)], [TFN(1, 2, 4), TFN(2, 3, 4)]]
r = fuzzy_codas(m, [0.6, 0.4], ["benefit", "cost"], alternatives=["Atlas", "Borealis", "Cedar"],
                criteria=["capacity", "price"])
print(r.ranking, {k: round(v, 4) for k, v in r.scores.items()})
print(r.operators)
print(r.extras["negative_ideal"]["capacity"])

dm = DecisionMatrix(["Atlas", "Borealis", "Cedar"], ["capacity", "price"], [m])
print(rank_fuzzy(dm, [0.6, 0.4], ["benefit", "cost"], method="fuzzy_codas").ranking)
```

**`fuzzy_waspas`** builds the weighted sum and the weighted product as numbers,

$$
\tilde Q^{(1)}_i = \bigoplus_j w_j \tilde x_{ij}, \qquad
\tilde Q^{(2)}_i = \bigotimes_j \tilde x_{ij}^{\,w_j},
$$

that is [`wa`][bulanik.aggregate.wa] and [`wg`][bulanik.aggregate.wg] of the row. `combine`
selects how they meet $\lambda$. With `combine="fuzzy"`, the default, they are combined as
numbers and scored once, $Q_i = S(\lambda \tilde Q^{(1)}_i \oplus (1 - \lambda) \tilde Q^{(2)}_i)$.
With `combine="scores"`
they are scored first, $Q_i = \lambda S(\tilde Q^{(1)}_i) + (1 - \lambda) S(\tilde Q^{(2)}_i)$,
the form of crisp WASPAS.

**`fuzzy_cocoso`** builds the same two numbers, here $\tilde S_i$ and $\tilde P_i$, scores
them, and enters the scores in the appraisal scores of crisp [CoCoSo](#cocoso). The default
score is `"normalized"` where the kind has it. This follows the form of the neutrosophic
CoCoSo of Rani et al. (2021).

`score="defuzzified"` of an [`IVIFN`][bulanik.numbers.IVIFN] is the defuzzification of
Bolturk and Kahraman (2022, Definition 4)

$$
D = \frac{\mu_L + \mu_U + (1 - \nu_L) + (1 - \nu_U) + \mu_L \mu_U - \sqrt{(1 - \nu_L)(1 - \nu_U)}}{4}.
$$

```python exec="on" source="above" result="text"
from bulanik import IFN, IVIFN
from bulanik.mcdm import fuzzy_codas, fuzzy_cocoso, fuzzy_waspas

m = [[IFN(0.6, 0.3), IFN(0.7, 0.2), IFN(0.6, 0.3)],
     [IFN(0.4, 0.5), IFN(0.6, 0.3), IFN(0.4, 0.5)],
     [IFN(0.5, 0.4), IFN(0.8, 0.1), IFN(0.5, 0.4)]]
w, t = [0.4, 0.35, 0.25], ["cost", "benefit", "benefit"]
names = dict(alternatives=["Atlas", "Borealis", "Cedar"], criteria=["price", "quality", "delivery"])

for method in (fuzzy_codas, fuzzy_waspas, fuzzy_cocoso):
    r = method(m, w, t, **names)
    print(f"{r.method:13}", r.ranking, {k: round(v, 4) for k, v in r.scores.items()})
    print(" " * 13, r.operators)

s = fuzzy_waspas(m, w, t, combine="scores", **names)
print("combine=scores", s.ranking, {k: round(v, 4) for k, v in s.scores.items()})
q = fuzzy_waspas(m, w, t, **names).extras["Q"]["Cedar"]
print("Q of Cedar    ", round(q.mu, 4), round(q.nu, 4))
print("defuzzified   ", round(IVIFN(0.4, 0.5, 0.3, 0.4).score("defuzzified"), 4))
```

Through `rank_fuzzy` the experts are aggregated first, and `operators` also records
`aggregation`. With `score=None` the choice stays with the method.

```python exec="on" source="above" result="text"
from bulanik import IFN
from bulanik.mcdm import DecisionMatrix, rank_fuzzy

dm = DecisionMatrix(["Atlas", "Borealis", "Cedar"], ["price", "quality"], [
    [[IFN(0.6, 0.3), IFN(0.7, 0.2)], [IFN(0.4, 0.5), IFN(0.6, 0.3)], [IFN(0.5, 0.4), IFN(0.8, 0.1)]],
    [[IFN(0.5, 0.4), IFN(0.7, 0.2)], [IFN(0.4, 0.4), IFN(0.5, 0.4)], [IFN(0.6, 0.3), IFN(0.7, 0.2)]],
])
for method in ("codas", "fuzzy_codas", "waspas", "fuzzy_waspas", "cocoso", "fuzzy_cocoso"):
    r = rank_fuzzy(dm, [0.5, 0.5], ["cost", "benefit"], method=method, score="normalized")
    print(f"{method:13}", r.ranking, r.operators["aggregation"])
```

[Example 24](../examples/24-fuzzy-ranking.md) compares both routes on a larger matrix.

## Pitfalls

**Signed default scores break the ratio methods.** CODAS, MARCOS, WISP and WASPAS need
strictly positive entries, COPRAS non-negative ones, and the default score of an
orthopair or interval-valued number is a signed difference in $[-1, 1]$. Through
`rank_fuzzy` this is a clear `ValueError`, never a sign-flipped ranking. For a kind that
has the score, pass `score="normalized"` (a score in $[0, 1]$). For a kind without it,
such as `IVNeutrosophicFN`, the message says to use a positive-valued score.

**`fuzzy_codas` on picture numbers.** `PictureFN.scale(w)` leaves the valid domain for
almost any weight below 1, so `fuzzy_codas` raises a `ValueError` naming the criterion and
the weight. Use `fuzzy_waspas`, `fuzzy_cocoso` or `rank_fuzzy(dm, w, t, method="codas",
score="normalized")` for picture numbers.

**Zero and negative entries.** CODAS, MARCOS and WISP divide by entries or take ratios,
so a zero or negative entry raises `ValueError` naming the method. COPRAS rejects negative
entries, a column that sums to zero, and an alternative whose cost sum $S^-$ is 0. CoCoSo and ELECTRE accept any finite numbers.

**CoCoSo fails when one alternative is worst everywhere.** See the warning above.

**ELECTRE's `outranks` is a partial relation and may be empty.** With strict thresholds
nobody outranks anybody, yet `ranking` is still complete, because it comes from the net
indices. Do not read the ranking as a statement that each alternative outranks the next.

**ELECTRE with `cost="inverse"` needs positive cost columns.** Use `"reverse"` for zeros.

**Ties and identical alternatives are defined.** Identical alternatives have
discordance 0 in ELECTRE; ties keep input order.

**Score-first and score-last need not agree.** They are different methods, and CODAS in
particular can change its order, since its distances are taken on different objects. Compare
ranks across them, not score values.

**A cost criterion needs `complement()`.** `fuzzy_waspas` and `fuzzy_cocoso` raise
`TypeError` for a cost criterion on a `TFN`, `TrFN` or `IT2TrFN`, and `fuzzy_codas` for one
on an `IT2TrFN` (it takes `TFN` and `TrFN` costs through its normalisation). Use
`rank_fuzzy` with a crisp method, or rate the criterion as a benefit.

**`fuzzy_cocoso` needs non-negative scores.** Its default is `"normalized"` where the kind
has it; a signed score such as `"difference"` raises `ValueError` as soon as one sum or
product scores below 0.

**`fuzzy_waspas` has two meanings of `lam`.** `combine="fuzzy"` and `combine="scores"` give
close values on close numbers, but they are not the same formula. State which one you report.

**`score="defuzzified"` is for `IVIFN`.** On any other rung it raises `ValueError`.

```python exec="on" source="above" result="text"
from bulanik import IVNeutrosophicFN
from bulanik.mcdm import DecisionMatrix, cocoso, codas, electre, marcos, rank_fuzzy, wisp

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

X, W, T = [[0, 2], [3, 4]], [0.5, 0.5], ["cost", "benefit"]
show(lambda: marcos(X, W, T))
show(lambda: codas(X, W, T))
show(lambda: cocoso([[1, 1], [1, 1]], W, T))                      # identical alternatives
show(lambda: cocoso([[1, 5], [2, 4], [3, 3]], W, T, alternatives=["A", "B", "C"]))
show(lambda: codas([[1, 2], [3, 4]], W, T, psi="paper"))
show(lambda: codas([[1, 2], [3, 4]], W, T, tau=2))
show(lambda: wisp([[1, 2], [3, 4]], W, T, product="mul"))
show(lambda: electre([[1, 2]], W, T))                             # needs two alternatives
show(lambda: electre(X, W, T, cost="inverse"))
show(lambda: electre(X, W, T, cost="flip"))

N = IVNeutrosophicFN
dm = DecisionMatrix(["A", "B"], ["c1"], [[[N(0, 0, 1, 1, 1, 1)], [N(0.6, 0.7, 0.1, 0.2, 0.2, 0.3)]]])
show(lambda: rank_fuzzy(dm, [1.0], ["benefit"], method="marcos"))   # a score of 0

from bulanik import IFN, TFN, IVPFN
from bulanik.mcdm import fuzzy_codas, fuzzy_cocoso, fuzzy_waspas

tfn = [[TFN(1, 2, 3), TFN(2, 3, 4)], [TFN(2, 3, 4), TFN(1, 2, 3)]]
show(lambda: fuzzy_waspas(tfn, W, T))                              # no complement() for a cost
show(lambda: fuzzy_codas([[TFN(-2, -1, 0), TFN(1, 2, 3)], [TFN(-3, -2, -1), TFN(2, 3, 4)]], W, T))
low = [[IFN(0.1, 0.8), IFN(0.2, 0.7)], [IFN(0.2, 0.7), IFN(0.1, 0.8)]]
show(lambda: fuzzy_cocoso(low, W, ["benefit", "benefit"], score="difference"))
show(lambda: IVPFN(0.4, 0.5, 0.3, 0.4).score("defuzzified"))
```

## See also

- [Ranking](ranking.md) for TOPSIS, VIKOR, WASPAS, EDAS and `rank_fuzzy`, and
  [Weighting](weighting.md) and [More weighting methods](weighting-more.md) for the
  weights.
- [Choosing a method](../concepts/choosing.md) for when to use which.
- [Interval-valued numbers](numbers-interval.md) and
  [Interval type-2 trapezoids](numbers-it2.md) for the fuzzy kinds that feed `rank_fuzzy`.
- API: [`bulanik.mcdm.scoring`][bulanik.mcdm.scoring],
  [`bulanik.mcdm.outranking`][bulanik.mcdm.outranking],
  [`bulanik.mcdm.fuzzy_ranking`][bulanik.mcdm.fuzzy_ranking].
- Keshavarz Ghorabaee, Zavadskas, Turskis and Antucheviciene (2016), *Economic Computation
  and Economic Cybernetics Studies and Research* 50(3), 25-44 (CODAS); Chen and Gou (2021)
  on its threshold function; Zavadskas, Kaklauskas and Sarka (1994), *Technological and
  Economic Development of Economy* 1(3), 131-139, and Razavi Hajiagha, Hashemi and
  Zavadskas (2013), *Technological and Economic Development of Economy* 19(1), 22-37
  (COPRAS); Stevic, Pamucar, Puska and Chatterjee (2020), *Computers & Industrial
  Engineering* 140, 106231, with the restatement of Ulutas et al. (2020), *Mathematics*
  8(10), 1672 (MARCOS); Yazdani, Zarate, Zavadskas and Turskis (2019), *Management Decision*
  57(9), 2501-2519, and Rasoanaivo, Rakotonirainy and Zarate (2024), *Expert Systems with
  Applications* 251, 124079 (CoCoSo); Stanujkic, Popovic, Karabasevic, Meidute-Kavaliauskiene
  and Ulutas (2023), *IEEE Transactions on Engineering Management* 70(5), 1933-1944, with
  the restatement in M. Stanujkic (2023), *Journal of Process Management and New
  Technologies* 11(3-4), 79-88 (WISP); Roy (1968), *Revue francaise d'informatique et de
  recherche operationnelle* 2(8), 57-75, Hwang and Yoon (1981), *Multiple Attribute
  Decision Making*, Springer, and van Delft and Nijkamp (1977), *Multi-criteria Analysis
  and Regional Decision-making*, Martinus Nijhoff (ELECTRE). Rani, Ali, Krishankumar,
  Mishra, Cavallaro and Ravichandran (2021), *Energies* 14(15), 4594 (SVN-CoCoSo).
