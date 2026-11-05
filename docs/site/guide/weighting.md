# Weighting methods

A weighting method turns judgements or data into **criteria weights** that sum to 1.
bulanik has four, all returning a [`WeightingResult`][bulanik.mcdm.results.WeightingResult],
and a fifth, the best-worst method, that needs an optimiser and so lives in the
[LumiX](bridge-lumix.md) and [sezgi](bridge-sezgi.md) bridges.

| Method | Function | Input | Kind |
|---|---|---|---|
| AHP | [`ahp`][bulanik.mcdm.weighting.ahp] | crisp pairwise comparison matrix | subjective |
| Fuzzy AHP | [`fuzzy_ahp`][bulanik.mcdm.weighting.fuzzy_ahp] | matrix of triangular numbers | subjective |
| SWARA | [`swara`][bulanik.mcdm.weighting.swara] | comparative importances of an ordered list | subjective |
| Entropy | [`entropy`][bulanik.mcdm.weighting.entropy] | decision matrix (alternatives by criteria) | objective |
| BWM | `bwm_linear`, `BWMProblem`, `FuzzyBWMProblem` | best-to-others and others-to-worst vectors | subjective |

*Subjective* methods encode what the decision makers think; *objective* ones read the
weights off the data. Every result carries `weights` (a dict), `ranking` (criteria,
largest weight first), `operators` (the choices behind the numbers) and, for the
AHP-based methods, `consistency_ratio`. Pass a result straight to a
[ranking method](ranking.md) as its `weights`.

<figure markdown="span">
  ![Three panels of horizontal bars for price, quality and delivery: AHP weights 0.540, 0.297, 0.163 with CR 0.008; SWARA weights 0.419, 0.349, 0.233; entropy weights with delivery largest at 0.597](../assets/figures/mcdm-weights-compare-light.svg#only-light)
  ![Three panels of horizontal bars for price, quality and delivery: AHP weights 0.540, 0.297, 0.163 with CR 0.008; SWARA weights 0.419, 0.349, 0.233; entropy weights with delivery largest at 0.597](../assets/figures/mcdm-weights-compare-dark.svg#only-dark)
  <figcaption markdown="span">Three methods, three answers for the same three criteria, drawn
  by [`plot.weights`][bulanik.plot.weights]. AHP and SWARA are judgement-based and agree
  on the order; entropy ignores judgement and rewards the criterion whose values differ
  most between the four suppliers, here delivery. The AHP panel shows its consistency
  ratio in the corner.</figcaption>
</figure>

## AHP

Saaty's **analytic hierarchy process** asks for a square matrix $A$ in which $a_{ij}$ says
how many times more important criterion $i$ is than $j$ (the usual scale is 1 to 9), with
$a_{ji} = 1/a_{ij}$ and ones on the diagonal. The weights are the **principal
eigenvector** of $A$ normalised to sum 1; bulanik finds it by power iteration.

The matrix is rarely perfectly consistent ($a_{ij} a_{jk} = a_{ik}$), so AHP also reports
how far it is from consistent. With the principal eigenvalue $\lambda_{\max}$ and order $n$,

$$
CI = \frac{\lambda_{\max} - n}{n - 1}, \qquad CR = \frac{\max(0, CI)}{RI_n},
$$

where $RI_n$ is Saaty's **random index**, the average $CI$ of random reciprocal matrices
of order $n$. The code uses this table:

| $n$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| $RI$ | 0 | 0 | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 |

Hence matrices larger than 10 are rejected (there is no random index beyond 10), and for
$n \le 2$ the ratio is 0 (every 1x1 and 2x2 reciprocal matrix is consistent). A ratio below
about 0.1 is conventionally acceptable.

```python exec="on" source="above" result="text"
from bulanik.mcdm import ahp

names = ["cost", "quality", "speed"]
r = ahp([[1, 3, 5], [1/3, 1, 2], [1/5, 1/2, 1]], names)
print({k: round(v, 3) for k, v in r.weights.items()})
print(r.ranking, round(r.consistency_ratio, 3), r.operators)

# cost beats quality 3:1 and quality beats speed 3:1, but speed beats cost 3:1: a cycle
cyclic = ahp([[1, 3, 1/3], [1/3, 1, 3], [3, 1/3, 1]], names)
print({k: round(v, 3) for k, v in cyclic.weights.items()}, round(cyclic.consistency_ratio, 3))
```

The cyclic matrix has equal weights and a ratio far above 0.1: AHP will always give you
*some* weights, so check `consistency_ratio` before trusting them.

## Fuzzy AHP

[`fuzzy_ahp`][bulanik.mcdm.weighting.fuzzy_ahp] takes the same matrix of triangular
numbers ([`TFN`][bulanik.TFN]), so each judgement can carry its own uncertainty. It
follows **Buckley's (1985)** geometric-mean method:

1. The fuzzy geometric mean of row $i$ is $\tilde r_i = (\prod_j \tilde a_{ij})^{1/n}$,
   taken componentwise on $(l, m, u)$.
2. The fuzzy weight is $\tilde w_i = \tilde r_i \otimes (\sum_k \tilde r_k)^{-1}$.
3. Each $\tilde w_i$ is defuzzified with `defuzzification` (a TFN score name: `"centroid"`,
   the default, `"graded_mean"` or `"mean"`) and the crisp values are normalised to sum 1.

The result keeps the fuzzy weights in `fuzzy_weights`. The consistency ratio is Saaty's,
computed on the **defuzzified** matrix with the table above. The matrix must be
reciprocal in the fuzzy sense, $\tilde a_{ji} = 1/\tilde a_{ij}$ (relative tolerance
$10^{-9}$), with every lower bound positive.
[`SAATY_TFN`][bulanik.scales.SAATY_TFN] and `LinguisticScale.reciprocal` build such
entries from the words of the 1-9 scale.

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.mcdm import fuzzy_ahp
from bulanik.scales import SAATY_TFN

one = SAATY_TFN["1"]
three, five = SAATY_TFN["3"], SAATY_TFN["5"]
matrix = [
    [one,                        three,                      five],
    [SAATY_TFN.reciprocal("3"),  one,                        three],
    [SAATY_TFN.reciprocal("5"),  SAATY_TFN.reciprocal("3"),  one],
]
r = fuzzy_ahp(matrix, ["cost", "quality", "speed"])
print({k: round(v, 3) for k, v in r.weights.items()})
print("CR:", round(r.consistency_ratio, 3), r.operators)
print("cost, fuzzy:", r.fuzzy_weights["cost"])
g = fuzzy_ahp(matrix, ["cost", "quality", "speed"], defuzzification="graded_mean")
print({k: round(v, 3) for k, v in g.weights.items()})
```

## SWARA

**Step-wise weight assessment ratio analysis** (Kersuliene et al. 2010) needs no matrix:
rank the criteria from most to least important, then for each criterion say how much
less important it is than the one before it. With these **comparative importances**
$s_2, \dots, s_n$ the code computes

$$
q_1 = 1, \qquad q_j = \frac{q_{j-1}}{s_j + 1}, \qquad w_j = \frac{q_j}{\sum_k q_k}.
$$

An $s_j$ of 0 means "as important as the previous one", and 0.3 means "30% less
important". The names are given **most important first** and there are $n - 1$ values.
The importances may be crisp (non-negative) or all fuzzy (any fuzzy number kind); fuzzy
ones are scored first, with `score=` or the kind's default, and `operators["score"]`
records which. There is no consistency ratio.

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.mcdm import swara

r = swara([0.3, 0.2], ["a", "b", "c"])
print({k: round(v, 3) for k, v in r.weights.items()}, r.operators)

fuzzy = swara([TFN(0.2, 0.3, 0.4), TFN(0.1, 0.2, 0.3)], ["a", "b", "c"])
print({k: round(v, 3) for k, v in fuzzy.weights.items()}, fuzzy.operators)
print(swara([TFN(0.2, 0.3, 0.4)], ["a", "b"], score="mean").operators)
```

## Entropy

The **entropy** method is objective: it uses only the decision matrix. With
$p_{ij} = x_{ij} / \sum_i x_{ij}$ for $m$ alternatives, criterion $j$ has entropy and
divergence

$$
e_j = -\frac{1}{\ln m} \sum_i p_{ij} \ln p_{ij}, \qquad d_j = 1 - e_j,
\qquad w_j = \frac{d_j}{\sum_k d_k},
$$

with $0 \ln 0 = 0$. A criterion on which the alternatives are all alike has entropy near
1, carries little information and gets little weight; one on which they differ a lot gets
more. Entropy says nothing about what *matters* to you, only about what
*discriminates*, so it is often combined with, not substituted for, a subjective method.

```python exec="on" source="above" result="text"
from bulanik.mcdm import entropy

r = entropy([[1, 9], [2, 8], [3, 7]], ["a", "b"])
print({k: round(v, 3) for k, v in r.weights.items()})   # a (1, 2, 3) differs more, relatively, than b (9, 8, 7)
print(r.operators, r.consistency_ratio)
```

## Best-worst method

The **best-worst method** (BWM, Rezaei 2015) asks for only $2n - 3$ comparisons: the best
criterion $B$ against every other ($a_{Bj}$) and every other against the worst criterion
$W$ ($a_{jW}$), on a 1 to 9 scale. The weights are those that best reproduce the
comparisons, which is an optimisation problem, solved by the
[LumiX bridge](bridge-lumix.md) (linear form) or the [sezgi bridge](bridge-sezgi.md)
(nonlinear and fuzzy forms). Its input checks and two consistency indices are
solver-free, so they live in [`bulanik.mcdm.bwm`][bulanik.mcdm.bwm]:

- [`bwm_consistency_index(a_bw)`][bulanik.mcdm.bwm.bwm_consistency_index] is Rezaei's
  largest attainable deviation $\xi$ for a best-over-worst comparison $a = a_{BW}$,
  $CI(a) = \bigl((1 + 2a) - \sqrt{8a + 1}\bigr)/2$, and
- [`fuzzy_bwm_consistency_index(a_bw)`][bulanik.mcdm.bwm.fuzzy_bwm_consistency_index] is
  Guo and Zhao's for a triangular $\tilde a_{BW}$ with upper bound $u$,
  $CI(u) = \bigl((1 + 2u) + \sqrt{8u + 1}\bigr)/2$ (the *larger* root, so it is never 0).

```python exec="on" source="above" result="text"
from bulanik.mcdm.bwm import bwm_consistency_index, fuzzy_bwm_consistency_index
from bulanik.scales import FUZZY_BWM_TFN

print("a_BW  ", *range(1, 10), sep="\t")
print("CI(a) ", *[f"{bwm_consistency_index(a):.2f}" for a in range(1, 10)], sep="\t")
print()
for term in FUZZY_BWM_TFN.terms:
    tfn = FUZZY_BWM_TFN[term]
    print(f"{term}  {str(tfn):<44} CI = {fuzzy_bwm_consistency_index(tfn):.2f}")
```

The consistency *ratio* of a BWM result is $\xi / CI$, which is why the ratio does not
exist for $a_{BW} = 1$ in the crisp model (see
[the sezgi bridge](bridge-sezgi.md#consistency_ratio-is-not-one-quantity)).

## Pitfalls

**AHP matrices must be exactly reciprocal.** The check is a relative tolerance of
$10^{-9}$ on $a_{ij} a_{ji} = 1$, the diagonal must be 1, entries must be positive and the
order at most 10. Write `1/3`, not `0.33`.

**A fuzzy AHP entry must be a `TFN` with $l > 0$.** Plain numbers and other number kinds
are rejected, and so is a $\tilde a_{ji}$ that is not the reciprocal of $\tilde a_{ij}$.

**SWARA wants the criteria sorted by you.** The function does not sort: the first name is
taken as the most important. With fuzzy importances a negative score is an error; use
`score="normalized"` if the default score can go negative.

**Entropy needs variation.** Fewer than 2 alternatives, a negative entry, a column that
sums to 0, or a matrix whose columns are all constant is an error (no criterion
discriminates).

**A small `consistency_ratio` is a necessary check, not proof.** It says the judgements
do not contradict each other, not that they are right.

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.mcdm import ahp, entropy, fuzzy_ahp, swara

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: ahp([[1, 3], [0.33, 1]]))                       # not reciprocal
show(lambda: ahp([[2, 1], [1, 2]]))                          # diagonal is not 1
show(lambda: ahp([[1, 0], [0, 1]]))                          # not positive
show(lambda: ahp([[1.0] * 11 for _ in range(11)]))           # no random index beyond 10
show(lambda: ahp([[1, 3], [1/3, 1]], ["a", "a"]))            # duplicate names
show(lambda: fuzzy_ahp([[1, 3], [1/3, 1]]))                  # not TFN entries
show(lambda: fuzzy_ahp([[TFN(1, 1, 1), TFN(2, 3, 4)], [TFN(1, 1, 1), TFN(1, 1, 1)]]))  # not reciprocal
show(lambda: swara([0.3], ["a", "b", "c"]))                  # needs n - 1 = 2 values
show(lambda: swara([0.3, TFN(0.1, 0.2, 0.3)], ["a", "b", "c"]))   # crisp and fuzzy mixed
show(lambda: swara([-0.5], ["a", "b"]))
show(lambda: entropy([[1, 2], [1, 2]]))                      # no criterion discriminates
show(lambda: entropy([[1, 2]]))                              # one alternative
```

## See also

- [More weighting methods](weighting-more.md) for CRITIC, PIPRECIA, `ahp_fuzzy_numbers`
  and DEMATEL.
- [Ranking](ranking.md): pass a result straight to `topsis`, `vikor`, `waspas`, `edas`.
- [LumiX bridge](bridge-lumix.md) and [sezgi bridge](bridge-sezgi.md): the best-worst
  method.
- [Aggregation](aggregation.md) to combine several experts' judgements before weighting,
  [Sets and scales](sets-and-scales.md) for [`SAATY_TFN`][bulanik.scales.SAATY_TFN] and
  friends.
- API: [`bulanik.mcdm`][bulanik.mcdm], [`bulanik.plot.weights`][bulanik.plot.weights].
- Examples: [03 fuzzy AHP](../examples/03-fuzzy-ahp.md),
  [04 SWARA and TOPSIS](../examples/04-swara-topsis-supplier.md).
- Saaty (1980), *The Analytic Hierarchy Process*, McGraw-Hill; Buckley (1985), "Fuzzy
  hierarchical analysis", *Fuzzy Sets and Systems* 17(3); Kersuliene, Zavadskas and
  Turskis (2010), "Selection of rational dispute resolution method by applying new
  step-wise weight assessment ratio analysis (SWARA)", *Journal of Business Economics and
  Management* 11(2); Shannon (1948), "A mathematical theory of communication", *Bell System
  Technical Journal* 27; Hwang and Yoon (1981), *Multiple Attribute Decision Making*,
  Springer (the entropy procedure; see also Zeleny 1982); Rezaei (2015), "Best-worst multi-criteria decision-making
  method", *Omega* 53; Guo and Zhao (2017), "Fuzzy best-worst multi-criteria
  decision-making method and its applications", *Knowledge-Based Systems* 121.
