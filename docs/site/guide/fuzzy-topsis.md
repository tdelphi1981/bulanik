# Fuzzy TOPSIS

[`fuzzy_topsis`][bulanik.mcdm.fuzzy.fuzzy_topsis] is **Chen's (2000) fuzzy TOPSIS** for
ratings that are triangular fuzzy numbers. Unlike [`rank_fuzzy`](ranking.md#fuzzy-ratings-decisionmatrix-and-rank_fuzzy),
which scores each fuzzy rating to a crisp value up front, it keeps the numbers fuzzy
through normalisation, weighting and the distances, and only the final closeness
coefficient is crisp. Use it when the ratings are already triangles, one per alternative
and criterion (aggregate several experts first with
[`aggregate_experts`][bulanik.aggregate.aggregate_experts]), and the weights are triangles
or crisp numbers in $[0, 1]$.

## The method

For a matrix of $\tilde x_{ij} = (l_{ij}, m_{ij}, u_{ij})$ the code does the following.

**1. Normalise** each column to $[0, 1]$ with a *linear scale*, not a vector norm:

- benefit: $\tilde r_{ij} = (l_{ij}, m_{ij}, u_{ij}) / c_j^*$ with $c_j^* = \max_i u_{ij}$;
- cost: $\tilde r_{ij} = (a_j^-/u_{ij},\; a_j^-/m_{ij},\; a_j^-/l_{ij})$ with
  $a_j^- = \min_i l_{ij}$.

**2. Weight**: $\tilde v_{ij} = \tilde r_{ij} \otimes \tilde w_j$, componentwise on
$(l, m, u)$.

**3. Ideal solutions** are fixed, not taken from the data: the fuzzy positive ideal
$(1,1,1)$ and negative ideal $(0,0,0)$ for every criterion.

**4. Distance** between triangles is the **vertex distance**

$$
d(\tilde a, \tilde b) = \sqrt{\tfrac13\bigl[(a_l - b_l)^2 + (a_m - b_m)^2 + (a_u - b_u)^2\bigr]}.
$$

**5. Closeness coefficient**: with $d_i^* = \sum_j d(\tilde v_{ij}, (1,1,1))$ and
$d_i^- = \sum_j d(\tilde v_{ij}, (0,0,0))$,

$$
CC_i = \frac{d_i^-}{d_i^* + d_i^-},
$$

larger is better (0.5 if both sums are 0). `extras` holds `d_star` and `d_minus`, and
`operators` is `{"normalization": "linear_scale_chen2000", "distance": "vertex"}`.

The weights come in the same forms as for the crisp methods: a sequence, a mapping by
criterion name or a weighting result. Each entry is a `TFN` or a crisp number, and
**must lie in $[0, 1]$**; they need not sum to 1.

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.mcdm import fuzzy_topsis

m = [[TFN(5, 7, 9), TFN(3, 5, 7)],
     [TFN(7, 9, 9), TFN(1, 3, 5)],
     [TFN(3, 5, 7), TFN(7, 9, 9)]]
r = fuzzy_topsis(m, {"quality": TFN(.5, .7, .9), "price": 0.3},
                 ["benefit", "cost"], alternatives=["X", "Y", "Z"])
print(r.ranking)
print({k: round(v, 3) for k, v in r.scores.items()})
print({k: round(v, 3) for k, v in r.extras["d_star"].items()})
print({k: round(v, 3) for k, v in r.extras["d_minus"].items()})
print(r.operators)
```

To see that the code does what the steps say, redo alternative X by hand. Quality is a
benefit with $c^* = 9$; price is a cost with $a^- = \min l = 1$.

```python exec="on" source="above" result="text"
import math
from bulanik import TFN

def vertex(a, b):
    return math.sqrt(((a.l - b.l) ** 2 + (a.m - b.m) ** 2 + (a.u - b.u) ** 2) / 3)

quality, price = TFN(5, 7, 9), TFN(3, 5, 7)          # alternative X
r_q = TFN(5 / 9, 7 / 9, 9 / 9)                       # benefit: divide by c* = 9
r_p = TFN(1 / 7, 1 / 5, 1 / 3)                       # cost: (1/u, 1/m, 1/l)
v_q = TFN(r_q.l * .5, r_q.m * .7, r_q.u * .9)        # times w = (.5, .7, .9)
v_p = TFN(r_p.l * .3, r_p.m * .3, r_p.u * .3)        # times w = 0.3 as (.3, .3, .3)
one, zero = TFN(1, 1, 1), TFN(0, 0, 0)
d_star = vertex(v_q, one) + vertex(v_p, one)
d_minus = vertex(v_q, zero) + vertex(v_p, zero)
print(round(d_star, 3), round(d_minus, 3), round(d_minus / (d_star + d_minus), 3))
```

## Chen's method or `rank_fuzzy`?

| | `fuzzy_topsis` | `rank_fuzzy(method="topsis")` |
|---|---|---|
| Number kinds | `TFN` only | any kind |
| Experts | one matrix (aggregate beforehand) | a `DecisionMatrix` of experts, aggregated inside |
| Normalisation | linear scale to $[0, 1]$ | vector norm, after scoring |
| Ideal solutions | fixed $(1,1,1)$ and $(0,0,0)$ | column best and worst |
| Distance | vertex distance on triangles | Euclidean on crisp scores |
| Weights | in $[0, 1]$, fuzzy allowed | sum to 1, crisp |

They answer the same question with different arithmetic and can order close alternatives
differently. Pick by the number kind and by whether you want the spread of the ratings to
influence the distances.

## Pitfalls

**Weights outside $[0, 1]$ are rejected.** Because the ideal solutions are $(1,1,1)$ and
$(0,0,0)$, a weight above 1 would push a weighted rating beyond the "ideal". A weight on
a 0-10 scale must be divided by 10. A `TFN` weight needs $l \ge 0$ and $u \le 1$.

**Sign requirements on the ratings.** A benefit column needs $l \ge 0$ and a positive
upper bound somewhere; a cost column needs $l > 0$ (the code divides by the ratings).

**Only `TFN` ratings.** For other number kinds use
[`rank_fuzzy`](ranking.md#fuzzy-ratings-decisionmatrix-and-rank_fuzzy).

**Normalisation depends on the whole column.** Adding or removing an alternative changes
$c_j^*$ and $a_j^-$ and therefore every score, so rank reversal is possible, as in
the crisp TOPSIS.

```python exec="on" source="above" result="text"
from bulanik import TFN
from bulanik.mcdm import fuzzy_topsis

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

m = [[TFN(5, 7, 9), TFN(3, 5, 7)], [TFN(7, 9, 9), TFN(1, 3, 5)]]
types = ["benefit", "cost"]
show(lambda: fuzzy_topsis(m, [5, 0.3], types))                       # weight above 1
show(lambda: fuzzy_topsis(m, [TFN(0.5, 0.9, 1.2), 0.3], types))      # TFN weight above 1
show(lambda: fuzzy_topsis(m, [-0.1, 0.3], types))
show(lambda: fuzzy_topsis([[TFN(-1, 2, 3)], [TFN(1, 2, 3)]], [0.5], ["benefit"]))
show(lambda: fuzzy_topsis([[TFN(0, 2, 3)], [TFN(1, 2, 3)]], [0.5], ["cost"]))
show(lambda: fuzzy_topsis([[5, 7], [1, 2]], [0.5, 0.5], types))      # crisp ratings
show(lambda: fuzzy_topsis(m, [0.5, 0.5], ["benefit"]))
```

## See also

- [Ranking](ranking.md) for the crisp methods and `rank_fuzzy`, [Weighting](weighting.md)
  for the weights, and
  [Triangular numbers](numbers-triangular.md) for `TFN`.
- API: [`fuzzy_topsis`][bulanik.mcdm.fuzzy.fuzzy_topsis], [`bulanik.mcdm`][bulanik.mcdm].
- Chen (2000), "Extensions of the TOPSIS for group decision-making under fuzzy
  environment", *Fuzzy Sets and Systems* 114(1), 1-9.
