# Orthopair numbers: IFN, PFN, FFN, QROFN

An **orthopair number** records an opinion as two degrees: how strongly you
*support* something (membership $\mu$) and how strongly you *oppose* it
(non-membership $\nu$). Unlike a classical fuzzy number, the two need not add up
to 1. What is left over is **hesitancy**: "I am not sure". Use them for ratings
from people, such as expert judgements, where a single degree hides doubt.

The constraint that keeps the pair meaningful depends on the **rung** $q$:

$$
\mu^q + \nu^q \le 1, \qquad \mu, \nu \in [0, 1].
$$

| Class | $q$ | Name | Constraint |
|---|---|---|---|
| [`IFN`][bulanik.IFN] | 1 | intuitionistic (Atanassov 1986) | $\mu + \nu \le 1$ |
| [`PFN`][bulanik.PFN] | 2 | Pythagorean (Yager 2013) | $\mu^2 + \nu^2 \le 1$ |
| [`FFN`][bulanik.FFN] | 3 | Fermatean (Senapati and Yager 2020) | $\mu^3 + \nu^3 \le 1$ |
| [`QROFN`][bulanik.QROFN] | any integer $\ge 1$ | q-rung orthopair (Yager 2017) | $\mu^q + \nu^q \le 1$ |

A higher rung admits more pairs: $(0.8, 0.5)$ is invalid for `IFN` but fine for
`PFN`. The hesitancy is $\pi = (1 - \mu^q - \nu^q)^{1/q}$.

<figure markdown="span">
  ![Nested feasible regions for q = 1, 2, 3 and 5 in the μ–ν plane, with one IFN, PFN, FFN and q = 5 number and their hesitancy segments](../assets/figures/orthopair-regions-light.svg#only-light)
  ![Nested feasible regions for q = 1, 2, 3 and 5 in the μ–ν plane, with one IFN, PFN, FFN and q = 5 number and their hesitancy segments](../assets/figures/orthopair-regions-dark.svg#only-dark)
  <figcaption markdown="span">The regions are nested: the feasible set grows with $q$.</figcaption>
</figure>

`IFN`, `PFN` and `FFN` are `QROFN` with the rung fixed, so everything below
holds for all four. You build `IFN(mu, nu)`, `PFN(mu, nu)`, `FFN(mu, nu)` and
`QROFN(mu, nu, q)`.

```python exec="on" source="above" result="text"
from bulanik import FFN, IFN, PFN, QROFN

x = IFN(0.6, 0.3)
print(x, x.q, round(x.hesitancy, 4))
print(PFN(0.8, 0.5).q, round(PFN(0.8, 0.5).hesitancy, 4))
print(FFN(0.8, 0.6), QROFN(0.9, 0.5, 3))
print(x.complement())                      # membership and non-membership swapped
print(isinstance(x, QROFN))                # the fixed-q classes are QROFN subclasses
```

## One implementation, two operator families

Every operation is the intuitionistic one applied to the $q$-th powers
$a = \mu^q$ and $b = \nu^q$, followed by the $q$-th root of each result part.
Two operator families exist, selected by the name `ops="algebraic"` (the default,
and what `+ * **` use) or `ops="einstein"`. For two numbers with powers
$(a_1, b_1)$ and $(a_2, b_2)$ the code computes:

| Operation | `"algebraic"` | `"einstein"` |
|---|---|---|
| `add` | $(a_1 + a_2 - a_1 a_2,\; b_1 b_2)$ | $\left(\dfrac{a_1 + a_2}{1 + a_1 a_2},\; \dfrac{b_1 b_2}{1 + (1 - b_1)(1 - b_2)}\right)$ |
| `mul` | $(a_1 a_2,\; b_1 + b_2 - b_1 b_2)$ | $\left(\dfrac{a_1 a_2}{1 + (1 - a_1)(1 - a_2)},\; \dfrac{b_1 + b_2}{1 + b_1 b_2}\right)$ |
| `scale` by $k$ | $(1 - (1 - a)^k,\; b^k)$ | $\left(\dfrac{(1+a)^k - (1-a)^k}{(1+a)^k + (1-a)^k},\; \dfrac{2 b^k}{(2-b)^k + b^k}\right)$ |
| `power` by $k$ | $(a^k,\; 1 - (1 - b)^k)$ | $\left(\dfrac{2 a^k}{(2-a)^k + a^k},\; \dfrac{(1+b)^k - (1-b)^k}{(1+b)^k + (1-b)^k}\right)$ |

Sum and product are not the same as arithmetic on crisp numbers: adding two
opinions *raises* membership and *lowers* non-membership, while multiplying them
does the opposite: membership falls and non-membership rises. Scale and power need
$k \ge 0$. The Einstein family gives similar but not identical results.

<figure markdown="span">
  ![Arrows from IFN x to the algebraic and Einstein sums and products of x and y](../assets/figures/orthopair-operators-light.svg#only-light)
  ![Arrows from IFN x to the algebraic and Einstein sums and products of x and y](../assets/figures/orthopair-operators-dark.svg#only-dark)
  <figcaption markdown="span">The two operator families send the same pair of numbers to
  different places.</figcaption>
</figure>

```python exec="on" source="above" result="text"
from bulanik import IFN, PFN

a, b = IFN(0.6, 0.3), IFN(0.5, 0.4)

def pair(n):
    return f"({n.mu:.4f}, {n.nu:.4f})"

print("a + b            ", pair(a + b))
print("a.add(b, einstein)", pair(a.add(b, ops="einstein")))
print("a * b            ", pair(a * b))
print("a.mul(b, einstein)", pair(a.mul(b, ops="einstein")))
print("2 * a            ", pair(2 * a))
print("a ** 2           ", pair(a ** 2))
print("PFN a ** 2       ", pair(PFN(0.6, 0.5) ** 2))
print(IFN.OPS, PFN.OPS)
```

## Scores, accuracy and distance

A score ranks numbers; an accuracy breaks ties. With $a = \mu^q$, $b = \nu^q$:

| Quantity | Names (default first) | Formula |
|---|---|---|
| score | `"difference"` | $\mu^q - \nu^q$ |
| | `"normalized"` | $(1 + \mu^q - \nu^q)/2$, a score in $[0, 1]$ |
| accuracy | `"sum"` | $\mu^q + \nu^q$ (the certainty: one minus the hesitancy power) |
| distance | `"hamming"` | $(\lvert \Delta a \rvert + \lvert \Delta b \rvert + \lvert \Delta p \rvert)/2$ |
| | `"euclidean"` | $\sqrt{(\Delta a^2 + \Delta b^2 + \Delta p^2)/2}$ |

where $\Delta$ is the difference between the two numbers and $p = 1 - a - b$ is the hesitancy power $\pi^q$. Two numbers can have the same
score and still differ in accuracy: `IFN(0.6, 0.3)` and `IFN(0.5, 0.2)` both
score 0.3, but the first is more certain.

```python exec="on" source="above" result="text"
from bulanik import IFN

x, y = IFN(0.6, 0.3), IFN(0.5, 0.2)
print(x.score(), y.score())                     # a tie on the score ...
print(x.accuracy(), y.accuracy())               # ... broken by the accuracy
print(x.score("normalized"))
print(round(x.distance(y), 4), round(x.distance(y, "euclidean"), 4))
print(IFN.SCORES, IFN.ACCURACIES, IFN.DISTANCES)
```

## Averaging opinions

The weighted average `wa` of `IFN`s is the classic IFWA operator
$(1 - \prod (1 - \mu_i)^{w_i},\; \prod \nu_i^{w_i})$, and the weighted geometric
mean `wg` is IFWG. See [Aggregation](aggregation.md).

<figure markdown="span">
  ![Three weighted IFN ratings with their weighted average WA and weighted geometric mean WG](../assets/figures/orthopair-aggregation-light.svg#only-light)
  ![Three weighted IFN ratings with their weighted average WA and weighted geometric mean WG](../assets/figures/orthopair-aggregation-dark.svg#only-dark)
</figure>

## Rounding and the boundary

Floating-point rounding can put a pair a hair outside the constraint. Inputs up to
$10^{-12}$ beyond it are accepted and projected back onto $\mu^q + \nu^q = 1$;
anything beyond raises. The exact rules, with a worked example, are in
[Conventions](../concepts/conventions.md#tolerance-and-boundary-projection).

## Pitfalls

**The constraint depends on the rung.** `IFN(0.8, 0.5)` fails while `PFN(0.8, 0.5)`
works. Pick the lowest rung that holds your data, or construct the rung you need
with `QROFN`.

**Rungs do not mix.** `IFN + PFN` raises `TypeError` even though both are
`QROFN`; so does a different `q` in `QROFN`. Convert deliberately by choosing the
degrees again.

**`q` must be an integer $\ge 1$**, and fixed-rung classes reject a conflicting `q`.

**Scale and power need $k \ge 0$.** There is no negative scale and no reciprocal,
and results that leave the constraint beyond rounding raise `ValueError`.

**Degrees must be plain numbers in $[0, 1]$.**

```python exec="on" source="above" result="text"
from bulanik import FFN, IFN, PFN, QROFN, TFN

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: IFN(0.8, 0.5))
show(lambda: PFN(0.8, 0.5))                       # fine
show(lambda: FFN(0.9, 0.8))                       # needs mu^3 + nu^3 <= 1
show(lambda: IFN(0.5, 0.4) + PFN(0.5, 0.4))
show(lambda: QROFN(0.5, 0.4, 2) + QROFN(0.5, 0.4, 3))
show(lambda: IFN(0.5, 0.4) + TFN(1, 2, 3))
show(lambda: QROFN(0.5, 0.4, 0))
show(lambda: QROFN(0.5, 0.4, 2.5))
show(lambda: IFN(0.5, 0.4, q=2))
show(lambda: IFN(1.2, 0.0))
show(lambda: IFN(True, 0.0))
show(lambda: IFN(0.5, 0.4).scale(-1))
show(lambda: IFN(0.5, 0.4).score("median"))
show(lambda: IFN(0.5, 0.4).add(IFN(0.1, 0.1), ops="hamacher"))
```

## See also

- [Conventions](../concepts/conventions.md) for validation and the named-operator rule.
- [Choosing a method](../concepts/choosing.md) for IFN versus PFN versus FFN.
- [Three-part numbers](numbers-threepart.md) when you also need a neutral degree.
- [Aggregation](aggregation.md), and [Sets and scales](sets-and-scales.md) for
  linguistic scales of `IFN` terms.
- API: [`QROFN`][bulanik.QROFN], [`IFN`][bulanik.IFN], [`PFN`][bulanik.PFN],
  [`FFN`][bulanik.FFN].
- Example: [05 intuitionistic pipeline](../examples/05-intuitionistic-pipeline.md).
