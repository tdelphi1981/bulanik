# Three-part numbers: spherical, picture, neutrosophic

When two degrees are not enough, bulanik offers three types with **three
parts** in $[0, 1]$. Each adds a way to say "neither yes nor no":

| Class | Parts | Constraint | Reads as |
|---|---|---|---|
| [`SphericalFN`][bulanik.SphericalFN] | $(\mu, \nu, \pi)$ | $\mu^2 + \nu^2 + \pi^2 \le 1$ | membership, non-membership, hesitancy |
| [`PictureFN`][bulanik.PictureFN] | $(\mu, \eta, \nu)$ | $\mu + \eta + \nu \le 1$ | positive, neutral, negative; the rest is *refusal* |
| [`NeutrosophicFN`][bulanik.NeutrosophicFN] | $(t, i, f)$ | none | truth, indeterminacy, falsity, each independent |

Use a picture number for a vote with four outcomes (yes, abstain, no, refuse to
vote), a spherical number when hesitancy is a separate quantity, and a
neutrosophic number when the three judgements are made independently and may
sum to more than 1. All three have one operator family, `"algebraic"`;
`scale(k)` and `power(k)` need $k \ge 0$; the distances are `"hamming"` (default)
and `"euclidean"`.

**Distances.** Take the differences $d_i$ of the two numbers' *simplex parts*;
then Hamming is $\sum_i \lvert d_i \rvert / D$ and Euclidean is
$\sqrt{\sum_i d_i^2 / D}$:

| Class | Simplex parts | $D$ |
|---|---|---|
| `SphericalFN` | $\mu^2,\; \nu^2,\; \pi^2$ and the refusal $1 - \mu^2 - \nu^2 - \pi^2$ | 2 |
| `PictureFN` | $\mu,\; \eta,\; \nu$ and the refusal $1 - \mu - \eta - \nu$ | 2 |
| `NeutrosophicFN` | $t,\; i,\; f$ | 3 |

## Spherical numbers

<figure markdown="span">
  ![Three spherical fuzzy numbers A, B, C inside the positive octant of the unit sphere](../assets/figures/spherical-octant-light.svg#only-light)
  ![Three spherical fuzzy numbers A, B, C inside the positive octant of the unit sphere](../assets/figures/spherical-octant-dark.svg#only-dark)
  <figcaption markdown="span">A valid spherical number is a point inside the positive octant
  of the unit ball.</figcaption>
</figure>

The operations are those of Kutlu Gundogdu and Kahraman (2019), evaluated on the
squares $(a, b, c) = (\mu^2, \nu^2, \pi^2)$ and the exact refusal
$r = 1 - a - b - c$; a square root takes each result back:

| Operation | $\mu^2$ | $\nu^2$ | $\pi^2$ |
|---|---|---|---|
| `add` | $1 - (1 - a_1)(1 - a_2)$ | $b_1 b_2$ | $(1 - a_1)(1 - a_2) - (b_1 + r_1)(b_2 + r_2)$ |
| `mul` | $a_1 a_2$ | $1 - (1 - b_1)(1 - b_2)$ | $(1 - b_1)(1 - b_2) - (a_1 + r_1)(a_2 + r_2)$ |
| `scale` by $k$ | $1 - (1 - a)^k$ | $b^k$ | $(1 - a)^k - (1 - a - c)^k$ |
| `power` by $k$ | $a^k$ | $1 - (1 - b)^k$ | $(1 - b)^k - (1 - b - c)^k$ |

Scores: `"pi_adjusted"` (default) $(\mu - \pi)^2 - (\nu - \pi)^2$ and
`"normalized"` $(1 + s)/2$. Accuracy `"sum"`: $\mu^2 + \nu^2 + \pi^2$.

```python exec="on" source="above" result="text"
from bulanik import SphericalFN

a, b = SphericalFN(0.5, 0.4, 0.3), SphericalFN(0.4, 0.3, 0.2)
print(a, a.parts)
print([round(p, 4) for p in (a + b).parts])
print([round(p, 4) for p in (a * b).parts])
print([round(p, 4) for p in a.scale(2).parts])
print(round(a.score(), 4), round(a.score("normalized"), 4), round(a.accuracy(), 4))
print(a.complement())
print(SphericalFN.SCORES, round(a.distance(b), 4), round(a.distance(b, "euclidean"), 4))
```

## Picture numbers

<figure markdown="span">
  ![Four picture fuzzy numbers on the μ–η–ν ternary triangle, with halos sized by their refusal degree](../assets/figures/picture-simplex-light.svg#only-light)
  ![Four picture fuzzy numbers on the μ–η–ν ternary triangle, with halos sized by their refusal degree](../assets/figures/picture-simplex-dark.svg#only-dark)
  <figcaption markdown="span">Picture numbers live on a triangle; the unused share
  $1 - \mu - \eta - \nu$ is the refusal degree.</figcaption>
</figure>

The operations are those of Wei (2017), as the code evaluates them (that paper was
retracted by the publisher in 2025; the closed forms are tested against the code, not the
paper, see the [references](../about/references.md)):

| Operation | $\mu$ | $\eta$ | $\nu$ |
|---|---|---|---|
| `add` | $\mu_1 + \mu_2 - \mu_1 \mu_2$ | $\eta_1 \eta_2$ | $\nu_1 \nu_2$ |
| `mul` | $(\mu_1 + \eta_1)(\mu_2 + \eta_2) - \eta_1 \eta_2$ | $\eta_1 \eta_2$ | $\nu_1 + \nu_2 - \nu_1 \nu_2$ |
| `scale` by $k$ | $1 - (1 - \mu)^k$ | $\eta^k$ | $\nu^k$ |
| `power` by $k$ | $(\mu + \eta)^k - \eta^k$ | $\eta^k$ | $1 - (1 - \nu)^k$ |

Scores: `"difference"` (default) $\mu - \nu$ and `"normalized"`
$(1 + \mu - \nu)/2$. Accuracy `"sum"`: $\mu + \eta + \nu$.

Scaling by $k < 1$ can leave the constraint set, and $k = 0$ always does (the
additive identity $(0, 1, 1)$ is not a valid picture number). That matters
because averaging is built from scaling by weights below 1. So
[`wa`][bulanik.aggregate.wa] and [`wg`][bulanik.aggregate.wg] use closed forms for
picture numbers and work regardless.

```python exec="on" source="above" result="text"
from bulanik import PictureFN
from bulanik.aggregate import wa

a, b = PictureFN(0.5, 0.2, 0.1), PictureFN(0.4, 0.3, 0.2)
print(a, round(a.accuracy(), 4), "refusal", round(1 - sum(a.parts), 4))
print([round(p, 4) for p in (a + b).parts])
print([round(p, 4) for p in (a * b).parts])
print([round(p, 4) for p in (a ** 2).parts])
print(round(a.score(), 4), round(a.score("normalized"), 4))
print(a.complement())
mean = wa([a, b], [0.5, 0.5])          # closed form: no scale(0.5) involved
print([round(p, 4) for p in mean.parts])
```

## Neutrosophic numbers

<figure markdown="span">
  ![Two neutrosophic numbers as triangles on truth, indeterminacy and falsity spokes](../assets/figures/neutrosophic-radar-light.svg#only-light)
  ![Two neutrosophic numbers as triangles on truth, indeterminacy and falsity spokes](../assets/figures/neutrosophic-radar-dark.svg#only-dark)
</figure>

A single-valued neutrosophic number (Wang, Smarandache, Zhang and Sunderraman
2010) has no joint constraint, so truth, indeterminacy and falsity can each be
anything in $[0, 1]$. Operations:

| Operation | $t$ | $i$ | $f$ |
|---|---|---|---|
| `add` | $t_1 + t_2 - t_1 t_2$ | $i_1 i_2$ | $f_1 f_2$ |
| `mul` | $t_1 t_2$ | $i_1 + i_2 - i_1 i_2$ | $f_1 + f_2 - f_1 f_2$ |
| `scale` by $k$ | $1 - (1 - t)^k$ | $i^k$ | $f^k$ |
| `power` by $k$ | $t^k$ | $1 - (1 - i)^k$ | $1 - (1 - f)^k$ |

The only score is `"average"`, $(2 + t - i - f)/3 \in [0, 1]$. Accuracy is
`"difference"` (default) $t - f$ or `"certainty"` $t$. The complement is
$(f, 1 - i, t)$ and distances divide by 3 instead of 2.

```python exec="on" source="above" result="text"
from bulanik import NeutrosophicFN

a, b = NeutrosophicFN(0.7, 0.2, 0.1), NeutrosophicFN(0.5, 0.3, 0.2)
print([round(p, 4) for p in (a + b).parts])
print([round(p, 4) for p in (a * b).parts])
print([round(p, 4) for p in a.scale(2).parts], [round(p, 4) for p in a.scale(0.5).parts])
print(round(a.score(), 4), a.accuracy(), a.accuracy("certainty"))
print(a.complement())
print(round(a.distance(b), 4), NeutrosophicFN(0.9, 0.9, 0.9))   # no joint constraint
```

## Pitfalls

**Each type has its own constraint, and neutrosophic has none.** A spherical number
that is fine as a picture number may be rejected, and a neutrosophic number that
is "too big" is accepted.

**The three types do not mix**, and none mixes with orthopair numbers.

**Picture scale below 1.** Never scale a picture number by a fraction by hand;
`wa` does it for you. `scale(0.5)` fails when the result leaves the set.

**Tiny spherical parts.** Spherical operations work on squares, so a part below
about $1.5 \times 10^{-154}$ behaves as 0.

```python exec="on" source="above" result="text"
from bulanik import IFN, NeutrosophicFN, PictureFN, SphericalFN

def show(make):
    try:
        print(make())
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: SphericalFN(0.8, 0.5, 0.5))
show(lambda: PictureFN(0.6, 0.3, 0.3))
show(lambda: PictureFN(0.5, 0.2, 0.1).scale(0.5))
show(lambda: PictureFN(0.5, 0.2, 0.1).scale(0))
show(lambda: PictureFN(0.5, 0.2, 0.1).scale(-1))
show(lambda: PictureFN(0.5, 0.2, 0.1) + NeutrosophicFN(0.5, 0.2, 0.1))
show(lambda: SphericalFN(0.5, 0.4, 0.3) + IFN(0.5, 0.4))
show(lambda: NeutrosophicFN(0.7, 0.2, 0.1).score("normalized"))
show(lambda: SphericalFN(0.5, 0.4, 0.3).accuracy("difference"))
show(lambda: SphericalFN(0.5, 0.4, 0.3).add(SphericalFN(0.1, 0.1, 0.1), ops="einstein"))
```

## See also

- [Orthopair numbers](numbers-orthopair.md) for the two-degree types.
- [Choosing a method](../concepts/choosing.md) and [Conventions](../concepts/conventions.md).
- [Aggregation](aggregation.md) for closed-form picture averages.
- API: [`SphericalFN`][bulanik.SphericalFN], [`PictureFN`][bulanik.PictureFN],
  [`NeutrosophicFN`][bulanik.NeutrosophicFN].
