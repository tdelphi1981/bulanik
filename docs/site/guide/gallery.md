# Fuzzy numbers, visually

Each figure on this page shows one idea about fuzzy numbers. They are drawn
with [`bulanik.plot`](../api/plot.md), so you can make the same pictures from
your own numbers. bulanik computes every value in them: α-cuts, sums,
products, scores and aggregates. The figures follow your colour scheme; switch
between light and dark mode to see the other variant.

??? note "How these figures are made"
    The scripts are in `docs/scripts/`. Each one calls `bulanik.plot` and then
    adds titles and annotations. Regenerate all the figures with
    `uv run --group docs python docs/scripts/build_figures.py`. Each figure is
    written as a light and a dark SVG, and the output is deterministic.

## Triangular and trapezoidal numbers

<figure markdown="span">
  ![Membership functions of TFN(2, 4, 7) and TrFN(1, 3, 5, 8), with support, core and the α-cut at 0.5 marked](../assets/figures/tfn-anatomy-light.svg#only-light)
  ![Membership functions of TFN(2, 4, 7) and TrFN(1, 3, 5, 8), with support, core and the α-cut at 0.5 marked](../assets/figures/tfn-anatomy-dark.svg#only-dark)
  <figcaption markdown="span">**Anatomy.** A triangle *(l, m, u)* and a trapezoid *(a, b, c, d)* are
  fixed by their corner points. The **support** is where the membership is above
  0, and the **core** is where it equals 1: one point for a triangle, a plateau
  for a trapezoid. Every α-cut between them is an interval; here the cut at
  α = 0.5 is [`alpha_cut(0.5)`][bulanik.numbers.TFN.alpha_cut]. See
  [Triangular numbers](numbers-triangular.md).</figcaption>
</figure>

<figure markdown="span">
  ![α-cuts of A = TFN(1, 2, 4), B = TFN(2, 3, 5), A + B and A × B, stacked on one x axis](../assets/figures/alpha-cut-arithmetic-light.svg#only-light)
  ![α-cuts of A = TFN(1, 2, 4), B = TFN(2, 3, 5), A + B and A × B, stacked on one x axis](../assets/figures/alpha-cut-arithmetic-dark.svg#only-dark)
  <figcaption markdown="span">**Arithmetic, one α at a time.** Fuzzy arithmetic works level by level on α-cuts, as interval
  arithmetic. At α = 0.5 the sum's cut is the sum of the two operand cuts,
  [1.5, 3] + [2.5, 4] = [4, 7]. Addition keeps a triangle exact. The exact
  product has curved sides (the outline marked "exact cuts"), so `A * B` returns the triangle
  through its corner products: at α = 0.5 that gives [4, 13] where the exact
  cut is [3.75, 12]. See [`TFN.mul`][bulanik.numbers.TFN.mul].</figcaption>
</figure>

<figure markdown="span">
  ![TFN(1, 2, 7) with stems at its graded mean 2.667 and its centroid and mean 3.333](../assets/figures/defuzzification-light.svg#only-light)
  ![TFN(1, 2, 7) with stems at its graded mean 2.667 and its centroid and mean 3.333](../assets/figures/defuzzification-dark.svg#only-dark)
  <figcaption markdown="span">**Defuzzification.** [`score`][bulanik.numbers.TFN.score] turns a number into
  one crisp value, and the method you choose matters for a skewed number. The
  graded mean weights the peak four times, so it stays near *m*. For a triangle
  the centroid equals the mean (*l + m + u*)/3, which the long right tail pulls
  further out.</figcaption>
</figure>

<figure markdown="span">
  ![The seven TFN terms of EXAMPLE_TFN_7 on 0 to 10, with a rating of 6.5 belonging to MG at 0.75 and to F at 0.25](../assets/figures/linguistic-scale-tfn-light.svg#only-light)
  ![The seven TFN terms of EXAMPLE_TFN_7 on 0 to 10, with a rating of 6.5 belonging to MG at 0.75 and to F at 0.25](../assets/figures/linguistic-scale-tfn-dark.svg#only-dark)
  <figcaption markdown="span">**A linguistic scale.** [`EXAMPLE_TFN_7`][bulanik.scales.EXAMPLE_TFN_7] maps seven
  words to triangles. Neighbouring terms overlap, so a crisp rating belongs
  partly to two words: 6.5 is "MG" to degree 0.75 and "F" to degree 0.25. Drawn
  with [`plot.variable`][bulanik.plot.variable]; see
  [Sets and scales](sets-and-scales.md).</figcaption>
</figure>

## Orthopair numbers

<figure markdown="span">
  ![Nested feasible regions for q = 1, 2, 3 and 5 in the μ–ν plane, with one IFN, PFN, FFN and q = 5 number and their hesitancy segments](../assets/figures/orthopair-regions-light.svg#only-light)
  ![Nested feasible regions for q = 1, 2, 3 and 5 in the μ–ν plane, with one IFN, PFN, FFN and q = 5 number and their hesitancy segments](../assets/figures/orthopair-regions-dark.svg#only-dark)
  <figcaption markdown="span">**One family, nested regions.** An orthopair number (μ, ν) of rung *q* must
  satisfy μ<sup>q</sup> + ν<sup>q</sup> ≤ 1. Raising *q* enlarges the region:
  intuitionistic (q = 1) ⊂ Pythagorean (2) ⊂ Fermatean (3) ⊂ q = 5. Each sample
  number sits outside the region of the rung below its own. The short segment
  from a point to its own boundary grows with the hesitancy
  [`π`][bulanik.numbers.QROFN.hesitancy]. See
  [Orthopair numbers](numbers-orthopair.md).</figcaption>
</figure>

<figure markdown="span">
  ![Arrows from IFN x to the algebraic and Einstein sums and products of x and y](../assets/figures/orthopair-operators-light.svg#only-light)
  ![Arrows from IFN x to the algebraic and Einstein sums and products of x and y](../assets/figures/orthopair-operators-dark.svg#only-dark)
  <figcaption markdown="span">**Operator families.** Adding *y* to *x* moves toward full membership
  (bottom right); multiplying moves toward full non-membership (top left). The
  Einstein operators (`ops="einstein"`) move further than the algebraic
  default in both directions: the Einstein t-conorm is larger than the
  algebraic one, and its t-norm is smaller. See
  [`QROFN.add`][bulanik.numbers.QROFN.add] and
  [`QROFN.mul`][bulanik.numbers.QROFN.mul].</figcaption>
</figure>

<figure markdown="span">
  ![Three weighted IFN ratings with their weighted average WA and weighted geometric mean WG](../assets/figures/orthopair-aggregation-light.svg#only-light)
  ![Three weighted IFN ratings with their weighted average WA and weighted geometric mean WG](../assets/figures/orthopair-aggregation-dark.svg#only-dark)
  <figcaption markdown="span">**WA vs WG.** Both [`wa`][bulanik.aggregate.wa] and
  [`wg`][bulanik.aggregate.wg] combine the three ratings with the same weights. They
  disagree in a systematic way. The weighted average (IFWA) leans to the high
  membership of the heaviest rating. The geometric mean (IFWG) is pulled toward
  the weak ratings, so it lands at lower μ and higher ν. See
  [Aggregation](aggregation.md).</figcaption>
</figure>

<figure markdown="span">
  ![The seven IFN terms of EXAMPLE_IFN_7 in the μ–ν plane, joined in scale order](../assets/figures/linguistic-scale-ifn-light.svg#only-light)
  ![The seven IFN terms of EXAMPLE_IFN_7 in the μ–ν plane, joined in scale order](../assets/figures/linguistic-scale-ifn-dark.svg#only-dark)
  <figcaption markdown="span">**An intuitionistic scale.** [`EXAMPLE_IFN_7`][bulanik.scales.EXAMPLE_IFN_7] runs
  from "EL" (extremely low) to "EH" (extremely high), and the hairline joins the
  terms in scale order. The distance of a term from the boundary μ + ν = 1 is
  its hesitancy. "L" and "M" have the most (π = 0.1), and "EH" lies on the
  boundary (π = 0).</figcaption>
</figure>

## Three-part numbers

<figure markdown="span">
  ![Four picture fuzzy numbers on the μ–η–ν ternary triangle, with halos sized by their refusal degree](../assets/figures/picture-simplex-light.svg#only-light)
  ![Four picture fuzzy numbers on the μ–η–ν ternary triangle, with halos sized by their refusal degree](../assets/figures/picture-simplex-dark.svg#only-dark)
  <figcaption markdown="span">**Picture fuzzy numbers.** A [`PictureFN`][bulanik.numbers.PictureFN] splits a
  judgement into positive μ, neutral η and negative ν. The rest,
  ρ = 1 − μ − η − ν, is refusal. Each point sits at its parts rescaled to sum to
  1, (μ, η, ν)/(1 − ρ), so the grid reads the rescaled shares and the labels give
  the parts themselves: D (0.3, 0.25, 0.15) sits on the η ≈ 0.36 line. The halo
  grows with ρ, and D leaves the most unsaid (ρ = 0.3). See
  [Three-part numbers](numbers-threepart.md).</figcaption>
</figure>

<figure markdown="span">
  ![Three spherical fuzzy numbers A, B, C inside the positive octant of the unit sphere](../assets/figures/spherical-octant-light.svg#only-light)
  ![Three spherical fuzzy numbers A, B, C inside the positive octant of the unit sphere](../assets/figures/spherical-octant-dark.svg#only-dark)
  <figcaption markdown="span">**Spherical fuzzy numbers.** A [`SphericalFN`][bulanik.numbers.SphericalFN]
  (μ, ν, π) only needs μ² + ν² + π² ≤ 1, so it can be any point of the unit
  ball's positive octant. A = (0.8, 0.3, 0.4), B = (0.4, 0.7, 0.5) and
  C = (0.3, 0.3, 0.85) are drawn with drop lines to the μ–ν floor so you can
  read their depth. C sits high on the π axis: mostly hesitant.</figcaption>
</figure>

<figure markdown="span">
  ![Two neutrosophic numbers as triangles on truth, indeterminacy and falsity spokes](../assets/figures/neutrosophic-radar-light.svg#only-light)
  ![Two neutrosophic numbers as triangles on truth, indeterminacy and falsity spokes](../assets/figures/neutrosophic-radar-dark.svg#only-dark)
  <figcaption markdown="span">**Neutrosophic numbers.** The truth, indeterminacy and falsity parts of a
  [`NeutrosophicFN`][bulanik.numbers.NeutrosophicFN] are independent; each can be
  anywhere in [0, 1]. A radar triangle shows all three at once. A is a
  confident "true", and B is mostly indeterminate.</figcaption>
</figure>

## Interval-valued orthopair numbers

<figure markdown="span">
  ![An interval-valued intuitionistic, Pythagorean and Fermatean number as rectangles over the q = 1, 2 and 3 feasible regions](../assets/figures/iv-orthopair-regions-light.svg#only-light)
  ![An interval-valued intuitionistic, Pythagorean and Fermatean number as rectangles over the q = 1, 2 and 3 feasible regions](../assets/figures/iv-orthopair-regions-dark.svg#only-dark)
  <figcaption markdown="span">**Rectangles instead of points.** An interval-valued orthopair number
  gives membership and non-membership as intervals, so it is the rectangle
  [μ<sub>L</sub>, μ<sub>U</sub>] × [ν<sub>L</sub>, ν<sub>U</sub>] of all the pairs it admits.
  The IVIFN, IVPFN and IVFFN shown here ([`IVIFN`][bulanik.numbers.IVIFN],
  [`IVPFN`][bulanik.numbers.IVPFN], [`IVFFN`][bulanik.numbers.IVFFN]) each have their
  upper-right corner on the boundary of their own rung: μ<sub>U</sub> = 0.3 on q = 1,
  0.5 on q = 2 and 0.88 on q = 3. Only that corner has to satisfy
  μ<sub>U</sub><sup>q</sup> + ν<sub>U</sub><sup>q</sup> ≤ 1, and then the whole rectangle
  lies inside the region.</figcaption>
</figure>

<figure markdown="span">
  ![Two interval-valued intuitionistic numbers A and B with their sum and product as rectangles](../assets/figures/iv-orthopair-operations-light.svg#only-light)
  ![Two interval-valued intuitionistic numbers A and B with their sum and product as rectangles](../assets/figures/iv-orthopair-operations-dark.svg#only-dark)
  <figcaption markdown="span">**Arithmetic on rectangles.** With the algebraic operators, A ⊕ B moves
  toward high membership and low non-membership, and A ⊗ B toward low membership and
  high non-membership. Both results are again rectangles inside the region. The
  corner formulas act on the interval ends, so no interval has to be sampled. See
  [`IVQROFN.add`][bulanik.numbers.IVQROFN.add] and
  [`IVQROFN.mul`][bulanik.numbers.IVQROFN.mul].</figcaption>
</figure>

## Interval type-2 numbers

<figure markdown="span">
  ![A seven-term interval type-2 trapezoidal scale from VL to VH with the DTraT score of each term](../assets/figures/it2-trapezoid-light.svg#only-light)
  ![A seven-term interval type-2 trapezoidal scale from VL to VH with the DTraT score of each term](../assets/figures/it2-trapezoid-dark.svg#only-dark)
  <figcaption markdown="span">**An interval type-2 scale.** Each term is an
  [`IT2TrFN`][bulanik.numbers.IT2TrFN]: an upper trapezoid (bold line), a lower trapezoid
  that is nested inside it (thin line) and, between them, the footprint of uncertainty
  (the wash). The numbers under the axis are the DTraT scores that bulanik computes for
  each term, which turn the type-2 term into one crisp value. The seven terms
  form an **illustrative** scale in the style of the interval type-2 TOPSIS
  literature (Chen and Lee 2010,
  [doi:10.1016/j.eswa.2009.09.012](https://doi.org/10.1016/j.eswa.2009.09.012)). The values are not a transcription of any one
  published table.</figcaption>
</figure>

## DEMATEL

<figure markdown="span">
  ![DEMATEL cause-effect diagram of five factors, with the strongest influences as arrows](../assets/figures/dematel-cause-effect-light.svg#only-light)
  ![DEMATEL cause-effect diagram of five factors, with the strongest influences as arrows](../assets/figures/dematel-cause-effect-dark.svg#only-dark)
  <figcaption markdown="span">**Cause and effect.** The 5 × 5 direct-relation matrix of the vignette
  of the R package `dematel` goes through [`dematel`][bulanik.mcdm.dematel]. Each factor is
  placed by its prominence D + R (how involved it is) and its relation D − R (above the
  line, a net giver of influence). F1, F4 and F5 are causes, and F2 and F3 are effects.
  Arrows mark the total influences above 0.45; the default threshold, the mean of T
  (0.33), would draw 13 of them. See [`plot.dematel`][bulanik.plot.dematel].</figcaption>
</figure>
