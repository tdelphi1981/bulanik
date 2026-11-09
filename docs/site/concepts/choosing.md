# Choosing a method

Decision tables: start from the question, read off the type or method, then follow
the link to the guide page. "Why" is the one-line reason; the guide pages give the
formulas.

## Which number type?

| Your information looks like | Use | Why |
|---|---|---|
| "about 5, between 3 and 8" | [`TFN`][bulanik.numbers.TFN] | One most-plausible value and a left and right spread: the simplest model of a vague quantity. Supports full arithmetic (`+ - * /`, powers, reciprocals). |
| "roughly between 4 and 6, never below 2 or above 9" | [`TrFN`][bulanik.numbers.TrFN] | A plateau of full membership instead of a single peak. |
| An expert gives a degree of support and a degree of opposition, and the two sum to at most 1 | [`IFN`][bulanik.numbers.IFN] | Membership $\mu$ and non-membership $\nu$ with $\mu + \nu \le 1$; the rest is hesitancy. |
| Support and opposition may sum to more than 1, but their squares sum to at most 1 | [`PFN`][bulanik.numbers.PFN] | $\mu^2 + \nu^2 \le 1$ admits more pairs than the intuitionistic constraint, for example $(0.8, 0.5)$. |
| Even less restricted | [`FFN`][bulanik.numbers.FFN], then [`QROFN`][bulanik.numbers.QROFN] | $\mu^3 + \nu^3 \le 1$, or any rung $q$: the larger the rung, the larger the set of admissible pairs. |
| Three separate degrees: positive, neutral, negative (with refusal as the remainder) | [`PictureFN`][bulanik.numbers.PictureFN] | Models a voting-like opinion with an abstention. |
| Support and opposition, but each only known as an interval | [`IVIFN`][bulanik.numbers.IVIFN], [`IVPFN`][bulanik.numbers.IVPFN], [`IVFFN`][bulanik.numbers.IVFFN], [`IVQROFN`][bulanik.numbers.IVQROFN] | The orthopair constraint on the upper pair with the rung of your choice, and a membership and a non-membership interval. See [Interval-valued numbers](../guide/numbers-interval.md). |
| Truth, indeterminacy and falsity, each only known as an interval | [`IVNeutrosophicFN`][bulanik.numbers.IVNeutrosophicFN] | Three intervals in $[0, 1]$ with no joint constraint. Its only score is `"average"`. |
| A linguistic term that is vague about its own shape (an upper and a lower trapezoid) | [`IT2TrFN`][bulanik.numbers.IT2TrFN] | Interval type-2 trapezoid with a footprint of uncertainty; scored by `"dtrat"` by default. See [Interval type-2 trapezoids](../guide/numbers-it2.md). |
| Three degrees whose squares sum to at most 1 | [`SphericalFN`][bulanik.numbers.SphericalFN] | Membership, non-membership and hesitancy on the unit sphere. |
| Truth, indeterminacy and falsity that are independent of each other | [`NeutrosophicFN`][bulanik.numbers.NeutrosophicFN] | Each part is in $[0, 1]$ with no joint constraint. |

For the orthopair family the smallest rung that admits a given pair is easy to
compute. This block runs when the site is built:

```python exec="on" source="above" result="text"
from bulanik import QROFN

for mu, nu in [(0.6, 0.3), (0.8, 0.5), (0.9, 0.5), (0.95, 0.6)]:
    q = next(q for q in range(1, 50) if mu**q + nu**q <= 1)
    print(f"(mu={mu}, nu={nu}) needs q >= {q}: {QROFN(mu, nu, q)}")
```

Rule of thumb: pick the **simplest** type that can hold your data. TFN for
numerical vagueness; an orthopair type when experts express support *and*
opposition; and choose the rung by what the elicitation allows, not by what
makes the numbers fit.

## Which inference system?

| You need | Use | Why |
|---|---|---|
| A rule base whose consequents are fuzzy sets, readable by experts | [`MamdaniFIS`][bulanik.inference.MamdaniFIS] | The classic: clip or scale each consequent, aggregate, defuzzify. Cost grows with `resolution`. |
| A crisp, fast output; control or regression; something you can tune | [`SugenoFIS`][bulanik.inference.SugenoFIS] | Constant or linear consequents; the output is a weighted average, no defuzzification step. |
| Outputs defined by monotonic membership functions | [`TsukamotoFIS`][bulanik.inference.TsukamotoFIS] | Each rule gives a crisp value by inverting its consequent; needs `linear_increasing`, `linear_decreasing` or `sigmoid` terms. |
| Membership functions you are not sure about (uncertain mean or width) | [`IT2MamdaniFIS`][bulanik.type2.IT2MamdaniFIS], [`IT2SugenoFIS`][bulanik.type2.IT2SugenoFIS] | Interval type-2 sets carry a footprint of uncertainty, and type reduction turns it into an output interval. |
| A Sugeno system learned from input-output data | [`ANFIS`][bulanik.anfis.ANFIS] | Grid partition, trained by hybrid (least squares plus gradient) or gradient steps; `to_sugeno` exports a plain `SugenoFIS`. |
| To tune a Sugeno system or ANFIS with a metaheuristic | the [sezgi bridge](../guide/bridge-sezgi.md) | `ANFISProblem` and `train_anfis` pose training as a sezgi problem. |

### Which type-reduction method? (interval type-2)

| Method | Use when | Why |
|---|---|---|
| `km` (default) | You want the exact interval | Karnik-Mendel iteration to the exact switch points. |
| `ekm` | You want the exact interval from the enhanced algorithm | Enhanced KM: returns the same interval as `km`. |
| `nie_tan` | You need speed and a closed form | $\sum y(\underline\mu + \overline\mu)/\sum(\underline\mu + \overline\mu)$, with no iteration. |
| `bmm` | You want a weighted blend of the lower and upper averages | $m\cdot\text{avg}_{lower} + n\cdot\text{avg}_{upper}$ (default $m = n = 0.5$); needs some positive lower weight, else `NoLowerFiringError`. |

### Which defuzzification? (Mamdani)

| Method | Output | Why |
|---|---|---|
| `centroid` (default) | The centre of gravity of the aggregate | Smooth: every active rule moves the output. |
| `bisector` | The point that halves the area under the aggregate | Splits the aggregate into two equal halves instead of balancing it. |
| `mom`, `som`, `lom` | Mean, smallest or largest of the maxima | Pick the plateau of the winning rule; the output jumps when the winner changes. |

## Which weighting method?

| You have | Use | Why |
|---|---|---|
| A full pairwise-comparison matrix from one decision maker | [`ahp`][bulanik.mcdm.ahp] | Principal eigenvector plus a consistency ratio to check the judgments. |
| A pairwise matrix with vague judgments (triangular numbers) | [`fuzzy_ahp`][bulanik.mcdm.fuzzy_ahp] | Keeps the vagueness in the weights (`fuzzy_weights`) as well as the crisp scores. |
| A ranking of the criteria and "how much less important is each than the one before" | [`swara`][bulanik.mcdm.swara] | Only $n - 1$ judgments, no matrix; crisp or fuzzy importances. |
| Data but no opinions | [`entropy`][bulanik.mcdm.entropy] | Objective: a criterion whose values differ more between alternatives gets more weight. |
| Data but no opinions, and you want conflicting criteria to count more | [`critic`][bulanik.mcdm.weighting.critic] | Objective: weight grows with the spread of a criterion and with its negative correlation to the others. |
| Relative importances of criteria listed in any order, optionally judged again backwards | [`piprecia`][bulanik.mcdm.weighting.piprecia] | Like SWARA without sorting; crisp or triangular judgements and an optional inverse pass. |
| A pairwise matrix of interval-valued orthopair judgments (IVIFN, IVPFN, IVFFN), as in the IVPF-, IVIF- and Fermatean AHP literature | [`ahp_interval_valued`][bulanik.mcdm.weighting.ahp_interval_valued] | Works on the interval endpoints: `"multiplicative"` for any rung spreads the weights widely, `"possibility"` for IVIFN also gives interval weights. |
| A pairwise matrix of interval-valued, orthopair, neutrosophic or interval type-2 judgments | [`ahp_fuzzy_numbers`][bulanik.mcdm.weighting.ahp_fuzzy_numbers] | AHP over any one number kind: aggregate the rows, score them, normalise. |
| The criteria influence each other and you want causes, effects and weights from that network | [`dematel`][bulanik.mcdm.dematel.dematel] | Total-relation matrix, prominence and relation of each factor; `as_weights()` gives a `WeightingResult`. See [More weighting methods](../guide/weighting-more.md). |
| Few comparisons and a consistency check, and a solver is installed | the best-worst method: [`bwm_linear`][bulanik.lumix.bwm_linear] (`lumix`), [`BWMProblem`][bulanik.sezgi.BWMProblem] and [`FuzzyBWMProblem`][bulanik.sezgi.FuzzyBWMProblem] (`sezgi`) | Two vectors of $n - 1$ comparisons each (best against the others, the others against the worst), more consistent than a full matrix. The linear model is an LP; the nonlinear and the fuzzy model are searched with sezgi. |

The `consistency_ratio` of a BWM result is not the same quantity in every
bridge; see the [warning on `sezgi`](../api/sezgi.md) before comparing them.

## Which ranking method?

| Situation | Use | Why |
|---|---|---|
| Crisp scores; a general-purpose default | [`topsis`][bulanik.mcdm.topsis] | Closeness to the ideal and distance from the anti-ideal; compensatory and intuitive. |
| You want a compromise solution and to see group utility against individual regret | [`vikor`][bulanik.mcdm.vikor] | Ranks by $Q$ built from the weighted sum $S$ and the worst single criterion $R$ (`v` balances them). **Lower is better.** |
| You want the weighted-sum and the weighted-product models combined | [`waspas`][bulanik.mcdm.waspas] | Blends the two with the weight `lam`; entries must be positive. |
| Criteria where distance from the *average* solution is meaningful | [`edas`][bulanik.mcdm.edas] | Positive and negative distances from the average; the average of each criterion must be positive. |
| A distance-based score from the negative-ideal point | [`codas`][bulanik.mcdm.scoring.codas] | Euclidean and taxicab distances; `psi` chooses between the paper's equation and its tables. |
| Shares of benefit and cost sums, as a percentage utility | [`copras`][bulanik.mcdm.scoring.copras] | Proportional assessment; the utility degree is relative to the best. |
| Utility relative to both an ideal and an anti-ideal alternative | [`marcos`][bulanik.mcdm.scoring.marcos] | Entries must be strictly positive. |
| Three compromise scores combined; a parameter `lam` for sum against power | [`cocoso`][bulanik.mcdm.scoring.cocoso] | Fails, with a clear error, if one alternative is worst on every criterion. |
| Both weighted sums and weighted products, with benefit and cost measured against each other | [`wisp`][bulanik.mcdm.scoring.wisp] | Four measures averaged; entries must be strictly positive. |
| Pairwise dominance, not a single score (outranking) | [`electre`][bulanik.mcdm.outranking.electre] | Concordance and discordance between pairs; `extras["outranks"]` is the partial relation, `ranking` comes from the net indices. See [More ranking methods](../guide/ranking-more.md). |
| Triangular fuzzy ratings and weights, with fuzziness kept to the end | [`fuzzy_topsis`][bulanik.mcdm.fuzzy_topsis] | Chen's (2000) TOPSIS on triangular numbers: normalise, weight, compare to fixed fuzzy ideals by vertex distance. |
| Several experts, any number type | [`rank_fuzzy`][bulanik.mcdm.rank_fuzzy] with a [`DecisionMatrix`][bulanik.mcdm.DecisionMatrix] | Aggregates the experts (`wa` or `wg`, any operator family), scores each fuzzy number, then ranks with any of ten methods (`topsis`, `vikor`, `waspas`, `edas`, `codas`, `copras`, `marcos`, `cocoso`, `wisp`, `electre`). |
| CODAS on orthopair, neutrosophic, triangular or trapezoidal ratings, with distances taken between the numbers | [`fuzzy_codas`][bulanik.mcdm.fuzzy_ranking.fuzzy_codas] | Score-last: distances of the weighted numbers from the negative-ideal numbers; a cost criterion uses `complement()`, or for `TFN` / `TrFN` the fuzzy CODAS normalisation of Keshavarz Ghorabaee et al. (2017). |
| WASPAS with the weighted sum and product kept as fuzzy numbers | [`fuzzy_waspas`][bulanik.mcdm.fuzzy_ranking.fuzzy_waspas] | Score-last; `combine="fuzzy"` mixes the numbers and scores once, `combine="scores"` scores first. |
| CoCoSo with the weighted sum and product kept as fuzzy numbers | [`fuzzy_cocoso`][bulanik.mcdm.fuzzy_ranking.fuzzy_cocoso] | Score-last sums and products, then CoCoSo's appraisal scores; `"normalized"` score by default. |

Different methods may disagree; that is information about how robust the ranking is,
not a bug. When it matters, rank with two methods and compare.

```python exec="on" source="above" result="text"
from bulanik.mcdm import edas, topsis, vikor, waspas

data = [[250, 16, 12], [200, 16, 8], [300, 32, 16], [275, 32, 8]]   # price, memory, camera
weights, types = [0.4, 0.35, 0.25], ["cost", "benefit", "benefit"]
names = ["P1", "P2", "P3", "P4"]
for method in (topsis, vikor, waspas, edas):
    result = method(data, weights, types, alternatives=names)
    print(f"{method.__name__:7s}", result.ranking)
```
