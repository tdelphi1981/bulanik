# bulanik

**Easy-to-use fuzzy numbers, fuzzy inference and fuzzy multi-criteria decision
making in pure Python, with no runtime dependencies.**

*Bulanık* is Turkish for *fuzzy*. Optional bridges connect bulanik to
[LumiX](https://pypi.org/project/lumix-opt/) (optimization modelling) and
[sezgi](https://github.com/tdelphi1981/sezgi) (metaheuristics), and the optional
[`bulanik.plot`](api/plot.md) module draws publication-quality figures of all of it.

<div class="grid cards" markdown>

-   **Fuzzy numbers**

    Triangular and trapezoidal numbers, intuitionistic, Pythagorean, Fermatean
    and q-rung orthopair numbers, spherical, picture and neutrosophic numbers.
    Immutable, validated on construction, with named operator families.

    [:octicons-arrow-right-24: Numbers](guide/numbers-triangular.md)

-   **Aggregation and scales**

    Weighted average and geometric mean, expert aggregation, membership
    functions, linguistic variables and ready-made linguistic scales.

    [:octicons-arrow-right-24: Aggregation](guide/aggregation.md)

-   **Fuzzy inference**

    Mamdani, Sugeno and Tsukamoto systems from plain-text rules, interval
    type-2 systems with type reduction, and ANFIS learning from data.

    [:octicons-arrow-right-24: Mamdani inference](guide/mamdani.md)

-   **Decision making**

    AHP, fuzzy AHP, SWARA, entropy and best-worst weights; TOPSIS, VIKOR, WASPAS,
    EDAS and Chen's fuzzy TOPSIS for the ranking.

    [:octicons-arrow-right-24: Weighting](guide/weighting.md)

-   **Bridges**

    Solve the best-worst method and Zimmermann's fuzzy programs with LumiX, tune
    Sugeno systems and train ANFIS with sezgi.

    [:octicons-arrow-right-24: LumiX bridge](guide/bridge-lumix.md)

-   **Figures**

    One validated palette, light and dark themes, direct labels:
    `bulanik.plot.membership`, `orthopair`, `variable`, `mamdani_output`,
    `ranking` and more.

    [:octicons-arrow-right-24: bulanik.plot](api/plot.md)

</div>

## A ten-line taste

Rate three suppliers with linguistic terms and rank them with fuzzy TOPSIS. The block runs when the site is built:

```python exec="on" source="above" result="text"
from bulanik.mcdm import fuzzy_topsis
from bulanik.scales import EXAMPLE_TFN_7 as scale

ratings = scale.convert([["G", "VG", "F"], ["F", "G", "G"], ["MP", "F", "VG"]])
weights = [scale["G"] / 10, scale["MG"] / 10, scale["F"] / 10]
result = fuzzy_topsis(ratings, weights, ["benefit", "benefit", "cost"],
                      alternatives=["A", "B", "C"], criteria=["quality", "service", "price"])
print(result.ranking)
print({k: round(v, 3) for k, v in result.scores.items()})
```

## Where to go next

- New here? [Install](install.md), then the [Quickstart](quickstart.md): five short
  recipes, each one runnable.
- [Conventions](concepts/conventions.md) explains what every function promises:
  immutability, validation, tolerances, and the one naming rule for operator
  families, scores and methods.
- [Choosing a method](concepts/choosing.md) maps problems to number types,
  inference systems and MCDM methods.
- The [user guide](guide/numbers-triangular.md) has one page per topic with the
  formulas the code uses; the [examples](examples/index.md) are 24 complete
  scripts with their real output; the [API reference](api/index.md) lists every
  public name.
