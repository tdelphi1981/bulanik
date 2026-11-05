# bulanik

Easy-to-use fuzzy numbers, fuzzy inference and fuzzy multi-criteria decision making
(MCDM) in pure Python, with no runtime dependencies. *Bulanık* is Turkish for *fuzzy*.
Optional bridges connect it to [LumiX](https://pypi.org/project/lumix-opt/) (optimization
modelling) and [sezgi](https://github.com/tdelphi1981/sezgi) (metaheuristics), and an
optional plotting module draws the results in a light and a dark theme.

**Documentation:** <https://tdelphi1981.github.io/bulanik/>

## Install

```bash
pip install bulanik                      # core, no runtime dependencies
pip install "bulanik[lumix]"             # + the LumiX bridge (bulanik.lumix)
pip install "bulanik[sezgi]"             # + the sezgi bridge (bulanik.sezgi)
pip install "bulanik[plot]"              # + figures (bulanik.plot, needs matplotlib)
pip install "bulanik[lumix,sezgi,plot]"  # everything
```

Python 3.10 to 3.13. For development, clone the repository and run `uv sync` (see
[Contributing](https://github.com/tdelphi1981/bulanik/blob/main/CONTRIBUTING.md)).

## Quickstart

```python
from bulanik import IFN, TFN
from bulanik.aggregate import wa

a, b = IFN(0.6, 0.3), IFN(0.5, 0.4)
print(a + b)                          # IFN(mu=0.8, nu=0.12)
print(a.add(b, ops="einstein"))       # the Einstein operator family
print(wa([a, b], [0.7, 0.3]))         # weighted average
print((TFN(1, 2, 3) + TFN(3, 4, 5)).score())   # crisp score for ranking
```

Every operator family, score and method is chosen by name (`ops="einstein"`,
`score("normalized")`); an unknown name raises a `ValueError` that lists the valid ones.

## What is in it

- **Fuzzy numbers:** triangular and trapezoidal; intuitionistic, Pythagorean, Fermatean
  and q-rung orthopair; spherical, picture and neutrosophic; their interval-valued
  versions (`IVIFN`, `IVPFN`, `IVFFN`, `IVQROFN`, `IVNeutrosophicFN`) and interval type-2
  trapezoidal numbers (`IT2TrFN`). Immutable and validated.
- **Aggregation, sets and scales:** weighted average and geometric mean, expert
  aggregation, membership functions, linguistic variables and linguistic scales.
- **Fuzzy inference:** Mamdani, Sugeno and Tsukamoto systems from plain-text rules;
  interval type-2 systems with KM, EKM, Nie-Tan and BMM type reduction; ANFIS.
- **Decision making:** AHP, fuzzy AHP (also over any number kind), SWARA, entropy, CRITIC
  and PIPRECIA weights; DEMATEL with CFCS; best-worst consistency; TOPSIS, VIKOR, WASPAS,
  EDAS, CODAS, COPRAS, MARCOS, CoCoSo, WISP and ELECTRE, on crisp matrices or fuzzy
  ratings (fuzzy CODAS, WASPAS and CoCoSo included), and Chen's fuzzy TOPSIS. Interval-valued
  AHP (`ahp_interval_valued`) with the multiplicative and possibility methods.
- **Bridges:** with LumiX, the linear best-worst method, Zimmermann fuzzy programming and
  MCDM scores as optimization data; with sezgi, nonlinear and fuzzy BWM and ANFIS training.
- **Figures:** `bulanik.plot` for numbers, scales, inference, rankings and DEMATEL.

The [documentation](https://tdelphi1981.github.io/bulanik/) has a user guide, an API
reference, and 24 runnable examples (also in [`examples/`](https://github.com/tdelphi1981/bulanik/tree/main/examples)).

## Development

```bash
uv sync
uv run python -m pytest -q
```

The dev group installs LumiX, OR-Tools, sezgi and matplotlib, so the bridge tests and
every example run; without an extra, its tests and examples are skipped. Coverage is
measured on the pure-Python core.

## Citing

Please cite bulanik using [`CITATION.cff`](https://github.com/tdelphi1981/bulanik/blob/main/CITATION.cff) (GitHub's "Cite this repository"),
and the original papers behind the methods you use, listed with DOIs in the
[references](https://tdelphi1981.github.io/bulanik/about/references.html).

## License

MIT. See [LICENSE](https://github.com/tdelphi1981/bulanik/blob/main/LICENSE).
