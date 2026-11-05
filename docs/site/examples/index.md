# Examples

Twenty-four runnable scripts live in the repository's `examples/` folder. Each page here shows the script, runs it when the site is built, and points to the guide and API pages behind it. All of them are also run by the test suite.

| # | Example | Needs |
|---|---|---|
| 01 | [Fuzzy numbers](01-fuzzy-numbers.md) | core |
| 02 | [Linguistic scales and expert aggregation](02-scale-and-experts.md) | core |
| 03 | [Buckley fuzzy AHP](03-fuzzy-ahp.md) | core |
| 04 | [SWARA weights and TOPSIS](04-swara-topsis-supplier.md) | core |
| 05 | [Intuitionistic pipeline](05-intuitionistic-pipeline.md) | core |
| 06 | [Zimmermann fuzzy goals with LumiX](06-zimmermann-lumix.md) | lumix |
| 07 | [Best-worst method with LumiX](07-bwm-lumix.md) | lumix |
| 08 | [Fuzzy BWM with sezgi](08-fuzzy-bwm-sezgi.md) | sezgi |
| 09 | [MCDM scores as LumiX data](09-weights-to-lumix.md) | lumix |
| 10 | [Mamdani tipper](10-mamdani-tipper.md) | core |
| 11 | [Sugeno controller](11-sugeno-controller.md) | core |
| 12 | [Tuning a Sugeno system with sezgi](12-tune-sugeno-with-sezgi.md) | sezgi |
| 13 | [Tsukamoto](13-tsukamoto.md) | core |
| 14 | [Interval type-2 versus type-1 under noise](14-it2-vs-t1-noise.md) | core |
| 15 | [ANFIS on the sinc surface](15-anfis-sinc.md) | core |
| 16 | [ANFIS trained with sezgi](16-anfis-with-sezgi.md) | sezgi |
| 17 | [Plotting with bulanik.plot](17-plotting.md) | plot |
| 18 | [Interval-valued AHP and CODAS](18-interval-valued-ahp-codas.md) | core |
| 19 | [Interval type-2 AHP and MARCOS](19-it2-ahp-marcos.md) | core |
| 20 | [Ranking methods compared](20-ranking-methods-compared.md) | core |
| 21 | [Objective and subjective weights](21-critic-piprecia.md) | core |
| 22 | [Fuzzy DEMATEL](22-fuzzy-dematel.md) | core |
| 23 | [Interval-valued AHP compared](23-interval-valued-ahp.md) | core |
| 24 | [Ranking on fuzzy numbers](24-fuzzy-ranking.md) | core |

Examples marked `lumix`, `sezgi` or `plot` need that [extra](../install.md); the documentation build has them installed.

To run one yourself from a checkout:

```bash
uv sync
uv run python examples/10_mamdani_tipper.py
```
