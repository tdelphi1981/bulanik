# API reference

One page per module. Every public name is listed here with its one-line
summary and a link to its full entry (signature, arguments, exceptions and
runnable examples). The table is generated from the package at build time, so
it cannot fall out of date.

The fuzzy number classes are also importable from the package root
(`from bulanik import TFN`); everything else is imported from its module
(`from bulanik.mcdm import topsis`).

| Module | Contents |
|---|---|
| [`bulanik.numbers`](numbers.md) | Fuzzy number types: triangular, orthopair, three-part, interval-valued and interval type-2 |
| [`bulanik.aggregate`](aggregate.md) | Weighted average, weighted geometric mean, expert aggregation |
| [`bulanik.sets`](sets.md) | Membership functions, linguistic variables, t-norms |
| [`bulanik.scales`](scales.md) | Linguistic scales and the built-in ones |
| [`bulanik.inference`](inference.md) | Rules, Mamdani, Sugeno and Tsukamoto systems |
| [`bulanik.type2`](type2.md) | Interval type-2 sets, systems and type reduction |
| [`bulanik.anfis`](anfis.md) | Adaptive neuro-fuzzy inference |
| [`bulanik.mcdm`](mcdm.md) | Weighting, ranking, outranking, DEMATEL and fuzzy ranking |
| [`bulanik.lumix`](lumix.md) | LumiX bridge (`lumix` extra) |
| [`bulanik.sezgi`](sezgi.md) | sezgi bridge (`sezgi` extra) |
| [`bulanik.plot`](plot.md) | Figures (`plot` extra) |

```python exec="on" html="1"
import subprocess
import sys

print(subprocess.run([sys.executable, "docs/scripts/api_index.py"], check=True, capture_output=True, text=True).stdout)
```

