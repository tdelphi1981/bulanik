# Example 24: Ranking on fuzzy numbers

Two experts' intuitionistic ratings of four suppliers go through `rank_fuzzy`, score-first with crisp CODAS, WASPAS and CoCoSo, and score-last with their fuzzy versions.

## Source

```python title="examples/24_fuzzy_ranking.py"
--8<-- "examples/24_fuzzy_ranking.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/24_fuzzy_ranking.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- WASPAS and CoCoSo give the same order score-first and score-last here. CODAS does not: score-first it picks Borealis, score-last Cedar. The crisp method turns a cost into a ratio of scores and measures distances on the scored matrix; `fuzzy_codas` takes the complement of the cost rating and measures distances between the weighted numbers.
- `combine="fuzzy"` and `combine="scores"` differ from the third decimal on, because a fuzzy mix of two close numbers scores close to the mix of their scores. They agree on the order in this example, though they need not in general.
- Scores from different methods live on different scales, so compare the ranks across methods and the scores only within one method.
- `extras["Q1"]` and `extras["Q2"]` keep the weighted sum and the weighted product as `IFN` numbers.

## Go further

- Guide: [More ranking methods](../guide/ranking-more.md#ranking-on-fuzzy-numbers)
- API: [`fuzzy_codas`][bulanik.mcdm.fuzzy_ranking.fuzzy_codas], [`fuzzy_waspas`][bulanik.mcdm.fuzzy_ranking.fuzzy_waspas], [`fuzzy_cocoso`][bulanik.mcdm.fuzzy_ranking.fuzzy_cocoso], [`rank_fuzzy`][bulanik.mcdm.fuzzy.rank_fuzzy]

Run it yourself: `python examples/24_fuzzy_ranking.py`.
