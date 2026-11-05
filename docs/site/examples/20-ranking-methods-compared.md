# Example 20: Ranking methods compared

All ten crisp ranking methods run on one decision matrix with the same weights, and the rank table shows where they agree.

## Source

```python title="examples/20_ranking_methods_compared.py"
--8<-- "examples/20_ranking_methods_compared.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/20_ranking_methods_compared.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- Nine methods put S5 first; CODAS prefers S4 because it ranks by distance from the negative-ideal solution.
- VIKOR ranks by ascending Q, but its `ranking` still lists the best alternative first, like every other method.
- The methods share one call shape, so swapping one for another is a one-word change.

## Go further

- Guide: [Ranking](../guide/ranking.md)
- Guide: [More ranking methods](../guide/ranking-more.md)
- API: [`bulanik.mcdm`][bulanik.mcdm]

Run it yourself: `python examples/20_ranking_methods_compared.py`.
