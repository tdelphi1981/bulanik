# Example 22: Fuzzy DEMATEL

Three experts' triangular influence matrices are converted by CFCS, analysed with DEMATEL, and the factor weights drive a WISP ranking.

## Source

```python title="examples/22_fuzzy_dematel.py"
--8<-- "examples/22_fuzzy_dematel.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/22_fuzzy_dematel.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- `cfcs` takes the minimum and maximum per cell across the experts, converts each expert, then averages.
- A factor is a cause when its net relation D - R is positive, an effect otherwise.
- `result.as_weights()` makes the DEMATEL weights usable by every ranking method.

## Go further

- Guide: [More weighting methods](../guide/weighting-more.md)
- Guide: [Ranking](../guide/ranking-more.md)
- API: [`bulanik.mcdm`][bulanik.mcdm]

Run it yourself: `python examples/22_fuzzy_dematel.py`.
