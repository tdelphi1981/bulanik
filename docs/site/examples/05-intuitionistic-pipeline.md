# Example 05: Intuitionistic pipeline

Seven-term IF ratings from three experts are aggregated into a decision matrix, criteria are weighted with IF-SWARA and the plans are ranked with VIKOR.

## Source

```python title="examples/05_intuitionistic_pipeline.py"
--8<-- "examples/05_intuitionistic_pipeline.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/05_intuitionistic_pipeline.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- The whole pipeline stays intuitionistic until the final score; nothing is defuzzified early.
- The `operators` line records the normalisation, aggregation, operator family and score the ranking used.
- VIKOR ranks by Q, lower is better, so Plan A (smallest Q) comes first.

## Go further

- Guide: [Aggregation](../guide/aggregation.md)
- Guide: [Weighting](../guide/weighting.md)
- Guide: [Ranking](../guide/ranking.md)
- API: [`bulanik.mcdm`][bulanik.mcdm]

Run it yourself: `python examples/05_intuitionistic_pipeline.py`.
