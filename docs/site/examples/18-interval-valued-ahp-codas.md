# Example 18: Interval-valued AHP and CODAS

IVIFN pairwise comparisons give criteria weights with `ahp_fuzzy_numbers`, IVIFN ratings are ranked with CODAS, and the same pipeline is repeated with interval-valued neutrosophic ratings.

## Source

```python title="examples/18_interval_valued_ahp_codas.py"
--8<-- "examples/18_interval_valued_ahp_codas.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/18_interval_valued_ahp_codas.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- Both steps use `score="normalized"`: the default orthopair score is signed and CODAS, which divides by entries, rejects it.
- `IVNeutrosophicFN` needs no such choice; its only score, `"average"`, is already non-negative.
- CODAS scores are assessment scores, higher is better, and can be negative.

## Go further

- Guide: [Aggregation](../guide/aggregation.md)
- Guide: [Weighting](../guide/weighting.md)
- Guide: [Interval-valued numbers](../guide/numbers-interval.md)
- Guide: [Ranking](../guide/ranking.md)
- API: [`bulanik.mcdm`][bulanik.mcdm]

Run it yourself: `python examples/18_interval_valued_ahp_codas.py`.
