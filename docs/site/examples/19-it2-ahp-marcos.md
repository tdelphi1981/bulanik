# Example 19: Interval type-2 AHP and MARCOS

Interval type-2 trapezoidal judgements weigh the criteria with the DTraT score, and two experts' ratings are ranked with MARCOS. The 7-term scale is illustrative, not a published table.

## Source

```python title="examples/19_it2_ahp_marcos.py"
--8<-- "examples/19_it2_ahp_marcos.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/19_it2_ahp_marcos.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- The linguistic scale is **illustrative**: its values are not a transcription of any published table.
- The reciprocal of an `IT2TrFN` is written out by hand (trapezoid points reversed and inverted, heights kept); the library has no reciprocal method for it.
- The default IT2 score, DTraT, is positive, so MARCOS accepts it without a `score=` argument.

## Go further

- Guide: [Interval type-2 numbers](../guide/numbers-it2.md)
- Guide: [Weighting](../guide/weighting.md)
- Guide: [Ranking](../guide/ranking.md)
- API: [`bulanik.mcdm`][bulanik.mcdm]

Run it yourself: `python examples/19_it2_ahp_marcos.py`.
