# Example 08: Fuzzy BWM with sezgi

The fuzzy best-worst method of Guo and Zhao, solved with the jDE metaheuristic from sezgi.

!!! note "Needs an extra"
    This example uses the sezgi bridge (`pip install "bulanik[sezgi]"`). The documentation build installs it, so the output below is real.

## Source

```python title="examples/08_fuzzy_bwm_sezgi.py"
--8<-- "examples/08_fuzzy_bwm_sezgi.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/08_fuzzy_bwm_sezgi.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- The comparisons are linguistic terms of the five-term fuzzy BWM scale.
- Weights are triangular numbers; the last number in each line is its crisp value.
- `k*` is the optimal inconsistency level and CR the consistency ratio derived from it. The run is seeded, so the output is reproducible.

## Go further

- Guide: [sezgi bridge](../guide/bridge-sezgi.md)
- Guide: [Weighting](../guide/weighting.md#best-worst-method)
- API: [`bulanik.sezgi`][bulanik.sezgi]

Run it yourself: `python examples/08_fuzzy_bwm_sezgi.py`.
