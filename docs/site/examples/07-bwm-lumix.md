# Example 07: Best-worst method with LumiX

The linear best-worst method on a slightly inconsistent set of comparisons.

!!! note "Needs an extra"
    This example uses the LumiX bridge (`pip install "bulanik[lumix]"`). The documentation build installs it, so the output below is real.

## Source

```python title="examples/07_bwm_lumix.py"
--8<-- "examples/07_bwm_lumix.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/07_bwm_lumix.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- The comparisons are deliberately inconsistent (2 times 3 is 6, not 4), so xi is greater than zero.
- Weights sum to 1; xi is the optimum of the linear model, a measure of inconsistency.
- Only the best and the worst criterion are compared with the others.

## Go further

- Guide: [Weighting](../guide/weighting.md#best-worst-method)
- Guide: [LumiX bridge](../guide/bridge-lumix.md)
- API: [`bulanik.lumix`][bulanik.lumix]

Run it yourself: `python examples/07_bwm_lumix.py`.
