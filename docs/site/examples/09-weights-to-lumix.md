# Example 09: MCDM scores as LumiX data

TOPSIS scores the suppliers, and a LumiX binary model picks the best pair within a setup budget.

!!! note "Needs an extra"
    This example uses the LumiX bridge (`pip install "bulanik[lumix]"`). The documentation build installs it, so the output below is real.

## Source

```python title="examples/09_weights_to_lumix.py"
--8<-- "examples/09_weights_to_lumix.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/09_weights_to_lumix.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- `scores_as_data` hands a ranking to LumiX as records, so the optimisation model can use the scores as coefficients.
- The best pair is not the two highest ranked suppliers: Cedar and Atlas would cost 95, over the budget of 75, so the model picks Cedar and Delta.
- The same suppliers and scores appear in example 04.

## Go further

- Guide: [LumiX bridge](../guide/bridge-lumix.md)
- Guide: [Ranking](../guide/ranking.md)
- API: [`bulanik.lumix`][bulanik.lumix]
- API: [`bulanik.mcdm`][bulanik.mcdm]

Run it yourself: `python examples/09_weights_to_lumix.py`.
