# Example 12: Tuning a Sugeno system with sezgi

A FIS is the evaluated function inside a sezgi problem: jDE searches the two rule constants that reproduce y = 2x + 1.

!!! note "Needs an extra"
    This example uses the sezgi bridge (`pip install "bulanik[sezgi]"`). The documentation build installs it, so the output below is real.

## Source

```python title="examples/12_tune_sugeno_with_sezgi.py"
--8<-- "examples/12_tune_sugeno_with_sezgi.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/12_tune_sugeno_with_sezgi.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- The system is a plain Python function from constants to error, so any sezgi algorithm can tune it.
- The printed mean squared error is the objective jDE minimised; it is near zero when the two constants reproduce the target.

## Go further

- Guide: [sezgi bridge](../guide/bridge-sezgi.md)
- Guide: [Sugeno and Tsukamoto](../guide/sugeno-tsukamoto.md)
- API: [`bulanik.sezgi`][bulanik.sezgi]
- API: [`bulanik.inference`][bulanik.inference]

Run it yourself: `python examples/12_tune_sugeno_with_sezgi.py`.
