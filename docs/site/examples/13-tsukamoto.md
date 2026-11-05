# Example 13: Tsukamoto

Monotonic consequents are inverted at each rule's firing strength.

## Source

```python title="examples/13_tsukamoto.py"
--8<-- "examples/13_tsukamoto.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/13_tsukamoto.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- Only monotonic terms (linear ramps and sigmoids) are allowed as Tsukamoto consequents.
- Each rule yields a crisp value `z_k` by inverting its term at the firing strength; the output is their weighted average.

## Go further

- Guide: [Sugeno and Tsukamoto](../guide/sugeno-tsukamoto.md)
- API: [`bulanik.inference`][bulanik.inference]

Run it yourself: `python examples/13_tsukamoto.py`.
