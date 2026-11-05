# Example 14: Interval type-2 versus type-1 under noise

Noise sensitivity of an interval type-2 and a type-1 Sugeno approximator of y = x.

## Source

```python title="examples/14_it2_vs_t1_noise.py"
--8<-- "examples/14_it2_vs_t1_noise.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/14_it2_vs_t1_noise.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- The measure is how much the output moves when the input is perturbed, not how well it fits.
- The uncertain-sigma footprint of uncertainty smooths the interval type-2 system, so it reacts less to noise here.
- The run is seeded with a fixed `random.Random`, so the numbers are reproducible.

## Go further

- Guide: [Interval type-2](../guide/type2.md)
- API: [`bulanik.type2`][bulanik.type2]

Run it yourself: `python examples/14_it2_vs_t1_noise.py`.
