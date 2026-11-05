# Example 10: Mamdani tipper

The classic restaurant tip: service 3 of 10 and food 8 of 10, with three rules.

## Source

```python title="examples/10_mamdani_tipper.py"
--8<-- "examples/10_mamdani_tipper.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/10_mamdani_tipper.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- `explain` shows the fuzzified inputs and each rule's activation, so the answer can be traced.
- OR rules take the larger of the two degrees; the output is the centroid of the clipped, aggregated consequents.
- A high `resolution` makes the sampled centroid match the closed-form value to many digits.

## Go further

- Guide: [Mamdani inference](../guide/mamdani.md)
- API: [`bulanik.inference`][bulanik.inference]

Run it yourself: `python examples/10_mamdani_tipper.py`.
