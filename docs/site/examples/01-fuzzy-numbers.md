# Example 01: Fuzzy numbers

Arithmetic on triangular, intuitionistic, Pythagorean and spherical numbers, a second operator family, scores and a weighted average.

## Source

```python title="examples/01_fuzzy_numbers.py"
--8<-- "examples/01_fuzzy_numbers.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/01_fuzzy_numbers.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- The algebraic and Einstein families give different sums for the same two IFNs; the result always reports the operator used.
- `score()` is a number you can sort on, but each type has its own default (centroid for triangular, difference for orthopair).
- Weights are validated, and `wa` returns the same number type you put in.

## Go further

- Guide: [Triangular numbers](../guide/numbers-triangular.md)
- Guide: [Orthopair numbers](../guide/numbers-orthopair.md)
- Guide: [Aggregation](../guide/aggregation.md)
- API: [`bulanik.numbers`][bulanik.numbers]
- API: [`bulanik.aggregate`][bulanik.aggregate]

Run it yourself: `python examples/01_fuzzy_numbers.py`.
