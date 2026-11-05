# Example 03: Buckley fuzzy AHP

Three criteria compared with the Saaty triangular scale, weighted with Buckley's geometric-mean method.

## Source

```python title="examples/03_fuzzy_ahp.py"
--8<-- "examples/03_fuzzy_ahp.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/03_fuzzy_ahp.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- Only the upper triangle of the comparison matrix is typed; the lower triangle is the reciprocal.
- Each criterion gets a fuzzy weight and a crisp one; the crisp weights sum to 1.
- The consistency ratio (CR) is reported; a value under about 0.1 is conventionally acceptable.

## Go further

- Guide: [Weighting](../guide/weighting.md)
- API: [`bulanik.mcdm`][bulanik.mcdm]

Run it yourself: `python examples/03_fuzzy_ahp.py`.
