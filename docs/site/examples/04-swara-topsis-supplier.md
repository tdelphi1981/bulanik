# Example 04: SWARA weights and TOPSIS

SWARA criteria weights from importance ordering, then TOPSIS over four suppliers with mixed cost and benefit criteria.

## Source

```python title="examples/04_swara_topsis_supplier.py"
--8<-- "examples/04_swara_topsis_supplier.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/04_swara_topsis_supplier.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- The criteria are listed most important first; SWARA needs no comparison matrix.
- `types` marks `price` as a cost criterion, so a lower price is better.
- Results carry the names of the methods used, so a printed ranking is self-describing.

## Go further

- Guide: [Weighting](../guide/weighting.md)
- Guide: [Ranking](../guide/ranking.md)
- API: [`bulanik.mcdm`][bulanik.mcdm]

Run it yourself: `python examples/04_swara_topsis_supplier.py`.
