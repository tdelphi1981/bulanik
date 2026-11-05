# Example 23: Interval-valued AHP compared

One set of linguistic judgements on four criteria is written as interval-valued Pythagorean and as interval-valued intuitionistic numbers, and goes through both methods of `ahp_interval_valued` and through `ahp_fuzzy_numbers`.

## Source

```python title="examples/23_interval_valued_ahp.py"
--8<-- "examples/23_interval_valued_ahp.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/23_interval_valued_ahp.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- All five runs give the same order; the spread is what changes. The multiplicative method, through $1000^{d/q}$, gives the top criterion about 0.6, while the possibility method and `ahp_fuzzy_numbers` give it between 0.31 and 0.35.
- `method="possibility"` is defined for $q = 1$, so the Pythagorean matrix raises `ValueError` and the intuitionistic version of the judgements takes its place. The multiplicative method takes either.
- The possibility method also reports interval weights; neighbouring ones overlap, which is why its crisp weights stay close together.
- The consistency ratio comes from the crisp Saaty matrix passed as `consistency_matrix`, so it is the same for every run.

## Go further

- Guide: [More weighting methods](../guide/weighting-more.md#interval-valued-ahp)
- API: [`ahp_interval_valued`][bulanik.mcdm.weighting.ahp_interval_valued], [`ahp_fuzzy_numbers`][bulanik.mcdm.weighting.ahp_fuzzy_numbers]

Run it yourself: `python examples/23_interval_valued_ahp.py`.
