# Example 02: Linguistic scales and expert aggregation

Three experts rate two suppliers on two criteria with words from a linguistic scale; the words become intuitionistic numbers and are aggregated.

## Source

```python title="examples/02_scale_and_experts.py"
--8<-- "examples/02_scale_and_experts.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/02_scale_and_experts.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- `convert` turns nested lists of words into numbers without a loop of your own.
- `aggregate_experts` takes `[expert][alternative][criterion]` and returns `[alternative][criterion]`.
- Supplier 2 / criterion 2 ends with a negative difference score: the experts' non-membership outweighs membership.

## Go further

- Guide: [Sets and scales](../guide/sets-and-scales.md)
- Guide: [Aggregation](../guide/aggregation.md)
- API: [`bulanik.scales`][bulanik.scales]
- API: [`bulanik.aggregate`][bulanik.aggregate]

Run it yourself: `python examples/02_scale_and_experts.py`.
