# Example 21: Objective and subjective weights

CRITIC weights from the data and PIPRECIA weights from judgements, both feeding COPRAS.

## Source

```python title="examples/21_critic_piprecia.py"
--8<-- "examples/21_critic_piprecia.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/21_critic_piprecia.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- CRITIC rewards criteria that vary and that conflict with the others; PIPRECIA reflects what the decision maker says.
- With `s_inverse` PIPRECIA runs the forward and the inverse pass and averages them.
- The two weightings agree on the best and worst alternative and swap the two in between.

## Go further

- Guide: [Weighting](../guide/weighting.md)
- Guide: [More weighting methods](../guide/weighting-more.md)
- Guide: [Ranking](../guide/ranking.md)
- API: [`bulanik.mcdm`][bulanik.mcdm]

Run it yourself: `python examples/21_critic_piprecia.py`.
