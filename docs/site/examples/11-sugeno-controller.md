# Example 11: Sugeno controller

A first-order Sugeno heater controller on error and change of error.

## Source

```python title="examples/11_sugeno_controller.py"
--8<-- "examples/11_sugeno_controller.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/11_sugeno_controller.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- Consequents are linear functions of the inputs, not fuzzy sets, so there is no defuzzification step.
- The output is the weighted average of the rule outputs, weighted by the firing strengths.
- The result is a single number, easy to check by hand.

## Go further

- Guide: [Sugeno and Tsukamoto](../guide/sugeno-tsukamoto.md)
- API: [`bulanik.inference`][bulanik.inference]

Run it yourself: `python examples/11_sugeno_controller.py`.
