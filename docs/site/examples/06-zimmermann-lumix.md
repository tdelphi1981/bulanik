# Example 06: Zimmermann fuzzy goals with LumiX

Two soft goals (a profit goal and a cost goal) on a LumiX model, solved with Zimmermann's max-min approach.

!!! note "Needs an extra"
    This example uses the LumiX bridge (`pip install "bulanik[lumix]"`). The documentation build installs it, so the output below is real.

## Source

```python title="examples/06_zimmermann_lumix.py"
--8<-- "examples/06_zimmermann_lumix.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/06_zimmermann_lumix.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- The model keeps its hard constraints; only the goals are fuzzy.
- The common satisfaction level lambda is what the solver maximises, and both goals end with the same membership.
- `solve_zimmermann` returns lambda together with each goal's value (`goal_values`) and membership (`memberships`).

## Go further

- Guide: [LumiX bridge](../guide/bridge-lumix.md)
- API: [`bulanik.lumix`][bulanik.lumix]

Run it yourself: `python examples/06_zimmermann_lumix.py`.
