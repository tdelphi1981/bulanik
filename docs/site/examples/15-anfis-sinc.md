# Example 15: ANFIS on the sinc surface

A grid ANFIS learns the two-dimensional sinc surface with Jang's hybrid rule, then exports to a Sugeno system.

## Source

```python title="examples/15_anfis_sinc.py"
--8<-- "examples/15_anfis_sinc.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/15_anfis_sinc.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- The hybrid rule solves the consequents by least squares and updates the premises by gradient descent.
- `to_sugeno` exports the trained model as an ordinary SugenoFIS with one rule per combination of input terms (16 here).

- The least-squares step solves the consequents in epoch 1; the premises barely move at `lr=0.01`, so the RMSE stays flat after that.

## Go further

- Guide: [ANFIS](../guide/anfis.md)
- API: [`bulanik.anfis`][bulanik.anfis]

Run it yourself: `python examples/15_anfis_sinc.py`.
