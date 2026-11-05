# Example 16: ANFIS trained with sezgi

The same model class trained without gradients by sezgi's jDE, compared with Jang's hybrid rule on a sine curve.

!!! note "Needs an extra"
    This example uses the sezgi bridge (`pip install "bulanik[sezgi]"`). The documentation build installs it, so the output below is real.

## Source

```python title="examples/16_anfis_with_sezgi.py"
--8<-- "examples/16_anfis_with_sezgi.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/16_anfis_with_sezgi.py", run_name="__main__")
print(buf.getvalue(), end="")
```

## What to notice

- `train_anfis` treats the parameters as one vector and spends a fixed budget of evaluations.
- On this smooth problem the hybrid rule reaches a much lower error with far less work; metaheuristics shine when gradients are unavailable.

## Go further

- Guide: [sezgi bridge](../guide/bridge-sezgi.md)
- Guide: [ANFIS](../guide/anfis.md)
- API: [`bulanik.sezgi`][bulanik.sezgi]
- API: [`bulanik.anfis`][bulanik.anfis]

Run it yourself: `python examples/16_anfis_with_sezgi.py`.
