# Example 17: Plotting with bulanik.plot

Figures of a fuzzy number, an orthopair number, a linguistic scale, a Mamdani system and an MCDM ranking, each in a light and a dark theme.

!!! note "Needs an extra"
    This example uses the plot extra (`pip install "bulanik[plot]"`). The documentation build installs it, so the output below is real.

## Source

```python title="examples/17_plotting.py"
--8<-- "examples/17_plotting.py"
```

## Output

Run when this page is built:

```python exec="on" result="text"
import contextlib, io, re, runpy
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path("examples/17_plotting.py", run_name="__main__")
print(re.sub(r"\S*bulanik-plots-\w+", "<tmp>", buf.getvalue()), end="")
```

The first figure the script saves, `membership-light.svg` and `membership-dark.svg`
(shown in your colour scheme; the docs build ships the same `plot.membership` call as a committed figure):

<figure markdown="span">
  ![Membership functions of TFN(1, 2, 4) and TrFN(3, 4, 5, 7), with the alpha-cut at 0.5 marked](../assets/figures/example-plotting-membership-light.svg#only-light)
  ![Membership functions of TFN(1, 2, 4) and TrFN(3, 4, 5, 7), with the alpha-cut at 0.5 marked](../assets/figures/example-plotting-membership-dark.svg#only-dark)
</figure>

## What to notice

- Every plot function takes an `ax` and returns it, so figures compose with ordinary matplotlib.
- Each figure is saved twice, once per theme; the files go to a temporary directory (shown as `<tmp>` below) and the docs [gallery](../guide/gallery.md) shows the same style.
- `bulanik.plot` needs the `plot` extra and matplotlib.

## Go further

- Guide: [Fuzzy numbers, visually](../guide/gallery.md)
- API: [`bulanik.plot`][bulanik.plot]

Run it yourself: `python examples/17_plotting.py`.
