"""Figure for example 17: the membership call of examples/17_plotting.py, drawn in both themes.

Run from the repository root: `uv run --group docs python docs/scripts/fig_example_plotting.py`.
The plot is exactly the `plot.membership` call of the example, with no added annotations.
"""

from __future__ import annotations

from _figutil import run

from bulanik import TFN, TrFN, plot


def membership():
    """Two fuzzy numbers and the alpha-cut at 0.5, as in examples/17_plotting.py."""
    return plot.membership([TFN(1, 2, 4), TrFN(3, 4, 5, 7)], labels=["A", "B"], alpha_cut=0.5)


FIGURES = {"example-plotting-membership": membership}

if __name__ == "__main__":
    run(FIGURES)
