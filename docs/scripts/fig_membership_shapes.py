"""Figure for the sets-and-scales guide: the seven membership-function shapes.

Run from the repository root: `uv run --group docs python docs/scripts/fig_membership_shapes.py`.
Every curve is evaluated by bulanik; the script only lays out the panels and titles.
"""

from __future__ import annotations

from _figutil import panels, run

from bulanik import plot
from bulanik.sets import (
    LinguisticVariable,
    bell,
    gaussian,
    linear_decreasing,
    linear_increasing,
    sigmoid,
    trapezoidal,
    triangular,
)

SHAPES = [
    ("triangular(1, 3, 7)", (0, 8), triangular(1, 3, 7)),
    ("trapezoidal(1, 3, 5, 7)", (0, 8), trapezoidal(1, 3, 5, 7)),
    ("gaussian(4, 1.2)", (0, 8), gaussian(4, 1.2)),
    ("bell(2, 3, 4)", (0, 8), bell(2, 3, 4)),
    ("sigmoid(2, 4)", (0, 8), sigmoid(2, 4)),
    ("linear_increasing(2, 6)", (0, 8), linear_increasing(2, 6)),
    ("linear_decreasing(2, 6)", (0, 8), linear_decreasing(2, 6)),
]


def shapes():
    """One panel per factory; the eighth cell stays empty."""
    fig, axes = panels(2, 4, figsize=(9.6, 4.6), sharey=True)
    flat = list(axes.flat)
    for ax, (title, universe, mf) in zip(flat, SHAPES):
        plot.variable(LinguisticVariable("x", universe, {"": mf}), ax=ax)
        ax.set_title(title, loc="left", fontsize=9.5)
    flat[-1].set_axis_off()
    for ax in flat[:-1]:
        ax.set_xlabel("")
    return fig


FIGURES = {"membership-shapes": shapes}

if __name__ == "__main__":
    run(FIGURES)
