"""Figure for the type-2 guide: the footprint of uncertainty of the three set builders.

Run from the repository root: `uv run --group docs python docs/scripts/fig_it2_fou.py`.
"""

from __future__ import annotations

from _figutil import panels, run

from bulanik import plot
from bulanik.type2 import gaussian_uncertain_mean, gaussian_uncertain_sigma, scaled_triangle

SETS = [
    ("gaussian_uncertain_mean(4, 6, 1.2)", (0, 10), gaussian_uncertain_mean(4, 6, 1.2)),
    ("gaussian_uncertain_sigma(5, 0.8, 1.8)", (0, 10), gaussian_uncertain_sigma(5, 0.8, 1.8)),
    ("scaled_triangle(1, 5, 9, 0.6)", (0, 10), scaled_triangle(1, 5, 9, 0.6)),
]


def fou():
    fig, axes = panels(1, 3, figsize=(10.0, 3.0), sharey=True)
    for ax, (title, universe, mf) in zip(axes, SETS):
        plot.it2_set(mf, universe=universe, ax=ax)
        ax.set_title(title, loc="left", fontsize=9)
    return fig


FIGURES = {"it2-fou": fou}

if __name__ == "__main__":
    run(FIGURES)
