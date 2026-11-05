"""Figure for the weighting guide: AHP, SWARA and entropy weights for the same three criteria.

Run from the repository root: `uv run --group docs python docs/scripts/fig_mcdm_weights.py`.
Every bar is drawn by `bulanik.plot.weights`; the script only lays out the panels.
"""

from __future__ import annotations

from _figutil import panels, run

from bulanik import plot
from bulanik.mcdm import ahp, entropy, swara

CRITERIA = ["price", "quality", "delivery"]
SUPPLIERS = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]


def compare():
    """One panel per method; bars follow each method's own ranking."""
    results = [
        ahp([[1, 2, 3], [1 / 2, 1, 2], [1 / 3, 1 / 2, 1]], CRITERIA),
        swara([0.2, 0.5], CRITERIA),
        entropy(SUPPLIERS, CRITERIA),
    ]
    fig, axes = panels(1, 3, figsize=(10.4, 2.5))
    fig.get_layout_engine().set(w_pad=0.12, wspace=0.08)  # room between a value label and the next panel
    for ax, result in zip(axes, results):
        plot.weights(result, ax=ax)
    return fig


FIGURES = {"mcdm-weights-compare": compare}

if __name__ == "__main__":
    run(FIGURES)
