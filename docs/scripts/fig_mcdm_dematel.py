"""Gallery figure for DEMATEL: the cause-effect diagram of the R `dematel` vignette example.

Run from the repository root: `uv run --group docs python docs/scripts/fig_mcdm_dematel.py`.
Every plotted value is computed by bulanik; the script only lays out the panel.
"""

from __future__ import annotations

from _figutil import panels, run

from bulanik import plot
from bulanik.mcdm import dematel

# the 5 x 5 direct-relation matrix of the R `dematel` package vignette (tests/test_dematel.py)
A = [[0, 2, 4, 3, 4], [3, 0, 2, 0, 0], [3, 3, 0, 0, 0], [3, 0, 4, 0, 2], [4, 0, 2, 0, 0]]


# the default threshold (mean of T, 0.33) keeps 13 influences and hides the pattern; 0.45 keeps the strongest
THRESHOLD = 0.45


def cause_effect():
    _, ax = panels(1, 1, figsize=(6.6, 5.0))
    plot.dematel(dematel(A, threshold=THRESHOLD), ax=ax)
    ax.set_title("DEMATEL cause-effect diagram", loc="left", fontsize=9.5)
    return ax


FIGURES = {"dematel-cause-effect": cause_effect}

if __name__ == "__main__":
    run(FIGURES)
