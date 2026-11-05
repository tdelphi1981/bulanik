"""Gallery figure for interval type-2 trapezoidal numbers: an illustrative 7-term linguistic scale.

Run from the repository root: `uv run --group docs python docs/scripts/fig_numbers_it2.py`.
The scale is illustrative, in the style of the IT2 TOPSIS literature: its term values are not
a transcription of any one paper's table. Every plotted value, including the DTraT score under
each term, is computed by bulanik; the script only adds labels.
"""

from __future__ import annotations

from _figutil import note, panels, run

from bulanik import IT2TrFN, plot

# (term, upper trapezoid, lower trapezoid): heights 1 above, 0.9 below
SCALE = [
    ("VL", (0, 0, 0, 0.1), (0, 0, 0, 0.05)),
    ("L", (0, 0.1, 0.1, 0.3), (0.05, 0.1, 0.1, 0.2)),
    ("ML", (0.1, 0.3, 0.3, 0.5), (0.2, 0.3, 0.3, 0.4)),
    ("M", (0.3, 0.5, 0.5, 0.7), (0.4, 0.5, 0.5, 0.6)),
    ("MH", (0.5, 0.7, 0.7, 0.9), (0.6, 0.7, 0.7, 0.8)),
    ("H", (0.7, 0.9, 0.9, 1.0), (0.8, 0.9, 0.9, 0.95)),
    ("VH", (0.9, 1.0, 1.0, 1.0), (0.95, 1.0, 1.0, 1.0)),
]


def trapezoid():
    nums = [IT2TrFN((*u, 1, 1), (*l, 0.9, 0.9)) for _, u, l in SCALE]
    names = [s for s, _, _ in SCALE]
    _, ax = panels(1, 1, figsize=(9.6, 4.2))
    plot.it2_number(nums, labels=names, ax=ax)
    ax.set_title("A 7-term interval type-2 scale and the DTraT score of each term", loc="left", fontsize=9.5)
    for n, term in zip(nums, names):
        peak = (n.upper[1] + n.upper[2]) / 2
        note(ax, peak, -0.19, f"{n.score():.3f}", va="top", transform=ax.get_xaxis_transform(), clip_on=False)
    note(ax, -0.045, -0.19, "DTraT", ha="right", va="top", ink="muted", transform=ax.get_xaxis_transform(),
         clip_on=False)
    return ax


FIGURES = {"it2-trapezoid": trapezoid}

if __name__ == "__main__":
    run(FIGURES)
