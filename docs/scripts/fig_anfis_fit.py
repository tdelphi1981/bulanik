"""Figure for the ANFIS guide: a sinc target, the hybrid fit, the learned terms and the RMSE history.

Run from the repository root: `uv run --group docs python docs/scripts/fig_anfis_fit.py`.
The fitted curve and the learned terms are drawn by `bulanik.plot.surface` / `bulanik.plot.variable`
from the exported Sugeno system; the script adds only the target, the samples and the history line.
"""

from __future__ import annotations

import math

from _figutil import key, panels, run, tokens

from bulanik import plot
from bulanik.anfis import ANFIS


def sinc(t: float) -> float:
    return 1.0 if t == 0 else math.sin(t) / t


def trained() -> tuple[ANFIS, list[float], list[float]]:
    xs = [-10 + 20 * i / 40 for i in range(41)]
    model = ANFIS(1, 5, ranges=[(-10, 10)])
    model.fit([[x] for x in xs], [sinc(x) for x in xs], epochs=20)
    return model, xs, [sinc(x) for x in xs]


def fit():
    model, xs, ys = trained()
    fis = model.to_sugeno(["x"], output_name="y")
    fig, axes = panels(1, 2, figsize=(9.0, 3.3))
    t = tokens()
    fine = [-10 + 20 * i / 400 for i in range(401)]
    ax = axes[0]
    plot.surface(fis, resolution=201, ax=ax)
    ax.plot(fine, [sinc(x) for x in fine], color=t["text_secondary"], lw=1.1, ls=(0, (4, 3)), zorder=5)
    ax.plot(xs, ys, linestyle="none", marker="o", ms=3.2, mfc=t["surface"], mec=t["muted"], mew=0.9, zorder=4)
    key(ax, [("", t["series"][0], "ANFIS fit"), ("", t["text_secondary"], "target sinc(x), dashed"), ("o", t["muted"], "training samples")], loc="upper right")
    ax.set_ylim(-0.3, 1.05)
    ax.set_title("fit to 41 samples (5 terms)", loc="left", fontsize=9)
    plot.variable(fis.inputs[0], ax=axes[1])
    axes[1].set_title("learned input terms", loc="left", fontsize=9)
    return fig


FIGURES = {"anfis-fit": fit}

if __name__ == "__main__":
    run(FIGURES)
