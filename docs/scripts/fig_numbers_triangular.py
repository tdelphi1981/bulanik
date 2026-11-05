"""Gallery figures for triangular and trapezoidal fuzzy numbers.

Run from the repository root: `uv run --group docs python docs/scripts/fig_numbers_triangular.py`.
Every plotted value is computed by bulanik; the script only adds labels.
"""

from __future__ import annotations

from _figutil import bracket, dot, note, panels, run, the_text, tokens

from bulanik import TFN, TrFN, plot
# private on purpose (docs scripts only): number text in bulanik.plot's own format
from bulanik.plot._style import fmt
from bulanik.scales import EXAMPLE_TFN_7

LEVEL = 0.5


def _corners(n) -> tuple[float, ...]:
    """The defining points, through the public fields."""
    return (n.l, n.m, n.u) if isinstance(n, TFN) else (n.a, n.b, n.c, n.d)


def _interval(lo: float, hi: float) -> str:
    return f"[{fmt(round(lo, 4))}, {fmt(round(hi, 4))}]"


def anatomy():
    """TFN and TrFN side by side: defining points, support, core and one alpha-cut."""
    fig, (left, right) = panels(1, 2, figsize=(7.4, 3.4))
    cases = [
        (left, TFN(2, 4, 7), ("l", "m", "u"), "TFN(2, 4, 7)"),
        (right, TrFN(1, 3, 5, 8), ("a", "b", "c", "d"), "TrFN(1, 3, 5, 8)"),
    ]
    for ax, n, names, title in cases:
        plot.membership(n, alpha_cut=LEVEL, ax=ax)
        pts = list(_corners(n))
        ax.set_xticks(pts, [f"{fmt(p)}\n${s}$" for p, s in zip(pts, names)])
        ax.set_title(title, loc="left")
        lo, hi = n.alpha_cut(0.0)
        note(ax, (lo + hi) / 2, 0.06, f"support {_interval(lo, hi)}")
        c0, c1 = n.alpha_cut(1.0)
        if c1 - c0 > 1e-12:
            bracket(ax, c0, c1, 1.07, f"core {_interval(c0, c1)}")
        else:
            note(ax, c0, 1.07, f"core {{{fmt(c0)}}}")
        a, b = n.alpha_cut(LEVEL)
        note(ax, (a + b) / 2, LEVEL + 0.05, rf"$\alpha$-cut {_interval(a, b)}")
        ax.set_ylim(0, 1.22)
    return fig


def _name(n) -> str:
    return f"{type(n).__name__}(" + ", ".join(fmt(v) for v in _corners(n)) + ")"


def arithmetic():
    """Alpha-cut arithmetic: A, B, A + B and A * B as stacked panels on one x axis."""
    a, b = TFN(1, 2, 4), TFN(2, 3, 5)
    rows = [("A", a), ("B", b), ("A + B", a + b), (r"A $\times$ B", a * b)]
    fig, axes = panels(len(rows), 1, figsize=(7.4, 6.6), sharex=True)
    t = tokens()
    right_edge = max(_corners(n)[-1] for _, n in rows) + 1
    for k, (ax, (name, n)) in enumerate(zip(axes, rows)):
        plot.alpha_cuts(n, levels=5, ax=ax)  # levels 0, 0.25, 0.5, 0.75, 1
        ax.set_xlabel("")
        ax.set_title(f"{name} = {_name(n)}", loc="left", fontsize=10.5)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, LEVEL, 1], ["0", fmt(LEVEL), "1"])
        lo, up = n.alpha_cut(LEVEL)
        for x in (lo, up):
            dot(ax, x, LEVEL, t["text"], size=6)
        # the interval sits right after the bar end it describes
        note(ax, up + 0.45, LEVEL, _interval(lo, up), ha="left", va="center", size=9.5, ink="text")
    # the exact cut of a product is the interval product of the operand cuts;
    # TFN multiplication keeps a triangle, an approximation of it
    ax = axes[-1]
    levels = [i / 40 for i in range(41)]
    lefts, rights = [], []
    for al in levels:
        (p0, p1), (q0, q1) = a.alpha_cut(al), b.alpha_cut(al)
        prods = [p0 * q0, p0 * q1, p1 * q0, p1 * q1]
        lefts.append(min(prods))
        rights.append(max(prods))
    ax.plot(lefts + rights[::-1], levels + levels[::-1], color=t["text_secondary"], lw=1.3, zorder=5)
    k = levels.index(0.8)
    note(ax, lefts[k] - 0.5, 0.8, "exact cuts", ha="right", va="center", ink="muted")
    lo, up = lefts[levels.index(LEVEL)], rights[levels.index(LEVEL)]
    for x in (lo, up):  # the exact cut's ends, small and muted beside the triangle's
        dot(ax, x, LEVEL, t["muted"], size=4.5)
    note(ax, 13.4, 0.86, f"exact cut at {fmt(LEVEL)}: {_interval(lo, up)}", ha="left", va="center", ink="muted")
    ax.set_xlabel("$x$")
    ax.set_xlim(0, right_edge)
    ticks = list(range(0, int(right_edge) + 1, 2))
    ax.set_xticks(ticks, [str(v) for v in ticks])
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    return fig


def defuzzification():
    """Centroid, graded mean and mean of one skewed triangle."""
    ax = plot.defuzzification(TFN(1, 2, 7))
    ax.set_title("TFN(1, 2, 7) defuzzified", loc="left")
    ax.figure.set_size_inches(6.4, 3.8)
    return ax


def scale():
    """Every term of the 7-term example scale on one axis."""
    ax = plot.variable(EXAMPLE_TFN_7, x=6.5)
    ax.set_title("EXAMPLE_TFN_7 and a rating of 6.5", loc="left")
    # the library writes the input tag at the foot of its marker, across the curves;
    # move it to the top, above the term names
    text = the_text(ax, startswith="x = ")
    text.xy = (6.5, 1.17)  # an annotation: anchor in data, offset in points
    text.set_position((0, 0))
    text.set_ha("center")
    text.set_va("bottom")
    ax.plot([6.5, 6.5], [1.0, 1.165], color=tokens()["muted"], lw=0.7, zorder=3)  # tie the tag to its line
    ax.set_ylim(0, 1.3)
    return ax


FIGURES = {
    "tfn-anatomy": anatomy,
    "alpha-cut-arithmetic": arithmetic,
    "defuzzification": defuzzification,
    "linguistic-scale-tfn": scale,
}

if __name__ == "__main__":
    run(FIGURES)
