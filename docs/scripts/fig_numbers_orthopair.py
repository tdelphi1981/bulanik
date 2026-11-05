"""Gallery figures for orthopair numbers (IFN, PFN, FFN, q-rung).

Run from the repository root: `uv run --group docs python docs/scripts/fig_numbers_orthopair.py`.
Every plotted value is computed by bulanik; the script only adds labels.
"""

from __future__ import annotations

from _figutil import arrow, callout, dot, key, label, panels, run, the_text, tokens

from bulanik import FFN, IFN, PFN, QROFN, plot
from bulanik.aggregate import wa, wg
# private on purpose (docs scripts only): number text in bulanik.plot's own format
from bulanik.plot._style import fmt
from bulanik.scales import EXAMPLE_IFN_7

# marks for results: (palette slot, marker); inputs keep bulanik.plot's own points
ALGEBRAIC = (2, "s")
EINSTEIN = (6, "D")
SQUARE = (5.8, 5.6)
NAMES = {"algebraic": "algebraic", "einstein": "Einstein"}
WA = (2, "s")
WG = (6, "D")


def _pair(n) -> str:
    return f"({fmt(round(n.mu, 3))}, {fmt(round(n.nu, 3))})"


def regions():
    """Nested feasible regions for q = 1, 2, 3, 5 with one number of each kind."""
    nums = [IFN(0.5, 0.35), PFN(0.75, 0.5), FFN(0.85, 0.6), QROFN(0.92, 0.8, 5)]
    labels = [f"{type(n).__name__} {_pair(n)}" for n in nums]
    _, ax = panels(1, 1, figsize=SQUARE)
    plot.orthopair(nums, regions=(1, 2, 3, 5), labels=labels, hesitancy=True, ax=ax)
    ax.set_title(r"Feasible regions $\mu^q + \nu^q \leq 1$", loc="left")
    # the placer puts this one across the q = 5 curve: two lines, west of the point,
    # fit between the q = 3 and q = 5 curves
    text = the_text(ax, startswith="QROFN")
    text.set_text(text.get_text().replace(" ", "\n", 1))
    text.set_multialignment("right")
    text.set_position((-7, 0))
    text.set_ha("right")
    text.set_va("center")
    return ax


def operators():
    """Algebraic vs Einstein sum and product of two IFNs, as arrows from x in the mu-nu plane."""
    t = tokens()
    x, y = IFN(0.35, 0.45), IFN(0.5, 0.3)
    _, ax = panels(1, 1, figsize=SQUARE)
    plot.orthopair([x, y], q=1, labels=["x", "y"], ax=ax)
    ax.set_title("Algebraic vs Einstein operators", loc="left")
    # sums: labels beside their markers, as (dx, dy points, ha, va); products: no room
    # between the markers and the y axis, so callouts into the empty area beyond the boundary
    places = {
        ("add", "algebraic"): (-9, -6, "right", "top"),
        ("add", "einstein"): (9, 2, "left", "bottom"),
    }
    callouts = {("mul", "einstein"): (0.30, 0.86), ("mul", "algebraic"): (0.40, 0.76)}
    for op, sign in (("add", "+"), ("mul", r"$\times$")):
        for (slot, marker), ops in ((ALGEBRAIC, "algebraic"), (EINSTEIN, "einstein")):
            r = getattr(x, op)(y, ops=ops)
            c = t["series"][slot]
            # arcs keep the two families apart: algebraic below-left, Einstein above-right
            bend = (0.14 if ops == "algebraic" else -0.14) * (1 if op == "add" else -1)
            arrow(ax, (x.mu, x.nu), (r.mu, r.nu), c, lw=1.3, shrink=5, shrink_end=10, rad=bend)
            dot(ax, r.mu, r.nu, c, marker=marker, size=8.5, zorder=10)
            text = f"x {sign} y  {NAMES[ops]}\n{_pair(r)}"
            if (op, ops) in callouts:
                callout(ax, (r.mu, r.nu), callouts[op, ops], text)
            else:
                dx, dy, ha, va = places[op, ops]
                label(ax, r.mu, r.nu, text, dx=dx, dy=dy, ha=ha, va=va)
    key(ax, [("o", t["series"][1], "operands x, y"),
             (ALGEBRAIC[1], t["series"][ALGEBRAIC[0]], "algebraic"),
             (EINSTEIN[1], t["series"][EINSTEIN[0]], "Einstein")], loc="upper right")
    return ax


def aggregation():
    """Weighted average (WA) and weighted geometric mean (WG) of three IFNs."""
    t = tokens()
    nums = [IFN(0.85, 0.1), IFN(0.4, 0.15), IFN(0.2, 0.65)]
    w = [0.5, 0.3, 0.2]
    # weight labels: (dx, dy, ha, va) chosen clear of the spokes
    spots = [(-8, -5, "right", "top"), (-8, -5, "right", "top"), (9, 0, "left", "center")]
    _, ax = panels(1, 1, figsize=SQUARE)
    results = (("WA", WA, wa(nums, w)), ("WG", WG, wg(nums, w)))
    # spokes first, so the bulanik.plot point labels are placed on top of them
    for _name, (slot, _marker), r in results:
        for n in nums:
            ax.plot([n.mu, r.mu], [n.nu, r.nu], color=t["series"][slot], lw=0.9, alpha=0.6, zorder=3)
    plot.orthopair(nums, q=1, ax=ax)
    ax.set_title("WA vs WG of three IFNs", loc="left")
    for n, wi, (dx, dy, ha, va) in zip(nums, w, spots):
        label(ax, n.mu, n.nu, f"w = {fmt(wi)}", dx=dx, dy=dy, ha=ha, va=va)
    for name, (slot, marker), r in results:
        dot(ax, r.mu, r.nu, t["series"][slot], marker=marker, size=9)
        if name == "WA":
            label(ax, r.mu, r.nu, f"WA {_pair(r)}", dx=10, dy=0, ha="left", va="center", ink="text")
        else:
            label(ax, r.mu, r.nu, f"WG {_pair(r)}", dx=-10, dy=0, ha="right", va="center", ink="text")
    key(ax, [("o", t["series"][1], "ratings with weights w"),
             (WA[1], t["series"][WA[0]], "WA (IFWA)"),
             (WG[1], t["series"][WG[0]], "WG (IFWG)")], loc="upper right")
    return ax


def scale():
    """The 7-term example IFN scale as points in the mu-nu plane."""
    _, ax = panels(1, 1, figsize=SQUARE)
    terms = EXAMPLE_IFN_7.terms
    nums = [EXAMPLE_IFN_7[s] for s in terms]
    # plot.variable(EXAMPLE_IFN_7) labels points on their east side, where these
    # labels would cross the boundary; the docs label them inside the region instead
    plot.orthopair(nums, q=1, ax=ax)
    ax.set_title("EXAMPLE_IFN_7", loc="left")
    # a hairline through the terms in scale order: the scale climbs from EL to EH
    ax.plot([n.mu for n in nums], [n.nu for n in nums], color=tokens()["muted"], lw=0.8, zorder=3)
    for term, n in zip(terms, nums):
        if n.mu < 0.1:  # no room on the west side: below the point
            label(ax, n.mu, n.nu, term, dx=0, dy=-8, ha="center", va="top", size=9)
        else:
            label(ax, n.mu, n.nu, term, dx=-8, dy=0, ha="right", va="center", size=9)
    return ax


FIGURES = {
    "orthopair-regions": regions,
    "orthopair-operators": operators,
    "orthopair-aggregation": aggregation,
    "linguistic-scale-ifn": scale,
}

if __name__ == "__main__":
    run(FIGURES)
