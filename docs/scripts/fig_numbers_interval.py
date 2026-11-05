"""Gallery figures for interval-valued orthopair numbers (IVIFN, IVPFN, IVFFN).

Run from the repository root: `uv run --group docs python docs/scripts/fig_numbers_interval.py`.
Every plotted value is computed by bulanik; the script only adds labels.
"""

from __future__ import annotations

from _figutil import label, panels, run, tokens

from bulanik import IVFFN, IVIFN, IVPFN, plot
from bulanik.plot._style import fmt

SQUARE = (5.8, 5.6)


def _on_boundary(mu: float, q: int) -> float:
    """The nu with mu^q + nu^q = 1."""
    return (1 - mu**q) ** (1 / q)


def regions():
    """One IVIFN, IVPFN and IVFFN rectangle over the q = 1, 2, 3 feasible regions."""
    # each upper-right corner (mu_U, nu_U) lies on the boundary mu^q + nu^q = 1 of its own rung
    nums = [
        IVIFN(0.15, 0.3, 0.45, _on_boundary(0.3, 1)),
        IVPFN(0.35, 0.5, 0.55, _on_boundary(0.5, 2)),
        IVFFN(0.78, 0.88, 0.2, _on_boundary(0.88, 3)),
    ]
    names = ["IVIFN", "IVPFN", "IVFFN"]
    _, ax = panels(1, 1, figsize=SQUARE)
    plot.iv_orthopair(nums, regions=(1, 2, 3), labels=names, ax=ax)
    ax.set_title(r"Rectangles $[\mu_L, \mu_U] \times [\nu_L, \nu_U]$ in the regions $\mu^q + \nu^q \leq 1$",
                 loc="left", fontsize=9.5)
    return ax


def operations():
    """A, B, A (+) B and A (x) B as rectangles in the q = 1 plane."""
    a, b = IVIFN(0.2, 0.35, 0.4, 0.5), IVIFN(0.4, 0.5, 0.2, 0.35)
    nums = [a, b, a.add(b), a.mul(b)]
    names = ["A", "B", r"A $\oplus$ B", r"A $\otimes$ B"]
    _, ax = panels(1, 1, figsize=SQUARE)
    plot.iv_orthopair(nums, q=1, labels=names, ax=ax)
    ax.set_title("Arithmetic on intervals, algebraic operators", loc="left", fontsize=9.5)
    return ax


FIGURES = {
    "iv-orthopair-regions": regions,
    "iv-orthopair-operations": operations,
}

if __name__ == "__main__":
    run(FIGURES)
