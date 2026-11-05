"""Gallery figures for three-part numbers (picture, spherical, neutrosophic).

Run from the repository root: `uv run --group docs python docs/scripts/fig_numbers_threepart.py`.
Every plotted value is computed by bulanik; the script only adds titles.
"""

from __future__ import annotations

import math

from _figutil import run, the_text

from bulanik import NeutrosophicFN, PictureFN, SphericalFN, plot

# private on purpose (docs scripts only): number text in bulanik.plot's own format
from bulanik.plot._style import fmt


def _triple(name: str, n) -> str:
    return f"{name} (" + ", ".join(fmt(v) for v in n.parts) + ")"


def picture():
    """Picture fuzzy numbers on the mu-eta-nu simplex."""
    nums = [PictureFN(0.65, 0.12, 0.13), PictureFN(0.2, 0.55, 0.2), PictureFN(0.1, 0.12, 0.7),
            PictureFN(0.3, 0.25, 0.15)]
    ax = plot.picture(nums, labels=[_triple(s, n) for n, s in zip(nums, "ABCD")])
    ax.set_title("Picture fuzzy numbers", loc="left")
    # the point is the parts rescaled to sum to 1, so the grid reads (mu, eta, nu) / (1 - rho);
    # the labels keep the numbers' own parts, and the footnote says how the two relate
    text = the_text(ax, startswith="halo radius")
    text.set_text("position: the parts rescaled to sum to 1, " r"$(\mu, \eta, \nu)\,/\,(1 - \rho)$"
                  "\nhalo radius grows with the refusal " r"$\rho = 1 - \mu - \eta - \nu$")
    text.set_linespacing(1.6)
    return ax


def spherical():
    """Spherical fuzzy numbers inside the unit octant."""
    nums = [SphericalFN(0.8, 0.3, 0.4), SphericalFN(0.4, 0.7, 0.5), SphericalFN(0.3, 0.3, 0.85)]
    ax = plot.spherical(nums, labels=["A", "B", "C"])
    ax.set_title("Spherical fuzzy numbers", loc="left")
    return ax


def neutrosophic():
    """Neutrosophic numbers as T/I/F radar triangles."""
    nums = [NeutrosophicFN(0.8, 0.2, 0.1), NeutrosophicFN(0.4, 0.7, 0.35)]
    ax = plot.neutrosophic(nums, labels=[_triple(s, n) for n, s in zip(nums, "AB")])
    ax.set_title("Neutrosophic numbers", loc="left")
    # the library writes the 0.5 level beside the truth spoke, where A's edge passes;
    # the falsity spoke is clear at 0.5, so the label moves there
    angle = math.radians(-30)  # the falsity spoke
    text = the_text(ax, equals="0.5")
    text.set_position((0.5 * math.cos(angle) + 0.02, 0.5 * math.sin(angle) + 0.03))
    text.set_ha("left")
    text.set_va("bottom")
    return ax


FIGURES = {
    "picture-simplex": picture,
    "spherical-octant": spherical,
    "neutrosophic-radar": neutrosophic,
}

if __name__ == "__main__":
    run(FIGURES)
