"""Publication-quality plots of fuzzy numbers, linguistic variables, inference and MCDM results.

Needs the `plot` extra (`pip install "bulanik[plot]"`, which installs
matplotlib); the rest of bulanik stays dependency-free.

Every function takes keyword-only options and an optional `ax`; without one it
creates a new figure. It returns the matplotlib `Axes` and never calls
`plt.show()`, so you can add annotations, save or show the figure yourself.
Inputs are validated with the library's usual errors (`TypeError` for a wrong
type, `ValueError` for a bad value).

- Fuzzy numbers: [`membership`][bulanik.plot.membership],
  [`alpha_cuts`][bulanik.plot.alpha_cuts],
  [`defuzzification`][bulanik.plot.defuzzification],
  [`orthopair`][bulanik.plot.orthopair],
  [`iv_orthopair`][bulanik.plot.iv_orthopair], [`picture`][bulanik.plot.picture],
  [`spherical`][bulanik.plot.spherical],
  [`neutrosophic`][bulanik.plot.neutrosophic].
- Sets and scales: [`variable`][bulanik.plot.variable],
  [`it2_set`][bulanik.plot.it2_set], [`it2_number`][bulanik.plot.it2_number].
- Inference: [`mamdani_output`][bulanik.plot.mamdani_output],
  [`surface`][bulanik.plot.surface].
- MCDM: [`ranking`][bulanik.plot.ranking], [`weights`][bulanik.plot.weights],
  [`dematel`][bulanik.plot.dematel].

House style: one validated categorical palette ([`PALETTE`][bulanik.plot.PALETTE])
in fixed slot order, 2 px lines with soft translucent fills, hairline axes and
grid, direct labels instead of legends. Wrap plotting and saving in
[`theme`][bulanik.plot.theme] to get the light or dark variant; outside a
theme the light variant's colours are used.

Examples:
    >>> from bulanik import TFN, plot
    >>> with plot.theme("light"):
    ...     ax = plot.membership([TFN(1, 2, 4), TFN(2, 4, 5)], labels=["A", "B"])
    >>> ax.get_xlabel()
    '$x$'
"""

try:
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise ImportError(
        'bulanik.plot needs matplotlib; install the plot extra: pip install "bulanik[plot]"'
    ) from exc

from ._style import PALETTE, current_theme, theme
from .inference import mamdani_output, surface
from .mcdm import dematel, ranking, weights
from .numbers import (
    alpha_cuts,
    defuzzification,
    it2_number,
    iv_orthopair,
    membership,
    neutrosophic,
    orthopair,
    picture,
    spherical,
)
from .sets import it2_set, variable

__all__ = [
    "PALETTE",
    "alpha_cuts",
    "current_theme",
    "defuzzification",
    "dematel",
    "it2_number",
    "it2_set",
    "iv_orthopair",
    "mamdani_output",
    "membership",
    "neutrosophic",
    "orthopair",
    "picture",
    "ranking",
    "spherical",
    "surface",
    "theme",
    "variable",
    "weights",
]
