"""Figures for the Mamdani guide: clipped consequents, aggregate and centroid; the tipper surface.

Run from the repository root: `uv run --group docs python docs/scripts/fig_mamdani_aggregate.py`.
Every curve comes from `bulanik.plot.mamdani_output` / `bulanik.plot.surface`.
"""

from __future__ import annotations

from _figutil import panels, run

from bulanik import plot
from bulanik.inference import MamdaniFIS
from bulanik.sets import LinguisticVariable, triangular


def tipper() -> MamdaniFIS:
    """The tipper of examples/10_mamdani_tipper.py."""
    service = LinguisticVariable(
        "service", (0, 10),
        {"poor": triangular(0, 0, 5), "average": triangular(0, 5, 10), "good": triangular(5, 10, 10)},
    )
    food = LinguisticVariable("food", (0, 10), {"rancid": triangular(0, 0, 5), "delicious": triangular(5, 10, 10)})
    tip = LinguisticVariable(
        "tip", (0, 30),
        {"cheap": triangular(0, 5, 10), "average": triangular(10, 15, 20), "generous": triangular(20, 25, 30)},
    )
    return MamdaniFIS(
        [service, food], [tip],
        [
            "IF service IS poor OR food IS rancid THEN tip IS cheap",
            "IF service IS average THEN tip IS average",
            "IF service IS good OR food IS delicious THEN tip IS generous",
        ],
    )


def aggregate():
    """Service 3, food 8: three clipped consequents, their aggregate and the centroid."""
    fis = tipper()
    fig, ax = panels(1, 1, figsize=(7.2, 3.6))
    plot.mamdani_output(fis, {"service": 3, "food": 8}, ax=ax)
    return fig


def surface():
    """The control surface of the tipper."""
    fig = panels(1, 1, figsize=(5.6, 4.4))[0]
    fig.clear()
    ax = fig.add_subplot(projection="3d")
    plot.surface(tipper(), resolution=31, ax=ax)
    ax.set_box_aspect((1, 1, 0.62), zoom=1.3)
    return fig


FIGURES = {"mamdani-aggregate": aggregate, "mamdani-surface": surface}

if __name__ == "__main__":
    run(FIGURES)
