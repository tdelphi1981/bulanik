"""Figures for the ranking guide: four methods on one matrix, and a SWARA -> TOPSIS pipeline.

Run from the repository root: `uv run --group docs python docs/scripts/fig_mcdm_ranking.py`.
Every dot and bar is drawn by `bulanik.plot.ranking` / `bulanik.plot.weights`.
"""

from __future__ import annotations

from _figutil import panels, run

from bulanik import plot
from bulanik.mcdm import DecisionMatrix, edas, rank_fuzzy, swara, topsis, vikor, waspas
from bulanik.scales import LinguisticScale
from bulanik import TFN
from bulanik.aggregate import aggregate_experts

CRITERIA = ["price", "quality", "delivery"]
TYPES = ["cost", "benefit", "benefit"]
SUPPLIERS = ["Atlas", "Borealis", "Cedar", "Delta"]
DATA = [[120, 8, 7], [100, 7, 5], [110, 9, 6], [130, 6, 9]]


def methods():
    """The same matrix and weights through TOPSIS, VIKOR, WASPAS and EDAS."""
    w = swara([0.2, 0.5], CRITERIA)
    results = [f(DATA, w, TYPES, alternatives=SUPPLIERS) for f in (topsis, vikor, waspas, edas)]
    fig, axes = panels(2, 2, figsize=(9.0, 4.6))
    for ax, result in zip(axes.flat, results):
        plot.ranking(result, ax=ax)
    return fig


IMPORTANCE = LinguisticScale("importance", {
    "slightly": TFN(0.1, 0.2, 0.3), "clearly": TFN(0.3, 0.4, 0.5), "much": TFN(0.5, 0.6, 0.7)})
RATING = LinguisticScale("rating", {
    "VL": TFN(0, 0, 3), "L": TFN(1, 3, 5), "M": TFN(3, 5, 7), "H": TFN(5, 7, 9), "VH": TFN(7, 10, 10)})
EXPERTS = [0.4, 0.35, 0.25]
WORDS = [[["slightly", "clearly"]], [["clearly", "much"]], [["slightly", "much"]]]
TERMS = [
    [["H", "M", "H"], ["VH", "H", "L"], ["M", "VH", "M"], ["L", "L", "VH"]],
    [["M", "H", "M"], ["H", "M", "L"], ["H", "H", "M"], ["L", "VL", "H"]],
    [["H", "M", "VH"], ["VH", "H", "M"], ["M", "VH", "L"], ["VL", "L", "VH"]],
]


def pipeline():
    """Expert words -> SWARA weights (left) -> TOPSIS ranking of the suppliers (right)."""
    importance = aggregate_experts(IMPORTANCE.convert(WORDS), EXPERTS)[0]
    weights = swara(importance, CRITERIA)
    dm = DecisionMatrix.from_terms(RATING, SUPPLIERS, CRITERIA, TERMS, EXPERTS)
    ranking = rank_fuzzy(dm, weights, TYPES)
    fig, axes = panels(1, 2, figsize=(9.0, 2.6))
    plot.weights(weights, ax=axes[0])
    plot.ranking(ranking, ax=axes[1])
    return fig


FIGURES = {"mcdm-ranking-methods": methods, "mcdm-pipeline": pipeline}

if __name__ == "__main__":
    run(FIGURES)
