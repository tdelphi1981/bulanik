"""Regenerate every committed docs figure, in the light and dark variants.

Run from the repository root:

    uv run --group docs python docs/scripts/build_figures.py

The output is deterministic (fixed SVG id salt, no timestamps), so an
unchanged figure regenerates byte for byte and does not show up in git.
"""

from __future__ import annotations

import fig_anfis_fit
import fig_example_plotting
import fig_it2_fou
import fig_mcdm_dematel
import fig_mcdm_ranking
import fig_mcdm_weights
import fig_mamdani_aggregate
import fig_membership_shapes
import fig_numbers_interval
import fig_numbers_it2
import fig_numbers_orthopair
import fig_numbers_threepart
import fig_numbers_triangular
from _figutil import run

MODULES = (
    fig_numbers_triangular,
    fig_numbers_orthopair,
    fig_numbers_threepart,
    fig_membership_shapes,
    fig_mamdani_aggregate,
    fig_it2_fou,
    fig_example_plotting,
    fig_anfis_fit,
    fig_mcdm_weights,
    fig_mcdm_ranking,
    fig_numbers_interval,
    fig_numbers_it2,
    fig_mcdm_dematel,
)

if __name__ == "__main__":
    for module in MODULES:
        run(module.FIGURES, [])
