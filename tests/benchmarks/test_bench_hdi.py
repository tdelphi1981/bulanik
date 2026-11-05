"""WASPAS on the committed HDI extract against UNDP's published rank (frozen in hdi_results.json)."""

import csv
import json
from pathlib import Path

import pytest
from _hdi_analysis import analyse

HERE = Path(__file__).parent / "data"
FROZEN = json.loads((HERE / "hdi_results.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def result():
    with (HERE / "hdi_2023_extract.csv").open(encoding="utf-8", newline="") as fh:
        rows = [
            {"iso3": x["iso3"], "rank": int(x["hdi_rank_2023"]), "hdi": float(x["hdi_2023"]),
             "health": float(x["health_index"]), "education": float(x["education_index"]),
             "income": float(x["income_index"])}
            for x in csv.DictReader(fh)
        ]
    return analyse(rows)


def test_matches_frozen_results(result):
    for form in ("product", "sum", "blend_0.5"):
        assert result[form]["spearman_with_undp_rank"] == pytest.approx(
            FROZEN["results"][form]["spearman_with_undp_rank"], abs=1e-9
        )
        assert result[form]["top5"] == FROZEN["results"][form]["top5"]


def test_product_form_reproduces_undp(result):
    product = result["product"]
    assert product["spearman_with_undp_rank"] > 0.9999
    assert product["same_rank_after_rounding_to_3_decimals"] == FROZEN["countries_used"] == 193
    assert product["max_abs_diff_rescaled_score_vs_geometric_mean"] < 1e-12


def test_goalposts_reproduce_published_hdi(result):
    assert result["recomputed_hdi"]["agree_at_3_decimals"] == 193
