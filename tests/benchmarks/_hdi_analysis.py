"""The WASPAS-on-HDI analysis, shared by benchmarks/hdi_undp.py and the offline test."""

from __future__ import annotations

from _bench_support import average_ranks, pearson
from bulanik.mcdm import waspas


def competition_ranks(values: list[float]) -> list[int]:
    """Largest value first, equal values share the first of their positions (1, 2, 2, 4, ...)."""
    return [1 + sum(other > v for other in values) for v in values]


def analyse(rows: list[dict]) -> dict:
    """Rank rows with keys iso3, rank, hdi, health, education, income; compare with UNDP's rank."""
    matrix = [[r["health"], r["education"], r["income"]] for r in rows]
    weights = [1 / 3, 1 / 3, 1 / 3]
    names = [r["iso3"] for r in rows]
    undp = [float(r["rank"]) for r in rows]
    geo = [(h * e * i) ** (1 / 3) for h, e, i in matrix]
    out: dict = {}
    for label, lam in (("product", 0.0), ("sum", 1.0), ("blend_0.5", 0.5)):
        res = waspas(matrix, weights, ["benefit"] * 3, alternatives=names, lam=lam, normalize_weights=True)
        scores = list(res.scores.values())
        ours = average_ranks(scores, True)
        scale = geo[0] / scores[0]
        rebuilt = [s * scale for s in scores]
        rounded = competition_ranks([round(v, 3) for v in rebuilt])
        out[label] = {
            "lam": lam,
            "spearman_with_undp_rank": pearson(ours, average_ranks(undp, False, 0.0)),
            "same_rank_unrounded": sum(a == b for a, b in zip(ours, undp)),
            "max_rank_difference_unrounded": max(abs(a - b) for a, b in zip(ours, undp)),
            "same_rank_after_rounding_to_3_decimals": sum(a == b for a, b in zip(rounded, undp)),
            "top5": list(res.ranking[:5]),
            "bottom3": list(res.ranking[-3:]),
            "max_abs_diff_rescaled_score_vs_geometric_mean": max(abs(a - b) for a, b in zip(rebuilt, geo)),
        }
    published = [r["hdi"] for r in rows]
    out["recomputed_hdi"] = {
        "max_abs_diff_to_published": max(abs(a - b) for a, b in zip(geo, published)),
        "agree_at_3_decimals": sum(round(a, 3) == b for a, b in zip(geo, published)),
        "countries": len(rows),
    }
    return out
