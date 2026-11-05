#!/usr/bin/env python3
"""Benchmark 2: WASPAS on the Human Development Index dimensions against UNDP's rank.

The HDI is the geometric mean of three dimension indices. WASPAS in product form (lam=0)
with equal weights is that same geometric mean divided by a constant, because it first divides
each column by its maximum. So its order should match the HDI order almost exactly; this
checks the implementation against a published ranking and is not evidence that
WASPAS finds anything new.

Run from the repository root:

    uv run --frozen --isolated --no-dev python benchmarks/hdi_undp.py

Writes tests/benchmarks/data/hdi_2023_extract.csv and hdi_results.json.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests" / "benchmarks"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bulanik  # noqa: E402
from _hdi_analysis import analyse  # noqa: E402
from _download import fetch  # noqa: E402

URL = "https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv"
SHA256 = "61ed82e5b66c88dfca8ff9fac775c63981ecab6a254862af97acacc41c143117"
YEAR = 2023
DATA = REPO / "tests" / "benchmarks" / "data"


def health_index(le: float) -> float:
    return (min(le, 85.0) - 20.0) / (85.0 - 20.0)


def education_index(eys: float, mys: float) -> float:
    return (min(eys, 18.0) / 18.0 + min(mys, 15.0) / 15.0) / 2.0


def income_index(gni: float) -> float:
    capped = min(max(gni, 100.0), 75000.0)
    return (math.log(capped) - math.log(100.0)) / (math.log(75000.0) - math.log(100.0))


def main() -> None:
    path = fetch(URL, "HDR25_Composite_indices_complete_time_series.csv", SHA256)
    with path.open(encoding="latin-1", newline="") as fh:
        source = list(csv.DictReader(fh))
    rows, dropped = [], []
    for s in source:
        if s["iso3"].startswith("ZZ"):
            continue  # regional and world aggregates
        fields = [s[f"{k}_{YEAR}"] for k in ("hdi", "le", "eys", "mys", "gnipc")] + [s[f"hdi_rank_{YEAR}"]]
        if any(v == "" for v in fields):
            dropped.append(s["country"])
            continue
        hdi, le, eys, mys, gni = (float(v) for v in fields[:5])
        rows.append({
            "iso3": s["iso3"], "country": s["country"], "rank": int(float(fields[5])), "hdi": hdi,
            "le": le, "eys": eys, "mys": mys, "gnipc": gni, "health": health_index(le),
            "education": education_index(eys, mys), "income": income_index(gni),
        })
    rows.sort(key=lambda r: (r["rank"], r["iso3"]))
    extract = DATA / f"hdi_{YEAR}_extract.csv"
    DATA.mkdir(parents=True, exist_ok=True)
    with extract.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["iso3", "country", f"hdi_rank_{YEAR}", f"hdi_{YEAR}", f"le_{YEAR}", f"eys_{YEAR}",
                    f"mys_{YEAR}", f"gnipc_{YEAR}", "health_index", "education_index", "income_index"])
        for r in rows:
            w.writerow([r["iso3"], r["country"], r["rank"], r["hdi"], r["le"], r["eys"], r["mys"], r["gnipc"],
                        f"{r['health']:.10f}", f"{r['education']:.10f}", f"{r['income']:.10f}"])
    # re-read so the frozen numbers come from the committed extract at its stored precision
    with extract.open(encoding="utf-8", newline="") as fh:
        stored = [
            {"iso3": x["iso3"], "rank": int(x[f"hdi_rank_{YEAR}"]), "hdi": float(x[f"hdi_{YEAR}"]),
             "health": float(x["health_index"]), "education": float(x["education_index"]),
             "income": float(x["income_index"])}
            for x in csv.DictReader(fh)
        ]
    result = {
        "produced_by": {"script": "benchmarks/hdi_undp.py", "bulanik": bulanik.__version__},
        "source": {"url": URL, "sha256": SHA256, "year": YEAR, "licence": "CC BY 3.0 IGO"},
        "countries_used": len(stored),
        "dropped_no_data": dropped,
        "tied_undp_ranks": len(stored) - len({r["rank"] for r in stored}),
        "results": analyse(stored),
    }
    (DATA / "hdi_results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
