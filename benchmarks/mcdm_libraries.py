#!/usr/bin/env python3
"""Benchmark 1: bulanik against pymcdm and pyrepo-mcda on identical decision matrices.

Run from the repository root (see benchmarks/README.md):

    uv run --frozen --isolated --no-dev --with pymcdm==1.4.0 --with pyrepo-mcda==0.1.15 \
        python benchmarks/mcdm_libraries.py

Writes tests/benchmarks/data/mcdm_results.json, read by the docs and the offline tests.
"""

from __future__ import annotations

import contextlib
import importlib.metadata as md
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pymcdm
from pymcdm.helpers import rankdata
from pymcdm.normalizations import vector_normalization as pymcdm_vector
from pyrepo_mcda import normalizations as repo_norms
from pyrepo_mcda import weighting_methods as repo_weights
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda.mcda_methods import AHP as RepoAHP
from pyrepo_mcda.mcda_methods import CODAS, COCOSO, COPRAS, EDAS, MARCOS, TOPSIS, VIKOR, WASPAS

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests" / "benchmarks"))

import bulanik  # noqa: E402
import bulanik.mcdm as bm  # noqa: E402
from _bench_support import average_ranks, spearman  # noqa: E402
from _mcdm_cases import FIXTURES, SHARED_REAL, bulanik_score, random_pairwise, random_problem  # noqa: E402

SEED = 20261105
N_RANDOM = 200
N_AHP = 100
N_FROZEN = 6
OUT = REPO / "tests" / "benchmarks" / "data" / "mcdm_results.json"
AGREE = 1e-9


def num_types(types):
    return np.array([1 if t == "benefit" else -1 for t in types])


def pm(method_factory):
    def call(matrix, weights, types):
        return method_factory()(np.array(matrix, float), np.array(weights, float), num_types(types))
    return call


def rp(method_factory):
    return pm(method_factory)


def repo_eigenvector(matrix):
    np.random.seed(0)  # ARPACK draws a random start vector
    return RepoAHP()._calculate_eigenvector(np.array(matrix, float))


LIB_SCORING = {
    "pymcdm": {
        "topsis": pm(pymcdm.methods.TOPSIS),
        "topsis/vector": pm(lambda: pymcdm.methods.TOPSIS(normalization_function=pymcdm_vector)),
        "vikor": pm(pymcdm.methods.VIKOR),
        "codas": pm(pymcdm.methods.CODAS),
        "waspas": pm(pymcdm.methods.WASPAS),
        "edas": pm(pymcdm.methods.EDAS),
        "marcos": pm(pymcdm.methods.MARCOS),
        "copras": pm(pymcdm.methods.COPRAS),
        "cocoso": pm(pymcdm.methods.COCOSO),
    },
    "pyrepo-mcda": {
        "topsis": rp(TOPSIS),
        "topsis/vector": rp(lambda: TOPSIS(normalization_method=repo_norms.vector_normalization)),
        "vikor": rp(VIKOR),
        "codas": rp(CODAS),
        "waspas": rp(WASPAS),
        "edas": rp(EDAS),
        "marcos": rp(MARCOS),
        "copras": rp(COPRAS),
        "cocoso": rp(COCOSO),
    },
}
LIB_WEIGHTS = {
    "pymcdm": {"entropy": pymcdm.weights.entropy_weights, "critic": pymcdm.weights.critic_weights},
    "pyrepo-mcda": {"entropy": repo_weights.entropy_weighting, "critic": repo_weights.critic_weighting},
}
LIB_AHP = {
    "pymcdm": lambda a: pymcdm.weights.subjective.AHP(matrix=np.array(a, float))(),
    "pyrepo-mcda": repo_eigenvector,
}
LOWER_IS_BETTER = {"vikor"}

# Why a comparison differs. Anything above AGREE without an entry here is reported UNCLASSIFIED.
EXPLAIN = {
    ("pymcdm", "topsis"): (
        "normalisation variant",
        "pymcdm's default is min-max; bulanik always uses vector normalisation and has no option "
        "for min-max. The topsis/vector row sets pymcdm to vector normalisation and agrees.",
    ),
    ("pyrepo-mcda", "topsis"): (
        "normalisation variant",
        "pyrepo-mcda's default is min-max; bulanik always uses vector normalisation and has no option "
        "for min-max. The topsis/vector row sets pyrepo-mcda to vector normalisation and agrees.",
    ),
    ("pymcdm", "copras"): (
        "formula form",
        "pymcdm 1.4.0 computes Q = S+ + (min(S-) * S-) / (S- * (min(S-) / S-)), which simplifies to "
        "S+ + S-: a larger cost raises the score. bulanik and pyrepo-mcda use "
        "Q = S+ + sum(S-) / (S- * sum(1/S-)). The copras/sum row rebuilds pymcdm's form from "
        "bulanik's extras (S_plus + S_minus, divided by its maximum) and agrees.",
    ),
}


def compare(a, b, lower_better=False):
    """Max absolute difference, Spearman correlation and identical ranking at the tie tolerance."""
    diff = max(abs(x - y) for x, y in zip(a, b))
    rho = spearman(a, b)
    desc = not lower_better
    same = average_ranks(a, desc) == average_ranks(b, desc)
    return diff, rho, same


class Tally:
    def __init__(self):
        self.diff, self.rho, self.same, self.n = 0.0, 1.0, 0, 0
        self.bulanik_errors, self.library_errors = [], []
        self.t_bul, self.t_lib = 0.0, 0.0

    def row(self, method, library, label, note_key=None):
        if self.diff <= AGREE:
            cls, why = "agree", "Same scores within 1e-9 on every matrix."
        else:
            cls, why = EXPLAIN.get(note_key or (library, method), ("UNCLASSIFIED", ""))
        return {
            "method": label,
            "library": library,
            "matrices": self.n,
            "max_abs_diff": self.diff,
            "min_spearman": self.rho,
            "identical_rankings": self.same,
            "bulanik_errors": self.bulanik_errors,
            "library_errors": self.library_errors,
            "seconds_bulanik": round(self.t_bul, 4),
            "seconds_library": round(self.t_lib, 4),
            "classification": cls,
            "explanation": why,
        }


def scoring_run(library, label, problems):
    method = label.split("/")[0]
    fn = LIB_SCORING[library][label]
    tally = Tally()
    for name, matrix, weights, types in problems:
        t0 = time.perf_counter()
        try:
            ours = bulanik_score(method, matrix, weights, types)
        except ValueError as exc:
            tally.bulanik_errors.append(f"{name}: {exc}")
            continue
        t1 = time.perf_counter()
        try:
            with np.errstate(all="ignore"):
                theirs = [float(v) for v in fn(matrix, weights, types)]
        except Exception as exc:  # the library's own refusal is a result
            tally.library_errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        t2 = time.perf_counter()
        tally.t_bul += t1 - t0
        tally.t_lib += t2 - t1
        if not all(math.isfinite(v) for v in theirs):
            tally.library_errors.append(f"{name}: non-finite scores")
            continue
        d, r, s = compare(ours, theirs, method in LOWER_IS_BETTER)
        tally.diff, tally.rho = max(tally.diff, d), min(tally.rho, r)
        tally.same += s
        tally.n += 1
    return tally.row(method, library, label)


def copras_sum_row(problems):
    """bulanik's extras rebuilt into pymcdm's S+ + S- form."""
    fn = LIB_SCORING["pymcdm"]["copras"]
    tally = Tally()
    for _name, matrix, weights, types in problems:
        r = bm.copras(matrix, weights, types)
        q = [a + b for a, b in zip(r.extras["S_plus"].values(), r.extras["S_minus"].values())]
        top = max(q)
        ours = [v / top for v in q]
        d, rho, s = compare(ours, [float(v) for v in fn(matrix, weights, types)])
        tally.diff, tally.rho = max(tally.diff, d), min(tally.rho, rho)
        tally.same += s
        tally.n += 1
    return tally.row("copras", "pymcdm", "copras/sum")


def weights_run(library, method, matrices, label=None, types_of=None):
    fn = LIB_WEIGHTS[library][method]
    tally = Tally()
    for name, matrix, types in matrices:
        t0 = time.perf_counter()
        if types_of is not None:
            ours = list(getattr(bm, method)(matrix, criteria_types=types).weights.values())
        else:
            ours = bulanik_score(method, matrix, None, None)
        t1 = time.perf_counter()
        theirs = [float(v) for v in fn(np.array(matrix, float))]
        t2 = time.perf_counter()
        tally.t_bul += t1 - t0
        tally.t_lib += t2 - t1
        d, r, s = compare(ours, theirs)
        tally.diff, tally.rho = max(tally.diff, d), min(tally.rho, r)
        tally.same += s
        tally.n += 1
    return tally.row(method, library, label or method, note_key=("any", label) if label else None)


EXPLAIN[("any", "critic/cost-types")] = (
    "parameter default",
    "bulanik can flip cost columns in the min-max step (criteria_types=...), which changes the sign "
    "of their correlations; both libraries ignore criterion types here. Left at its default "
    "(all benefit) bulanik agrees, see the critic rows.",
)


def ahp_run(library, matrices):
    fn = LIB_AHP[library]
    tally = Tally()
    for name, matrix in matrices:
        t0 = time.perf_counter()
        ours = list(bm.ahp(matrix).weights.values())
        t1 = time.perf_counter()
        theirs = [float(v) for v in fn(matrix)]
        t2 = time.perf_counter()
        tally.t_bul += t1 - t0
        tally.t_lib += t2 - t1
        d, r, s = compare(ours, theirs)
        tally.diff, tally.rho = max(tally.diff, d), min(tally.rho, r)
        tally.same += s
        tally.n += 1
    row = tally.row("ahp", library, "ahp", note_key=("any", "ahp"))
    return row


EXPLAIN[("any", "ahp")] = (
    "numerical tolerance",
    "All compute the principal eigenvector: bulanik by power iteration, pymcdm by numpy.linalg.eig, "
    "pyrepo-mcda by scipy eigs (a private method, _calculate_eigenvector). The gap is solver "
    "precision only.",
)


def ahp_cr(matrices):
    """Consistency ratios: bulanik against pymcdm, raw and with pymcdm's random index undone."""
    raw, aligned = 0.0, 0.0
    table = pymcdm.weights.subjective.AHP.RI_M
    ours_ri = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    for _name, matrix in matrices:
        a = bm.ahp(matrix).consistency_ratio
        b = pymcdm.weights.subjective.AHP(matrix=np.array(matrix, float)).get_cr()
        n = len(matrix)
        raw = max(raw, abs(a - b))
        aligned = max(aligned, abs(a - b * table[n - 1] / ours_ri[n]))
    return {
        "matrices": len(matrices),
        "max_abs_diff": raw,
        "max_abs_diff_same_random_index": aligned,
        "classification": "parameter default",
        "explanation": "pymcdm uses another random-index table (0.5799, 0.8921, ... ); bulanik and "
        "pyrepo-mcda use Saaty's 0.58, 0.90, ... With the same table the ratios agree.",
    }


EDGE_CASES = [
    ("tied rows", [[3, 5, 2], [3, 5, 2], [4, 1, 7], [1, 2, 9]], [0.5, 0.3, 0.2], ["benefit", "cost", "benefit"]),
    ("constant column", [[3, 5, 2], [4, 5, 7], [1, 5, 9], [2, 5, 4]], [0.5, 0.3, 0.2], ["benefit", "cost", "benefit"]),
    ("zero entry in a benefit column", [[0, 5, 2], [4, 3, 7], [1, 6, 9], [2, 4, 4]], [0.5, 0.3, 0.2], ["benefit", "cost", "benefit"]),
    ("alternative worst on every criterion", [[1, 9, 1], [4, 3, 7], [3, 6, 9], [2, 4, 4]], [0.5, 0.3, 0.2], ["benefit", "cost", "benefit"]),
    ("benefit criteria only", [[3, 5, 2], [4, 3, 7], [1, 6, 9], [2, 4, 4]], [0.5, 0.3, 0.2], ["benefit", "benefit", "benefit"]),
    ("all alternatives identical", [[2, 2, 2], [2, 2, 2], [2, 2, 2]], [0.5, 0.3, 0.2], ["benefit", "cost", "benefit"]),
]


def outcome(call):
    try:
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            v = [float(x) for x in call()]
    except Exception as exc:
        return f"raises {type(exc).__name__}: {str(exc)[:90]}"
    if not all(math.isfinite(x) for x in v):
        return "returns non-finite: " + ", ".join(f"{x:.4g}" for x in v)
    return [round(x, 6) for x in v]


def edge_rows():
    rows = []
    for case, matrix, weights, types in EDGE_CASES:
        for method in ("topsis/vector", "vikor", "codas", "waspas", "edas", "marcos", "copras", "cocoso"):
            base = method.split("/")[0]
            row = {"case": case, "method": base, "bulanik": outcome(lambda: bulanik_score(base, matrix, weights, types))}
            for lib in LIB_SCORING:
                row[lib] = outcome(lambda: LIB_SCORING[lib][method](matrix, weights, types))
            rows.append(row)
        for method in ("entropy", "critic"):
            row = {"case": case, "method": method, "bulanik": outcome(lambda: bulanik_score(method, matrix, None, None))}
            for lib in LIB_WEIGHTS:
                row[lib] = outcome(lambda: LIB_WEIGHTS[lib][method](np.array(matrix, float)))
            rows.append(row)
    return rows


def tie_handling():
    """How each library's own ranking helper treats two identical alternatives."""
    _case, matrix, weights, types = EDGE_CASES[0]
    ours = bm.waspas(matrix, weights, types)
    scores = np.array(list(ours.scores.values()))
    order = {a: i + 1 for i, a in enumerate(ours.ranking)}
    return {
        "scores": [round(float(v), 6) for v in scores],
        "bulanik_ranking_positions": [order[a] for a in ours.alternatives],
        "pymcdm_rankdata": [float(v) for v in rankdata(scores, reverse=True)],
        "pyrepo_rank_preferences": [int(v) for v in rank_preferences(scores, reverse=True)],
        "classification": "tie handling",
        "explanation": "bulanik returns a strict order in which ties keep input order; pymcdm gives tied "
        "alternatives their average rank; pyrepo-mcda gives them the same integer rank and does not skip the next one.",
    }


FIXTURE_DIFFERENCES = {
    "pymcdm-copras-1": ("formula form", "pymcdm's test expects its S+ + S- form; see the copras row."),
    "pymcdm-copras-2": ("formula form", "pymcdm's test expects its S+ + S- form; see the copras row."),
    "pymcdm-topsis": ("normalisation variant", "pymcdm's test uses its min-max default; bulanik uses vector normalisation."),
    "pymcdm-waspas": (
        "parameter default",
        "The fixture's weights sum to 1.0155 and pymcdm's test turns validation off. bulanik rejects "
        "them; with normalize_weights=True the scores shrink by up to 0.0059 and the ranking stays the same.",
    ),
}


def fixture_rows():
    rows = []
    for f in FIXTURES:
        kw = dict(f["params"])
        refused = None
        try:
            ours = bulanik_score(f["method"], f["matrix"], f["weights"], f["types"], **kw)
        except ValueError as exc:
            refused = str(exc)
            ours = bulanik_score(f["method"], f["matrix"], f["weights"], f["types"], normalize_weights=True, **kw)
        diff = max(abs(a - b) for a, b in zip(ours, f["expected"]))
        tol = 0.5 * 10 ** -f["decimals"] + 1e-9
        cls, why = FIXTURE_DIFFERENCES.get(f["id"], ("", ""))
        if diff > tol and not cls:
            cls = "UNCLASSIFIED"
        rows.append({
            "id": f["id"], "library": f["library"], "test": f["test"], "method": f["method"], "study": f["study"],
            "decimals": f["decimals"], "max_abs_diff": diff, "within_rounding": diff <= tol,
            "weights_refused_without_normalize": refused, "classification": cls, "explanation": why,
        })
    return rows


def main() -> None:
    rng = random.Random(SEED)
    problems = [(n, m, w, t) for n, m, w, t in SHARED_REAL]
    rand = []
    for i in range(N_RANDOM):
        m, w, t = random_problem(rng)
        rand.append((f"random-{i}", m, w, t))
    problems += rand
    rows = []
    for lib in LIB_SCORING:
        for label in LIB_SCORING[lib]:
            rows.append(scoring_run(lib, label, problems))
    rows.append(copras_sum_row(problems))
    mats = [(n, m, t) for n, m, _w, t in problems]
    for lib in LIB_WEIGHTS:
        rows.append(weights_run(lib, "entropy", mats))
        rows.append(weights_run(lib, "critic", mats))
        rows.append(weights_run(lib, "critic", mats, label="critic/cost-types", types_of=True))
    pair_rng = random.Random(SEED + 1)
    pairs = [(f"pairwise-{i}", random_pairwise(pair_rng, pair_rng.randint(3, 10))) for i in range(N_AHP)]
    pairs.append(("wang-chin-study", next(f["matrix"] for f in FIXTURES if f["id"] == "pymcdm-ahp")))
    for lib in LIB_AHP:
        rows.append(ahp_run(lib, pairs))

    frozen = {}
    subset = problems[: len(SHARED_REAL) + N_FROZEN]
    for lib, methods in LIB_SCORING.items():
        frozen[lib] = {}
        for label, fn in methods.items():
            with np.errstate(all="ignore"):
                frozen[lib][label] = [[float(v) for v in fn(m, w, t)] for _n, m, w, t in subset]
        for name, fn in LIB_WEIGHTS[lib].items():
            frozen[lib][name] = [[float(v) for v in fn(np.array(m, float))] for _n, m, _w, _t in subset]
        frozen[lib]["ahp"] = [[float(v) for v in LIB_AHP[lib](a)] for _n, a in pairs[:N_FROZEN]]

    data = {
        "produced_by": {
            "script": "benchmarks/mcdm_libraries.py",
            "bulanik": bulanik.__version__,
            "pymcdm": md.version("pymcdm"),
            "pyrepo-mcda": md.version("pyrepo-mcda"),
            "numpy": md.version("numpy"),
            "scipy": md.version("scipy"),
            "python": sys.version.split()[0],
        },
        "setup": {
            "seed": SEED,
            "random_matrices": N_RANDOM,
            "study_matrices": len(SHARED_REAL),
            "ahp_matrices": len(pairs),
            "tie_tolerance": 1e-9,
            "agree_threshold": AGREE,
        },
        "comparison": rows,
        "ahp_consistency_ratio": ahp_cr(pairs),
        "tie_handling": tie_handling(),
        "edge_cases": edge_rows(),
        "fixtures": fixture_rows(),
        "frozen": {
            "n_random": N_FROZEN,
            "seed": SEED,
            "problems": [{"name": n, "matrix": m, "weights": w, "types": t} for n, m, w, t in subset],
            "pairwise": [{"name": n, "matrix": a} for n, a in pairs[:N_FROZEN]],
            "scores": frozen,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['method']:18} {r['library']:12} n={r['matrices']:3} diff={r['max_abs_diff']:.2e} "
              f"rho={r['min_spearman']:.4f} same={r['identical_rankings']:3} err={len(r['bulanik_errors'])}/"
              f"{len(r['library_errors'])} t={r['seconds_bulanik']:.3f}/{r['seconds_library']:.3f} {r['classification']}")


if __name__ == "__main__":
    main()
