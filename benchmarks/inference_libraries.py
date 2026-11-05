#!/usr/bin/env python3
"""Benchmark 4: Mamdani and Sugeno against simpful and scikit-fuzzy, over an input grid.

Run from the repository root:

    uv run --frozen --isolated --no-dev --with simpful==2.12.0 --with scikit-fuzzy==0.5.0 \
        --with pymcdm==1.4.0 python benchmarks/inference_libraries.py

pymcdm is here only because scikit-fuzzy 0.5.0 declares no dependencies yet imports `packaging`,
which pymcdm brings in through matplotlib.

scikit-fuzzy's rule engine (skfuzzy.control) imports networkx, which the package does not declare
and this suite does not install. So the scikit-fuzzy column uses the library's membership and
defuzzification functions, with the firing strengths and the clip-and-aggregate step written
below. That compares its primitives, not its control engine. Writes
tests/benchmarks/data/inference_results.json.
"""

from __future__ import annotations

import contextlib
import importlib.metadata as md
import importlib.util
import io
import json
import pkgutil
import sys
import time
from pathlib import Path

import numpy as np
import simpful as sf
import skfuzzy as fuzz

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests" / "benchmarks"))

import bulanik  # noqa: E402
from _inference_cases import (  # noqa: E402
    CLIMATE, GAP, MIXED, SUGENO_OUTPUTS, TERMS, UNIVERSE, bulanik_mamdani, bulanik_sugeno, grid,
)
from bulanik.inference import NoRuleFiredError  # noqa: E402

OUT = REPO / "tests" / "benchmarks" / "data" / "inference_results.json"


def sf_mf(spec):
    kind, *p = spec
    return {"tri": sf.Triangular_MF, "trap": sf.Trapezoidal_MF, "gauss": sf.Gaussian_MF}[kind](*p)


def sf_var(name):
    sets = [sf.FuzzySet(function=sf_mf(spec), term=term) for term, spec in TERMS[name].items()]
    return sf.LinguisticVariable(sets, concept=name, universe_of_discourse=list(UNIVERSE[name]))


def sf_ant(node):
    op = node[0]
    if op == "is":
        return f"({node[1]} IS {node[2]})"
    if op == "not":
        return f"(NOT{sf_ant(node[1])})"
    return f"({sf_ant(node[1])} {op.upper()} {sf_ant(node[2])})"


def sf_rules(rules):
    return [
        f"IF {sf_ant(a)} THEN (fan IS {c})" + (f" WEIGHT {w}" if w != 1.0 else "") for a, c, w in rules
    ]


def simpful_system(rules, and_method, sugeno):
    ops = ["AND_PRODUCT"] if and_method == "prod" else None
    with contextlib.redirect_stdout(io.StringIO()):
        fs = sf.FuzzySystem(operators=ops, show_banner=False, verbose=False)
        fs.add_linguistic_variable("temp", sf_var("temp"))
        fs.add_linguistic_variable("hum", sf_var("hum"))
        if sugeno:
            for name, (c, coef) in SUGENO_OUTPUTS.items():
                if coef:
                    fs.set_output_function(name, f"{coef['temp']}*temp+{coef['hum']}*hum+{c}")
                else:
                    fs.set_crisp_output_value(name, c)
        else:
            fs.add_linguistic_variable("fan", sf_var("fan"))
        fs.add_rules(sf_rules(rules))
    return fs


def simpful_eval(fs, t, h, sugeno, subdivisions):
    with contextlib.redirect_stdout(io.StringIO()):
        fs.set_variable("temp", t)
        fs.set_variable("hum", h)
        if sugeno:
            return float(fs.Sugeno_inference(["fan"])["fan"])
        return float(fs.Mamdani_inference(["fan"], subdivisions=subdivisions)["fan"])


def sk_mf(spec, x):
    kind, *p = spec
    x = np.atleast_1d(np.asarray(x, float))
    if kind == "tri":
        return fuzz.trimf(x, p)
    if kind == "trap":
        return fuzz.trapmf(x, p)
    return fuzz.gaussmf(x, *p)


def sk_alpha(node, values, and_method):
    op = node[0]
    if op == "is":
        return float(sk_mf(TERMS[node[1]][node[2]], values[node[1]])[0])
    if op == "not":
        return 1.0 - sk_alpha(node[1], values, and_method)
    a, b = sk_alpha(node[1], values, and_method), sk_alpha(node[2], values, and_method)
    if op == "or":
        return max(a, b)
    return min(a, b) if and_method == "min" else a * b


class SkMamdani:
    def __init__(self, rules, resolution, and_method, mode):
        self.rules, self.and_method, self.mode = rules, and_method, mode
        self.x = np.linspace(*UNIVERSE["fan"], resolution)
        self.mfs = {t: sk_mf(s, self.x) for t, s in TERMS["fan"].items()}

    def __call__(self, t, h):
        agg = np.zeros_like(self.x)
        values = {"temp": t, "hum": h}
        for ant, cons, weight in self.rules:
            alpha = sk_alpha(ant, values, self.and_method) * weight
            agg = np.fmax(agg, np.fmin(alpha, self.mfs[cons]))
        return float(fuzz.defuzz(self.x, agg, self.mode))


def stats(a, b):
    d = [abs(x - y) for x, y in zip(a, b)]
    return {"max_abs_diff": max(d), "mean_abs_diff": sum(d) / len(d), "points": len(d)}


def timed(fn, points):
    t0 = time.perf_counter()
    values = [fn(t, h) for t, h in points]
    return values, time.perf_counter() - t0


def mamdani_block(label, rules, and_method, resolution, points):
    fis = bulanik_mamdani(rules, resolution=resolution, and_method=and_method)
    ours, t_b = timed(lambda t, h: fis.evaluate({"temp": t, "hum": h})["fan"], points)
    fs = simpful_system(rules, and_method, False)
    sim, t_s = timed(lambda t, h: simpful_eval(fs, t, h, False, resolution), points)
    sk, t_k = timed(SkMamdani(rules, resolution, and_method, "centroid"), points)
    return {
        "rule_base": label, "and": and_method, "resolution": resolution,
        "vs_simpful": stats(ours, sim), "vs_scikit_fuzzy_primitives": stats(ours, sk),
        "seconds": {"bulanik": round(t_b, 3), "simpful": round(t_s, 3), "scikit_fuzzy_primitives": round(t_k, 3)},
    }, ours, sim, sk


def rectangle_rule_check(rules, resolution, points):
    """The sum-based centroid applied to scikit-fuzzy's aggregate set; should equal bulanik's."""
    sk = SkMamdani(rules, resolution, "min", "centroid")
    fis = bulanik_mamdani(rules, resolution=resolution)
    worst = 0.0
    for t, h in points:
        agg = np.zeros_like(sk.x)
        values = {"temp": t, "hum": h}
        for ant, cons, weight in rules:
            agg = np.fmax(agg, np.fmin(sk_alpha(ant, values, "min") * weight, sk.mfs[cons]))
        rect = float((sk.x * agg).sum() / agg.sum())
        worst = max(worst, abs(rect - fis.evaluate(values)["fan"]))
    return worst


def gap_behaviour():
    """What each library does when no rule fires (temp = 35: GAP's only rule has membership 0)."""
    out = {}
    fis = bulanik_mamdani(GAP, resolution=101)
    try:
        out["bulanik"] = fis.evaluate({"temp": 35.0, "hum": 50.0})["fan"]
    except NoRuleFiredError as exc:
        out["bulanik"] = f"raises NoRuleFiredError: {str(exc)[:70]}"
    fs = simpful_system(GAP, "min", False)
    out["simpful"] = f"returns {simpful_eval(fs, 35.0, 50.0, False, 101)} (prints a warning)"
    try:
        out["scikit_fuzzy_defuzz"] = f"returns {SkMamdani(GAP, 101, 'min', 'centroid')(35.0, 50.0)}"
    except Exception as exc:
        out["scikit_fuzzy_defuzz"] = f"raises {type(exc).__name__}: {str(exc)[:70]}"
    return out


def main() -> None:
    full = grid(21)
    sparse = grid(5)
    subset = grid(11)
    blocks, frozen = [], {}
    plan = [
        ("climate", CLIMATE, "min", 101, full),
        ("climate", CLIMATE, "min", 1001, full),
        ("climate", CLIMATE, "min", 10001, sparse),
        ("climate", CLIMATE, "prod", 1001, full),
        ("mixed", MIXED, "min", 1001, full),
    ]
    for label, rules, and_method, res, points in plan:
        block, ours, sim, sk = mamdani_block(label, rules, and_method, res, points)
        blocks.append(block)
        print(block, flush=True)
        if res == 1001 and points is full and and_method == "min":
            keep = [i for i, p in enumerate(full) if p in set(subset)]
            frozen[label] = {
                "simpful": [sim[i] for i in keep], "scikit_fuzzy_primitives": [sk[i] for i in keep],
                "bulanik_vs_scikit_tolerance": max(abs(ours[i] - sk[i]) for i in keep) * 1.01,
            }

    defuzz = []
    for mode in ("bisector", "mom", "som", "lom"):
        fis = bulanik_mamdani(CLIMATE, resolution=1001, defuzzification=mode)
        ours, _ = timed(lambda t, h: fis.evaluate({"temp": t, "hum": h})["fan"], full)
        sk, _ = timed(SkMamdani(CLIMATE, 1001, "min", mode), full)
        defuzz.append({"method": mode, **stats(ours, sk)})
        print(defuzz[-1], flush=True)

    sugeno = []
    frozen_sugeno = {}
    for label, rules, and_method in (("climate", CLIMATE, "min"), ("climate", CLIMATE, "prod"), ("mixed", MIXED, "min")):
        fis = bulanik_sugeno(rules, and_method=and_method)
        ours, t_b = timed(lambda t, h: fis.evaluate({"temp": t, "hum": h})["fan"], full)
        fs = simpful_system(rules, and_method, True)
        sim, t_s = timed(lambda t, h: simpful_eval(fs, t, h, True, 0), full)
        sugeno.append({"rule_base": label, "and": and_method, "vs_simpful": stats(ours, sim),
                       "seconds": {"bulanik": round(t_b, 4), "simpful": round(t_s, 4)}})
        print(sugeno[-1], flush=True)
        keep = [i for i, p in enumerate(full) if p in set(subset)]
        frozen_sugeno[f"{label}/{and_method}"] = [sim[i] for i in keep]

    plain_mixed = [(a, c, 1.0) for a, c, _w in MIXED]
    fis = bulanik_sugeno(plain_mixed)
    ours, t_b = timed(lambda t, h: fis.evaluate({"temp": t, "hum": h})["fan"], full)
    fs = simpful_system(plain_mixed, "min", True)
    sim, t_s = timed(lambda t, h: simpful_eval(fs, t, h, True, 0), full)
    sugeno.append({"rule_base": "mixed, weights removed", "and": "min", "vs_simpful": stats(ours, sim),
                   "seconds": {"bulanik": round(t_b, 4), "simpful": round(t_s, 4)}})
    print(sugeno[-1], flush=True)

    rebuilt = 0.0
    fis = bulanik_sugeno(MIXED)
    fs = simpful_system(MIXED, "min", True)
    for t, h in full:
        strengths = [r.strength for r in fis.explain({"temp": t, "hum": h}).activations]
        raw = sum(a / w for a, (_x, _c, w) in zip(strengths, MIXED))
        num = 0.0
        for a, (_x, c, _w) in zip(strengths, MIXED):
            const, coef = SUGENO_OUTPUTS[c]
            num += a * (const + coef.get("temp", 0.0) * t + coef.get("hum", 0.0) * h)
        rebuilt = max(rebuilt, abs(num / raw - simpful_eval(fs, t, h, True, 0)))

    sk_files = [m.name for m in pkgutil.walk_packages(fuzz.__path__, "skfuzzy.")]
    root = Path(fuzz.__file__).parent
    mentions = sum("sugeno" in p.read_text(encoding="utf-8", errors="ignore").lower() for p in root.rglob("*.py"))
    data = {
        "produced_by": {
            "script": "benchmarks/inference_libraries.py", "bulanik": bulanik.__version__,
            "simpful": md.version("simpful"), "scikit-fuzzy": md.version("scikit-fuzzy"),
            "numpy": md.version("numpy"), "python": sys.version.split()[0],
        },
        "grid": {"full": "21 x 21 temp, hum", "res_10001": "5 x 5, to keep the run short", "frozen": "11 x 11"},
        "scikit_fuzzy_scope": {
            "control_engine_importable_without_networkx": importlib.util.find_spec("networkx") is not None,
            "python_files_mentioning_sugeno": mentions,
            "modules_seen": len(sk_files),
            "note": "skfuzzy.control imports networkx, which scikit-fuzzy 0.5.0 does not declare and "
            "this suite does not install. The column uses skfuzzy membership functions and defuzz, "
            "with firing and aggregation written in this script. No file in the package mentions Sugeno.",
        },
        "classification": {
            "mamdani_vs_simpful": "Same scores within rounding: same operators, same sampled centroid.",
            "mamdani_vs_scikit_fuzzy": {
                "label": "formula form (quadrature)",
                "explanation": "bulanik and simpful take sum(x mu) / sum(mu) over the samples, a rectangle rule that "
                "counts the universe endpoints in full; skfuzzy integrates the piecewise-linear set exactly. "
                "The gap shrinks about tenfold for each tenfold rise in resolution.",
                "rectangle_rule_on_scikit_aggregate_vs_bulanik": rectangle_rule_check(CLIMATE, 1001, full),
            },
            "bisector_vs_scikit_fuzzy": "formula form (quadrature): bulanik returns the first sample at which the running "
            "sum reaches half; skfuzzy solves for the exact area split. On a low flat stretch the endpoint weight moves "
            "the split by more than one grid step.",
            "sugeno_vs_simpful_with_weights": {
                "label": "formula form",
                "explanation": "With rule weights, bulanik computes sum(w a f) / sum(w a), simpful sum(w a f) / sum(a): simpful "
                "weights the numerator only. Without weights they agree (the row 'mixed, weights removed'). bulanik "
                "has no option for simpful's form; rebuilding it from bulanik's rule activations reproduces simpful.",
                "rebuilt_simpful_form_vs_simpful": rebuilt,
            },
        },
        "mamdani": blocks,
        "defuzzification_vs_scikit_fuzzy": defuzz,
        "sugeno": sugeno,
        "no_rule_fired": gap_behaviour(),
        "frozen": {"mamdani": frozen, "sugeno": frozen_sugeno},
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(data["no_rule_fired"], data["scikit_fuzzy_scope"])


if __name__ == "__main__":
    main()
