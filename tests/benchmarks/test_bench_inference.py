"""Mamdani and Sugeno over a grid against values frozen from simpful and scikit-fuzzy primitives."""

import json
from pathlib import Path

import pytest
from _inference_cases import CLIMATE, MIXED, SUGENO_OUTPUTS, bulanik_mamdani, bulanik_sugeno, grid

RESULT = json.loads((Path(__file__).parent / "data" / "inference_results.json").read_text(encoding="utf-8"))
FROZEN = RESULT["frozen"]
POINTS = grid(11)
BASES = {"climate": CLIMATE, "mixed": MIXED}


def test_provenance_is_recorded():
    who = RESULT["produced_by"]
    assert who["simpful"] == "2.12.0" and who["scikit-fuzzy"] == "0.5.0"


@pytest.mark.parametrize("name", ["climate", "mixed"])
def test_mamdani_against_frozen_values(name):
    fis = bulanik_mamdani(BASES[name], resolution=1001)
    ours = [fis.evaluate({"temp": t, "hum": h})["fan"] for t, h in POINTS]
    block = FROZEN["mamdani"][name]
    assert ours == pytest.approx(block["simpful"], abs=1e-9)
    tol = block["bulanik_vs_scikit_tolerance"]
    assert max(abs(a - b) for a, b in zip(ours, block["scikit_fuzzy_primitives"])) <= tol


@pytest.mark.parametrize("key", ["climate/min", "climate/prod"])
def test_sugeno_against_frozen_values(key):
    name, and_method = key.split("/")
    fis = bulanik_sugeno(BASES[name], and_method=and_method)
    ours = [fis.evaluate({"temp": t, "hum": h})["fan"] for t, h in POINTS]
    assert ours == pytest.approx(FROZEN["sugeno"][key], abs=1e-9)


def test_weighted_sugeno_differs_from_simpful_by_its_denominator():
    """simpful divides by the unweighted firing strengths; the rest of the formula is the same."""
    fis = bulanik_sugeno(MIXED)
    rebuilt = []
    ours = []
    for t, h in POINTS:
        e = fis.explain({"temp": t, "hum": h})
        strengths = [r.strength for r in e.activations]
        raw = sum(a / w for a, (_x, _c, w) in zip(strengths, MIXED))
        num = 0.0
        for a, (_x, c, _w) in zip(strengths, MIXED):
            const, coef = SUGENO_OUTPUTS[c]
            num += a * (const + coef.get("temp", 0.0) * t + coef.get("hum", 0.0) * h)
        rebuilt.append(num / raw)
        ours.append(e.outputs["fan"])
    frozen = FROZEN["sugeno"]["mixed/min"]
    assert rebuilt == pytest.approx(frozen, abs=1e-9)
    assert max(abs(a - b) for a, b in zip(ours, frozen)) > 1.0
