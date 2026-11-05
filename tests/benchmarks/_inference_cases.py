"""Rule bases written for this benchmark, in a neutral form, and their bulanik build.

Standard library plus bulanik only. Antecedents are nested tuples: ``("is", var, term)``,
``("and", a, b)``, ``("or", a, b)``, ``("not", ("is", var, term))``.
"""

from __future__ import annotations

from bulanik.inference import Linear, MamdaniFIS, SugenoFIS, SugenoOutput
from bulanik.sets import LinguisticVariable, gaussian, trapezoidal, triangular

UNIVERSE = {"temp": (0.0, 40.0), "hum": (0.0, 100.0), "fan": (0.0, 100.0)}
TERMS = {
    "temp": {"cold": ("trap", 0, 0, 10, 18), "mild": ("tri", 12, 20, 28), "hot": ("trap", 22, 30, 40, 40)},
    "hum": {"dry": ("gauss", 0, 20), "normal": ("gauss", 50, 15), "humid": ("gauss", 100, 20)},
    "fan": {
        "off": ("tri", 0, 0, 20), "low": ("tri", 10, 30, 50), "mid": ("tri", 35, 50, 65),
        "high": ("tri", 50, 70, 90), "full": ("tri", 80, 100, 100),
    },
}


def _is(var, term):
    return ("is", var, term)


_T = lambda t: _is("temp", t)  # noqa: E731
_H = lambda t: _is("hum", t)  # noqa: E731

CLIMATE = [
    (("and", _T("cold"), _H("dry")), "off", 1.0),
    (("and", _T("cold"), _H("normal")), "off", 1.0),
    (("and", _T("cold"), _H("humid")), "low", 1.0),
    (("and", _T("mild"), _H("dry")), "low", 1.0),
    (("and", _T("mild"), _H("normal")), "mid", 1.0),
    (("and", _T("mild"), _H("humid")), "high", 1.0),
    (("and", _T("hot"), _H("dry")), "mid", 1.0),
    (("and", _T("hot"), _H("normal")), "high", 1.0),
    (("and", _T("hot"), _H("humid")), "full", 1.0),
]
MIXED = [
    (("or", _T("cold"), _H("dry")), "off", 1.0),
    (("and", _T("mild"), ("not", _H("humid"))), "low", 0.8),
    (("or", _T("mild"), _H("normal")), "mid", 0.6),
    (("and", _T("hot"), ("not", _H("dry"))), "high", 1.0),
    (("and", _T("hot"), _H("humid")), "full", 0.5),
]
GAP = [(_T("cold"), "off", 1.0)]

SUGENO_OUTPUTS = {
    "off": (0.0, {}),
    "low": (20.0, {}),
    "mid": (5.0, {"temp": 1.0, "hum": 0.2}),
    "high": (10.0, {"temp": 1.5, "hum": 0.4}),
    "full": (100.0, {}),
}


def grid(points: int) -> list[tuple[float, float]]:
    """``points`` by ``points`` evenly spaced (temp, hum) inputs, universe bounds included."""
    t0, t1 = UNIVERSE["temp"]
    h0, h1 = UNIVERSE["hum"]
    return [
        (t0 + (t1 - t0) * i / (points - 1), h0 + (h1 - h0) * j / (points - 1))
        for i in range(points)
        for j in range(points)
    ]


def _mf(spec):
    kind, *p = spec
    return {"tri": triangular, "trap": trapezoidal, "gauss": gaussian}[kind](*p)


def _variable(name):
    return LinguisticVariable(name, UNIVERSE[name], {t: _mf(s) for t, s in TERMS[name].items()})


def antecedent_text(node) -> str:
    """Rule text in bulanik's grammar."""
    op = node[0]
    if op == "is":
        return f"{node[1]} IS {node[2]}"
    if op == "not":
        inner = node[1]  # the grammar negates an atom: "var IS NOT term"
        return f"{inner[1]} IS NOT {inner[2]}"
    return f"({antecedent_text(node[1])}) {op.upper()} ({antecedent_text(node[2])})"


def rule_texts(rules) -> list[str]:
    return [
        f"IF {antecedent_text(a)} THEN fan IS {c}" + (f" WITH {w}" if w != 1.0 else "")
        for a, c, w in rules
    ]


def bulanik_mamdani(rules, *, resolution=1001, and_method="min", defuzzification="centroid") -> MamdaniFIS:
    return MamdaniFIS(
        [_variable("temp"), _variable("hum")], [_variable("fan")], rule_texts(rules),
        and_method=and_method, resolution=resolution, defuzzification=defuzzification,
    )


def bulanik_sugeno(rules, *, and_method="min") -> SugenoFIS:
    funcs = {
        n: Linear(coef, c) if coef else c for n, (c, coef) in SUGENO_OUTPUTS.items()
    }
    return SugenoFIS(
        [_variable("temp"), _variable("hum")], [SugenoOutput("fan", funcs)], rule_texts(rules),
        and_method=and_method,
    )
