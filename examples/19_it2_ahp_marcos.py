"""Interval type-2 AHP and MARCOS with an illustrative 7-term linguistic scale.

The scale below is **illustrative**, in the style of the interval type-2 literature:
its term values are not a transcription of any published table (the table of Chen
and Lee is not reproduced here). Seven IT2TrFN terms (upper trapezoid with height 1,
lower trapezoid with height 0.9) rate the criteria pairwise and the alternatives;
`ahp_fuzzy_numbers` weighs the criteria with the DTraT score and `rank_fuzzy`
ranks with MARCOS, which needs strictly positive entries (the DTraT score is).
"""

from bulanik import IT2TrFN
from bulanik.mcdm import DecisionMatrix, ahp_fuzzy_numbers, rank_fuzzy

# (upper trapezoid, lower trapezoid); heights 1 above, 0.9 below
_TERMS = {
    "VL": ((0, 0, 0, 0.1), (0, 0, 0, 0.05)),
    "L": ((0, 0.1, 0.1, 0.3), (0.05, 0.1, 0.1, 0.2)),
    "ML": ((0.1, 0.3, 0.3, 0.5), (0.2, 0.3, 0.3, 0.4)),
    "M": ((0.3, 0.5, 0.5, 0.7), (0.4, 0.5, 0.5, 0.6)),
    "MH": ((0.5, 0.7, 0.7, 0.9), (0.6, 0.7, 0.7, 0.8)),
    "H": ((0.7, 0.9, 0.9, 1.0), (0.8, 0.9, 0.9, 0.95)),
    "VH": ((0.9, 1.0, 1.0, 1.0), (0.95, 1.0, 1.0, 1.0)),
}
SCALE = {k: IT2TrFN((*u, 1, 1), (*l, 0.9, 0.9)) for k, (u, l) in _TERMS.items()}

criteria = ["cost", "quality", "service"]
alternatives = ["Vendor 1", "Vendor 2", "Vendor 3"]
types = ["cost", "benefit", "benefit"]

# Pairwise comparison of the criteria. IT2TrFN has no reciprocal method, so the lower
# triangle uses the trapezoid reciprocal (1/d, 1/c, 1/b, 1/a) written out here; the
# scale terms are shifted by 1 so that they read as ratios of at least 1.
def shifted(term):
    up, lo = term.upper, term.lower
    return IT2TrFN((*(1 + x for x in up[:4]), *up[4:]), (*(1 + x for x in lo[:4]), *lo[4:]))


def recip(x):
    u, l = x.upper, x.lower
    return IT2TrFN((1 / u[3], 1 / u[2], 1 / u[1], 1 / u[0], u[4], u[5]),
                   (1 / l[3], 1 / l[2], 1 / l[1], 1 / l[0], l[4], l[5]))


cq, cs, qs = shifted(SCALE["MH"]), shifted(SCALE["H"]), shifted(SCALE["M"])
one = IT2TrFN((1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 0.9, 0.9))
comparisons = [
    [one, cq, cs],
    [recip(cq), one, qs],
    [recip(cs), recip(qs), one],
]
weights = ahp_fuzzy_numbers(comparisons, criteria)
print("weights:", {k: round(v, 4) for k, v in weights.weights.items()})
print("score:", weights.operators["score"])

# two experts rate the alternatives with words of the scale
words = [
    [["M", "H", "MH"], ["H", "M", "H"], ["ML", "VH", "M"]],
    [["MH", "H", "M"], ["H", "MH", "H"], ["M", "H", "ML"]],
]
experts = [[[SCALE[w] for w in row] for row in table] for table in words]
matrix = DecisionMatrix(alternatives, criteria, experts)
result = rank_fuzzy(matrix, weights, types, method="marcos")
print("IT2 ranking:", " > ".join(result.ranking))
for name in result.ranking:
    print(f"{name}: f(K) {result.scores[name]:.4f}")
print("score used:", result.operators["score"])

assert abs(sum(weights.weights.values()) - 1.0) < 1e-9
assert weights.ranking[0] == "cost"
expected_w = {"cost": 0.4687, "quality": 0.3056, "service": 0.2257}
assert all(abs(weights.weights[k] - v) < 5e-5 for k, v in expected_w.items())
assert result.ranking == ("Vendor 3", "Vendor 1", "Vendor 2")
expected_f = {"Vendor 3": 0.7471, "Vendor 1": 0.6437, "Vendor 2": 0.5416}
assert all(abs(result.scores[k] - v) < 5e-5 for k, v in expected_f.items())
assert result.operators["score"] == "dtrat"
