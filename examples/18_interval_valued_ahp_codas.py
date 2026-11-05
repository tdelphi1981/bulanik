"""Interval-valued AHP and CODAS: IVIFN criteria weights, IVIFN ratings, plus an IVNN variant.

The criteria are compared pairwise with interval-valued intuitionistic numbers
(Saaty-style: a high membership means "row more important than column", the
reciprocal swaps membership and non-membership). `ahp_fuzzy_numbers` turns the
matrix into weights, and `rank_fuzzy(method="codas")` ranks the alternatives.
CODAS divides by entries, so the signed default score would be rejected; both
steps use `score="normalized"`, a score in [0, 1].
"""

from bulanik import IVIFN, IVNeutrosophicFN
from bulanik.mcdm import DecisionMatrix, ahp_fuzzy_numbers, rank_fuzzy

criteria = ["cost", "quality", "delivery"]
alternatives = ["Supplier A", "Supplier B", "Supplier C"]
types = ["cost", "benefit", "benefit"]

# pairwise comparison matrix of the criteria; the diagonal is "equally important"
equal = IVIFN(0.5, 0.5, 0.5, 0.5)
strong = IVIFN(0.7, 0.8, 0.1, 0.2)  # row clearly more important
mild = IVIFN(0.5, 0.6, 0.3, 0.4)  # row slightly more important
comparisons = [
    [equal, mild, strong],
    [mild.complement(), equal, mild],
    [strong.complement(), mild.complement(), equal],
]
weights = ahp_fuzzy_numbers(comparisons, criteria, score="normalized")
print("weights:", {k: round(v, 4) for k, v in weights.weights.items()})

# one expert's ratings: IVIFN [alternative][criterion]
ratings = [
    [IVIFN(0.3, 0.4, 0.4, 0.5), IVIFN(0.7, 0.8, 0.1, 0.2), IVIFN(0.5, 0.6, 0.3, 0.4)],
    [IVIFN(0.5, 0.6, 0.3, 0.4), IVIFN(0.5, 0.6, 0.3, 0.4), IVIFN(0.7, 0.8, 0.1, 0.2)],
    [IVIFN(0.6, 0.7, 0.2, 0.3), IVIFN(0.3, 0.4, 0.4, 0.5), IVIFN(0.3, 0.4, 0.4, 0.5)],
]
matrix = DecisionMatrix(alternatives, criteria, [ratings])
result = rank_fuzzy(matrix, weights, types, method="codas", score="normalized")
print("IVIFN ranking:", " > ".join(result.ranking))
for name in result.ranking:
    print(f"{name}: H {result.scores[name]:+.4f}")

# the same pipeline with interval-valued neutrosophic ratings (truth, indeterminacy,
# falsity); their only score, "average", is already non-negative
def nn(t, i, f):
    return IVNeutrosophicFN(t - 0.05, t + 0.05, i - 0.05, i + 0.05, f - 0.05, f + 0.05)

nn_ratings = [
    [nn(0.4, 0.2, 0.4), nn(0.8, 0.2, 0.2), nn(0.6, 0.3, 0.3)],
    [nn(0.6, 0.2, 0.3), nn(0.6, 0.3, 0.3), nn(0.8, 0.1, 0.2)],
    [nn(0.7, 0.2, 0.2), nn(0.4, 0.3, 0.5), nn(0.4, 0.3, 0.5)],
]
nn_matrix = DecisionMatrix(alternatives, criteria, [nn_ratings])
nn_result = rank_fuzzy(nn_matrix, weights, types, method="codas")
print("IVNN ranking:", " > ".join(nn_result.ranking))

assert result.operators["score"] == "normalized" and weights.operators["score"] == "normalized"
assert abs(sum(weights.weights.values()) - 1.0) < 1e-9
assert weights.ranking[0] == "cost"
expected_w = {"cost": 0.4279, "quality": 0.3391, "delivery": 0.2331}
assert all(abs(weights.weights[k] - v) < 5e-5 for k, v in expected_w.items())
assert result.ranking == ("Supplier A", "Supplier B", "Supplier C")
expected_h = {"Supplier A": 0.7845, "Supplier B": 0.1177, "Supplier C": -0.9023}
assert all(abs(result.scores[k] - v) < 5e-5 for k, v in expected_h.items())
assert nn_result.ranking == result.ranking
