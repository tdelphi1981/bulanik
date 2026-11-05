"""Intuitionistic pipeline: IF expert ratings, IF-SWARA weights, VIKOR ranking."""

from bulanik.mcdm import DecisionMatrix, rank_fuzzy, swara
from bulanik.scales import EXAMPLE_IFN_7

alternatives = ["Plan A", "Plan B", "Plan C"]
criteria = ["cost", "risk", "benefit"]
types = ["cost", "cost", "benefit"]

# three experts rate [alternative][criterion] with 7-term IF words
experts = [
    [["M", "L", "H"], ["H", "M", "VH"], ["L", "VL", "M"]],
    [["M", "M", "H"], ["VH", "M", "H"], ["L", "L", "M"]],
    [["L", "L", "VH"], ["H", "H", "VH"], ["M", "VL", "H"]],
]
matrix = DecisionMatrix.from_terms(
    EXAMPLE_IFN_7, alternatives, criteria, experts, expert_weights=[0.4, 0.3, 0.3]
)

# criteria listed most important first; comparative importances are IF numbers,
# mapped to [0, 1] with the "normalized" score (1 + mu - nu) / 2
importance = [EXAMPLE_IFN_7["L"], EXAMPLE_IFN_7["M"]]
weights = swara(importance, criteria, score="normalized")
print("weights:", {k: round(v, 4) for k, v in weights.weights.items()})

result = rank_fuzzy(
    matrix, weights, types, method="vikor", aggregation="wa", ops="algebraic"
)
print("ranking:", " > ".join(result.ranking))
print("operators:", result.operators)
for name in result.ranking:
    print(f"{name}: Q {result.scores[name]:.4f}")

# VIKOR ranks by ascending Q: Plans A and C are close, Plan B is clearly worst (Q = 1)
assert result.ranking == ("Plan A", "Plan C", "Plan B")
expected_q = {"Plan A": 0.0889, "Plan C": 0.0905, "Plan B": 1.0}
assert all(abs(result.scores[k] - v) < 5e-5 for k, v in expected_q.items())
expected_w = {"cost": 0.4507, "risk": 0.3339, "benefit": 0.2154}
assert all(abs(weights.weights[k] - v) < 5e-5 for k, v in expected_w.items())
assert result.operators["score"] == "difference" and result.operators["aggregation"] == "wa"
assert abs(sum(weights.weights.values()) - 1.0) < 1e-9
