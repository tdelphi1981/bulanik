"""Fuzzy DEMATEL: three experts' triangular matrices, CFCS, causes and effects, then WISP.

Each expert states how strongly factor i influences factor j with a triangular fuzzy
number. `cfcs` (Opricovic and Tzeng 2003) converts the three matrices into one crisp
matrix; `dematel` then finds the total-relation matrix, the cause and effect groups
and factor weights. The factor weights (`as_weights()`) become the criteria weights
of a WISP ranking of three projects.
"""

from bulanik import TFN
from bulanik.mcdm import dematel, wisp
from bulanik.mcdm.dematel import cfcs

factors = ["quality", "cost", "delivery", "service"]

# linguistic influence words as triangular numbers
NO, LOW, MID, HIGH = TFN(0, 0, 0.25), TFN(0, 0.25, 0.5), TFN(0.25, 0.5, 0.75), TFN(0.5, 0.75, 1)
VERY = TFN(0.75, 1, 1)

expert_1 = [
    [NO, MID, HIGH, HIGH],
    [LOW, NO, MID, LOW],
    [LOW, LOW, NO, MID],
    [LOW, NO, LOW, NO],
]
expert_2 = [
    [NO, HIGH, MID, HIGH],
    [MID, NO, MID, LOW],
    [LOW, MID, NO, HIGH],
    [NO, LOW, LOW, NO],
]
expert_3 = [
    [NO, MID, VERY, MID],
    [LOW, NO, HIGH, NO],
    [NO, LOW, NO, MID],
    [LOW, LOW, NO, NO],
]

crisp = cfcs([expert_1, expert_2, expert_3])
result = dematel(crisp, factors)

print("total relation T (rounded):")
for name, row in zip(factors, result.total):
    print(f"  {name:<9}" + "".join(f"{x:8.3f}" for x in row))
print("prominence D+R:", {k: round(v, 3) for k, v in result.prominence.items()})
print("relation D-R  :", {k: round(v, 3) for k, v in result.relation.items()})
print("causes:", ", ".join(result.causes))
print("effects:", ", ".join(result.effects))

weights = result.as_weights()
print("weights:", {k: round(v, 4) for k, v in weights.weights.items()})

projects = ["Project X", "Project Y", "Project Z"]
scores = [
    [8, 6, 7, 5],  # quality, cost (cost-type: lower is better), delivery, service
    [7, 4, 8, 7],
    [9, 7, 5, 6],
]
ranked = wisp(scores, weights, ["benefit", "cost", "benefit", "benefit"],
              alternatives=projects, criteria=factors)
print("WISP ranking:", " > ".join(ranked.ranking))

assert abs(sum(weights.weights.values()) - 1.0) < 1e-9
assert result.causes == ("quality", "cost") and result.effects == ("delivery", "service")
expected_w = {"quality": 0.2877, "cost": 0.2332, "delivery": 0.2551, "service": 0.224}
assert all(abs(weights.weights[k] - v) < 5e-5 for k, v in expected_w.items())
assert ranked.ranking == ("Project Y", "Project X", "Project Z")
assert set(result.causes) | set(result.effects) == set(factors)
assert abs(sum(result.relation.values())) < 1e-9
# a TFN matrix handed to dematel directly is defuzzified by CFCS with one expert
single = dematel(expert_1, factors)
assert single.operators["defuzzification"] == "cfcs"
