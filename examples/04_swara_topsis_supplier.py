"""SWARA criteria weights, then TOPSIS over four suppliers."""

from bulanik.mcdm import swara, topsis

criteria = ["price", "quality", "delivery"]
types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
#            price  quality  delivery score
data = [
    [120, 8, 7],  # Atlas
    [100, 7, 5],  # Borealis
    [110, 9, 6],  # Cedar
    [130, 6, 9],  # Delta
]

# Experts name price most important; quality is 0.2 less important, delivery 0.5 less.
# SWARA: q = 1, 1/1.2, 1/1.2/1.5 = 1, .8333, .5556 -> w = (.4186, .3488, .2326).
weights = swara([0.2, 0.5], criteria)
print("weights:", {k: round(v, 4) for k, v in weights.weights.items()})

result = topsis(data, weights, types, alternatives=suppliers, criteria=criteria)
for name in result.ranking:
    print(f"{name}: closeness {result.scores[name]:.4f}")
best = result.ranking[0]
print("best supplier:", best)

# Plausibility check (a weighted-majority argument, not a TOPSIS derivation):
# Cedar beats Atlas and Delta on price and quality, which carry weight .77
# together, and beats Borealis (the cheapest) on quality and delivery, which
# carry weight .58, while paying only 10 more. TOPSIS agrees: closeness .5978
# for Cedar against .5231 for Atlas, the runner-up.
assert best == "Cedar"
assert result.ranking == ("Cedar", "Atlas", "Delta", "Borealis")
expected = {"Cedar": 0.5978, "Atlas": 0.5231, "Delta": 0.4338, "Borealis": 0.4199}
assert all(abs(result.scores[k] - v) < 5e-5 for k, v in expected.items())
assert abs(weights.weights["price"] - 0.4186046511627907) < 1e-12
