"""All ten crisp ranking methods on one decision matrix, printed as a rank table.

Five suppliers are judged on price (cost), quality, delivery and service (benefits).
TOPSIS, VIKOR, WASPAS, EDAS, CODAS, COPRAS, MARCOS, CoCoSo, WISP and ELECTRE use the
same weights, so the table shows where the methods agree and where they do not.
"""

from bulanik.mcdm import cocoso, codas, copras, edas, electre, marcos, topsis, vikor, waspas, wisp

alternatives = ["S1", "S2", "S3", "S4", "S5"]
criteria = ["price", "quality", "delivery", "service"]
types = ["cost", "benefit", "benefit", "benefit"]
weights = [0.35, 0.30, 0.20, 0.15]
matrix = [
    [120, 8, 7, 6],
    [100, 6, 8, 7],
    [140, 9, 6, 8],
    [90, 5, 9, 5],
    [110, 7, 7, 9],
]

methods = {
    "topsis": topsis,
    "vikor": vikor,
    "waspas": waspas,
    "edas": edas,
    "codas": codas,
    "copras": copras,
    "marcos": marcos,
    "cocoso": cocoso,
    "wisp": wisp,
    "electre": electre,
}
results = {
    name: fn(matrix, weights, types, alternatives=alternatives, criteria=criteria)
    for name, fn in methods.items()
}

print("rank table (1 = best)")
print(f"{'method':<8}" + "".join(f"{a:>5}" for a in alternatives) + "   best first")
for name, result in results.items():
    position = {a: i + 1 for i, a in enumerate(result.ranking)}
    print(f"{name:<8}" + "".join(f"{position[a]:>5}" for a in alternatives) + "   " + " > ".join(result.ranking))

winners = {name: r.ranking[0] for name, r in results.items()}
print("winners:", winners)

assert len(results) == 10 and all(len(r.ranking) == 5 for r in results.values())
assert all(sorted(r.ranking) == alternatives for r in results.values())
# nine of ten methods put S5 first; CODAS, which is driven by distance from the
# negative-ideal solution, prefers S4 (cheapest, weakest quality)
assert [n for n, w in winners.items() if w == "S5"] == [n for n in methods if n != "codas"]
assert winners["codas"] == "S4"
assert results["topsis"].ranking == ("S5", "S1", "S2", "S3", "S4")
assert results["copras"].ranking == ("S5", "S2", "S3", "S4", "S1")
assert results["electre"].ranking == ("S5", "S2", "S1", "S4", "S3")
