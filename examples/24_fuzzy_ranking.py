"""Score first or score last: CODAS, WASPAS and CoCoSo on an intuitionistic matrix.

Two experts rate four suppliers on price (a cost), quality and delivery with
intuitionistic numbers (mu, nu). `rank_fuzzy` aggregates them and then either

- scores every aggregated number first and ranks the crisp matrix with the
  crisp method (`method="codas"`, `"waspas"`, `"cocoso"`), or
- hands the aggregated *fuzzy* matrix to the fuzzy method
  (`method="fuzzy_codas"`, `"fuzzy_waspas"`, `"fuzzy_cocoso"`), which weights,
  sums and multiplies the numbers with intuitionistic arithmetic and scores
  only at the end.

The crisp ratio methods need positive entries, so every run uses
`score="normalized"`, (1 + mu - nu) / 2. The fuzzy WASPAS is run both ways of
combining its weighted sum and weighted product: as numbers
(`combine="fuzzy"`) and as scores (`combine="scores"`).
"""

from bulanik import IFN
from bulanik.mcdm import DecisionMatrix, rank_fuzzy

alternatives = ["Atlas", "Borealis", "Cedar", "Delta"]
criteria = ["price", "quality", "delivery"]
types = ["cost", "benefit", "benefit"]
weights = {"price": 0.40, "quality": 0.35, "delivery": 0.25}

# ratings [expert][alternative][criterion]; for price a high mu means "expensive"
expert1 = [
    [IFN(0.60, 0.30), IFN(0.70, 0.20), IFN(0.60, 0.30)],
    [IFN(0.40, 0.50), IFN(0.60, 0.30), IFN(0.40, 0.50)],
    [IFN(0.50, 0.40), IFN(0.80, 0.10), IFN(0.50, 0.40)],
    [IFN(0.70, 0.20), IFN(0.50, 0.40), IFN(0.80, 0.10)],
]
expert2 = [
    [IFN(0.50, 0.40), IFN(0.70, 0.20), IFN(0.70, 0.20)],
    [IFN(0.40, 0.40), IFN(0.50, 0.40), IFN(0.50, 0.40)],
    [IFN(0.60, 0.30), IFN(0.70, 0.20), IFN(0.40, 0.50)],
    [IFN(0.80, 0.10), IFN(0.60, 0.30), IFN(0.70, 0.20)],
]
matrix = DecisionMatrix(alternatives, criteria, [expert1, expert2])

runs = {
    "codas (score first)": dict(method="codas"),
    "fuzzy_codas": dict(method="fuzzy_codas"),
    "waspas (score first)": dict(method="waspas"),
    "fuzzy_waspas, fuzzy": dict(method="fuzzy_waspas", combine="fuzzy"),
    "fuzzy_waspas, scores": dict(method="fuzzy_waspas", combine="scores"),
    "cocoso (score first)": dict(method="cocoso"),
    "fuzzy_cocoso": dict(method="fuzzy_cocoso"),
}
results = {name: rank_fuzzy(matrix, weights, types, score="normalized", **kw)
           for name, kw in runs.items()}

print("fuzzy ranking table (1 = best)")
print(f"{'run':<22}" + "".join(f"{a:>10}" for a in alternatives))
for name, r in results.items():
    position = {a: i + 1 for i, a in enumerate(r.ranking)}
    print(f"{name:<22}" + "".join(f"{position[a]:>10}" for a in alternatives))

print()
print("scores (larger is better; compare within a method, not across methods)")
for name, r in results.items():
    print(f"{name:<22}" + "".join(f"{r.scores[a]:>10.4f}" for a in alternatives))

fw = results["fuzzy_waspas, fuzzy"]
print()
print("fuzzy_waspas keeps the numbers: Q1 (weighted sum) and Q2 (weighted product) of Cedar")
q1, q2 = fw.extras["Q1"]["Cedar"], fw.extras["Q2"]["Cedar"]
print(f"  Q1 = IFN({q1.mu:.4f}, {q1.nu:.4f}), Q2 = IFN({q2.mu:.4f}, {q2.nu:.4f})")

assert all(sorted(r.ranking) == sorted(alternatives) for r in results.values())
