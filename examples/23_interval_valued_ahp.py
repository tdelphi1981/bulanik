"""Interval-valued AHP: one set of judgements, two number kinds, three ways to get weights.

One decision maker compares four criteria with linguistic terms. The same terms are
written twice, as interval-valued Pythagorean numbers (IVPFN, q = 2) and as
interval-valued intuitionistic numbers (IVIFN, q = 1), and the matrices go through

- `ahp_interval_valued(method="multiplicative")`, which takes any rung;
- `ahp_interval_valued(method="possibility")`, which is defined for IVIFN only, so it
  gets the intuitionistic version of the judgements;
- `ahp_fuzzy_numbers`, the generic AHP over any one number kind: row geometric mean, then
  score, then normalise.

The scale values are illustrative.
"""

from bulanik import IVIFN, IVPFN
from bulanik.mcdm import ahp_fuzzy_numbers, ahp_interval_valued
from bulanik.scales import LinguisticScale

criteria = ["cost", "quality", "delivery", "service"]

# EQ equal, then more important (MOD, HI, VHI) and less important (LOW, VLO, ELO)
pythagorean = LinguisticScale("IVPF terms", {
    "EQ": IVPFN(0.50, 0.60, 0.50, 0.60),
    "MOD": IVPFN(0.60, 0.70, 0.30, 0.40),
    "HI": IVPFN(0.70, 0.80, 0.20, 0.30),
    "VHI": IVPFN(0.80, 0.95, 0.05, 0.15),
    "LOW": IVPFN(0.30, 0.40, 0.60, 0.70),
    "VLO": IVPFN(0.20, 0.30, 0.70, 0.80),
    "ELO": IVPFN(0.05, 0.15, 0.80, 0.95),
})
intuitionistic = LinguisticScale("IVIF terms", {
    "EQ": IVIFN(0.45, 0.45, 0.45, 0.45),
    "MOD": IVIFN(0.50, 0.60, 0.20, 0.30),
    "HI": IVIFN(0.60, 0.70, 0.15, 0.25),
    "VHI": IVIFN(0.70, 0.80, 0.05, 0.15),
    "LOW": IVIFN(0.20, 0.30, 0.50, 0.60),
    "VLO": IVIFN(0.15, 0.25, 0.60, 0.70),
    "ELO": IVIFN(0.05, 0.15, 0.70, 0.80),
})

judgements = [
    ["EQ", "MOD", "HI", "VHI"],
    ["LOW", "EQ", "MOD", "HI"],
    ["VLO", "LOW", "EQ", "MOD"],
    ["ELO", "VLO", "LOW", "EQ"],
]
ivpf = pythagorean.convert(judgements)
ivif = intuitionistic.convert(judgements)

# The crisp Saaty values of the same terms, for the consistency ratio
saaty = [[1, 3, 5, 7], [1 / 3, 1, 3, 5], [1 / 5, 1 / 3, 1, 3], [1 / 7, 1 / 5, 1 / 3, 1]]

runs = {
    "IVPF multiplicative": ahp_interval_valued(
        ivpf, criteria, method="multiplicative", consistency_matrix=saaty),
    "IVIF multiplicative": ahp_interval_valued(
        ivif, criteria, method="multiplicative", consistency_matrix=saaty),
    "IVIF possibility": ahp_interval_valued(
        ivif, criteria, method="possibility", consistency_matrix=saaty),
    "IVPF ahp_fuzzy_numbers": ahp_fuzzy_numbers(
        ivpf, criteria, score="normalized", consistency_matrix=saaty),
    "IVIF ahp_fuzzy_numbers": ahp_fuzzy_numbers(
        ivif, criteria, score="normalized", consistency_matrix=saaty),
}

print("weights compared")
print(f"{'run':<24}" + "".join(f"{c:>10}" for c in criteria) + f"{'CR':>8}")
for name, r in runs.items():
    print(f"{name:<24}" + "".join(f"{r.weights[c]:>10.3f}" for c in criteria)
          + f"{r.consistency_ratio:>8.3f}")

print()
print("the possibility method also gives interval weights")
intervals = runs["IVIF possibility"].extras["interval_weights"]
for c, (low, high) in zip(criteria, intervals):
    print(f"  {c:<10} [{low:.3f}, {high:.3f}]")

print()
print("possibility on the Pythagorean matrix:")
try:
    ahp_interval_valued(ivpf, criteria, method="possibility")
except ValueError as message:
    print(f"  {message}")

for r in runs.values():
    assert r.ranking == tuple(criteria)
