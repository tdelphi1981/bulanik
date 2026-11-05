"""Fuzzy numbers: arithmetic, operator families, scores and weighted aggregation."""

import math

from bulanik import IFN, PFN, TFN, SphericalFN
from bulanik.aggregate import wa

# Triangular numbers use interval-style arithmetic.
cost = TFN(1, 2, 3) + TFN(3, 4, 5)
print("TFN sum:", cost, "centroid score:", cost.score())

# Intuitionistic numbers: the algebraic sum is the probabilistic sum of mu.
a, b = IFN(0.6, 0.3), IFN(0.5, 0.4)
total = a + b
assert math.isclose(total.mu, 0.8, rel_tol=1e-9)
print("IFN sum:", total)

# Scalar multiplication, and the algebraic vs Einstein operator families.
print("2 x a  :", a * 2)
print("algebraic:", a.add(b, ops="algebraic"))
print("einstein :", a.add(b, ops="einstein"))

# Scores are named too; results report the name you picked.
print("score difference:", round(total.score("difference"), 4), "normalized:", round(total.score("normalized"), 4))

# Pythagorean and spherical numbers allow more room than intuitionistic ones.
p = PFN(0.8, 0.5)  # 0.64 + 0.25 <= 1, impossible as an IFN
s = SphericalFN(0.6, 0.3, 0.4)
print("PFN:", p, "score:", round(p.score(), 4), "| Spherical:", s, "score:", round(s.score(), 4))

# Weighted average of two opinions.
print("WA:", wa([a, b], [0.7, 0.3]))
