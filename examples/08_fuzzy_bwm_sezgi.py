"""Fuzzy best-worst method (Guo & Zhao 2017) solved with sezgi's jDE.

Needs the sezgi extra: pip install "bulanik[sezgi]".
"""

import math

from bulanik.mcdm.bwm import fuzzy_bwm_consistency_index
from bulanik.scales import FUZZY_BWM_TFN as S
from bulanik.sezgi import FuzzyBWMProblem

# C1 is the best criterion and C3 the worst, rated with Guo & Zhao's linguistic terms.
best_to_others = {"C1": S["EI"], "C2": S["FI"], "C3": S["VI"]}
others_to_worst = {"C1": S["VI"], "C2": S["FI"], "C3": S["EI"]}

result = FuzzyBWMProblem(best_to_others, others_to_worst).solve(seed=1)
print(f"best = {result.best}, worst = {result.worst}")
for c, w in result.fuzzy_weights.items():
    print(f"  {c}: ({w.l:.4f}, {w.m:.4f}, {w.u:.4f}) -> {result.weights[c]:.4f}")
print(f"k* = {result.xi:.4f}, CR = {result.consistency_ratio:.4f}")

# Hand check: the middle values alone are the crisp BWM a12 = 2, a13 = 3, a23 = 2,
# whose optimum solves (2 - k)(2 - k) = 3 + k, i.e. k^2 - 5k + 1 = 0, k = (5 - sqrt 21) / 2.
# That is a lower bound on k*, and the fuzzy model attains it.
k_star = (5 - math.sqrt(21)) / 2
assert abs(result.xi - k_star) < 1e-3
# CI(VI) is the larger root of k^2 - 8k + 35/4 = 0 (u = 7/2): (8 + sqrt 29) / 2 = 6.69
assert abs(fuzzy_bwm_consistency_index(S["VI"]) - (8 + math.sqrt(29)) / 2) < 1e-12
assert result.ranking == ("C1", "C2", "C3")
assert abs(sum(result.weights.values()) - 1) < 1e-9
