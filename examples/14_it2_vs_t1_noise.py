"""Noise sensitivity of an interval type-2 vs a type-1 Sugeno approximator.

Both systems approximate y = x on [0, 10] with three rules (centres 0, 5, 10,
constant consequents 0, 5, 10). The type-1 terms are gaussians with sigma 1.5;
the type-2 terms have an uncertain sigma in [1.0, 2.0] (KM type reduction).

What is measured: 200 inputs x are drawn with random.Random(7); each gets
gaussian noise with sigma 1.0 and is clipped to [0, 10]. A system's noise
sensitivity is the RMSE between its output on the noisy input and its OWN
output on the clean input, so it measures how much the noise moves the output,
not how well the system fits y = x. The fit RMSE (clean output vs y = x) is
printed for context only.

Final numbers (seed 7): sensitivity T1 1.2981, IT2 1.1204; fit T1 0.7249,
IT2 0.3949. Seeds 11, 23, 42 and 101 give the same ordering.

Caveat: the comparison is against this type-1 baseline. The type-2 system
behaves like a smoother type-1 one, and a type-1 system with sigma 2.0 is
itself less sensitive (1.0491 at seed 7); type-2 is not more robust than every
type-1 choice.
"""

import math
import random

from bulanik.inference import SugenoFIS, SugenoOutput
from bulanik.sets import LinguisticVariable, gaussian
from bulanik.type2 import IT2LinguisticVariable, IT2SugenoFIS, gaussian_uncertain_sigma

CENTRES = {"low": 0.0, "mid": 5.0, "high": 10.0}
RULES = [f"IF x IS {term} THEN y IS {term}" for term in CENTRES]
y = SugenoOutput("y", dict(CENTRES))

t1 = SugenoFIS(
    [LinguisticVariable("x", (0, 10), {t: gaussian(c, 1.5) for t, c in CENTRES.items()})], [y], RULES
)
it2 = IT2SugenoFIS(
    [IT2LinguisticVariable("x", (0, 10), {t: gaussian_uncertain_sigma(c, 1.0, 2.0) for t, c in CENTRES.items()})],
    [y],
    RULES,
)

rng = random.Random(7)
n = 200
sens_t1 = sens_it2 = fit_t1 = fit_it2 = 0.0
for _ in range(n):
    x = rng.uniform(0, 10)
    noisy = min(max(x + rng.gauss(0, 1.0), 0.0), 10.0)
    clean1, clean2 = t1.evaluate({"x": x})["y"], it2.evaluate({"x": x})["y"]
    sens_t1 += (t1.evaluate({"x": noisy})["y"] - clean1) ** 2
    sens_it2 += (it2.evaluate({"x": noisy})["y"] - clean2) ** 2
    fit_t1 += (clean1 - x) ** 2
    fit_it2 += (clean2 - x) ** 2
sens_t1, sens_it2 = math.sqrt(sens_t1 / n), math.sqrt(sens_it2 / n)
fit_t1, fit_it2 = math.sqrt(fit_t1 / n), math.sqrt(fit_it2 / n)

assert sens_it2 < sens_t1
print(f"noise sensitivity: T1 RMSE = {sens_t1:.4f}, IT2 RMSE = {sens_it2:.4f}")
print(f"fit to y = x (clean inputs, context): T1 RMSE = {fit_t1:.4f}, IT2 RMSE = {fit_it2:.4f}")
