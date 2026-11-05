"""Tune the constants of a zero-order Sugeno system with sezgi.

A FIS can be the evaluated function inside a sezgi.Problem: here jDE searches
the two rule constants that make the system reproduce y = 2x + 1.

Needs the sezgi extra: pip install "bulanik[sezgi]".
"""

import sezgi

from bulanik.inference import SugenoFIS, SugenoOutput
from bulanik.sets import LinguisticVariable, triangular

x_var = LinguisticVariable("x", (0, 10), {"low": triangular(0, 0, 10), "high": triangular(0, 10, 10)})
RULES = ["IF x IS low THEN y IS c1", "IF x IS high THEN y IS c2"]
XS = list(range(11))
TARGET = [2 * x + 1 for x in XS]


def build(c1, c2):
    y = SugenoOutput("y", {"c1": c1, "c2": c2})
    return SugenoFIS([x_var], [y], RULES)


class FitSugeno(sezgi.Problem):
    """Mean squared error of the Sugeno system on x = 0..10."""

    def space(self):
        return sezgi.Float(-50, 50, 2)

    def evaluate(self, c):
        fis = build(c[0], c[1])
        return sum((fis.evaluate({"x": x})["y"] - t) ** 2 for x, t in zip(XS, TARGET)) / len(XS)


result = sezgi.DifferentialEvolution(pop_size=40, variant="jde").run(FitSugeno(), budget=4000, seed=1)
c1, c2 = result.best_x
print(f"fitted c1={c1:.2f} c2={c2:.2f} (MSE {result.best_f:.2e})")

# Hand check: low(x) = 1 - x/10 and high(x) = x/10 sum to 1, so the output is
# c1 (1 - x/10) + c2 x/10. It equals 2x + 1 at x = 0 (c1 = 1) and x = 10 (c2 = 21).
assert abs(c1 - 1) < 0.05 and abs(c2 - 21) < 0.05
