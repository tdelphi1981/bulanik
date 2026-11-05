"""First-order Sugeno heater controller on error and change of error."""

import math

from bulanik.inference import Linear, SugenoFIS, SugenoOutput
from bulanik.sets import LinguisticVariable, triangular

error = LinguisticVariable(
    "e", (-5, 5), {"neg": triangular(-5, -5, 5), "pos": triangular(-5, 5, 5)}
)
d_error = LinguisticVariable(
    "de", (-5, 5), {"neg": triangular(-5, -5, 5), "pos": triangular(-5, 5, 5)}
)
u = SugenoOutput(
    "u",
    {
        "cool": Linear({"e": 0.5, "de": 0.2}, -1.0),
        "hold": Linear({"e": 1.0}, 0.0),
        "heat": Linear({"e": 2.0, "de": 0.5}, 1.0),
    },
)
fis = SugenoFIS(
    [error, d_error],
    [u],
    [
        "IF e IS neg AND de IS neg THEN u IS cool",
        "IF e IS neg AND de IS pos THEN u IS hold",
        "IF e IS pos AND de IS neg THEN u IS hold",
        "IF e IS pos AND de IS pos THEN u IS heat",
    ],
)

result = fis.evaluate({"e": 2, "de": -1})["u"]
# e=2: neg .3, pos .7 ; de=-1: neg .6, pos .4 ; min-AND strengths: .3, .3, .6, .4
# cool = -1 + 1 - .2 = -.2 ; hold = 2 ; hold = 2 ; heat = 1 + 4 - .5 = 4.5
# u = (.3*-.2 + .3*2 + .6*2 + .4*4.5) / 1.6 = 3.54 / 1.6 = 2.2125
assert math.isclose(result, 2.2125)
print(f"u(e=2, de=-1) = {result:.4f}")
