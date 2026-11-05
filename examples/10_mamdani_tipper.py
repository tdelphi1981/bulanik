"""Mamdani tipper: how much to tip for service 3/10 and food 8/10."""

import math

from bulanik.inference import MamdaniFIS
from bulanik.sets import LinguisticVariable, triangular

service = LinguisticVariable(
    "service",
    (0, 10),
    {"poor": triangular(0, 0, 5), "average": triangular(0, 5, 10), "good": triangular(5, 10, 10)},
)
food = LinguisticVariable(
    "food", (0, 10), {"rancid": triangular(0, 0, 5), "delicious": triangular(5, 10, 10)}
)
tip = LinguisticVariable(
    "tip",
    (0, 30),
    {"cheap": triangular(0, 5, 10), "average": triangular(10, 15, 20), "generous": triangular(20, 25, 30)},
)

fis = MamdaniFIS(
    [service, food],
    [tip],
    [
        "IF service IS poor OR food IS rancid THEN tip IS cheap",
        "IF service IS average THEN tip IS average",
        "IF service IS good OR food IS delicious THEN tip IS generous",
    ],
    resolution=3001,
)

result = fis.evaluate({"service": 3, "food": 8})["tip"]
# strengths .4, .6, .6 clip three disjoint triangles: areas 5h(2-h) = 3.2, 4.2, 4.2 with
# centres 5, 15, 25, so the centroid is (3.2*5 + 4.2*15 + 4.2*25) / 11.6 = 184 / 11.6
assert math.isclose(result, 184 / 11.6, abs_tol=1e-6)
print(f"tip(service=3, food=8) = {result:.2f}")

explanation = fis.explain({"service": 3, "food": 8})
print("input degrees:")
for variable, terms in explanation.degrees.items():
    print(f"  {variable}: " + ", ".join(f"{t}={mu:.2f}" for t, mu in terms.items()))
print("rule activations:")
for activation in explanation.activations:
    print(f"  {activation.strength:.2f}  {activation.rule}")
