"""Tsukamoto: monotonic consequents are inverted at each rule's firing strength."""

import math

from bulanik.inference import TsukamotoFIS
from bulanik.sets import LinguisticVariable, linear_decreasing, linear_increasing, triangular

x = LinguisticVariable("x", (0, 10), {"low": triangular(0, 0, 10), "high": triangular(0, 10, 10)})
z = LinguisticVariable(
    "z", (0, 100), {"small": linear_decreasing(20, 60), "large": linear_increasing(40, 100)}
)
fis = TsukamotoFIS([x], [z], ["IF x IS low THEN z IS small", "IF x IS high THEN z IS large"])

result = fis.evaluate({"x": 3})["z"]
# x=3: low .7, high .3 ; z_small = 60 - 40*.7 = 32 ; z_large = 40 + 60*.3 = 58
# output = (.7*32 + .3*58) / (.7 + .3) = 39.8
assert math.isclose(result, 39.8)
print(f"z(x=3) = {result:.2f}")
