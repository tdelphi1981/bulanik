import pytest
from hypothesis import settings

from bulanik.inference import Linear, SugenoOutput
from bulanik.sets import LinguisticVariable, linear_decreasing, linear_increasing, triangular

settings.register_profile("bulanik", deadline=None, max_examples=100)
settings.load_profile("bulanik")

TIPPER_RULES = [
    "IF service IS poor OR food IS rancid THEN tip IS cheap",
    "IF service IS average THEN tip IS average",
    "IF service IS good OR food IS delicious THEN tip IS generous",
]

SERVICE = LinguisticVariable(
    "service",
    (0, 10),
    {"poor": triangular(0, 0, 5), "average": triangular(0, 5, 10), "good": triangular(5, 10, 10)},
)
FOOD = LinguisticVariable(
    "food", (0, 10), {"rancid": triangular(0, 0, 5), "delicious": triangular(5, 10, 10)}
)
TIP = LinguisticVariable(
    "tip",
    (0, 30),
    {
        "cheap": triangular(0, 5, 10),
        "average": triangular(10, 15, 20),
        "generous": triangular(20, 25, 30),
    },
)
SPEED = LinguisticVariable(
    "speed", (0, 1), {"slow": triangular(0, 0, 1), "fast": triangular(0, 1, 1)}
)
# x on [0,10]: at x=4 low .6 / high .4 ; x=3 .7/.3 ; x=8 .2/.8
X = LinguisticVariable("x", (0, 10), {"low": triangular(0, 0, 10), "high": triangular(0, 10, 10)})
OUT = SugenoOutput("y", {"f_low": Linear({"x": 0.5}, 1.0), "f_high": 8.0})
Z = LinguisticVariable(
    "z", (0, 100), {"small": linear_decreasing(20, 60), "large": linear_increasing(40, 100)}
)
TIPPER_INPUTS = [SERVICE, FOOD]
TIPPER_OUTPUTS = [TIP]


@pytest.fixture
def tipper():
    """Factory for the textbook tipper (step 0.01, so 5, 15 and 25 are grid points)."""
    from bulanik.inference import MamdaniFIS

    def make(rules=TIPPER_RULES, with_speed=False, **fis_kwargs):
        outputs = list(TIPPER_OUTPUTS)
        rules = list(rules)
        if with_speed:
            outputs.append(SPEED)
            rules[0] += " AND speed IS slow"
            rules[2] += " AND speed IS fast"
        fis_kwargs.setdefault("resolution", 3001)
        return MamdaniFIS(TIPPER_INPUTS, outputs, rules, **fis_kwargs)

    return make


@pytest.fixture
def overlap_fis():
    """Two overlapping rules on [0,10]; strengths .6 and .4 at x=4."""
    from bulanik.inference import MamdaniFIS

    def make(**fis_kwargs):
        x = LinguisticVariable("x", (0, 10), {"a": triangular(0, 0, 10), "b": triangular(0, 10, 10)})
        y = LinguisticVariable("y", (0, 10), {"m": triangular(0, 5, 10), "r": triangular(5, 10, 10)})
        rules = ["IF x IS a THEN y IS m", "IF x IS b THEN y IS r"]
        return MamdaniFIS([x], [y], rules, resolution=1001, **fis_kwargs)

    return make
