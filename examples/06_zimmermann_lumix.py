"""Zimmermann's fuzzy goals on a LumiX model: a profit goal and a cost goal.

Needs the lumix extra: pip install "bulanik[lumix]".
"""

from lumix import LXModel, LXVariable
from lumix.core import LXLinearExpression

from bulanik.lumix import FuzzyGoal, solve_zimmermann

# Two activity levels; x1 is capped at 8 by a hard bound.
x1 = LXVariable[None, float]("x1").continuous().bounds(lower=0, upper=8)
x2 = LXVariable[None, float]("x2").continuous().bounds(lower=0)
model = LXModel("plan").add_variable(x1).add_variable(x2)

profit = LXLinearExpression().add_term(x1, 2.0).add_term(x2, 1.0)  # z1 = 2 x1 + x2
cost = LXLinearExpression().add_term(x1, 1.0).add_term(x2, 2.0)  # z2 = x1 + 2 x2
goals = [
    FuzzyGoal("profit", profit, aspiration=14, tolerance=4, sense="ge"),  # "about 14 or more"
    FuzzyGoal("cost", cost, aspiration=6, tolerance=4, sense="le"),  # "about 6 or less"
]

result = solve_zimmermann(model, goals)
print(f"lambda = {result.lambda_:.4f}")
for name, z in result.goal_values.items():
    print(f"{name}: z = {z:.4f}, membership {result.memberships[name]:.4f}")

# Hand check: rows 2x1 + x2 - 4λ >= 10 and x1 + 2x2 + 4λ <= 10. One third of the
# first plus two thirds of the second gives 4λ <= 10/3 - x2, so λ* = 5/6 at x2 = 0;
# both rows tight then give x1 = 20/3, z1 = 40/3 and z2 = 20/3.
assert abs(result.lambda_ - 5 / 6) < 1e-7
assert abs(result.goal_values["profit"] - 40 / 3) < 1e-6
assert abs(result.goal_values["cost"] - 20 / 3) < 1e-6
