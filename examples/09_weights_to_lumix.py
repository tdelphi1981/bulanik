"""MCDM scores as LumiX data: pick the best pair of suppliers within a budget.

TOPSIS scores the suppliers, scores_as_data turns them into records, and a LumiX
binary model chooses at most two suppliers that maximise the total score
without exceeding the setup budget.

Needs the lumix extra: pip install "bulanik[lumix]".
"""

from lumix import LXConstraint, LXModel, LXOptimizer, LXVariable
from lumix.core import LXLinearExpression

from bulanik.lumix import scores_as_data
from bulanik.mcdm import swara, topsis

# Same supplier data as example 04.
criteria = ["price", "quality", "delivery"]
types = ["cost", "benefit", "benefit"]
suppliers = ["Atlas", "Borealis", "Cedar", "Delta"]
data = [
    [120, 8, 7],  # Atlas
    [100, 7, 5],  # Borealis
    [110, 9, 6],  # Cedar
    [130, 6, 9],  # Delta
]
weights = swara([0.2, 0.5], criteria)
ranking = topsis(data, weights, types, alternatives=suppliers, criteria=criteria)
records = scores_as_data(ranking)
for r in records:
    print(f"{r.key}: score {r.value:.4f}, rank {r.rank}")

setup_cost = {"Atlas": 50, "Borealis": 25, "Cedar": 45, "Delta": 30}  # thousand EUR
budget = 75

pick = LXVariable[object, int]("pick").binary().indexed_by(lambda r: r.key).from_data(records)
model = (
    LXModel("supplier_pair")
    .add_variable(pick)
    .add_constraint(
        LXConstraint("budget")
        .expression(LXLinearExpression().add_term(pick, lambda r: setup_cost[r.key]))
        .le()
        .rhs(budget)
    )
    .add_constraint(
        LXConstraint("at_most_two").expression(LXLinearExpression().add_term(pick, 1.0)).le().rhs(2)
    )
    .maximize(LXLinearExpression().add_term(pick, lambda r: r.value))
)
solution = LXOptimizer().use_solver("ortools").solve(model)
assert solution.is_optimal(), solution.status
selected = sorted(k for k, v in solution.variables["pick"].items() if v > 0.5)
print("selected:", ", ".join(selected))

# Hand check: the TOPSIS scores are Cedar .5978, Atlas .5231, Delta .4338 and
# Borealis .4199 (example 04). The top two, Cedar and Atlas, cost 95 > 75. The
# pairs within budget are Cedar+Delta (75, 1.0316), Cedar+Borealis (70, 1.0177),
# Atlas+Borealis (75, .9429) and Borealis+Delta (55, .8537); a single supplier
# scores at most .5978. So the best feasible pair is Cedar and Delta.
assert selected == ["Cedar", "Delta"]
