"""Objective versus subjective weights (CRITIC and PIPRECIA), then COPRAS.

CRITIC derives weights from the data alone: a criterion earns weight for varying and
for conflicting with the others. PIPRECIA turns a decision maker's judgements of
"is this criterion more or less important than the previous one" into weights; the
inverse pass (`s_inverse`) is run as well and the two are averaged. The ranking by
COPRAS shows how much the choice of weights matters.
"""

from bulanik.mcdm import copras, critic, piprecia

alternatives = ["P1", "P2", "P3", "P4"]
criteria = ["cost", "reliability", "speed", "support"]
types = ["cost", "benefit", "benefit", "benefit"]
matrix = [
    [250, 16, 12, 5],
    [200, 16, 8, 3],
    [300, 32, 16, 4],
    [275, 32, 8, 4],
]

objective = critic(matrix, criteria, criteria_types=types)
print("CRITIC  :", {k: round(v, 4) for k, v in objective.weights.items()})

# judgements s_2..s_n: criterion j against criterion j-1 (> 1 more important, < 1 less);
# the inverse pass judges the same pairs from the other end (s'_1..s'_{n-1})
subjective = piprecia([0.9, 1.2, 0.8], criteria, s_inverse=[1.1, 0.7, 1.15])
print("PIPRECIA:", {k: round(v, 4) for k, v in subjective.weights.items()})
print("operators:", subjective.operators["inverse"])

by_critic = copras(matrix, objective, types, alternatives=alternatives, criteria=criteria)
by_piprecia = copras(matrix, subjective, types, alternatives=alternatives, criteria=criteria)
print("COPRAS ranking:")
print("  objective  :", " > ".join(by_critic.ranking))
print("  subjective :", " > ".join(by_piprecia.ranking))

assert abs(sum(objective.weights.values()) - 1.0) < 1e-9
assert abs(sum(subjective.weights.values()) - 1.0) < 1e-9
assert subjective.operators["inverse"] == "mean_of_forward_and_inverse"
exp_c = {"cost": 0.3119, "reliability": 0.2998, "speed": 0.2058, "support": 0.1825}
exp_p = {"cost": 0.2483, "reliability": 0.2246, "speed": 0.2862, "support": 0.2409}
assert all(abs(objective.weights[k] - v) < 5e-5 for k, v in exp_c.items())
assert all(abs(subjective.weights[k] - v) < 5e-5 for k, v in exp_p.items())
# the two weightings agree on the best and the worst but swap P1 and P4
assert by_critic.ranking == ("P3", "P4", "P1", "P2")
assert by_piprecia.ranking == ("P3", "P1", "P4", "P2")
