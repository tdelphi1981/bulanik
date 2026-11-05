"""Linear best-worst method (Rezaei 2016) solved with LumiX.

Needs the lumix extra: pip install "bulanik[lumix]".
"""

from bulanik.lumix import bwm_linear

# C1 is the best criterion and C3 the worst. The decision maker's comparisons are
# slightly inconsistent: C1 over C2 (2) times C2 over C3 (3) is 6, not C1 over C3 (4).
best_to_others = {"C1": 1, "C2": 2, "C3": 4}
others_to_worst = {"C1": 4, "C2": 3, "C3": 1}

result = bwm_linear(best_to_others, others_to_worst)
print(f"best = {result.best}, worst = {result.worst}")
print("weights:", {k: round(v, 4) for k, v in result.weights.items()})
print(f"xi = {result.xi:.4f}")

# Hand check: |w1 - 2 w2|, |w1 - 4 w3| and |w2 - 3 w3| are all tight at xi, with
# w1 - 2 w2 = -xi, w1 - 4 w3 = xi and w2 - 3 w3 = -xi. Then w3 = 2 xi, w2 = 5 xi,
# w1 = 9 xi and the sum 16 xi = 1, so xi = 1/16 and w = (9, 5, 2) / 16.
assert abs(result.xi - 1 / 16) < 1e-7
for got, want in zip(result.weights.values(), (9 / 16, 5 / 16, 2 / 16)):
    assert abs(got - want) < 1e-7
assert result.ranking == ("C1", "C2", "C3")
