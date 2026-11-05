"""Buckley fuzzy AHP: three criteria compared with the Saaty triangular scale."""

from bulanik.mcdm import fuzzy_ahp
from bulanik.scales import SAATY_TFN

criteria = ["cost", "quality", "delivery"]

# upper triangle as Saaty terms; the lower triangle is the reciprocal
judgements = {(0, 1): "3", (0, 2): "5", (1, 2): "2"}
matrix = [[SAATY_TFN["1"] for _ in criteria] for _ in criteria]
for (i, j), term in judgements.items():
    matrix[i][j] = SAATY_TFN[term]
    matrix[j][i] = SAATY_TFN.reciprocal(term)

result = fuzzy_ahp(matrix, criteria)

for name in criteria:
    print(f"{name}: weight {result.weights[name]:.4f}, fuzzy {result.fuzzy_weights[name]}")
print(f"CR = {result.consistency_ratio:.4f}")

assert abs(sum(result.weights.values()) - 1.0) < 1e-9
assert result.consistency_ratio < 0.1
assert result.ranking == ("cost", "quality", "delivery")
