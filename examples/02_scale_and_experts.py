"""Linguistic scales: three experts rate two suppliers on two criteria."""

from bulanik.aggregate import aggregate_experts
from bulanik.scales import EXAMPLE_IFN_5

scale = EXAMPLE_IFN_5

# ratings[expert][supplier][criterion], written as scale terms
words = [
    [["H", "M"], ["VH", "L"]],  # expert 1
    [["M", "M"], ["H", "VL"]],  # expert 2
    [["VH", "L"], ["M", "L"]],  # expert 3
]
ratings = scale.convert(words)
weights = [0.4, 0.35, 0.25]  # expert importance

matrix = aggregate_experts(ratings, weights, method="wa")

for s, row in enumerate(matrix):
    for c, agg in enumerate(row):
        mus = [r[s][c].mu for r in ratings]
        nus = [r[s][c].nu for r in ratings]
        assert 0.0 <= agg.mu and 0.0 <= agg.nu and agg.mu + agg.nu <= 1.0
        # mu and nu are weighted geometric means of (1 - mu) and nu, so each stays
        # inside the range of the opinions it mixes (the score mu - nu need not)
        assert min(mus) - 1e-9 <= agg.mu <= max(mus) + 1e-9
        assert min(nus) - 1e-9 <= agg.nu <= max(nus) + 1e-9
        print(f"aggregated: supplier {s + 1}, criterion {c + 1}: {agg} score {agg.score():.3f}")
