"""Train an ANFIS without gradients (sezgi's jDE) and compare with Jang's hybrid rule.

Needs the sezgi extra: pip install "bulanik[sezgi]".
"""

import math

from bulanik.anfis import ANFIS
from bulanik.sezgi import train_anfis

X = [[2 * math.pi * i / 49] for i in range(50)]
y = [math.sin(x[0]) for x in X]

start = ANFIS(1, 3, ranges=[(0, 2 * math.pi)])  # 3 gaussian MFs -> 3 rules, 12 parameters
before = start.get_params()
result = train_anfis(start, X, y, budget=20000, seed=1)
print(f"sezgi RMSE {result.rmse:.4f} ({result.evals_used} evaluations)")

hybrid = ANFIS(1, 3, ranges=[(0, 2 * math.pi)])
history = hybrid.fit(X, y, epochs=10)
print(f"hybrid RMSE {history[-1]:.4f} (10 epochs)")

assert result.rmse < 0.1
assert start.get_params() == before  # train_anfis trains a copy
assert abs(result.anfis.rmse(X, y) - result.rmse) < 1e-12
