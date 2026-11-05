"""Grid ANFIS learns the 2-D sinc surface with Jang's hybrid rule, then exports to Sugeno."""

import math

from bulanik.anfis import ANFIS


def sinc(t):
    return 1.0 if t == 0 else math.sin(t) / t


# 11 x 11 grid over [-10, 10]^2, z = sinc(x) * sinc(y)
grid = [-10 + 2 * k for k in range(11)]
X = [[x, y] for x in grid for y in grid]
z = [sinc(x) * sinc(y) for x, y in X]

anfis = ANFIS(2, 4)  # 16 rules, ranges taken from the data
history = anfis.fit(X, z, epochs=30, method="hybrid", lr=0.01)
final = history[-1]
print(f"epoch 1 RMSE {history[0]:.4f}")
print(f"final RMSE {final:.4f}")
assert final < 0.1  # prototype: 0.0926
# on this data, the training RMSE never gets worse after the first epoch
assert all(b <= a + 1e-9 for a, b in zip(history[1:], history[2:]))

# the trained network as a plain first-order Sugeno system (product AND)
fis = anfis.to_sugeno(["x", "y"], output_name="z")
same = fis.evaluate({"x": 1.5, "y": -2.5})["z"]
assert math.isclose(same, anfis.predict([[1.5, -2.5]])[0], rel_tol=0, abs_tol=1e-12)
print(f"Sugeno export z(1.5, -2.5) = {same:.4f} ({len(fis.rules)} rules)")
