#!/usr/bin/env python3
"""Benchmark 3: bulanik ANFIS on a self-generated Mackey-Glass series.

Series: RK4 with step 0.1, tau = 17, beta = 0.2, gamma = 0.1, power 10, x(0) = 1.2, x(t<0) = 0.
Inputs x(t-18), x(t-12), x(t-6), x(t); target x(t+6); 500 train then 500 test points from t = 118.
Two Gaussian (or bell) terms per input give 16 rules. `fit` needs no extra: it is pure Python.
The sezgi extra is only needed for the optional derivative-free `train_anfis`.

Every setting tried is written to the result file; none is selected for reporting. There is no seed:
training is deterministic. Run from the repository root:

    uv run --frozen --isolated --no-dev python benchmarks/anfis_mackey_glass.py

Writes tests/benchmarks/data/anfis_mackey_glass_results.json.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests" / "benchmarks"))

import bulanik  # noqa: E402
from _bench_support import mackey_glass_dataset  # noqa: E402
from bulanik._linalg import normal_equations, solve_spd  # noqa: E402
from bulanik.anfis import ANFIS, ExtrapolationWarning  # noqa: E402

OUT = REPO / "tests" / "benchmarks" / "data" / "anfis_mackey_glass_results.json"
CHECKPOINTS = (1, 10, 100, 500)
SETTINGS = [(mf, lr) for lr in (0.01, 0.1, 1.0, 10.0, 100.0) for mf in ("gaussian", "bell")]


def rmse(pred, target):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pred, target)) / len(target))


def linear_baseline(xtr, ytr, xte, yte):
    design = [row + [1.0] for row in xtr]
    gram, aty = normal_equations(design, ytr)
    for i in range(len(gram)):
        gram[i][i] += 1e-10
    theta = solve_spd(gram, aty)
    predict = lambda rows: [sum(t * v for t, v in zip(theta, r + [1.0])) for r in rows]  # noqa: E731
    return rmse(predict(xtr), ytr), rmse(predict(xte), yte)


def run(mf, lr, xtr, ytr, xte, yte):
    model = ANFIS(4, 2, mf=mf)
    out, done, seconds = [], 0, 0.0
    for stop in CHECKPOINTS:
        t0 = time.perf_counter()
        model.fit(xtr, ytr, epochs=stop - done, lr=lr)
        seconds += time.perf_counter() - t0
        done = stop
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ExtrapolationWarning)
            test = model.rmse(xte, yte)
        out.append({"epochs": stop, "train_rmse": model.rmse(xtr, ytr), "test_rmse": test,
                    "fit_seconds": round(seconds, 2), "extrapolation_warned": bool(caught)})
    return {"mf": mf, "lr": lr, "default_lr": lr == 0.01, "n_rules": model.n_rules, "checkpoints": out}


def main() -> None:
    xtr, ytr, xte, yte = mackey_glass_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        runs = [run(mf, lr, xtr, ytr, xte, yte) for mf, lr in SETTINGS]
    lin_tr, lin_te = linear_baseline(xtr, ytr, xte, yte)
    data = {
        "produced_by": {"script": "benchmarks/anfis_mackey_glass.py", "bulanik": bulanik.__version__,
                        "python": sys.version.split()[0], "machine": platform.machine()},
        "series": {"method": "RK4", "dt": 0.1, "tau": 17, "beta": 0.2, "gamma": 0.1, "power": 10, "x0": 1.2,
                   "first_t": 118, "n_train": len(ytr), "n_test": len(yte),
                   "x_118": xtr[0][3], "target_last": yte[-1]},
        "baselines": {
            "persist_last_value": {"train_rmse": rmse([r[3] for r in xtr], ytr), "test_rmse": rmse([r[3] for r in xte], yte)},
            "linear_least_squares": {"train_rmse": lin_tr, "test_rmse": lin_te},
        },
        "runs": runs,
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
