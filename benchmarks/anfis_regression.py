#!/usr/bin/env python3
"""Benchmark 5: bulanik ANFIS on public regression data against scikit-learn, same folds.

UCI Airfoil Self-Noise (doi:10.24432/C5VW2C) and Auto MPG (doi:10.24432/C5859H), both CC BY 4.0.
Fixed 5-fold split from a seeded shuffle; every configuration listed below is reported.
Headline settings are the defaults on raw inputs: ANFIS Gaussian terms, 20 epochs, lr 0.01;
scikit-learn LinearRegression and GradientBoostingRegressor(random_state=0).
The default ANFIS runs blew up on some folds, so two diagnostic variants were added after seeing
that and are labelled as such: inputs scaled to [0, 1], and lr 1e-12 (consequents only).

Run from the repository root (needs network the first time; data stay in benchmarks/.cache):

    uv run --frozen --isolated --no-dev --with scikit-learn==1.9.1 python benchmarks/anfis_regression.py

Writes tests/benchmarks/data/anfis_regression_results.json.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import math
import random
import statistics
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bulanik  # noqa: E402
from _download import fetch  # noqa: E402
from bulanik.anfis import ANFIS, ExtrapolationWarning  # noqa: E402

OUT = REPO / "tests" / "benchmarks" / "data" / "anfis_regression_results.json"
SEED = 20261105
FOLDS = 5
EPOCHS = 20

AIRFOIL = ("https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip",
           "airfoil+self+noise.zip", "5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c")
AUTO = ("https://archive.ics.uci.edu/static/public/9/auto+mpg.zip",
        "auto+mpg.zip", "5fc1507522b73b8050d3aa45df51eefe1b666ae5f428a5f1347b6483339f1003")

AIRFOIL_NAMES = ["frequency", "attack_angle", "chord", "velocity", "thickness"]
AUTO_NAMES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]


def load_airfoil():
    with zipfile.ZipFile(fetch(*AIRFOIL)) as z:
        text = z.read("airfoil_self_noise.dat").decode()
    rows = [[float(v) for v in line.split()] for line in text.splitlines() if line.strip()]
    return [r[:5] for r in rows], [r[5] for r in rows], 0


def load_auto():
    with zipfile.ZipFile(fetch(*AUTO)) as z:
        text = z.read("auto-mpg.data").decode()
    X, y, dropped = [], [], 0
    for line in text.splitlines():
        fields = line.split('"')[0].split()
        if not fields:
            continue
        if "?" in fields:
            dropped += 1
            continue
        y.append(float(fields[0]))
        X.append([float(v) for v in fields[1:8]])
    return X, y, dropped


def folds(n):
    order = list(range(n))
    random.Random(SEED).shuffle(order)
    return [order[i::FOLDS] for i in range(FOLDS)]


def rmse(pred, target):
    return math.sqrt(sum((float(a) - b) ** 2 for a, b in zip(pred, target)) / len(target))


def cross_validate(make_fit_predict, X, y):
    split = folds(len(y))
    rows, fit_times, pred_times, warned = [], [], [], 0
    for k in range(FOLDS):
        test = set(split[k])
        tr = [i for i in range(len(y)) if i not in test]
        xtr, ytr = [X[i] for i in tr], [y[i] for i in tr]
        xte, yte = [X[i] for i in split[k]], [y[i] for i in split[k]]
        fit_predict = make_fit_predict()
        t0 = time.perf_counter()
        predict = fit_predict(xtr, ytr)
        t1 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ExtrapolationWarning)
            pred = predict(xte)
        t2 = time.perf_counter()
        warned += bool(caught)
        rows.append(rmse(pred, yte))
        fit_times.append(t1 - t0)
        pred_times.append(t2 - t1)
    return {
        "fold_rmse": rows,
        "mean_rmse": statistics.fmean(rows),
        "std_rmse": statistics.stdev(rows),
        "mean_fit_seconds": round(statistics.fmean(fit_times), 3),
        "mean_predict_seconds": round(statistics.fmean(pred_times), 4),
        "folds_with_extrapolation": warned,
    }


def anfis_maker(columns, terms, mode="default", history=None):
    """``mode``: "default"; "scaled", inputs mapped to [0, 1] by the training fold's range;
    "consequents", lr=1e-12, so the premise stays put and only least squares runs."""

    def make():
        def fit_predict(xtr, ytr):
            model = ANFIS(len(columns), terms)
            rows = [[r[c] for c in columns] for r in xtr]
            lo = [min(col) for col in zip(*rows)]
            span = [max(col) - low for col, low in zip(zip(*rows), lo)]

            def prepare(raw):
                picked = [[r[c] for c in columns] for r in raw]
                if mode != "scaled":
                    return picked
                return [[(v - a) / b for v, a, b in zip(r, lo, span)] for r in picked]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                trace = model.fit(prepare(xtr), ytr, epochs=EPOCHS, lr=1e-12 if mode == "consequents" else 0.01)
            if history is not None and not history:
                history.extend(trace)
            return lambda rows: model.predict(prepare(rows))

        return fit_predict

    return make


def sklearn_maker(factory):
    def make():
        def fit_predict(xtr, ytr):
            model = factory().fit(np.array(xtr), np.array(ytr))
            return lambda rows: model.predict(np.array(rows)).tolist()
        return fit_predict
    return make


def run_dataset(name, X, y, names, configs):
    out = []
    for columns, terms in configs:
        label = f"ANFIS {len(columns)} inputs, {terms} terms ({terms ** len(columns)} rules)"
        for mode, suffix in (("default", ""), ("scaled", ", inputs scaled"), ("consequents", ", lr 1e-12")):
            history = []
            result = cross_validate(anfis_maker(columns, terms, mode, history), X, y)
            out.append({"model": label + suffix, "mode": mode, "inputs": [names[c] for c in columns],
                        "train_rmse_history_fold_1": history, **result})
            print(name, label + suffix, result["mean_rmse"], result["mean_fit_seconds"],
                  result["folds_with_extrapolation"], flush=True)
    for label, factory in (
        ("scikit-learn LinearRegression (all inputs)", LinearRegression),
        ("scikit-learn GradientBoostingRegressor (all inputs)", lambda: GradientBoostingRegressor(random_state=0)),
    ):
        result = cross_validate(sklearn_maker(factory), X, y)
        out.append({"model": label, "inputs": names, **result})
        print(name, label, result["mean_rmse"], result["mean_fit_seconds"], flush=True)
    return out


def main() -> None:
    ax, ay, _ = load_airfoil()
    mx, my, dropped = load_auto()
    data = {
        "produced_by": {"script": "benchmarks/anfis_regression.py", "bulanik": bulanik.__version__,
                        "scikit-learn": md.version("scikit-learn"), "numpy": md.version("numpy"),
                        "python": sys.version.split()[0]},
        "setup": {"seed": SEED, "folds": FOLDS, "anfis_epochs": EPOCHS, "anfis_lr": 0.01, "anfis_mf": "gaussian",
                  "diagnostic_variants_added_after_seeing_default_results": ["inputs scaled", "lr 1e-12"]},
        "datasets": {
            "airfoil": {"rows": len(ay), "licence": "CC BY 4.0", "doi": "10.24432/C5VW2C",
                        "results": run_dataset("airfoil", ax, ay, AIRFOIL_NAMES,
                                               [([0, 4, 1], 2), ([0, 4, 1], 3), ([0, 1, 2, 3, 4], 2)])},
            "auto_mpg": {"rows": len(my), "rows_dropped_unknown_horsepower": dropped, "licence": "CC BY 4.0",
                         "doi": "10.24432/C5859H",
                         "results": run_dataset("auto_mpg", mx, my, AUTO_NAMES,
                                                [([3, 2, 5], 2), ([3, 2, 5], 3)])},
        },
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
