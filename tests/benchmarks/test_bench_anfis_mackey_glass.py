"""A short deterministic ANFIS run on the Mackey-Glass data, against the full benchmark's checkpoints."""

import json
import warnings
from pathlib import Path

import pytest
from _bench_support import mackey_glass_dataset

from bulanik.anfis import ANFIS, ExtrapolationWarning

RESULT = json.loads((Path(__file__).parent / "data" / "anfis_mackey_glass_results.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dataset():
    return mackey_glass_dataset()


def test_series_is_the_benchmark_series(dataset):
    xtr, ytr, xte, yte = dataset
    assert (len(ytr), len(yte)) == (500, 500)
    assert xtr[0][3] == pytest.approx(RESULT["series"]["x_118"], abs=1e-12)
    assert yte[-1] == pytest.approx(RESULT["series"]["target_last"], abs=1e-12)


@pytest.mark.parametrize("mf", ["gaussian", "bell"])
def test_ten_epochs_match_checkpoint(dataset, mf):
    xtr, ytr, xte, yte = dataset
    frozen = next(r for r in RESULT["runs"] if r["mf"] == mf and r["default_lr"])
    point = next(c for c in frozen["checkpoints"] if c["epochs"] == 10)
    model = ANFIS(4, 2, mf=mf)
    assert model.n_rules == 16
    model.fit(xtr, ytr, epochs=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        test = model.rmse(xte, yte)
    assert model.rmse(xtr, ytr) == pytest.approx(point["train_rmse"], abs=1e-8)
    assert test == pytest.approx(point["test_rmse"], abs=1e-8)


def test_premise_gradient_is_the_mean_squared_error_gradient():
    """Checks the flat training curve is a matter of step size, not a wrong gradient."""
    xtr, ytr, _, _ = mackey_glass_dataset(100, 1)
    model = ANFIS(4, 2)
    model.fit(xtr, ytr, epochs=1)
    grad, params = model.premise_gradient(xtr, ytr), model.get_params()

    def mse(values):
        model.set_params(values)
        return model.rmse(xtr, ytr) ** 2

    for j in range(8):
        up, down = list(params), list(params)
        up[j] += 1e-6
        down[j] -= 1e-6
        assert (mse(up) - mse(down)) / 2e-6 == pytest.approx(grad[j], rel=1e-3, abs=1e-9)
