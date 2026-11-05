import pytest

from bulanik._validate import TOL, check_weights, choose, finite, unit


def test_tol_value():
    assert TOL == 1e-12


def test_finite_rejects_nan_and_inf():
    assert finite(2, "x") == 2.0
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="x"):
            finite(bad, "x")


def test_unit_clamps_within_tol():
    assert unit(1 + 5e-13, "mu") == 1.0 and unit(-5e-13, "mu") == 0.0
    assert unit(0.3, "mu") == 0.3


def test_unit_rejects_beyond_tol():
    with pytest.raises(ValueError, match="mu"):
        unit(1 + 1e-9, "mu")
    with pytest.raises(ValueError, match="mu"):
        unit(-1e-9, "mu")


def test_unit_rejects_nan():
    with pytest.raises(ValueError):
        unit(float("nan"), "mu")


def test_choose_lists_valid_names():
    with pytest.raises(
        ValueError, match=r"unknown ops 'foo' for IFN; valid: algebraic, einstein"
    ):
        choose("foo", ("algebraic", "einstein"), "ops", "IFN")


def test_choose_default_and_known_name():
    assert choose(None, ("a", "b"), "ops", "X", default="a") == "a"
    assert choose("b", ("a", "b"), "ops", "X", default="a") == "b"
    with pytest.raises(ValueError, match="valid: a, b"):
        choose(None, ("a", "b"), "ops", "X")


def test_check_weights():
    assert check_weights(None, 4) == [0.25] * 4
    assert check_weights([2, 6], 2, normalize=True) == [0.25, 0.75]
    assert check_weights([0.5, 0.5], 2) == [0.5, 0.5]
    with pytest.raises(ValueError, match="sum to 1"):
        check_weights([0.5, 0.6], 2)
    with pytest.raises(ValueError, match="non-negative"):
        check_weights([1.5, -0.5], 2)
    with pytest.raises(ValueError):
        check_weights([0, 0], 2, normalize=True)
    with pytest.raises(ValueError, match="2"):
        check_weights([1.0], 2)


def test_check_weights_overflowing_sum_message():
    with pytest.raises(ValueError, match="overflow"):
        check_weights([1e308, 1e308], 2)
    with pytest.raises(ValueError, match="overflow"):
        check_weights([1e308, 1e308], 2, normalize=True)
    # a plain wrong sum still says so
    with pytest.raises(ValueError, match="sum to 1, got 1.2"):
        check_weights([0.6, 0.6], 2)
