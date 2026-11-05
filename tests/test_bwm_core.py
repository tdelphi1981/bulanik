import pytest
from pytest import approx

from bulanik import TFN
from bulanik.mcdm.bwm import bwm_consistency_index, resolve_best_worst
from bulanik.mcdm.results import BWMResult


def test_resolve_best_worst():
    crit, b, w = resolve_best_worst({"C1": 1, "C2": 2, "C3": 4}, {"C1": 4, "C2": 2, "C3": 1})
    assert (crit, b, w) == (["C1", "C2", "C3"], "C1", "C3")
    with pytest.raises(ValueError, match="best="):
        resolve_best_worst({"a": 1, "b": 1, "c": 3}, {"a": 3, "b": 3, "c": 1})
    with pytest.raises(ValueError, match="a_BW"):
        resolve_best_worst({"C1": 1, "C2": 2, "C3": 4}, {"C1": 5, "C2": 2, "C3": 1})


def test_resolve_best_worst_explicit_and_order():
    # ties on BO=1 resolved by best=; criteria order comes from the argument
    crit, b, w = resolve_best_worst(
        {"a": 1, "b": 1, "c": 3}, {"a": 3, "b": 3, "c": 1}, ["c", "b", "a"], best="a"
    )
    assert (crit, b, w) == (["c", "b", "a"], "a", "c")
    with pytest.raises(ValueError, match="worst="):
        resolve_best_worst({"a": 1, "b": 2, "c": 2}, {"a": 2, "b": 1, "c": 1})


def test_resolve_best_worst_errors():
    bo, ow = {"C1": 1, "C2": 2, "C3": 4}, {"C1": 4, "C2": 2, "C3": 1}
    with pytest.raises(ValueError, match="keys"):
        resolve_best_worst(bo, {"C1": 4, "C2": 2})
    with pytest.raises(ValueError, match="keys"):
        resolve_best_worst(bo, ow, ["C1", "C2"])
    with pytest.raises(ValueError, match=">= 1"):
        resolve_best_worst({"C1": 1, "C2": 0.5, "C3": 4}, ow)
    with pytest.raises(ValueError, match="no criterion"):
        resolve_best_worst({"C1": 2, "C2": 2, "C3": 4}, ow)
    with pytest.raises(ValueError, match="unknown best 'X'; valid: C1, C2, C3"):
        resolve_best_worst(bo, ow, best="X")
    with pytest.raises(ValueError, match="must be 1"):
        resolve_best_worst(bo, ow, best="C2")
    with pytest.raises(ValueError, match="differ"):
        resolve_best_worst({"a": 1, "b": 1}, {"a": 1, "b": 1}, best="a", worst="a")
    with pytest.raises(ValueError, match="at least two"):
        resolve_best_worst({"a": 1}, {"a": 1})
    with pytest.raises(ValueError, match="duplicate"):
        resolve_best_worst(bo, ow, ["C1", "C1", "C3"])
    with pytest.raises(TypeError):
        resolve_best_worst({"C1": True, "C2": 2, "C3": 4}, ow)
    with pytest.raises(ValueError, match="finite"):
        resolve_best_worst({"C1": 1, "C2": float("nan"), "C3": 4}, ow)


def test_resolve_best_worst_fuzzy_values():
    # fuzzy values pass through and are compared with ==, for the fuzzy BWM
    one, fi, vi = TFN(1, 1, 1), TFN(1.5, 2, 2.5), TFN(2.5, 3, 3.5)
    crit, b, w = resolve_best_worst({"a": one, "b": fi, "c": vi}, {"a": vi, "b": fi, "c": one}, one=one)
    assert (crit, b, w) == (["a", "b", "c"], "a", "c")


def test_bwm_ci_roots():
    assert bwm_consistency_index(1) == approx(0.0)
    assert bwm_consistency_index(2) == approx((5 - 17**0.5) / 2)  # .438447 (table .44)
    assert bwm_consistency_index(3) == approx(1.0)  # roots of ξ²-7ξ+6
    assert bwm_consistency_index(9) == approx((19 - 73**0.5) / 2)  # 5.227998 (table 5.23)
    with pytest.raises(ValueError):
        bwm_consistency_index(0.5)


def test_bwm_result_is_frozen():
    r = BWMResult("bwm", ("a", "b"), {"a": 0.6, "b": 0.4}, ("a", "b"), 0.0, 0.0, "a", "b", {})
    assert r.fuzzy_weights is None and r.extras == {}
    with pytest.raises(AttributeError):
        r.xi = 1.0  # type: ignore[misc]
