import importlib
import sys

import pytest
from pytest import approx

lumix = pytest.importorskip("lumix")

from lumix import LXConstraint, LXModel, LXOptimizer, LXVariable  # noqa: E402
from lumix.core import LXLinearExpression  # noqa: E402

from bulanik.lumix import (  # noqa: E402
    FuzzyGoal,
    ScoreRecord,
    bwm_linear,
    scores_as_data,
    solve_zimmermann,
    zimmermann,
)
from bulanik.mcdm import ahp, topsis  # noqa: E402


def test_bwm_linear_consistent():
    r = bwm_linear({"C1": 1, "C2": 2, "C3": 4}, {"C1": 4, "C2": 2, "C3": 1})
    assert list(r.weights.values()) == approx([4 / 7, 2 / 7, 1 / 7], abs=1e-7) and r.xi == approx(0, abs=1e-9)
    assert r.consistency_ratio == approx(0, abs=1e-9) and r.best == "C1" and r.worst == "C3"
    assert r.ranking == ("C1", "C2", "C3") and r.method == "bwm_linear"
    assert r.operators == {"model": "linear", "consistency": "xi_L"}
    assert isinstance(r.extras["model"], LXModel)


def test_bwm_linear_inconsistent_golden():
    # |w1-2w2|, |w1-4w3|, |w2-3w3| all tight at ξ: w1-2w2=-ξ, w1-4w3=ξ, w2-3w3=-ξ
    # -> w3=2ξ, w2=5ξ, w1=9ξ, Σ=16ξ=1 -> ξ=1/16, w=(9,5,2)/16   (checked with OR-Tools)
    r = bwm_linear({"C1": 1, "C2": 2, "C3": 4}, {"C1": 4, "C2": 3, "C3": 1})
    assert r.xi == approx(1 / 16, abs=1e-7) and list(r.weights.values()) == approx([9 / 16, 5 / 16, 2 / 16], abs=1e-7)
    assert r.consistency_ratio == approx(1 / 16, abs=1e-7) and r.operators["consistency"] == "xi_L"


def test_bwm_linear_criteria_order_and_rows():
    # same golden as above, criteria listed in another order; weights follow it
    r = bwm_linear({"C1": 1, "C2": 2, "C3": 4}, {"C1": 4, "C2": 3, "C3": 1}, ["C3", "C1", "C2"])
    assert r.criteria == ("C3", "C1", "C2")
    assert list(r.weights.values()) == approx([2 / 16, 9 / 16, 5 / 16], abs=1e-7)
    names = {c.name for c in r.extras["model"].constraints}
    # BO rows for j≠B (C2, C3), OW rows for j≠W (C1, C2), and the sum row
    assert names == {"bo_C2_pos", "bo_C2_neg", "bo_C3_pos", "bo_C3_neg",
                     "ow_C1_pos", "ow_C1_neg", "ow_C2_pos", "ow_C2_neg", "sum_w"}


def _xy_model():
    x = LXVariable[None, float]("x1").continuous().bounds(0, 10)
    return LXModel("m").add_variable(x), x


def test_zimmermann_ge_and_eq_golden():
    m, x = _xy_model()
    goals = [FuzzyGoal("big", LXLinearExpression().add_term(x, 1.0), 6, 2, "ge"),
             FuzzyGoal("three", LXLinearExpression().add_term(x, 1.0), 3, 3, "eq")]
    # max λ: (x-4)/2 = 1-(x-3)/3 -> x=24/5, λ=.4   (checked with OR-Tools: 4.8, 0.4)
    r = solve_zimmermann(m, goals)
    assert r.lambda_ == approx(0.4, abs=1e-7) and r.goal_values["big"] == approx(4.8, abs=1e-7)
    assert r.memberships == approx({"big": 0.4, "three": 0.4}, abs=1e-7)


def test_zimmermann_two_variable_golden():
    # z1=2x1+x2 ≥ 14 (t=4), z2=x1+2x2 ≤ 6 (t=4), x1 ≤ 8 hard.
    # rows: 2x1+x2-4λ ≥ 10 and x1+2x2+4λ ≤ 10 ; 1/3·row1 + 2/3·row2 gives 4λ ≤ 10/3 - x2
    # -> λ* = 5/6 at x2=0, then both rows tight: 2x1-10 = 10-x1 -> x1=20/3 ; z1=40/3, z2=20/3
    x1 = LXVariable[None, float]("x1").continuous().bounds(0, 8)
    x2 = LXVariable[None, float]("x2").continuous().bounds(lower=0)
    m = LXModel("m").add_variable(x1).add_variable(x2)
    r = solve_zimmermann(m, [FuzzyGoal("profit", LXLinearExpression().add_term(x1, 2.0).add_term(x2, 1.0), 14, 4, "ge"),
                             FuzzyGoal("cost", LXLinearExpression().add_term(x1, 1.0).add_term(x2, 2.0), 6, 4, "le")])
    assert r.lambda_ == approx(5 / 6, abs=1e-7) and r.goal_values == approx({"profit": 40 / 3, "cost": 20 / 3}, abs=1e-6)
    # memberships: (40/3-10)/4 = 5/6 and (10-20/3)/4 = 5/6
    assert r.memberships == approx({"profit": 5 / 6, "cost": 5 / 6}, abs=1e-6)


def test_zimmermann_goal_variable_is_free_and_expression_untouched():
    # z = x - 5 with a constant; at x=0 the goal value is -5, below 0 (needs a free goal_ variable)
    # goal le with g=-5, t=1 -> μ=1 at x=0 (x ≥ 0 hard), λ=1
    m, x = _xy_model()
    e = LXLinearExpression().add_term(x, 1.0).add_constant(-5.0)
    before = (dict(e.terms), e.constant)
    r = solve_zimmermann(m, [FuzzyGoal("low", e, -5, 1, "le")])
    assert r.lambda_ == approx(1.0, abs=1e-7) and r.goal_values["low"] == approx(-5.0, abs=1e-7)
    assert (dict(e.terms), e.constant) == before


def test_zimmermann_returns_extendable_model():
    m, x = _xy_model()
    out = zimmermann(m, [FuzzyGoal("g", LXLinearExpression().add_term(x, 1.0), 6, 2, "ge")])
    assert out is m
    out.add_constraint(LXConstraint("cap").expression(LXLinearExpression().add_term(x, 1.0)).le().rhs(5))
    assert LXOptimizer().use_solver("ortools").solve(out).variables["fuzzy_lambda"] == approx(0.5, abs=1e-7)  # (5-4)/2


def test_zimmermann_errors():
    m, x = _xy_model()
    e = LXLinearExpression().add_term(x, 1.0)
    with pytest.raises(ValueError, match="valid: ge, le, eq"):
        FuzzyGoal("g", e, 1, 1, "gt")
    with pytest.raises(ValueError, match="tolerance must be > 0"):
        FuzzyGoal("g", e, 1, 0, "ge")
    with pytest.raises(ValueError, match="finite"):
        FuzzyGoal("g", e, float("inf"), 1, "ge")
    with pytest.raises(TypeError):
        FuzzyGoal("g", "x", 1, 1, "ge")
    with pytest.raises(ValueError, match="duplicate"):
        zimmermann(m, [FuzzyGoal("g", e, 1, 1, "ge")] * 2)
    with pytest.raises(ValueError, match="at least one"):
        zimmermann(m, [])
    assert m.constraints == [] and len(m.variables) == 1  # failed calls leave the model alone
    m2, x2 = _xy_model()  # x ≤ 10 cannot reach 20-1
    with pytest.raises(RuntimeError, match="status"):
        solve_zimmermann(m2, [FuzzyGoal("g", LXLinearExpression().add_term(x2, 1.0), 20, 1, "ge")])


def test_zimmermann_name_collision_leaves_model_unchanged():
    # goal_g already exists in the user's model: nothing may be added before the error
    m, x = _xy_model()
    m.add_variable(LXVariable[None, float]("goal_g").continuous())
    goal = FuzzyGoal("g", LXLinearExpression().add_term(x, 1.0), 6, 2, "ge")
    with pytest.raises(ValueError, match="collide.*goal_g"):
        zimmermann(m, [goal])
    assert [v.name for v in m.variables] == ["x1", "goal_g"] and m.constraints == []
    m2, x2 = _xy_model()  # lambda_name clashes with x1
    with pytest.raises(ValueError, match="collide.*x1"):
        zimmermann(m2, [goal], lambda_name="x1")
    assert [v.name for v in m2.variables] == ["x1"] and m2.constraints == []
    with pytest.raises(ValueError, match="collide.*goal_g"):  # λ named like a goal variable
        zimmermann(m2, [goal], lambda_name="goal_g")
    assert [v.name for v in m2.variables] == ["x1"] and m2.constraints == []


def test_scores_as_data():
    recs = scores_as_data(topsis([[7, 9], [8, 7], [9, 8]], [0.6, 0.4], ["benefit", "cost"]))
    assert [(r.key, r.rank) for r in recs] == [("A1", 3), ("A2", 2), ("A3", 1)] and recs[1].value == approx(0.625)
    assert [r.rank for r in scores_as_data({"x": 0.2, "y": 0.8})] == [2, 1]


def test_scores_as_data_weighting_and_bwm():
    # AHP on a consistent 2x2 (a12 = 3) -> w = (.75, .25)
    w = scores_as_data(ahp([[1, 3], [1 / 3, 1]]))
    assert [(r.key, r.rank) for r in w] == [("C1", 1), ("C2", 2)] and w[0].value == approx(0.75)
    b = scores_as_data(bwm_linear({"C1": 1, "C2": 2, "C3": 4}, {"C1": 4, "C2": 3, "C3": 1}, ["C3", "C1", "C2"]))
    assert [(r.key, r.rank) for r in b] == [("C3", 3), ("C1", 1), ("C2", 2)]
    assert isinstance(b[0], ScoreRecord)
    with pytest.raises(TypeError):
        scores_as_data([0.1, 0.2])
    with pytest.raises(TypeError):
        scores_as_data({"x": True})


def test_bwm_linear_non_optimal_raises(monkeypatch):
    class _Bad:
        status = "infeasible"
        def is_optimal(self):
            return False
    monkeypatch.setattr(LXOptimizer, "solve", lambda self, model: _Bad())
    with pytest.raises(RuntimeError, match="infeasible"):
        bwm_linear({"C1": 1, "C2": 2}, {"C1": 2, "C2": 1})


def test_missing_extra_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "lumix", None)
    monkeypatch.delitem(sys.modules, "bulanik.lumix", raising=False)
    with pytest.raises(ImportError, match=r'pip install "bulanik\[lumix\]"'):
        importlib.import_module("bulanik.lumix")


def test_module_restored_after_missing_extra():
    # runs after test_missing_extra_message: the original module must be back,
    # or a later import would build a second module with its own FuzzyGoal class
    assert sys.modules["bulanik.lumix"].FuzzyGoal is FuzzyGoal


def test_bwm_linear_result_feeds_topsis():
    # consistent comparisons give w = (4/7, 2/7, 1/7); a BWMResult is accepted as weights
    bw = bwm_linear({"C1": 1, "C2": 2, "C3": 4}, {"C1": 4, "C2": 2, "C3": 1})
    m = [[7, 9, 3], [8, 7, 5], [9, 8, 4]]
    types = ["benefit", "cost", "benefit"]
    r = topsis(m, bw, types, criteria=["C3", "C1", "C2"])
    plain = topsis(m, [bw.weights[c] for c in ("C3", "C1", "C2")], types)
    assert list(r.scores.values()) == approx(list(plain.scores.values()))
