import pytest
from pytest import approx

from bulanik._linalg import lstsq, solve_spd


def test_exact_and_overdetermined():
    assert lstsq([[1, 0], [1, 1], [1, 2]], [1, 3, 5], ridge=0) == approx([1, 2])
    # y=[1,2,2] at x=0,1,2: slope Σ(x-1)(y-5/3)/Σ(x-1)^2 = (2/3+1/3)/2 = .5 ; intercept 5/3-.5 = 7/6
    assert lstsq([[1, 0], [1, 1], [1, 2]], [1, 2, 2], ridge=0) == approx([7 / 6, 0.5])
    assert solve_spd([[4, 2], [2, 3]], [2, 1]) == approx([0.5, 0.0])
    with pytest.raises(ValueError):
        solve_spd([[1, 2], [2, 1]], [1, 1])


def test_solve_spd_three_by_three_golden():
    # A = L L^T with L = [[2,0,0],[1,1,0],[1,1,1]] -> A = [[4,2,2],[2,2,2],[2,2,3]];
    # x = [1,-1,2] gives b = A x = [4-2+4, 2-2+4, 2-2+6] = [6, 4, 6]
    assert solve_spd([[4, 2, 2], [2, 2, 2], [2, 2, 3]], [6, 4, 6]) == approx([1, -1, 2])


def test_solve_spd_rejects_singular_and_bad_shapes():
    # [[1,1],[1,1]]: second pivot 1 - 1*1/1 = 0 exactly -> not positive definite
    with pytest.raises(ValueError, match="positive definite"):
        solve_spd([[1, 1], [1, 1]], [1, 1])
    with pytest.raises(ValueError, match="square"):
        solve_spd([[1, 0]], [1])
    with pytest.raises(ValueError, match="right-hand side"):
        solve_spd([[1, 0], [0, 1]], [1])
    with pytest.raises(ValueError, match="symmetric"):
        solve_spd([[1, 0.5], [0, 1]], [1, 1])
    with pytest.raises(ValueError, match="finite"):
        solve_spd([[float("nan")]], [1])


def test_ridge_handles_constant_duplicate_column():
    # A = [[1,1]]*3, y = 3: A^T A = [[3,3],[3,3]], A^T y = [9,9]; by symmetry
    # theta = (t, t) with (6 + ridge) t = 9 -> t = 1.5 as ridge -> 0 (the minimum-norm solution)
    assert lstsq([[1, 1]] * 3, [3, 3, 3]) == approx([1.5, 1.5], rel=1e-7)
    # without a ridge the normal matrix is exactly singular
    with pytest.raises(ValueError, match="positive definite"):
        lstsq([[1, 1]] * 3, [3, 3, 3], ridge=0)


def test_ridge_handles_duplicate_rows():
    # one distinct row (1, 2) twice, y = 5: theta1 + 2 theta2 = 5 has a line of solutions;
    # the ridge picks the minimum-norm one, 5 (1, 2) / |(1, 2)|^2 = (1, 2)
    assert lstsq([[1, 2], [1, 2]], [5, 5]) == approx([1, 2], rel=1e-7)


def test_ridge_shrinks_toward_zero():
    # one row [1], y = [2]: (1 + ridge) theta = 2 -> theta = 2 / 1.5 with ridge .5
    assert lstsq([[1]], [2], ridge=0.5) == approx([4 / 3])


def test_lstsq_input_errors():
    with pytest.raises(ValueError, match="at least one row"):
        lstsq([], [])
    with pytest.raises(ValueError, match="rows"):
        lstsq([[1], [1]], [1])
    with pytest.raises(ValueError, match="same length"):
        lstsq([[1], [1, 2]], [1, 2])
    with pytest.raises(ValueError, match="ridge"):
        lstsq([[1]], [1], ridge=-1)
    with pytest.raises(ValueError, match="finite"):
        lstsq([[float("inf")]], [1])


def test_inverse():
    from bulanik._linalg import inverse

    assert [x for r in inverse([[2, 1], [1, 1]]) for x in r] == approx([1, -1, -1, 2])
    # needs pivoting: a[0][0] == 0.  A = [[0,2,1],[1,0,0],[3,0,1]], det = -2.
    # checked through A @ inv = I below
    a = [[0, 2, 1], [1, 0, 0], [3, 0, 1]]
    inv = inverse(a)
    for i in range(3):
        for j in range(3):
            assert sum(a[i][k] * inv[k][j] for k in range(3)) == approx(1.0 if i == j else 0.0)
    with pytest.raises(ValueError, match="singular"):
        inverse([[1, 2], [2, 4]])
    with pytest.raises(ValueError, match="square"):
        inverse([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="finite"):
        inverse([[1, float("nan")], [0, 1]])
