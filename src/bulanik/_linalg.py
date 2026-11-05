r"""Small pure-Python dense linear algebra: Cholesky solve and ridge least squares.

Private helpers of [`ANFIS`][bulanik.anfis.ANFIS] training (not public API, no
dependency on NumPy): `solve_spd` solves a symmetric positive definite system by
Cholesky factorisation, `normal_equations` forms $A^T A$ and $A^T y$ for a
design matrix, and `lstsq` combines them into a ridge least-squares solve,
$(A^T A + \lambda I)\theta = A^T y$.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["inverse", "lstsq", "normal_equations", "solve_spd"]


def inverse(a: Sequence[Sequence[float]]) -> list[list[float]]:
    """Inverse of a square matrix by Gauss-Jordan elimination with partial pivoting.

    Raises ValueError when ``a`` is not square or not finite, or is singular,
    i.e. a pivot satisfies ``|pivot| <= 1e-12 * max|a_ij|``.
    """
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("inverse needs a non-empty square matrix")
    m = [[float(v) for v in row] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    if not all(math.isfinite(v) for row in m for v in row):
        raise ValueError("inverse needs finite entries")
    scale = max(abs(v) for row in m for v in row[:n])
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) <= 1e-12 * scale:
            raise ValueError("matrix is singular")
        m[c], m[p] = m[p], m[c]
        piv = m[c][c]
        m[c] = [v / piv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0.0:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return [row[n:] for row in m]


def solve_spd(a: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Solve ``a x = b`` for a symmetric positive definite ``a`` (Cholesky).

    Raises ValueError when ``a`` is not square, not symmetric, not finite or
    not (numerically) positive definite, i.e. a Cholesky pivot is <= 0.
    """
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("solve_spd needs a non-empty square matrix")
    if len(b) != n:
        raise ValueError(f"right-hand side has {len(b)} entries, matrix is {n}x{n}")
    m = [[float(v) for v in row] for row in a]
    rhs = [float(v) for v in b]
    if not all(math.isfinite(v) for row in m for v in row) or not all(map(math.isfinite, rhs)):
        raise ValueError("solve_spd needs finite entries")
    for i in range(n):
        for j in range(i):
            if abs(m[i][j] - m[j][i]) > 1e-12 * max(1.0, abs(m[i][j]), abs(m[j][i])):
                raise ValueError("solve_spd needs a symmetric matrix")
    # In-place lower Cholesky factor L (row-major), a = L L^T.
    low = [[0.0] * n for _ in range(n)]
    for j in range(n):
        lj = low[j]
        d = m[j][j] - sum(v * v for v in lj[:j])
        if not d > 0.0:
            raise ValueError("matrix is not positive definite (Cholesky pivot <= 0)")
        ljj = math.sqrt(d)
        lj[j] = ljj
        for i in range(j + 1, n):
            li = low[i]
            li[j] = (m[i][j] - sum(li[k] * lj[k] for k in range(j))) / ljj
    # Forward substitution L z = b, then back substitution L^T x = z.
    z = [0.0] * n
    for i in range(n):
        li = low[i]
        z[i] = (rhs[i] - sum(li[k] * z[k] for k in range(i))) / li[i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - sum(low[k][i] * x[k] for k in range(i + 1, n))) / low[i][i]
    return x


def normal_equations(
    rows: Sequence[Sequence[float]], y: Sequence[float]
) -> tuple[list[list[float]], list[float]]:
    """``(A^T A, A^T y)`` for the design matrix ``A`` given by ``rows``."""
    if not rows:
        raise ValueError("lstsq needs at least one row")
    if len(rows) != len(y):
        raise ValueError(f"lstsq got {len(rows)} rows but {len(y)} targets")
    p = len(rows[0])
    gram = [[0.0] * p for _ in range(p)]
    aty = [0.0] * p
    for row, target in zip(rows, y):
        if len(row) != p:
            raise ValueError("lstsq rows must all have the same length")
        r = [float(v) for v in row]
        t = float(target)
        if not math.isfinite(t) or not all(map(math.isfinite, r)):
            raise ValueError("lstsq needs finite rows and targets")
        for i in range(p):
            ri = r[i]
            if ri == 0.0:
                continue
            aty[i] += ri * t
            gi = gram[i]
            for j in range(i, p):
                gi[j] += ri * r[j]
    for i in range(p):
        for j in range(i):
            gram[i][j] = gram[j][i]
    return gram, aty


def lstsq(
    rows: Sequence[Sequence[float]], y: Sequence[float], ridge: float = 1e-8
) -> list[float]:
    """Ridge least squares: solve ``(A^T A + ridge I) theta = A^T y``.

    A positive ``ridge`` makes rank-deficient designs (a constant column,
    duplicate rows) solvable; the solution then tends to the minimum-norm
    least-squares solution as ``ridge -> 0``. ``ridge=0`` is the plain normal
    equation and raises ValueError when ``A^T A`` is singular.
    """
    ridge = float(ridge)
    if not (math.isfinite(ridge) and ridge >= 0.0):
        raise ValueError(f"ridge must be finite and >= 0, got {ridge}")
    gram, aty = normal_equations(rows, y)
    for i in range(len(gram)):
        gram[i][i] += ridge
    return solve_spd(gram, aty)
