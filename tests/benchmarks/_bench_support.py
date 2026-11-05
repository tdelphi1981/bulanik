"""Pure-Python helpers shared by the offline benchmark tests and ``benchmarks/``.

Standard library only, so the offline tests stay free of third-party imports.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def average_ranks(values: Sequence[float], descending: bool = True, tie_tol: float = 1e-9) -> list[float]:
    """Ranks starting at 1; values within ``tie_tol`` of the next one share their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=descending)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and abs(values[order[stop]] - values[order[stop - 1]]) <= tie_tol:
            stop += 1
        for k in range(start, stop):
            ranks[order[k]] = (start + 1 + stop) / 2
        start = stop
    return ranks


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation; ``nan`` if a side is constant."""
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = sum((x - ma) ** 2 for x in a)
    sb = sum((x - mb) ** 2 for x in b)
    if sa == 0 or sb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(sa * sb)


def spearman(a: Sequence[float], b: Sequence[float], tie_tol: float = 1e-9) -> float:
    """Spearman rank correlation with average ranks for ties."""
    return pearson(average_ranks(a, True, tie_tol), average_ranks(b, True, tie_tol))


def mackey_glass(
    n_steps: int, tau: float = 17.0, dt: float = 0.1, x0: float = 1.2,
    beta: float = 0.2, gamma: float = 0.1, power: float = 10.0,
) -> list[float]:
    """Mackey-Glass series at ``t = k * dt`` by RK4, with ``x(t) = 0`` for ``t < 0``.

    $dx/dt = \\beta x(t-\\tau) / (1 + x(t-\\tau)^{p}) - \\gamma x(t)$. The delayed value at
    an RK4 half step is the mean of its two neighbouring grid values (linear interpolation).
    """
    lag = round(tau / dt)
    xs = [x0]

    def delayed(k: int) -> float:
        return xs[k - lag] if k >= lag else 0.0

    def rhs(x: float, xd: float) -> float:
        return beta * xd / (1.0 + xd**power) - gamma * x

    for k in range(n_steps):
        d0, d1 = delayed(k), delayed(k + 1)
        dm = (d0 + d1) / 2.0
        x = xs[k]
        k1 = rhs(x, d0)
        k2 = rhs(x + dt / 2 * k1, dm)
        k3 = rhs(x + dt / 2 * k2, dm)
        k4 = rhs(x + dt * k3, d1)
        xs.append(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return xs


def mackey_glass_dataset(
    n_train: int = 500, n_test: int = 500, first_t: int = 118,
) -> tuple[list[list[float]], list[float], list[list[float]], list[float]]:
    """Rows ``[x(t-18), x(t-12), x(t-6), x(t)]`` with target ``x(t+6)``, at ``t = first_t, first_t + 1, ...``."""
    dt = 0.1
    last = first_t + n_train + n_test - 1 + 6
    xs = mackey_glass(round(last / dt), dt=dt)
    at = lambda t: xs[round(t / dt)]  # noqa: E731
    rows, target = [], []
    for t in range(first_t, first_t + n_train + n_test):
        rows.append([at(t - 18), at(t - 12), at(t - 6), at(t)])
        target.append(at(t + 6))
    return rows[:n_train], target[:n_train], rows[n_train:], target[n_train:]
