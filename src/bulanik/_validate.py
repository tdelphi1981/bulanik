"""Shared validation helpers (private).

Argument checking used across the package: finite numbers, values in
$[0, 1]$ (with a tolerance of $10^{-12}$ for rounding), method-name resolution
and weight vectors. Nothing here is public API; the helpers define the error
messages that the public functions raise.
"""

from __future__ import annotations

import math
from typing import Sequence

TOL = 1e-12


def _num(v: float) -> str:
    """Short form of ``v`` that still round-trips (``%g`` when exact, else repr)."""
    short = f"{v:g}"
    return short if float(short) == v else repr(v)


def finite(x: float, name: str) -> float:
    """Return ``x`` as a float; raise ValueError on NaN or infinity."""
    if isinstance(x, (bool, str)):
        raise TypeError(f"{name} must be a number, got {type(x).__name__} {x!r}")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x}")
    return x


def unit(x: float, name: str) -> float:
    """Return ``x`` clamped into [0, 1] when it is outside by at most TOL."""
    x = finite(x, name)
    if x < 0.0:
        if x < -TOL:
            raise ValueError(f"{name} must be in [0, 1], got {x}")
        return 0.0
    if x > 1.0:
        if x > 1.0 + TOL:
            raise ValueError(f"{name} must be in [0, 1], got {x}")
        return 1.0
    return x


def choose(
    name: str | None,
    valid: Sequence[str],
    what: str,
    owner: str,
    default: str | None = None,
) -> str:
    """Resolve a method name, defaulting when None; list valid names on error."""
    if name is None:
        if default is None:
            raise ValueError(f"no {what} given for {owner}; valid: {', '.join(valid)}")
        return default
    if name not in valid:
        raise ValueError(
            f"unknown {what} '{name}' for {owner}; valid: {', '.join(valid)}"
        )
    return name


def check_weights(
    weights: Sequence[float] | None, n: int, normalize: bool = False
) -> list[float]:
    """Validate ``n`` weights (None means equal weights 1/n).

    Without ``normalize`` the weights must sum to 1 within 1e-9; the weights
    are returned as given (callers such as :func:`bulanik.aggregate.wa` rescale
    them to sum to exactly 1). With ``normalize`` they are divided by their sum.
    """
    if n < 1:
        raise ValueError("need at least one weight")
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"expected {n} weights, got {len(weights)}")
    w = [finite(x, "weight") for x in weights]
    if any(x < 0 for x in w):
        raise ValueError("weights must be non-negative")
    total = sum(w)
    if not math.isfinite(total):  # each weight is finite but their sum overflowed
        raise ValueError("weights are too large: their sum overflows a float")
    if normalize:
        if total <= 0:
            raise ValueError("weights must not all be zero")
        return [x / total for x in w]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1, got {total}")
    return w
