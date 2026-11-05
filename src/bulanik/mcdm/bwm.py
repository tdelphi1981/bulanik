r"""Best-worst method (BWM) helpers shared by the LumiX and sezgi bridges.

The best-worst method (Rezaei 2015) weights $n$ criteria from two vectors of
pairwise comparisons: the *best-to-others* vector $a_{Bj}$ (how much more
important the best criterion $B$ is than $j$) and the *others-to-worst* vector
$a_{jW}$ (how much more important $j$ is than the worst criterion $W$), on a
1-9 scale. The weights solve a small optimisation problem, which needs a
solver, so the models live in [`bulanik.lumix`][bulanik.lumix] (linear) and
[`bulanik.sezgi`][bulanik.sezgi] (nonlinear and fuzzy). This module holds the
solver-free parts: input validation and the consistency indices.

References:
    Rezaei (2015), Omega 53, 49-57; Guo & Zhao (2017), Knowledge-Based Systems
    121, 23-31.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .._validate import finite
from ..numbers import TFN, FuzzyNumber

__all__ = ["resolve_best_worst", "bwm_consistency_index", "fuzzy_bwm_consistency_index"]


def _check_value(v: Any, what: str) -> None:
    """Crisp comparisons must be finite numbers >= 1; fuzzy ones pass through."""
    if isinstance(v, FuzzyNumber):
        return
    x = finite(v, what)
    if x < 1.0:
        raise ValueError(f"{what} must be >= 1, got {x}")


def _pick(
    given: str | None,
    values: Mapping[str, Any],
    criteria: list[str],
    one: Any,
    role: str,
    vector: str,
) -> str:
    if given is not None:
        if given not in values:
            raise ValueError(f"unknown {role} '{given}'; valid: {', '.join(criteria)}")
        if values[given] != one:
            raise ValueError(f"{vector}[{given}] must be {one} for the {role} criterion, "
                             f"got {values[given]}")
        return given
    found = [c for c in criteria if values[c] == one]
    if not found:
        raise ValueError(f"no criterion has {vector} value {one}; cannot infer the {role}")
    if len(found) > 1:
        raise ValueError(f"several criteria have {vector} value {one} ({', '.join(found)}); "
                         f"pass {role}= to choose the {role} criterion")
    return found[0]


def resolve_best_worst(
    best_to_others: Mapping[str, Any],
    others_to_worst: Mapping[str, Any],
    criteria: Sequence[str] | None = None,
    best: str | None = None,
    worst: str | None = None,
    *,
    one: Any = 1.0,
) -> tuple[list[str], str, str]:
    r"""Validate BWM comparison vectors and resolve the best and worst criteria.

    Both mappings must have exactly the keys of `criteria`. If `best` (`worst`)
    is not given it is the unique key whose best-to-others (others-to-worst)
    value equals `one`. The comparison of best over worst must agree in both
    vectors: $a_{BW}$ = `best_to_others[worst]` = `others_to_worst[best]`.

    Args:
        best_to_others: Comparison of the best criterion over each criterion.
        others_to_worst: Comparison of each criterion over the worst criterion.
        criteria: Criterion names, fixing the order. Defaults to the order of
            `best_to_others`.
        best: The best criterion. Inferred from the vector when `None`.
        worst: The worst criterion. Inferred from the vector when `None`.
        one: The "equally important" value, `1.0` for crisp comparisons and
            `TFN(1, 1, 1)` for fuzzy ones.

    Returns:
        A tuple `(criteria, best, worst)`; `criteria` is a list.

    Raises:
        TypeError: If a comparison value is a `bool` or `str` instead of a number.
        ValueError: If criteria are duplicated or fewer than two, a vector's
            keys differ from the criteria, a crisp value is not finite or is
            below 1, the best or worst is unknown, cannot be inferred or is
            ambiguous, best equals worst, or the two $a_{BW}$ values differ.

    Examples:
        >>> from bulanik.mcdm.bwm import resolve_best_worst
        >>> resolve_best_worst({"a": 1, "b": 2, "c": 8}, {"a": 8, "b": 4, "c": 1})
        (['a', 'b', 'c'], 'a', 'c')
    """
    crit = list(best_to_others) if criteria is None else list(criteria)
    if len(set(crit)) != len(crit):
        raise ValueError(f"duplicate criteria in {crit}")
    if len(crit) < 2:
        raise ValueError("BWM needs at least two criteria")
    for vec, name in ((best_to_others, "best_to_others"), (others_to_worst, "others_to_worst")):
        if set(vec) != set(crit):
            raise ValueError(f"{name} keys {sorted(map(str, vec))} must equal the criteria "
                             f"{sorted(map(str, crit))}")
        for c in crit:
            _check_value(vec[c], f"{name}[{c}]")
    b = _pick(best, best_to_others, crit, one, "best", "best_to_others")
    w = _pick(worst, others_to_worst, crit, one, "worst", "others_to_worst")
    if b == w:
        raise ValueError(f"best and worst criteria must differ, both are '{b}'")
    if best_to_others[w] != others_to_worst[b]:
        raise ValueError(f"a_BW mismatch: best_to_others[{w}] = {best_to_others[w]} but "
                         f"others_to_worst[{b}] = {others_to_worst[b]}; they must be equal")
    return crit, b, w


def bwm_consistency_index(a_bw: float) -> float:
    r"""Rezaei's (2015) consistency index for a best-over-worst comparison.

    The index is the largest $\xi$ the model can reach for the given
    $a = a_{BW}$: the root of $\xi^2 - (1 + 2a)\xi + (a^2 - a) = 0$ that is
    at most $a$,

    $$CI(a) = \frac{(1 + 2a) - \sqrt{8a + 1}}{2}.$$

    It is 0, 0.44, 1.00, 1.63, 2.30, 3.00, 3.73, 4.47 and 5.23 for
    $a = 1, \dots, 9$. The consistency ratio of a solution is $\xi / CI(a_{BW})$
    ([`BWMProblem`][bulanik.sezgi.BWMProblem]); it does not exist for $a = 1$.

    Args:
        a_bw: The comparison of the best over the worst criterion, at least 1.

    Returns:
        The index, never negative.

    Raises:
        ValueError: If `a_bw` is not finite or is below 1.

    Examples:
        >>> from bulanik.mcdm.bwm import bwm_consistency_index
        >>> [round(bwm_consistency_index(a), 2) for a in (1, 5, 9)]
        [0.0, 2.3, 5.23]

    References:
        Rezaei (2015), Omega 53, 49-57.
    """
    a = finite(a_bw, "a_bw")
    if a < 1.0:
        raise ValueError(f"a_bw must be >= 1, got {a}")
    return max(0.0, ((1.0 + 2.0 * a) - math.sqrt(8.0 * a + 1.0)) / 2.0)


def fuzzy_bwm_consistency_index(a_bw: TFN) -> float:
    r"""Guo & Zhao's (2017) consistency index for a fuzzy best-over-worst comparison.

    With $u$ the upper bound of $a_{BW}$ it is the *larger* root of
    $\xi^2 - (1 + 2u)\xi + (u^2 - u) = 0$,

    $$CI(u) = \frac{(1 + 2u) + \sqrt{8u + 1}}{2},$$

    which gives 3.00, 3.80, 5.29, 6.69 and 8.04 for the terms EI, WI, FI, VI and
    AI of [`FUZZY_BWM_TFN`][bulanik.scales.FUZZY_BWM_TFN]. Unlike the crisp index
    it is never 0.

    Args:
        a_bw: The fuzzy comparison of the best over the worst criterion.

    Returns:
        The index.

    Raises:
        TypeError: If `a_bw` is not a [`TFN`][bulanik.numbers.TFN].
        ValueError: If `a_bw.u` is below 1.

    Examples:
        >>> from bulanik import TFN
        >>> from bulanik.mcdm.bwm import fuzzy_bwm_consistency_index
        >>> round(fuzzy_bwm_consistency_index(TFN(1, 1, 1)), 2)
        3.0

    References:
        Guo & Zhao (2017), Knowledge-Based Systems 121, 23-31.
    """
    if not isinstance(a_bw, TFN):
        raise TypeError(f"a_bw must be a TFN, got {type(a_bw).__name__}")
    u = a_bw.u
    if u < 1.0:
        raise ValueError(f"a_bw.u must be >= 1, got {u}")
    return ((1.0 + 2.0 * u) + math.sqrt(8.0 * u + 1.0)) / 2.0
