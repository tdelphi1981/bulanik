"""Plots of linguistic variables, linguistic scales and interval type-2 sets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes

from .._validate import finite
from ..numbers import QROFN, TFN, NeutrosophicFN, PictureFN, SphericalFN, TrFN
from ..scales import LinguisticScale
from ..sets import LinguisticVariable, MembershipFunction
from ..type2 import IT2LinguisticVariable, IT2MF
from . import numbers as _numbers
from ._style import (
    wash,
    place_point_labels,
    to_display,
    check_count,
    fill_alpha,
    finish_unit_y,
    fmt,
    gaps_at_zero,
    halo,
    new_axes,
    pad_range,
    peak_labels,
    series_colors,
    style_axes,
    themed,
    tokens,
)

_SAMPLES = 601
_BREAKPOINT_KINDS = {
    "triangular", "trapezoidal", "linear_increasing", "linear_decreasing", "scaled_triangular",
}


def _extent(mf: MembershipFunction) -> tuple[float, float]:
    """A finite x range that shows the interesting part of `mf`."""
    p = mf.params
    k = mf.kind
    if k in ("triangular", "trapezoidal", "linear_increasing", "linear_decreasing"):
        return p[0], p[-1]
    if k == "scaled_triangular":
        return p[0], p[2]
    if k == "gaussian":
        return p[0] - 3.5 * p[1], p[0] + 3.5 * p[1]
    if k in ("gaussian_flat_top", "gaussian_min_pair"):
        return p[0] - 3.5 * p[2], p[1] + 3.5 * p[2]
    if k == "bell":
        a, _, c = p
        return c - 3 * abs(a), c + 3 * abs(a)
    slope, center = p  # sigmoid
    return center - 6 / abs(slope), center + 6 / abs(slope)


def _grid(lo: float, hi: float, mfs: Sequence[MembershipFunction]) -> list[float]:
    """Even samples on [lo, hi] plus the breakpoints of piecewise-linear functions."""
    xs = [lo + (hi - lo) * i / (_SAMPLES - 1) for i in range(_SAMPLES)]
    for mf in mfs:
        if mf.kind in _BREAKPOINT_KINDS:
            pts = mf.params[:3] if mf.kind == "scaled_triangular" else mf.params
            xs.extend(x for x in pts if lo <= x <= hi)
    return sorted(set(xs))


def _peak_x(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Centre of the region where `ys` is maximal, and the maximum."""
    top = max(ys)
    idx = [i for i, y in enumerate(ys) if y >= top - 1e-9]
    return (xs[idx[0]] + xs[idx[-1]]) / 2, top


def _universe(universe: Any, owner: str) -> tuple[float, float]:
    try:
        lo, hi = universe
    except (TypeError, ValueError):
        raise ValueError(f"{owner} universe must be a (lo, hi) pair, got {universe!r}") from None
    lo, hi = finite(lo, "universe lo"), finite(hi, "universe hi")
    if not lo < hi:
        raise ValueError(f"{owner} universe needs lo < hi, got ({fmt(lo)}, {fmt(hi)})")
    return lo, hi


def _input_marker(
    ax: Axes, x: float, name: str, degrees: Sequence[tuple[float, str, str]], t: dict[str, Any]
) -> None:
    """Mark a crisp input and its membership degree in every term it touches.

    `degrees` holds `(mu, colour, term)`. Terms with the same degree share one
    nested dot (one ring per term) and one label, e.g. "MP · F  0.5".
    """
    ax.plot([x, x], [0, 1], color=t["text"], lw=1.0, zorder=7, gid="input")
    x0, x1 = ax.get_xlim()
    right = x > x0 + 0.8 * (x1 - x0)  # near the right edge the label goes left of the line
    ax.annotate(f"{name} = {fmt(x)}", (x, 0.02), xytext=(-6 if right else 6, 0), textcoords="offset points",
                ha="right" if right else "left", va="bottom", fontsize=8.5, color=t["text"], zorder=9,
                path_effects=halo(t))
    groups: dict[str, list[tuple[float, str, str]]] = {}
    for d in sorted(degrees, key=lambda d: -d[0]):
        groups.setdefault(fmt(round(d[0], 3)), []).append(d)
    anchors, texts = [], []
    for value, members in groups.items():
        mu = members[0][0]
        for k, (_, c, _) in enumerate(members):
            ax.plot([x], [mu], marker="o", ms=max(3.0, 8.0 - 3.2 * k), mfc=c, mec=t["surface"], mew=1.4,
                    zorder=8 + k * 0.01, gid="degree")
        anchors.append((x, mu))
        texts.append(" · ".join(m[2] for m in members) + f"  {value}")
    ax.figure.draw_without_rendering()
    line = (to_display(ax, (x, 0.0)), to_display(ax, (x, 1.0)))
    side = ("W", "NW", "SW") if right else ("E", "NE", "SE")
    place_point_labels(ax, anchors, texts, t, segments=[line], prefer=[side] * len(anchors), fontsize=8.5,
                       color=t["text"])


@themed
def variable(
    var_or_scale: LinguisticVariable | LinguisticScale | IT2LinguisticVariable,
    *,
    x: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Every term of a linguistic variable or scale, direct-labelled.

    - A [`LinguisticVariable`][bulanik.sets.LinguisticVariable], or a
      [`LinguisticScale`][bulanik.scales.LinguisticScale] of `TFN` / `TrFN`
      terms: all membership functions on one axis with soft fills, each named
      above its peak. Up to 8 terms take the categorical palette in order; longer
      scales (Saaty's 1-9) take an ordered blue ramp. With `x`, a crisp input is
      marked and its membership degree in every term it touches is shown.
    - A scale of orthopair, picture, spherical or neutrosophic terms is drawn by
      [`orthopair`][bulanik.plot.orthopair], [`picture`][bulanik.plot.picture],
      [`spherical`][bulanik.plot.spherical] or
      [`neutrosophic`][bulanik.plot.neutrosophic], labelled with the term names.
    - An [`IT2LinguisticVariable`][bulanik.type2.IT2LinguisticVariable] is drawn
      by [`it2_set`][bulanik.plot.it2_set].

    Args:
        var_or_scale: The variable or scale.
        x: A crisp input to mark (type-1 variables and `TFN`/`TrFN` scales only).
        ax: Axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If `var_or_scale` is none of the supported types.
        ValueError: If `x` is outside the universe, or given for a scale or
            variable that has no crisp axis.

    Examples:
        >>> from bulanik import plot
        >>> from bulanik.scales import EXAMPLE_TFN_7
        >>> ax = plot.variable(EXAMPLE_TFN_7, x=4)
        >>> type(ax).__name__
        'Axes'
    """
    owner = "variable"
    if isinstance(var_or_scale, LinguisticScale):
        scale = var_or_scale
        terms = list(scale.terms)
        nums = [scale[s] for s in terms]
        kind = scale.number_type
        if not issubclass(kind, (TFN, TrFN)):
            if x is not None:
                raise ValueError(f"{owner} takes a crisp input only for TFN/TrFN scales, not {kind.__name__}")
            if issubclass(kind, QROFN):
                return _numbers.orthopair(nums, labels=terms, ax=ax)
            if issubclass(kind, PictureFN):
                return _numbers.picture(nums, labels=terms, ax=ax)
            if issubclass(kind, SphericalFN):
                return _numbers.spherical(nums, labels=terms, ax=ax)
            if issubclass(kind, NeutrosophicFN):
                return _numbers.neutrosophic(nums, labels=terms, ax=ax)
            raise TypeError(f"{owner} cannot draw a scale of {kind.__name__}")  # pragma: no cover
        lo = min(n._pts()[0] for n in nums)
        hi = max(n._pts()[-1] for n in nums)
        t = tokens()
        ax = new_axes(ax, owner)
        style_axes(ax, t)
        colors = series_colors(len(nums), t, ordered=len(nums) > 8)
        ax.set_xlim(*pad_range(lo, hi, 0.03))
        _numbers._draw_piecewise(ax, nums, colors, t, True)
        ytop = peak_labels(ax, [(_numbers._peak(n), 1.0, s) for n, s in zip(nums, terms)], t)
        if x is not None:
            x = finite(x, "x")
            if not lo <= x <= hi:
                raise ValueError(f"{owner} x={fmt(x)} is outside the scale range [{fmt(lo)}, {fmt(hi)}]")
            degrees = [(n.mu(x), c, s) for n, c, s in zip(nums, colors, terms) if n.mu(x) > 0]
        finish_unit_y(ax, t, ytop)
        ax.set_xlabel(scale.name)
        if x is not None:
            _input_marker(ax, x, "x", degrees, t)
        return ax
    if isinstance(var_or_scale, IT2LinguisticVariable):
        if x is not None:
            raise ValueError(f"{owner} takes a crisp input only for type-1 variables")
        return it2_set(var_or_scale, ax=ax)
    if not isinstance(var_or_scale, LinguisticVariable):
        raise TypeError(
            f"{owner} needs a LinguisticVariable, LinguisticScale or IT2LinguisticVariable, "
            f"got {type(var_or_scale).__name__}"
        )
    v = var_or_scale
    lo, hi = v.universe
    mfs = [v[s] for s in v.term_names]
    t = tokens()
    ax = new_axes(ax, owner)
    style_axes(ax, t)
    ax.set_xlim(lo, hi)
    colors = series_colors(len(mfs), t, ordered=len(mfs) > 8)
    xs = _grid(lo, hi, mfs)
    peaks = []
    for name, mf, c in zip(v.term_names, mfs, colors):
        ys = [mf(xx) for xx in xs]
        ax.fill_between(xs, ys, 0, color=c, alpha=fill_alpha(t, len(mfs), c), linewidth=0, gid="fill", zorder=2)
        ax.plot(xs, gaps_at_zero(ys), color=c, gid="membership", zorder=4, clip_on=False)
        px, py = _peak_x(xs, ys)
        peaks.append((px, py, name))
    ytop = peak_labels(ax, peaks, t)
    if x is not None:
        x = finite(x, "x")
        if not lo <= x <= hi:
            raise ValueError(f"{owner} x={fmt(x)} is outside the universe [{fmt(lo)}, {fmt(hi)}] of '{v.name}'")
        degrees = [(mf(x), c, s) for mf, c, s in zip(mfs, colors, v.term_names) if mf(x) > 0]
    finish_unit_y(ax, t, ytop)
    ax.set_xlabel(v.name)
    if x is not None:
        _input_marker(ax, x, v.name, degrees, t)
    return ax


@themed
def it2_set(
    mf_or_term: IT2MF | IT2LinguisticVariable,
    *,
    universe: tuple[float, float] | None = None,
    label: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Footprint of uncertainty of interval type-2 sets.

    The region between the lower $\underline{\mu}$ and upper $\overline{\mu}$
    membership functions (the footprint of uncertainty, FOU) is shaded; the upper
    bound is a full line, the lower bound a thinner one. Given an
    [`IT2LinguisticVariable`][bulanik.type2.IT2LinguisticVariable], every term is
    drawn and named above its peak.

    Args:
        mf_or_term: An [`IT2MF`][bulanik.type2.IT2MF] or an
            `IT2LinguisticVariable`.
        universe: The `(lo, hi)` x range; default: the variable's universe, or
            for a single `IT2MF` a range that covers it (e.g. 3.5 sigma around a
            Gaussian).
        label: Name written above a single set's peak.
        ax: Axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If `mf_or_term` is neither an `IT2MF` nor an
            `IT2LinguisticVariable`.
        ValueError: If `universe` is not a pair with `lo < hi`, or the variable
            has more than 8 terms.

    Examples:
        >>> from bulanik import plot
        >>> from bulanik.type2 import gaussian_uncertain_mean
        >>> ax = plot.it2_set(gaussian_uncertain_mean(4, 6, 1), label="medium")
        >>> type(ax).__name__
        'Axes'
    """
    owner = "it2_set"
    if isinstance(mf_or_term, IT2MF):
        names = [label] if label is not None else [None]
        sets = [mf_or_term]
        if universe is None:
            spans = [_extent(mf_or_term.upper), _extent(mf_or_term.lower)]
            lo, hi = pad_range(min(s[0] for s in spans), max(s[1] for s in spans), 0.04)
        else:
            lo, hi = _universe(universe, owner)
        xlabel = "$x$"
    elif isinstance(mf_or_term, IT2LinguisticVariable):
        names = list(mf_or_term.term_names)
        sets = [mf_or_term[s] for s in names]
        lo, hi = mf_or_term.universe if universe is None else _universe(universe, owner)
        xlabel = mf_or_term.name
    else:
        raise TypeError(f"{owner} needs an IT2MF or an IT2LinguisticVariable, got {type(mf_or_term).__name__}")
    check_count(len(sets), owner, "sets")
    t = tokens()
    ax = new_axes(ax, owner)
    style_axes(ax, t)
    ax.set_xlim(lo, hi)
    colors = series_colors(len(sets), t)
    xs = _grid(lo, hi, [m for s in sets for m in (s.upper, s.lower)])
    peaks = []
    for name, s, c in zip(names, sets, colors):
        pairs = [s(xx) for xx in xs]
        lower = [p[0] for p in pairs]
        upper = [p[1] for p in pairs]
        ax.fill_between(xs, lower, upper, color=c, alpha=wash(t, c, t["fill_alpha"] * 1.4), linewidth=0, gid="fou", zorder=2)
        ax.plot(xs, gaps_at_zero(upper), color=c, gid="upper", zorder=4, clip_on=False)
        ax.plot(xs, gaps_at_zero(lower), color=c, lw=1.1, gid="lower", zorder=4, clip_on=False)
        if name is not None:
            px, py = _peak_x(xs, upper)
            peaks.append((px, py, name))
    ytop = peak_labels(ax, peaks, t) if peaks else 1.08
    finish_unit_y(ax, t, ytop)
    ax.set_xlabel(xlabel)
    return ax


__all__ = ["it2_set", "variable"]
