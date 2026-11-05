"""Plots of fuzzy numbers: membership functions, alpha-cuts, scores, orthopairs, three-part numbers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from .._validate import choose, unit
from ..numbers import IT2TrFN, IVQROFN, QROFN, TFN, NeutrosophicFN, PictureFN, SphericalFN, TrFN
from ._style import (
    place_point_labels,
    to_display,
    grid3d,
    as_list,
    check_count,
    check_labels,
    fill_alpha,
    finish_unit_y,
    fmt,
    LINE_WIDTH,
    halo,
    new_axes,
    pad_range,
    peak_labels,
    ramp_colors,
    series_colors,
    style_axes,
    themed,
    tokens,
    wash,
)

_PIECEWISE = (TFN, TrFN)
_PIECEWISE_WHAT = "TFN or TrFN numbers"


def _points(n: TFN | TrFN) -> tuple[list[float], list[float]]:
    pts = list(n._pts())
    ys = [0.0, 1.0, 0.0] if len(pts) == 3 else [0.0, 1.0, 1.0, 0.0]
    return pts, ys


def _short(n: Any) -> str:
    vals = n._pts() if isinstance(n, _PIECEWISE) else n.parts
    return "(" + ", ".join(fmt(v) for v in vals) + ")"


def _peak(n: TFN | TrFN) -> float:
    lo, hi = n._core()
    return (lo + hi) / 2


def _draw_piecewise(
    ax: Axes, numbers: Sequence[Any], colors: Sequence[str], t: dict[str, Any], fill: bool
) -> None:
    for n, c in zip(numbers, colors):
        xs, ys = _points(n)
        if fill:
            ax.fill_between(xs, ys, 0, color=c, alpha=fill_alpha(t, len(numbers), c), linewidth=0, gid="fill", zorder=2)
        ax.plot(xs, ys, color=c, gid="membership", zorder=4)


@themed
def membership(
    numbers: TFN | TrFN | Sequence[TFN | TrFN],
    *,
    labels: Sequence[str] | None = None,
    alpha_cut: float | None = None,
    fill: bool = True,
    ax: Axes | None = None,
) -> Axes:
    r"""Membership functions of triangular and trapezoidal numbers on one axis.

    Each number is drawn as its exact piecewise-linear membership function with
    a soft translucent fill, one palette slot per number, and is labelled above
    its peak. A single number gets its defining points as x ticks. With
    `alpha_cut`, every number's $\alpha$-cut
    ([`alpha_cut`][bulanik.numbers.TFN.alpha_cut]) is drawn as a band at
    height $\alpha$ with guides down to the axis.

    Args:
        numbers: A [`TFN`][bulanik.numbers.TFN] / [`TrFN`][bulanik.numbers.TrFN]
            or a sequence of up to 8 of them.
        labels: One label per number; `None` labels several numbers with their
            points and leaves a single number unlabelled.
        alpha_cut: Level in $[0, 1]$ of an $\alpha$-cut band to draw, or `None`.
        fill: Shade the area under each curve.
        ax: Axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If a number is not a `TFN` or `TrFN`.
        ValueError: If there are no numbers or more than 8, the label count
            differs from the number count, or `alpha_cut` is outside $[0, 1]$.

    Examples:
        >>> from bulanik import TFN, TrFN, plot
        >>> ax = plot.membership([TFN(1, 2, 4), TrFN(3, 4, 5, 7)], labels=["A", "B"], alpha_cut=0.5)
        >>> type(ax).__name__
        'Axes'
    """
    owner = "membership"
    nums = as_list(numbers, _PIECEWISE, _PIECEWISE_WHAT, owner)
    if not nums:
        raise ValueError(f"{owner} needs at least one number")
    check_count(len(nums), owner)
    labels = check_labels(labels, len(nums), owner)
    if alpha_cut is not None:
        alpha_cut = unit(alpha_cut, "alpha_cut")
    if labels is None and len(nums) > 1:
        labels = [_short(n) for n in nums]
    t = tokens()
    ax = new_axes(ax, owner)
    style_axes(ax, t)
    colors = series_colors(len(nums), t)
    lo = min(n._pts()[0] for n in nums)
    hi = max(n._pts()[-1] for n in nums)
    ax.set_xlim(*pad_range(lo, hi))
    _draw_piecewise(ax, nums, colors, t, fill)
    if len(nums) == 1:
        pts = sorted(set(nums[0]._pts()))
        ax.set_xticks(pts)
        ax.set_xticklabels([fmt(p) for p in pts])
    ytop = 1.12
    if labels is not None:
        ytop = peak_labels(ax, [(_peak(n), 1.0, s) for n, s in zip(nums, labels)], t)
    if alpha_cut is not None:
        x0, x1 = ax.get_xlim()
        ax.axhline(alpha_cut, color=t["muted"], lw=0.7, zorder=1)
        ax.text(
            x1, alpha_cut + 0.015, rf"$\alpha$ = {fmt(alpha_cut)}", ha="right", va="bottom",
            fontsize=8.5, color=t["muted"], zorder=6,
        )
        for n, c in zip(nums, colors):
            a, b = n.alpha_cut(alpha_cut)
            for x in (a, b):
                ax.plot([x, x], [0, alpha_cut], color=c, lw=0.8, alpha=0.6, zorder=3)
            ax.plot(
                [a, b], [alpha_cut, alpha_cut], color=c, lw=3.6, solid_capstyle="butt",
                gid="alpha-cut", zorder=5,
            )
            ax.plot(
                [a, b], [alpha_cut, alpha_cut], linestyle="none", marker="o", ms=5.5,
                mfc=c, mec=t["surface"], mew=1.4, zorder=6,
            )
    finish_unit_y(ax, t, ytop)
    ax.set_xlabel("$x$")
    return ax


def _levels(levels: int | Sequence[float], owner: str) -> list[float]:
    if isinstance(levels, bool):
        raise TypeError(f"{owner} levels must be an integer or a sequence of levels, got bool")
    if isinstance(levels, int):
        if levels < 2:
            raise ValueError(f"{owner} levels must be at least 2, got {levels}")
        return [i / (levels - 1) for i in range(levels)]
    out = sorted(unit(a, "alpha level") for a in levels)
    if not out:
        raise ValueError(f"{owner} needs at least one alpha level")
    return out


@themed
def alpha_cuts(
    number: TFN | TrFN,
    *,
    levels: int | Sequence[float] = 11,
    ax: Axes | None = None,
) -> Axes:
    r"""The $\alpha$-cuts of a number as stacked horizontal intervals.

    Every level $\alpha$ is drawn as the interval
    [`alpha_cut(α)`][bulanik.numbers.TFN.alpha_cut] at height $\alpha$, coloured
    from light (support, $\alpha = 0$) to strong (core, $\alpha = 1$), inside the
    outline of the membership function. This is the picture behind interval
    arithmetic on fuzzy numbers: an operation acts on each cut separately.

    Args:
        number: A [`TFN`][bulanik.numbers.TFN] or [`TrFN`][bulanik.numbers.TrFN].
        levels: Number of evenly spaced levels from 0 to 1 (at least 2), or the
            levels themselves.
        ax: Axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If `number` is not a `TFN` or `TrFN`, or `levels` is a bool.
        ValueError: If `levels` is below 2, empty, or holds a level outside
            $[0, 1]$.

    Examples:
        >>> from bulanik import TFN, plot
        >>> ax = plot.alpha_cuts(TFN(1, 2, 4), levels=6)
        >>> type(ax).__name__
        'Axes'
    """
    owner = "alpha_cuts"
    if not isinstance(number, _PIECEWISE):
        raise TypeError(f"{owner} needs a TFN or TrFN, got {type(number).__name__}")
    alphas = _levels(levels, owner)
    t = tokens()
    ax = new_axes(ax, owner)
    style_axes(ax, t, grid=None)
    pts = number._pts()
    ax.set_xlim(*pad_range(pts[0], pts[-1]))
    xs, ys = _points(number)
    ax.fill_between(xs, ys, 0, color=t["series"][0], alpha=t["fill_alpha"] * 0.5, linewidth=0, zorder=1)
    ax.plot(xs, ys, color=t["muted"], lw=1.0, gid="membership", zorder=2)
    colors = ramp_colors(len(alphas), t)
    lw = max(1.6, min(4.0, 26.0 / len(alphas)))
    for a, c in zip(alphas, colors):
        lo, hi = number.alpha_cut(a)
        # unclipped and above the spine, so the alpha = 0 cut on the baseline stays visible
        ax.plot([lo, hi], [a, a], color=c, lw=lw, solid_capstyle="round", gid="alpha-cut", zorder=3,
                clip_on=False)
        if hi - lo <= 1e-12 * max(1.0, abs(hi)):  # a one-point cut (the peak of a triangle)
            ax.plot([lo], [a], marker="o", ms=lw * 1.9, mfc=c, mec=t["surface"], mew=1.2, zorder=4,
                    clip_on=False)
    uniq = sorted(set(pts))
    ax.set_xticks(uniq)
    ax.set_xticklabels([fmt(p) for p in uniq])
    finish_unit_y(ax, t, 1.06, label=r"$\alpha$")
    ax.set_xlabel("$x$")
    return ax


@themed
def defuzzification(
    number: TFN | TrFN,
    *,
    scores: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """A number with markers at the crisp values of its score functions.

    Each score ([`score`][bulanik.numbers.TFN.score]: `"centroid"`,
    `"graded_mean"`, `"mean"`) is a stem from the axis up to the membership
    curve, labelled with its name and value. Scores that coincide (the centroid
    and the mean of a triangle) share one stem.

    Args:
        number: A [`TFN`][bulanik.numbers.TFN] or [`TrFN`][bulanik.numbers.TrFN].
        scores: Score names to show; `None` shows all of them.
        ax: Axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If `number` is not a `TFN` or `TrFN`.
        ValueError: If `scores` is empty or names an unknown score.

    Examples:
        >>> from bulanik import TrFN, plot
        >>> ax = plot.defuzzification(TrFN(1, 2, 3, 6))
        >>> type(ax).__name__
        'Axes'
    """
    owner = "defuzzification"
    if not isinstance(number, _PIECEWISE):
        raise TypeError(f"{owner} needs a TFN or TrFN, got {type(number).__name__}")
    names = list(number.SCORES if scores is None else scores)
    if not names:
        raise ValueError(f"{owner} needs at least one score name")
    names = [choose(s, number.SCORES, "score", type(number).__name__) for s in names]
    groups: list[tuple[float, list[str]]] = []
    for s in dict.fromkeys(names):
        v = number.score(s)
        for g in groups:
            if math.isclose(g[0], v, rel_tol=1e-9, abs_tol=1e-12):
                g[1].append(s)
                break
        else:
            groups.append((v, [s]))
    t = tokens()
    ax = new_axes(ax, owner)
    style_axes(ax, t)
    pts = number._pts()
    ax.set_xlim(*pad_range(pts[0], pts[-1]))
    _draw_piecewise(ax, [number], [t["series"][0]], t, True)
    groups.sort(key=lambda g: g[0])
    marks = t["series"][1:]
    items = []
    for k, (v, members) in enumerate(groups):
        c = marks[k % len(marks)]
        h = number.mu(v)
        ax.plot([v, v], [0, h], color=c, lw=1.5, gid="score", zorder=5)
        ax.plot([v], [h], marker="o", ms=7, mfc=c, mec=t["surface"], mew=1.6, zorder=6)
        name = " = ".join(m.replace("_", " ") for m in members)
        items.append((v, h, c, f"{name}  {fmt(round(v, 4))}"))
    # labels: one row each, above the curve, ordered left to right, with leaders
    n = len(items)
    for k, (v, h, c, text) in enumerate(items):
        ly = 1.12 + 0.13 * (n - 1 - k)
        ax.plot([v, v], [h + 0.03, ly - 0.025], color=c, lw=0.7, alpha=0.7, zorder=3)
        ax.plot([v], [ly], marker="o", ms=3.5, mfc=c, mec=c, zorder=6)
        ax.text(v, ly, "  " + text, ha="left", va="center", fontsize=8.5, color=t["text_secondary"], zorder=6)
    uniq = sorted(set(pts))
    ax.set_xticks(uniq)
    ax.set_xticklabels([fmt(p) for p in uniq])
    finish_unit_y(ax, t, 1.12 + 0.13 * (n - 1) + 0.09)
    ax.set_xlabel("$x$")
    return ax


def _boundary_hit(mu: float, nu: float, q: int) -> float:
    """Step t >= 0 with (mu + t)^q + (nu + t)^q = 1 (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (mu + mid) ** q + (nu + mid) ** q > 1.0:
            hi = mid
        else:
            lo = mid
    return lo


_RUNG_NAMES = {1: "IFN", 2: "PFN", 3: "FFN"}
_LABEL_ANGLES = (45, 40, 50, 35, 55, 30, 60)


def _curve_label_spot(
    q: int, points: Sequence[tuple[float, float]], half: float, clearance: float = 0.06
) -> tuple[float, float, float, tuple[float, float]]:
    """Where to write the name of the boundary mu^q + nu^q = 1.

    The label runs along the curve's tangent, `half` data units either side of
    its anchor, lifted off the curve along the outward normal. The anchor slides
    along the curve (polar angles 30-60 degrees, 45 first) to the first spot
    whose label centreline keeps `clearance` from every point, else the spot
    that keeps the most. Returns the anchor on the curve, the text rotation (the
    tangent, degrees) and the outward unit normal.
    """
    best = None
    for deg in _LABEL_ANGLES:
        th = math.radians(deg)
        ct, st = math.cos(th), math.sin(th)
        rad = (ct**q + st**q) ** (-1.0 / q)
        x, y = rad * ct, rad * st
        gx, gy = x ** (q - 1), y ** (q - 1)  # gradient of mu^q + nu^q (up to the factor q)
        norm = math.hypot(gx, gy)
        nx, ny = gx / norm, gy / norm
        mx, my = x + _LIFT * nx, y + _LIFT * ny  # centreline of the text
        tx, ty = ny, -nx
        gap = min(
            (_dist_to_segment(px, py, (mx - half * tx, my - half * ty), (mx + half * tx, my + half * ty))
             for px, py in points),
            default=math.inf,
        )
        if best is None or gap > best[0]:
            best = (gap, x, y, nx, ny)
        if gap >= clearance:
            best = (gap, x, y, nx, ny)
            break
    _, x, y, nx, ny = best  # type: ignore[misc]
    rot = math.degrees(math.atan2(-nx, ny))  # tangent direction, pointing down-right
    return x, y, rot, (nx, ny)


_LIFT = 0.042  # data units from the curve to the middle of its label


def _text_segments(ax: Axes, text: Any) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Display-space centreline and edges of a rotated, centred text."""
    rot = text.get_rotation()
    text.set_rotation(0)
    bb = text.get_window_extent()
    text.set_rotation(rot)
    cx, cy = to_display(ax, text.get_position())
    tx, ty = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    hw, hh = bb.width / 2, bb.height / 2
    out = []
    for off in (-hh, 0.0, hh):
        ox, oy = -ty * off, tx * off
        out.append(((cx - hw * tx + ox, cy - hw * ty + oy), (cx + hw * tx + ox, cy + hw * ty + oy)))
    return out


def _dist_to_segment(px: float, py: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    f = max(0.0, min(1.0, ((px - a[0]) * vx + (py - a[1]) * vy) / (vx * vx + vy * vy)))
    return math.hypot(px - a[0] - f * vx, py - a[1] - f * vy)


def _rung(q: Any, what: str, owner: str = "orthopair") -> int:
    if isinstance(q, bool) or not isinstance(q, int) or q < 1:
        raise ValueError(f"{owner} {what} must be an integer >= 1, got {q!r}")
    return q


def _region_rungs(nums: Sequence[Any], q: Any, regions: Any, owner: str) -> list[int]:
    """The rungs whose regions are drawn: `regions`, or `(q,)`, or the numbers' rungs, or `(1,)`."""
    if q is not None:
        q = _rung(q, "q", owner)
        if regions is not None:
            raise ValueError(f"{owner} takes q or regions, not both")
    if regions is None:
        return [q] if q is not None else sorted({n.q for n in nums}) or [1]
    rungs = sorted({_rung(r, "region", owner) for r in regions})
    if not rungs:
        raise ValueError(f"{owner} regions must name at least one rung")
    return rungs


def _mu_nu_axes(ax: Axes | None, owner: str, t: dict[str, Any]) -> Axes:
    """The unit square of the mu-nu plane, equal aspect, quarter ticks."""
    ax = new_axes(ax, owner, figsize=(4.8, 4.6))
    style_axes(ax, t, grid="both")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(ticks, ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks(ticks, ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel(r"membership $\mu$")
    ax.set_ylabel(r"non-membership $\nu$")
    return ax


def _draw_regions(ax: Axes, rungs: Sequence[int], t: dict[str, Any], pts: Sequence[tuple[float, float]]) -> list[Text]:
    """Boundary curves, washes and rotated names of the regions of `rungs`.

    The inner (smallest) region is darkest. Each name slides along its curve to
    keep clear of `pts`. Returns the name texts.
    """
    # inner (smallest) region darkest: ramp from strong to light as q grows
    colors = list(reversed(ramp_colors(len(rungs), t)))
    m = 241
    ax.figure.draw_without_rendering()
    char = 0.55 * 8.5 * ax.figure.dpi / 72 / max(ax.get_window_extent().width, 1.0)  # data units per glyph
    curve_labels = []
    for r, c in zip(rungs, colors):
        xs = [i / (m - 1) for i in range(m)]
        ys = [max(0.0, 1.0 - x**r) ** (1.0 / r) for x in xs]
        ax.fill_between(xs, ys, 0, color=c, alpha=0.07 if t["name"] == "light" else 0.1, linewidth=0, zorder=1)
        ax.plot(xs, ys, color=c, lw=1.3, gid="boundary", zorder=2, clip_on=False)
        name = _RUNG_NAMES.get(r)
        text = f"q = {r}" + (f"  {name}" if name else "")
        x, y, rot, (nx, ny) = _curve_label_spot(r, pts, half=len(text) * char / 2)
        curve_labels.append(ax.text(
            x + _LIFT * nx, y + _LIFT * ny, text, rotation=rot, rotation_mode="anchor",
            ha="center", va="center", fontsize=8.5, color=t["muted"], zorder=3, gid="region-label",
        ))
    return curve_labels


@themed
def orthopair(
    numbers: QROFN | Sequence[QROFN],
    *,
    q: int | None = None,
    regions: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    hesitancy: bool = False,
    ax: Axes | None = None,
) -> Axes:
    r"""Orthopair numbers as points in the $\mu$–$\nu$ plane with their feasible regions.

    The region of rung $q$ is $\mu^q + \nu^q \le 1$; each requested rung is drawn
    as its boundary curve with a light wash, so nested regions (IFN $\subset$
    PFN $\subset$ FFN) read as steps of one blue ramp. Numbers are points; with
    `hesitancy=True` each point gets a segment along the diagonal to the boundary
    of its own rung, whose length grows with the hesitancy
    [`π`][bulanik.numbers.QROFN.hesitancy] (a number on its boundary has none).

    Args:
        numbers: A [`QROFN`][bulanik.numbers.QROFN] (`IFN`, `PFN`, `FFN`) or a
            sequence of them; may be empty to draw regions only.
        q: Rung of the one region to draw when `regions` is `None`.
        regions: Rungs whose regions are drawn, e.g. `(1, 2, 3)`. Default: the
            rungs of the numbers, or `(q,)`, or `(1,)`.
        labels: One label per number, written next to its point.
        hesitancy: Draw the hesitancy segments.
        ax: Axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If a number is not an orthopair number.
        ValueError: If `q` or a region is not an integer of at least 1,
            `regions` is empty, both `q` and `regions` are given, or the label
            count differs from the number count.

    Examples:
        >>> from bulanik import IFN, PFN, plot
        >>> ax = plot.orthopair([IFN(0.6, 0.3), PFN(0.7, 0.6)], regions=(1, 2, 3), hesitancy=True)
        >>> type(ax).__name__
        'Axes'
    """
    owner = "orthopair"
    nums = as_list(numbers, (QROFN,), "orthopair numbers (IFN, PFN, FFN, QROFN)", owner)
    labels = check_labels(labels, len(nums), owner)
    rungs = _region_rungs(nums, q, regions, owner)
    t = tokens()
    ax = _mu_nu_axes(ax, owner, t)
    pts = [(n.mu, n.nu) for n in nums]
    curve_labels = _draw_regions(ax, rungs, t, pts)
    # points take slot 2 so they stand out from the blue ramp of the regions
    c = t["series"][1]
    segments = []
    for n in nums:
        if hesitancy and n.hesitancy > 0:
            step = _boundary_hit(n.mu, n.nu, n.q)
            ax.plot(
                [n.mu, n.mu + step], [n.nu, n.nu + step], color=c, lw=1.2, alpha=0.65,
                solid_capstyle="butt", gid="hesitancy", zorder=4,
            )
            ax.plot([n.mu + step], [n.nu + step], marker="o", ms=3, mfc=c, mec=c, alpha=0.65, zorder=4)
            segments.append(((n.mu, n.nu), (n.mu + step, n.nu + step)))
        ax.scatter([n.mu], [n.nu], s=46, color=c, edgecolors=t["surface"], linewidths=1.5, gid="point", zorder=5)
    if labels is not None:
        ax.figure.draw_without_rendering()
        seg_disp = [(to_display(ax, p0), to_display(ax, p1)) for p0, p1 in segments]
        prefer = [("NW", "SW", "W", "N", "S") if n.mu > 0.72 else ("SE", "NE", "E", "S", "N") for n in nums]
        for lab in curve_labels:  # rotated labels as three parallel segments, not their loose box
            seg_disp.extend(_text_segments(ax, lab))
        place_point_labels(ax, pts, labels, t, segments=seg_disp, prefer=prefer)
    ax.spines["left"].set_bounds(0, 1)
    ax.spines["bottom"].set_bounds(0, 1)
    return ax


_IV_SLOTS = 7  # palette slots 2-8: slot 1 (blue) is the ramp of the regions


@themed
def iv_orthopair(
    numbers: IVQROFN | Sequence[IVQROFN],
    *,
    q: int | None = None,
    regions: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Interval-valued orthopair numbers as rectangles in the $\mu$–$\nu$ plane.

    An [`IVQROFN`][bulanik.numbers.IVQROFN] $([\mu_L, \mu_U], [\nu_L, \nu_U])$
    is the rectangle $[\mu_L, \mu_U] \times [\nu_L, \nu_U]$: every orthopair
    $(\mu, \nu)$ it admits. Since $\mu_U^q + \nu_U^q \le 1$, its upper-right
    corner, and so the whole rectangle, lies in the feasible region of rung
    $q$, drawn as in [`orthopair`][bulanik.plot.orthopair]. Each number takes
    the next palette slot from the second on (the first, blue, is the region
    ramp), as a solid outline over a translucent fill. The fill sits on a
    knockout of the surface, so its hue stays clean over the region washes
    (on the dark theme the knockout stays bare: a warm tint over the slate
    surface turns brown); the region boundaries are drawn over it. A degenerate interval still
    shows: a zero-width side is a line and a crisp pair is a dot.

    Args:
        numbers: An `IVQROFN` (`IVIFN`, `IVPFN`, `IVFFN`) or a sequence of at
            most 7; may be empty to draw regions only.
        q: Rung of the one region to draw when `regions` is `None`.
        regions: Rungs whose regions are drawn, e.g. `(1, 2, 3)`. Default: the
            rungs of the numbers, or `(q,)`, or `(1,)`.
        labels: One label per number, written by the upper-right corner of its
            rectangle.
        ax: Axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If a number is not an interval-valued orthopair number.
        ValueError: If `q` or a region is not an integer of at least 1,
            `regions` is empty, both `q` and `regions` are given, there are more
            than 7 numbers, or the label count differs from the number count.

    Examples:
        >>> from bulanik import IVIFN, IVPFN, plot
        >>> ax = plot.iv_orthopair([IVIFN(0.5, 0.6, 0.2, 0.3), IVPFN(0.6, 0.8, 0.3, 0.5)], labels=["a", "b"])
        >>> type(ax).__name__
        'Axes'

    References:
        Atanassov and Gargov (1989), Fuzzy Sets and Systems 31(3), 343-349,
        doi:10.1016/0165-0114(89)90205-4 (interval-valued intuitionistic sets).
    """
    owner = "iv_orthopair"
    nums = as_list(numbers, (IVQROFN,), "interval-valued orthopair numbers (IVIFN, IVPFN, IVFFN, IVQROFN)", owner)
    if len(nums) > _IV_SLOTS:
        raise ValueError(
            f"{owner} draws at most {_IV_SLOTS} numbers (one palette slot each besides the region blue), "
            f"got {len(nums)}; split them over several axes"
        )
    labels = check_labels(labels, len(nums), owner)
    rungs = _region_rungs(nums, q, regions, owner)
    t = tokens()
    ax = _mu_nu_axes(ax, owner, t)
    pts = []  # corners, edge midpoints and centre: the name of a region keeps clear of them
    for n in nums:
        xs = (n.mu_l, (n.mu_l + n.mu_u) / 2, n.mu_u)
        ys = (n.nu_l, (n.nu_l + n.nu_u) / 2, n.nu_u)
        pts.extend((x, y) for x in xs for y in ys)
    curve_labels = _draw_regions(ax, rungs, t, pts)
    colors = list(t["series"][1:1 + len(nums)])
    # the region lines run above the fills (so a rectangle shows where it leaves a region), below the outlines
    for art in [*ax.lines, *curve_labels]:
        if art.get_gid() in ("boundary", "region-label"):
            art.set_zorder(3.5)
    # each fill sits on a knockout of the surface, so its hue is not mixed with the blue region wash
    # on the dark theme any warm tint over the slate turns brown, so the knockout stays bare there
    tint = t["fill_alpha"] if t["name"] == "light" else 0.0
    rects = []
    for n, c in zip(nums, colors):
        corner, w, h = (n.mu_l, n.nu_l), n.mu_u - n.mu_l, n.nu_u - n.nu_l
        ax.add_patch(Rectangle(corner, w, h, facecolor=t["surface"], edgecolor="none", gid="interval-knockout",
                               zorder=3))
        ax.add_patch(Rectangle(corner, w, h, facecolor=to_rgba(c, wash(t, c, tint)), edgecolor="none",
                               gid="interval-fill", zorder=3.1))
        rect = Rectangle(corner, w, h, facecolor="none", edgecolor=c, linewidth=LINE_WIDTH,
                         joinstyle="round", capstyle="round", gid="interval", zorder=4)
        ax.add_patch(rect)
        rects.append(rect)
        if w <= 0 and h <= 0:  # a crisp pair: its outline has no length; a dot as heavy as an outline
            ax.scatter([n.mu_u], [n.nu_u], s=64, color=c, edgecolors=t["surface"], linewidths=1.5,
                       gid="degenerate", zorder=5)
    if labels is not None:
        ax.figure.draw_without_rendering()
        grow = LINE_WIDTH * ax.figure.dpi / 72
        boxes = [r.get_window_extent().expanded(1.0, 1.0) for r in rects]
        boxes = [Bbox.from_extents(b.x0 - grow, b.y0 - grow, b.x1 + grow, b.y1 + grow) for b in boxes]
        seg_disp = []
        for line in ax.lines:  # the boundary curves, so no label sits on a line
            if line.get_gid() == "boundary":
                cx, cy = line.get_data()
                curve = [to_display(ax, (cx[i], cy[i])) for i in range(0, len(cx), 8)]
                seg_disp.extend(zip(curve, curve[1:]))
        for lab in curve_labels:
            seg_disp.extend(_text_segments(ax, lab))
        prefer = [("NW", "N", "W", "SW", "NE") if n.mu_u > 0.72 else ("NE", "N", "E", "SE", "NW") for n in nums]
        anchors = [(n.mu_u, n.nu_u) for n in nums]
        place_point_labels(ax, anchors, labels, t, segments=seg_disp, avoid=boxes, prefer=prefer,
                           point_radius_pt=3.0)
    ax.spines["left"].set_bounds(0, 1)
    ax.spines["bottom"].set_bounds(0, 1)
    return ax


@themed
def it2_number(
    numbers: IT2TrFN | Sequence[IT2TrFN],
    *,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Interval type-2 trapezoidal numbers: upper and lower trapezoids and the footprint between them.

    Each [`IT2TrFN`][bulanik.numbers.IT2TrFN] is drawn as its upper trapezoid
    $(a^U_1, 0), (a^U_2, h^U_1), (a^U_3, h^U_2), (a^U_4, 0)$ in a full line,
    its lower trapezoid (same form, heights $h^L_1, h^L_2$, which may be below 1)
    in a thinner line, and the footprint of uncertainty, the area between
    them, as a soft wash of the same colour (lighter on the dark theme and when
    many footprints overlap). Numbers take the palette slots in order. Heights
    below 1 get their own ticks on the membership axis.

    Args:
        numbers: An `IT2TrFN` or a sequence of at most 8.
        labels: One label per number, written above the peak of its upper
            trapezoid.
        ax: Axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If a number is not an `IT2TrFN`.
        ValueError: If there are no numbers or more than 8, or the label count
            differs from the number count.

    Examples:
        >>> from bulanik import IT2TrFN, plot
        >>> low = IT2TrFN((0, 0.1, 0.1, 0.3, 1, 1), (0.05, 0.1, 0.1, 0.2, 0.9, 0.9))
        >>> ax = plot.it2_number(low, labels=["low"])
        >>> [line.get_gid() for line in ax.lines]
        ['upper', 'lower']

    References:
        Mendel, John and Liu (2006), IEEE Transactions on Fuzzy Systems 14(6),
        808-821, doi:10.1109/TFUZZ.2006.879986 (footprint of uncertainty).
        Chen and Lee (2010), Expert Systems with Applications 37(1), 824-833,
        doi:10.1016/j.eswa.2009.06.094 (the trapezoidal form).
    """
    owner = "it2_number"
    nums = as_list(numbers, (IT2TrFN,), "IT2TrFN numbers", owner)
    if not nums:
        raise ValueError(f"{owner} needs at least one number")
    check_count(len(nums), owner)
    labels = check_labels(labels, len(nums), owner)
    t = tokens()
    ax = new_axes(ax, owner)
    style_axes(ax, t)
    lo, hi = pad_range(min(n.upper[0] for n in nums), max(n.upper[3] for n in nums), 0.04)
    ax.set_xlim(lo, hi)
    def fou_alpha(c: str) -> float:  # the wash of it2_set, lighter when many footprints overlap
        dark = 1.0 if t["name"] == "light" else 0.5  # overlapping washes turn brown on the slate surface
        return wash(t, c, t["fill_alpha"] * 1.4 * dark * min(1.0, 3.0 / len(nums)))

    peaks = []
    for i, (n, c) in enumerate(zip(nums, series_colors(len(nums), t))):
        a1, a2, a3, a4, h1, h2 = n.upper
        b1, b2, b3, b4, g1, g2 = n.lower
        up = [(a1, 0.0), (a2, h1), (a3, h2), (a4, 0.0)]
        low = [(b1, 0.0), (b2, g1), (b3, g2), (b4, 0.0)]
        ax.add_patch(Polygon(up + low[::-1], closed=True, facecolor=c, alpha=fou_alpha(c),
                             edgecolor="none", gid="fou", zorder=2))
        ax.plot([p[0] for p in up], [p[1] for p in up], color=c, gid="upper", zorder=4, clip_on=False)
        ax.plot([p[0] for p in low], [p[1] for p in low], color=c, lw=1.1, gid="lower", zorder=4, clip_on=False)
        if labels is not None:
            peaks.append(((a2 + a3) / 2, max(h1, h2), labels[i]))
    ytop = peak_labels(ax, peaks, t) if peaks else 1.08
    finish_unit_y(ax, t, ytop)
    # heights below 1 (a few at most) get their own ticks, so h^U and h^L read off the axis
    ticks = [0.0, 1.0]
    for h in sorted({h for n in nums for h in (*n.upper[4:], *n.lower[4:])}):
        if all(abs(h - k) >= 0.06 for k in ticks):
            ticks.append(h)
    if all(abs(0.5 - k) >= 0.15 for k in ticks):
        ticks.append(0.5)
    if len(ticks) <= 6:
        ticks.sort()
        ax.set_yticks(ticks, [fmt(k) for k in ticks])
    ax.set_xlabel("$x$")
    return ax


_SQ3 = math.sqrt(3) / 2
_V_MU, _V_ETA, _V_NU = (0.0, 0.0), (0.5, _SQ3), (1.0, 0.0)


def _bary(mu: float, eta: float, nu: float) -> tuple[float, float]:
    return (mu * _V_MU[0] + eta * _V_ETA[0] + nu * _V_NU[0], mu * _V_MU[1] + eta * _V_ETA[1] + nu * _V_NU[1])


@themed
def picture(
    numbers: PictureFN | Sequence[PictureFN],
    *,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Picture fuzzy numbers on the $\mu$–$\eta$–$\nu$ ternary simplex.

    A [`PictureFN`][bulanik.numbers.PictureFN] $(\mu, \eta, \nu)$ has
    $\mu + \eta + \nu \le 1$; the rest $\rho = 1 - \mu - \eta - \nu$ is the
    refusal degree. The point sits at the barycentric position of
    $(\mu, \eta, \nu)$ normalised to sum 1 (its balance between positive,
    neutral and negative membership), and a translucent halo whose radius grows
    with $\rho$ shows how much is refused.

    Args:
        numbers: A `PictureFN` or a sequence of them (at least one).
        labels: One label per number, written next to its point.
        ax: Axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib `Axes` (its frame is hidden).

    Raises:
        TypeError: If a number is not a `PictureFN`.
        ValueError: If there are no numbers, a number has all three parts 0,
            or the label count differs from the number count.

    Examples:
        >>> from bulanik import PictureFN, plot
        >>> ax = plot.picture([PictureFN(0.5, 0.2, 0.1), PictureFN(0.2, 0.3, 0.4)], labels=["p", "r"])
        >>> type(ax).__name__
        'Axes'
    """
    owner = "picture"
    nums = as_list(numbers, (PictureFN,), "PictureFN numbers", owner)
    if not nums:
        raise ValueError(f"{owner} needs at least one number")
    labels = check_labels(labels, len(nums), owner)
    for n in nums:
        if n.mu + n.eta + n.nu == 0:
            raise ValueError(f"{owner} cannot place {n!r}: all three parts are 0 (pure refusal)")
    t = tokens()
    ax = new_axes(ax, owner, figsize=(5.0, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")
    tri = Polygon([_V_MU, _V_ETA, _V_NU], closed=True, facecolor=t["series"][0], alpha=0.05,
                  edgecolor="none", gid="simplex", zorder=0)
    ax.add_patch(tri)
    edge = [_V_MU, _V_ETA, _V_NU, _V_MU]
    ax.plot([p[0] for p in edge], [p[1] for p in edge], color=t["axis"], lw=0.9, zorder=2)
    for k in range(1, 5):
        f = k / 5
        # each isoline: (end on the labelled edge, other end); the tick extends it past the edge
        for end, other in (
            (_bary(f, 0, 1 - f), _bary(f, 1 - f, 0)),  # constant mu, labelled on the bottom edge
            (_bary(1 - f, f, 0), _bary(0, f, 1 - f)),  # constant eta, labelled on the left edge
            (_bary(0, 1 - f, f), _bary(1 - f, 0, f)),  # constant nu, labelled on the right edge
        ):
            ax.plot([end[0], other[0]], [end[1], other[1]], color=t["grid"], lw=0.6, zorder=1)
            dx, dy = end[0] - other[0], end[1] - other[1]
            norm = math.hypot(dx, dy)
            ux, uy = dx / norm, dy / norm
            ax.plot([end[0], end[0] + 0.03 * ux], [end[1], end[1] + 0.03 * uy], color=t["axis"], lw=0.8,
                    zorder=2, gid="tick")
            ax.text(end[0] + 0.065 * ux, end[1] + 0.065 * uy, fmt(f), ha="center", va="center", fontsize=7.5,
                    color=t["muted"])
    sec = t["text_secondary"]
    ax.text(_V_MU[0], _V_MU[1] - 0.085, r"$\mu$  positive", ha="center", va="top", fontsize=9.5, color=sec)
    ax.text(_V_NU[0], _V_NU[1] - 0.085, r"$\nu$  negative", ha="center", va="top", fontsize=9.5, color=sec)
    ax.text(_V_ETA[0], _V_ETA[1] + 0.04, r"$\eta$  neutral", ha="center", va="bottom", fontsize=9.5, color=sec)
    c = t["series"][0]
    any_refusal = False
    anchors = []
    for n in nums:
        s = n.mu + n.eta + n.nu
        x, y = _bary(n.mu / s, n.eta / s, n.nu / s)
        anchors.append((x, y))
        rho = max(0.0, 1.0 - s)
        if rho > 1e-9:
            any_refusal = True
            ax.add_patch(Circle((x, y), 0.015 + 0.14 * rho, facecolor=c, alpha=0.16, edgecolor="none",
                                gid="refusal", zorder=3))
        ax.scatter([x], [y], s=46, color=c, edgecolors=t["surface"], linewidths=1.5, gid="point", zorder=5)
    if any_refusal:
        ax.text(0.5, -0.27, r"halo radius grows with the refusal $\rho = 1 - \mu - \eta - \nu$",
                ha="center", va="top", fontsize=8, color=t["muted"])
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.33 if any_refusal else -0.22, _SQ3 + 0.12)
    if labels is not None:
        place_point_labels(ax, anchors, labels, t)
    return ax


@themed
def spherical(
    numbers: SphericalFN | Sequence[SphericalFN],
    *,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Spherical fuzzy numbers inside the positive octant of the unit sphere.

    A [`SphericalFN`][bulanik.numbers.SphericalFN] $(\mu, \nu, \pi)$ satisfies
    $\mu^2 + \nu^2 + \pi^2 \le 1$: it is a point of the unit ball's positive
    octant. The sphere patch is drawn translucent with a few parallels and
    meridians; each point has a drop line to the $\mu$–$\nu$ floor to make its
    depth readable.

    Args:
        numbers: A `SphericalFN` or a sequence of them (at least one).
        labels: One label per number.
        ax: A 3-D axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib 3-D `Axes`.

    Raises:
        TypeError: If a number is not a `SphericalFN`.
        ValueError: If there are no numbers, `ax` is not a 3-D axes, or the
            label count differs from the number count.

    Examples:
        >>> from bulanik import SphericalFN, plot
        >>> ax = plot.spherical([SphericalFN(0.5, 0.4, 0.3)], labels=["a"])
        >>> ax.name
        '3d'
    """
    owner = "spherical"
    nums = as_list(numbers, (SphericalFN,), "SphericalFN numbers", owner)
    if not nums:
        raise ValueError(f"{owner} needs at least one number")
    labels = check_labels(labels, len(nums), owner)
    t = tokens()
    ax = new_axes(ax, owner, projection="3d", figsize=(5.2, 5.0))
    _style_3d(ax, t)
    m = 16  # 225 facets: the patch is uniform and unshaded, the outline comes from the arcs below
    us = [i / (m - 1) * math.pi / 2 for i in range(m)]
    import numpy as np

    th, ph = np.meshgrid(us, us)
    X, Y, Z = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
    ax.plot_surface(X, Y, Z, color=t["series"][0], alpha=0.10, linewidth=0, shade=False, zorder=1)
    arc = np.linspace(0, math.pi / 2, 90)
    for k in (0, 1, 2, 3):  # meridians (k=0, 3 are the coordinate-plane arcs)
        p = k * math.pi / 6
        ax.plot(np.sin(arc) * math.cos(p), np.sin(arc) * math.sin(p), np.cos(arc),
                color=t["axis"] if k in (0, 3) else t["grid"], lw=0.9 if k in (0, 3) else 0.7)
    for k in (0, 1, 2):  # parallels (k=0 is the floor arc)
        th0 = math.pi / 2 - k * math.pi / 6
        ax.plot(math.sin(th0) * np.cos(arc), math.sin(th0) * np.sin(arc), np.full_like(arc, math.cos(th0)),
                color=t["axis"] if k == 0 else t["grid"], lw=0.9 if k == 0 else 0.7)
    c = t["series"][0]
    for n in nums:
        ax.plot([n.mu, n.mu], [n.nu, n.nu], [0, n.pi], color=t["muted"], lw=0.8, zorder=3)
        ax.scatter([n.mu], [n.nu], [0], s=10, color=t["muted"], depthshade=False, zorder=3)
        ax.scatter([n.mu], [n.nu], [n.pi], s=48, color=c, edgecolors=t["surface"], linewidths=1.5,
                   depthshade=False, gid="point", zorder=5)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\nu$")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$\pi$", rotation=0)
    if labels is not None:
        # place in the 2-D projection of the fixed view, clear of points and drop lines
        ax.figure.draw_without_rendering()
        anchors = [_projected(ax, (n.mu, n.nu, n.pi)) for n in nums]
        floors = [_projected(ax, (n.mu, n.nu, 0.0)) for n in nums]
        segments = [
            (to_display(ax, a_, "figure fraction"), to_display(ax, f_, "figure fraction"))
            for a_, f_ in zip(anchors, floors)
        ]
        positions = [to_display(ax, a_, "figure fraction") for a_ in anchors]
        follow = [_follow3d(ax, n.parts) for n in nums]
        place_point_labels(ax, anchors, labels, t, coords=follow, segments=segments, positions=positions)
    return ax


def _follow3d(ax: Axes, xyz: tuple[float, float, float]) -> Any:
    """Annotation coordinate system pinned to a 3-D point, re-projected at draw time."""
    from matplotlib.transforms import Bbox
    from mpl_toolkits.mplot3d import proj3d

    def where(renderer: Any) -> Bbox:
        x2, y2, _ = proj3d.proj_transform(*xyz, ax.get_proj())
        x, y = ax.transData.transform((x2, y2))
        return Bbox.from_bounds(x, y, 0, 0)

    return where


def _projected(ax: Axes, xyz: tuple[float, float, float]) -> tuple[float, float]:
    """Figure-fraction position of a 3-D data point in the axes' current view."""
    from mpl_toolkits.mplot3d import proj3d

    x2, y2, _ = proj3d.proj_transform(*xyz, ax.get_proj())
    disp = ax.transData.transform((x2, y2))
    fx, fy = ax.figure.transFigure.inverted().transform(disp)
    return float(fx), float(fy)


def _style_3d(ax: Axes, t: dict[str, Any]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=40)
    ticks = [0, 0.5, 1]
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis.pane.set_edgecolor(t["grid"])
        axis.line.set_color(t["axis"])
        axis.line.set_linewidth(0.8)
        axis.set_ticks(ticks)
        # the nu axis drops its "1": it meets the mu axis's "1" at the front corner
        axis.set_ticklabels(["0", "0.5", "" if axis is ax.yaxis else "1"])
        grid3d(axis, t, ticks=True)
        axis.label.set_color(t["text_secondary"])
    ax.tick_params(colors=t["text_secondary"], labelsize=8.5, pad=1)
    ax.set_facecolor(t["surface"])


_TIF = (("T", "truth"), ("I", "indeterminacy"), ("F", "falsity"))
_SPOKES = tuple(math.pi / 2 + k * 2 * math.pi / 3 for k in range(3))


def _spoke(r: float, k: int) -> tuple[float, float]:
    return r * math.cos(_SPOKES[k]), r * math.sin(_SPOKES[k])


@themed
def neutrosophic(
    numbers: NeutrosophicFN | Sequence[NeutrosophicFN],
    *,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Neutrosophic numbers as $T$/$I$/$F$ radar triangles.

    A [`NeutrosophicFN`][bulanik.numbers.NeutrosophicFN] $(t, i, f)$ has three
    independent parts in $[0, 1]$; each number is a filled triangle on three
    spokes (truth at the top, indeterminacy lower left, falsity lower right)
    over a triangular grid at 0.25 steps. Up to 8 numbers, one palette slot
    each; several numbers (or given labels) are named in a legend.

    Args:
        numbers: A `NeutrosophicFN` or a sequence of up to 8 of them.
        labels: One label per number; `None` uses the parts.
        ax: A 2-D axes to draw into (its frame is hidden); `None` creates a
            new figure.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If a number is not a `NeutrosophicFN`.
        ValueError: If there are no numbers or more than 8, `ax` is not a 2-D
            axes, or the label count differs from the number count.

    Examples:
        >>> from bulanik import NeutrosophicFN, plot
        >>> ax = plot.neutrosophic([NeutrosophicFN(0.7, 0.2, 0.1), NeutrosophicFN(0.3, 0.6, 0.5)])
        >>> type(ax).__name__
        'Axes'
    """
    owner = "neutrosophic"
    nums = as_list(numbers, (NeutrosophicFN,), "NeutrosophicFN numbers", owner)
    if not nums:
        raise ValueError(f"{owner} needs at least one number")
    check_count(len(nums), owner)
    given = check_labels(labels, len(nums), owner)
    labels = given or [_short(n) for n in nums]
    t = tokens()
    ax = new_axes(ax, owner, figsize=(4.8, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")
    for r in (0.25, 0.5, 0.75, 1.0):
        pts = [_spoke(r, k) for k in (0, 1, 2, 0)]
        outer = r == 1.0
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=t["axis"] if outer else t["grid"],
                lw=0.8 if outer else 0.6, zorder=1)
        if r in (0.5, 1.0):  # two level labels on the truth spoke are enough to read the grid
            x, y = _spoke(r, 0)
            ax.text(x + 0.035, y, fmt(r), ha="left", va="center", fontsize=7.5, color=t["muted"], zorder=5,
                    path_effects=halo(t))
    sec = t["text_secondary"]
    for k, (sym, word) in enumerate(_TIF):
        x, y = _spoke(1.0, k)
        ax.plot([0, x], [0, y], color=t["grid"], lw=0.6, zorder=1)
        lx, ly = _spoke(1.07, k)
        ha = "center" if k == 0 else "right" if k == 1 else "left"
        ax.text(lx, ly + (0.02 if k == 0 else -0.04), f"${sym}$  {word}", ha=ha,
                va="bottom" if k == 0 else "top", fontsize=9.5, color=sec)
    colors = series_colors(len(nums), t)
    for n, c, lab in zip(nums, colors, labels):
        pts = [_spoke(r, k) for k, r in enumerate(n.parts)]
        ring = pts + pts[:1]
        ax.fill([p[0] for p in ring], [p[1] for p in ring], color=c, alpha=fill_alpha(t, len(nums), c), linewidth=0, zorder=2)
        ax.plot([p[0] for p in ring], [p[1] for p in ring], color=c, gid="radar", label=lab, zorder=3)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle="none", marker="o", ms=6, mfc=c,
                mec=t["surface"], mew=1.4, zorder=4)
    low = -0.5
    if len(nums) > 1 or given is not None:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=min(len(nums), 3),
                  handlelength=1.4, columnspacing=1.6)
        low = -0.8
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(low - 0.12, 1.22)
    return ax


__all__ = [
    "alpha_cuts",
    "defuzzification",
    "it2_number",
    "iv_orthopair",
    "membership",
    "neutrosophic",
    "orthopair",
    "picture",
    "spherical",
]
