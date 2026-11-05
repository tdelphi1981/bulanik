"""Plots of MCDM results: rankings, criteria weights and DEMATEL cause-effect diagrams."""

from __future__ import annotations

import math
from typing import Any

from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import offset_copy

from ..mcdm.results import BWMResult, DEMATELResult, RankingResult, WeightingResult
from ._style import fmt, new_axes, pad_range, place_point_labels, style_axes, themed, to_display, tokens


LOWER_IS_BETTER = frozenset({"vikor"})
"""Ranking methods whose smaller score is better (VIKOR's Q)."""


_DISPLAY = {
    "ahp": "AHP", "fuzzy_ahp": "Fuzzy AHP", "swara": "SWARA", "entropy": "Entropy", "bwm": "BWM",
    "bwm_linear": "BWM", "bwm_nonlinear": "BWM", "fuzzy_bwm": "Fuzzy BWM",
    "topsis": "TOPSIS", "vikor": "VIKOR", "waspas": "WASPAS", "edas": "EDAS",
    "fuzzy_topsis": "Fuzzy TOPSIS",
}


def _title(method: str) -> str:
    """Method name for display: `"fuzzy_topsis"` -> `"Fuzzy TOPSIS"`; unknown names are title-cased."""
    return _DISPLAY.get(method) or method.replace("_", " ").title()


def _rows_height(n: int) -> tuple[float, float]:
    return (6.4, max(1.6, 0.42 * n + 0.9))


@themed
def ranking(result: RankingResult, *, ax: Axes | None = None) -> Axes:
    """Alternatives sorted best first, as a dot plot of their scores.

    Rows follow `result.ranking` (best at the top, numbered), so methods where
    a lower score is better (VIKOR) read the same way as the others; the x label
    says which direction is better (VIKOR is known to be lower-is-better; for
    other methods the direction is read off the ranking). The best alternative is
    emphasised in the accent colour; every dot carries its score, the others in
    small muted ink.

    Args:
        result: A [`RankingResult`][bulanik.mcdm.results.RankingResult].
        ax: Axes to draw into; `None` creates a new figure sized to the number
            of alternatives.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If `result` is not a `RankingResult`.

    Examples:
        >>> from bulanik import plot
        >>> from bulanik.mcdm import topsis
        >>> r = topsis([[250, 16], [200, 16], [300, 32]], [0.5, 0.5], ["cost", "benefit"])
        >>> ax = plot.ranking(r)
        >>> type(ax).__name__
        'Axes'
    """
    owner = "ranking"
    if not isinstance(result, RankingResult):
        raise TypeError(f"{owner} needs a RankingResult, got {type(result).__name__}")
    names = list(result.ranking)
    scores = [result.scores[a] for a in names]
    n = len(names)
    t = tokens()
    ax = new_axes(ax, owner, figsize=_rows_height(n))
    style_axes(ax, t, grid=None)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    lo, hi = pad_range(min(scores), max(scores), 0.1)
    hi += 0.08 * (hi - lo)  # room for the value labels right of the dots
    ax.set_xlim(lo, hi)
    ys = [n - 1 - i for i in range(n)]
    for i, (y, s) in enumerate(zip(ys, scores)):
        best = i == 0
        ax.plot([lo, hi], [y, y], color=t["grid"], lw=0.6, zorder=1)
        ax.scatter(
            [s], [y], s=70 if best else 46, color=t["series"][0] if best else t["muted"],
            edgecolors=t["surface"], linewidths=1.6, gid="dot", zorder=4,
        )
        ax.annotate(
            fmt(round(s, 4)), (s, y), xytext=(9 if best else 7, 0), textcoords="offset points", va="center",
            fontsize=9 if best else 8, color=t["text"] if best else t["muted"], zorder=5,
        )
    ax.set_yticks(ys, [f"{i + 1}   {a}" for i, a in enumerate(names)])
    ax.tick_params(axis="y", length=0)
    ax.get_yticklabels()[0].set_color(t["text"])
    ax.set_ylim(-0.7, n - 0.3)
    if result.method in LOWER_IS_BETTER:
        better = "lower"
    else:  # unknown method: read the direction off the ranking (ties read as higher)
        better = "lower" if n > 1 and scores[0] < scores[-1] else "higher"
    ax.set_xlabel(f"{_title(result.method)} score ({better} is better)")
    return ax


@themed
def weights(result: WeightingResult | BWMResult, *, ax: Axes | None = None) -> Axes:
    """Criteria weights as horizontal bars, largest first, value at each tip.

    Bars follow `result.ranking`. If the method reports a consistency ratio
    (AHP, fuzzy AHP, BWM), it is shown in the corner.

    Args:
        result: A [`WeightingResult`][bulanik.mcdm.results.WeightingResult] or
            [`BWMResult`][bulanik.mcdm.results.BWMResult].
        ax: Axes to draw into; `None` creates a new figure sized to the number
            of criteria.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If `result` is neither a `WeightingResult` nor a `BWMResult`.

    Examples:
        >>> from bulanik import plot
        >>> from bulanik.mcdm import ahp
        >>> r = ahp([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]], criteria=["cost", "quality", "risk"])
        >>> ax = plot.weights(r)
        >>> type(ax).__name__
        'Axes'
    """
    owner = "weights"
    if not isinstance(result, (WeightingResult, BWMResult)):
        raise TypeError(f"{owner} needs a WeightingResult or BWMResult, got {type(result).__name__}")
    names = list(result.ranking)
    ws = [result.weights[c] for c in names]
    n = len(names)
    t = tokens()
    ax = new_axes(ax, owner, figsize=_rows_height(n))
    style_axes(ax, t, grid=None)
    ys = [n - 1 - i for i in range(n)]
    bars = ax.barh(ys, ws, height=0.5, color=t["series"][0], zorder=3)
    for b in bars:
        b.set_gid("bar")
    top = max(ws) if ws else 1.0
    for y, w in zip(ys, ws):
        ax.annotate(f"{w:.3f}", (w, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=t["text_secondary"], zorder=4)
    ax.set_xlim(0, top * 1.18)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(ys, names)
    ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.axvline(0, color=t["axis"], lw=0.8, zorder=4)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{_title(result.method)} weights", loc="left", fontsize=10.5)
    if result.consistency_ratio is not None:
        ax.set_title(f"CR = {result.consistency_ratio:.3f}", loc="right", fontsize=8.5, fontweight="normal",
                     color=t["muted"])
    return ax


_DOT = 52  # factor marker area (points^2)
_RING = 1.5  # surface ring around a factor (points)
_BEND = 0.15  # arc3 curvature of a two-way relation; the reverse arrow bends to the other side
_END_GAP = math.radians(48)  # least angle between two arrowheads on one marker's circumference


def _control(p: tuple[float, float], q: tuple[float, float], rad: float) -> tuple[float, float]:
    """Control point of matplotlib's arc3 connection from `p` to `q`."""
    return ((p[0] + q[0]) / 2 + rad * (q[1] - p[1]), (p[1] + q[1]) / 2 - rad * (q[0] - p[0]))


def _spread(angles: list[float], gap: float) -> list[float]:
    """`angles` (radians) moved apart to at least `gap`, each as little as possible, input order kept.

    Close angles are merged into clusters spaced `gap` apart around their mean (the
    1-D label de-overlap), on the circle cut at its widest empty arc.
    """
    n = len(angles)
    if n < 2:
        return list(angles)
    if n * gap >= 2 * math.pi:
        gap = 2 * math.pi / n
    order = sorted(range(n), key=lambda i: angles[i] % (2 * math.pi))
    vals = [angles[i] % (2 * math.pi) for i in order]
    widest = max(range(n), key=lambda k: (vals[(k + 1) % n] - vals[k]) % (2 * math.pi) or 2 * math.pi)
    start = (widest + 1) % n
    seq = [vals[(start + k) % n] for k in range(n)]
    seq = [seq[0] + (v - seq[0]) % (2 * math.pi) for v in seq]  # unwrapped, increasing
    clusters: list[list[float]] = []  # [centre, count]
    for v in seq:
        clusters.append([v, 1])
        while len(clusters) > 1:
            (c1, n1), (c2, n2) = clusters[-2], clusters[-1]
            if c2 - (n2 - 1) / 2 * gap - (c1 + (n1 - 1) / 2 * gap) >= gap:
                break
            clusters[-2:] = [[(c1 * n1 + c2 * n2) / (n1 + n2), n1 + n2]]
    placed = [c + (k - (m - 1) / 2) * gap for c, m in clusters for k in range(m)]
    if placed[0] + 2 * math.pi - placed[-1] < gap:  # the spread wrapped round: space them evenly
        placed = [placed[0] + k * 2 * math.pi / n for k in range(n)]
    out = [0.0] * n
    for k, v in enumerate(placed):
        out[order[(start + k) % n]] = v
    return out


def _arc(
    p: tuple[float, float], q: tuple[float, float], rad: float, k: int = 12
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Display-space polyline of matplotlib's arc3 connection from `p` to `q`."""
    cx = (p[0] + q[0]) / 2 + rad * (q[1] - p[1])
    cy = (p[1] + q[1]) / 2 - rad * (q[0] - p[0])
    pts = []
    for i in range(k + 1):
        f = i / k
        g = 1 - f
        pts.append((g * g * p[0] + 2 * g * f * cx + f * f * q[0], g * g * p[1] + 2 * g * f * cy + f * f * q[1]))
    return list(zip(pts, pts[1:]))


def _group_label(ax: Axes, t: Any, text: str, color: str, up: bool) -> None:
    """`text` in the right margin, just above or below the zero line, after a key dot in `color`."""
    dy = 4 if up else -4
    ax.annotate(
        text, (1, 0), xycoords=("axes fraction", "data"), xytext=(16, dy), textcoords="offset points",
        ha="left", va="bottom" if up else "top", fontsize=8.5, color=t["text_secondary"],
        annotation_clip=False, zorder=3,
    )
    # the key dot sits at the text's mid-height, 9 points right of the end of the zero line
    at = offset_copy(ax.get_yaxis_transform(), fig=ax.figure, x=9, y=dy + (5 if up else -5), units="points")
    ax.scatter([1], [0], s=22, color=color, edgecolors=t["surface"], linewidths=1.0, transform=at,
               clip_on=False, zorder=3)


@themed
def dematel(result: DEMATELResult, *, arrows: bool = True, ax: Axes | None = None) -> Axes:
    r"""DEMATEL cause-effect diagram: prominence against relation, with the strong influences.

    Each factor is a point at $(D + R,\, D - R)$: right means more involved in
    the system (prominence), up means a net giver of influence (relation). A
    hairline at $D - R = 0$ splits the **cause** group above from the
    **effect** group below; causes take the first palette slot and effects the
    second. With `arrows`, every influence $t_{ij}$ above the threshold
    (`result.relations`) is a thin muted arrow from factor $i$ to factor $j$,
    stopping at the edge of its marker; two factors that influence each other
    get two arrows bent to opposite sides. A stronger influence is a heavier,
    darker arrow, and the arrow ends that meet on one marker are spread around
    its edge so the heads stay apart. Factor names are placed clear of the
    points, the arrows and each other.

    The number of arrows is set by the analysis, not by the plot: raise
    `dematel(..., threshold=...)` to keep only the strongest influences (the
    default threshold, the mean of $T$, often keeps half of them), or pass
    `arrows=False`.

    The group key ("cause group", "effect group") sits in the right margin,
    just past the end of the zero line. On a figure of your own with tight
    subplots, leave some room on the right (constrained layout does this) or
    the key can be clipped.

    Args:
        result: A [`DEMATELResult`][bulanik.mcdm.results.DEMATELResult].
        arrows: Draw the relations above the threshold.
        ax: Axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If `result` is not a `DEMATELResult`.

    Examples:
        >>> from bulanik import plot
        >>> from bulanik.mcdm import dematel
        >>> r = dematel([[0, 3, 2], [1, 0, 3], [2, 1, 0]], ["a", "b", "c"])
        >>> ax = plot.dematel(r)
        >>> ax.get_xlabel()
        'prominence $D + R$'

    References:
        Gabus and Fontela (1973), *Perceptions of the World Problematique:
        Communication Procedure, Communicating With Those Bearing Collective
        Responsibility*, DEMATEL Report No. 1, Battelle Geneva Research Centre
        (the cause-effect diagram).
    """
    owner = "dematel"
    if not isinstance(result, DEMATELResult):
        raise TypeError(f"{owner} needs a DEMATELResult, got {type(result).__name__}")
    names = list(result.factors)
    xy = {f: (result.prominence[f], result.relation[f]) for f in names}
    t = tokens()
    ax = new_axes(ax, owner, figsize=(6.4, 4.4))
    style_axes(ax, t, grid=None)
    xs = [p[0] for p in xy.values()]
    ys = [p[1] for p in xy.values()]
    x0, x1 = pad_range(min(xs), max(xs), 0.12)
    y0, y1 = pad_range(min(min(ys), 0.0), max(max(ys), 0.0), 0.16)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("prominence $D + R$")
    ax.set_ylabel("relation $D - R$")
    ax.axhline(0, color=t["axis"], lw=0.8, gid="zero-line", zorder=1)
    for group, color in ((result.causes, t["series"][0]), (result.effects, t["series"][1])):
        if group:
            ax.scatter([xy[f][0] for f in group], [xy[f][1] for f in group], s=_DOT, color=color,
                       edgecolors=t["surface"], linewidths=_RING, gid="factor", zorder=5)
    # group names in the right margin, by the end of the zero line, clear of every point
    _group_label(ax, t, "cause group", t["series"][0], True)
    _group_label(ax, t, "effect group", t["series"][1], False)
    ax.figure.draw_without_rendering()
    segments = [(to_display(ax, (x0, 0.0)), to_display(ax, (x1, 0.0)))]  # names stay off the zero line
    if arrows and result.relations:
        fig = ax.figure
        disp = {f: to_display(ax, xy[f]) for f in names}
        pairs = {(a, b) for a, b, _ in result.relations}
        rads = [_BEND if (b, a) in pairs else 0.0 for a, b, _ in result.relations]
        # each end meets its marker along the arrow's tangent (towards the arc3 control point); the
        # heads that arrive at one marker are then spread around it, so they do not pile up at a hub
        heads: dict[str, list[tuple[float, int]]] = {f: [] for f in names}
        angle: dict[tuple[int, int], float] = {}
        for i, (a, b, _) in enumerate(result.relations):
            c = _control(disp[a], disp[b], rads[i])
            angle[(i, 0)] = math.atan2(c[1] - disp[a][1], c[0] - disp[a][0])
            heads[b].append((math.atan2(c[1] - disp[b][1], c[0] - disp[b][0]), i))
        for lst in heads.values():
            for phi, (_, i) in zip(_spread([e[0] for e in lst], _END_GAP), lst):
                angle[(i, 1)] = phi
        # points: the marker edge plus a little air; on that circle two heads _END_GAP apart are a
        # head's width apart at the tip
        reach = _DOT**0.5 / 2 + _RING / 2 + 2.0
        px = reach * fig.dpi / 72
        strengths = [v for *_, v in result.relations]
        lo = min(strengths)
        span = max(max(strengths) - lo, 1e-12)
        for i, (a, b, v) in enumerate(result.relations):
            w = (v - lo) / span  # 0 for the weakest drawn influence, 1 for the strongest
            # each end is the factor's data point shifted in points onto its marker's edge, so it
            # stays on the edge whatever the final layout of the figure
            start, end = (
                offset_copy(ax.transData, fig=fig, x=reach * math.cos(phi), y=reach * math.sin(phi), units="points")
                for phi in (angle[(i, 0)], angle[(i, 1)])
            )
            ann = ax.annotate(
                "", xy=xy[b], xycoords=end, xytext=xy[a], textcoords=start, annotation_clip=False,
                zorder=3, arrowprops=dict(
                    arrowstyle=f"-|>,head_length={3.6 + 1.4 * w:.3f},head_width={1.6 + 0.8 * w:.3f}",
                    connectionstyle=f"arc3,rad={rads[i]}", shrinkA=0, shrinkB=0, patchA=None, patchB=None,
                    mutation_scale=1, color=t["muted"], lw=0.5 + 1.5 * w, alpha=0.35 + 0.55 * w,
                ),
            )
            ann.set_gid("relation")
            p = tuple(disp[a][k] + px * (math.cos, math.sin)[k](angle[(i, 0)]) for k in range(2))
            q = tuple(disp[b][k] + px * (math.cos, math.sin)[k](angle[(i, 1)]) for k in range(2))
            segments.extend(_arc(p, q, rads[i]))  # type: ignore[arg-type]
    place_point_labels(ax, [xy[f] for f in names], names, t, segments=segments,
                       point_radius_pt=_DOT**0.5 / 2 + _RING)
    return ax


__all__ = ["dematel", "ranking", "weights"]
