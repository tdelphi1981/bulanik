"""House style of `bulanik.plot`: palette, ink tokens, themes and shared helpers."""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

from .._validate import choose

PALETTE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "light": (
            "#2a78d6",  # blue
            "#eb6834",  # orange
            "#1baf7a",  # aqua
            "#eda100",  # yellow
            "#e87ba4",  # magenta
            "#008300",  # green
            "#6250d6",  # violet
            "#e34948",  # red
        ),
        "dark": (
            "#3987e5",
            "#d95926",
            "#199e70",
            "#c98500",
            "#d55181",
            "#2ea32e",
            "#9085e9",
            "#e66767",
        ),
    }
)
"""The categorical palette: eight hues in a fixed order, one tuple per theme.

Slots are assigned in order and never cycled; a ninth series is an error, not a
generated colour. The order passes the dataviz palette validator (lightness band,
chroma floor, colour-vision-deficiency separation, normal-vision floor) on a
white surface (`"light"`) and on the Material for MkDocs slate surface `#1e2129`
(`"dark"`). Three light slots (aqua, yellow, magenta) are below 3:1 contrast on
white, so every plot that uses them also labels its series directly.

Examples:
    >>> from bulanik.plot import PALETTE
    >>> PALETTE["light"][:2]
    ('#2a78d6', '#eb6834')
    >>> len(PALETTE["dark"])
    8
"""

# Ordinal (ordered, discrete) blue ramps: five validated steps each, nearest the
# surface first. Light starts at blue-250 (2.1:1 on white); dark starts at
# blue-550 (2.4:1 on slate). Sequences longer than five sample the ramp
# continuously and rely on position and labels for identity.
_ORDINAL = {
    "light": ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"),
    "dark": ("#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"),
}
# Sequential ramp (magnitude), near-surface end first. The near end is clamped at
# the validated ordinal step (blue-250 on light, blue-550 on dark) so the lowest
# values never fade into the surface.
_SEQUENTIAL = {
    "light": ("#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
    "dark": ("#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
}
_INK = {
    "light": {
        "surface": "#ffffff",
        "text": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e6e5df",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1e2129",
        "text": "#e8eaf0",
        "text_secondary": "#b9bdc8",
        "muted": "#8b909c",
        "grid": "#2e323d",
        "axis": "#4a4f5c",
    },
}
_FILL_ALPHA = {"light": 0.18, "dark": 0.2}
_THEMES = ("light", "dark")
_FONTS = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
LINE_WIDTH = 1.8  # points; about 2 px on screen
MAX_SERIES = len(PALETTE["light"])

_stack: list[str] = []


def _active() -> str:
    return _stack[-1] if _stack else "light"


def _tokens(name: str) -> dict[str, Any]:
    t: dict[str, Any] = dict(_INK[name])
    t.update(
        name=name,
        series=PALETTE[name],
        ramp=_ORDINAL[name],
        sequential=_SEQUENTIAL[name],
        fill_alpha=_FILL_ALPHA[name],
    )
    return t


def current_theme() -> Mapping[str, Any]:
    """The tokens of the active theme, as a read-only mapping.

    Keys: `name` (`"light"` or `"dark"`), `surface` (background), `text`,
    `text_secondary`, `muted` (ink for labels and annotations), `grid`, `axis`
    (hairlines), `series` (the categorical palette), `ramp` (five ordinal blue
    steps, nearest the surface first), `sequential` (the blue magnitude ramp)
    and `fill_alpha` (opacity of the soft fills under curves). Use them to
    annotate a figure in the house style.

    Returns:
        The tokens of the innermost active [`theme`][bulanik.plot.theme], or of
        `"light"` outside any theme.

    Examples:
        >>> from bulanik import plot
        >>> plot.current_theme()["name"]
        'light'
        >>> with plot.theme("dark"):
        ...     plot.current_theme()["surface"]
        '#1e2129'
    """
    return MappingProxyType(_tokens(_active()))


def _rc(name: str) -> dict[str, Any]:
    t = _tokens(name)
    return {
        "figure.facecolor": t["surface"],
        "figure.edgecolor": t["surface"],
        "figure.figsize": (6.4, 3.6),
        "figure.dpi": 100,
        "axes.facecolor": t["surface"],
        "axes.edgecolor": t["axis"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": t["text_secondary"],
        "axes.labelsize": 10,
        "axes.labelpad": 6,
        "axes.titlecolor": t["text"],
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "axes.prop_cycle": mpl.cycler(color=list(t["series"])),
        "grid.color": t["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "grid.alpha": 1.0,
        "text.color": t["text"],
        "font.family": "sans-serif",
        "font.sans-serif": _FONTS,
        "font.size": 9.5,
        "mathtext.fontset": "stixsans",
        "xtick.color": t["axis"],
        "ytick.color": t["axis"],
        "xtick.labelcolor": t["text_secondary"],
        "ytick.labelcolor": t["text_secondary"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,
        "lines.linewidth": LINE_WIDTH,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round",
        "patch.linewidth": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": t["text_secondary"],
        "savefig.facecolor": t["surface"],
        "savefig.edgecolor": t["surface"],
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "svg.fonttype": "path",
        "polaraxes.grid": True,
        "axes3d.grid": True,
    }


@contextlib.contextmanager
def theme(name: str = "light") -> Iterator[Mapping[str, Any]]:
    """Context manager that applies the house style in a light or dark variant.

    Inside the block, matplotlib's rc parameters (fonts, sizes, surface and ink
    colours, hairline axes, 2 px lines, tight saving at 200 dpi) follow the
    theme, and every `bulanik.plot` function picks the matching palette. Save
    figures inside the block so the background matches. Themes nest; the
    previous settings come back on exit.

    Args:
        name: `"light"` (white surface) or `"dark"` (the Material for MkDocs
            slate surface `#1e2129`).

    Yields:
        The theme tokens, as returned by [`current_theme`][bulanik.plot.current_theme].

    Raises:
        ValueError: If `name` is not `"light"` or `"dark"`.

    Examples:
        >>> from bulanik import TFN, plot
        >>> with plot.theme("dark") as tokens:
        ...     ax = plot.membership(TFN(1, 2, 4))
        >>> tokens["name"], type(ax).__name__
        ('dark', 'Axes')
    """
    name = choose(name, _THEMES, "theme", "bulanik.plot.theme")
    _stack.append(name)
    try:
        with mpl.rc_context(_rc(name)):
            yield MappingProxyType(_tokens(name))
    finally:
        _stack.pop()


F = TypeVar("F", bound=Callable[..., Any])


def themed(fn: F) -> F:
    """Run a plotting function inside the rc settings of the active theme."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with mpl.rc_context(_rc(_active())):
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


# -- helpers shared by the plotting modules ---------------------------------------


def tokens() -> dict[str, Any]:
    return _tokens(_active())


def new_axes(
    ax: Axes | None,
    owner: str,
    *,
    projection: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Axes:
    """Return `ax` (checked for the projection) or a new figure's axes."""
    want = projection or "rectilinear"
    if ax is None:
        # constrained layout collapses 3-D axes; they get the full figure instead
        layout = None if projection == "3d" else "constrained"
        fig = plt.figure(figsize=figsize, layout=layout)
        return fig.add_subplot(projection=projection)
    if not isinstance(ax, Axes):
        raise TypeError(f"{owner} ax must be a matplotlib Axes, got {type(ax).__name__}")
    if ax.name != want:
        kind = {"3d": "3-D axes", "polar": "polar axes"}.get(want, "2-D (rectilinear) axes")
        raise ValueError(f"{owner} needs {kind}, got '{ax.name}' axes")
    return ax


def style_axes(ax: Axes, t: Mapping[str, Any], *, grid: str | None = "y") -> None:
    """Hairline chrome: left/bottom spines, quiet ticks, optional solid grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["axis"], labelcolor=t["text_secondary"])
    if grid:
        ax.grid(True, axis=grid, color=t["grid"], linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def series_colors(n: int, t: Mapping[str, Any], *, ordered: bool = False) -> list[str]:
    """`n` colours: categorical slots in order, or the ordinal ramp past 8 / when ordered."""
    if not ordered and n <= MAX_SERIES:
        return list(t["series"][:n])
    return ramp_colors(n, t)


def ramp_colors(n: int, t: Mapping[str, Any]) -> list[str]:
    """`n` ordered colours from the ordinal ramp (validated steps when n <= 5)."""
    steps = t["ramp"]
    if n == 1:
        return [steps[2]]
    if n <= len(steps):
        idx = [round(i * (len(steps) - 1) / (n - 1)) for i in range(n)]
        return [steps[i] for i in idx]
    cmap = LinearSegmentedColormap.from_list("bulanik-ramp", [to_rgb(c) for c in steps])
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def sequential_cmap(t: Mapping[str, Any]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "bulanik-seq", [to_rgb(c) for c in t["sequential"]]
    )


def as_list(items: Any, kinds: tuple[type, ...], what: str, owner: str) -> list[Any]:
    """A single instance or a sequence of instances of `kinds`, as a list."""
    if isinstance(items, kinds):
        return [items]
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError(f"{owner} needs {what}, got {type(items).__name__}")
    out = list(items)
    for v in out:
        if not isinstance(v, kinds):
            raise TypeError(f"{owner} needs {what}, got {type(v).__name__}")
    return out


def check_labels(labels: Sequence[str] | None, n: int, owner: str) -> list[str] | None:
    if labels is None:
        return None
    if isinstance(labels, str):
        labels = [labels]
    labels = [str(s) for s in labels]
    if len(labels) != n:
        raise ValueError(f"{owner} got {len(labels)} labels for {n} numbers")
    return labels


def check_count(n: int, owner: str, what: str = "numbers") -> None:
    if n > MAX_SERIES:
        raise ValueError(
            f"{owner} draws at most {MAX_SERIES} {what} (one palette slot each), got {n}; "
            "split them over several axes"
        )


def fmt(v: float) -> str:
    """Short number text: up to 4 significant digits, no trailing zeros."""
    if v == 0:
        return "0"
    return f"{v:.4g}"


def place_rows(
    ax: Axes, items: Sequence[tuple[float, str]], fontsize: float, gap_px: float = 6.0
) -> list[int]:
    """Greedy row assignment so labels centred at data x never overlap.

    Returns a row index per item (0 = lowest row). Text width is estimated from
    the font size and the axes width in pixels.
    """
    x0, x1 = ax.get_xlim()
    width_px = max(ax.get_window_extent().width, 1.0)
    per_px = (x1 - x0) / width_px
    char_px = 0.58 * fontsize * ax.figure.dpi / 72
    rows_end: list[float] = []
    out = [0] * len(items)
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    for i in order:
        x, text = items[i]
        half = (len(text) * char_px / 2 + gap_px / 2) * per_px
        for r, end in enumerate(rows_end):
            if x - half >= end:
                rows_end[r] = x + half
                out[i] = r
                break
        else:
            rows_end.append(x + half)
            out[i] = len(rows_end) - 1
    return out


def peak_labels(
    ax: Axes,
    peaks: Sequence[tuple[float, float, str]],
    t: Mapping[str, Any],
    *,
    fontsize: float = 9.0,
    top: float = 1.0,
) -> float:
    """Direct labels above curve peaks, staggered into rows on collision.

    Returns the y extent the labels need (to set the upper y-limit).
    """
    rows = place_rows(ax, [(x, s) for x, _, s in peaks], fontsize)
    x0, x1 = ax.get_xlim()
    per_px = (x1 - x0) / max(ax.get_window_extent().width, 1.0)
    char = 0.58 * fontsize * ax.figure.dpi / 72 * per_px
    step = 0.11
    ymax = top
    for (x, y, s), r in zip(peaks, rows):
        ly = y + 0.035 if r == 0 else top + 0.035 + r * step
        if r:
            ax.plot([x, x], [y + 0.02, ly - 0.01], color=t["axis"], lw=0.6, zorder=2)
        half = len(s) * char / 2
        ha = "left" if x - half < x0 else "right" if x + half > x1 else "center"
        ax.text(
            x, ly, s, ha=ha, va="bottom", fontsize=fontsize,
            color=t["text_secondary"], zorder=6,
        )
        ymax = max(ymax, ly + 0.1)
    return ymax


def gaps_at_zero(ys: Sequence[float]) -> list[float]:
    """`ys` with NaN inside runs of zeros, so a curve does not trace the x axis.

    The zeros that touch a non-zero neighbour stay, so the curve still meets the
    axis where it rises or falls.
    """
    n = len(ys)
    out = list(ys)
    for i, y in enumerate(ys):
        if y == 0 and (i == 0 or ys[i - 1] == 0) and (i == n - 1 or ys[i + 1] == 0):
            out[i] = math.nan
    return out


def finish_unit_y(ax: Axes, t: Mapping[str, Any], ytop: float, label: str = r"$\mu(x)$") -> None:
    """Membership axis: 0, 0.5, 1 ticks, spine ending at 1, room for labels above."""
    ax.set_ylim(0, max(ytop, 1.05))
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.spines["left"].set_bounds(0, 1)
    ax.set_ylabel(label)


def pad_range(lo: float, hi: float, frac: float = 0.06) -> tuple[float, float]:
    span = hi - lo
    if span <= 0:
        span = max(abs(lo) * 0.5, 1.0)
        return lo - span, hi + span
    return lo - frac * span, hi + frac * span


def fill_alpha(t: Mapping[str, Any], n: int, color: str | None = None) -> float:
    """Opacity of the soft fills when `n` filled shapes overlap: lighter as `n` grows."""
    return wash(t, color, max(0.08, t["fill_alpha"] * min(1.0, 3.0 / max(n, 1))))


# Warm dark-theme slots (orange, yellow, magenta, red): their translucent fills
# mix into brown over the slate surface and each other, so they are washed lighter.
_WARM_DARK = frozenset({"#d95926", "#c98500", "#d55181", "#e66767"})


def wash(t: Mapping[str, Any], color: str | None, alpha: float) -> float:
    """`alpha` for a fill of `color`, reduced for warm hues on the dark theme."""
    if t["name"] == "dark" and color is not None and to_hex(color) in _WARM_DARK:
        return alpha * 0.6
    return alpha


def halo(t: Mapping[str, Any], width: float = 3.0) -> list[Any]:
    """Path effect: a soft outline in the surface colour, so lines never cross a label's glyphs."""
    return [patheffects.withStroke(linewidth=width, foreground=t["surface"], alpha=0.9)]


# -- collision-aware point labels ----------------------------------------------------

# compass offsets in points: (dx, dy, ha, va)
_COMPASS = {
    "NE": (6, 4, "left", "bottom"),
    "SE": (6, -4, "left", "top"),
    "NW": (-6, 4, "right", "bottom"),
    "SW": (-6, -4, "right", "top"),
    "E": (8, 0, "left", "center"),
    "W": (-8, 0, "right", "center"),
    "N": (0, 7, "center", "bottom"),
    "S": (0, -7, "center", "top"),
}
_ORDER = ("NE", "SE", "NW", "SW", "E", "W", "N", "S")


def _rect_hits_circle(bb: Any, x: float, y: float, r: float) -> bool:
    cx = min(max(x, bb.x0), bb.x1)
    cy = min(max(y, bb.y0), bb.y1)
    return (cx - x) ** 2 + (cy - y) ** 2 < r * r


def _rect_hits_segment(bb: Any, p: tuple[float, float], q: tuple[float, float]) -> bool:
    for i in range(41):
        f = i / 40
        x, y = p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])
        if bb.x0 <= x <= bb.x1 and bb.y0 <= y <= bb.y1:
            return True
    return False


def to_display(ax: Axes, xy: tuple[float, float], coords: str = "data") -> tuple[float, float]:
    tr = ax.transData if coords == "data" else ax.figure.transFigure
    x, y = tr.transform(xy)
    return float(x), float(y)


def place_point_labels(
    ax: Axes,
    anchors: Sequence[tuple[float, float]],
    labels: Sequence[str],
    t: Mapping[str, Any],
    *,
    coords: Any = "data",
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]] = (),
    avoid: Sequence[Any] = (),
    prefer: Sequence[Sequence[str]] | None = None,
    fontsize: float = 9.0,
    color: str | None = None,
    point_radius_pt: float = 5.0,
    positions: Sequence[tuple[float, float]] | None = None,
) -> list[Any]:
    """Label points without collisions, trying compass offsets in turn.

    `anchors` are in `coords`: "data", or a list of annotation coordinate
    callables (one per label, returning a zero-size Bbox at the point; each
    label then follows its point at draw time, which keeps
    3-D labels on their points through tight-bbox saving; `positions` then gives
    their current display positions). `segments` are
    display-space line segments to keep clear of; `avoid` are display-space
    bounding boxes (e.g. other labels). A candidate is kept when its text box
    hits no point, no segment, no placed label, stays inside the figure, and its
    centre is nearer its own point than any other. If no candidate is clean, the
    one with the fewest conflicts wins.
    """
    from matplotlib.transforms import Bbox

    fig = ax.figure
    fig.draw_without_rendering()
    artists = None if isinstance(coords, str) else list(coords)
    if artists is None:
        disp = [to_display(ax, a, coords) for a in anchors]
    else:  # the display positions of the artists come from the caller
        disp = [tuple(p) for p in positions]  # type: ignore[union-attr]
        anchors = [(0.0, 0.0)] * len(artists)
    r_px = point_radius_pt * fig.dpi / 72
    pad = 2.0 * fig.dpi / 72
    placed: list[Any] = [Bbox(b.get_points()) for b in avoid]
    frame = fig.bbox
    out = []
    for i, (anchor, text) in enumerate(zip(anchors, labels)):
        ann = ax.annotate(
            text, anchor, xycoords=coords if artists is None else artists[i], xytext=(0, 0), textcoords="offset points",
            fontsize=fontsize, color=color or t["text_secondary"], zorder=8, path_effects=halo(t),
        )
        order = list(prefer[i]) if prefer is not None else list(_ORDER)
        order += [d for d in _ORDER if d not in order]
        cands = [(d, 1.0) for d in order] + [(d, 2.2) for d in order]
        best = None
        for d, scale in cands:
            dx, dy, ha, va = _COMPASS[d]
            ann.set_position((dx * scale, dy * scale))
            ann.set_ha(ha)
            ann.set_va(va)
            bb = ann.get_window_extent().expanded(1.0, 1.0)
            bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
            cost = 0
            cost += sum(_rect_hits_circle(bb, x, y, r_px) for x, y in disp)
            cost += sum(_rect_hits_segment(bb, p, q) for p, q in segments)
            cost += sum(bb.overlaps(o) for o in placed)
            cost += int(not (frame.x0 <= bb.x0 and bb.x1 <= frame.x1 and frame.y0 <= bb.y0 and bb.y1 <= frame.y1))
            cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
            own = (cx - disp[i][0]) ** 2 + (cy - disp[i][1]) ** 2
            cost += 5 * sum(  # sitting nearer another point misattributes the label
                (cx - x) ** 2 + (cy - y) ** 2 < own for j, (x, y) in enumerate(disp) if j != i
            )
            if best is None or cost < best[0]:
                best = (cost, dx * scale, dy * scale, ha, va, bb)
            if cost == 0:
                break
        _, dx, dy, ha, va, bb = best  # type: ignore[misc]
        ann.set_position((dx, dy))
        ann.set_ha(ha)
        ann.set_va(va)
        placed.append(bb)
        out.append(ann)
    return out


def grid3d(axis: Any, t: Mapping[str, Any], *, ticks: bool = False) -> None:
    """Hairline 3-D grid; uses matplotlib's private `_axinfo`, so it degrades gracefully."""
    try:
        axis._axinfo["grid"].update(color=t["grid"], linewidth=0.6)
        if ticks:
            axis._axinfo["tick"].update(inward_factor=0.0, outward_factor=0.2)
    except (AttributeError, KeyError, TypeError):  # pragma: no cover - other matplotlib versions
        pass  # the rc grid settings of the theme still apply
