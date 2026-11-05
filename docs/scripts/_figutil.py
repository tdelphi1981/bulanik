"""Shared plumbing for the docs figure scripts.

All plotting goes through `bulanik.plot`; this module only

- renders a figure in both house themes and saves `<name>-light.svg` and
  `<name>-dark.svg` into `docs/site/assets/figures/` with deterministic output
  (fixed SVG id salt, no creation date), and
- offers page-level layout helpers: multi-panel figures and annotations
  (notes, brackets, arrows, markers) drawn with the active theme's tokens from
  `bulanik.plot.current_theme()`.

A figure is a function `draw() -> Figure | Axes` that runs inside the theme.
"""

from __future__ import annotations

import io
import os
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from bulanik import plot  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "site" / "assets" / "figures"
THEMES = ("light", "dark")
Draw = Callable[[], "Figure | Axes"]


def tokens() -> Mapping[str, Any]:
    """The active theme's tokens (`bulanik.plot.current_theme()`)."""
    return plot.current_theme()


def render(name: str, draw: Draw, out: Path = FIGURES) -> list[Path]:
    """Draw the figure once per theme and save `<name>-<theme>.svg`."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for mode in THEMES:
        with plot.theme(mode), mpl.rc_context({"svg.hashsalt": f"bulanik-{name}"}):
            res = draw()
            fig = res.figure if isinstance(res, Axes) else res
            path = out / f"{name}-{mode}.svg"
            buffer = io.StringIO()
            fig.savefig(buffer, format="svg", metadata={"Date": None, "Creator": "bulanik docs"})
            plt.close(fig)
            path.write_text(compact(buffer.getvalue()), encoding="utf-8")
        paths.append(path)
    return paths


_DECIMALS = re.compile(r"(\d\.\d{2})\d+")
_PATH_DATA = re.compile(r' d="[^"]*"')  # only path geometry; transforms keep full precision


def compact(svg: str) -> str:
    """Round path coordinates to 0.01 pt: invisible at any zoom, and the files shrink by ~40%.

    Haloed labels are written as outline paths (not glyph references), so the
    coordinate text dominates the file size.
    """
    return _PATH_DATA.sub(lambda m: _DECIMALS.sub(r"\1", m.group(0)), svg)


def the_text(ax: Axes, *, startswith: str | None = None, equals: str | None = None):
    """The one text of `ax` that matches; fail loudly on zero or several matches."""
    found = [
        t
        for t in ax.texts
        if (t.get_text().startswith(startswith) if startswith is not None else t.get_text() == equals)
    ]
    if len(found) != 1:
        raise AssertionError(f"expected exactly one text matching {startswith or equals!r}, found {len(found)}")
    return found[0]


def run(figures: Mapping[str, Draw], argv: list[str] | None = None) -> None:
    """Command-line entry: render all figures, or only the names given."""
    names = (argv if argv is not None else sys.argv[1:]) or list(figures)
    for name in names:
        for path in render(name, figures[name]):
            print(path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path)


# -- layout ---------------------------------------------------------------------------


def panels(nrows: int, ncols: int, *, figsize: tuple[float, float], **kw: Any) -> tuple[Figure, Any]:
    """A constrained-layout figure with a grid of axes, in the active theme."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained", **kw)
    return fig, axes


def halo(width: float = 3.0) -> list[Any]:
    """Surface-coloured outline so lines never cut through a label."""
    return [patheffects.withStroke(linewidth=width, foreground=tokens()["surface"], alpha=0.9)]


def note(ax: Axes, x: float, y: float, text: str, *, ink: str = "text_secondary", size: float = 8.5, **kw: Any) -> Any:
    """Annotation text in a text token (never a series colour), with a halo."""
    kw.setdefault("ha", "center")
    kw.setdefault("va", "bottom")
    kw.setdefault("zorder", 9)
    return ax.text(x, y, text, color=tokens()[ink], fontsize=size, path_effects=halo(), **kw)


def bracket(ax: Axes, x0: float, x1: float, y: float, text: str, *, tick: float = 0.035, below: bool = False) -> None:
    """A hairline bracket over [x0, x1] at height y with a centred label."""
    t = tokens()
    s = -1 if below else 1
    ax.plot([x0, x0, x1, x1], [y - s * tick, y, y, y - s * tick], color=t["muted"], lw=0.8,
            solid_joinstyle="miter", zorder=7, clip_on=False)
    note(ax, (x0 + x1) / 2, y + s * 0.02, text, va="top" if below else "bottom", ink="text_secondary")


def arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float], color: str, *,
          lw: float = 1.2, shrink: float = 6.0, alpha: float = 1.0, rad: float = 0.0,
          shrink_end: float | None = None) -> None:
    """A thin arrow in a series colour (a mark, so it may carry the hue).

    `shrink_end` (points) stops the head short of the target marker; it defaults to `shrink`.
    """
    ax.annotate(
        "", xy=end, xytext=start, zorder=6,
        arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=color, lw=lw,
                        shrinkA=shrink, shrinkB=shrink if shrink_end is None else shrink_end, alpha=alpha,
                        connectionstyle=f"arc3,rad={rad}"),
    )


def dot(ax: Axes, x: float, y: float, color: str, *, size: float = 8.0, marker: str = "o", zorder: float = 8) -> None:
    """A marker with the house 2 px surface ring."""
    ax.plot([x], [y], marker=marker, ms=size, mfc=color, mec=tokens()["surface"], mew=1.6,
            linestyle="none", zorder=zorder, clip_on=False)


def key(ax: Axes, items: list[tuple[str, str, str]], *, loc: str = "upper right", **kw: Any) -> Any:
    """A compact legend: (marker, colour, text) per entry, text in ink tokens."""
    from matplotlib.lines import Line2D

    t = tokens()
    handles = [
        Line2D([], [], marker=m, linestyle="none", ms=7, mfc=c, mec=t["surface"], mew=1.4, label=s)
        if m else Line2D([], [], color=c, lw=1.8, label=s)
        for m, c, s in items
    ]
    kw.setdefault("fontsize", 8.5)
    leg = ax.legend(handles=handles, loc=loc, frameon=False, labelcolor=t["text_secondary"],
                    handletextpad=0.4, borderaxespad=0.3, **kw)
    leg.set_zorder(10)
    return leg


def label(ax: Axes, x: float, y: float, text: str, *, dx: float = 7, dy: float = 0, ha: str = "left",
          va: str = "center", size: float = 8.5, ink: str = "text_secondary") -> Any:
    """Point label offset by (dx, dy) points, in a text token with a halo."""
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                       fontsize=size, color=tokens()[ink], path_effects=halo(), zorder=9)


def callout(ax: Axes, xy: tuple[float, float], at: tuple[float, float], text: str, *, ha: str = "left",
            va: str = "center", size: float = 8.5, ink: str = "text_secondary") -> Any:
    """A label placed at data point `at`, tied to `xy` by a hairline leader."""
    t = tokens()
    return ax.annotate(text, xy, xytext=at, textcoords="data", ha=ha, va=va, fontsize=size, color=t[ink],
                       path_effects=halo(), zorder=9,
                       arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.6, shrinkA=3, shrinkB=6))
