"""Plots of fuzzy inference: Mamdani output sets and control surfaces."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ..inference import MamdaniFIS, NoRuleFiredError
from ..inference._common import aggregate as _aggregate
from ..inference._common import imply as _imply
from ..type2 import NoLowerFiringError
from ._style import (
    wash,
    grid3d,
    finish_unit_y,
    fmt,
    gaps_at_zero,
    halo,
    new_axes,
    peak_labels,
    sequential_cmap,
    series_colors,
    style_axes,
    themed,
    tokens,
)
from .sets import _peak_x


def _output(fis: Any, output: str | None, owner: str) -> Any:
    outs = list(fis.outputs)
    if output is None:
        return outs[0]
    for v in outs:
        if v.name == output:
            return v
    names = ", ".join(v.name for v in outs)
    raise ValueError(f"{owner}: unknown output '{output}'; outputs: {names}")


@themed
def mamdani_output(
    fis: MamdaniFIS,
    inputs: Mapping[str, float],
    output: str | None = None,
    *,
    ax: Axes | None = None,
) -> Axes:
    r"""How a Mamdani system reaches its crisp output for one set of inputs.

    The output variable's terms are drawn as thin outlines. Every rule that
    fires contributes its consequent set, clipped (`implication="min"`) or
    scaled (`"prod"`) by its firing strength, as a filled area in its term's
    colour. The aggregated set (the system's `aggregation` of the implied sets,
    on the system's own sampling grid) is the dark outline, and the crisp output
    from [`MamdaniFIS.evaluate`][bulanik.inference.mamdani.MamdaniFIS.evaluate]
    is the vertical marker. When no rule fires and the system has a `default`,
    only the default value is marked.

    Args:
        fis: A [`MamdaniFIS`][bulanik.inference.mamdani.MamdaniFIS].
        inputs: Crisp value of every input variable, by name.
        output: Output variable name; `None` takes the first output.
        ax: Axes to draw into; `None` creates a new figure.

    Returns:
        The matplotlib `Axes`.

    Raises:
        TypeError: If `fis` is not a `MamdaniFIS`.
        ValueError: If `output` is unknown, or an input is missing or outside
            its universe (as in `MamdaniFIS.explain`).
        NoRuleFiredError: If no rule fires for the output and the system has
            no `default`.

    Examples:
        >>> from bulanik import plot
        >>> from bulanik.sets import LinguisticVariable, triangular
        >>> from bulanik.inference import MamdaniFIS
        >>> service = LinguisticVariable("service", (0, 10), {
        ...     "poor": triangular(0, 0, 5), "good": triangular(5, 10, 10)})
        >>> tip = LinguisticVariable("tip", (0, 20), {
        ...     "low": triangular(0, 0, 10), "high": triangular(10, 20, 20)})
        >>> fis = MamdaniFIS([service], [tip], [
        ...     "IF service IS poor THEN tip IS low", "IF service IS good THEN tip IS high"])
        >>> ax = plot.mamdani_output(fis, {"service": 7.5})
        >>> type(ax).__name__
        'Axes'
    """
    owner = "mamdani_output"
    if not isinstance(fis, MamdaniFIS):
        raise TypeError(f"{owner} needs a MamdaniFIS, got {type(fis).__name__}")
    v = _output(fis, output, owner)
    info = fis.explain(inputs)
    crisp = info.outputs[v.name]
    lo, hi = v.universe
    n = fis.resolution
    step = (hi - lo) / (n - 1)
    xs = [lo + i * step for i in range(n - 1)] + [hi]
    terms = list(v.term_names)
    t = tokens()
    colors = dict(zip(terms, series_colors(len(terms), t, ordered=len(terms) > 8)))
    ax = new_axes(ax, owner)
    style_axes(ax, t)
    ax.set_xlim(lo, hi)
    peaks = []
    curves = {}
    for name in terms:
        mf = v[name]
        ys = [mf(x) for x in xs]
        curves[name] = ys
        ax.plot(xs, gaps_at_zero(ys), color=colors[name], lw=1.0, alpha=0.75, zorder=3, clip_on=False)
        px, py = _peak_x(xs, ys)
        peaks.append((px, py, name))
    agg = [0.0] * n
    fired = 0
    for rule, act in zip(fis.rules, info.activations):
        term = rule.consequents.get(v.name)
        alpha = act.strength
        if term is None or alpha == 0.0:
            continue
        ys = [_imply(alpha, mu, fis.implication) for mu in curves[term]]
        agg = [_aggregate(a, y, fis.aggregation) for a, y in zip(agg, ys)]
        c = colors[term]
        ax.fill_between(xs, ys, 0, color=c, alpha=wash(t, c, t["fill_alpha"] * 1.5), linewidth=0, gid="consequent", zorder=2)
        ax.plot(xs, gaps_at_zero(ys), color=c, lw=1.3, zorder=4)
        fired += 1
    ax.plot(xs, gaps_at_zero(agg), color=t["text"], lw=2.0, gid="aggregate", zorder=5)
    top = max(1.0, max(agg))
    ytop = max(peak_labels(ax, peaks, t, top=top), top + 0.12)
    # the crisp value: a marker line up to the top of the sets, labelled beside its top
    how = fis.defuzzification if fired else "default"
    ax.plot([crisp, crisp], [0, top], color=t["text"], lw=1.1, gid="crisp", zorder=6)
    ax.plot([crisp], [0], marker="o", ms=7.5, mfc=t["text"], mec=t["surface"], mew=1.6, zorder=7, clip_on=False)
    x0, x1 = ax.get_xlim()
    right = crisp > x0 + 0.75 * (x1 - x0)
    ax.annotate(
        f"{v.name} = {fmt(round(crisp, 4))}  ({how})", (crisp, top), xytext=(-7 if right else 7, -2),
        textcoords="offset points", ha="right" if right else "left", va="top", fontsize=9,
        color=t["text"], zorder=8, path_effects=halo(t),
    )
    finish_unit_y(ax, t, ytop)
    if top > 1.0:
        ax.spines["left"].set_bounds(0, top)
    ax.set_xlabel(v.name)
    return ax


def _safe(fis: Any, inputs: dict[str, float], name: str) -> float:
    try:
        return float(fis.evaluate(inputs)[name])
    except (NoRuleFiredError, NoLowerFiringError):
        return math.nan


@themed
def surface(
    fis: Any,
    output: str | None = None,
    *,
    resolution: int = 41,
    ax: Axes | None = None,
) -> Axes:
    """Input-output map of a fuzzy system: a 3-D surface (2 inputs) or a curve (1 input).

    Works with every system that has `inputs`, `outputs` and `evaluate`
    ([`MamdaniFIS`][bulanik.inference.mamdani.MamdaniFIS],
    [`SugenoFIS`][bulanik.inference.sugeno.SugenoFIS],
    [`TsukamotoFIS`][bulanik.inference.tsukamoto.TsukamotoFIS] and the interval
    type-2 systems). Each input universe is sampled at `resolution` evenly spaced
    points; where no rule fires (and there is no `default`) the output is left
    as a gap. A surface is coloured by height with a one-hue sequential ramp.

    Args:
        fis: The fuzzy inference system.
        output: Output variable name; `None` takes the first output.
        resolution: Samples per input, an integer of at least 2.
        ax: Axes to draw into (3-D for two inputs); `None` creates a new figure.

    Returns:
        The matplotlib `Axes` (3-D for two inputs).

    Raises:
        TypeError: If `fis` has no `inputs`, `outputs` and `evaluate`.
        ValueError: If the system has more than 2 inputs, `output` is unknown,
            `resolution` is not an integer of at least 2, or `ax` has the wrong
            projection.

    Examples:
        >>> from bulanik import plot
        >>> from bulanik.sets import LinguisticVariable, triangular
        >>> from bulanik.inference import SugenoFIS, SugenoOutput
        >>> x = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 10), "hi": triangular(0, 10, 10)})
        >>> y = SugenoOutput("y", {"a": 1.0, "b": 5.0})
        >>> fis = SugenoFIS([x], [y], ["IF x IS lo THEN y IS a", "IF x IS hi THEN y IS b"])
        >>> ax = plot.surface(fis, resolution=21)
        >>> type(ax).__name__
        'Axes'
    """
    owner = "surface"
    if not all(hasattr(fis, a) for a in ("inputs", "outputs", "evaluate")):
        raise TypeError(f"{owner} needs a fuzzy inference system, got {type(fis).__name__}")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2:
        raise ValueError(f"{owner} resolution must be an integer >= 2, got {resolution!r}")
    ins = list(fis.inputs)
    if len(ins) not in (1, 2):
        raise ValueError(f"{owner} draws systems with 1 or 2 inputs, got {len(ins)}")
    name = _output(fis, output, owner).name
    t = tokens()

    def grid(v: Any) -> list[float]:
        lo, hi = v.universe
        return [lo + (hi - lo) * i / (resolution - 1) for i in range(resolution - 1)] + [hi]

    if len(ins) == 1:
        (v,) = ins
        ax = new_axes(ax, owner)
        style_axes(ax, t)
        xs = grid(v)
        ys = [_safe(fis, {v.name: x}, name) for x in xs]
        ax.plot(xs, ys, color=t["series"][0], gid="surface", zorder=3)
        ax.set_xlim(*v.universe)
        ax.set_xlabel(v.name)
        ax.set_ylabel(name)
        return ax
    import numpy as np

    vx, vy = ins
    ax = new_axes(ax, owner, projection="3d", figsize=(6.0, 5.0))
    xs, ys = grid(vx), grid(vy)
    Z = np.array([[_safe(fis, {vx.name: x, vy.name: y}, name) for x in xs] for y in ys])
    X, Y = np.meshgrid(xs, ys)
    surf = ax.plot_surface(
        X, Y, Z, cmap=sequential_cmap(t), rstride=1, cstride=1, linewidth=0.25,
        edgecolor=t["surface"], antialiased=True, shade=False, alpha=1.0,
    )
    surf.set_gid("surface")
    ax.set_xlim(*vx.universe)
    ax.set_ylim(*vy.universe)
    ax.view_init(elev=28, azim=-128)
    ax.set_box_aspect((1, 1, 0.62))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis.pane.set_edgecolor(t["grid"])
        axis.line.set_color(t["axis"])
        grid3d(axis, t)
        axis.label.set_color(t["text_secondary"])
    ax.tick_params(colors=t["text_secondary"], labelsize=8.5, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(5))
        axis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    ax.set_xlabel(vx.name)
    ax.set_ylabel(vy.name)
    ax.set_zlabel(name)
    return ax


__all__ = ["mamdani_output", "surface"]
