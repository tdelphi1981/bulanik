r"""Grid-partition first-order Sugeno ANFIS (Jang 1993) with hybrid or gradient training.

[`ANFIS`][bulanik.anfis.ANFIS] fits a first-order Sugeno system to numeric
data. Every input has its own Gaussian or generalized-bell membership
functions; the rules are the grid of all combinations of one MF per input (in
`itertools.product` order); AND is the product; and rule $r$ has a linear
consequent $p_r \cdot x + r_r$. For an input vector $x$ the output is

$$f(x) = \sum_r \bar w_r(x)\,(p_r \cdot x + r_r), \qquad
\bar w_r = \frac{w_r}{\sum_s w_s}, \qquad w_r = \prod_i \mu_{i, k_i(r)}(x_i),$$

with $\mu(x) = \exp\bigl(-(x - c)^2 / (2\sigma^2)\bigr)$ for `gaussian` and
$\mu(x) = 1 / (1 + |(x - c)/a|^{2b})$ for `bell`. After training,
[`to_sugeno`][bulanik.anfis.ANFIS.to_sugeno] exports an equivalent
[`SugenoFIS`][bulanik.inference.sugeno.SugenoFIS] with `and_method="prod"`.

Firing strengths are computed in the log domain ($\log w_r = \sum_i \log \mu$,
then a max-shifted softmax), so far-away samples never give 0/0. The result
equals the plain ratio $w_r / \sum w$ up to rounding, which is what the exported
`SugenoFIS` computes.

Training is deterministic; there is no random initialisation:

- `hybrid`: least squares (ridge normal equation, pure Python) for all
  consequents, then one batch gradient step on the premise parameters;
- `gradient`: one batch gradient step on all parameters per epoch.

Guard: after every step widths (Gaussian $\sigma$, bell $a$) are floored at
`MIN_WIDTH * span` of their input and bell $b$ at `MIN_B`. A step that would
raise the training RMSE, or make it or any parameter non-finite, is halved until
it does not, and dropped after 60 halvings, so `lr` is the largest step tried and
not the step taken. A `fit` that dropped any step warns once (`RuntimeWarning`)
with the count.

The least-squares step works on inputs normalised to $[0, 1]$ over each
input's range and maps the solution back, so it does not depend on the input
offset or scale. The gradient step `lr` is not scale-invariant.

`predict` outside the trained `ranges` extrapolates and warns
([`ExtrapolationWarning`][bulanik.anfis.ExtrapolationWarning]).

Examples:
    >>> from bulanik.anfis import ANFIS
    >>> X = [[i / 10] for i in range(21)]
    >>> y = [x[0] ** 2 for x in X]
    >>> model = ANFIS(1, 3, ranges=[(0, 2)])
    >>> history = model.fit(X, y, epochs=5)
    >>> history[-1] < 0.02
    True
    >>> round(model.predict([[1.0]])[0], 1)
    1.0

References:
    Jang (1993): "ANFIS: adaptive-network-based fuzzy inference system", IEEE
    Transactions on Systems, Man, and Cybernetics.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Sequence

from ._linalg import normal_equations, solve_spd
from ._validate import _num, choose, finite
from .inference import Linear, SugenoFIS, SugenoOutput
from .sets import LinguisticVariable, bell, gaussian

__all__ = ["ANFIS", "ExtrapolationWarning", "MIN_B", "MIN_WIDTH"]

MIN_WIDTH = 1e-3
"""Training floors a width (Gaussian sigma, bell a) at `MIN_WIDTH * (hi - lo)` of its input."""
MIN_B = 1e-2
"""Training floors the bell exponent `b` at `MIN_B`."""

_MFS = ("gaussian", "bell")
_METHODS = ("hybrid", "gradient")
_WARN_RULES = 27  # above 3 inputs x 3 MFs
_MAX_RULES = 1024
_MAX_HALVINGS = 60
_MAX_RIDGE_RETRIES = 30


_EPS = 2.220446049250313e-16  # float64 machine epsilon


class ExtrapolationWarning(RuntimeWarning):
    """[`ANFIS.predict`][bulanik.anfis.ANFIS.predict] was asked about inputs outside the trained ranges.

    A `RuntimeWarning` subclass, emitted once per call and naming the first
    offending sample. The prediction is still returned (the consequents
    extrapolate linearly, so it can be far off); the `SugenoFIS` exported by
    [`to_sugeno`][bulanik.anfis.ANFIS.to_sugeno] instead rejects such inputs.

    Examples:
        >>> import warnings
        >>> from bulanik.anfis import ANFIS, ExtrapolationWarning
        >>> model = ANFIS(1, 2, ranges=[(0, 2)])
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     _ = model.predict([[3.0]])
        >>> caught[0].category is ExtrapolationWarning
        True
    """


def _sq(d: float) -> float:
    return d * d  # inf on overflow, where ``d ** 2`` would raise OverflowError


def _softplus(u: float) -> float:
    return u + math.log1p(math.exp(-u)) if u > 0.0 else math.log1p(math.exp(u))


def _gauss_terms(c: float, s: float, x: float) -> tuple[float, tuple[float, float]]:
    """log mu and d(log mu)/d(c, sigma) for exp(-(x-c)^2 / (2 sigma^2))."""
    z = (x - c) / s
    zz = z * z
    return -0.5 * zz, (z / s, zz / s)


def _bell_terms(a: float, b: float, c: float, x: float) -> tuple[float, tuple[float, float, float]]:
    """log mu and d(log mu)/d(a, b, c) for 1 / (1 + |(x-c)/a|^(2b))."""
    z = (x - c) / a
    if z == 0.0:
        return 0.0, (0.0, 0.0, 0.0)
    lz = math.log(abs(z))
    u = 2.0 * b * lz  # log t
    if u > 700.0:  # t would overflow: stay in the log domain
        log_mu = -_softplus(u)
        s = 1.0 / (1.0 + math.exp(-u))
    else:
        t = abs(z) ** (2.0 * b)
        log_mu = -math.log1p(t)
        s = t / (1.0 + t)  # t * mu
    return log_mu, (2.0 * b * s / a, -2.0 * lz * s, 2.0 * b * s / (a * z))


class ANFIS:
    r"""Grid-partition first-order Sugeno ANFIS (Jang 1993).

    The structure and the formulas are given in the module description. Initial
    centres are `linspace(lo, hi, n)` for each input with $n$ MFs; Gaussian
    $\sigma$ and bell $a$ are $(hi - lo) / (2 (n - 1))$ (or $(hi - lo) / 2$ for
    one MF), bell $b$ is 2, and all consequents are 0. The `history` attribute
    holds the RMSE per epoch of the last [`fit`][bulanik.anfis.ANFIS.fit].

    More than 27 rules emits a `UserWarning` (training cost grows with the rule
    count); more than 1024 raises `ValueError` unless `allow_large=True`.

    Args:
        n_inputs: Number of inputs, an integer of at least 1.
        n_mfs: Membership functions per input: one integer for every input, or
            one per input.
        mf: Membership function family, `"gaussian"` or `"bell"`.
        ranges: Each input's `(lo, hi)` with `lo < hi`. When `None` they are
            taken from the data at the first `fit`.
        allow_large: Allow more than 1024 grid rules.

    Raises:
        ValueError: If `n_inputs` is not an integer of at least 1, `n_mfs` is
            not a positive integer (or `n_inputs` of them), `mf` is unknown,
            `ranges` has the wrong length or an invalid pair, or the grid has
            more than 1024 rules without `allow_large`.

    Attributes:
        history: The training RMSE after each epoch of the last `fit`
            (empty before the first).

    Examples:
        >>> from bulanik.anfis import ANFIS
        >>> model = ANFIS(2, [2, 3], mf="bell", ranges=[(0, 1), (0, 5)])
        >>> model.n_inputs, model.n_rules, model.mf
        (2, 6, 'bell')
        >>> ANFIS(11, 3)
        Traceback (most recent call last):
        ...
        ValueError: grid partition gives 177147 rules (> 1024); pass allow_large=True to build it anyway

    References:
        Jang (1993): "ANFIS: adaptive-network-based fuzzy inference system", IEEE
        Transactions on Systems, Man, and Cybernetics.
    """

    def __init__(
        self,
        n_inputs: int,
        n_mfs: int | Sequence[int] = 2,
        *,
        mf: str = "gaussian",
        ranges: Sequence[tuple[float, float]] | None = None,
        allow_large: bool = False,
    ) -> None:
        if isinstance(n_inputs, bool) or not isinstance(n_inputs, int) or n_inputs < 1:
            raise ValueError(f"n_inputs must be an integer >= 1, got {n_inputs!r}")
        counts = [n_mfs] * n_inputs if isinstance(n_mfs, int) else list(n_mfs)
        if len(counts) != n_inputs or any(
            isinstance(m, bool) or not isinstance(m, int) or m < 1 for m in counts
        ):
            raise ValueError(
                f"n_mfs must be an integer >= 1 or {n_inputs} such integers, got {n_mfs!r}"
            )
        self._mf = choose(mf, _MFS, "mf", "ANFIS")
        n_rules = math.prod(counts)
        if n_rules > _MAX_RULES and not allow_large:
            raise ValueError(
                f"grid partition gives {n_rules} rules (> {_MAX_RULES}); "
                "pass allow_large=True to build it anyway"
            )
        if n_rules > _WARN_RULES:
            warnings.warn(
                f"grid partition gives {n_rules} rules; training cost grows with the rule count",
                UserWarning,
                stacklevel=2,
            )
        self._n = n_inputs
        self._counts = counts
        self._rules = list(itertools.product(*(range(m) for m in counts)))
        self._per_mf = 2 if self._mf == "gaussian" else 3
        self._ranges: list[tuple[float, float]] | None = None
        self._premise: list[list[list[float]]] = []
        self._conseq = [[0.0] * (n_inputs + 1) for _ in self._rules]
        self.history: list[float] = []
        if ranges is not None:
            self._init_premise(self._check_ranges(ranges))

    # ------------------------------------------------------------------ setup

    @property
    def n_inputs(self) -> int:
        """Number of inputs."""
        return self._n

    @property
    def n_rules(self) -> int:
        """Number of grid rules, the product of the MF counts."""
        return len(self._rules)

    @property
    def mf(self) -> str:
        """Membership function family: `"gaussian"` or `"bell"`."""
        return self._mf

    @property
    def ranges(self) -> list[tuple[float, float]] | None:
        """Each input's `(lo, hi)`, or `None` before the first fit without `ranges`.

        Examples:
            >>> from bulanik.anfis import ANFIS
            >>> ANFIS(1, 2).ranges is None
            True
            >>> ANFIS(1, 2, ranges=[(0, 1)]).ranges
            [(0.0, 1.0)]
        """
        return None if self._ranges is None else list(self._ranges)

    def _check_ranges(self, ranges: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        ranges = list(ranges)
        if len(ranges) != self._n:
            raise ValueError(f"ranges needs {self._n} (lo, hi) pairs, got {len(ranges)}")
        out = []
        for i, (lo, hi) in enumerate(ranges):
            lo, hi = finite(lo, "ranges lo"), finite(hi, "ranges hi")
            if not lo < hi:
                raise ValueError(f"ranges of input {i} needs lo < hi, got ({lo}, {hi})")
            out.append((lo, hi))
        return out

    def _init_premise(self, ranges: list[tuple[float, float]]) -> None:
        self._ranges = ranges
        self._premise = []
        for (lo, hi), n in zip(ranges, self._counts):
            span = hi - lo
            width = span / (2 * (n - 1)) if n > 1 else span / 2
            centres = [lo + span * k / (n - 1) for k in range(n)] if n > 1 else [lo]
            if self._mf == "gaussian":
                self._premise.append([[c, width] for c in centres])
            else:
                self._premise.append([[width, 2.0, c] for c in centres])

    def _infer_ranges(self, rows: list[list[float]]) -> None:
        """Take each input's range from checked data ``rows`` unless it already has one."""
        if self._ranges is not None:
            return
        ranges = []
        for i in range(self._n):
            col = [r[i] for r in rows]
            lo, hi = min(col), max(col)
            if not lo < hi:
                raise ValueError(f"input {i} is constant in X ({lo:g}); pass ranges=... explicitly")
            ranges.append((lo, hi))
        self._init_premise(ranges)

    def _need_ranges(self) -> None:
        if self._ranges is None:
            raise RuntimeError("ANFIS has no input ranges yet: pass ranges=... or call fit first")

    def _check_x(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        rows = [list(r) for r in X]
        if not rows:
            raise ValueError("X needs at least one sample")
        for r in rows:
            if len(r) != self._n:
                raise ValueError(f"every sample needs {self._n} input(s), got {len(r)}")
        return [[finite(v, "input value") for v in r] for r in rows]

    def _check_xy(
        self, X: Sequence[Sequence[float]], y: Sequence[float]
    ) -> tuple[list[list[float]], list[float]]:
        rows = self._check_x(X)
        ys = [finite(v, "target") for v in y]
        if len(ys) != len(rows):
            raise ValueError(f"got {len(rows)} samples but {len(ys)} targets")
        return rows, ys

    # ------------------------------------------------------------- parameters

    def param_layout(self) -> list[tuple[str, int, int]]:
        """A `(kind, input, index)` tag per parameter, in [`get_params`][bulanik.anfis.ANFIS.get_params] order.

        Premise parameters come first, input by input and MF by MF: Gaussian
        `("c", i, k)`, `("sigma", i, k)`; bell `("a", i, k)`, `("b", i, k)`,
        `("c", i, k)` for input `i` and MF `k`. Consequents follow, rule by
        rule: `("p", i, r)` for the coefficient of input `i` in rule `r`, then
        `("r", -1, r)` for its constant.

        Returns:
            A list with one tag per parameter.

        Examples:
            >>> from bulanik.anfis import ANFIS
            >>> ANFIS(1, 2).param_layout()
            [('c', 0, 0), ('sigma', 0, 0), ('c', 0, 1), ('sigma', 0, 1), ('p', 0, 0), ('r', -1, 0), ('p', 0, 1), ('r', -1, 1)]
        """
        kinds = ("c", "sigma") if self._mf == "gaussian" else ("a", "b", "c")
        lay = [(kind, i, k) for i, m in enumerate(self._counts) for k in range(m) for kind in kinds]
        for r in range(len(self._rules)):
            lay.extend(("p", i, r) for i in range(self._n))
            lay.append(("r", -1, r))
        return lay

    def get_params(self) -> list[float]:
        """All parameters as a flat list.

        Premise parameters (input-major, MF-major) come first, then the
        consequents (rule-major, `[p_1, ..., p_n, r]` per rule); see
        [`param_layout`][bulanik.anfis.ANFIS.param_layout].

        Returns:
            The parameter values, a copy.

        Raises:
            RuntimeError: If the input ranges are not known yet (pass `ranges`
                or call `fit` first).

        Examples:
            >>> from bulanik.anfis import ANFIS
            >>> ANFIS(1, 2, ranges=[(0, 2)]).get_params()
            [0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            >>> ANFIS(1, 2).get_params()
            Traceback (most recent call last):
            ...
            RuntimeError: ANFIS has no input ranges yet: pass ranges=... or call fit first
        """
        self._need_ranges()
        flat = [v for inp in self._premise for mfp in inp for v in mfp]
        flat.extend(v for c in self._conseq for v in c)
        return flat

    def set_params(self, values: Sequence[float]) -> None:
        """Replace every parameter (layout as in [`get_params`][bulanik.anfis.ANFIS.get_params]).

        Values must be finite numbers (not bools). Widths (sigma, bell a) must
        be at least `MIN_WIDTH * (hi - lo)` of their input and bell b at least
        `MIN_B`, the same floors that training keeps.

        Args:
            values: The new parameter values.

        Raises:
            RuntimeError: If the input ranges are not known yet.
            ValueError: If the count is wrong, a value is a `bool`, not finite,
                or below its floor.

        Examples:
            >>> from bulanik.anfis import ANFIS
            >>> model = ANFIS(1, 2, ranges=[(0, 2)])
            >>> params = model.get_params()
            >>> params[-2:] = [3.0, 1.0]          # last rule: y = 3x + 1
            >>> model.set_params(params)
            >>> model.get_params()[-2:]
            [3.0, 1.0]
            >>> model.set_params([1.0])
            Traceback (most recent call last):
            ...
            ValueError: expected 8 parameters, got 1
        """
        self._need_ranges()
        assert self._ranges is not None
        values = list(values)
        expected = self._n_premise() + len(self._rules) * (self._n + 1)
        if len(values) != expected:
            raise ValueError(f"expected {expected} parameters, got {len(values)}")
        vals = []
        for v, (kind, i, k) in zip(values, self.param_layout()):
            if kind == "r":
                name = f"r of rule {k}"
            elif kind == "p":
                name = f"p of input {i} in rule {k}"
            else:
                name = f"{kind} of input {i} MF {k}"
            if isinstance(v, bool):
                raise ValueError(f"{name} must be a number, got bool {v!r}")
            v = finite(v, name)
            if kind in ("sigma", "a"):
                lo, hi = self._ranges[i]
                least = MIN_WIDTH * (hi - lo)
                if v < least:
                    raise ValueError(f"{name} must be >= MIN_WIDTH * span = {least:g}, got {v}")
            elif kind == "b" and v < MIN_B:
                raise ValueError(f"{name} must be >= MIN_B = {MIN_B:g}, got {v}")
            vals.append(v)
        self._unflatten(vals)

    def _n_premise(self) -> int:
        return sum(self._counts) * self._per_mf

    def _unflatten(self, vals: list[float]) -> None:
        it = iter(vals)
        self._premise = [
            [[next(it) for _ in range(self._per_mf)] for _ in range(m)] for m in self._counts
        ]
        self._conseq = [[next(it) for _ in range(self._n + 1)] for _ in self._rules]

    # ---------------------------------------------------------------- forward

    def _mf_terms(self, x: list[float]) -> tuple[list[list[float]], list[list[tuple[float, ...]]]]:
        logs, derivs = [], []
        for i, inp in enumerate(self._premise):
            li, di = [], []
            for p in inp:
                lm, d = _gauss_terms(p[0], p[1], x[i]) if self._mf == "gaussian" else _bell_terms(
                    p[0], p[1], p[2], x[i]
                )
                li.append(lm)
                di.append(d)
            logs.append(li)
            derivs.append(di)
        return logs, derivs

    def _sample(self, x: list[float]) -> tuple[list[float], list[float], float, list[list[tuple[float, ...]]]]:
        """Normalised firing, rule outputs, output and MF log-derivatives for one sample."""
        logs, derivs = self._mf_terms(x)
        lw = [math.fsum(logs[i][k] for i, k in enumerate(rule)) for rule in self._rules]
        top = max(lw)
        if top == -math.inf:
            raise ValueError(f"no rule fires at input {x}: every firing strength underflows")
        e = [math.exp(v - top) for v in lw]
        total = sum(e)
        wbar = [v / total for v in e]
        zr = [c[-1] + math.fsum(c[i] * x[i] for i in range(self._n)) for c in self._conseq]
        return wbar, zr, math.fsum(w * z for w, z in zip(wbar, zr)), derivs

    def predict(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Crisp outputs for the rows of `X`.

        The model is only trained inside [`ranges`][bulanik.anfis.ANFIS.ranges].
        An input outside them still gets a prediction (the consequents
        extrapolate linearly, so it can be far off), but one
        [`ExtrapolationWarning`][bulanik.anfis.ExtrapolationWarning] (a
        `RuntimeWarning`) names the first offending sample. The `SugenoFIS` exported
        by [`to_sugeno`][bulanik.anfis.ANFIS.to_sugeno] instead rejects such inputs.

        Args:
            X: Samples, each a sequence of `n_inputs` finite numbers.

        Returns:
            One prediction per sample.

        Raises:
            RuntimeError: If the input ranges are not known yet.
            ValueError: If `X` is empty, a sample has the wrong length, a
                value is not finite, or no rule fires at a sample (every
                firing strength underflows).

        Examples:
            >>> from bulanik.anfis import ANFIS
            >>> model = ANFIS(1, 2, ranges=[(0, 2)])
            >>> model.predict([[0.5], [1.5]])      # untrained: every consequent is 0
            [0.0, 0.0]
        """
        self._need_ranges()
        rows = self._check_x(X)
        self._warn_extrapolation(rows)
        return [self._sample(x)[2] for x in rows]

    def _warn_extrapolation(self, rows: list[list[float]]) -> None:
        """Warn once about the first sample outside :attr:`ranges` (beyond rounding noise)."""
        assert self._ranges is not None
        for k, x in enumerate(rows):
            for i, v in enumerate(x):
                lo, hi = self._ranges[i]
                slack = 4 * _EPS * max(abs(lo), abs(hi), 1.0)  # a few ulps of float noise, e.g. 0.1*3 vs 0.3
                if lo - slack <= v <= hi + slack:
                    continue
                warnings.warn(
                    f"ANFIS.predict: sample {k} has input {i} = {_num(v)} outside the "
                    f"trained range [{_num(lo)}, {_num(hi)}]; the prediction extrapolates",
                    ExtrapolationWarning,
                    stacklevel=3,
                )
                return

    def rmse(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> float:
        """Root mean squared error of [`predict`][bulanik.anfis.ANFIS.predict] against `y`.

        Warns on extrapolation like `predict`.

        Args:
            X: Samples, each a sequence of `n_inputs` finite numbers.
            y: One finite target per sample.

        Returns:
            The RMSE.

        Raises:
            RuntimeError: If the input ranges are not known yet.
            ValueError: If `X` is empty, the lengths differ, a value is not
                finite, no rule fires at a sample (every firing strength
                underflows), or the error overflows.

        Examples:
            >>> from bulanik.anfis import ANFIS
            >>> ANFIS(1, 2, ranges=[(0, 2)]).rmse([[0.5], [1.5]], [3.0, 4.0])   # all-zero model
            3.535...
        """
        self._need_ranges()
        rows, ys = self._check_xy(X, y)
        self._warn_extrapolation(rows)
        err = math.sqrt(
            math.fsum(_sq(self._sample(x)[2] - t) for x, t in zip(rows, ys)) / len(rows)
        )
        if not math.isfinite(err):
            raise ValueError(f"RMSE is not finite ({err}); the parameters overflow on this data")
        return err

    def _rmse_or_inf(self, rows: list[list[float]], ys: list[float]) -> float:
        """RMSE on checked data, or ``inf`` where :meth:`rmse` would raise (a penalty value)."""
        try:
            err = math.sqrt(
                math.fsum(_sq(self._sample(x)[2] - t) for x, t in zip(rows, ys)) / len(rows)
            )
        except (OverflowError, ValueError):
            return math.inf
        return err if math.isfinite(err) else math.inf

    # --------------------------------------------------------------- gradient

    def premise_gradient(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> list[float]:
        """Gradient of the mean squared error with respect to the premise parameters.

        In [`get_params`][bulanik.anfis.ANFIS.get_params] order, premise
        parameters only (the consequents are solved by least squares in
        `hybrid` training).

        Args:
            X: Samples, each a sequence of `n_inputs` finite numbers.
            y: One finite target per sample.

        Returns:
            One partial derivative per premise parameter.

        Raises:
            RuntimeError: If the input ranges are not known yet.
            ValueError: If `X` is empty, the lengths differ, a value is not
                finite, or no rule fires at a sample (every firing strength
                underflows).

        Examples:
            >>> from bulanik.anfis import ANFIS
            >>> model = ANFIS(1, 2, ranges=[(0, 2)])
            >>> grad = model.premise_gradient([[0.5], [1.5]], [3.0, 4.0])
            >>> len(grad)
            4
        """
        self._need_ranges()
        rows, ys = self._check_xy(X, y)
        return self._gradient(rows, ys, consequents=False)

    def _gradient(self, rows: list[list[float]], ys: list[float], consequents: bool) -> list[float]:
        n_in, per = self._n, self._per_mf
        gp = [[[0.0] * per for _ in range(m)] for m in self._counts]
        gc = [[0.0] * (n_in + 1) for _ in self._rules] if consequents else []
        scale = 2.0 / len(rows)
        for x, t in zip(rows, ys):
            wbar, zr, f, derivs = self._sample(x)
            err = scale * (f - t)
            if err == 0.0:
                continue
            # d f / d log mu_ik = sum over rules using MF k of input i of wbar_r (z_r - f)
            g = [[0.0] * m for m in self._counts]
            for rule, w, z in zip(self._rules, wbar, zr):
                if w == 0.0:
                    continue
                v = w * (z - f)
                for i, k in enumerate(rule):
                    g[i][k] += v
            for i in range(n_in):
                for k, gik in enumerate(g[i]):
                    if gik:
                        acc, d = gp[i][k], derivs[i][k]
                        for j in range(per):
                            acc[j] += err * gik * d[j]
            for r, w in enumerate(wbar) if consequents else ():
                if w:
                    c = gc[r]
                    ew = err * w
                    for i in range(n_in):
                        c[i] += ew * x[i]
                    c[n_in] += ew
        flat = [v for inp in gp for mfp in inp for v in mfp]
        flat.extend(v for c in gc for v in c)
        return flat

    # --------------------------------------------------------------- training

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        *,
        epochs: int = 50,
        method: str = "hybrid",
        lr: float = 0.01,
        ridge: float = 1e-8,
    ) -> list[float]:
        """Train for `epochs` epochs; return (and store as `history`) the RMSE per epoch.

        Training is deterministic and continues from the current parameters.
        `hybrid` solves the consequents by ridge least squares before the first
        epoch, then each epoch takes one gradient step on the premise and solves
        the consequents again; `gradient` takes one gradient step on all
        parameters per epoch. Each history entry is the RMSE of the model as it
        stands after that epoch, so the last one is the RMSE of the returned model.
        If the model has no `ranges` yet they are taken from `X`.

        The least-squares step normalises every input to $[0, 1]$ over its
        range, so it is invariant to input offset and scale; `ridge` applies
        to the normalised coefficients. If the normal matrix is singular to
        working precision (for example `ridge=0` with a rank-deficient design),
        the ridge is raised automatically (10x per retry, from at least 1e-12
        times the largest diagonal entry) instead of failing. The gradient step
        `lr` is *not* scale-invariant: rescaling an input changes the effective
        step.

        A step that would raise the training RMSE, or make it or a parameter
        non-finite, is halved until it does not, and dropped after 60 halvings
        (or at once for a non-finite gradient). The next epoch starts from twice
        the step taken, up to `lr`. So the gradient method's history
        never rises, and neither does the hybrid history apart from the small
        effect of `ridge`. A step shrunk that far no longer changes the parameters,
        so a dropped step means `lr` is far too large; a `fit` that dropped any
        emits one `RuntimeWarning` with the count.

        Args:
            X: Training samples, each a sequence of `n_inputs` finite numbers.
            y: One finite target per sample.
            epochs: Number of epochs, an integer of at least 1.
            method: `"hybrid"` or `"gradient"`.
            lr: Largest gradient step tried per epoch, positive.
            ridge: Ridge penalty of the least-squares step, non-negative.

        Returns:
            The RMSE after each epoch (a copy of `history`).

        Raises:
            ValueError: If an option is invalid (`method`, `epochs`, `lr`,
                `ridge`), `X` is empty, a sample has the wrong length, the
                lengths of `X` and `y` differ, a value is not finite, an
                input is constant in `X` while `ranges` is unknown, or no rule
                fires at a sample (every firing strength underflows).

        Examples:
            >>> from bulanik.anfis import ANFIS
            >>> X = [[i / 10] for i in range(21)]
            >>> y = [x[0] ** 2 for x in X]
            >>> model = ANFIS(1, 3, ranges=[(0, 2)])
            >>> model.rmse(X, y) > 1
            True
            >>> history = model.fit(X, y, epochs=5)
            >>> len(history), history == model.history
            (5, True)
            >>> round(model.rmse(X, y), 2)
            0.01
            >>> tuned = ANFIS(1, 3, mf="bell", ranges=[(0, 2)])
            >>> losses = tuned.fit(X, y, epochs=3, method="gradient", lr=0.05)
            >>> losses[-1] < losses[0]
            True
        """
        method = choose(method, _METHODS, "method", "ANFIS.fit")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"epochs must be an integer >= 1, got {epochs!r}")
        lr = finite(lr, "lr")
        if not lr > 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        ridge = finite(ridge, "ridge")
        if ridge < 0.0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")
        rows, ys = self._check_xy(X, y)
        self._infer_ranges(rows)
        history = []
        n_prem = self._n_premise()
        skipped = 0
        hybrid = method == "hybrid"
        if hybrid:
            self._solve_consequents(rows, ys, ridge)
        start = lr
        for _ in range(epochs):
            grad = self._gradient(rows, ys, consequents=not hybrid)
            err, taken = self._step(rows, ys, grad, start, n_prem)
            if taken is None:
                skipped += 1
            else:
                start = min(lr, 2.0 * taken)
            if hybrid:
                self._solve_consequents(rows, ys, ridge)
                err = self._rmse_or_inf(rows, ys)
            history.append(err)
        self.history = history
        if skipped:
            warnings.warn(
                f"ANFIS.fit skipped {skipped} of {epochs} gradient steps: none lowered the "
                f"error within {_MAX_HALVINGS} halvings or the gradient was not finite; "
                "lower lr",
                RuntimeWarning,
                stacklevel=2,
            )
        return list(history)

    def _solve_consequents(self, rows: list[list[float]], ys: list[float], ridge: float) -> None:
        assert self._ranges is not None
        los = [lo for lo, _ in self._ranges]
        spans = [hi - lo for lo, hi in self._ranges]
        design = []
        for x in rows:
            wbar = self._sample(x)[0]
            u = [(v - lo) / s for v, lo, s in zip(x, los, spans)]  # inputs on [0, 1]
            row: list[float] = []
            for w in wbar:
                row.extend(w * v for v in u)
                row.append(w)
            design.append(row)
        gram, aty = normal_equations(design, ys)
        bump = ridge
        floor = 1e-12 * max(gram[i][i] for i in range(len(gram)))
        for _ in range(_MAX_RIDGE_RETRIES):
            a = [list(r) for r in gram]
            for i in range(len(a)):
                a[i][i] += bump
            try:
                theta = solve_spd(a, aty)
                break
            except ValueError:  # singular to working precision: raise the ridge
                bump = max(bump * 10.0, floor)
        else:  # pragma: no cover - a 1e18-relative ridge is always positive definite
            return
        # back from u = (x - lo) / span: p = p' / span, r = r' - sum(p' lo / span)
        it = iter(theta)
        conseq = []
        for _ in self._rules:
            pn = [next(it) for _ in range(self._n)]
            rn = next(it)
            p = [v / s for v, s in zip(pn, spans)]
            conseq.append(p + [rn - math.fsum(v * lo for v, lo in zip(p, los))])
        self._conseq = conseq

    def _floor(self, vals: list[float], layout: list[tuple[str, int, int]]) -> None:
        assert self._ranges is not None
        for j, (kind, i, _) in enumerate(layout):
            if kind in ("sigma", "a"):
                lo, hi = self._ranges[i]
                vals[j] = max(vals[j], MIN_WIDTH * (hi - lo))
            elif kind == "b":
                vals[j] = max(vals[j], MIN_B)

    def _step(
        self, rows: list[list[float]], ys: list[float], grad: list[float], lr: float, n_prem: int
    ) -> tuple[float, float | None]:
        """One descent step: the RMSE after it and the step size taken (``None`` if dropped)."""
        old = self.get_params()
        layout = self.param_layout()
        current = self._rmse_or_inf(rows, ys)
        if all(math.isfinite(g) for g in grad):
            step = lr
            for _ in range(_MAX_HALVINGS):
                cand = list(old)
                for j, g in enumerate(grad):
                    cand[j] -= step * g
                self._floor(cand, layout[:n_prem])
                if all(math.isfinite(v) for v in cand):
                    self._unflatten(cand)
                    err = self._rmse_or_inf(rows, ys)
                    if math.isfinite(err) and err <= current:
                        return err, step
                step *= 0.5
        self._unflatten(old)
        return current, None

    # ----------------------------------------------------------------- export

    def to_sugeno(
        self, input_names: Sequence[str] | None = None, output_name: str = "y"
    ) -> SugenoFIS:
        """An equivalent [`SugenoFIS`][bulanik.inference.sugeno.SugenoFIS] (`and_method="prod"`).

        One variable per input over its range with terms `mf1..mfn`, one
        `Linear` output term `r1..rR` per rule. It reproduces
        [`predict`][bulanik.anfis.ANFIS.predict] inside the ranges (the FIS
        rejects inputs outside). Where every membership underflows to 0 in plain
        floats the two differ: ANFIS still predicts (its firing is normalised in
        the log domain) while the FIS raises `NoRuleFiredError`.

        Args:
            input_names: Names for the input variables; defaults to `x1`, `x2`, ...
            output_name: Name of the output variable.

        Returns:
            The exported fuzzy inference system.

        Raises:
            RuntimeError: If the input ranges are not known yet.
            ValueError: If `input_names` has the wrong length, or a name is
                not a valid rule name.

        Examples:
            >>> from bulanik.anfis import ANFIS
            >>> X = [[i / 10] for i in range(21)]
            >>> y = [x[0] ** 2 for x in X]
            >>> model = ANFIS(1, 3, ranges=[(0, 2)])
            >>> _ = model.fit(X, y, epochs=5)
            >>> fis = model.to_sugeno(["x"])
            >>> print(fis.rules[0])
            IF x IS mf1 THEN y IS r1
            >>> abs(fis.evaluate({"x": 1.0})["y"] - model.predict([[1.0]])[0]) < 1e-9
            True
        """
        self._need_ranges()
        assert self._ranges is not None
        names = [f"x{i + 1}" for i in range(self._n)] if input_names is None else list(input_names)
        if len(names) != self._n:
            raise ValueError(f"input_names needs {self._n} names, got {len(names)}")
        make = gaussian if self._mf == "gaussian" else bell
        variables = [
            LinguisticVariable(
                name, rng, {f"mf{k + 1}": make(*p) for k, p in enumerate(self._premise[i])}
            )
            for i, (name, rng) in enumerate(zip(names, self._ranges))
        ]
        output = SugenoOutput(
            output_name,
            {
                f"r{r + 1}": Linear(dict(zip(names, c[: self._n])), c[self._n])
                for r, c in enumerate(self._conseq)
            },
        )
        rules = [
            "IF "
            + " AND ".join(f"{names[i]} IS mf{k + 1}" for i, k in enumerate(rule))
            + f" THEN {output_name} IS r{r + 1}"
            for r, rule in enumerate(self._rules)
        ]
        return SugenoFIS(variables, [output], rules, and_method="prod")
