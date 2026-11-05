# Conventions

What every part of bulanik promises, so the guide pages do not have to repeat it.

## Immutability

Every fuzzy number is a frozen dataclass: you cannot assign to a field, and
every operation returns a **new** number. Numbers compare by value and are
hashable, so they work as dictionary keys and set members. The same holds for
membership functions, rules and terms, interval type-2 sets and the result
objects of [`bulanik.mcdm`][bulanik.mcdm] (`WeightingResult`, `RankingResult`,
`BWMResult`).

```python exec="on" source="above" result="text"
from bulanik import TFN

a = TFN(1, 2, 3)
b = a + TFN(1, 1, 1)
print(a, b)                          # a is unchanged
print(a == TFN(1, 2, 3), hash(a) == hash(TFN(1, 2, 3)))
try:
    a.l = 5
except Exception as error:
    print(type(error).__name__ + ":", error)
```

Stateful objects are the exception, and they say so: a trained
[`ANFIS`][bulanik.anfis.ANFIS] changes its parameters (and `history`) in `fit`
and `set_params`.

## Validation and error types

Inputs are checked when an object is built or a function is called, never later.
Two exception types carry nearly all the information:

| Type | Raised for |
|---|---|
| `TypeError` | A value of the wrong **kind**: a `bool` or `str` where a number is expected, a number of another type or rung in a binary operation (`TFN + IFN`, `IFN + PFN`), a non-`TFN` where a `TFN` is required. |
| `ValueError` | A right kind but a bad **value**: NaN or infinity, a degree outside $[0, 1]$, a violated constraint (`l <= m <= u`, $\mu^q + \nu^q \le 1$), an unknown method name, an empty or ragged table, mismatched lengths, weights that do not sum to 1. |

Subclasses of `ValueError` make a specific failure catchable:
[`NoRuleFiredError`][bulanik.inference.NoRuleFiredError] (no rule fires for an
output; its `.explanation` shows why) and
[`NoLowerFiringError`][bulanik.type2.NoLowerFiringError] (the `bmm` type reducer
without any positive lower firing). Two more types appear at the edges:
`RuntimeError` when a solver finds no optimum (the bridges) or the ANFIS ranges
are not known yet, and `ImportError` when a bridge or `bulanik.plot` is imported
without its extra.

Booleans are rejected as numbers on purpose (`True` would silently mean 1), and
NaN and infinity are always rejected. Messages name the argument and the
offending value:

```python exec="on" source="above" result="text"
from bulanik import IFN, TFN

def show(make):
    try:
        make()
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: TFN(3, 2, 1))                 # order violated
show(lambda: TFN(1, float("nan"), 3))      # not finite
show(lambda: TFN(True, 2, 3))              # bool is not a number
show(lambda: TFN("1", 2, 3))               # nor is a string
show(lambda: IFN(0.5, 0.4) + TFN(1, 2, 3)) # numbers of different types
show(lambda: IFN(1.2, 0.0))                # degree outside [0, 1]
```

## Tolerance and boundary projection

Floating-point rounding can push a value that is mathematically on a constraint
slightly past it. bulanik treats this in two steps, both with the module constant
`TOL = 1e-12`:

1. **Inputs and results within `TOL` of the legal range are accepted.** A degree
   in $[-10^{-12}, 1 + 10^{-12}]$ is clamped to $[0, 1]$. A pair $(\mu, \nu)$ with
   $\mu^q + \nu^q \le 1 + 10^{-12}$ is accepted as rounding. Anything beyond
   raises `ValueError`.
2. **Orthopair numbers are projected back onto the boundary** when the excess is
   more than 4 ulps. One ulp (unit in the last place of a float) is
   $2.2 \times 10^{-16}$ at 1, so the threshold is about $8.9 \times 10^{-16}$. The pair is rescaled so that
   $\mu^q + \nu^q = 1$. An excess of at most 4 ulps is left as given, so a
   deliberate boundary value such as `PFN(0.6, 0.8)` stays bit-identical. The
   projection matters because an amplifying operation (`power`, `scale` by
   $k > 1$) would otherwise magnify the excess until a later check rejects a
   value the constructor had accepted.

```python exec="on" source="above" result="text"
from bulanik import IFN

on_boundary = IFN(0.6, 0.4)
within_ulps = IFN(0.6, 0.4 + 2.0**-52)         # 1 ulp past: kept as given
projected = IFN(0.6, 0.4 + 1e-13)              # within TOL, more than 4 ulps: projected
print(repr(within_ulps.nu), repr(projected.mu), repr(projected.nu))
print(projected.mu + projected.nu)             # back on the boundary
try:
    IFN(0.6, 0.4 + 1e-11)                      # beyond TOL: an error
except ValueError as error:
    print("ValueError:", error)
```

Operation results get the same treatment: a result that misses its constraint
only by rounding is put back on the boundary, and one that misses it by more
raises `ValueError` (for example a picture fuzzy number scaled by $k < 1$ can leave
its constraint set).

## Named operator families, scores, accuracies and distances

Everything that has more than one definition is selected by a **name**, a plain
string: `ops="einstein"`, `x.score("normalized")`, `x.distance(y, "euclidean")`.
Every number class lists the names it supports in class attributes (`OPS`,
`SCORES`, `ACCURACIES`, `DISTANCES`) and its defaults (`DEFAULT_SCORE` and so on).
The table below is generated from those attributes; the default of each column is
in **bold**, and for operator families the first name is the default.

```python exec="on"
from bulanik import FFN, IFN, PFN, QROFN, TFN, NeutrosophicFN, PictureFN, SphericalFN, TrFN


def names(values, default):
    if not values:
        return "*none*"
    return ", ".join(f"**{v}**" if v == default else v for v in values)


print("| Type | Operator families (`ops`) | Scores | Accuracies | Distances |")
print("|---|---|---|---|---|")
for cls in (TFN, TrFN, IFN, PFN, FFN, QROFN, SphericalFN, PictureFN, NeutrosophicFN):
    print(
        f"| `{cls.__name__}` | {names(cls.OPS, cls.OPS[0])} | {names(cls.SCORES, cls.DEFAULT_SCORE)} "
        f"| {names(cls.ACCURACIES, cls.DEFAULT_ACCURACY)} | {names(cls.DISTANCES, cls.DEFAULT_DISTANCE)} |"
    )
```

`IFN`, `PFN`, `FFN` and `QROFN` share one implementation, so they have the same
names (applied to $\mu^q$ and $\nu^q$). A number type without accuracy functions
(`TFN`, `TrFN`) raises `ValueError` from `accuracy()`; the formulas of each name are
on the [user guide](../guide/numbers-triangular.md) pages.

### The naming rule

1. A name is always a string and always optional where a default exists:
   `None` means "the default".
2. An **unknown name raises `ValueError`** that lists every valid name:
   `unknown ops 'foo' for IFN; valid: algebraic, einstein`.
3. MCDM results **report the names they used** in `result.operators`, so a ranking can
   be reproduced from its output.

```python exec="on" source="above" result="text"
from bulanik import IFN, TFN

def show(make):
    try:
        make()
    except Exception as error:
        print(f"{type(error).__name__}: {error}")

show(lambda: IFN(0.6, 0.3).add(IFN(0.5, 0.4), ops="foo"))
show(lambda: TFN(1, 2, 3).score("foo"))
show(lambda: TFN(1, 2, 3).accuracy())
print(IFN(0.6, 0.3).score(), IFN(0.6, 0.3).score(None), IFN(0.6, 0.3).score("normalized"))
```

### Names in the other modules

The same rule covers the method options of the sets, inference, type-2, ANFIS and
MCDM modules. This table is generated from the code too (the valid names come from
the modules, the defaults from the function signatures):

```python exec="on"
import inspect

from bulanik import anfis, mcdm, sets, type2
from bulanik.aggregate import aggregate_experts
from bulanik.anfis import ANFIS
from bulanik.inference import MamdaniFIS, SugenoFIS, TsukamotoFIS, mamdani, rules, sugeno
from bulanik.mcdm import rank_fuzzy
from bulanik.type2 import IT2MamdaniFIS, IT2SugenoFIS, type_reduce


def default(fn, parameter):
    value = inspect.signature(fn).parameters[parameter].default
    return f"`{value}`" if isinstance(value, str) else "*required*"


def row(where, parameter, valid, fn):
    shown = ", ".join(valid)
    print(f"| `{where}` | `{parameter}` | {shown} | {default(fn, parameter)} |")


print("| Where | Option | Valid names | Default |")
print("|---|---|---|---|")
row("t_norm", "kind", sets._T_NORMS, sets.t_norm)
row("t_conorm", "kind", sets._T_CONORMS, sets.t_conorm)
row("MamdaniFIS, SugenoFIS, TsukamotoFIS", "and_method", rules._AND_METHODS, MamdaniFIS.__init__)
row("MamdaniFIS, SugenoFIS, TsukamotoFIS", "or_method", rules._OR_METHODS, MamdaniFIS.__init__)
row("MamdaniFIS", "implication", mamdani._IMPLICATION, MamdaniFIS.__init__)
row("MamdaniFIS", "aggregation", mamdani._AGGREGATION, MamdaniFIS.__init__)
row("MamdaniFIS, defuzzify_samples", "defuzzification", mamdani._DEFUZZ, MamdaniFIS.__init__)
row("SugenoFIS", "output_method", sugeno._OUTPUT, SugenoFIS.__init__)
row("IT2MamdaniFIS, IT2SugenoFIS", "type_reduction", type2._METHODS, IT2MamdaniFIS.__init__)
row("type_reduce", "method", type2._METHODS, type_reduce)
row("IT2MamdaniFIS", "implication", type2._IMPLICATION, IT2MamdaniFIS.__init__)
row("IT2MamdaniFIS", "aggregation", type2._AGGREGATION, IT2MamdaniFIS.__init__)
row("ANFIS", "mf", anfis._MFS, ANFIS.__init__)
row("ANFIS.fit", "method", anfis._METHODS, ANFIS.fit)
row("aggregate_experts", "method", ("wa", "wg"), aggregate_experts)
row("rank_fuzzy", "method", tuple(mcdm.fuzzy._METHODS), rank_fuzzy)
row("rank_fuzzy", "aggregation", ("wa", "wg"), rank_fuzzy)
```

Criterion types are the names `"benefit"` and `"cost"`; `FuzzyGoal.sense` takes
`"ge"`, `"le"` or `"eq"`; rule keywords (`IF`, `IS`, `AND`, `OR`, `NOT`, `THEN`,
`WITH`) are case-insensitive while variable and term names are case-sensitive.

## Warnings

bulanik raises errors for wrong input and keeps warnings for results that are
valid but probably not what you want. All of them are `RuntimeWarning` or
`UserWarning`, so the standard `warnings` filters apply.

| Warning | Emitted when |
|---|---|
| [`ExtrapolationWarning`][bulanik.anfis.ExtrapolationWarning] (a `RuntimeWarning`) | `ANFIS.predict` or `rmse` is asked about an input outside the trained ranges. One warning per call, naming the first offending sample; the prediction is still returned. The `SugenoFIS` exported by `to_sugeno` rejects such inputs instead. |
| `RuntimeWarning` | `ANFIS.fit` skipped gradient steps. A step whose result would be non-finite is halved up to 60 times and skipped after that (or at once for a non-finite gradient); the history then repeats the previous RMSE. One warning per `fit`, with the count. Lower `lr`. |
| `UserWarning` | An `ANFIS` grid partition produces more than 27 rules (more than 1024 raises `ValueError` unless `allow_large=True`). |

```python exec="on" source="above" result="text"
import warnings

from bulanik.anfis import ANFIS

X = [[i / 10] for i in range(21)]
y = [x[0] ** 2 for x in X]

model = ANFIS(1, 3, ranges=[(0, 2)])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model.predict([[5.0]])                       # outside the trained range [0, 2]
print(caught[0].category.__name__, "-", caught[0].message)

wild = ANFIS(1, 3, mf="bell", ranges=[(0, 2)])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    wild.fit(X, y, epochs=3, method="gradient", lr=1e100)   # absurd step size
print(caught[0].category.__name__, "-", caught[0].message)
```
