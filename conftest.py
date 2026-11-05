"""Root pytest configuration: doctest collection over ``src/bulanik``.

``bulanik.lumix`` and ``bulanik.sezgi`` import optional extras at module level,
so collecting their doctests would fail without the extra. Skip a bridge module
when its dependencies are not importable; likewise ``bulanik.plot`` without
matplotlib. With matplotlib, doctests and tests draw on the non-interactive
Agg backend and every figure is closed after each test.
"""

import importlib.util

import pytest

collect_ignore = []

if any(importlib.util.find_spec(m) is None for m in ("lumix", "ortools")):
    collect_ignore.append("src/bulanik/lumix.py")
if importlib.util.find_spec("sezgi") is None:
    collect_ignore.append("src/bulanik/sezgi.py")
if importlib.util.find_spec("matplotlib") is None:
    collect_ignore.append("src/bulanik/plot")
else:
    import matplotlib

    matplotlib.use("Agg")

    @pytest.fixture(autouse=True)
    def _close_matplotlib_figures():
        yield
        import matplotlib.pyplot as plt

        plt.close("all")
