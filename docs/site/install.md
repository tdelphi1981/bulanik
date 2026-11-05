# Install

bulanik is pure Python with **no runtime dependencies**. The core installs on its
own; three extras add the optional parts.

## From PyPI

```bash
pip install bulanik                 # core: numbers, aggregation, sets, inference, type-2, ANFIS, MCDM
pip install "bulanik[lumix]"        # + the LumiX bridge (bulanik.lumix)
pip install "bulanik[sezgi]"        # + the sezgi bridge (bulanik.sezgi)
pip install "bulanik[plot]"         # + figures (bulanik.plot)
pip install "bulanik[lumix,sezgi,plot]"   # everything
```

With [uv](https://docs.astral.sh/uv/): `uv add bulanik` or
`uv add "bulanik[lumix,sezgi,plot]"`.

| Extra | Adds | Installs |
|---|---|---|
| *(none)* | everything except the three parts below | nothing |
| `lumix` | [`bulanik.lumix`](api/lumix.md): linear BWM, Zimmermann fuzzy programming, scores as LumiX data | `lumix-opt>=0.3.2`, `ortools>=9.8` |
| `sezgi` | [`bulanik.sezgi`](api/sezgi.md): nonlinear and fuzzy BWM, ANFIS training with metaheuristics | `sezgi>=0.1.6` |
| `plot` | [`bulanik.plot`](api/plot.md): membership, orthopair, inference and MCDM figures in a light and a dark theme | `matplotlib>=3.8` |

Importing a bridge or `bulanik.plot` without its extra raises an `ImportError`
that names the extra to install. The rest of bulanik never imports them.

## Python versions

bulanik supports Python **3.10, 3.11, 3.12 and 3.13**. Building *this
documentation site* needs Python 3.11 or newer, because the docs build pins a
matplotlib release that has no Python 3.10 wheel.

## From a checkout (development)

```bash
git clone https://github.com/tdelphi1981/bulanik.git
cd bulanik
uv sync                      # core + the dev group (pytest, hypothesis, LumiX, OR-Tools, sezgi, matplotlib)
uv run pytest -q             # tests and doctests

# or with pip
pip install -e ".[lumix,sezgi,plot]"
```

To build this site locally (Python 3.11 or newer):

```bash
uv sync --group docs
uv run --group docs mkdocs serve          # live preview
uv run --group docs mkdocs build --strict # the same check CI runs
```

The `docs` group includes the bridge packages and matplotlib, so every code block
in the documentation runs when the site is built.

## Verify the install

This block runs when the site is built, so what you see is real output from
the installed version:

```python exec="on" source="above" result="text"
import importlib.util

import bulanik

print("bulanik", bulanik.__version__)
for module, extra in [("lumix", "lumix"), ("sezgi", "sezgi"), ("matplotlib", "plot")]:
    found = importlib.util.find_spec(module) is not None
    print(f"extra {extra!r}: {'available' if found else 'not installed'}")
```
