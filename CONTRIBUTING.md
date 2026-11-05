# Contributing to bulanik

## Development setup

```bash
git clone https://github.com/tdelphi1981/bulanik
cd bulanik
uv sync            # dev group: pytest, hypothesis, coverage, matplotlib, lumix-opt, ortools, sezgi
```

## Tests

```bash
uv run python -m pytest -q
```

The suite runs every script in `examples/`. Coverage is measured only when you ask for
it, and then it must stay at or above 95% (`fail_under` in `pyproject.toml`):

```bash
uv run python -m pytest -q --cov=bulanik
```

Do not run pytest with `-W error`: OR-Tools' SWIG bindings emit warnings that
turn into crashes. Property tests use Hypothesis.

## Documentation

The site is MkDocs Material, sources in `docs/site/`, configuration in `mkdocs.yml`.

```bash
uv sync --group docs
uv run mkdocs serve                # live preview
uv run mkdocs build --strict       # must finish with zero warnings
```

`tests/test_docs.py` runs the strict build only when the docs group is installed
(`uv sync --group docs`); without it the test is skipped. Deselect it with
`-m "not docs"`. Building the site needs **Python 3.11 or newer** (the docs group pins a
matplotlib release without Python 3.10 wheels) and **network access** (the privacy plugin
downloads external assets such as the MathJax script). Python snippets in the docs are executed at
build time with markdown-exec, so the output shown is real.

## Style

- Python 3.10+, type hints on public API, no runtime dependencies in the core.
- Numbers and sets are immutable; invalid input raises a specific error, never returns silently.
- Docstrings use Google style (`Args:`, `Returns:`, `Raises:`, `Examples:`).
- Cited works need a verified reference: add them to `docs/site/about/references.md` with a
  DOI that resolves (check it against Crossref or the publisher), never from memory.
- Keep commits focused; do not change library behaviour in documentation commits.
