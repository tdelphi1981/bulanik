"""The generated API index reads cleanly: no raw markup, no truncated sentences."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("griffe") is None, reason="docs dependency group not installed"
)


def _summaries():
    proc = subprocess.run(
        [sys.executable, "docs/scripts/api_index.py"],
        cwd=REPO, capture_output=True, text=True, timeout=300, check=True,
    )
    rows = re.findall(r"<tr><td>.*?</td><td>[^<]*</td><td>(.*?)</td></tr>", proc.stdout, re.S)
    assert len(rows) > 100
    return rows


def _outside_math(cell):
    """The cell without rendered math and code spans (where backslashes and brackets are legitimate)."""
    cell = re.sub(r'<span class="arithmatex">.*?</span>', "", cell, flags=re.S)
    return re.sub(r"<code>.*?</code>", "", cell, flags=re.S)


def test_summaries_are_clean():
    for cell in _summaries():
        text = _outside_math(cell)
        for bad in ("\\", "][", "mathbb", "frac", "operatorname", "$"):
            assert bad not in text, (bad, cell)
        plain = re.sub(r"<[^>]+>", "", cell).strip()
        assert plain, cell
        assert not plain.endswith(("et al.", "...", "e.g.", "i.e.", "(")), cell
        # math, when present, must be balanced
        assert cell.count("\\(") == cell.count("\\)"), cell
