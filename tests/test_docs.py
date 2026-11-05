"""The documentation site builds cleanly in strict mode."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

SITE = REPO / "docs" / "site"
FIGURE_REF = re.compile(r"assets/figures/([\w.-]+?)-(?:light|dark)\.svg")


def test_referenced_figures_exist_in_both_themes():
    refs = {
        (page.relative_to(SITE).as_posix(), name)
        for page in SITE.rglob("*.md")
        for name in FIGURE_REF.findall(page.read_text(encoding="utf-8"))
    }
    assert refs, "no figures referenced; the gallery should reference some"
    missing = [
        f"{page}: {name}-{mode}.svg"
        for page, name in sorted(refs)
        for mode in ("light", "dark")
        if not (SITE / "assets" / "figures" / f"{name}-{mode}.svg").is_file()
    ]
    assert not missing, missing


@pytest.mark.docs
@pytest.mark.skipif(importlib.util.find_spec("mkdocs") is None, reason="docs dependency group not installed")
def test_strict_build(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(tmp_path / "site")],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "WARNING" not in output, output


def test_sdist_excludes_planning_docs():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    sdist = config["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert "docs/superpowers" in sdist["exclude"]
    assert ".github" in sdist["exclude"]
