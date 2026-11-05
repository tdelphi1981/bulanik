import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# file name -> (expected stdout substring, required module or None)
EXAMPLES: dict[str, tuple[str, str | None]] = {
    "01_fuzzy_numbers.py": ("IFN sum", None),
    "02_scale_and_experts.py": ("aggregated:", None),
    "03_fuzzy_ahp.py": ("CR =", None),
    "04_swara_topsis_supplier.py": ("best supplier:", None),
    "05_intuitionistic_pipeline.py": ("ranking:", None),
    "06_zimmermann_lumix.py": ("lambda = 0.8333", "lumix"),
    "07_bwm_lumix.py": ("xi = 0.0625", "lumix"),
    "08_fuzzy_bwm_sezgi.py": ("k* = 0.2087", "sezgi"),
    "09_weights_to_lumix.py": ("selected: Cedar, Delta", "lumix"),
    "10_mamdani_tipper.py": ("tip(service=3, food=8) = 15.86", None),
    "11_sugeno_controller.py": ("u(e=2, de=-1) = 2.2125", None),
    "12_tune_sugeno_with_sezgi.py": ("fitted c1=1.00 c2=21.00", "sezgi"),
    "13_tsukamoto.py": ("z(x=3) = 39.80", None),
    "14_it2_vs_t1_noise.py": ("IT2 RMSE", None),
    "15_anfis_sinc.py": ("final RMSE", None),
    "16_anfis_with_sezgi.py": ("sezgi RMSE", "sezgi"),
    "17_plotting.py": ("saved 10 figures", "matplotlib"),  # the plot extra
    "18_interval_valued_ahp_codas.py": ("IVIFN ranking:", None),
    "19_it2_ahp_marcos.py": ("IT2 ranking:", None),
    "20_ranking_methods_compared.py": ("rank table", None),
    "21_critic_piprecia.py": ("COPRAS ranking:", None),
    "22_fuzzy_dematel.py": ("causes:", None),
    "23_interval_valued_ahp.py": ("weights compared", None),
    "24_fuzzy_ranking.py": ("fuzzy ranking table", None),
}


@pytest.mark.parametrize("name", sorted(EXAMPLES) or [None])
def test_example(name):
    if name is None:
        pytest.skip("no examples registered yet")
    expected, required = EXAMPLES[name]
    if required is not None and importlib.util.find_spec(required) is None:
        pytest.skip(f"{required} is not installed")
    result = subprocess.run(
        [sys.executable, f"examples/{name}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


_EXTRA_MODULES = {"lumix": {"lumix", "ortools"}, "sezgi": {"sezgi"}, "matplotlib": {"matplotlib"}}


def _foreign_imports(source: str, required: str | None) -> list[str]:
    """Top-level modules imported by ``source`` beyond stdlib, bulanik and its extra."""
    import ast

    allowed = set(sys.stdlib_module_names) | {"bulanik"} | _EXTRA_MODULES.get(required, set())
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods = [node.module or ""]
        else:
            continue
        found += [m for m in mods if m.split(".")[0] not in allowed]
    return found


def test_examples_import_only_runtime_dependencies():
    """Examples must run for a plain user who installed an extra from a checkout."""
    for path in sorted((REPO / "examples").glob("*.py")):
        # each example may import only the extra it declares in EXAMPLES
        required = EXAMPLES[path.name][1]
        assert _foreign_imports(path.read_text(), required) == [], path.name


def test_import_guard_is_per_example():
    assert _foreign_imports("import sezgi", "sezgi") == []
    assert _foreign_imports("import sezgi", "lumix") == ["sezgi"]
    assert _foreign_imports("from lumix import X", None) == ["lumix"]
    assert _foreign_imports("import numpy", "lumix") == ["numpy"]
