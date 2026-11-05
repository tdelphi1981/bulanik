# Benchmarks

Scripts comparing bulanik with other libraries and public data. They need network or third-party
packages, so they are not part of the test suite or the sdist. Each one runs alone from the
repository root and rewrites its result file in `tests/benchmarks/data/`. The offline tests in
`tests/benchmarks/` and the docs page `docs/site/guide/benchmarks.md` read those files, so neither
needs the network.

`--frozen` keeps `uv.lock` untouched, and `--isolated --no-dev` keeps the extra packages out of
the project environment. bulanik itself stays stdlib-only.

| Script | Command |
|---|---|
| `mcdm_libraries.py` | `uv run --frozen --isolated --no-dev --with pymcdm==1.4.0 --with pyrepo-mcda==0.1.15 python benchmarks/mcdm_libraries.py` |
| `hdi_undp.py` | `uv run --frozen --isolated --no-dev python benchmarks/hdi_undp.py` |
| `anfis_mackey_glass.py` | `uv run --frozen --isolated --no-dev python benchmarks/anfis_mackey_glass.py` |
| `inference_libraries.py` | `uv run --frozen --isolated --no-dev --with simpful==2.12.0 --with scikit-fuzzy==0.5.0 --with pymcdm==1.4.0 python benchmarks/inference_libraries.py` |
| `anfis_regression.py` | `uv run --frozen --isolated --no-dev --with scikit-learn==1.9.1 python benchmarks/anfis_regression.py` |

`inference_libraries.py` installs pymcdm only because scikit-fuzzy 0.5.0 declares no dependencies
and imports `packaging`, which pymcdm brings in through matplotlib. It does not install networkx,
so `skfuzzy.control` is not run; see the script's docstring.

Downloads go to the git-ignored `benchmarks/.cache/`, and `_download.py` checks each file against
the SHA-256 recorded in the script. Run times in the result files are from one machine and change
between runs; everything else is deterministic for fixed package versions.

Data and third-party licences are in `THIRD_PARTY_NOTICES.md`. That file is copied to
`tests/benchmarks/`, because the copied values ship in the sdist with the tests and this directory
does not.
