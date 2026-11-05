# Benchmarks

This page compares bulanik with other open-source libraries and with public data. It is there
to show how the library behaves, including where it differs and where it does poorly. The tables are
read from the result files in `tests/benchmarks/data/` when the docs are built. The scripts that
produce them are in `benchmarks/`; [its README](https://github.com/tdelphi1981/bulanik/tree/main/benchmarks)
gives the commands. Offline tests in `tests/benchmarks/` keep the frozen values honest.

**In short**

- On matching formulas, every crisp MCDM method gives the same scores as pymcdm and pyrepo-mcda, to
  rounding, on 209 matrices. Most real differences are choices of variant; pymcdm's COPRAS uses another formula.
- bulanik is stricter than the libraries about rounded weights, and it has no min-max option for TOPSIS.
- Mamdani output equals simpful's and differs from an exact integral by a discretisation error.
- ANFIS gets nearly all of its fit from the least-squares step. At default settings the premise
  step changes almost nothing on any dataset here, so gradient training adds little. This is the
  weakest result on the page.

## 1. MCDM against pymcdm and pyrepo-mcda

**Method.** bulanik and each library run on the same inputs: nine decision matrices taken from the
studies named in the libraries' test suites, 200 random matrices from a seeded generator (4 to 12
alternatives, 3 to 8 criteria, mixed benefit and cost), and for AHP 100 random reciprocal matrices on
Saaty's scale plus one from a study. A few of the study weight vectors were adjusted in the last entry so
that they sum to 1 and every library accepts them. Ranks are compared with a tie tolerance of $10^{-9}$.
The `/vector` and `/sum` rows set a variant so that the two formulas match.

**Data and licence.** Matrices and expected values come from the pymcdm 1.4.0 and pyrepo-mcda 0.1.15
source distributions (both MIT; see `benchmarks/THIRD_PARTY_NOTICES.md`).

```python exec="on"
import json
from pathlib import Path

d = json.loads(Path("tests/benchmarks/data/mcdm_results.json").read_text(encoding="utf-8"))
who = d["produced_by"]
print(f"Produced with bulanik {who['bulanik']}, pymcdm {who['pymcdm']}, pyrepo-mcda {who['pyrepo-mcda']}, "
      f"numpy {who['numpy']}, scipy {who['scipy']}, Python {who['python']}; seed {d['setup']['seed']}.\n")
print("| Method | Library | Matrices | Max abs. score diff. | Min. Spearman | Identical rankings | Classification |")
print("|---|---|---|---|---|---|---|")
for r in sorted(d["comparison"], key=lambda r: (r["method"], r["library"])):
    print(f"| {r['method']} | {r['library']} | {r['matrices']} | {r['max_abs_diff']:.2e} | {r['min_spearman']:.4f} "
          f"| {r['identical_rankings']} of {r['matrices']} | {r['classification']} |")
```

**Known differences**

```python exec="on"
import json
from pathlib import Path

d = json.loads(Path("tests/benchmarks/data/mcdm_results.json").read_text(encoding="utf-8"))
seen = set()
for r in d["comparison"]:
    if r["classification"] != "agree" and r["explanation"] not in seen:
        seen.add(r["explanation"])
        print(f"- **{r['method']}, {r['library']} ({r['classification']}).** {r['explanation']}")
for r in d["comparison"]:
    if r["bulanik_errors"]:
        print(f"- **{r['method']}, {r['library']}.** bulanik refused one random matrix, which is why the row has "
              f"{r['matrices']} matrices: {r['bulanik_errors'][0]}")
        break
c = d["ahp_consistency_ratio"]
print(f"- **AHP consistency ratio, pymcdm ({c['classification']}).** Largest gap {c['max_abs_diff']:.4f}; "
      f"{c['max_abs_diff_same_random_index']:.1e} with the same random index. {c['explanation']}")
t = d["tie_handling"]
print(f"- **Ranks of tied alternatives ({t['classification']}).** For scores {t['scores']}: bulanik's positions "
      f"{t['bulanik_ranking_positions']}, pymcdm {t['pymcdm_rankdata']}, pyrepo-mcda {t['pyrepo_rank_preferences']}. "
      f"{t['explanation']}")
```

The formulas of the pymcdm COPRAS row come from reading its source; the study its test cites was not
read here, so this page does not say which form the study prints or which is right.

**Expected values from the libraries' test suites.** bulanik against the numbers printed in those
tests, at the rounding each test uses. "Normalised" means bulanik rejected the rounded weights until
`normalize_weights=True` was set. The studies are the ones the test suites name; they were not read here.

```python exec="on"
import json
from pathlib import Path

d = json.loads(Path("tests/benchmarks/data/mcdm_results.json").read_text(encoding="utf-8"))
print("| Fixture | Study named by the test | Decimals | Max abs. diff. | Within rounding | Note |")
print("|---|---|---|---|---|---|")
for f in d["fixtures"]:
    note = f["explanation"] or ("Normalised." if f["weights_refused_without_normalize"] else "")
    print(f"| `{f['id']}` | {f['study']} | {f['decimals']} | {f['max_abs_diff']:.5f} | "
          f"{'yes' if f['within_rounding'] else 'no'} | {note} |")
```

**Edge cases.** The libraries and bulanik differ most at the edges. These rows show only the cases where
the three do not give the same outcome. Numbers are scores; a COPRAS row that differs only as above is left out.

```python exec="on"
import json
from pathlib import Path

d = json.loads(Path("tests/benchmarks/data/mcdm_results.json").read_text(encoding="utf-8"))


def show(v):
    return v.split(":")[0] if isinstance(v, str) else ", ".join(f"{x:.3f}" for x in v)


print("| Case | Method | bulanik | pymcdm | pyrepo-mcda |")
print("|---|---|---|---|---|")
for e in d["edge_cases"]:
    if e["bulanik"] == e["pymcdm"] == e["pyrepo-mcda"]:
        continue
    if e["method"] == "copras" and e["bulanik"] == e["pyrepo-mcda"] and not isinstance(e["pymcdm"], str):
        continue
    print(f"| {e['case']} | {e['method']} | {show(e['bulanik'])} | {show(e['pymcdm'])} | {show(e['pyrepo-mcda'])} |")
```

Where bulanik is stricter: it refuses a zero in MARCOS, which both libraries accept, and in WASPAS, which
pyrepo-mcda accepts and pymcdm refuses. For all-identical alternatives it refuses CoCoSo and entropy, which the
libraries score. Where an alternative is worst on every CoCoSo criterion it raises an error, and the libraries return
non-finite scores. On a constant column CoCoSo sets the normalised value to 0 where both libraries set it to 1,
which changes the scores (not the order, in this case).

## 2. Human Development Index

**Method.** The HDI of a country is the geometric mean of three dimension indices: health, education and
income. The published file holds the raw indicators and not the indices, so they are computed from life
expectancy, expected and mean years of schooling, and GNI per capita, as
$(LE - 20) / 65$, $(\min(EYS, 18) / 18 + \min(MYS, 15) / 15) / 2$ and
$(\ln \min(GNI, 75000) - \ln 100) / (\ln 75000 - \ln 100)$. These goalposts were not read from UNDP's
technical notes for this benchmark: they are checked by reproducing the published HDI (below).

Countries are then ranked by [`waspas`][bulanik.mcdm.ranking.waspas] with equal weights and `lam=0`, the
product form, and compared with UNDP's rank. WASPAS first divides each column by its maximum, so
its product form is the HDI divided by a constant, and **it should match nearly exactly by construction**.
This checks the implementation against a published ranking. It does not show that WASPAS learns anything
about countries. The sum form (`lam=1`) and the default blend show how far other forms drift.

**Data and licence.** [HDR 2025 composite indices](https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv),
United Nations Development Programme, used under
[CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/legalcode) as stated in its
[terms of use](https://hdr.undp.org/copyright-and-terms-use). A 2023 extract with computed index columns is
committed at `tests/benchmarks/data/hdi_2023_extract.csv`; `benchmarks/THIRD_PARTY_NOTICES.md` lists the changes.
UNDP does not endorse this work.

```python exec="on"
import json
from pathlib import Path

d = json.loads(Path("tests/benchmarks/data/hdi_results.json").read_text(encoding="utf-8"))
res = d["results"]
n = d["countries_used"]
print(f"{n} countries for {d['source']['year']}; left out for missing data: {', '.join(d['dropped_no_data'])}. "
      f"Recomputed HDI equals the published value at 3 decimals for {res['recomputed_hdi']['agree_at_3_decimals']} of "
      f"{res['recomputed_hdi']['countries']} countries.\n")
print("| Form | lam | Spearman with UNDP rank | Same rank, unrounded | Largest rank gap, unrounded | Same rank after rounding to 3 decimals |")
print("|---|---|---|---|---|---|")
for key in ("product", "blend_0.5", "sum"):
    r = res[key]
    print(f"| {key} | {r['lam']} | {r['spearman_with_undp_rank']:.6f} | {r['same_rank_unrounded']} of {n} | "
          f"{r['max_rank_difference_unrounded']:.1f} | {r['same_rank_after_rounding_to_3_decimals']} of {n} |")
```

**Known differences.** UNDP ranks on the HDI rounded to 3 decimals, so 34 countries share a rank with
another (1, 2, 2, 4, ...). Unrounded scores split those ties, which accounts for the whole gap of the
product form: once its values are rounded to 3 decimals, its ranks are UNDP's.

## 3. ANFIS on Mackey-Glass

**Method.** The series is generated here by RK4 from
$\dot x = 0.2\,x(t-17) / (1 + x(t-17)^{10}) - 0.1\,x(t)$ with step 0.1, $x(0) = 1.2$ and $x(t) = 0$ for
$t < 0$. Inputs are $x(t-18), x(t-12), x(t-6), x(t)$, the target is $x(t+6)$, with 500 training points from
$t = 118$ and the next 500 for testing. Two terms per input give 16 rules. Training is
[`fit`](anfis.md), the hybrid method; it is pure Python and needs no extra (the sezgi extra is only for the
optional derivative-free `train_anfis`). Training is deterministic, so there is no seed. **Every step
size tried is listed; none was selected.** No figure from the literature is quoted, because
none was read for this benchmark and the series could differ.

**Data and licence.** Generated by `tests/benchmarks/_bench_support.py`; no external data.

```python exec="on"
import json
from pathlib import Path

d = json.loads(Path("tests/benchmarks/data/anfis_mackey_glass_results.json").read_text(encoding="utf-8"))
b = d["baselines"]
print("| Model | lr | Epochs | Train RMSE | Test RMSE | Fit seconds |")
print("|---|---|---|---|---|---|")
print(f"| Last value carried forward | | | {b['persist_last_value']['train_rmse']:.5f} | {b['persist_last_value']['test_rmse']:.5f} | |")
print(f"| Linear least squares | | | {b['linear_least_squares']['train_rmse']:.5f} | {b['linear_least_squares']['test_rmse']:.5f} | |")
for r in d["runs"]:
    for c in r["checkpoints"]:
        if c["epochs"] in (1, 500):
            print(f"| ANFIS {r['mf']}, {r['n_rules']} rules | {r['lr']:g}{' (default)' if r['default_lr'] else ''} | "
                  f"{c['epochs']} | {c['train_rmse']:.5f} | {c['test_rmse']:.5f} | {c['fit_seconds']} |")
```

**What this shows.** At the default `lr=0.01`, the fit after one epoch and after 500 are the same to
five decimals: the least-squares consequents do the work and the premise barely moves. The premise gradient
is correct (an offline test checks it against finite differences on a 100-point subset, as the gradient of the
mean squared error), but its entries there are around $10^{-7}$ once the error is this small, so a small step
does nothing. Larger values of `lr` help the bell terms a little and the Gaussian terms not at all, and the
Gaussian rows have the best test RMSE at every `lr`. The test set
goes slightly outside the training range, so the extrapolation warning fires on every run.

## 4. Mamdani and Sugeno against simpful and scikit-fuzzy

**Method.** Two rule bases written for this benchmark (`tests/benchmarks/_inference_cases.py`), with
triangular, trapezoidal and Gaussian terms, are evaluated over a grid of input pairs. All libraries use
AND as stated, OR max, NOT $1-\mu$, clip implication and max aggregation.

**Scope of scikit-fuzzy.** `skfuzzy.control` imports networkx, which scikit-fuzzy 0.5.0 does not declare and
this suite does not install, so **its control engine was not run**. The scikit-fuzzy column uses the library's
membership functions and `defuzz`, with firing strengths and aggregation written in the benchmark script. No
file in the package mentions Sugeno, so it has no Sugeno row.

**Data and licence.** No dataset. simpful and scikit-fuzzy outputs are frozen for the offline test.

```python exec="on"
import json
from pathlib import Path

d = json.loads(Path("tests/benchmarks/data/inference_results.json").read_text(encoding="utf-8"))
who = d["produced_by"]
print(f"Produced with bulanik {who['bulanik']}, simpful {who['simpful']}, scikit-fuzzy {who['scikit-fuzzy']}.\n")
print("| Mamdani rule base | AND | Resolution | Grid points | Max diff. vs simpful | Max diff. vs scikit-fuzzy primitives | Seconds: bulanik, simpful, scikit-fuzzy |")
print("|---|---|---|---|---|---|---|")
for m in d["mamdani"]:
    s = m["seconds"]
    print(f"| {m['rule_base']} | {m['and']} | {m['resolution']} | {m['vs_simpful']['points']} | {m['vs_simpful']['max_abs_diff']:.2e} "
          f"| {m['vs_scikit_fuzzy_primitives']['max_abs_diff']:.2e} | {s['bulanik']}, {s['simpful']}, {s['scikit_fuzzy_primitives']} |")
print()
print("| Defuzzification (climate, resolution 1001) | Max diff. vs scikit-fuzzy | Mean diff. |")
print("|---|---|---|")
for m in d["defuzzification_vs_scikit_fuzzy"]:
    print(f"| {m['method']} | {m['max_abs_diff']:.2e} | {m['mean_abs_diff']:.2e} |")
print()
print("| Sugeno rule base | AND | Max diff. vs simpful | Seconds: bulanik, simpful |")
print("|---|---|---|---|")
for m in d["sugeno"]:
    print(f"| {m['rule_base']} | {m['and']} | {m['vs_simpful']['max_abs_diff']:.2e} | {m['seconds']['bulanik']}, {m['seconds']['simpful']} |")
print()
c = d["classification"]
print("**Known differences**\n")
print(f"- **Mamdani against simpful.** {c['mamdani_vs_simpful']}")
q = c["mamdani_vs_scikit_fuzzy"]
print(f"- **Centroid against scikit-fuzzy ({q['label']}).** {q['explanation']} Applying the rectangle rule to the aggregate "
      f"set built with scikit-fuzzy's functions gives bulanik's value to {q['rectangle_rule_on_scikit_aggregate_vs_bulanik']:.1e}.")
print(f"- **Bisector against scikit-fuzzy.** {c['bisector_vs_scikit_fuzzy']}")
s = c["sugeno_vs_simpful_with_weights"]
print(f"- **Weighted Sugeno against simpful ({s['label']}).** {s['explanation']} Largest gap of the rebuilt form: "
      f"{s['rebuilt_simpful_form_vs_simpful']:.1e}.")
g = d["no_rule_fired"]
print(f"- **When no rule fires.** bulanik: {g['bulanik']}. simpful: {g['simpful']}. scikit-fuzzy's defuzz: {g['scikit_fuzzy_defuzz']}.")
```

For bulanik's centroid the error against the exact integral is largest when a term has non-zero membership
at the end of the output universe. At a resolution of 1001 it reached about 0.08 on a universe of width 100
here. Raise `resolution` if that matters.

## 5. ANFIS on public regression data

**Method.** Five fixed folds from a seeded shuffle, the same for all models, on raw inputs. ANFIS uses
defaults: Gaussian terms, `lr=0.01`, 20 epochs; ranges come from the training fold. The inputs were
chosen before running: for Airfoil, frequency, suction-side displacement thickness and attack angle,
plus a run with all five; for Auto MPG, weight, horsepower and model year. scikit-learn's models use all
inputs and default settings. Nothing was tuned. The script runs only in `benchmarks/`.

The plain ANFIS rows are the result. The rows marked "inputs scaled" (inputs mapped to $[0, 1]$ by the
training fold's range) and "lr 1e-12" (premise frozen, least squares only) are diagnostics. They were
**added after seeing the first results**, which exposed a bug in `fit`, since fixed; the figures below are
from the fixed code.

**Data and licence.** Both CC BY 4.0, downloaded at run time and not committed:
Brooks, Pope & Marcolini (1989), [Airfoil Self-Noise](https://doi.org/10.24432/C5VW2C), and Quinlan (1993),
[Auto MPG](https://doi.org/10.24432/C5859H), UCI Machine Learning Repository. Auto MPG rows with unknown
horsepower are dropped.

```python exec="on"
import json
from pathlib import Path

d = json.loads(Path("tests/benchmarks/data/anfis_regression_results.json").read_text(encoding="utf-8"))
for name, block in d["datasets"].items():
    print(f"\n**{name}** ({block['rows']} rows)\n")
    print("| Model | Mean RMSE | Std. over folds | Worst fold | Mean fit seconds | Folds with extrapolation warning |")
    print("|---|---|---|---|---|---|")
    for r in block["results"]:
        print(f"| {r['model']} | {r['mean_rmse']:.3f} | {r['std_rmse']:.3f} | {max(r['fold_rmse']):.3f} | {r['mean_fit_seconds']} | {r['folds_with_extrapolation']} |")
```

**The premise step adds little.** The plain rows are within 0.005 of the "lr 1e-12" rows, so
the fit comes from least squares. A premise step is kept only if it does not raise the training error, so on
data like this, with a large target and a small input range, `lr` is shrunk rather than harming the model. A
small case shows it:

```python exec="on" source="above" result="text"
import math
from bulanik.anfis import ANFIS

X = [[0.05 * i / 199] for i in range(200)]
y = [125 + 5 * math.sin(120 * x[0]) for x in X]

history = ANFIS(1, 2).fit(X, y, epochs=10)
print("RMSE per epoch, defaults:", [round(v, 2) for v in history])
print("least squares only (lr=1e-12):", round(ANFIS(1, 2).fit(X, y, epochs=1, lr=1e-12)[0], 3))
```

The least-squares step itself looks sound: in a one-off check on three
Airfoil folds, it gave the same train and test RMSE as `numpy.linalg.solve` on the same design matrix, to 9 decimals.

**Against scikit-learn.** Gradient boosting beats every ANFIS row on both datasets. The best ANFIS rows do beat
linear regression on Airfoil; on Auto MPG they are about level with it, and their worst fold is well above its worst.

**Known differences.** This is not a like-for-like contest: gradient boosting sees every input, and the
ANFIS grid cannot afford to, since rules multiply with each input. Runtime is a result too: the five-input
grid takes seconds per fold where scikit-learn's models take hundredths of a second or less.
