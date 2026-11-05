# bulanik: an easy-to-use fuzzy library for LumiX and sezgi (design)

Date: 2026-11-03. Written without a question round, per the user's standing instruction ("bana soru
sorma doğrudan başlat ve subagentler ile tamamla"). The user asked: "start with an easy-to-use fuzzy
logic library with detailed unit tests and examples; the goal is to use it within LumiX and sezgi."

## 1. Intent

Most of the fuzzy literature our group meets needs the
same small set of pieces:
- fuzzy numbers of many flavours;
- operations and aggregation over them;
- score functions;
- linguistic scales;
- a handful of crisp MCDM methods fed by those numbers.

The optimization part of that literature (fuzzy goal programming, the Best-Worst Method, fuzzy BWM)
belongs in LumiX (exact) and sezgi (metaheuristic).

`bulanik` ("fuzzy" in Turkish, a sibling to `sezgi`, "intuition") is that layer. It has three parts:
- a small, readable, dependency-free core;
- optional bridges to LumiX and sezgi;
- examples that run.

Success means three things:
1. A researcher can write `IFN(0.6, 0.3)`, aggregate expert opinions and rank alternatives in a
   few lines.
2. Every operator and score function in a result is named and switchable, so a paper's exact
   choice can be reproduced later.
3. The same weights feed straight into a LumiX model or a sezgi problem.

## 2. Goals (v0.1)

1. **Fuzzy numbers (immutable, validated, operator overloading).**
   - **TFN and TrFN:**
     - triangular `TFN(l, m, u)` and trapezoidal `TrFN(a, b, c, d)`;
     - `+`, `-`, `*` and `/` (the standard approximate formulas), plus scalar multiplication and
       power;
     - `alpha_cut(alpha)`, membership `mu(x)`, the vertex distance;
     - `defuzzify(method=...)` with the methods `centroid`, `graded_mean` ((l+4m+u)/6) and `mean`.
   - **q-rung orthopair, `QROFN(mu, nu, q)`:** the constraint is μ^q + ν^q ≤ 1. The named classes
     share this one implementation:
     - `IFN` (q = 1, intuitionistic);
     - `PFN` (q = 2, Pythagorean);
     - `FFN` (q = 3, Fermatean).
   - **Three-part types:**
     - `SphericalFN(mu, nu, pi)`, where μ² + ν² + π² ≤ 1;
     - `PictureFN(mu, eta, nu)`, where μ + η + ν ≤ 1;
     - `NeutrosophicFN(t, i, f)`, single-valued, each part in [0, 1].
   - **Each type has:**
     - complement;
     - algebraic `+` and `*` and scalar multiplication and power;
     - Einstein versions where the literature defines them for that type;
     - `score(method=...)` and `accuracy(method=...)`, with a documented default and named
       alternatives;
     - a normalised distance (Hamming and Euclidean).
2. **Operator families are explicit.**
   - Every operation and aggregation takes `ops="algebraic" | "einstein"`; the default is
     algebraic.
   - Every score function is selected by name.
   - Results carry the names that were used.
   - An unknown name, or a family a type doesn't support, raises a clear `ValueError` that lists the
     valid names.
3. **Aggregation.**
   - Weighted averaging (WA) and weighted geometric (WG) for every type and family.
   - Weights must be non-negative and sum to 1 within 1e-9, or pass `normalize=True`.
   - Expert aggregation is a helper over the same functions.
4. **Fuzzy sets and membership functions** (the basics of fuzzy logic, needed for fuzzy goal
   programming):
   - triangular, trapezoidal, gaussian, sigmoid and linear (increasing/decreasing) membership
     functions;
   - `LinguisticVariable` (a name, a universe, and terms that map to membership functions);
   - `fuzzify(x)`, which returns each term's degree;
   - min/max/product t-norms and t-conorms.
4b. **Fuzzy inference systems (`bulanik.inference`).** The user added this on 2026-11-03: "mamdani ve sugeno içerelim".
   - **Rules:** a `Rule` is an IF-part over linguistic variable terms, combined with AND/OR, with
     an optional NOT on a term and an optional rule weight. Rules can be written as readable strings,
     e.g. `"IF service IS good AND food IS rancid THEN tip IS average"`, or built from Python objects.
   - **`MamdaniFIS`:**
     - an AND method (min or prod) and an OR method (max or probor);
     - implication (min or prod) and aggregation (max, sum or probor);
     - defuzzification over a sampled output universe: `centroid` (default), `bisector`, `mom`,
       `som` and `lom`;
     - the resolution is configurable.
   - **`SugenoFIS`:** zero-order (constant) and first-order (linear in the inputs) consequents. The
     output is the firing-strength-weighted average, or the weighted sum by name.
   - **Both systems:**
     - multiple inputs and outputs;
     - `evaluate({"service": 7, "food": 3})` returns a dict of outputs;
     - `explain(...)` returns each rule's firing strength;
     - when no rule fires, the system raises a clear error or uses a declared default output
       (configurable).
   - **Validation:** reference behaviour is the classic tipping problem, checked against
     hand-computed values. Where the builder is confident of published reference numbers (e.g.
     scikit-fuzzy or MATLAB's tipper), it may use those too, naming the source.
   - **Examples:** `examples/10_mamdani_tipper.py` and `examples/11_sugeno_controller.py`.
   - **Fit with the sezgi bridge:** a FIS can be the evaluated function inside a `sezgi.Problem`.
     One example tunes Sugeno consequent parameters to data with sezgi:
     `examples/12_tune_sugeno_with_sezgi.py`.
4c. **Tsukamoto, type-2 and ANFIS.** The user added these on 2026-11-03: "ANFIS training, Tsukamoto
   and type-2 inference systems içerelim".
   - **`TsukamotoFIS`:**
     - consequents are monotonic membership functions (linear increasing/decreasing, sigmoid);
     - each rule's crisp output is the inverse of its consequent at the firing strength;
     - the output is the weighted average;
     - a non-monotonic consequent is refused at construction.
   - **Interval type-2 FIS (`IT2MamdaniFIS`, `IT2SugenoFIS`):**
     - **Sets:** interval type-2 fuzzy sets, a footprint of uncertainty given by an upper and a lower
       membership function, e.g. a gaussian with uncertain mean or sigma, or a scaled lower
       triangle.
     - **Rules:** each rule fires with an interval [f_lower, f_upper].
     - **Type reduction:** Karnik–Mendel (KM) by default, with Enhanced KM, Nie–Tan and
       Begian–Melek–Mendel selectable by name.
     - **Output:** the midpoint of the type-reduced interval, with the interval itself also
       reported.
     - **Scope:** general type-2 is out of scope. Interval type-2 is the common case.
   - **ANFIS (`bulanik.anfis`):**
     - **Structure:** a first-order Sugeno ANFIS with gaussian or generalized-bell input membership
       functions and grid-partitioned rules.
     - **`fit(X, y, epochs=..., method="hybrid")`:**
       - the hybrid rule is Jang 1993: least-squares estimation of the consequents in the forward
         pass, gradient descent on the premise parameters in the backward pass;
       - `method="gradient"` is pure gradient descent;
       - training reports the RMSE per epoch;
       - the least-squares step uses a small pure-Python solver (QR or a regularised normal
         equation) and needs no numpy.
     - **Use:** `predict(X)`, plus export to an equivalent `SugenoFIS`, so the trained system plugs
       into goals 4b and 8.
     - **Size limit:** the grid partition grows fast with the number of inputs. A warning is given
       above 3 inputs × 3 MFs, and the system is refused above 1024 rules unless the caller passes
       `allow_large=True`.
     - **sezgi:** a sezgi-based trainer, `bulanik.sezgi.train_anfis(...)`, tunes all parameters with
       a metaheuristic, as the derivative-free alternative.
   - **Reference behaviour:**
     - ANFIS learns a smooth 1-D or 2-D function (for example sin(x) or the sinc surface) to an
       RMSE threshold within a fixed number of epochs and a fixed seed;
     - the hybrid method does not end with a higher training RMSE than gradient-only on the same
       budget;
     - the KM type reducer matches hand-computed switch points on a small example;
     - with lower = upper membership functions, IT2 collapses to the type-1 result;
     - Tsukamoto matches hand-computed outputs.
   - **Examples:**
     - `examples/13_tsukamoto.py`;
     - `examples/14_it2_vs_t1_noise.py`, which shows IT2 is more robust to noisy inputs than the
       type-1 system;
     - `examples/15_anfis_sinc.py`;
     - `examples/16_anfis_with_sezgi.py`.
5. **Linguistic scales.**
   - `LinguisticScale` maps terms to fuzzy numbers of one type and can be user-defined.
   - A few built-in scales ship as well, such as Saaty's TFN 1–9 scale for fuzzy AHP and a 5-term
     and a 7-term IFN rating scale.
   - Each built-in scale's docstring states its source. A scale whose source the builder can't
     state is shipped as an "example scale", without attribution.
6. **MCDM methods (`bulanik.mcdm`).** All return a frozen result dataclass with `weights` or
   `scores`, a `ranking`, and the names of the operators used.
   - **Weighting:**
     - `ahp` (crisp; plus fuzzy AHP by Buckley's geometric mean for TFN), with the consistency ratio
       of the defuzzified matrix;
     - `swara` (crisp, and fuzzy via the chosen score);
     - `entropy` (objective weights from a crisp matrix).
   - **Ranking on crisp matrices:** `topsis`, `vikor`, `waspas` and `edas`, with benefit/cost
     criteria. TOPSIS normalises vectors; the other methods use the normalisation of their original
     papers.
   - **Fuzzy pipeline:** a `DecisionMatrix` of fuzzy numbers is reduced in three steps:
     1. aggregate experts (WA or WG);
     2. score or defuzzify by name;
     3. run the crisp method.

     This is one documented function, `rank_fuzzy(matrix, weights, method="topsis", ...)`.
   - **Fully fuzzy reference method:** `fuzzy_topsis` for TFN (Chen 2000, vertex distance), as the
     reference fully-fuzzy method.
7. **LumiX bridge (`bulanik.lumix`, extra `[lumix]`).**
   - `bwm_linear(best_to_others, others_to_worst, criteria)`: Rezaei's linear BWM as an LXModel
     solved with OR-Tools. It returns the weights, ξ\* and the consistency ratio.
   - `zimmermann(...)`: max–min fuzzy goal programming. Each fuzzy goal is a LumiX linear
     expression with an aspiration level, a tolerance and a sense. The bridge builds a linear
     membership for each goal, adds λ, and maximizes λ, so the result is an LXModel the user can
     still extend.
   - `scores_as_data(...)`: turns `bulanik` results (weights, scores) into plain dataclass records
     LumiX can index with `from_data`.
8. **sezgi bridge (`bulanik.sezgi`, extra `[sezgi]`).**
   - `BWMProblem`: the original nonlinear min–max BWM.
   - `FuzzyBWMProblem`: TFN fuzzy BWM, Guo & Zhao 2017.
   - Both are `sezgi.Problem` subclasses over a Float space, with the weight-sum and ordering
     constraints as penalties. Each has a `solve(...)` convenience that returns weights and ξ.
9. **Examples** (`examples/`, each a runnable script printing its result):
   1. fuzzy number basics;
   2. a linguistic scale and expert aggregation;
   3. fuzzy AHP;
   4. SWARA + TOPSIS supplier selection;
   5. an intuitionistic pipeline: IF ratings, IF-SWARA weights, ranking;
   6. fuzzy goal programming with LumiX (Zimmermann);
   7. linear BWM with LumiX;
   8. fuzzy BWM with sezgi;
   9. weights feeding a LumiX selection model (the #23/#161 pattern).

   The test suite runs every example. Examples whose extra is missing are skipped.
10. **README.** Quickstart, the type table, the operator/score naming rule, the bridges, and how to
    run the tests.

## 3. Non-goals (v0.1)

- **Out of this release, the next rounds:**
  - general (non-interval) type-2 FIS, ANFIS with scatter or cluster partitioning, and other
    neuro-fuzzy models. Mamdani, Sugeno, Tsukamoto, interval type-2 and grid ANFIS are in v0.1
    (goals 4b and 4c).
  - interval-valued and type-2 fuzzy numbers. The class design must leave room for them.
- **Out of scope here:**
  - neutrosophic types beyond single-valued;
  - plotting;
  - exact reproduction of specific published papers. That is the next project; this release builds
    the switchable pieces it needs.
- **Not without the user's approval:** publishing to PyPI or creating a GitHub repo.

## 4. Anti-goals (a failure even if it works)

- **An invalid fuzzy number that exists silently.**
  - Constructors validate.
  - Operations keep results valid, clamping float error of 1e-12 or less and raising beyond that.
- **A hidden operator, score or normalisation choice.** Every one is named, documented, overridable
  and reported in results.
- **Runtime dependencies in the core.** The core imports only the standard library. The bridges
  import `lumix` or `sezgi` lazily, with a clear message when the extra is missing.
- **Examples or docstrings that don't run.**
- **Tests that only check that code runs.** Every numeric test asserts a hand-computed value or a
  mathematical property.

## 5. Decisions (my calls)

1. **Location:** a new local git repo,
   `/Volumes/Work/01_Akademik/Yayinlar/Articles/optimization_package/bulanik`, beside `sezgi` and
   `lumix-sezgi`. Branch `main`, no remote.
2. **Packaging:**
   - `src/bulanik`, hatchling, uv, Python ≥ 3.10, MIT;
   - author Tolga Berber <tolga.berber@fen.ktu.edu.tr>;
   - typed, with `py.typed`;
   - extras `lumix = ["lumix-opt>=0.3.2", "ortools>=9.8"]` and `sezgi = ["sezgi>=0.1.6"]`;
   - dev group: pytest, hypothesis, pytest-cov, plus both extras for the bridge tests.
3. **API names:**
   - classes `TFN`, `TrFN`, `QROFN`, `IFN`, `PFN`, `FFN`, `SphericalFN`, `PictureFN` and
     `NeutrosophicFN`, all exported from `bulanik`;
   - modules `bulanik.numbers`, `bulanik.aggregate`, `bulanik.sets`, `bulanik.inference`, `bulanik.type2`, `bulanik.anfis`, `bulanik.scales`,
     `bulanik.mcdm`, `bulanik.lumix` and `bulanik.sezgi`.
4. **Language:** English API, docs and examples.
5. **Floats only:** no numpy. Matrices are lists of lists or a tiny `DecisionMatrix` class.
6. **Test layout:** `tests/` mirrors the modules. Property tests use hypothesis strategies per type,
   covering:
   - closure (results stay valid);
   - commutativity and associativity where they hold;
   - identities;
   - idempotency of WA/WG on equal inputs;
   - boundedness and monotonicity of score functions;
   - weights summing to 1.

   Hand-computed golden values are written with their derivation in a comment. Core coverage must be
   at least 95%.
7. **Git:** commits as `git -c user.name="Tolga Berber" -c user.email=tolga.berber@fen.ktu.edu.tr`,
   with no attribution lines.

## 6. Left to the builder

- Internal class structure: a shared base for the three-part types, and how the q-rung Einstein
  operations are expressed.
- Which score alternatives to ship beyond the default per type. Each must cite its source in the
  docstring.
- The exact README layout.
