# Changelog

All notable changes to bulanik are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-11-09

Fuzzy CODAS on triangular and trapezoidal numbers. Every existing result is unchanged: the
default distance of `TFN` and `TrFN` is still `"vertex"`, and `fuzzy_codas` on the other
number kinds computes what it did in 0.3.0.

### Added

- `TFN` and `TrFN` distances `"euclidean"` and `"hamming"`: the weighted distances of
  Li (2007), with weights (1, 2, 2, 1)/6 on the trapezoid points and (1, 4, 1)/6 on the
  triangle points (a triangle is the trapezoid (l, m, m, u)).
- `fuzzy_codas` (and `rank_fuzzy(method="fuzzy_codas")`) accepts `TFN` and `TrFN`
  matrices, with benefit and cost criteria, following the fuzzy CODAS of Keshavarz
  Ghorabaee et al. (2017): each column is normalised by its largest score (the centroid
  by default), x / max D for a benefit and 1 - x / max D for a cost; the weighted
  distances from the negative ideal are summed over the criteria. `operators` also holds
  `"normalization": "max_score"` for these kinds.

### Changed

- `fuzzy_codas` no longer raises `TypeError` for `TFN` and `TrFN` matrices (benefit or
  cost criteria). It still raises it for `IT2TrFN`, which has no Euclidean or Hamming
  distance. `fuzzy_waspas` and `fuzzy_cocoso` still need `complement()` for a cost
  criterion.

[0.4.0]: https://github.com/tdelphi1981/bulanik/releases/tag/v0.4.0

### Changed

- The intuitionistic `fuzzy_codas` docstring and the ranking guide now cite the CODAS of Keshavarz Ghorabaee et al. (2016) with the distances of Szmidt & Kacprzyk (2000) instead of an unlisted source; the computation is unchanged.

## [0.3.0] - 2026-11-05

Interval-valued AHP and fuzzy versions of three more ranking methods. The public v0.2 API is unchanged apart from
the two changes listed below.

### Added

- `ahp_interval_valued`: AHP over interval-valued orthopair comparison matrices, with the
  `multiplicative` and `possibility` methods.
- `fuzzy_codas`, `fuzzy_waspas` (with `combine="fuzzy"` or `combine="scores"`) and
  `fuzzy_cocoso`, also reachable through `rank_fuzzy(method=...)`.
- An `IVQROFN` score named `"defuzzified"` (q = 1).
- Examples 23 and 24 (24 executed examples in all).

### Changed

- Crisp `codas` accepts zeros in benefit columns.
- `ANFIS.fit` results differ from v0.2: `lr` is now the largest step tried, and hybrid training
  solves the consequents once more per `fit`, so `history` values change.

### Fixed

- Crisp `cocoso` treats rounding-size gaps at a column extreme as zero.
- `ANFIS.fit` kept gradient steps that raised the training error and could return a model far worse
  than least squares alone, with no warning. A step that raises the error is now halved or dropped,
  and hybrid training ends on solved consequents.

[0.3.0]: https://github.com/tdelphi1981/bulanik/releases/tag/v0.3.0

## [0.2.0] - 2026-11-04

Interval-valued and interval type-2 numbers, six new ranking methods, new weighting
methods, DEMATEL and three new figures. The public v0.1 API is unchanged: this release
only adds names.

### Added

- Interval-valued numbers: `IVQROFN` (interval-valued q-rung orthopair) with the aliases
  `IVIFN` (q = 1), `IVPFN` (q = 2) and `IVFFN` (q = 3), and `IVNeutrosophicFN` (interval
  neutrosophic). Frozen and validated, with the algebraic and Einstein operator families,
  scores, accuracies and distances. The aliases are separate classes: they neither compare
  equal to nor mix with an `IVQROFN` of the same rung.
- `IT2TrFN`: interval type-2 trapezoidal fuzzy numbers (upper and lower trapezoid with
  heights, lower nested inside upper), with the arithmetic of Chen and Lee (2010a) and the
  DTraT score of Kahraman et al. (2014).
- Ranking methods: CODAS, COPRAS, MARCOS, CoCoSo, WISP and ELECTRE I (with net concordance
  and discordance indices for a complete ranking), all registered in
  `rank_fuzzy(method=...)`. Zero, negative and constant columns, and signed default
  scores, raise a `ValueError` naming the method where the method cannot handle them
  (constant columns are neutral in CoCoSo, CRITIC and COPRAS; ELECTRE and CoCoSo accept zeros
  and negatives); a magnitude spread so large that a scaled entry underflows to 0 raises the
  same way in CODAS, MARCOS and WISP.
- Weighting methods: CRITIC (objective), PIPRECIA (crisp and `TFN` judgements, with an
  optional inverse pass) and `ahp_fuzzy_numbers` (AHP over a comparison matrix of any one
  fuzzy number kind). PIPRECIA requires every judgement `s_j` to lie in (0, 2) and names the
  offending index otherwise.
- DEMATEL: `dematel` for crisp or fuzzy direct-relation matrices, `cfcs` (Opricovic and
  Tzeng 2003) to convert triangular matrices from several experts, and `DEMATELResult` with
  `as_weights()` so the factor weights feed every ranking method. A singular or
  ill-conditioned $I - N$ (spectral radius numerically 1) raises instead of returning a huge
  total relation.
- `bulanik.plot`: `dematel` (cause-effect diagram), `iv_orthopair` (interval rectangles over
  the feasible region) and `it2_number` (upper and lower trapezoids with the footprint of
  uncertainty).
- Documentation: guide pages for interval-valued numbers, interval type-2 numbers, the new
  ranking and weighting methods and DEMATEL; new gallery figures; API reference for all new
  names; examples 18 to 22 (22 executed examples in all); the reference list extended to
  every source cited in the v0.2 docstrings, each DOI checked against Crossref.

### Changed

- The dark categorical plot palette, slot 6 (green), is `#2ea32e` instead of `#008300`,
  which was too dark on the slate surface. Figures that use the sixth colour change in dark
  mode only.

### Notes on sourcing

- The 7-term interval type-2 linguistic scale in example 19 and in the gallery is
  illustrative: it follows the style of the type-2 literature, and its values are not a
  transcription of any published table.
- The CRITIC tests are checked against values derived by hand from the published formulas,
  not against a worked example printed in the paper.
- The DEMATEL tests include a cross-check against the vignette of the R package `dematel`.

[0.2.0]: https://github.com/tdelphi1981/bulanik/releases/tag/v0.2.0

## [0.1.0] - 2026-11-03

First release.

### Added

- Fuzzy numbers: triangular (`TFN`) and trapezoidal (`TrFN`); intuitionistic (`IFN`),
  Pythagorean (`PFN`), Fermatean (`FFN`) and q-rung orthopair (`QROFN`); spherical
  (`SphericalFN`), picture (`PictureFN`) and single-valued neutrosophic
  (`NeutrosophicFN`). Immutable, validated on construction, with named operator
  families (`algebraic`, `einstein`), scores, accuracies and distances.
- Aggregation: weighted average and weighted geometric mean, and expert aggregation of
  `[expert][alternative][criterion]` ratings.
- Fuzzy sets: membership functions (triangular, trapezoidal, gaussian, bell, sigmoid,
  linear ramps), linguistic variables, t-norms and t-conorms by name, and linguistic
  scales (`SAATY_TFN`, `FUZZY_BWM_TFN` and illustrative IF/TFN scales).
- Fuzzy inference: Mamdani, Sugeno (zero and first order) and Tsukamoto systems from
  plain-text rules, with `explain` and a no-rule-fired policy.
- Interval type-2 inference: interval type-2 membership functions and Mamdani and
  Sugeno systems, with KM, EKM, Nie-Tan and BMM type reduction.
- ANFIS: grid-partition first-order Sugeno ANFIS with hybrid (Jang) or gradient
  training, exported to a Sugeno system.
- MCDM: AHP, fuzzy AHP (Buckley), SWARA and entropy weights; best-worst consistency
  indices (crisp and fuzzy); TOPSIS, VIKOR, WASPAS and EDAS on crisp matrices;
  `DecisionMatrix` and `rank_fuzzy` for fuzzy-number matrices; Chen's fuzzy TOPSIS.
- `bulanik.lumix` (extra `lumix`): linear best-worst method, Zimmermann fuzzy
  programming and MCDM scores as LumiX data.
- `bulanik.sezgi` (extra `sezgi`): nonlinear and fuzzy best-worst method and ANFIS
  training with sezgi metaheuristics.
- `bulanik.plot` (extra `plot`): figures for numbers, scales, inference, type-2 sets and
  MCDM results in matching light and dark themes.
- Documentation site (MkDocs Material) with a user guide, API reference, 17 executed
  examples and a verified reference list; 17 runnable scripts in `examples/`.

[0.1.0]: https://github.com/tdelphi1981/bulanik/releases/tag/v0.1.0
