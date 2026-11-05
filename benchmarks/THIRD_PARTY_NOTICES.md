# Third-party notices

The benchmark suite compares bulanik with other libraries and uses public data. This file
records what was copied, from where, and under which licence. No third-party code is copied.

## pymcdm 1.4.0 (MIT)

Decision matrices, weights and expected values in `tests/benchmarks/_mcdm_cases.py` are copied
from `test/test_mcdm.py` and `test/test_weights.py` in the pymcdm 1.4.0 source distribution on
PyPI. Scores in `tests/benchmarks/data/mcdm_results.json` were produced by running pymcdm.

```text
Copyright (c) 2024-2026 Andrii Shekhovtsov
Copyright (c) 2024-2026 Bartłomiej Kizielewicz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## pyrepo-mcda 0.1.15 (MIT)

Decision matrices, weights and expected values in `tests/benchmarks/_mcdm_cases.py` are copied
from `tests/test_mcda_methods_1.py`, `tests/test_mcda_methods_2.py` and
`tests/test_weighting_methods.py` in the pyrepo-mcda 0.1.15 source distribution on PyPI. Scores
in `tests/benchmarks/data/mcdm_results.json` were produced by running pyrepo-mcda.

```text
MIT License

Copyright (c) 2022 energyinpython

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## simpful 2.12.0 and scikit-fuzzy 0.5.0

The frozen numbers in `tests/benchmarks/data/inference_results.json` are outputs of these
libraries, run on rule bases written for this benchmark. Nothing from their source is copied.
simpful is under the Academic Free License 3.0; scikit-fuzzy under a BSD-style licence.

## scikit-learn 1.9.1

`tests/benchmarks/data/anfis_regression_results.json` holds error figures from running it.
Nothing from its source is copied.

## Human Development Index data (CC BY 3.0 IGO)

`tests/benchmarks/data/hdi_2023_extract.csv` is an extract of the file
<https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv>
from the United Nations Development Programme, Human Development Report Office, used under the
[Creative Commons Attribution 3.0 IGO licence](https://creativecommons.org/licenses/by/3.0/igo/legalcode)
as stated at <https://hdr.undp.org/copyright-and-terms-use>.

Changes made: kept the year 2023 and the columns for rank, HDI, life expectancy, expected and mean
years of schooling and GNI per capita; dropped regional aggregates and the two countries without
data; re-encoded as UTF-8; added the three computed dimension-index columns. UNDP does not
endorse this work or its use of the data.

## UCI Machine Learning Repository data (CC BY 4.0)

Used by `benchmarks/anfis_regression.py`. The data are downloaded at run time and not committed;
only error figures are.

- Brooks, T., Pope, D. & Marcolini, M. (1989). Airfoil Self-Noise [Dataset]. UCI Machine Learning
  Repository. <https://doi.org/10.24432/C5VW2C>
- Quinlan, R. (1993). Auto MPG [Dataset]. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C5859H>

Licence: <https://creativecommons.org/licenses/by/4.0/>. No changes other than dropping the rows
with unknown horsepower from Auto MPG.
