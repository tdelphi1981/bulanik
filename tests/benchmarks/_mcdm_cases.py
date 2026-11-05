"""Decision problems for the MCDM comparison: screened fixtures and seeded random matrices.

The fixture matrices, weights and printed values are copied from the test suites in the
PyPI source distributions of pymcdm 1.4.0 and pyrepo-mcda 0.1.15 (both MIT; see
``THIRD_PARTY_NOTICES.md``). ``study`` is the source each suite itself names.
``decimals`` is the rounding the suite compares at. Standard library only.
"""

from __future__ import annotations

import random

B, C = "benefit", "cost"

TUS = "Tus & Aytac Adali (2019), Opsearch 56(2), 528-538"
ERSOY = "Ersoy (2021), LogForum 17(3)"
BADI_ICMNEE = "Badi, Shetwan & Abdulshahed (2017), Proc. 1st ICMNEE, 395-407"
BADI_DMAME = "Badi, Abdulshahed & Shetwan (2018), Decision Making: Appl. Manag. Eng. 1(1), 1-12"
YAZDANI = "Yazdani, Zarate, Zavadskas & Turskis (2019), Management Decision 57(9), 2501-2519"
ZAVADSKAS = "Zavadskas, Kaklauskas, Peldschus & Turskis (2007), Baltic J. Road Bridge Eng. 2(4), 195-203"
PODVEZKO = "Zavadskas & Podvezko (2016), IJITDM 15(2), 267-283"
OPRICOVIC = "Opricovic & Tzeng (2004), EJOR 156(2), 445-455"

_CODAS_M = [[45, 3600, 45, 0.9], [25, 3800, 60, 0.8], [23, 3100, 35, 0.9],
            [14, 3400, 50, 0.7], [15, 3300, 40, 0.8], [28, 3000, 30, 0.6]]
_COCOSO_M = [[60, 0.4, 2540, 500, 990], [6.35, 0.15, 1016, 3000, 1041], [6.8, 0.1, 1727.2, 1500, 1676],
             [10, 0.2, 1000, 2000, 965], [2.5, 0.1, 560, 500, 915], [4.5, 0.08, 1016, 350, 508],
             [3, 0.1, 1778, 1000, 920]]
_COPRAS_M = [[30, 12.487, 6.261, 10.880, 7.610], [20, 12.372, 5.961, 10.880, 7.460],
             [27, 11.096, 6.262, 9.920, 6.690], [18, 10.982, 5.962, 9.920, 6.540],
             [24, 11.017, 6.283, 9.980, 7.000], [16, 10.903, 5.983, 9.980, 6.850]]
_EDAS_M = [[3873, 39.55, 0.27, 0.87, 150, 0.07, 12, 2130], [5067, 67.26, 0.23, 0.23, 40, 0.02, 21, 2200],
           [2213, 24.69, 0.08, 0.17, 200, 0.04, 35, 570], [6243, 132, 0.07, 0.25, 100, 0.04, 16, 100],
           [8312, 460.47, 0.05, 0.21, 25, 0.1, 25, 200]]
_MARCOS_M = [[660, 1000, 1600, 18, 1200], [800, 1000, 1600, 24, 900], [980, 1000, 2500, 24, 900],
             [920, 1500, 1600, 24, 900], [1380, 1500, 1500, 24, 1150], [1230, 1000, 1600, 24, 1150],
             [680, 1500, 1600, 18, 1100], [960, 2000, 1600, 12, 1150]]
_VIKOR_YANG_M = [[78, 56, 34, 6], [4, 45, 3, 97], [18, 2, 50, 63], [9, 14, 11, 92], [85, 9, 100, 29]]
_OPRICOVIC_M = [[1, 3000], [2, 3750], [5, 4500]]
_WASPAS_M = [[30, 23, 5, 0.745, 0.745, 1500, 5000], [18, 13, 15, 0.745, 0.745, 1300, 6000],
             [15, 12, 10, 0.500, 0.500, 950, 7000], [25, 20, 13, 0.745, 0.745, 1200, 4000],
             [14, 18, 14, 0.255, 0.745, 950, 3500], [17, 15, 9, 0.745, 0.500, 1250, 5250],
             [23, 18, 20, 0.500, 0.745, 1100, 3000], [16, 8, 14, 0.255, 0.500, 1500, 3000]]
_TUS_M = [[5000, 3, 3, 4, 3, 2], [680, 5, 3, 2, 2, 1], [2000, 3, 2, 3, 4, 3],
          [600, 4, 3, 1, 2, 2], [800, 2, 4, 3, 3, 4]]
_ERSOY_M = [[256, 8, 41, 1.6, 1.77, 7347.16], [256, 8, 32, 1.0, 1.8, 6919.99], [256, 8, 53, 1.6, 1.9, 8400],
            [256, 8, 41, 1.0, 1.75, 6808.9], [512, 8, 35, 1.6, 1.7, 8479.99], [256, 4, 35, 1.6, 1.7, 7499.99]]
_ERSOY_W = [0.405, 0.221, 0.134, 0.199, 0.007, 0.034]
_ERSOY_T = [B, B, B, B, C, C]
_VIKOR_PAP_M = [[8, 7, 2, 1], [5, 3, 7, 5], [7, 5, 6, 4], [9, 9, 7, 3], [11, 10, 3, 7], [6, 9, 5, 4]]
_DESAL_M = [[8, 8, 10, 9, 5], [8, 9, 9, 9, 8], [9, 9, 7, 8, 6], [8, 8, 7, 8, 9], [9, 8, 7, 7, 4]]
_GOSWAMI_M = [[80, 16, 2, 5], [110, 32, 2, 9], [130, 64, 4, 9], [185, 64, 4, 1], [135, 64, 3, 4],
              [140, 32, 3, 5], [185, 64, 6, 7], [110, 16, 3, 3], [120, 16, 4, 3], [340, 128, 6, 5]]
_STEVIC_M = [
    [6.257, 4.217, 4.217, 6.257, 3.000, 4.217, 5.000, 3.557, 3.557, 3.557, 3.000, 5.000, 4.718, 3.557, 3.557, 2.080, 3.557, 3.000, 4.718, 3.557, 2.080],
    [4.217, 6.804, 7.000, 5.000, 7.000, 6.804, 5.593, 5.593, 6.804, 7.000, 5.000, 7.612, 5.593, 6.257, 5.000, 7.000, 5.593, 5.593, 6.804, 5.593, 5.593],
    [4.718, 5.593, 6.257, 5.000, 4.718, 5.000, 5.593, 4.718, 5.593, 5.000, 3.557, 6.257, 5.000, 4.718, 4.718, 5.000, 3.557, 5.593, 5.593, 3.557, 4.217],
    [5.000, 6.804, 5.000, 3.000, 5.000, 6.257, 7.612, 3.557, 5.000, 6.257, 6.257, 5.593, 6.257, 5.000, 5.593, 7.000, 5.000, 6.257, 5.000, 3.557, 4.217],
    [3.557, 5.593, 6.804, 3.000, 5.000, 7.000, 5.593, 5.000, 6.257, 7.000, 5.593, 7.612, 6.257, 6.257, 5.000, 6.257, 5.593, 7.000, 5.000, 4.718, 5.000],
    [6.257, 3.000, 4.217, 5.000, 3.557, 3.000, 4.217, 3.000, 4.217, 3.000, 2.466, 3.000, 4.217, 3.557, 5.000, 3.000, 4.217, 3.557, 2.080, 5.000, 3.000],
    [4.217, 5.000, 6.257, 5.593, 3.557, 5.593, 4.217, 5.593, 5.000, 6.257, 3.557, 5.000, 6.257, 5.593, 5.593, 7.000, 6.257, 5.000, 6.257, 5.593, 5.000],
    [7.612, 1.442, 3.000, 9.000, 2.080, 3.000, 1.442, 2.080, 1.442, 3.000, 1.000, 1.442, 2.080, 3.000, 3.000, 1.000, 1.442, 1.442, 3.000, 2.080, 1.000],
]
_STEVIC_W = [0.127, 0.159, 0.060, 0.075, 0.043, 0.051, 0.075, 0.061, 0.053, 0.020, 0.039, 0.022, 0.017, 0.027, 0.022, 0.039, 0.017, 0.035, 0.015, 0.024, 0.016]
_STEVIC_T = [C, B, B, C] + [B] * 17
_ELSAYED_M = [[7, 312, 1891, 6613, 163], [72, 88, 728, 941804, 1078], [10, 252, 2594, 3466, 117471],
              [10, 145, 980, 2371, 86329], [65, 54, 350, 501, 29897], [29, 48, 380, 912, 34051],
              [7, 476, 3300, 78470, 2212], [70, 87, 650, 733, 819000], [4, 86, 591, 3015, 103],
              [12, 79, 579, 3240, 96098], [21, 45, 261, 1253, 453], [1, 72, 530, 4333, 0.104800]]
_XU_M = [[30, 30, 38, 29], [19, 54, 86, 29], [19, 15, 85, 28.9], [68, 70, 60, 29]]
_PODVEZKO_M = [[3.0, 100, 10, 7], [2.5, 80, 8, 5], [1.8, 50, 20, 11], [2.2, 70, 12, 9]]
_LEE_M = [[4550, 30, 6.74, 20, 15, 5, 85, 150, 0.87, 4.76], [3005, 60.86, 2.4, 35, 27, 4, 26, 200, 0.17, 4.51],
          [2040, 14.85, 1.7, 90, 25, 5, 26, 500, 0.27, 4.19], [3370, 99.4, 3.25, 25.3, 54, 3, 45, 222, 0.21, 3.78],
          [3920, 112.6, 4.93, 11.4, 71.7, 2, 50, 100, 0.25, 4.11]]
_WANG_M = [[1, 4, 3, 1, 3, 4], [1 / 4, 1, 7, 3, 1 / 5, 1], [1 / 3, 1 / 7, 1, 1 / 5, 1 / 5, 1 / 6],
           [1, 1 / 3, 5, 1, 1, 1 / 3], [1 / 3, 5, 5, 1, 1, 3], [1 / 4, 1, 6, 3, 1 / 3, 1]]
_PAP_CRIT = [[1, 1, 5, 3], [1, 1, 5, 3], [1 / 5, 1 / 5, 1, 1 / 3], [1 / 3, 1 / 3, 3, 1]]


def _fx(fid, lib, test, method, study, matrix, weights, types, expected, decimals, **params):
    return dict(id=fid, library=lib, test=test, method=method, study=study, matrix=matrix,
                weights=weights, types=types, expected=expected, decimals=decimals, params=params)


# Every entry: scoring methods carry weights and types; weighting methods have weights=None.
FIXTURES = [
    _fx("pymcdm-cocoso", "pymcdm", "test_mcdm.py::TestCOCOSO", "cocoso", YAZDANI, _COCOSO_M,
        [0.036, 0.192, 0.326, 0.326, 0.12], [B, C, B, B, B], [2.041, 2.788, 2.882, 2.416, 1.299, 1.443, 2.519], 3),
    _fx("pymcdm-codas", "pymcdm", "test_mcdm.py::TestCODAS", "codas", BADI_ICMNEE, _CODAS_M,
        [0.2857, 0.3036, 0.2321, 0.1786], [B, C, B, B], [1.3914, 0.3411, -0.2170, -0.5381, -0.7292, -0.2481], 4),
    _fx("pymcdm-copras-1", "pymcdm", "test_mcdm.py::TestCOPRAS", "copras", ZAVADSKAS, _COPRAS_M,
        [0.5300, 0.1175, 0.1175, 0.1175, 0.1175], [B, C, C, C, C], [1.0, 0.797, 0.9078, 0.7243, 0.8537, 0.6898], 4),
    _fx("pymcdm-copras-2", "pymcdm", "test_mcdm.py::TestCOPRAS2", "copras", ZAVADSKAS, _COPRAS_M,
        [0.075, 0.700, 0.075, 0.075, 0.075], [B, C, C, C, C], [1.0, 0.9585, 0.8984, 0.86, 0.8886, 0.8532], 4),
    _fx("pymcdm-edas", "pymcdm", "test_mcdm.py::TestEDAS", "edas",
        "Yazdani, Torkayesh, Santibanez-Gonzalez & Otaghsara (2020), Sustainable Operations and Computers 1, 35-42",
        _EDAS_M, [0.131, 0.113, 0.126, 0.125, 0.126, 0.129, 0.132, 0.117], [C, C, C, B, B, C, B, B],
        [0.841, 0.632, 0.883, 0.457, 0.104], 3),
    _fx("pymcdm-marcos", "pymcdm", "test_mcdm.py::TestMARCOS", "marcos",
        "Ulutas, Karabasevic, Popovic, Stanujkic, Nguyen & Karakoy (2020), Mathematics 8(10), 1672",
        _MARCOS_M, [0.1061, 0.3476, 0.3330, 0.1185, 0.0949], [C, B, B, B, B],
        [0.5649, 0.5543, 0.6410, 0.6174, 0.6016, 0.5453, 0.6282, 0.6543], 4),
    _fx("pymcdm-topsis", "pymcdm", "test_mcdm.py::TestTOPSIS", "topsis", OPRICOVIC, _OPRICOVIC_M,
        [0.5, 0.5], [C, B], [0.500, 0.617, 0.500], 3),
    _fx("pymcdm-vikor-yang", "pymcdm", "test_mcdm.py::TestVIKOR::test_output_yang", "vikor",
        "Yang & Wu (2020), IEEE Access 8, 21261-21271", _VIKOR_YANG_M, [0.25] * 4, [B] * 4,
        [0.5679, 0.7667, 1, 0.7493, 0], 4),
    _fx("pymcdm-vikor-opricovic", "pymcdm", "test_mcdm.py::TestVIKOR::test_output_opricovic", "vikor", OPRICOVIC,
        _OPRICOVIC_M, [0.5, 0.5], [C, B], [1, 0, 1], 9),
    _fx("pymcdm-waspas", "pymcdm", "test_mcdm.py::TestWASPAS", "waspas",
        "Chakraborty, Zavadskas & Antucheviciene (2015), ECECSR 49(1), 5-22", _WASPAS_M,
        [0.1181, 0.1181, 0.0445, 0.1181, 0.2861, 0.2861, 0.0445], [B, B, B, B, B, C, C],
        [0.8329, 0.7884, 0.6987, 0.8831, 0.7971, 0.7036, 0.8728, 0.5749], 4),
    _fx("pymcdm-entropy", "pymcdm", "test_weights.py::TestEntropyWeights", "entropy",
        "Elsayed, Dawood & Karthikeyan (2017), Int. J. Eng. Trends Technol. 46(2), 60-66", _ELSAYED_M, None, None,
        [0.104991, 0.069124, 0.071373, 0.449937, 0.304575], 6),
    _fx("pymcdm-critic", "pymcdm", "test_weights.py::TestCRITICWeights", "critic", TUS, _TUS_M, None, None,
        [0.157, 0.249, 0.168, 0.121, 0.154, 0.151], 3),
    _fx("pymcdm-ahp", "pymcdm", "test_weights.py::TestAHPWeights", "ahp",
        "Wang & Chin (2011), Information Sciences 181(23), 5240-5248", _WANG_M, None, None,
        [0.3208, 0.1395, 0.0348, 0.1285, 0.2374, 0.1391], 4),
    _fx("pyrepo-vikor", "pyrepo-mcda", "test_mcda_methods_1.py::Test_VIKOR", "vikor",
        "Papathanasiou & Ploskas (2018), Multiple Criteria Decision Aid, 31-55", _VIKOR_PAP_M,
        [0.4, 0.3, 0.1, 0.2], [B] * 4, [0.640, 1.000, 0.693, 0.271, 0.000, 0.694], 3, v=0.625),
    _fx("pyrepo-topsis", "pyrepo-mcda", "test_mcda_methods_1.py::Test_TOPSIS", "topsis", ERSOY, _ERSOY_M,
        _ERSOY_W, _ERSOY_T, [0.308, 0.254, 0.327, 0.263, 0.857, 0.192], 2),
    _fx("pyrepo-codas-1", "pyrepo-mcda", "test_mcda_methods_1.py::Test_CODAS::test_codas", "codas", BADI_DMAME,
        _CODAS_M, [0.2857, 0.3036, 0.2321, 0.1786], [B, C, B, B], [1.3914, 0.3411, -0.2170, -0.5381, -0.7292, -0.2481], 4),
    _fx("pyrepo-codas-2", "pyrepo-mcda", "test_mcda_methods_1.py::Test_CODAS::test_codas2", "codas",
        "Badi, Ballem & Shetwan (2018), Int. J. Quality Research 12(3), 609", _DESAL_M,
        [0.19, 0.26, 0.24, 0.17, 0.14], [C, B, B, B, C], [0.4463, 0.1658, -0.2544, -0.4618, 0.1041], 4),
    _fx("pyrepo-edas", "pyrepo-mcda", "test_mcda_methods_1.py::Test_EDAS", "edas", ERSOY, _ERSOY_M,
        _ERSOY_W, _ERSOY_T, [0.414, 0.130, 0.461, 0.212, 0.944, 0.043], 3),
    _fx("pyrepo-marcos", "pyrepo-mcda", "test_mcda_methods_2.py::Test_MARCOS", "marcos",
        "Stevic, Pamucar, Puska & Chatterjee (2020), Computers & Industrial Engineering 140, 106231",
        _STEVIC_M, _STEVIC_W, _STEVIC_T, [0.524, 0.846, 0.703, 0.796, 0.843, 0.499, 0.722, 0.291], 3),
    _fx("pyrepo-copras", "pyrepo-mcda", "test_mcda_methods_2.py::Test_COPRAS", "copras",
        "Goswami & Mitra (2020), Int. J. Data and Network Science 4(1), 27-42", _GOSWAMI_M,
        [0.60338, 0.13639, 0.19567, 0.06456], [C, B, B, B],
        [1, 0.85262, 0.91930, 0.68523, 0.80515, 0.72587, 0.83436, 0.79758, 0.79097, 0.79533], 5),
    _fx("pyrepo-cocoso", "pyrepo-mcda", "test_mcda_methods_2.py::Test_COCOSO", "cocoso", YAZDANI, _COCOSO_M,
        [0.036, 0.192, 0.326, 0.326, 0.12], [B, C, B, B, B], [2.041, 2.788, 2.882, 2.416, 1.3, 1.443, 2.52], 1),
    _fx("pyrepo-entropy-xu", "pyrepo-mcda", "test_weighting_methods.py::Test_Entropy::test_Entropy", "entropy",
        "Xu (2004), EJOR 156(2), 530-532", _XU_M, None, None, [0.4630, 0.3992, 0.1378, 0.0000], 4),
    _fx("pyrepo-entropy-podvezko", "pyrepo-mcda", "test_weighting_methods.py::Test_Entropy::test_Entropy2", "entropy",
        PODVEZKO, _PODVEZKO_M, None, None, [0.1146, 0.1981, 0.4185, 0.2689], 4),
    _fx("pyrepo-entropy-ersoy", "pyrepo-mcda", "test_weighting_methods.py::Test_Entropy::test_Entropy3", "entropy",
        ERSOY, _ERSOY_M, None, None, _ERSOY_W, 3),
    _fx("pyrepo-entropy-lee", "pyrepo-mcda", "test_weighting_methods.py::Test_Entropy::test_Entropy4", "entropy",
        "Lee & Chang (2018), Renewable and Sustainable Energy Reviews 92, 883-896", _LEE_M, None, None,
        [0.026, 0.154, 0.089, 0.199, 0.115, 0.04, 0.08, 0.123, 0.172, 0.002], 3),
    _fx("pyrepo-critic", "pyrepo-mcda", "test_weighting_methods.py::Test_CRITIC", "critic", TUS, _TUS_M, None, None,
        [0.157, 0.249, 0.168, 0.121, 0.154, 0.151], 3),
]

# Matrices of the screened fixtures, reused as shared inputs for the library comparison.
SHARED_REAL = [
    ("cocoso-study", _COCOSO_M, [0.036, 0.192, 0.326, 0.326, 0.12], [B, C, B, B, B]),
    ("codas-study", _CODAS_M, [0.2857, 0.3036, 0.2321, 0.1786], [B, C, B, B]),
    ("copras-study", _COPRAS_M, [0.53, 0.1175, 0.1175, 0.1175, 0.1175], [B, C, C, C, C]),
    ("edas-study", _EDAS_M, [0.131, 0.113, 0.126, 0.125, 0.126, 0.129, 0.132, 0.118], [C, C, C, B, B, C, B, B]),
    ("marcos-study", _MARCOS_M, [0.1061, 0.3476, 0.3330, 0.1185, 0.0948], [C, B, B, B, B]),
    ("waspas-study", _WASPAS_M, [0.1181, 0.1181, 0.0445, 0.1181, 0.2861, 0.2861, 0.0290], [B, B, B, B, B, C, C]),
    ("ersoy-study", _ERSOY_M, _ERSOY_W, _ERSOY_T),
    ("goswami-study", _GOSWAMI_M, [0.6, 0.14, 0.2, 0.06], [C, B, B, B]),
    ("tus-study", _TUS_M, [0.157, 0.249, 0.168, 0.121, 0.154, 0.151], [C, B, B, B, B, B]),
]

SAATY_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def random_problem(rng: random.Random) -> tuple[list[list[float]], list[float], list[str]]:
    """A positive matrix (4-12 alternatives, 3-8 criteria), weights summing to 1, mixed types."""
    m, n = rng.randint(4, 12), rng.randint(3, 8)
    scales = [10 ** rng.uniform(-2, 4) for _ in range(n)]
    matrix = [[round(scales[j] * rng.lognormvariate(0, 0.6), 6) for j in range(n)] for _ in range(m)]
    raw = [rng.uniform(0.05, 1) for _ in range(n)]
    total = sum(raw)
    weights = [v / total for v in raw]
    types = [B] + [C] + [rng.choice((B, C)) for _ in range(n - 2)]
    rng.shuffle(types)
    return matrix, weights, types


def random_pairwise(rng: random.Random, n: int) -> list[list[float]]:
    """A reciprocal matrix with entries on Saaty's 1-9 scale; usually inconsistent."""
    a = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = rng.choice(SAATY_VALUES)
            a[i][j], a[j][i] = (v, 1 / v) if rng.random() < 0.5 else (1 / v, v)
    return a


def bulanik_score(method: str, matrix, weights, types, **params) -> list[float]:
    """Scores of a bulanik method: the COPRAS utility as a fraction, otherwise ``scores``."""
    import bulanik.mcdm as mcdm

    if method == "ahp":
        return list(mcdm.ahp(matrix).weights.values())
    if method == "entropy":
        return list(mcdm.entropy(matrix).weights.values())
    if method == "critic":
        return list(mcdm.critic(matrix).weights.values())
    r = getattr(mcdm, method)(matrix, weights, types, **params)
    if method == "copras":
        return [v / 100 for v in r.extras["utility"].values()]
    return list(r.scores.values())
