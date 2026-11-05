"""bulanik: easy-to-use fuzzy numbers, fuzzy inference and fuzzy MCDM.

The package root re-exports the fuzzy number types of [`bulanik.numbers`][bulanik.numbers]
so that the most common objects are one import away:

- [`TFN`][bulanik.numbers.TFN], [`TrFN`][bulanik.numbers.TrFN]: triangular and trapezoidal numbers.
- [`IFN`][bulanik.numbers.IFN], [`PFN`][bulanik.numbers.PFN], [`FFN`][bulanik.numbers.FFN],
  [`QROFN`][bulanik.numbers.QROFN]: orthopair numbers (q = 1, 2, 3, any q).
- [`IVIFN`][bulanik.numbers.IVIFN], [`IVPFN`][bulanik.numbers.IVPFN], [`IVFFN`][bulanik.numbers.IVFFN],
  [`IVQROFN`][bulanik.numbers.IVQROFN]: their interval-valued versions.
- [`SphericalFN`][bulanik.numbers.SphericalFN], [`PictureFN`][bulanik.numbers.PictureFN],
  [`NeutrosophicFN`][bulanik.numbers.NeutrosophicFN]: three-part numbers.
- [`IVNeutrosophicFN`][bulanik.numbers.IVNeutrosophicFN]: interval neutrosophic numbers.
- [`IT2TrFN`][bulanik.numbers.IT2TrFN]: interval type-2 trapezoidal numbers.

Inference, MCDM, aggregation, sets, scales and the optional LumiX / sezgi bridges
live in submodules (`bulanik.inference`, `bulanik.mcdm`, `bulanik.aggregate`,
`bulanik.sets`, `bulanik.scales`, `bulanik.lumix`, `bulanik.sezgi`).

Attributes:
    __version__: The installed package version string.

Examples:
    >>> import bulanik
    >>> bulanik.TFN(1, 2, 4).score()
    2.3333
"""

from .numbers import (
    FFN,
    IFN,
    IT2TrFN,
    IVFFN,
    IVIFN,
    IVNeutrosophicFN,
    IVPFN,
    IVQROFN,
    PFN,
    QROFN,
    TFN,
    NeutrosophicFN,
    PictureFN,
    SphericalFN,
    TrFN,
)

__version__ = "0.3.0"

__all__ = [
    "TFN",
    "TrFN",
    "QROFN",
    "IFN",
    "PFN",
    "FFN",
    "IVQROFN",
    "IVIFN",
    "IVPFN",
    "IVFFN",
    "SphericalFN",
    "PictureFN",
    "NeutrosophicFN",
    "IVNeutrosophicFN",
    "IT2TrFN",
    "__version__",
]
