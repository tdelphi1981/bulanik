r"""Fuzzy number types.

Every type implements the common interface of
[`FuzzyNumber`][bulanik.numbers.FuzzyNumber]: the operations `add`, `mul`,
`scale` and `power`, and `score`, `accuracy` and `distance`. Operations are
selected by name (`ops="algebraic"`, plus `"einstein"` for orthopair numbers),
and combining numbers of different types raises `TypeError`.

Triangular and trapezoidal numbers (`TFN`, `TrFN`)
:   Ordered points on the real line; closed under the operators of interval
    arithmetic on alpha-cuts.

Orthopair numbers (`QROFN`, `IFN`, `PFN`, `FFN`)
:   A pair $(\mu, \nu)$ with $\mu^q + \nu^q \le 1$ for rung $q$
    ($q = 1$ intuitionistic, $q = 2$ Pythagorean, $q = 3$ Fermatean).

Interval-valued orthopair numbers (`IVQROFN`, `IVIFN`, `IVPFN`, `IVFFN`)
:   A membership interval and a non-membership interval with
    $\mu_U^q + \nu_U^q \le 1$; the point laws of the orthopair numbers act on
    the lower and the upper pair.

Three-part numbers (`SphericalFN`, `PictureFN`, `NeutrosophicFN`)
:   A triple of membership, neutral/hesitant and non-membership parts.

Interval neutrosophic numbers (`IVNeutrosophicFN`)
:   Intervals of truth, indeterminacy and falsity; the laws of
    `NeutrosophicFN` act on the lower and the upper bounds.

Interval type-2 trapezoids (`IT2TrFN`)
:   An upper and a lower generalised trapezoid (the footprint of uncertainty);
    the type-1 rules act on both and the heights combine with `min`.

Examples:
    >>> from bulanik.numbers import TFN, IFN
    >>> TFN(1, 2, 3) + TFN(2, 3, 4)
    TFN(l=3.0, m=5.0, u=7.0)
    >>> IFN(0.6, 0.3).score()
    0.3
"""

from ._base import FuzzyNumber
from .it2 import IT2TrFN
from .iv_neutrosophic import IVNeutrosophicFN
from .iv_orthopair import IVFFN, IVIFN, IVPFN, IVQROFN
from .orthopair import FFN, IFN, PFN, QROFN
from .threepart import NeutrosophicFN, PictureFN, SphericalFN
from .triangular import TFN, TrFN

__all__ = [
    "FuzzyNumber",
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
]
