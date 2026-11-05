"""Linguistic scales: named terms mapped to fuzzy numbers.

A [`LinguisticScale`][bulanik.scales.LinguisticScale] turns the words experts use
("high", "very important") into fuzzy numbers of one type, so that rating
tables written as strings can be converted with
[`convert`][bulanik.scales.LinguisticScale.convert] before aggregation.

Built-in scales: [`SAATY_TFN`][bulanik.scales.SAATY_TFN] (Saaty's 1-9 values as
triangles) and [`FUZZY_BWM_TFN`][bulanik.scales.FUZZY_BWM_TFN] (the best-worst
method scale of Guo and Zhao). Besides these, the built-ins
[`EXAMPLE_IFN_5`][bulanik.scales.EXAMPLE_IFN_5],
[`EXAMPLE_IFN_7`][bulanik.scales.EXAMPLE_IFN_7] and
[`EXAMPLE_TFN_7`][bulanik.scales.EXAMPLE_TFN_7] are example scales: illustrative
values shipped without attribution (`source=None`).

Examples:
    >>> from bulanik.scales import SAATY_TFN
    >>> SAATY_TFN["3"]
    TFN(l=2.0, m=3.0, u=4.0)
"""

from __future__ import annotations

from typing import Any, Mapping

from .numbers import IFN, TFN, FuzzyNumber

__all__ = [
    "LinguisticScale",
    "SAATY_TFN",
    "FUZZY_BWM_TFN",
    "EXAMPLE_IFN_5",
    "EXAMPLE_IFN_7",
    "EXAMPLE_TFN_7",
]


class LinguisticScale:
    """An ordered set of linguistic terms, all fuzzy numbers of one type.

    Terms are looked up with `scale["term"]` and tested with `"term" in scale`;
    the term order is the insertion order.

    Args:
        name: A descriptive name of the scale.
        terms: Mapping from term name to fuzzy number; all numbers must be of
            the same class (and rung).
        source: Where the values come from, or `None` for illustrative scales.

    Raises:
        ValueError: If `terms` is empty.
        TypeError: If a term is not a fuzzy number, or the numbers mix types.

    Attributes:
        name: The scale name.
        source: The attribution string, or `None`.
        number_type: The class shared by all term numbers.

    Examples:
        >>> from bulanik import TFN
        >>> from bulanik.scales import LinguisticScale
        >>> s = LinguisticScale("mine", {"low": TFN(0, 0, 5), "high": TFN(5, 10, 10)})
        >>> s.terms
        ('low', 'high')
        >>> s["high"]
        TFN(l=5.0, m=10.0, u=10.0)
        >>> "low" in s
        True
        >>> s.number_type.__name__
        'TFN'
    """

    def __init__(
        self,
        name: str,
        terms: Mapping[str, FuzzyNumber],
        source: str | None = None,
    ) -> None:
        if not terms:
            raise ValueError(f"scale '{name}' needs at least one term")
        for term, v in terms.items():
            if not isinstance(v, FuzzyNumber):
                raise TypeError(
                    f"term '{term}' of scale '{name}' must be a fuzzy number, "
                    f"got {type(v).__name__}"
                )
        types = {type(v) for v in terms.values()}
        if len(types) != 1:
            names = ", ".join(sorted(t.__name__ for t in types))
            raise TypeError(f"scale '{name}' mixes number types: {names}")
        self.name = name
        self.source = source
        self._terms = dict(terms)
        self.number_type: type = types.pop()

    @property
    def terms(self) -> tuple[str, ...]:
        """The term names, in insertion order."""
        return tuple(self._terms)

    def __getitem__(self, term: str) -> Any:
        """The fuzzy number of `term`.

        Args:
            term: A term name.

        Returns:
            The number the term maps to.

        Raises:
            KeyError: If the term does not exist (the message lists the valid terms).
        """
        try:
            return self._terms[term]
        except KeyError:
            raise KeyError(
                f"unknown term '{term}' in scale '{self.name}'; valid: {', '.join(self._terms)}"
            ) from None

    def __contains__(self, term: object) -> bool:
        return term in self._terms

    def convert(self, nested: Any) -> Any:
        """Map term strings at any list depth to their numbers (lists stay lists).

        Args:
            nested: A term name, or a list or tuple of them, nested to any depth.

        Returns:
            The number for a term name; a list with the same nesting otherwise
            (tuples become lists).

        Raises:
            TypeError: If an item is neither a string nor a list or tuple.
            KeyError: If a term name is not in the scale.

        Examples:
            >>> from bulanik.scales import SAATY_TFN
            >>> SAATY_TFN.convert([["1", "9"], "5"])
            [[TFN(l=1.0, m=1.0, u=1.0), TFN(l=9.0, m=9.0, u=9.0)], TFN(l=4.0, m=5.0, u=6.0)]
        """
        if isinstance(nested, str):
            return self[nested]
        if isinstance(nested, (list, tuple)):
            return [self.convert(item) for item in nested]
        raise TypeError(f"expected a term name or a list of them, got {type(nested).__name__}")

    def reciprocal(self, term: str) -> TFN:
        """Reciprocal of a term's number (TFN scales only).

        Args:
            term: A term name.

        Returns:
            The reciprocal `TFN` $1 / A$ (see [`TFN.reciprocal`][bulanik.numbers.TFN.reciprocal]).

        Raises:
            TypeError: If the scale does not hold `TFN` numbers.
            KeyError: If the term does not exist.
            ValueError: If the support of the term's number contains 0.

        Examples:
            >>> from bulanik.scales import SAATY_TFN
            >>> r = SAATY_TFN.reciprocal("3")
            >>> round(r.l, 4), round(r.m, 4), round(r.u, 4)
            (0.25, 0.3333, 0.5)
        """
        if self.number_type is not TFN:
            raise TypeError(
                f"reciprocal needs a TFN scale; '{self.name}' holds {self.number_type.__name__}"
            )
        return self[term].reciprocal()

    def __repr__(self) -> str:
        return f"LinguisticScale({self.name!r}, terms={self.terms})"


SAATY_TFN = LinguisticScale(
    "Saaty 1-9 triangular scale",
    {
        "1": TFN(1, 1, 1),
        **{str(k): TFN(k - 1, k, k + 1) for k in range(2, 9)},
        "9": TFN(9, 9, 9),
    },
    source="Saaty 1980 (crisp 1-9 values); the +/-1 triangular spread is a common convention",
)
"""Saaty's 1-9 scale as triangular numbers: `"1"` = (1,1,1), `"k"` = (k-1,k,k+1)
for k = 2..8, `"9"` = (9,9,9). The crisp 1-9 values are from Saaty (1980); the +/-1
triangular spread is a common convention with no single canonical source."""

FUZZY_BWM_TFN = LinguisticScale(
    "fuzzy BWM linguistic scale",
    {
        "EI": TFN(1, 1, 1),
        "WI": TFN(2 / 3, 1, 3 / 2),
        "FI": TFN(3 / 2, 2, 5 / 2),
        "VI": TFN(5 / 2, 3, 7 / 2),
        "AI": TFN(7 / 2, 4, 9 / 2),
    },
    source="Guo & Zhao 2017, Knowledge-Based Systems 121:23-31",
)
"""The fuzzy best-worst method scale of Guo and Zhao (2017, Knowledge-Based Systems
121:23-31): equally (EI), weakly (WI), fairly (FI), very (VI) and absolutely (AI)
important, as triangular numbers. Their consistency indices are given by
[`fuzzy_bwm_consistency_index`][bulanik.mcdm.bwm.fuzzy_bwm_consistency_index]
(3.00, 3.80, 5.29, 6.69, 8.04)."""

# The three example scales below are illustrations with no stated source.
EXAMPLE_IFN_5 = LinguisticScale(
    "example IFN 5-term scale",
    {
        "VL": IFN(0.1, 0.8),
        "L": IFN(0.3, 0.6),
        "M": IFN(0.5, 0.4),
        "H": IFN(0.7, 0.2),
        "VH": IFN(0.9, 0.05),
    },
)
"""Example 5-term intuitionistic scale (VL, L, M, H, VH); illustrative values with no source."""
EXAMPLE_IFN_7 = LinguisticScale(
    "example IFN 7-term scale",
    {
        "EL": IFN(0.05, 0.9),
        "VL": IFN(0.15, 0.8),
        "L": IFN(0.3, 0.6),
        "M": IFN(0.5, 0.4),
        "H": IFN(0.7, 0.25),
        "VH": IFN(0.85, 0.1),
        "EH": IFN(0.95, 0.05),
    },
)
"""Example 7-term intuitionistic scale (EL, VL, L, M, H, VH, EH); illustrative values with no source."""
EXAMPLE_TFN_7 = LinguisticScale(
    "example TFN 7-term scale",
    {
        "VP": TFN(0, 0, 1),
        "P": TFN(0, 1, 3),
        "MP": TFN(1, 3, 5),
        "F": TFN(3, 5, 7),
        "MG": TFN(5, 7, 9),
        "G": TFN(7, 9, 10),
        "VG": TFN(9, 10, 10),
    },
)
"""Example 7-term triangular rating scale on 0-10 (VP, P, MP, F, MG, G, VG); illustrative values with no source."""
