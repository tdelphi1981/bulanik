"""Fuzzy rules and inference systems (type-1).

- Rules: write them as text (see [`parse_rule`][bulanik.inference.rules.parse_rule]
  for the syntax) or build them from [`Term`][bulanik.inference.rules.Term],
  [`And`][bulanik.inference.rules.And], [`Or`][bulanik.inference.rules.Or] and
  [`Rule`][bulanik.inference.rules.Rule]; [`firing_strength`][bulanik.inference.rules.firing_strength]
  evaluates an antecedent.
- Systems: [`MamdaniFIS`][bulanik.inference.mamdani.MamdaniFIS] (fuzzy output sets,
  defuzzified), [`SugenoFIS`][bulanik.inference.sugeno.SugenoFIS] (constant or
  [`Linear`][bulanik.inference.sugeno.Linear] consequents) and
  [`TsukamotoFIS`][bulanik.inference.tsukamoto.TsukamotoFIS] (monotonic consequents
  inverted at the firing strength). All three offer `evaluate` for the crisp
  outputs and `explain` for an [`Explanation`][bulanik.inference.rules.Explanation].
- Interval type-2 versions live in `bulanik.type2`; the grid-partition ANFIS in
  `bulanik.anfis`.
"""

from .mamdani import MamdaniFIS, defuzzify_samples
from .rules import (
    And,
    Explanation,
    NoRuleFiredError,
    Or,
    Rule,
    RuleActivation,
    Term,
    firing_strength,
    parse_rule,
)
from .sugeno import Linear, SugenoFIS, SugenoOutput
from .tsukamoto import TsukamotoFIS

__all__ = [
    "And",
    "Explanation",
    "Linear",
    "MamdaniFIS",
    "NoRuleFiredError",
    "Or",
    "Rule",
    "RuleActivation",
    "SugenoFIS",
    "SugenoOutput",
    "Term",
    "TsukamotoFIS",
    "defuzzify_samples",
    "firing_strength",
    "parse_rule",
]
