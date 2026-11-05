"""Multi-criteria decision making: weighting and ranking methods.

The package has two kinds of methods and three result containers.

- **Weighting** (criteria weights): [`ahp`][bulanik.mcdm.weighting.ahp],
  [`fuzzy_ahp`][bulanik.mcdm.weighting.fuzzy_ahp],
  [`ahp_fuzzy_numbers`][bulanik.mcdm.weighting.ahp_fuzzy_numbers],
  [`ahp_interval_valued`][bulanik.mcdm.weighting.ahp_interval_valued],
  [`swara`][bulanik.mcdm.weighting.swara],
  [`piprecia`][bulanik.mcdm.weighting.piprecia],
  [`entropy`][bulanik.mcdm.weighting.entropy] and
  [`critic`][bulanik.mcdm.weighting.critic] return a
  [`WeightingResult`][bulanik.mcdm.results.WeightingResult]. The best-worst
  method needs an optimiser, so it lives in the bridges
  ([`bulanik.lumix`][bulanik.lumix], [`bulanik.sezgi`][bulanik.sezgi]) and
  returns a [`BWMResult`][bulanik.mcdm.results.BWMResult].
- **Ranking** (alternatives): [`topsis`][bulanik.mcdm.ranking.topsis],
  [`vikor`][bulanik.mcdm.ranking.vikor], [`waspas`][bulanik.mcdm.ranking.waspas]
  and [`edas`][bulanik.mcdm.ranking.edas] work on a crisp matrix, as do the
  scoring methods [`codas`][bulanik.mcdm.scoring.codas],
  [`copras`][bulanik.mcdm.scoring.copras], [`marcos`][bulanik.mcdm.scoring.marcos],
  [`cocoso`][bulanik.mcdm.scoring.cocoso] and [`wisp`][bulanik.mcdm.scoring.wisp],
  and the outranking method [`electre`][bulanik.mcdm.outranking.electre];
  [`rank_fuzzy`][bulanik.mcdm.fuzzy.rank_fuzzy] first aggregates expert fuzzy
  ratings from a [`DecisionMatrix`][bulanik.mcdm.fuzzy.DecisionMatrix] and
  scores them; [`fuzzy_topsis`][bulanik.mcdm.fuzzy.fuzzy_topsis] is Chen's
  (2000) method on triangular numbers; and
  [`fuzzy_codas`][bulanik.mcdm.fuzzy_ranking.fuzzy_codas],
  [`fuzzy_waspas`][bulanik.mcdm.fuzzy_ranking.fuzzy_waspas] and
  [`fuzzy_cocoso`][bulanik.mcdm.fuzzy_ranking.fuzzy_cocoso] compute on an
  aggregated matrix of fuzzy numbers and score only at the end. All return a
  [`RankingResult`][bulanik.mcdm.results.RankingResult].

- **Influence analysis**: [`dematel`][bulanik.mcdm.dematel.dematel] (crisp, or
  fuzzy with CFCS defuzzification) returns a
  [`DEMATELResult`][bulanik.mcdm.results.DEMATELResult] whose
  [`as_weights`][bulanik.mcdm.results.DEMATELResult.as_weights] feeds the
  ranking methods.

Every result records the operators it used in its `operators` mapping. The
ranking methods take the criteria weights as a sequence, a mapping by criterion
name, or a weighting result directly.

Examples:
    >>> from bulanik.mcdm import ahp, topsis
    >>> w = ahp([[1, 3], [1/3, 1]], ["cost", "quality"])
    >>> round(w.weights["cost"], 2), w.ranking
    (0.75, ('cost', 'quality'))
    >>> r = topsis([[250, 16], [200, 12], [300, 32]], w, ["cost", "benefit"],
    ...            alternatives=["A", "B", "C"])
    >>> r.ranking
    ('B', 'C', 'A')
"""

from .dematel import dematel
from .fuzzy import DecisionMatrix, fuzzy_topsis, rank_fuzzy
from .fuzzy_ranking import fuzzy_cocoso, fuzzy_codas, fuzzy_waspas
from .outranking import electre
from .ranking import edas, topsis, vikor, waspas
from .results import BWMResult, DEMATELResult, RankingResult, WeightingResult
from .scoring import cocoso, codas, copras, marcos, wisp
from .weighting import ahp, ahp_fuzzy_numbers, ahp_interval_valued, critic, entropy, fuzzy_ahp, piprecia, swara

__all__ = [
    "WeightingResult",
    "RankingResult",
    "BWMResult",
    "DEMATELResult",
    "dematel",
    "ahp",
    "fuzzy_ahp",
    "ahp_fuzzy_numbers",
    "ahp_interval_valued",
    "swara",
    "entropy",
    "piprecia",
    "critic",
    "topsis",
    "vikor",
    "waspas",
    "edas",
    "codas",
    "copras",
    "marcos",
    "cocoso",
    "wisp",
    "electre",
    "DecisionMatrix",
    "rank_fuzzy",
    "fuzzy_topsis",
    "fuzzy_codas",
    "fuzzy_waspas",
    "fuzzy_cocoso",
]
