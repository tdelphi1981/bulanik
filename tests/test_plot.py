"""Tests for bulanik.plot (Agg backend; artists, limits, validation, themes)."""

import importlib
import math
import sys

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from conftest import SERVICE, FOOD, TIPPER_RULES, TIP, X, OUT  # noqa: E402

from bulanik import (  # noqa: E402
    FFN,
    IFN,
    IT2TrFN,
    IVIFN,
    IVPFN,
    PFN,
    QROFN,
    TFN,
    NeutrosophicFN,
    PictureFN,
    SphericalFN,
    TrFN,
)
from bulanik import plot  # noqa: E402
from bulanik.inference import Linear, MamdaniFIS, SugenoFIS, SugenoOutput  # noqa: E402
from bulanik.mcdm import ahp, dematel, topsis, vikor  # noqa: E402
from bulanik.mcdm.results import BWMResult  # noqa: E402
from bulanik.scales import EXAMPLE_IFN_5, EXAMPLE_TFN_7, FUZZY_BWM_TFN, SAATY_TFN, LinguisticScale  # noqa: E402
from bulanik.sets import LinguisticVariable, gaussian, sigmoid, triangular  # noqa: E402
from bulanik.type2 import (  # noqa: E402
    IT2LinguisticVariable,
    IT2MF,
    gaussian_uncertain_mean,
    gaussian_uncertain_sigma,
    scaled_triangle,
)


@pytest.fixture(autouse=True)
def _close_figures(monkeypatch):
    def no_show(*a, **k):
        raise AssertionError("bulanik.plot must never call plt.show()")

    monkeypatch.setattr(plt, "show", no_show)
    yield
    plt.close("all")


def tagged(ax, gid):
    """Artists of ``ax`` whose gid is ``gid`` (lines, collections, patches, texts)."""
    pool = list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.texts)
    return [a for a in pool if a.get_gid() == gid]


def texts(ax):
    return [t.get_text() for t in ax.texts]


def fis_tipper():
    return MamdaniFIS([SERVICE, FOOD], [TIP], TIPPER_RULES, resolution=301)


# -- style -------------------------------------------------------------------


def test_palette_has_eight_validated_slots_per_theme():
    assert set(plot.PALETTE) == {"light", "dark"}
    assert len(plot.PALETTE["light"]) == len(plot.PALETTE["dark"]) == 8
    assert plot.PALETTE["light"][0] == "#2a78d6"
    assert plot.PALETTE["dark"][0] == "#3987e5"
    with pytest.raises(TypeError):
        plot.PALETTE["light"] = ()  # read-only


def test_theme_sets_and_restores_rc():
    before = matplotlib.rcParams["figure.facecolor"]
    assert plot.current_theme()["name"] == "light"
    with plot.theme("dark"):
        assert plot.current_theme()["name"] == "dark"
        assert matplotlib.colors.to_hex(matplotlib.rcParams["figure.facecolor"]) == "#1e2129"
    assert matplotlib.rcParams["figure.facecolor"] == before
    assert plot.current_theme()["name"] == "light"


def test_theme_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown theme 'sepia'"):
        with plot.theme("sepia"):
            pass  # pragma: no cover


def test_current_theme_is_read_only():
    t = plot.current_theme()
    assert {"surface", "text", "text_secondary", "muted", "grid", "axis", "series", "ramp"} <= set(t)
    with pytest.raises(TypeError):
        t["surface"] = "#000000"


def test_dark_theme_uses_dark_palette_and_surface():
    with plot.theme("dark"):
        ax = plot.membership(TFN(1, 2, 4))
    (line,) = tagged(ax, "membership")
    assert matplotlib.colors.to_hex(line.get_color()) == plot.PALETTE["dark"][0]
    assert matplotlib.colors.to_hex(ax.get_facecolor()) == "#1e2129"


def test_import_without_matplotlib_gives_install_hint(monkeypatch):
    for name in [m for m in sys.modules if m == "bulanik.plot" or m.startswith("bulanik.plot.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match=r'pip install "bulanik\[plot\]"'):
        importlib.import_module("bulanik.plot")


def test_given_axes_is_used_and_no_figure_created():
    fig, ax = plt.subplots()
    n = len(plt.get_fignums())
    assert plot.membership([TFN(1, 2, 4)], ax=ax) is ax
    assert len(plt.get_fignums()) == n


# -- membership --------------------------------------------------------------


def test_membership_lines_fills_labels_and_limits():
    ax = plot.membership([TFN(1, 2, 4), TrFN(3, 4, 5, 7)], labels=["A", "B"])
    assert len(tagged(ax, "membership")) == 2
    assert len(tagged(ax, "fill")) == 2
    assert {"A", "B"} <= set(texts(ax))
    lo, hi = ax.get_xlim()
    assert lo < 1 and hi > 7
    assert ax.get_ylim()[0] == 0 and ax.get_ylim()[1] >= 1
    xs, ys = tagged(ax, "membership")[0].get_data()
    assert list(xs) == [1, 2, 4] and list(ys) == [0, 1, 0]


def test_membership_single_number_ticks_its_points():
    ax = plot.membership(TFN(1, 2, 4))
    assert [t for t in ax.get_xticks()] == [1, 2, 4]


def test_membership_without_fill_and_with_alpha_cut():
    ax = plot.membership([TFN(1, 2, 4), TFN(2, 3, 6)], fill=False, alpha_cut=0.5)
    assert tagged(ax, "fill") == []
    cuts = tagged(ax, "alpha-cut")
    assert len(cuts) == 2
    xs, ys = cuts[0].get_data()
    assert tuple(xs) == TFN(1, 2, 4).alpha_cut(0.5) and tuple(ys) == (0.5, 0.5)
    assert any("0.5" in t for t in texts(ax))


def test_membership_crisp_number_is_drawn():
    ax = plot.membership(TFN(2, 2, 2))
    lo, hi = ax.get_xlim()
    assert lo < 2 < hi


@pytest.mark.parametrize(
    "kwargs, err, match",
    [
        ({"numbers": []}, ValueError, "at least one"),
        ({"numbers": [IFN(0.5, 0.3)]}, TypeError, "TFN or TrFN"),
        ({"numbers": "abc"}, TypeError, "TFN or TrFN"),
        ({"numbers": [TFN(1, 2, 3)], "labels": ["a", "b"]}, ValueError, "2 labels for 1"),
        ({"numbers": [TFN(1, 2, 3)], "alpha_cut": 1.5}, ValueError, "alpha_cut"),
        ({"numbers": [TFN(i, i + 1, i + 2) for i in range(9)]}, ValueError, "at most 8"),
    ],
)
def test_membership_rejects_bad_input(kwargs, err, match):
    numbers = kwargs.pop("numbers")
    with pytest.raises(err, match=match):
        plot.membership(numbers, **kwargs)


# -- alpha_cuts --------------------------------------------------------------


def test_alpha_cuts_draws_one_interval_per_level():
    n = TFN(1, 2, 4)
    ax = plot.alpha_cuts(n, levels=5)
    cuts = tagged(ax, "alpha-cut")
    assert len(cuts) == 5
    for line, a in zip(cuts, [0, 0.25, 0.5, 0.75, 1]):
        xs, ys = line.get_data()
        assert tuple(xs) == pytest.approx(n.alpha_cut(a)) and ys[0] == pytest.approx(a)
    assert len(tagged(ax, "membership")) == 1


def test_alpha_cuts_accepts_explicit_levels_and_trapezoid():
    ax = plot.alpha_cuts(TrFN(1, 2, 3, 5), levels=[0.2, 0.9])
    assert len(tagged(ax, "alpha-cut")) == 2


@pytest.mark.parametrize(
    "number, levels, err",
    [
        (TFN(1, 2, 3), 1, ValueError),
        (TFN(1, 2, 3), True, TypeError),
        (TFN(1, 2, 3), [], ValueError),
        (TFN(1, 2, 3), [0.5, 1.2], ValueError),
        (IFN(0.4, 0.4), 5, TypeError),
    ],
)
def test_alpha_cuts_rejects_bad_input(number, levels, err):
    with pytest.raises(err):
        plot.alpha_cuts(number, levels=levels)


# -- defuzzification ----------------------------------------------------------


def test_defuzzification_merges_coincident_scores():
    n = TFN(1, 2, 4)
    ax = plot.defuzzification(n)
    marks = tagged(ax, "score")
    assert len(marks) == 2  # centroid == mean for a triangle
    xs = sorted(m.get_xdata()[0] for m in marks)
    assert xs == pytest.approx(sorted({n.score("graded_mean"), n.score("centroid")}))
    assert any("centroid = mean" in t for t in texts(ax))


def test_defuzzification_trapezoid_and_subset():
    assert len(tagged(plot.defuzzification(TrFN(1, 2, 3, 6)), "score")) == 3
    assert len(tagged(plot.defuzzification(TFN(1, 2, 4), scores=["graded_mean"]), "score")) == 1


def test_defuzzification_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown score 'median'"):
        plot.defuzzification(TFN(1, 2, 3), scores=["median"])
    with pytest.raises(ValueError, match="at least one"):
        plot.defuzzification(TFN(1, 2, 3), scores=[])
    with pytest.raises(TypeError):
        plot.defuzzification(PFN(0.5, 0.5))


# -- orthopair ---------------------------------------------------------------


def test_orthopair_points_regions_and_hesitancy():
    nums = [IFN(0.6, 0.3), PFN(0.7, 0.6), FFN(0.8, 0.7)]
    ax = plot.orthopair(nums, regions=(1, 2, 3), labels=["a", "b", "c"], hesitancy=True)
    assert len(tagged(ax, "point")) == 3
    assert len(tagged(ax, "boundary")) == 3
    assert len(tagged(ax, "hesitancy")) == 3
    assert {"a", "b", "c"} <= set(texts(ax))
    assert ax.get_xlim() == pytest.approx((0, 1.0), abs=0.05)
    assert ax.get_aspect() == 1.0
    # the hesitancy segment of the IFN ends on its own boundary mu + nu = 1
    xs, ys = tagged(ax, "hesitancy")[0].get_data()
    assert xs[-1] + ys[-1] == pytest.approx(1.0)


def test_orthopair_defaults_regions_to_rungs_and_allows_no_points():
    ax = plot.orthopair([IFN(0.5, 0.2), QROFN(0.9, 0.5, 5)])
    assert len(tagged(ax, "boundary")) == 2
    ax = plot.orthopair([], q=2)
    assert len(tagged(ax, "boundary")) == 1 and tagged(ax, "point") == []
    assert len(tagged(plot.orthopair([]), "boundary")) == 1  # q = 1


def test_orthopair_hesitancy_zero_draws_nothing():
    ax = plot.orthopair([IFN(0.6, 0.4)], hesitancy=True)
    assert tagged(ax, "hesitancy") == []


@pytest.mark.parametrize(
    "numbers, kwargs, err",
    [
        ([TFN(1, 2, 3)], {}, TypeError),
        ([IFN(0.5, 0.3)], {"q": 0}, ValueError),
        ([IFN(0.5, 0.3)], {"regions": ()}, ValueError),
        ([IFN(0.5, 0.3)], {"regions": (1, 2.5)}, ValueError),
        ([IFN(0.5, 0.3)], {"labels": ["a", "b"]}, ValueError),
    ],
)
def test_orthopair_rejects_bad_input(numbers, kwargs, err):
    with pytest.raises(err):
        plot.orthopair(numbers, **kwargs)


# -- picture / spherical / neutrosophic -------------------------------------------


def test_picture_ternary():
    ax = plot.picture([PictureFN(0.5, 0.2, 0.1), PictureFN(0.2, 0.3, 0.4)], labels=["p", "r"])
    assert len(tagged(ax, "point")) == 2
    assert len(tagged(ax, "simplex")) == 1
    assert {"p", "r"} <= set(texts(ax))
    assert not ax.axison or not ax.xaxis.get_visible()


def test_picture_point_position_is_barycentric():
    ax = plot.picture(PictureFN(1.0, 0.0, 0.0))
    (pt,) = tagged(ax, "point")
    x, y = pt.get_offsets()[0]
    v = tagged(ax, "simplex")[0].get_xy()[0]  # the mu vertex comes first
    assert (x, y) == pytest.approx(tuple(v))


def test_picture_rejects_bad_input():
    with pytest.raises(TypeError):
        plot.picture([IFN(0.5, 0.3)])
    with pytest.raises(ValueError):
        plot.picture([])
    with pytest.raises(ValueError, match="all three parts are 0"):
        plot.picture(PictureFN(0, 0, 0))


def test_spherical_is_3d():
    ax = plot.spherical([SphericalFN(0.5, 0.4, 0.3), SphericalFN(0.8, 0.2, 0.1)], labels=["a", "b"])
    assert ax.name == "3d"
    assert len(tagged(ax, "point")) == 2
    with pytest.raises(ValueError, match="3-D axes"):
        plot.spherical([SphericalFN(0.5, 0.4, 0.3)], ax=plt.subplots()[1])
    with pytest.raises(TypeError):
        plot.spherical([PictureFN(0.5, 0.2, 0.1)])


def test_neutrosophic_radar():
    ax = plot.neutrosophic([NeutrosophicFN(0.7, 0.2, 0.1), NeutrosophicFN(0.3, 0.6, 0.5)])
    assert ax.name == "rectilinear"
    assert len(tagged(ax, "radar")) == 2
    xs, ys = tagged(ax, "radar")[0].get_data()
    assert (xs[0], ys[0]) == pytest.approx((0, 0.7), abs=1e-12)  # truth on the upward spoke
    assert ax.get_legend() is not None
    assert plot.neutrosophic(NeutrosophicFN(0.7, 0.2, 0.1)).get_legend() is None
    with pytest.raises(ValueError, match="2-D"):
        plot.neutrosophic([NeutrosophicFN(0.7, 0.2, 0.1)], ax=plt.figure().add_subplot(projection="polar"))
    with pytest.raises(TypeError):
        plot.neutrosophic([SphericalFN(0.5, 0.4, 0.3)])
    with pytest.raises(ValueError, match="at most 8"):
        plot.neutrosophic([NeutrosophicFN(0.1, 0.1, 0.1)] * 9)


def test_three_part_accept_given_axes():
    fig = plt.figure()
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    axp = fig.add_subplot(1, 2, 2)
    assert plot.spherical(SphericalFN(0.5, 0.4, 0.3), ax=ax3) is ax3
    assert plot.neutrosophic(NeutrosophicFN(0.5, 0.4, 0.3), ax=axp) is axp


# -- variable ------------------------------------------------------------------


def test_variable_of_linguistic_variable_with_input_marker():
    ax = plot.variable(SERVICE, x=3)
    assert len(tagged(ax, "membership")) == 3
    assert set(SERVICE.term_names) <= set(texts(ax))
    dots = tagged(ax, "degree")
    assert len(dots) == 2  # poor .4 and average .6; good is 0
    assert sorted(d.get_ydata()[0] for d in dots) == pytest.approx([0.4, 0.6])
    assert ax.get_xlim() == pytest.approx(SERVICE.universe)
    assert ax.get_xlabel() == "service"


def test_variable_of_smooth_terms():
    v = LinguisticVariable("t", (0, 10), {"g": gaussian(5, 1), "s": sigmoid(2, 7)})
    ax = plot.variable(v)
    assert len(tagged(ax, "membership")) == 2


def test_variable_of_tfn_scales():
    ax = plot.variable(EXAMPLE_TFN_7, x=4)
    assert len(tagged(ax, "membership")) == 7
    assert len(tagged(plot.variable(SAATY_TFN), "membership")) == 9  # ordinal ramp past 8
    assert len(tagged(plot.variable(FUZZY_BWM_TFN), "membership")) == 5


def test_variable_delegates_other_scales():
    ax = plot.variable(EXAMPLE_IFN_5)
    assert len(tagged(ax, "point")) == 5 and set(EXAMPLE_IFN_5.terms) <= set(texts(ax))
    pics = LinguisticScale("p", {"lo": PictureFN(0.2, 0.1, 0.6), "hi": PictureFN(0.7, 0.1, 0.1)})
    assert len(tagged(plot.variable(pics), "point")) == 2
    neu = LinguisticScale("n", {"lo": NeutrosophicFN(0.2, 0.1, 0.6)})
    assert len(tagged(plot.variable(neu), "radar")) == 1
    sph = LinguisticScale("s", {"lo": SphericalFN(0.2, 0.1, 0.6)})
    assert plot.variable(sph).name == "3d"
    it2 = IT2LinguisticVariable.from_type1(X)
    assert len(tagged(plot.variable(it2), "fou")) == 2


def test_variable_rejects_bad_input():
    with pytest.raises(ValueError, match="outside"):
        plot.variable(SERVICE, x=11)
    with pytest.raises(ValueError, match="outside the scale range"):
        plot.variable(EXAMPLE_TFN_7, x=11)
    with pytest.raises(ValueError, match="crisp input"):
        plot.variable(EXAMPLE_IFN_5, x=0.5)
    with pytest.raises(TypeError):
        plot.variable([TFN(1, 2, 3)])
    big = LinguisticVariable("b", (0, 1), {f"t{i}": triangular(0, i / 20, 1) for i in range(9)})
    assert len(tagged(plot.variable(big), "membership")) == 9


# -- inference -------------------------------------------------------------------


def test_mamdani_output_shows_consequents_aggregate_and_crisp_value():
    fis = fis_tipper()
    inputs = {"service": 3, "food": 8}
    ax = plot.mamdani_output(fis, inputs)
    fired = [a for a in fis.explain(inputs).activations if a.strength > 0]
    assert len(tagged(ax, "consequent")) == len(fired)
    assert len(tagged(ax, "aggregate")) == 1
    (crisp,) = tagged(ax, "crisp")
    assert crisp.get_xdata()[0] == pytest.approx(fis.evaluate(inputs)["tip"])
    assert ax.get_xlim() == pytest.approx(TIP.universe)


def test_mamdani_output_with_prod_sum_and_default():
    fis = MamdaniFIS([SERVICE, FOOD], [TIP], TIPPER_RULES, implication="prod", aggregation="sum")
    assert len(tagged(plot.mamdani_output(fis, {"service": 5, "food": 5}, "tip"), "aggregate")) == 1
    lo = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 3)})
    y = LinguisticVariable("y", (0, 10), {"m": triangular(0, 5, 10)})
    fis = MamdaniFIS([lo], [y], ["IF x IS lo THEN y IS m"], default={"y": 2})
    ax = plot.mamdani_output(fis, {"x": 8})
    (crisp,) = tagged(ax, "crisp")
    assert crisp.get_xdata()[0] == 2
    assert tagged(ax, "consequent") == []


def test_mamdani_output_rejects_bad_input():
    fis = fis_tipper()
    with pytest.raises(ValueError, match="unknown output 'speed'"):
        plot.mamdani_output(fis, {"service": 3, "food": 8}, "speed")
    with pytest.raises(TypeError, match="MamdaniFIS"):
        plot.mamdani_output(SugenoFIS([X], [OUT], ["IF x IS low THEN y IS f_low"]), {"x": 1})
    with pytest.raises(ValueError, match="outside"):
        plot.mamdani_output(fis, {"service": 30, "food": 8})


def test_surface_two_inputs_is_3d_surface():
    fis = fis_tipper()
    ax = plot.surface(fis, resolution=9)
    assert ax.name == "3d"
    (surf,) = tagged(ax, "surface")
    assert ax.get_xlabel() == "service" and ax.get_ylabel() == "food"


def test_surface_one_input_is_curve_and_gaps_on_no_fire():
    sug = SugenoFIS([X], [OUT], ["IF x IS low THEN y IS f_low", "IF x IS high THEN y IS f_high"])
    ax = plot.surface(sug, "y", resolution=11)
    (curve,) = tagged(ax, "surface")
    xs, ys = curve.get_data()
    assert len(xs) == 11 and ys[0] == pytest.approx(sug.evaluate({"x": 0})["y"])
    lo = LinguisticVariable("x", (0, 10), {"lo": triangular(0, 0, 3)})
    y = LinguisticVariable("y", (0, 10), {"m": triangular(0, 5, 10)})
    fis = MamdaniFIS([lo], [y], ["IF x IS lo THEN y IS m"])
    xs, ys = tagged(plot.surface(fis, resolution=11), "surface")[0].get_data()
    assert math.isnan(ys[-1]) and not math.isnan(ys[0])


def test_surface_rejects_bad_input():
    fis = fis_tipper()
    with pytest.raises(ValueError, match="resolution"):
        plot.surface(fis, resolution=1)
    with pytest.raises(ValueError, match="3-D axes"):
        plot.surface(fis, resolution=5, ax=plt.subplots()[1])
    three = MamdaniFIS(
        [SERVICE, FOOD, LinguisticVariable("z", (0, 1), {"a": triangular(0, 0, 1)})],
        [TIP],
        TIPPER_RULES,
    )
    with pytest.raises(ValueError, match="1 or 2 inputs"):
        plot.surface(three)
    with pytest.raises(TypeError):
        plot.surface("fis")
    with pytest.raises(ValueError, match="unknown output"):
        plot.surface(fis, "nope")


def test_it2_set_single_mf_and_variable():
    mf = gaussian_uncertain_mean(4, 6, 1)
    ax = plot.it2_set(mf, label="medium")
    assert len(tagged(ax, "fou")) == 1
    assert len(tagged(ax, "upper")) == 1 and len(tagged(ax, "lower")) == 1
    assert "medium" in texts(ax)
    lo, hi = ax.get_xlim()
    assert lo < 4 - 2 and hi > 6 + 2
    for mf in (
        gaussian_uncertain_sigma(5, 1, 2),
        scaled_triangle(0, 5, 10, 0.6),
        IT2MF(upper=sigmoid(1, 5), lower=sigmoid(1, 6)),
        IT2MF(upper=triangular(0, 5, 10), lower=triangular(2, 5, 8)),
    ):
        assert len(tagged(plot.it2_set(mf), "fou")) == 1
    v = IT2LinguisticVariable("x", (0, 10), {"lo": scaled_triangle(0, 0, 10, 0.5), "hi": scaled_triangle(0, 10, 10, 0.5)})
    ax = plot.it2_set(v)
    assert len(tagged(ax, "fou")) == 2 and {"lo", "hi"} <= set(texts(ax))
    assert ax.get_xlim() == pytest.approx((0, 10))
    assert ax.get_xlabel() == "x"


def test_it2_set_universe_and_errors():
    ax = plot.it2_set(gaussian_uncertain_mean(4, 6, 1), universe=(0, 20))
    assert ax.get_xlim() == pytest.approx((0, 20))
    with pytest.raises(ValueError, match="universe"):
        plot.it2_set(gaussian_uncertain_mean(4, 6, 1), universe=(5, 1))
    with pytest.raises(TypeError):
        plot.it2_set(triangular(0, 1, 2))


# -- MCDM --------------------------------------------------------------------------


MATRIX = [[250, 16, 12, 5], [200, 16, 8, 3], [300, 32, 16, 4], [275, 32, 8, 4]]
W = [0.3, 0.3, 0.2, 0.2]
TYPES = ["cost", "benefit", "benefit", "benefit"]


def test_ranking_orders_best_first():
    r = topsis(MATRIX, W, TYPES)
    ax = plot.ranking(r)
    dots = tagged(ax, "dot")
    assert len(dots) == len(r.alternatives)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert [lab.split()[-1] for lab in labels] == list(r.ranking)
    assert labels[0].startswith("1")
    assert "higher is better" in ax.get_xlabel()


def test_ranking_vikor_is_lower_better():
    ax = plot.ranking(vikor(MATRIX, W, TYPES))
    assert "lower is better" in ax.get_xlabel()


def test_weights_bars_match_result():
    r = ahp([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]], criteria=["cost", "quality", "risk"])
    ax = plot.weights(r)
    bars = tagged(ax, "bar")
    assert len(bars) == 3
    widths = sorted(b.get_width() for b in bars)
    assert widths == pytest.approx(sorted(r.weights.values()))
    assert ax.get_title(loc="right").startswith("CR = ")


def test_weights_bwm_result():
    r = BWMResult(
        method="bwm",
        criteria=("a", "b"),
        weights={"a": 0.7, "b": 0.3},
        ranking=("a", "b"),
        xi=0.05,
        consistency_ratio=None,
        best="a",
        worst="b",
        operators={},
    )
    assert len(tagged(plot.weights(r), "bar")) == 2


def test_mcdm_reject_wrong_results():
    r = topsis(MATRIX, W, TYPES)
    with pytest.raises(TypeError, match="RankingResult"):
        plot.ranking({"a": 1})
    with pytest.raises(TypeError, match="WeightingResult or BWMResult"):
        plot.weights(r)


DEMATEL_A = [[0, 2, 4, 3, 4], [3, 0, 2, 0, 0], [3, 3, 0, 0, 0], [3, 0, 4, 0, 2], [4, 0, 2, 0, 0]]
DEMATEL_F = ["cost", "quality", "service", "delivery", "risk"]
IV_NUMS = [IVIFN(0.5, 0.6, 0.2, 0.3), IVIFN(0.3, 0.45, 0.4, 0.5), IVPFN(0.6, 0.8, 0.3, 0.5)]
IV_NAMES = ["A", "B", "C"]
IT2_NUMS = [
    IT2TrFN((0, 0.1, 0.1, 0.3, 1, 1), (0.05, 0.1, 0.1, 0.2, 0.9, 0.9)),
    IT2TrFN((0.3, 0.5, 0.5, 0.7, 1, 1), (0.4, 0.5, 0.5, 0.6, 0.8, 0.8)),
]


# -- smoke: every function in both themes, saved as SVG ------------------------------


def _all_figures():
    fis = fis_tipper()
    return {
        "membership": lambda: plot.membership([TFN(1, 2, 4), TrFN(3, 4, 5, 7)], labels=["A", "B"], alpha_cut=0.4),
        "alpha_cuts": lambda: plot.alpha_cuts(TFN(1, 2, 4)),
        "defuzzification": lambda: plot.defuzzification(TrFN(1, 2, 3, 6)),
        "orthopair": lambda: plot.orthopair([IFN(0.6, 0.3), PFN(0.7, 0.6)], regions=(1, 2, 3), hesitancy=True),
        "picture": lambda: plot.picture([PictureFN(0.5, 0.2, 0.1)]),
        "spherical": lambda: plot.spherical([SphericalFN(0.5, 0.4, 0.3)]),
        "neutrosophic": lambda: plot.neutrosophic([NeutrosophicFN(0.7, 0.2, 0.1)]),
        "variable": lambda: plot.variable(SERVICE, x=3),
        "mamdani_output": lambda: plot.mamdani_output(fis, {"service": 3, "food": 8}),
        "surface": lambda: plot.surface(fis, resolution=7),
        "it2_set": lambda: plot.it2_set(gaussian_uncertain_mean(4, 6, 1)),
        "ranking": lambda: plot.ranking(topsis(MATRIX, W, TYPES)),
        "weights": lambda: plot.weights(ahp([[1, 3], [1 / 3, 1]])),
        "dematel": lambda: plot.dematel(dematel(DEMATEL_A, DEMATEL_F)),
        "iv_orthopair": lambda: plot.iv_orthopair(IV_NUMS, labels=IV_NAMES),
        "it2_number": lambda: plot.it2_number(IT2_NUMS, labels=["low", "medium"]),
    }


@pytest.mark.parametrize("name", sorted(_all_figures()))
@pytest.mark.parametrize("mode", ["light", "dark"])
def test_smoke_saves_svg(name, mode, tmp_path):
    with plot.theme(mode):
        ax = _all_figures()[name]()
        path = tmp_path / f"{name}-{mode}.svg"
        ax.figure.savefig(path)
    assert path.stat().st_size > 1000


def test_every_public_function_is_smoke_tested():
    public = {n for n in plot.__all__ if callable(getattr(plot, n))} - {"theme", "current_theme"}
    assert public == set(_all_figures())


# -- edge cases ----------------------------------------------------------------------


def test_axes_argument_must_be_axes():
    with pytest.raises(TypeError, match="matplotlib Axes"):
        plot.membership(TFN(1, 2, 3), ax="axes")


def test_single_string_label_and_q_with_regions():
    assert "A" in texts(plot.membership(TFN(1, 2, 3), labels="A"))
    with pytest.raises(ValueError, match="q or regions"):
        plot.orthopair([], q=2, regions=(1,))


def test_empty_three_part_inputs():
    with pytest.raises(ValueError, match="at least one"):
        plot.spherical([])
    with pytest.raises(ValueError, match="at least one"):
        plot.neutrosophic([])


def test_it2_variable_rejects_crisp_input_and_bad_universe():
    with pytest.raises(ValueError, match="type-1"):
        plot.variable(IT2LinguisticVariable.from_type1(X), x=3)
    with pytest.raises(ValueError, match="pair"):
        plot.it2_set(gaussian_uncertain_mean(4, 6, 1), universe=5)


def test_it2_extent_of_bell_terms():
    from bulanik.sets import bell

    ax = plot.it2_set(IT2MF(upper=bell(2, 2, 5), lower=bell(1, 2, 5)))
    lo, hi = ax.get_xlim()
    assert lo < 5 - 4 and hi > 5 + 4


def test_mamdani_output_sum_aggregation_above_one():
    from bulanik.sets import trapezoidal

    x = LinguisticVariable("x", (0, 10), {"a": trapezoidal(0, 0, 6, 10), "b": trapezoidal(0, 4, 10, 10)})
    y = LinguisticVariable("y", (0, 10), {"m": triangular(0, 5, 10)})
    fis = MamdaniFIS([x], [y], ["IF x IS a THEN y IS m", "IF x IS b THEN y IS m"], aggregation="sum")
    ax = plot.mamdani_output(fis, {"x": 5})
    assert max(tagged(ax, "aggregate")[0].get_ydata()) == pytest.approx(2.0)
    assert ax.get_ylim()[1] > 2.0


# -- fix round 1: positions, values and label placement ----------------------------------


def _label_centres(ax, names):
    ax.figure.draw_without_rendering()
    out = {}
    for txt in ax.texts:
        if txt.get_text() in names:
            bb = txt.get_window_extent()
            out[txt.get_text()] = ((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2, bb)
    assert set(out) == set(names)
    return out


def _nearest_is_own(centres, names, disp):
    for name, (cx, cy, _) in centres.items():
        d = [(cx - x) ** 2 + (cy - y) ** 2 for x, y in disp]
        assert d.index(min(d)) == names.index(name), name


CROWD = [IFN(0.6, 0.3), PFN(0.7, 0.6), FFN(0.85, 0.7)]
CROWD_NAMES = ["IFN(0.6, 0.3)", "PFN(0.7, 0.6)", "FFN(0.85, 0.7)"]


def test_orthopair_point_positions():
    ax = plot.orthopair(CROWD)
    got = [float(v) for p in tagged(ax, "point") for v in p.get_offsets()[0]]
    assert got == pytest.approx([v for n in CROWD for v in (n.mu, n.nu)])


def test_orthopair_labels_sit_by_their_own_point_and_clear_of_segments():
    ax = plot.orthopair(CROWD, regions=(1, 2, 3, 5), labels=CROWD_NAMES, hesitancy=True)
    centres = _label_centres(ax, CROWD_NAMES)
    disp = [tuple(ax.transData.transform((n.mu, n.nu))) for n in CROWD]
    _nearest_is_own(centres, CROWD_NAMES, disp)
    boxes = [bb for *_, bb in centres.values()]
    for i, a in enumerate(boxes):  # no two labels overlap
        assert not any(a.overlaps(b) for b in boxes[i + 1:])
    for seg in tagged(ax, "hesitancy"):  # no label crosses a hesitancy segment
        xs, ys = seg.get_data()
        for f in (0.25, 0.5, 0.75):
            x, y = ax.transData.transform((xs[0] + f * (xs[1] - xs[0]), ys[0] + f * (ys[1] - ys[0])))
            assert not any(bb.contains(x, y) for bb in boxes)


def test_orthopair_curve_labels_keep_clear_of_points():
    crowd = CROWD + [FFN(0.8, 0.78)]  # sits where the q = 3 label would go at 45 degrees
    ax = plot.orthopair(crowd, regions=(1, 2, 3, 5))
    labels = tagged(ax, "region-label")
    assert len(labels) == 4
    ax.figure.draw_without_rendering()
    per_px = 1.0 / ax.get_window_extent().width  # data units per pixel (x range is 1)
    for lab in labels:
        rot = lab.get_rotation()
        lab.set_rotation(0)
        half = lab.get_window_extent().width / 2 * per_px
        lab.set_rotation(rot)
        x, y = lab.get_position()  # centre of the text (ha=va=center)
        tx, ty = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        for n in crowd:  # distance from the point to the text's centreline
            vx, vy = n.mu - (x - half * tx), n.nu - (y - half * ty)
            f = max(0.0, min(2 * half, vx * tx + vy * ty))
            gap = math.hypot(vx - f * tx, vy - f * ty)
            assert gap >= 0.05, (lab.get_text(), n)


def test_picture_and_spherical_labels_sit_by_their_own_point():
    nums = [PictureFN(0.5, 0.2, 0.1), PictureFN(0.45, 0.25, 0.15), PictureFN(0.1, 0.1, 0.8)]
    ax = plot.picture(nums, labels=["a", "b", "c"])
    disp = [tuple(ax.transData.transform(p.get_offsets()[0])) for p in tagged(ax, "point")]
    _nearest_is_own(_label_centres(ax, ["a", "b", "c"]), ["a", "b", "c"], disp)
    sph = [SphericalFN(0.5, 0.4, 0.3), SphericalFN(0.55, 0.4, 0.35), SphericalFN(0.2, 0.3, 0.8)]
    ax = plot.spherical(sph, labels=["a", "b", "c"])
    centres = _label_centres(ax, ["a", "b", "c"])
    from bulanik.plot.numbers import _projected

    disp = [tuple(ax.figure.transFigure.transform(_projected(ax, n.parts))) for n in sph]
    _nearest_is_own(centres, ["a", "b", "c"], disp)


def test_spherical_point_coordinates():
    nums = [SphericalFN(0.5, 0.4, 0.3), SphericalFN(0.8, 0.2, 0.1)]
    ax = plot.spherical(nums)
    for pt, n in zip(tagged(ax, "point"), nums):
        assert tuple(float(v[0]) for v in pt._offsets3d) == pytest.approx(n.parts)


def test_spherical_sphere_patch_is_light():
    # the uniform, unshaded patch needs few facets; each facet is one SVG path
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax = plot.spherical(SphericalFN(0.5, 0.4, 0.3))
    ax.figure.canvas.draw()  # facets are projected at draw time
    patches = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
    assert len(patches) == 1
    assert len(patches[0].get_paths()) <= 400


def test_variable_merges_coincident_degrees():
    ax = plot.variable(EXAMPLE_TFN_7, x=4)
    assert "MP · F  0.5" in texts(ax)
    assert len(tagged(ax, "degree")) == 2  # one ring per term, nested
    assert "poor  0.4" in texts(plot.variable(SERVICE, x=3))


def test_mamdani_aggregate_peak_and_centroid():
    fis = fis_tipper()
    inputs = {"service": 3, "food": 8}
    ax = plot.mamdani_output(fis, inputs)
    xs, ys = tagged(ax, "aggregate")[0].get_data()
    ys = [0.0 if math.isnan(y) else y for y in ys]
    strongest = max(a.strength for a in fis.explain(inputs).activations)
    assert max(ys) == pytest.approx(strongest)
    centroid = sum(x * y for x, y in zip(xs, ys)) / sum(ys)
    assert centroid == pytest.approx(fis.evaluate(inputs)["tip"], abs=0.05)


def test_surface_heights_are_fis_outputs():
    from bulanik.sets import linear_decreasing, linear_increasing

    a = LinguisticVariable("a", (0, 10), {"lo": linear_decreasing(0, 10), "hi": linear_increasing(0, 10)})
    b = LinguisticVariable("b", (0, 4), {"lo": linear_decreasing(0, 4), "hi": linear_increasing(0, 4)})
    out = SugenoOutput("y", {"p": Linear({"a": 1.0, "b": 3.0}, 0.0), "z": 0.0})
    fis = SugenoFIS([a, b], [out], ["IF a IS hi OR b IS hi THEN y IS p", "IF a IS lo AND b IS lo THEN y IS z"])
    ax = plot.surface(fis, resolution=3)
    (surf,) = tagged(ax, "surface")
    ga, gb = [0.0, 5.0, 10.0], [0.0, 2.0, 4.0]
    z = {(i, j): fis.evaluate({"a": ga[i], "b": gb[j]})["y"] for i in range(3) for j in range(3)}
    faces = sorted(
        (z[(i, j)] + z[(i + 1, j)] + z[(i, j + 1)] + z[(i + 1, j + 1)]) / 4 for i in range(2) for j in range(2)
    )
    assert sorted(surf.get_array()) == pytest.approx(faces)


def test_alpha_zero_cut_is_drawn_above_the_baseline_spine():
    cut0 = tagged(plot.alpha_cuts(TFN(1, 2, 4), levels=3), "alpha-cut")[0]
    assert cut0.get_ydata()[0] == 0 and not cut0.get_clip_on()


def test_neutrosophic_level_labels_right_of_truth_spoke():
    ax = plot.neutrosophic(NeutrosophicFN(0.7, 0.2, 0.1))
    levels = [t for t in ax.texts if t.get_text() in ("0.5", "1")]
    assert len(levels) == 2 and all(t.get_position()[0] > 0 and t.get_ha() == "left" for t in levels)


def test_ranking_and_weights_titles_and_labels():
    r = topsis(MATRIX, W, TYPES)
    ax = plot.ranking(r)
    assert ax.get_xlabel().startswith("TOPSIS score")
    assert len([t for t in ax.texts if t.get_text()]) == len(r.alternatives)  # a value per dot
    flat = vikor([[1, 1], [1, 1]], [0.5, 0.5], ["benefit", "benefit"])  # all tied
    assert "lower is better" in plot.ranking(flat).get_xlabel()
    from bulanik.mcdm.results import RankingResult

    tied = RankingResult("custom", ("a", "b"), {"a": 1.0, "b": 1.0}, ("a", "b"), {})
    assert "higher is better" in plot.ranking(tied).get_xlabel()
    assert plot.weights(ahp([[1, 3], [1 / 3, 1]])).get_title(loc="left") == "AHP weights"


def test_mcdm_display_names():
    from bulanik.mcdm import edas, entropy, fuzzy_ahp, swara, waspas
    from bulanik.mcdm.results import RankingResult, WeightingResult

    assert plot.weights(entropy([[1, 9], [2, 8], [3, 7]])).get_title(loc="left") == "Entropy weights"
    assert plot.weights(swara([0.3], ["a", "b"])).get_title(loc="left") == "SWARA weights"
    fz = fuzzy_ahp([[TFN(1, 1, 1), TFN(2, 3, 4)], [TFN(1 / 4, 1 / 3, 1 / 2), TFN(1, 1, 1)]])
    assert plot.weights(fz).get_title(loc="left") == "Fuzzy AHP weights"
    custom = WeightingResult("my_method", ("a", "b"), {"a": 0.6, "b": 0.4}, ("a", "b"), {})
    assert plot.weights(custom).get_title(loc="left") == "My Method weights"
    assert plot.ranking(waspas(MATRIX, W, TYPES)).get_xlabel().startswith("WASPAS score")
    assert plot.ranking(edas(MATRIX, W, TYPES)).get_xlabel().startswith("EDAS score")
    odd = RankingResult("fuzzy_topsis", ("a", "b"), {"a": 0.6, "b": 0.4}, ("a", "b"), {})
    assert plot.ranking(odd).get_xlabel().startswith("Fuzzy TOPSIS score")


def test_dark_warm_fills_are_lighter_and_surface_ramp_is_clamped():
    with plot.theme("dark"):
        ax = plot.membership([TFN(1, 2, 4), TFN(3, 4, 5)])
        blue, orange = tagged(ax, "fill")
        assert orange.get_alpha() < blue.get_alpha()
        assert matplotlib.colors.to_hex(plot.current_theme()["sequential"][0]) == "#1c5cab"
    assert plot.current_theme()["sequential"][0] == "#86b6ef"


def test_picture_has_edge_ticks():
    assert len(tagged(plot.picture(PictureFN(0.5, 0.2, 0.1)), "tick")) == 12


# -- v0.2: dematel, iv_orthopair, it2_number ------------------------------------------


def test_iv_orthopair_rectangles_match_bounds():
    ax = plot.iv_orthopair(IV_NUMS, labels=IV_NAMES)
    rects = tagged(ax, "interval")
    assert len(rects) == 3
    for r, n in zip(rects, IV_NUMS):
        bb = r.get_bbox()
        assert (bb.x0, bb.y0, bb.x1, bb.y1) == pytest.approx((n.mu_l, n.nu_l, n.mu_u, n.nu_u))
    assert len(tagged(ax, "boundary")) == 2  # rungs 1 and 2 of the numbers
    assert set(IV_NAMES) <= set(texts(ax))
    assert ax.get_aspect() == 1.0
    # distinct series colours, never the blue slot of the region ramp
    edges = [matplotlib.colors.to_hex(r.get_edgecolor()) for r in rects]
    assert len(set(edges)) == 3 and plot.PALETTE["light"][0] not in edges


def test_iv_orthopair_regions_and_degenerate_intervals():
    ax = plot.iv_orthopair([IVIFN(0.5, 0.5, 0.2, 0.3), IVIFN(0.4, 0.4, 0.3, 0.3)], regions=(1, 2, 3))
    assert len(tagged(ax, "boundary")) == 3
    line, point = tagged(ax, "interval")
    assert line.get_bbox().width == 0 and line.get_linewidth() > 0
    (dot,) = tagged(ax, "degenerate")  # the point-like interval gets a dot as heavy as an outline
    assert dot.get_sizes()[0] >= 60
    assert len(tagged(plot.iv_orthopair([], q=2), "boundary")) == 1


def test_iv_orthopair_labels_clear_of_other_rectangles():
    ax = plot.iv_orthopair(IV_NUMS, labels=IV_NAMES)
    centres = _label_centres(ax, IV_NAMES)
    rects = [r.get_window_extent() for r in tagged(ax, "interval")]
    for name, (_, _, bb) in centres.items():
        assert not any(bb.overlaps(r) for r in rects), name


def test_it2_number_polygons_and_fou():
    ax = plot.it2_number(IT2_NUMS, labels=["low", "medium"])
    uppers, lowers, fous = tagged(ax, "upper"), tagged(ax, "lower"), tagged(ax, "fou")
    assert len(uppers) == len(lowers) == len(fous) == 2
    for line, low, n in zip(uppers, lowers, IT2_NUMS):
        a1, a2, a3, a4, h1, h2 = n.upper
        xs, ys = line.get_data()
        assert list(xs) == pytest.approx([a1, a2, a3, a4]) and list(ys) == pytest.approx([0, h1, h2, 0])
        b1, b2, b3, b4, g1, g2 = n.lower
        xs, ys = low.get_data()
        assert list(xs) == pytest.approx([b1, b2, b3, b4]) and list(ys) == pytest.approx([0, g1, g2, 0])
        assert low.get_linewidth() < line.get_linewidth()
    colors = [matplotlib.colors.to_hex(u.get_color()) for u in uppers]
    assert colors == list(plot.PALETTE["light"][:2])
    assert {"low", "medium"} <= set(texts(ax))
    assert ax.get_ylim()[0] == 0 and ax.get_ylim()[1] >= 1


def test_it2_number_height_ticks_and_fallback():
    ax = plot.it2_number(IT2TrFN((1, 2, 3, 5, 0.8, 0.6), (2, 2.5, 2.5, 4, 0.4, 0.4)))
    assert list(ax.get_yticks()) == pytest.approx([0, 0.4, 0.6, 0.8, 1])
    many = [IT2TrFN((0, 1, 2, 3, h, h), (1, 1.5, 1.5, 2, h / 2, h / 2)) for h in (0.3, 0.5, 0.7, 0.9)]
    ax = plot.it2_number(many)  # too many heights to tick: back to 0, 0.5, 1
    assert list(ax.get_yticks()) == pytest.approx([0, 0.5, 1])


def test_dark_fills_are_lighter_than_light_fills():
    def alphas(mode):
        with plot.theme(mode):
            iv = plot.iv_orthopair(IV_NUMS)
            it2 = plot.it2_number(IT2_NUMS)
        rect = [matplotlib.colors.to_rgba(r.get_facecolor())[3] for r in tagged(iv, "interval-fill")]
        fou = [f.get_alpha() for f in tagged(it2, "fou")]
        return rect, fou

    (lr, lf), (dr, df) = alphas("light"), alphas("dark")
    assert all(d < lo for d, lo in zip(dr, lr)) and all(d < lo for d, lo in zip(df, lf))


def test_iv_orthopair_fills_sit_on_a_surface_knockout():
    ax = plot.iv_orthopair(IV_NUMS)
    knock = tagged(ax, "interval-knockout")
    assert len(knock) == len(tagged(ax, "interval-fill")) == len(tagged(ax, "interval")) == 3
    assert {matplotlib.colors.to_hex(k.get_facecolor()) for k in knock} == {"#ffffff"}
    zs = [max(a.get_zorder() for a in knock), min(a.get_zorder() for a in tagged(ax, "boundary"))]
    assert zs[0] < zs[1]  # the region boundaries stay visible through a rectangle


def test_it2_number_single_and_too_many():
    ax = plot.it2_number(IT2TrFN((1, 2, 3, 5, 0.8, 0.6), (2, 2.5, 2.5, 4, 0.4, 0.4)))
    assert len(tagged(ax, "upper")) == 1
    with pytest.raises(ValueError, match="at most 8"):
        plot.it2_number([IT2_NUMS[0]] * 9)


def _arrow_display_path(ann):
    """The display-space path of a relation arrow (an annotation's arrow patch)."""
    patch = ann.arrow_patch
    return patch.get_transform().transform_path(patch.get_path())


def _tip(ann):
    """Display position of the arrow's head (the annotated point, offset from the target centre)."""
    return tuple(ann.xycoords.transform(ann.xy))


def test_dematel_heads_spread_around_a_hub():
    r = dematel(DEMATEL_A, DEMATEL_F)
    ax = plot.dematel(r)
    ax.figure.draw_without_rendering()
    arrows = tagged(ax, "relation")
    for target in DEMATEL_F:
        cx, cy = ax.transData.transform((r.prominence[target], r.relation[target]))
        tips = [_tip(a) for a, (_, b, _) in zip(arrows, r.relations) if b == target]
        angles = sorted(math.degrees(math.atan2(y - cy, x - cx)) % 360 for x, y in tips)
        if len(angles) < 2:
            continue
        gaps = [b - a for a, b in zip(angles, angles[1:])] + [angles[0] + 360 - angles[-1]]
        assert min(gaps) >= 20 - 1e-6, (target, angles)  # the heads do not pile up
        for x, y in tips:  # every head stops on the marker's edge, not at its centre
            assert 5 < math.hypot(x - cx, y - cy) < 14


@pytest.mark.parametrize(
    "angles, gap_deg",
    [
        ([0.3], 48),  # one end: unchanged
        ([0.0, 0.0, 0.1], 48),  # a clump spreads around its mean
        ([6.2, 0.0, 0.05], 48),  # a clump across 0 / 2 pi
        ([0.0] * 10, 48),  # no room for the gap: evenly spaced
        ([0.17, 0.42, 5.0, 6.16, 2.71], 70),  # the spread would wrap round: evenly spaced
    ],
)
def test_arrow_end_spread_keeps_the_gap(angles, gap_deg):
    from bulanik.plot.mcdm import _spread

    gap = math.radians(gap_deg)
    out = _spread(angles, gap)
    assert len(out) == len(angles)
    if len(out) == 1:
        assert out == angles
        return
    ring = sorted(v % (2 * math.pi) for v in out)
    gaps = [b - a for a, b in zip(ring, ring[1:])] + [ring[0] + 2 * math.pi - ring[-1]]
    assert min(gaps) >= min(gap, 2 * math.pi / len(out)) - 1e-9


def test_dematel_arrow_weight_grows_with_strength():
    r = dematel(DEMATEL_A, DEMATEL_F)
    ax = plot.dematel(r)
    arrows = sorted(zip((v for *_, v in r.relations), tagged(ax, "relation")), key=lambda p: p[0])
    lws = [a.arrow_patch.get_linewidth() for _, a in arrows]
    alphas = [a.arrow_patch.get_alpha() for _, a in arrows]
    assert lws == sorted(lws) and alphas == sorted(alphas)
    assert lws[0] == pytest.approx(0.5) and lws[-1] == pytest.approx(2.0)
    assert alphas[0] == pytest.approx(0.35) and alphas[-1] == pytest.approx(0.9)


def test_dematel_points_arrows_and_groups():
    r = dematel(DEMATEL_A, DEMATEL_F)
    ax = plot.dematel(r)
    pts = tagged(ax, "factor")
    got = [tuple(float(v) for v in off) for p in pts for off in p.get_offsets()]
    want = [(r.prominence[f], r.relation[f]) for f in r.causes + r.effects]
    assert [v for p in sorted(got) for v in p] == pytest.approx([v for p in sorted(want) for v in p])
    colors = {matplotlib.colors.to_hex(p.get_facecolor()[0]) for p in pts}
    assert colors == set(plot.PALETTE["light"][:2])
    assert len(tagged(ax, "zero-line")) == 1
    assert len(tagged(ax, "relation")) == len(r.relations) > 0
    assert {"cause", "effect"} <= {s.split()[0] for s in texts(ax) if s}
    assert ax.get_ylim()[0] < 0 < ax.get_ylim()[1]
    centres = _label_centres(ax, DEMATEL_F)
    disp = [tuple(ax.transData.transform((r.prominence[f], r.relation[f]))) for f in DEMATEL_F]
    _nearest_is_own(centres, DEMATEL_F, disp)
    boxes = [bb for *_, bb in centres.values()]
    for i, a in enumerate(boxes):
        assert not any(a.overlaps(b) for b in boxes[i + 1:])
    for arrow in tagged(ax, "relation"):  # no label sits on an arrow (shaft or head)
        path = _arrow_display_path(arrow)
        for curve, _ in path.iter_bezier():
            for x, y in curve([0.0, 0.25, 0.5, 0.75, 1.0]):
                assert not any(bb.contains(x, y) for bb in boxes)
    assert tagged(plot.dematel(r, arrows=False), "relation") == []


def test_dematel_two_way_relations_curve_apart():
    r = dematel([[0, 3, 1], [3, 0, 1], [1, 1, 0]], ["a", "b", "c"], threshold=0.0)
    ax = plot.dematel(r)
    arrows = tagged(ax, "relation")
    assert len(arrows) == len(r.relations) == 6
    # the same arc3 bend on A->B and B->A puts the two arrows on opposite sides
    assert {a.arrow_patch.get_connectionstyle().rad for a in arrows} == {0.15}


def test_new_plots_reject_bad_input():
    with pytest.raises(TypeError, match="dematel needs a DEMATELResult"):
        plot.dematel(topsis(MATRIX, W, TYPES))
    with pytest.raises(TypeError, match="iv_orthopair"):
        plot.iv_orthopair([IFN(0.5, 0.3)])
    with pytest.raises(TypeError, match="it2_number"):
        plot.it2_number([TFN(1, 2, 3)])
    with pytest.raises(ValueError):
        plot.iv_orthopair(IV_NUMS, q=0)
    with pytest.raises(ValueError, match="q or regions"):
        plot.iv_orthopair(IV_NUMS, q=1, regions=(1,))
    with pytest.raises(ValueError):
        plot.iv_orthopair(IV_NUMS, regions=())
    with pytest.raises(ValueError):
        plot.iv_orthopair(IV_NUMS, labels=["a"])
    with pytest.raises(ValueError):
        plot.it2_number(IT2_NUMS, labels=["a"])
    with pytest.raises(ValueError, match="at least one"):
        plot.it2_number([])
    with pytest.raises(ValueError, match="at most 7"):
        plot.iv_orthopair([IV_NUMS[0]] * 8)
