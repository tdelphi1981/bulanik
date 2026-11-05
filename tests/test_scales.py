import pytest
from pytest import approx

from bulanik import IFN, TFN
from bulanik.scales import (
    EXAMPLE_IFN_5,
    EXAMPLE_IFN_7,
    EXAMPLE_TFN_7,
    SAATY_TFN,
    LinguisticScale,
)


def test_saaty():
    assert SAATY_TFN["3"] == TFN(2, 3, 4) and SAATY_TFN["1"] == TFN(1, 1, 1)
    assert SAATY_TFN["9"] == TFN(9, 9, 9)
    assert SAATY_TFN.reciprocal("3") == TFN(0.25, 1 / 3, 0.5)  # 1/(4,3,2)
    assert "Saaty 1980" in SAATY_TFN.source
    assert SAATY_TFN.terms == tuple("123456789")
    assert SAATY_TFN.number_type is TFN


def test_example_scales_valid_and_unattributed():
    for s in (EXAMPLE_IFN_5, EXAMPLE_IFN_7, EXAMPLE_TFN_7):
        assert s.source is None
    assert len(EXAMPLE_IFN_5.terms) == 5 and len(EXAMPLE_IFN_7.terms) == 7
    assert len(EXAMPLE_TFN_7.terms) == 7
    for s in (EXAMPLE_IFN_5, EXAMPLE_IFN_7, EXAMPLE_TFN_7):
        scores = [s[t].score() for t in s.terms]
        assert scores == sorted(scores)
    assert EXAMPLE_IFN_7["H"] == IFN(0.7, 0.25) and EXAMPLE_IFN_5["H"] == IFN(0.7, 0.2)
    assert EXAMPLE_TFN_7["G"] == TFN(7, 9, 10)
    for doc in (EXAMPLE_IFN_5.name, EXAMPLE_IFN_7.name, EXAMPLE_TFN_7.name):
        assert "example" in doc.lower()


def test_convert_and_errors():
    assert EXAMPLE_IFN_5.convert([["H", "M"], ["VL"]]) == [[IFN(0.7, 0.2), IFN(0.5, 0.4)], [IFN(0.1, 0.8)]]
    assert EXAMPLE_IFN_5.convert("M") == IFN(0.5, 0.4)
    with pytest.raises(KeyError, match="VH"):
        EXAMPLE_IFN_5["XL"]
    with pytest.raises(KeyError):
        EXAMPLE_IFN_5.convert([["H", "nope"]])
    with pytest.raises(TypeError):
        EXAMPLE_IFN_5.convert([3])
    with pytest.raises(TypeError):
        LinguisticScale("mix", {"a": TFN(1, 2, 3), "b": IFN(0.5, 0.2)})
    with pytest.raises(ValueError):
        LinguisticScale("empty", {})
    with pytest.raises(TypeError):
        EXAMPLE_IFN_5.reciprocal("H")
    assert "H" in EXAMPLE_IFN_5 and "XL" not in EXAMPLE_IFN_5


def test_scale_copies_terms():
    src = {"a": TFN(1, 2, 3)}
    s = LinguisticScale("s", src)
    src["b"] = TFN(2, 3, 4)
    assert s.terms == ("a",)


def test_repr():
    assert "Saaty" in repr(SAATY_TFN)


def test_scale_type_checks_terms():
    with pytest.raises(TypeError, match="term 'x' of scale 's'.*fuzzy number"):
        LinguisticScale("s", {"a": TFN(1, 2, 3), "x": 2.0})
    with pytest.raises(TypeError, match="term 'x'"):
        LinguisticScale("s", {"x": "M"})
