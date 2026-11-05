"""Hypothesis strategies shared by the test modules."""

from hypothesis import strategies as st

from bulanik import TFN, TrFN


def _floats(positive: bool):
    if positive:
        return st.floats(0.1, 100, allow_nan=False, allow_infinity=False)
    return st.floats(-100, 100, allow_nan=False, allow_infinity=False)


@st.composite
def tfns(draw, positive: bool = False):
    l, m, u = sorted(draw(st.lists(_floats(positive), min_size=3, max_size=3)))
    return TFN(l, m, u)


@st.composite
def trfns(draw, positive: bool = False):
    a, b, c, d = sorted(draw(st.lists(_floats(positive), min_size=4, max_size=4)))
    return TrFN(a, b, c, d)


@st.composite
def qrofns(draw, q: int):
    from bulanik import QROFN

    mu = draw(st.floats(0, 1, allow_nan=False))
    cap = max(0.0, 1.0 - mu**q) ** (1.0 / q)
    nu = draw(st.floats(0, 1, allow_nan=False)) * cap
    return QROFN(mu, nu, q)


def _no_underflow(x: float) -> float:
    """Squares of parts below 1e-100 would underflow; treat them as 0."""
    return x if x > 1e-100 else 0.0


def _fractions():
    """Fractions of the remaining room: interior, tiny, none, and all of it."""
    return st.one_of(
        st.floats(0, 1, allow_nan=False).map(_no_underflow),
        st.floats(0, 1e-6, allow_nan=False).map(_no_underflow),
        st.sampled_from([0.0, 1.0, 1.0, 1e-9, 0.5]),
    )


@st.composite
def sphericals(draw):
    """Spherical numbers, many of them on or within rounding of the boundary."""
    import math

    from bulanik import SphericalFN

    mu = draw(_fractions())
    nu = draw(_fractions()) * math.sqrt(max(0.0, 1.0 - mu * mu))
    pi = draw(_fractions()) * math.sqrt(max(0.0, 1.0 - mu * mu - nu * nu))
    a, b, c = draw(st.permutations([mu, nu, pi]))
    return SphericalFN(a, b, c)


@st.composite
def pictures(draw):
    """Picture numbers, many of them on or within rounding of mu+eta+nu = 1."""
    from bulanik import PictureFN

    mu = draw(_fractions())
    eta = draw(_fractions()) * (1.0 - mu)
    nu = draw(_fractions()) * max(0.0, 1.0 - mu - eta)
    a, b, c = draw(st.permutations([mu, eta, nu]))
    return PictureFN(a, b, c)


@st.composite
def neutrosophics(draw):
    from bulanik import NeutrosophicFN

    part = st.one_of(st.floats(0, 1, allow_nan=False), st.sampled_from([0.0, 1.0, 1e-9]))
    return NeutrosophicFN(draw(part), draw(part), draw(part))


@st.composite
def ivqrofns(draw, q: int):
    """Interval-valued q-rung numbers: an upper pair like ``qrofns``, lower bounds as
    (boundary-heavy) fractions of the upper ones."""
    from bulanik import IVQROFN

    mu_u = draw(st.floats(0, 1, allow_nan=False).map(_no_underflow))
    cap = max(0.0, 1.0 - mu_u**q) ** (1.0 / q)
    nu_u = draw(_fractions()) * cap
    mu_l = draw(_fractions()) * mu_u
    nu_l = draw(_fractions()) * nu_u
    return IVQROFN(mu_l, mu_u, nu_l, nu_u, q)


@st.composite
def ivneutrosophics(draw):
    """Interval neutrosophic numbers; each interval is two sorted parts (0, 1, 1e-9 included)."""
    from bulanik import IVNeutrosophicFN

    part = st.one_of(st.floats(0, 1, allow_nan=False), st.sampled_from([0.0, 1.0, 1e-9]))
    pairs = [sorted((draw(part), draw(part))) for _ in range(3)]
    return IVNeutrosophicFN(*(x for p in pairs for x in p))


@st.composite
def it2trfns(draw, positive: bool = False):
    """IT2 trapezoids whose lower trapezoid is nested in the upper one.

    The lower core lies inside the upper core, the lower support inside the upper
    support and the lower heights are at most min(h1U, h2U); that guarantees the
    lower membership stays below the upper one.
    """
    from bulanik import IT2TrFN

    a1, a2, a3, a4 = sorted(draw(st.lists(_floats(positive), min_size=4, max_size=4)))
    h = st.one_of(st.floats(0.1, 1, allow_nan=False), st.sampled_from([0.1, 1.0]))
    h1, h2 = draw(h), draw(h)
    cap = min(h1, h2)
    frac = st.floats(0, 1, allow_nan=False)

    def between(lo, hi):
        return min(max(lo + draw(frac) * (hi - lo), lo), hi)

    b2 = between(a2, a3)
    b3 = between(b2, a3)
    b1 = between(a1, b2)
    b4 = between(b3, a4)
    g1, g2 = draw(frac) * cap, draw(frac) * cap
    return IT2TrFN((a1, a2, a3, a4, h1, h2), (b1, b2, b3, b4, g1, g2))
