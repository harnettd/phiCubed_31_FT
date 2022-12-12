"""Docstring
"""
import numpy as np
from sympy import expand, I
import spatial_integral as spint

def omega(n, beta):
    """Compute a Matsubara frequency.

    Parameters
    ----------
    n : integer
        sequence index
    beta : real
        inverse temperature

    Returns
    -------
    real
        Matsubara frequency
    """
    return 2*np.pi*n/beta


def corr_finite_temp_n(p1, p2, m1, m2, m3, beta, n):
    """Compute the nth finite temperature spatial integral.

    Parameters
    ----------
    p1 : complex 4-tuple
        Minkowski external 4-momentum.
    p2 : complex 4-tuple
        Minkowski external 4-momentum.
    m1 : complex
        Propagator mass.
    m2 : complex
        Propagator mass.
    m3 : complex
        Propagator mass.
    beta : real
        Inverse temperature.
    n : integer
        Series index.

    Returns
    -------
    sympy expression
        The nth finite temperature spatial integral.
    """
    p1_0_eucl = -1j*p1[0]
    p2_0_eucl = -1j*p2[0]
    p1_space = p1[1:]
    p2_space = p2[1:]
    wn = omega(n, beta)
    M1M1 = m1**2 + (wn + p2_0_eucl)**2
    M2M2 = m2**2 + (wn - p1_0_eucl)**2
    M3M3 = m3**2 + wn**2
    return expand(-I/beta*spint.use_psd(p1_space, p2_space, M1M1, M2M2, M3M3))


def corr_finite_temp(p1, p2, m1, m2, m3, beta, nmin, nmax):
    """Compute the finite temperature correlator.

    Parameters
    ----------
    p1 : complex 4-tuple
        Minkowski external 4-momentum.
    p2 : complex 4-tuple
        Minkowski external 4-momentum.
    m1 : complex
        Propagator mass.
    m2 : complex
        Propagator mass.
    m3 : complex
        Propagator mass.
    beta : real
        Inverse temperature.
    nmin : integer
        Minimum series index (included).
    nmax : integer
        Maximum series index (excluded).

    Returns
    -------
    real
        The finite temperature correlator.
    """
    data = [corr_finite_temp_n(p1, p2, m1, m2, m3, beta, n)
            for n in range(nmin, nmax)]
    return sum(data)
