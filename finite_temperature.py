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


def correlator_term(p1, p2, mass_1, mass_2, mass_3, beta, n):
    """Compute the nth finite temperature spatial integral.

    Parameters
    ----------
    p1 : complex 4-tuple
        Minkowski external 4-momentum.
    p2 : complex 4-tuple
        Minkowski external 4-momentum.
    mass_1 : complex
        Propagator mass.
    mass_2 : complex
        Propagator mass.
    mass_3 : complex
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
    omega_n = omega(n, beta)
    Delta_1 = mass_1**2 + (omega_n + p2_0_eucl)**2
    Delta_2 = mass_2**2 + (omega_n - p1_0_eucl)**2
    Delta_3 = mass_3**2 + omega_n**2
    return expand(-I / beta * spint.use_psd(p1_space, p2_space, Delta_1, Delta_2, Delta_3))


def correlator_sequence(p1, p2, mass_1, mass_2, mass_3, beta, n_min, n_max):
    """Compute a sequence of finite temperature correlator terms.

    Parameters
    ----------
    p1 : complex 4-tuple
        Minkowski external 4-momentum.
    p2 : complex 4-tuple
        Minkowski external 4-momentum.
    mass_1 : complex
        Propagator mass.
    mass_2 : complex
        Propagator mass.
    mass_3 : complex
        Propagator mass.
    beta : real
        Inverse temperature.
    nmin : integer
        Minimum series index (included).
    nmax : integer
        Maximum series index (excluded).

    Returns
    -------
    list
        A sequence of finite temperature correlator terms.
    """
    return [correlator_term(p1, p2, mass_1, mass_2, mass_3, beta, n)
            for n in range(n_min, n_max)]


def correlator_partial_sum(p1, p2, mass_1, mass_2, mass_3, beta, n_min, n_max):
    """Compute the finite temperature correlator.

    Parameters
    ----------
    p1 : complex 4-tuple
        Minkowski external 4-momentum.
    p2 : complex 4-tuple
        Minkowski external 4-momentum.
    mass_1 : complex
        Propagator mass.
    mass_2 : complex
        Propagator mass.
    mass_3 : complex
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
    seq = correlator_sequence(p1, p2, mass_1, mass_2, mass_3, beta, n_min, n_max)
    return sum(seq)


def vertex_function_partial_sum(l1, q1, l2, q2, xi1, xi2, xi3, a, n_min, n_max, M):
    """Compute the dimensionless vertex function (Gamma)"""
    M_over_a = M / a
    p1t = 1j * l1 * M_over_a
    p1 = [x * M_over_a for x in q1]  # spatial components
    p1.insert(0, p1t)
    p2t = 1j * l2 * M_over_a
    p2 = [x * M_over_a for x in q2]  # spatial components
    p2.insert(0, p2t)
    m1 = M * xi1
    m2 = M * xi2
    m3 = M * xi3
    beta = 2 * np.pi / M_over_a
    Pi = correlator_partial_sum(p1, p2, m1, m2, m3, beta, n_min, n_max)
    return expand(M**2 * I * Pi)
