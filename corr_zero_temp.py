"""Docstring
"""
from pySecDec.integral_interface import IntegralLibrary
import numpy as np
from sympy import expand, I
from pysecdec_output_tools import *
import spatial_integral as spint
from scipy.integrate import trapezoid

REL_ERROR = 1e-6
ABS_ERROR = 1e-9
MAX_ITER = 1000000

# load pySecDec libraries for the zero temperature spacetime integrals
spacetime_int_psd = IntegralLibrary('phiCubed_31/phiCubed_31_pylink.so')
spacetime_int_psd.use_Vegas(flags=0, epsrel=REL_ERROR, epsabs=ABS_ERROR,
                              maxeval=MAX_ITER)

def use_psd(p1, p2, m1, m2, m3):
    """
    Compute the zero temperature correlator using pySecDec.

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

    Returns
    -------
    sympy expression
        Zero temperature correlator.
    """
    p1p1 = spint.dot_product(p1, p1)
    p2p2 = spint.dot_product(p2, p2)
    p1p2 = spint.dot_product(p1, p2)
    result_str = spacetime_int_psd(complex_parameters=
                                   [p1p1, p2p2, p1p2, m1**2, m2**2, m3**2])[2]
    missing_prefactor = I  # missing from the generate file
    return expand(missing_prefactor*get_value(psd_to_sympy(result_str)))


def use_trap(p1, p2, m1, m2, m3, k0_eucl_max, num_grid_pts):
    """
    Compute the zero temperature correlator through iterated integrals.

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
    k0_eucl_max : real
        k0_eucl upper limit of integration
    num_grid_pts : integer
        Number of grid points to use in the trapezoid rule.

    Returns
    -------
    sympy expression
        Zero temperature correlator.
    """
    p1_0_eucl = -1j*p1[0]
    p2_0_eucl = -1j*p2[0]
    p1_space = p1[1:]
    p2_space = p2[1:]
    k0_eucl_grid = np.linspace(-k0_eucl_max, k0_eucl_max, num_grid_pts)
    spatial_int_vals = []
    for k0_eucl in k0_eucl_grid:
        M1M1 = m1**2 + (k0_eucl + p2_0_eucl)**2
        M2M2 = m2**2 + (k0_eucl - p1_0_eucl)**2
        M3M3 = m3**2 + k0_eucl**2
        spatial_int_vals.append(spint.use_psd(p1_space, p2_space, M1M1,
                                                 M2M2, M3M3))
    return expand(-I/(2*np.pi)*trapezoid(spatial_int_vals, k0_eucl_grid))


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
