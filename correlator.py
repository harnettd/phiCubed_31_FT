"""Docstring
"""
import numpy as np
from scipy.integrate import trapezoid
from sympy import expand, I

from pySecDec.integral_interface import IntegralLibrary

from pysecdec_output_tools import *
import spatial_integral as spint

REL_ERROR = 1e-6
ABS_ERROR = 1e-9
MAX_ITER = 1000000

# load pySecDec libraries for the zero temperature spacetime integrals
spacetime_int_psd = IntegralLibrary('phiCubed_31/phiCubed_31_pylink.so')
spacetime_int_psd.use_Vegas(flags=0, epsrel=REL_ERROR, epsabs=ABS_ERROR, maxeval=MAX_ITER)

def zero_temp_use_psd(p1, p2, mass_1, mass_2, mass_3):
    """Compute the zero temperature correlator using pySecDec.

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
    psd_str_result = spacetime_int_psd(complex_parameters=[p1p1, p2p2, p1p2, mass_1**2, mass_2**2, mass_3**2])[2]
    missing_prefactor = I  # missing from the generate file
    return expand(missing_prefactor*get_value(psd_to_sympy(psd_str_result)))


def zero_temp_use_trap(p1, p2, mass_1, mass_2, mass_3, k0_eucl_grid):
    """Compute the zero temperature correlator through iterated integrals.

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
    k0_eucl_grid : array-like
        grid to be used in the trapezoid rule

    Returns
    -------
    sympy expression
        Zero temperature correlator.
    """
    p1_0_eucl = -1j*p1[0]
    p2_0_eucl = -1j*p2[0]
    p1_space = p1[1:]
    p2_space = p2[1:]
    spatial_integral_data = []
    for k0_eucl in k0_eucl_grid:
        Delta_1 = mass_1**2 + (k0_eucl + p2_0_eucl)**2
        Delta_2 = mass_2**2 + (k0_eucl - p1_0_eucl)**2
        Delta_3 = mass_3**2 + k0_eucl**2
        spatial_integral_data.append(spint.use_psd(p1_space, p2_space, Delta_1, Delta_2, Delta_3))
    return expand(-I/(2*np.pi)*trapezoid(spatial_integral_data, k0_eucl_grid))


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


def finite_temp_term(p1, p2, mass_1, mass_2, mass_3, beta, n):
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
    omega_n = omega(n, beta)
    Delta_1 = mass_1**2 + (omega_n + p2_0_eucl)**2
    Delta_2 = mass_2**2 + (omega_n - p1_0_eucl)**2
    Delta_3 = mass_3**2 + omega_n**2
    return expand(-I/beta*spint.use_psd(p1_space, p2_space, Delta_1, Delta_2, Delta_3))


def finite_temp(p1, p2, mass_1, mass_2, mass_3, beta, n_min, n_max):
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
    data = [finite_temp_term(p1, p2, mass_1, mass_2, mass_3, beta, n) for n in range(n_min, n_max)]
    return sum(data)


def p1_p2(qq):
    """Compute 4-vectors corresponding to the symmetric point qq.

    Parameters
    ----------
    qq : real
        Symmetric point value.

    Returns
    -------
    p1 : 4d complex array
        4-momentum.
    p2 : 4d complex array
        4-momentum.
    """
    if qq >= 0:
        p1x = np.sqrt(qq)*1j
        p2t = 0.5*np.sqrt(3*qq)
    elif qq < 0:
        p1x = np.sqrt(np.abs(qq))
        p2t = 0.5*np.sqrt(3*np.abs(qq))*1j
    else:
        pass  # error
    p1t = 0
    p2x = -0.5*p1x
    p1 = np.array([p1t, p1x, 0, 0])
    p2 = np.array([p2t, p2x, 0, 0])
    return p1, p2