"""Docstring
"""
import numpy as np
from scipy.integrate import trapezoid
from sympy import expand, I

from pySecDec.integral_interface import IntegralLibrary

from dot_product import dot_product
from pysecdec_output_tools import get_value, psd_to_sympy
import spatial_integral as spint

REL_ERROR = 1e-6
ABS_ERROR = 1e-9
MAX_ITER = 1000000

# load pySecDec libraries for the zero temperature spacetime integrals
spacetime_int_psd = IntegralLibrary('phiCubed_31/phiCubed_31_pylink.so')
spacetime_int_psd.use_Vegas(flags=0, epsrel=REL_ERROR, epsabs=ABS_ERROR, maxeval=MAX_ITER)


def correlator_use_psd(p1, p2, mass_1, mass_2, mass_3):
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
    p1p1 = dot_product(p1, p1)
    p2p2 = dot_product(p2, p2)
    p1p2 = dot_product(p1, p2)
    psd_str_result = spacetime_int_psd(
        complex_parameters=[p1p1, p2p2, p1p2, mass_1**2, mass_2**2, mass_3**2])[2]
    missing_prefactor = I  # missing from the generate file
    return expand(missing_prefactor*get_value(psd_to_sympy(psd_str_result)))


def correlator_use_trapezoid(p1, p2, mass_1, mass_2, mass_3, k0_eucl_grid):
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


def dimensionless_vertex_function(q1, q2, xi1, xi2, xi3, M):
    p1 = [x * M for x in q1[1:]]
    p1.insert(0, 1j * q1[0])
    p2 = [x * M for x in q2[1:]]
    p2.insert(0, 1j * q2[0])
    m1 = xi1 * M
    m2 = xi2 * M
    m3 = xi3 * M
    Sigma = correlator_use_psd(p1, p2, m1, m2, m3)
    return expand(I * Sigma * M**2)
