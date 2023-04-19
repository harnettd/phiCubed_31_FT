"""Compute the zero temperature correlator.

Functions:
    correlator_use_psd(sequence, sequence, sequence)
        -> sympy object, sympy object

    correlator_use_trapezoid(sequence, sequence, sequence, array)
        -> sympy object

    dimensionless_vertex_funxtion(sequence, sequence, sequence, float)
        -> sympy object, sympy object
"""
import numpy as np
from scipy.integrate import trapezoid
from sympy import expand, I, symbols

from pySecDec.integral_interface import IntegralLibrary

from dot_product import dot_product
from mult import mult
from pysecdec_output_tools import get_uncertainty, get_value, psd_to_sympy
import spatial_integral as spint
from wick_rotation import to_euclidean

REL_ERROR = 1e-6
ABS_ERROR = 1e-9
MAX_ITER = 1_000_000

# load pySecDec libraries for the zero temperature spacetime integrals
spacetime_int_psd = IntegralLibrary('phiCubed_31/phiCubed_31_pylink.so')
spacetime_int_psd.use_Vegas(
    flags=0, epsrel=REL_ERROR, epsabs=ABS_ERROR, maxeval=MAX_ITER)

eps = symbols('eps')


def correlator_use_psd(p1, p2, masses):
    """Compute the zero temperature correlator using pySecDec.

    Parameters:
        p1 (four-element sequence of complex): First Minkowski external
            four-momentum
        p2 (four-element sequence of complex): Second Minkowski external
            four-momentum
        masses (three-element sequence of float): Propagator masses

    Returns:
        sympy object: Zero temperature correlator value
        sympy object: Zero temperature correlator uncertainty
    """
    dot_products =\
        dot_product(p1, p1), dot_product(p2, p2), dot_product(p1, p2)
    squared_masses = [mass**2 for mass in masses]
    psd_str_results =\
        spacetime_int_psd(complex_parameters=[*dot_products, *squared_masses])
    psd_str_result = psd_str_results[2]

    psd_sympy_result = psd_to_sympy(psd_str_result)
    psd_sympy_val = get_value(psd_sympy_result)
    psd_sympy_err = get_uncertainty(psd_sympy_result)

    missing_factor = 1j  # missing factor from the pySecDec generate file
    return mult(missing_factor, [psd_sympy_val, psd_sympy_err])


def correlator_use_trapezoid(p1, p2, masses, k0_eucl_grid):
    """Compute the zero temperature correlator through iterated integrals.

    Parameters:
        p1 (four-element sequence of complex): First Minkowski external
            four-momentum
        p2 (four-element sequence of complex): Second Minkowski external
            four-momentum
        masses (three-element sequence of float): Propagator masses
        k0_eucl_grid (array): Grid to be used in the trapezoid rule

    Returns:
        sympy object: Zero temperature correlator
    """
    p1_eucl = to_euclidean(p1)
    p2_eucl = to_euclidean(p2)
    p1_t_eucl = p1_eucl[0]
    p2_t_eucl = p2_eucl[0]
    p1_space = p1[1:]
    p2_space = p2[1:]

    spatial_integral_data = []
    for k0_eucl in k0_eucl_grid:
        delta_1 = masses[0]**2 + (k0_eucl + p2_t_eucl)**2
        delta_2 = masses[1]**2 + (k0_eucl - p1_t_eucl)**2
        delta_3 = masses[2]**2 + k0_eucl**2
        deltas = delta_1, delta_2, delta_3
        spatial_integral_data.append(
            spint.use_psd(p1_space, p2_space, deltas)[0])

    return expand(-I / (2 * np.pi) *
                  trapezoid(spatial_integral_data, k0_eucl_grid))


def dimensionless_correlator_use_psd(q1, q2, xis, M):
    """Compute the dimensionless zero temperature correlator using pySecDec.

    Parameters:
        q1 (four-element sequence of complex): First dimensionless Minkowski
            external momentum
        q2 (four-element sequence of complex): Second dimensionless Minkowski
            external momentum
        xis (three-element sequence of float): Dimensionless propagator masses
        M (real): Mass scale, i.e., the largest propagator mass

    One element of xis should be 1. The other two should be less than or equal
        to 1.

    Returns:
        sympy object: Dimensionless zero temperature correlator value
        sympy object: Dimensionless zero temperature correlator uncertainty
    """
    def scale_up(seq, scale):
        """Multiply each element of the sequence seq by scale."""
        return [s * scale for s in seq]

    p1 = scale_up(q1, M)
    p2 = scale_up(q2, M)
    masses = scale_up(xis, M)

    correlator_result = correlator_use_psd(p1, p2, masses)
    value = expand(correlator_result[0] * M**2)
    uncertainty = expand(correlator_result[1] * M**2)

    return value, uncertainty
