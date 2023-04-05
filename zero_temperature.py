"""Compute the zero temperature correlator.

Functions:
    correlator_use_psd(array-like, array-like, complex, complex, complex)
        -> sympy object

    correlator_use_trapezoid(array-like, array-like, complex, complex, complex, array)
        -> sympy object

    dimensionless_vertex_funxtion(sequence, sequence, sequence, float) 
        -> sympy object
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
MAX_ITER = 1_000_000

# load pySecDec libraries for the zero temperature spacetime integrals
spacetime_int_psd = IntegralLibrary('phiCubed_31/phiCubed_31_pylink.so')
spacetime_int_psd.use_Vegas(flags=0, epsrel=REL_ERROR, epsabs=ABS_ERROR, maxeval=MAX_ITER)


def correlator_use_psd(p1, p2, mass_1, mass_2, mass_3):
    """Compute the zero temperature correlator using pySecDec.

    Parameters:
        p1 (rank-one with four complex elements): First Minkowski external
            four-momentum
        p2 (rank-one with four complex elements): Second Minkowski external
            four-momentum
        mass_1 (complex): First propagator mass
        mass_2 (complex): Second propagator mass
        mass_3 (complex): Third propagator mass

    Returns:
        sympy object: Zero temperature correlator as a Laurent series
    """
    p1p1 = dot_product(p1, p1)
    p2p2 = dot_product(p2, p2)
    p1p2 = dot_product(p1, p2)

    psd_str_result = spacetime_int_psd(
            complex_parameters=[p1p1, p2p2, p1p2, mass_1**2, mass_2**2, mass_3**2]
        )[2]

    missing_prefactor = I  # missing from the generate file
    return expand(missing_prefactor * get_value(psd_to_sympy(psd_str_result)))


def correlator_use_trapezoid(p1, p2, mass_1, mass_2, mass_3, k0_eucl_grid):
    """Compute the zero temperature correlator through iterated integrals.

    Parameters:
        p1 (rank-one with four complex elements): First Minkowski external
            four-momentum
        p2 (rank-one with four complex elements): Second Minkowski external
            four-momentum
        mass_1 (complex): First propagator mass
        mass_2 (complex): Second propagator mass
        mass_3 (complex): Third propagator mass
        k0_eucl_grid (array): Grid to be used in the trapezoid rule

    Returns:
        sympy object: Zero temperature correlator
    """
    p1_0_eucl = -1j * p1[0]
    p2_0_eucl = -1j * p2[0]
    p1_space = p1[1:]
    p2_space = p2[1:]

    spatial_integral_data = []
    for k0_eucl in k0_eucl_grid:
        delta_1 = mass_1**2 + (k0_eucl + p2_0_eucl)**2
        delta_2 = mass_2**2 + (k0_eucl - p1_0_eucl)**2
        delta_3 = mass_3**2 + k0_eucl**2
        spatial_integral_data.append(spint.use_psd(p1_space, p2_space, delta_1, delta_2, delta_3))

    return expand(-I / (2 * np.pi) * trapezoid(spatial_integral_data, k0_eucl_grid))


def dimensionless_correlator_use_psd(q1, q2, xis, M):
    """Compute the dimensionless zero temperature correlator using pySecDec.

    Parameters:
        q1 (four-element complex sequence): First dimensionless Minkowski
            external momentum
        q2 (four-element complex sequence): Second dimensionless Minkowski
            external momentum
        xis (three-element positive real sequence): Dimensionless propagator
            masses
        M (real): Mass scale, i.e., the largest propagator mass

    One element of xis should be 1. The other two should be less than or equal
        to 1.

    Returns:
        sympy object: Dimensionless zero temperature correlator
    """
    def scale_up(seq, scale):
        """Multiply each element of the sequence seq by scale."""
        return [s * scale for s in seq]


    p1 = scale_up(q1, M)
    p2 = scale_up(q2, M)
    masses = scale_up(xis, M)

    return expand(correlator_use_psd(p1, p2, *masses) * M**2)
