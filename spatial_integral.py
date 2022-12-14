"""Docstring
"""
from pySecDec.integral_interface import IntegralLibrary
from pysecdec_output_tools import *
import numpy as np
from sympy import expand
from scipy.integrate import tplquad

REL_ERROR = 1e-6
ABS_ERROR = 1e-9
MAX_ITER = 1000000

# load pySecDec libraries for the spatial integrals
spatial_int_psd = IntegralLibrary('phiCubed_31_space/phiCubed_31_space_pylink.so')
spatial_int_psd.use_Vegas(flags=0, epsrel=REL_ERROR, epsabs=ABS_ERROR,
                      maxeval=MAX_ITER)


def dot_product(vector_1, vector_2):
    """Compute the Minkowski dot product of v1 and v2.

    v1 and v2 should be array_like objects with equal positive-integer
    dimensions and with complex elements.
    The first component is assumed to be the timelike one.

    Parameters
    ----------
    v1 : array_like, complex
        Minkowski vector.
    v2 : array_like, complex
        Minkowski vector.

    Returns
    -------
    Complex.
        Dot product of v1 and v2.
    """
    v1_arr = np.array(vector_1)
    v2_arr = np.array(vector_2)
    if v1_arr.shape != v2_arr.shape:
        print('Error in dot_product: v1 and v2 are incompatible')
        return None
    if v1_arr.ndim == 0:
        return v1_arr*v2_arr
    elif v1_arr.ndim == 1:
        return v1_arr[0]*v2_arr[0] - sum(v1_arr[1:]*v2_arr[1:])
    else:
        print('Error in dot_product: v1 and/or v2 not rank-1 objects')
        return None


def use_psd(p1_space, p2_space, Delta_1, Delta_2, Delta_3):
    """
    Compute the spatial integral.

    Parameters
    ----------
    p1_space : array_like, rank-0 or rank-1, complex
        External (Euclidean) spatial momentum.
    p2_space : array_like, rank-0 or rank-1, complex
        External (Euclidean) spatial momentum.
    Delta_1 : complex
        Squared mass parameter.
    Delta_2 : complex
        Squared mass parameter.
    Delta_3 : complex
        Squared mass parameter.

    Returns
    -------
    sympy expression
        Spatial integral.
    """
    # TODO: Should I test p1_space and p2_space before the next two lines?
    p1_space_eucl = np.array(p1_space, dtype=complex)
    p2_space_eucl = np.array(p2_space, dtype=complex)

    if p1_space_eucl.ndim != 1:
        print('Error in spatial_integral: p1_space is not rank 1')
        return None
    if p1_space_eucl.size != 3:
        print('Error in spatial_integral: p1_space does not have 3 elements')
        return None
    if p2_space_eucl.ndim != 1:
        print('Error in spatial_integral: p2_space is not rank 1')
        return None
    if p2_space_eucl.size != 3:
        print('Error in spatial_integral: p2_space does not have 3 elements')
        return None

    p1_space_mink = np.insert(p1_space_eucl[1:], 0, p1_space_eucl[0]*1j)
    p2_space_mink = np.insert(p2_space_eucl[1:], 0, p2_space_eucl[0]*1j)
    p1p1 = dot_product(p1_space_mink, p1_space_mink)
    p2p2 = dot_product(p2_space_mink, p2_space_mink)
    p1p2 = dot_product(p1_space_mink, p2_space_mink)
    missing_factor = -1  # factor omitted from pySecDec generate file
    spatial_int_str = spatial_int_psd(complex_parameters=[p1p1, p2p2, p1p2, Delta_1, Delta_2, Delta_3])[2]
    return expand(missing_factor*get_value(psd_to_sympy(spatial_int_str)))


def use_tplquad(p1_space, p2_space, Delta_1, Delta_2, Delta_3):
    """
    Compute the spatial integral.

    Parameters
    ----------
    p1_space : array_like, rank-0 or rank-1, complex
        External (Euclidean) spatial momentum.
    p2_space : array_like, rank-0 or rank-1, complex
        External (Euclidean) spatial momentum.
    Delta_1 : complex
        Squared mass parameter.
    Delta_2 : complex
        Squared mass parameter.
    Delta_3 : complex
        Squared mass parameter.

    Returns
    -------
    sympy expression
        Spatial integral.
    """
    # TODO: Should I test p1_space and p2_space before the next two lines?
    p1_space_eucl = np.array(p1_space, dtype=float)
    p2_space_eucl = np.array(p2_space, dtype=float)

    if p1_space_eucl.ndim != 1:
        print('Error in spatial_integral: p1_space is not rank 1')
        return None
    if p1_space_eucl.size != 3:
        print('Error in spatial_integral: p1_space does not have 3 elements')
        return None
    if p2_space_eucl.ndim != 1:
        print('Error in spatial_integral: p2_space is not rank 1')
        return None
    if p2_space_eucl.size != 3:
        print('Error in spatial_integral: p2_space does not have 3 elements')
        return None

    p1x, p1y, p1z = p1_space_eucl
    p2x, p2y, p2z = p2_space_eucl
    def integrand(kx, ky, kz):
        denom_factor_1 = kx**2 + ky**2 + kz**2 + Delta_3
        denom_factor_2 = (kx + p2x)**2 + (ky + p2y)**2 + (kz + p2z)**2 + Delta_1
        denom_factor_3 = (kx - p1x)**2 + (ky - p1y)**2 + (kz - p1z)**2 + Delta_2
        return 1/(8*np.pi**3 * denom_factor_1 * denom_factor_2 * denom_factor_3)

    result = tplquad(integrand, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)
    return result
