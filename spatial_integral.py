"""Compute the 3d spatial integral.

Functions:
    use_psd(sequence, sequence, sequence) -> sympy object, sympy object
    use_tplquad(sequence, sequence, sequence) -> complex, complex
"""
import numpy as np
from scipy.integrate import tplquad
from sympy import expand

from pySecDec.integral_interface import IntegralLibrary

from dot_product import dot_product
from pysecdec_output_tools import get_value, get_uncertainty, psd_to_sympy

REL_ERROR = 1e-3
ABS_ERROR = 1e-8
MAX_ITER = 5_000_000

# load pySecDec libraries for the spatial integrals
spatial_int_psd =\
    IntegralLibrary('phiCubed_31_space/phiCubed_31_space_pylink.so')

spatial_int_psd.use_Vegas(flags=0, epsrel=REL_ERROR, epsabs=ABS_ERROR,
                      maxeval=MAX_ITER)


def use_psd(p1_space, p2_space, deltas):
    """Compute the 3d spatial integral using pySecDec.

    Parameters:
        p1_space (three-element sequence of complex): First external
            (Euclidean) spatial momentum
        p2_space (three-element sequence of complex): Second external
            (Euclidean) spatial momentum
        deltas (three-element sequence of complex): Squared mass parameters

    Returns:
        sympy object: The spatial integral value as a Laurent series
        sympy object: The spatial integral uncertainty as a Laurent series
    """
    p1_space_eucl = np.array(p1_space, dtype=complex)
    p2_space_eucl = np.array(p2_space, dtype=complex)

    # reverse Wick rotate the spatial momenta
    p1_space_mink = np.insert(p1_space_eucl[1:], 0, p1_space_eucl[0] * 1j)
    p2_space_mink = np.insert(p2_space_eucl[1:], 0, p2_space_eucl[0] * 1j)

    p1p1 = dot_product(p1_space_mink, p1_space_mink)
    p2p2 = dot_product(p2_space_mink, p2_space_mink)
    p1p2 = dot_product(p1_space_mink, p2_space_mink)
    dot_products = p1p1, p2p2, p1p2

    spatial_int_str =\
        spatial_int_psd(complex_parameters=[*dot_products, *deltas])[2]
    spatial_int_sympy = psd_to_sympy(spatial_int_str)

    missing_factor = -1  # factor omitted from the pySecDec generate file
    value = expand(missing_factor * get_value(spatial_int_sympy))
    # missing_factor does not affect uncertainty here
    uncertainty = expand(get_uncertainty(spatial_int_sympy))

    return value, uncertainty


def use_tplquad(p1_space, p2_space, deltas):
    """Compute the 3d spatial integral using tplquad from scipy.integrate.

    Parameters:
        p1_space (three-element sequence of complex): First external
            (Euclidean) spatial momentum
        p2_space (three-element sequence of complex): Second external
            (Euclidean) spatial momentum
        deltas (three-element sequence of complex): Squared mass parameters

    Returns:
        complex: The value of the integral
        complex: The uncertainty in the value of the integral
    """
    p1_space_eucl = np.array(p1_space, dtype=complex)
    p2_space_eucl = np.array(p2_space, dtype=complex)

    p1x, p1y, p1z = p1_space_eucl
    p2x, p2y, p2z = p2_space_eucl


    def integrand(kx, ky, kz):
        """Compute the integral's integrand."""
        denom_factor_1 = kx**2 + ky**2 + kz**2 + deltas[2]
        denom_factor_2 = (kx + p2x)**2 + (ky + p2y)**2 + (kz + p2z)**2 + deltas[0]
        denom_factor_3 = (kx - p1x)**2 + (ky - p1y)**2 + (kz - p1z)**2 + deltas[1]
        return 1 / (8 * np.pi**3 * denom_factor_1 * denom_factor_2 * denom_factor_3)


    def integrand_real(kx, ky, kz):
        """Compute the real part of the integral's integrand."""
        return (integrand(kx, ky, kz)).real


    def integrand_imag(kx, ky, kz):
        """Compute the imaginary part of the integral's integrand."""
        return (integrand(kx, ky, kz)).imag


    result_real = tplquad(integrand_real, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)
    result_imag = tplquad(integrand_imag, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)
    return result_real[0] + 1j * result_imag[0], result_real[1] + 1j * result_imag[1]
