#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute the three-point, one-loop function at zero and finite temperature.

A pySecDec integrate file for evaluating the one-loop three-point function at
zero and finite temperature.

@author: Derek Harnett
@email: derek.harnett@ufv.ca
"""

from __future__ import print_function
from pySecDec.integral_interface import IntegralLibrary
import numpy as np
from sympy import symbols, sympify, I, expand
from scipy.integrate import trapezoid

# declare sympy symbols, i.e., variables
value, uncertainty, eps = symbols('value uncertainty eps')
indeterminate = symbols('indeterminate')

REL_ERROR = 1e-6
ABS_ERROR = 1e-9
MAX_ITER = 1000000

# load pySecDec libraries for the zero temperature spacetime integrals
spacetime_int_psd = IntegralLibrary('phiCubed_31/phiCubed_31_pylink.so')
spacetime_int_psd.use_Vegas(flags=0, epsrel=REL_ERROR, epsabs=ABS_ERROR,
                              maxeval=MAX_ITER)

# load pySecDec libraries for the finite temperature spatial integrals
space_int_psd = IntegralLibrary('phiCubed_31_space/phiCubed_31_space_pylink.so')
space_int_psd.use_Vegas(flags=0, epsrel=REL_ERROR, epsabs=ABS_ERROR,
                      maxeval=MAX_ITER)

def psd_to_sympy(expr):
    """Convert a pySecDec string output to a sympy expression.

    Parameters
    ----------
    expr : string
        string output from pySecDec

    Returns
    -------
    sympy expression
        an expansion in the symbol eps with both value and uncertainty parts
    """
    # TODO: Check for NaN before converting.
    # What follows is a pretty dangerous hack.
    return sympify(expr.replace('nan', 'indeterminate').
                     replace(' +/- ', '*value+uncertainty*').
                     replace(',', '+I*'))


def get_value(psd_sympy_result):
    """Get the value part of a sympified pySecDec result.

    Parameters
    ----------
    psd_sympy_result : sympy expression
        sympified version of a pySecDec result

    Returns
    -------
    sympy expression
        sympified pySecDec result with value=1 and uncertainty=0
    """
    return psd_sympy_result.subs(value, 1).subs(uncertainty, 0)


def get_uncertainty(psd_sympy_result):
    """Get the uncertainty part of a sympified pySecDec result.

    Parameters
    ----------
    psd_sympy_result : sympy expression
        sympified version of a pySecDec result

    Returns
    -------
    sympy expression
        sympified pySecDec result with value=0 and uncertainty=1
    """
    return psd_sympy_result.subs(value, 0).subs(uncertainty, 1)


def dot_product(v1, v2):
    """Compute the Minkowski dot product of v1 and v2.

    v1 & v2 can be arbitrary integer dimension. The first component is assumed
    to be the timelike one.

    Parameters
    ----------
    v1 : complex tuple
        Minkowski vector.
    v2 : complex tuple
        Minkowski vector.

    Returns
    -------
    Complex.
        Dot product of v1 and v2.
    """
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)
    return v1_arr[0]*v2_arr[0] - sum(v1_arr[1:]*v2_arr[1:])


def spatial_integral(p1_space, p2_space, M1M1, M2M2, M3M3):
    """
    Compute the spatial integral.

    Parameters
    ----------
    p1_space : complex 3-tuple
        External 3-momentum.
    p2_space : complex 3-tuple
        External 3-momentum.
    M1M1 : complex
        Squared mass parameter.
    M2M2 : complex
        Squared mass parameter.
    M3M3 : complex
        Squared mass parameter.

    Returns
    -------
    sympy expression
        Spatial integral.
    """
    p1_space_mink = [1j*p1_space[0]] + p1_space[1:]
    p2_space_mink = [1j*p2_space[0]] + p2_space[1:]
    p1p1 = dot_product(p1_space_mink, p1_space_mink)
    p2p2 = dot_product(p2_space_mink, p2_space_mink)
    p1p2 = dot_product(p1_space_mink, p2_space_mink)
    missing_factor = -1
    space_int_str = space_int_psd(complex_parameters=
                               [p1p1, p2p2, p1p2, M1M1, M2M2, M3M3])[2]
    return expand(missing_factor*get_value(psd_to_sympy(space_int_str)))


def corr_zero_temp(p1, p2, m1, m2, m3, method="psd", k0_eucl_max=10,
                   num_grid_pts=21):
    """
    Compute the zero temperature correlator.

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
    method: "psd" or "iter"
        Computational method to use.
    k0_eucl_max : real
        k0 upper limit of integration
    num_grid_pts : integer
        Number of grid points to use in the trapezoid rule.

    Returns
    -------
    sympy expression
        Zero temperature correlator.
    """
    if method == "psd":
        return corr_zero_temp_psd(p1, p2, m1, m2, m3)
    elif method == "iter":
        return corr_zero_temp_iter(p1, p2, m1, m2, m3, k0_eucl_max,
                                   num_grid_pts)
    else:
        return  # error


def corr_zero_temp_psd(p1, p2, m1, m2, m3):
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
    p1p1 = dot_product(p1, p1)
    p2p2 = dot_product(p2, p2)
    p1p2 = dot_product(p1, p2)
    result_str = spacetime_int_psd(complex_parameters=
                                   [p1p1, p2p2, p1p2, m1**2, m2**2, m3**2])[2]
    missing_prefactor = I  # missing from the generate file
    return expand(missing_prefactor*get_value(psd_to_sympy(result_str)))


def corr_zero_temp_iter(p1, p2, m1, m2, m3, k0_eucl_max, num_grid_pts):
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
        spatial_int_vals.append(spatial_integral(p1_space, p2_space, M1M1,
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
    return expand(-I/beta*spatial_integral(p1_space, p2_space, M1M1, M2M2,
                                               M3M3))


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
        p1x = np.sqrt(np.abs(qq))*1j
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
    