#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the one-loop, three-point vertex function at the symmetric point.

@author: Derek Harnett
@email: derek.harnett@ufv.ca
"""

import integrate_phiCubed_31 as ip
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, I, expand


def zero_temp_vertex_sym(qq, m1, m2, m3, method='psd'):
    """ 
    Compute the zero temperature vertex function at the symmetric point.

    Parameters
    ----------
    qq : real
        Symmetric point.
    m1 : complex
        Propagator mass.
    m2 : complex
        Propagator mass.
    m3 : complex
        Propagator mass.

    Returns
    -------
    sympy expression
        Zero temperature vetex function.

    """
    momenta = ip.p1_p2(qq)
    p1 = momenta[0]
    p2 = momenta[1]
    Pi = ip.corr_zero_temp(p1, p2, m1, m2, m3, method)
    Gamma = expand(I*Pi) 
    return Gamma

    
# declare sympy symbols, i.e., variables
eps = symbols('eps')

qq_grid = np.linspace(-10, 10, 42)
m1, m2, m3 = 2, 2, 2

zero_temp_data = [zero_temp_vertex_sym(qq, m1, m2, m3, method='psd')
                  for qq in qq_grid]
zero_temp_re = np.array([complex(x.subs('eps', 0)).real
                         for x in zero_temp_data])
zero_temp_im = np.array([complex(x.subs('eps', 0)).imag
                         for x in zero_temp_data])

fig, ax = plt.subplots()
ax.plot(qq_grid, zero_temp_re, marker='.', color='blue', label='real')
ax.plot(qq_grid, zero_temp_im, marker='.', color='red', label='imag')
ax.set_xlabel(r'$q^2$')
ax.set_ylabel(r'$\Gamma(q^2) = \mathrm{i}\Pi(q^2)$')
ax.legend(loc='upper left')
