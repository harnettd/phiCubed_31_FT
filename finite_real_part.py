"""Extract the finite, real part of pySecDec results."""
import sympy as sp

eps = sp.symbols('eps')

def finite_real_part(expr):
    """Return the finite, real part of expr"""
    return complex(expr.subs(eps, 0)).real
