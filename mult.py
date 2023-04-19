"""Multiply a complex number and a truncated Laurent series from pySecDec.

Functions:
    mult(complex, sequence) -> sequence
"""
import numpy as np
import sympy as sp

eps = sp.symbols('eps')


def mult(scalar, psd_result):
    """Multiply a pySecDec result (value and uncertainty) by a complex number.

    Parameters:
        scalar (complex): A complex number
        psd_result (two-element sequence of sympy objects): A
            (value, uncertainty) pair from pySecDec

    Returns:
        sympy object: Resulting product value
        sympy object: Resulting product uncertainty
    """

    def finite_parts(expr):
        """Return the real and imaginary finite parts of the sympy object expr.

        Parameters:
            expr (sympy object): A truncated Laurent series

        Returns:
            sympy object: The real finite part of expr
            sympy object: The real imaginary part of expr
        """
        finite_part = expr.subs(eps, 0)
        return sp.re(finite_part), sp.im(finite_part)

    re_scalar = scalar.real
    im_scalar = scalar.imag

    value = psd_result[0]
    re_val, im_val = finite_parts(value)

    uncertainty = psd_result[1]
    re_err, im_err = finite_parts(uncertainty)

    re_return_val = re_scalar * re_val - im_scalar * im_val
    im_return_val = re_scalar * im_val + im_scalar * re_val
    re_return_err = np.abs(re_scalar) * re_err + np.abs(im_scalar) * im_err
    im_return_err = np.abs(re_scalar) * im_err + np.abs(im_scalar) * re_err

    return_val = re_return_val + sp.I * im_return_val + sp.Order(eps)
    return_err = re_return_err + sp.I * im_return_err + sp.Order(eps)

    return return_val, return_err

# testing
if __name__ == '__main__':
    result = 2.0 + sp.I * 3.0 + sp.Order(eps), 0.2 + sp.I * 0.3 + sp.Order(eps)

    print(mult(2 + 3j, result))
    print(mult(1j, result))
    print(mult(-1, result))
