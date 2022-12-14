from sympy import symbols, sympify

eps, value, uncertainty, indeterminate = symbols('eps value uncertainty indeterminate')

def psd_to_sympy(psd_str_result):
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
    #TODO: Check for NaN before converting.
    # What follows is a pretty dangerous hack.
    return sympify(psd_str_result.replace('nan', 'indeterminate').
        replace(' +/- ', '*value+uncertainty*').replace(',', '+I*'))


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
