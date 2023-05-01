"""Convert a three-point correlator to a vertex function.

Functions:
    correlator_to_vertex(sequence) -> sequence 
"""
from mult import mult


def correlator_to_vertex(psd_result):
    """Convert a three-point correlator to a vertex function.
    
    Parameters:
        psd_result (two-element sequence of sympy objects): A
            (value, uncertainty) pair from pySecDec

    Returns:
        sympy object: Resulting product value
        sympy object: Resulting product uncertainty 
    """
    return mult(1j, psd_result)


# testing
if __name__ == '__main__':
    import sympy as sp
    val = 2 - 3 * sp.I
    err = 0.5 + 1.5 * sp.I
    print(correlator_to_vertex((val, err)))
