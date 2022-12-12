"""Docstring
"""
import numpy as np

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
    