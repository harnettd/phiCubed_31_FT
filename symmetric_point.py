"""Determine 4-vectors at the symmetric point for a 3-point correlator.

Functions:
    p1_p2(float) -> array, array
"""
import numpy as np

def p1_p2(qq):
    """Compute 4-vectors corresponding to the symmetric point qq.

    Parameters
        qq (float): Symmetric point value

    Returns:
        p1 (4d array of complex): First 4-momentum
        p2 (4d array of complex): Second 4-momentum
    """
    if qq >= 0:
        p1x = np.sqrt(qq) * 1j
        p2t = 0.5 * np.sqrt(3 * qq)
    elif qq < 0:
        p1x = np.sqrt(np.abs(qq))
        p2t = 0.5 * np.sqrt(3*np.abs(qq)) * 1j
    else:
        pass  # error
    p1t = 0
    p2x = -0.5 * p1x
    p1 = np.array([p1t, p1x, 0, 0])
    p2 = np.array([p2t, p2x, 0, 0])

    return p1, p2
