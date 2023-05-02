"""Convert dimensionless correlator inputs to dimensionful inputs.

Functions:
    make_minkowski_vector(sequence, float, float) => sequence

    dimless_to_dimful(sequence, sequence, sequence, float, float) =>
        sequence, sequence, sequence, float
"""
import numpy as np

from sequence_tools import scale
from wick_rotation import to_minkowski


def make_minkowski_vector(q_eucl, a, mass_scale=1):
    """Return a Minkowski four-vector.
    
    Parameters:
        q_eucl (four-element sequence of complex): Euclidean four-vector
        a (float): Inverse dimensionless temperature
        mass_scale (float): The largest propagator mass

    Returns:
        four-element sequence of complex: Minkowski four-vector

    For applications to finite temperature vertex function calculations,
    the first component of q should be an integer.
    """
    p_eucl = scale(q_eucl, mass_scale / a)
    return to_minkowski(p_eucl)


def dimless_to_dimful(q1_eucl, q2_eucl, xis, a, mass_scale=1):
    """Convert dimensionless correlator inputs to dimensionful inputs.
    
    Parameters:
        q1_eucl (four-element sequence of complex): First dimensionless
            Euclidean momentum
        q2_eucl (four-element sequence of complex): Second dimensionless
            Euclidean momentum
        xis (three-element sequence of float): Dimensionless propagator masses
        a (float): Inverse dimensionless temperature
    
    Returns:
        four-element sequence of complex: First dimensionful Minkowski
            momentum
        four-element sequence of complex: Second dimensionful Minkowski
            momentum        
        three-element sequence of float: Dimensionful propagator masses
        float: Dimensionful inverse temperature
    """
    p1_mink = make_minkowski_vector(q1_eucl, a, mass_scale)
    p2_mink = make_minkowski_vector(q2_eucl, a, mass_scale)
    masses = scale(xis, mass_scale)
    beta = 2 * np.pi * a / mass_scale

    return p1_mink, p2_mink, masses, beta


if __name__ == '__main__':
    q1_eucl = 1, 2, 3, 4
    q2_eucl = -1, -2, -3, -4
    xis = 1, 1, 1
    a = 3
    m_scale = 2
    result = dimless_to_dimful(q1_eucl, q2_eucl, xis, a, m_scale)
    print(result[:-1])
