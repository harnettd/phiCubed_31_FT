"""Compute the dimensionless, zero-temperature, three-point vertex.

Functions:
    dimless_vertex_function(sequence, sequence, sequence, float) ->
        sympy object, sympy object
"""
from correlator_to_vertex import correlator_to_vertex
from dimless_to_dimful import dimless_to_dimful
from mult import mult
import zero_temperature_correlator as ztc


def dimless_vertex_use_psd(q1_eucl, q2_eucl, xis, a, mass_scale=1):
    """Compute the dimensionless, zero-temperature, vertex function.

    Parameters:
        q1_eucl (four-element sequence of complex): First dimensionless
            Euclidean external momentum
        q2-eucl (four-element sequence of complex): Second dimensionless
            Euclidean external momentum
        xis (three-element sequence of float): Dimensionless propagator masses
        mass_scale (float): Mass scale, i.e., the largest propagator mass

    One element of xis should be 1. The other two should be less than or equal
        to 1 and greater than or equal to 0.

    The results are independent of mass_scale.

    Returns:
        sympy object: Dimensionless zero temperature vertex function 
            value
        sympy object: Dimensionless zero temperature vertex function
            uncertainty
    """
    dimful_params = dimless_to_dimful(q1_eucl, q2_eucl, xis, a, mass_scale)
    corr = ztc.correlator_use_psd(*dimful_params[:-1])
    vert = correlator_to_vertex(corr)
    dimless_vert = mult(mass_scale**2, vert)
    return dimless_vert
