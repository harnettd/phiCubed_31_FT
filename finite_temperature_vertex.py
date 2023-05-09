"""Compute dimensionless finite temperature vertex function.

Functions:
    dimless_vertex_term(
        sequence, sequence, sequence, float, int, float) ->
        sequence

    dimless_vertex_sequence(
        sequence, sequence, sequence, float, iterator, float) ->
        sequence

    dimless_vertex_sum(
        sequence, sequence, sequence, float, iterator, float) ->
        sequence, sequence

    dimless_vertex_partial_sum_sequence(
        sequence, sequence, sequence, float, int, float) ->
        sequence, sequence, sequence, sequence
"""
from correlator_to_vertex import correlator_to_vertex
from dimless_to_dimful import dimless_to_dimful
import finite_temperature_correlator as ftc
from mult import mult
from sequence_tools import get_col


def dimless_vertex_term(q1_eucl, q2_eucl, xis, a, index, mass_scale=1.):
    """Compute a finite temperature dimensionless vertex function term.

    Parameters:
        q1_eucl (four-element sequence of complex): First external
            dimensionless Euclidean momentum
        q2_eucl (four-element sequence of complex): Second external
            dimensionless Euclidean momentum
        xis (three-element sequence of float): Dimensionless propagator
            masses
        a (float): Dimensionless inverse temperature
        index (int): Series index
        mass_scale (float): The largest mass in the problem

    Returns:
        sympy object: dimensionless vertex term value
        sympy object: dimensionless vertex term uncertainty
    """
    dimful_params = dimless_to_dimful(q1_eucl, q2_eucl, xis, a, mass_scale)
    corr = ftc.correlator_term(*dimful_params, index)
    vert = correlator_to_vertex(corr)
    dimless_vert = mult(mass_scale**2, vert)
    return dimless_vert


def dimless_vertex_sequence(q1_eucl, q2_eucl, xis, a, indices, mass_scale=1.):
    """Return a list of finite temperature dimensionless vertex function terms.

    Parameters:
        q1_eucl (four-element sequence of complex): First external
            dimensionless Euclidean momentum
        q2_eucl (four-element sequence of complex): Second external
            dimensionless Euclidean momentum
        xis (three-element sequence of float): Dimensionless propagator
            masses
        a (float): Dimensionless inverse temperature
        indices (range of int): Series indices
        mass_scale (float): The largest mass in the problem

    Returns:
        list of sympy object two-tuples: a list of dimensionless vertex function
            value, uncertainty pairs
    """
    return [dimless_vertex_term(q1_eucl, q2_eucl, xis, a, index, mass_scale)
            for index in indices]


def dimless_vertex_sum(q1_eucl, q2_eucl, xis, a, indices, mass_scale=1.):
    """Return a sum of finite temperature dimensionless vertex function terms.

    Parameters:
        q1_eucl (four-element sequence of complex): First external
            dimensionless Euclidean momentum
        q2_eucl (four-element sequence of complex): Second external
            dimensionless Euclidean momentum
        xis (three-element sequence of float): Dimensionless propagator
            masses
        a (float): Dimensionless inverse temperature
        indices (range of int): Series indices
        mass_scale (float): The largest mass in the problem

    Returns:
        sympy object two-tuples: a sum of dimensionless vertex function
            values
        sympy object two-tuples: a sum of dimensionless vertex function
            uncertainties
    """
    dimless_vert_seq =\
        dimless_vertex_sequence(q1_eucl, q2_eucl, xis,
                                a, indices, mass_scale=1)
    dimless_vert_sum_val = sum(get_col(dimless_vert_seq, 0))
    dimless_vert_sum_err = sum(get_col(dimless_vert_seq, 1))
    return dimless_vert_sum_val, dimless_vert_sum_err


def dimless_vertex_partial_sum_sequence(q1_eucl, q2_eucl, xis, a, max_index,
                                        mass_scale=1.):
    """Return a list of partial sums of finite temperature vertex function terms.

    Parameters:
        q1_eucl (four-element sequence of complex): First external
            dimensionless Euclidean momentum
        q2_eucl (four-element sequence of complex): Second external
            dimensionless Euclidean momentum
        xis (three-element sequence of float): Dimensionless propagator
            masses
        a (float): Dimensionless inverse temperature
        max_index (int): Maximum series index
        mass_scale (float): The largest mass in the problem

    Returns:
        list of sympy objects: dimensionless vertex function sequence
            of partial sum values
        list of sympy objects: dimensionless vertex function sequence
            of partial sum uncertainties
        list of sympy objects: dimensionless vertex function sequence
            values
        list of sympy objects: dimensionless vertex function sequence
            uncertainties
    """
    dimful_params = dimless_to_dimful(q1_eucl, q2_eucl, xis, a, mass_scale)
    corr_parsum_vals, corr_parsum_errs, corr_seq_vals, corr_seq_errs =\
        ftc.correlator_partial_sum_sequence(*dimful_params, max_index)

    num_cols = len(corr_parsum_vals)
    vert_parsum_results =\
        [correlator_to_vertex([corr_parsum_vals[col], corr_parsum_errs[col]])
         for col in range(num_cols)]
    vert_seq_results =\
        [correlator_to_vertex([corr_seq_vals[col], corr_seq_errs[col]])
         for col in range(num_cols)]

    dimless_vert_parsum_results =\
        [mult(mass_scale**2, result) for result in vert_parsum_results]
    dimless_vert_seq_results =\
        [mult(mass_scale**2, result) for result in vert_seq_results]

    dimless_vert_parsum_vals = get_col(dimless_vert_parsum_results, 0)
    dimless_vert_parsum_errs = get_col(dimless_vert_parsum_results, 1)
    dimless_vert_seq_vals = get_col(dimless_vert_seq_results, 0)
    dimless_vert_seq_errs = get_col(dimless_vert_seq_results, 1)

    return dimless_vert_parsum_vals, dimless_vert_parsum_errs,\
        dimless_vert_seq_vals, dimless_vert_seq_errs
