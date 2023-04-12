"""Compute the finite temperature three-point correlator.

Functions:
    omega(int, float) -> float
    correlator_term(sequence, sequence, sequence, float, int) -> sympy object
    correlator_sequence(sequence, sequence, sequence, float, range) ->
        list of sympy objects
    correlator_partial_sum(sequence, sequence, sequence, float, range) ->
        sympy object
    accumulate(sequence) -> list
    correlator_partial_sum_sequence(sequence, sequence, sequence, float, range) ->
        list of sympy objects, list of sympy objects
"""
import numpy as np
from sympy import expand, I

import spatial_integral as spint


def omega(index, beta):
    """Compute a Matsubara frequency.

    Parameters:
        index (int): sequence index
        beta (float): inverse temperature

    Returns:
        float: Matsubara frequency
    """
    return 2 * np.pi * index / beta


def correlator_term(p1, p2, masses, beta, index):
    """Compute a finite temperature spatial integral.

    Parameters:
        p1 (four-element sequence of complex): First external Minkowski
            momentum
        p2 (four-element sequence of complex): Second external Minkowski
            momentum
        masses (three-element sequence of float): Propagator masses
        beta (float): Inverse temperature
        index (int): Series index

    Returns:
        sympy object: a finite temperature spatial integral
    """
    p1_0_eucl = -1j * p1[0]
    p2_0_eucl = -1j * p2[0]
    p1_space = p1[1:]
    p2_space = p2[1:]
    omega_n = omega(index, beta)
    delta_1 = masses[0]**2 + (omega_n + p2_0_eucl)**2
    delta_2 = masses[1]**2 + (omega_n - p1_0_eucl)**2
    delta_3 = masses[2]**2 + omega_n**2
    return expand(-I / beta * spint.use_psd(p1_space, p2_space, delta_1, delta_2, delta_3))


def correlator_sequence(p1, p2, masses, beta, indices):
    """Compute a list of finite temperature spatial integrals.

    Parameters:
        p1 (four-element sequence of complex): First external Minkowski
            momentum
        p2 (four-element sequence of complex): Second external Minkowski
            momentum
        masses (three-element sequence of float): Propagator masses
        beta (float): Inverse temperature
        indices (range of int): Series indices

    Returns:
        list of sympy objects: a list of finite temperature spatial integrals
    """
    return [correlator_term(p1, p2, masses, beta, index) for index in indices]


def correlator_partial_sum(p1, p2, masses, beta, indices):
    """Compute a sum of finite temperature spatial integrals.

    Parameters:
        p1 (four-element sequence of complex): First external Minkowski
            momentum
        p2 (four-element sequence of complex): Second external Minkowski
            momentum
        masses (three-element sequence of float): Propagator masses
        beta (float): Inverse temperature
        indices (range of int): Series indices

    Returns:
        sympy object: a sum of finite temperature spatial integrals
    """
    seq = correlator_sequence(p1, p2, masses, beta, indices)
    return sum(seq)


def accumulate(seq):
    """Returns a partial sum list corresponding to the sequence seq."""
    if len(seq) == 1:
        return list(seq)
    else:
        partial_sums = accumulate(seq[:-1])
        partial_sums.append(partial_sums[-1] + seq[-1])
        return partial_sums


def correlator_partial_sum_sequence(p1, p2, masses, beta, max_index):
    """Return a sequence of sums of finite temperature spatial integrals.

    Parameters:
        p1 (four-element sequence of complex): First external Minkowski
            momentum
        p2 (four-element sequence of complex): Second external Minkowski
            momentum
        masses (three-element sequence of float): Propagator masses
        beta (float): Inverse temperature
        max_index (positive int): Series index (included)

    Returns:
        list of sympy objects: a list of sums of finite temperature spatial
            integrals
        list of sympy objects: a list of finite temperature spatial
            integrals
    """
    def add_lists(list_1, list_2):
        """Add two sequences element-by-element.
      
        It is assumed that the two lists have equal length.
        """
        return [list_1[index] + list_2[index] for index in range(len(list_1))]

    seq = correlator_sequence(p1, p2, masses, beta, range(-max_index, max_index + 1))
    neg_index_corr_vals = seq[:max_index]
    neg_index_corr_vals.reverse()
    zero_index_corr_val = seq[max_index]
    pos_index_corr_vals = seq[max_index + 1:]

    corr_seq = add_lists(neg_index_corr_vals, pos_index_corr_vals)
    corr_seq.insert(0, zero_index_corr_val)
    partial_sum_seq = accumulate(corr_seq)

    return partial_sum_seq, corr_seq


def dimensionless_vertex_function_partial_sum(l1, q1, l2, q2, xi1, xi2, xi3, a, n_min, n_max, M):
    """Compute the dimensionless vertex function (Gamma)"""
    M_over_a = M / a
    p1t = 1j * l1 * M_over_a
    p1 = [x * M_over_a for x in q1]  # spatial components
    p1.insert(0, p1t)
    p2t = 1j * l2 * M_over_a
    p2 = [x * M_over_a for x in q2]  # spatial components
    p2.insert(0, p2t)
    m1 = M * xi1
    m2 = M * xi2
    m3 = M * xi3
    beta = 2 * np.pi / M_over_a
    Pi = correlator_partial_sum(p1, p2, m1, m2, m3, beta, n_min, n_max)
    return expand(M**2 * I * Pi)
