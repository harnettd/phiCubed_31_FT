"""Compute the finite-temperature three-point correlator.

Functions:
    omega(int, float) -> float
    correlator_term(sequence, sequence, sequence, float, int) -> sympy object
    correlator_sequence(sequence, sequence, sequence, float, range) ->
        list of sympy objects
    correlator_sum(sequence, sequence, sequence, float, range) -> sympy object
    partial_sums(sequence) -> sequence, sequence
    correlator_partial_sum_sequence(sequence, sequence, sequence, float, range) ->
        sequence, sequence, sequence, sequence
"""
import numpy as np

from mult import mult
from sequence_tools import add_lists, get_col
import spatial_integral as spint
from wick_rotation import to_euclidean


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
        sympy object: a finite temperature spatial integral value
        sympy object: a finite temperature spatial integral uncertainty
    """
    p1_eucl = to_euclidean(p1)
    p2_eucl = to_euclidean(p2)
    p1_t_eucl = p1_eucl[0]
    p2_t_eucl = p2_eucl[0]
    p1_space = p1_eucl[1:]
    p2_space = p2_eucl[1:]

    omega_n = omega(index, beta)

    delta_1 = masses[0]**2 + (omega_n + p2_t_eucl)**2
    delta_2 = masses[1]**2 + (omega_n - p1_t_eucl)**2
    delta_3 = masses[2]**2 + omega_n**2
    deltas = delta_1, delta_2, delta_3

    spatial_integral_result = spint.use_psd(p1_space, p2_space, deltas)
    missing_factor = -1 / beta * 1j  # omitted from the generate file
    return mult(missing_factor, spatial_integral_result)


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
        list of 2-tuples of sympy objects: a list of finite temperature spatial
            integral values and uncertainties
    """
    return [correlator_term(p1, p2, masses, beta, index) for index in indices]


def correlator_sum(p1, p2, masses, beta, indices):
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
        sympy object: The sum of finite temperature spatial integrals values
        sympy object: The sum of finite temperature spatial integrals
            uncertainties
    """
    seq = correlator_sequence(p1, p2, masses, beta, indices)
    val = sum(get_col(seq, 0))
    err = sum(get_col(seq, 1))
    return val, err


def partial_sums(seq):
    """Re-arrange and partial sum a sequence.

    Parameters:
        seq (sequence): Sequence to be partial summed

    Returns:
        sequence: Sequence of partial sums
        sequence: Re-arranged sequence
    """

    def accumulate(seq):
        """Returns a partial sum list corresponding to the sequence seq."""
        if len(seq) == 1:
            return list(seq)
        else:
            accumulation = accumulate(seq[:-1])
            accumulation.append(accumulation[-1] + seq[-1])
            return accumulation

    mid_index = int((len(seq) - 1) / 2)
    left_index_seq = seq[:mid_index]
    left_index_seq.reverse()
    right_index_seq = seq[mid_index + 1:]
    re_arranged_seq = add_lists(left_index_seq, right_index_seq)
    re_arranged_seq.insert(0, seq[mid_index])
    return accumulate(re_arranged_seq), re_arranged_seq


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
        list of sympy objects: partial sums of values of finite temperature
            spatial integrals
        list of sympy objects: partial sums of uncertainties in finite
            temperature spatial integrals
        list of sympy objects: values of finite temperature spatial integrals
        list of sympy objects: uncertainties in finite temperature spatial
            integrals
    """
    corr_results = correlator_sequence(p1, p2, masses, beta,
                                       range(-max_index, max_index + 1))
    corr_vals = get_col(corr_results, 0)
    corr_errs = get_col(corr_results, 1)

    corr_vals_partial_sums = partial_sums(corr_vals)
    corr_errs_partial_sums = partial_sums(corr_errs)

    return corr_vals_partial_sums[0], corr_errs_partial_sums[0],\
        corr_vals_partial_sums[1], corr_errs_partial_sums[1]
