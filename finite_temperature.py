"""Compute the finite temperature three-point correlator.

Functions:
    omega(int, float) -> float
    correlator_term(sequence, sequence, sequence, float, int) -> sympy object
    correlator_sequence(sequence, sequence, sequence, float, range) ->
        list of sympy objects
    get_col(array, index) -> sequence
    correlator_sum(sequence, sequence, sequence, float, range) -> sympy object
    partial_sums(sequence) -> sequence, sequence
    correlator_partial_sum_sequence(sequence, sequence, sequence, float, range) ->
        sequence, sequence, sequence, sequence
    dimensionless_vertex_function_partial_sum(
        int, sequence, int, sequence, sequence, float, indices, float) ->
        sympy object
"""
import numpy as np

from mult import mult
import spatial_integral as spint
from wick_rotation import to_euclidean, to_minkowski


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
        list of sympy objects: a list of finite temperature spatial integrals
    """
    return [correlator_term(p1, p2, masses, beta, index) for index in indices]


def get_col(arr, col):
    """Return the column col of array arr."""
    return [row[col] for row in arr]


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

    def add_lists(seq_1, seq_2):
        """Add two sequences element-by-element.

        It is assumed that the two lists have equal length.
        """
        return [seq_1[index] + seq_2[index] for index in range(len(seq_1))]

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
        list of sympy objects: partial sums of values of finite temperature spatial integrals
        list of sympy objects: partial sums of uncertainties in finite temperature spatial integrals
        list of sympy objects: values of finite temperature spatial integrals
        list of sympy objects: uncertainties in finite temperature spatial integrals
    """
    corr_results = correlator_sequence(p1, p2, masses, beta,
                                       range(-max_index, max_index + 1))
    corr_vals = get_col(corr_results, 0)
    corr_errs = get_col(corr_results, 1)

    corr_vals_partial_sums = partial_sums(corr_vals)
    corr_errs_partial_sums = partial_sums(corr_errs)

    return corr_vals_partial_sums[0], corr_errs_partial_sums[0],\
        corr_vals_partial_sums[1], corr_errs_partial_sums[1],


def dimensionless_vertex_function_partial_sum(el_1, q1_space, el_2, q2_space, xis, a, indices, mass_scale=1):
    """Compute the dimensionless vertex function (i.e., Gamma).

    Parameters:
        el_1 (int): Integer corresponding to first Matsubara frequency
        q1_space (three-element sequence of complex): First dimensionless
            spatial momentum
        el_2 (int): Integer corresponding to second Matsubara frequency
        q2_space (three-element sequence of complex): Second dimensionless
            spatial momentum
        xis (three-element sequence of float): Dimensionless propagator masses
        a (float): Dimensionless inverse temperature
        indices (range of int): Series indices
        mass_scale (float): The largest propagator mass

    Returns:
        sympy object: dimensionless vertex function value
        sympy object: dimensionless vertex function uncertainty
    """

    def rescale(sequence, scalar):
        """Multiply each element of sequence by scalar."""
        return [s * scalar for s in sequence]

    def make_minkowski_vector(el, q_space):
        """Return a Minkowski vector.
        
        Parameters:
            el (int): integer indicating a Matsubara frequency
            q_space (three-element sequence of floar): spatial momentum

        Returns:
            four-element sequence of complex: Minkowski 4-vector
        """
        p_t_eucl = el * mass_scale / a
        p_eucl = rescale(q_space, mass_scale / a)  # spatial components
        p_eucl.insert(0, p_t_eucl)
        return to_minkowski(p_eucl)

    p1_mink = make_minkowski_vector(el_1, q1_space)
    p2_mink = make_minkowski_vector(el_2, q2_space)
    masses = rescale(xis, mass_scale)
    beta = 2 * np.pi * a / mass_scale

    corr = correlator_sum(p1_mink, p2_mink, masses, beta, indices)
    corr_to_dimensionless_vert_factor = mass_scale**2 * 1j
    return mult(corr_to_dimensionless_vert_factor, corr)
