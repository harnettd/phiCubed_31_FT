"""Convert between Minkowski and Euclidean vectors.

    to_euclidean(sequence) -> sequence
    to_minkowski(sequence) -> sequence
"""


def to_euclidean(vector):
    """Convert a Minkowski to a Euclidean vector.

    Paramaters:
        vector (sequence of complex): a Minkowski vector

    Returns:
        sequence of complex: a Euclidean vector
    """
    vector_list = list(vector)
    vector_t_mink = vector_list[0]
    vector_t_eucl = -1j * vector_t_mink
    vector_eucl = vector_list[1:]
    vector_eucl.insert(0, vector_t_eucl)
    return vector_eucl


def to_minkowski(vector):
    """Convert a Euclidean to a Minkowski vector

    Paramaters:
        vector (sequence): a Euclidean vector

    Returns:
        sequence: a Minkowski vector
    """
    vector_list = list(vector)
    vector_t_eucl = vector_list[0]
    vector_t_mink = 1j * vector_t_eucl
    vector_mink = vector_list[1:]
    vector_mink.insert(0, vector_t_mink)
    return vector_mink
