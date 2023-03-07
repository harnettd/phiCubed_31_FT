"""Docstring
"""


def to_euclidean(vector):
    """Convert a Minkowski to a Euclidean vector.

    Paramaters:
        vector: arraylike
            a Minkowski vector

    Returns:
        arraylike
            a Euclidean vector
    """
    vector_t_mink = vector[0]
    vector_t_eucl = -1j * vector_t_mink
    vector_eucl = vector[1:]
    vector_eucl.insert(0, vector_t_eucl)
    return vector_eucl


def to_minkowski(vector):
    """Convert a Euclidean to a Minkowski vector

    Paramaters:
        vector: arraylike
            a Euclidean vector

    Returns:
        arraylike
            a Minkowski vector
    """
    vector_t_eucl = vector[0]
    vector_t_mink = 1j * vector_t_eucl
    vector_mink = vector[1:]
    vector_mink.insert(0, vector_t_mink)
    return vector_mink
