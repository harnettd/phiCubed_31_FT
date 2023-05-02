"""?

Functions:
    scale(sequence) -> sequence
"""


def scale(seq, scalar):
    """Multiply each element of the sequence seq by scalar."""
    return [s * scalar for s in seq]


def get_col(arr, col):
    """Return the column col of array arr."""
    return [row[col] for row in arr]


def add_lists(seq_1, seq_2):
    """Add two sequences element-by-element.

    It is assumed that the two lists have equal length.
    """
    return [seq_1[index] + seq_2[index] for index in range(len(seq_1))]
