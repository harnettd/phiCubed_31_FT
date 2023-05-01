"""?

Functions:
    scale(sequence) -> sequence
"""

def scale(seq, scalar):
    """Multiply each element of the sequence seq by scalar."""
    return [s * scalar for s in seq]
