"""Calculate the dot product of two Minkowski vectors.

Functions:
    dot_product(sequence, sequence) -> complex
"""
import numpy as np


def dot_product(vector_1, vector_2):
    """Compute the Minkowski dot product of vector_1 and vector_2.

    vector_1 and vector_2 should be complex sequences with equal 
    lengths. The first components are assumed to be timelike.

    Parameters:
        vector_1 (sequence of complex): Minkowski vector
        vector_2 (sequence of complex): Minkowski vector

    Returns
        complex: Dot product of v1 and v2
    """
    v1_arr = np.array(vector_1)
    v2_arr = np.array(vector_2)
    if v1_arr.shape != v2_arr.shape:
        print('Error in dot_product: v1 and v2 are incompatible')
        return None
    if v1_arr.ndim == 0:
        return v1_arr * v2_arr
    elif v1_arr.ndim == 1:
        return v1_arr[0] * v2_arr[0] - sum(v1_arr[1:] * v2_arr[1:])
    else:
        print('Error in dot_product: v1 and/or v2 not rank-1 objects')
        return None
    

if __name__ == '__main__':
    print(dot_product(2, 3))
    print(dot_product([1, 1], [3, 2]))
    print(dot_product([1j, 1], [3j, 2]))
