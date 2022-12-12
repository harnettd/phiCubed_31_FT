def dot_product(v1, v2):
    """Compute the Minkowski dot product of v1 and v2.

    v1 and v2 should be array_like objects with equal positive-integer
    dimensions and with complex elements.
    The first component is assumed to be the timelike one.

    Parameters
    ----------
    v1 : array_like, complex
        Minkowski vector.
    v2 : array_like, complex
        Minkowski vector.

    Returns
    -------
    Complex.
        Dot product of v1 and v2.
    """
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)
    if v1_arr.shape != v2_arr.shape:
        print('Error in dot_product: v1 and v2 are incompatible')
        return None
    if v1_arr.ndim == 0:
        return v1_arr*v2_arr
    elif v1_arr.ndim == 1:
        return v1_arr[0]*v2_arr[0] - sum(v1_arr[1:]*v2_arr[1:])
    else:
        print('Error in dot_product: v1 and/or v2 not rank-1 objects')
        return None