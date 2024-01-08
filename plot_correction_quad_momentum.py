"""Plot finite-temperature vertex correction vs qx."""
from math import ceil, sqrt
import matplotlib.pyplot as plt
import numpy as np
import pickle


def get_data(
        directory: str = './data',
        filename: str = 'finite_temp_vertex_correction_momentum_quad_data.pkl')\
        -> dict:
    """
    :param str directory: The directory of the pickled file.
    :param str filename: The name of the pickled file.
    :return : The finite-temperature vertex correction data.
    :rtype dict:
    """
    file = f'{directory}/{filename}'
    with open(file, 'rb') as f:
        data = pickle.load(file=f)
    f.close()
    return data


def negate(measurement: tuple[float, float]) -> tuple[float, float]:
    """Negate a measurement value, leaving its error unmodified."""
    val, err = measurement[0], measurement[1]
    return -val, err


def strip_n(vertex_n_map: dict) -> list:
    """Strip the keys from a map of finite-temperature vertex values."""
    return list(vertex_n_map.values())


def get_col(pairs: list[tuple[float, float]], idx: int) -> list[float]:
    """Return a column from a list of 2-tuples."""
    return [pair[idx] for pair in pairs]


def sum_quadrature(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """Sum pairs, combining errors in quadrature."""
    vals = get_col(pairs, 0)
    errs = get_col(pairs, 1)
    val = sum(vals)
    err = sqrt(sum([err ** 2 for err in errs]))
    return val, err


def transpose(pairs: list[tuple[float, float]]) -> tuple[np.array, np.array]:
    """Transpose a list of tuples to a tuple of numpy arrays."""
    vals = np.array(get_col(pairs, 0))
    errs = np.array(get_col(pairs, 1))
    return vals, errs


def make_fig(qx_vals: list, data: dict) -> None:
    """Generate a figure of finite-temperature vertex correction plots."""
    data_len = len(data)
    a_vals = list(data.keys())
    num_rows, num_cols = ceil(data_len / 2), 2
    print(f'len(data), num_rows, num_cols = {len(data)}, {num_rows}, {num_cols}')
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols)
    qx_arr = np.array(qx_vals)

    for row in range(num_rows):
        for col in range(num_cols):
            idx = 2 * row + col
            print(f'row, col, idx = {row}, {col}, {idx}')
            if idx < data_len:
                vals, errs = transpose(data[a_vals[idx]])
                axs[row, col].errorbar(qx_arr, vals, errs)
    plt.show()


def main() -> None:
    print(__doc__)

    # Import pickled finite-temperature vertex correction data.
    data_vertex = get_data()
    qx_vals = data_vertex['qx_vals']
    data_zero = data_vertex['zero']
    data_finite = data_vertex['finite']

    print(data_vertex)

    # Negate all vertex values in data_zero.
    minus_data_zero = {a: [negate(pair) for pair in pairs]
                       for a, pairs in data_zero.items()}

    # Strip n-values from data_finite.
    data_finite_n_stripped = {a: [strip_n(n_map) for n_map in n_maps]
                              for a, n_maps in data_finite.items()}

    # Append data_zero values to data_finite_n_stripped.
    data_correction = data_finite_n_stripped
    for a in data_correction:
        for idx in range(len(data_correction[a])):
            data_correction[a][idx].append(minus_data_zero[a][idx])

    # Sum vertex correction values in quadrature.
    data_correction_summed =\
        {a: [sum_quadrature(val_list) for val_list in nested_val_list]
         for a, nested_val_list in data_correction.items()}

    # Generate figure of plots.
    make_fig(qx_vals, data_correction_summed)


if __name__ == '__main__':
    main()
