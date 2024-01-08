"""Plot finite-temperature vertex correction vs a."""
from math import ceil, sqrt
import matplotlib.pyplot as plt
import numpy as np
import pickle


def get_data(
        directory: str = './data',
        filename: str = 'finite_temp_vertex_correction_quad_data.pkl')\
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


def make_fig(a_vals: list, data: dict) -> None:
    """Generate a figure of finite-temperature vertex correction plots."""
    data_len = len(data)
    num_rows, num_cols = ceil(data_len / 2), 2
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols,
                            sharex=True, sharey=False, figsize=(6, 3.7),
                            layout='constrained')
    a_arr = np.array(a_vals)

    params = {
        0: [-1, '.-k', 'upper right', None, r'$\tilde{\Gamma}_T - \tilde{\Gamma}_0$'],
        1: [-1, '.--k', 'upper right', None, None],
        2: [-1, '.-.k', 'lower right', r'$a$', r'$\tilde{\Gamma}_T - \tilde{\Gamma}_0$'],
        3: [-1, '.:k', 'lower right', r'$a$', None],
    }

    for row in range(num_rows):
        for col in range(num_cols):
            idx = 2 * row + col
            print(f'row, col, idx = {row}, {col}, {idx}')
            if idx < data_len:
                vals, errs = transpose(data[idx])
                axs[row, col].errorbar(a_arr, vals, errs, fmt=params[idx][1])
                axs[row, col].set_title(r'$\ell =$ {}'.format(idx))
                axs[row, col].set_xlabel(params[idx][3])
                axs[row, col].set_ylabel(params[idx][4])
                if row == 0 and col == 0:
                    axs[row, col].set_yscale('log')
    plt.show()


def main() -> None:
    print(__doc__)

    # Import pickled finite-temperature vertex correction data.
    data_vertex = get_data()
    a_vals = data_vertex['a_vals']
    data_zero = data_vertex['zero']
    data_finite = data_vertex['finite']

    # Negate all vertex values in data_zero.
    minus_data_zero = {ell: [negate(pair) for pair in pairs]
                       for ell, pairs in data_zero.items()}

    # Strip n-values from data_finite.
    data_finite_n_stripped = {ell: [strip_n(n_map) for n_map in n_maps]
                              for ell, n_maps in data_finite.items()}

    # Append data_zero values to data_finite_n_stripped.
    data_correction = data_finite_n_stripped
    for ell in data_correction:
        for idx in range(len(data_correction[ell])):
            data_correction[ell][idx].append(minus_data_zero[ell][idx])

    # Sum vertex correction values in quadrature.
    data_correction_summed =\
        {ell: [sum_quadrature(val_list) for val_list in nested_val_list]
         for ell, nested_val_list in data_correction.items()}

    # Generate figure of plots.
    make_fig(a_vals, data_correction_summed)


if __name__ == '__main__':
    main()
