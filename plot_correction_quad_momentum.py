"""Plot finite-temperature vertex correction vs qx."""
from math import ceil, sqrt
import matplotlib.pyplot as plt
import numpy as np
import pickle

from finite_temperature_vertex import zeta_function_correction


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
    plot_options = [
        {'format': '.-k', 'x_label': None,
         'y_label': r'$\tilde{\Gamma}_T - \tilde{\Gamma}_0$',
         'legend_loc': 'upper right'},
        {'format': '.--k', 'x_label': None,
         'y_label': None,
         'legend_loc': 'upper right'},
        {'format': '.-.k', 'x_label': r'$q^1$',
         'y_label': r'$\tilde{\Gamma}_T - \tilde{\Gamma}_0$',
         'legend_loc': 'upper right'},
        {'format': '.:k', 'x_label': r'$q^1$',
         'y_label': None,
         'legend_loc': 'upper right'}
    ]

    data_len = len(data)
    a_vals = list(data.keys())
    num_rows, num_cols = ceil(data_len / 2), 2
    print(f'len(data), num_rows, num_cols = {len(data)}, {num_rows}, {num_cols}')
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols,
                            figsize=(6, 3.7), layout='constrained')
    qx_arr = np.array(qx_vals)

    for row in range(num_rows):
        for col in range(num_cols):
            idx = 2 * row + col
            if idx < data_len:
                vals, errs = transpose(data[a_vals[idx]])
                axs[row, col].errorbar(qx_arr, vals, errs, fmt=plot_options[idx]['format'],
                                       label=r'$a=$' + '{:4.2f}'.format(a_vals[idx]))
                axs[row, col].set_title(r'$a$ = {:4.1f}'.format(a_vals[idx]))
                axs[row, col].set_xlabel(plot_options[idx]['x_label'])
                axs[row, col].set_ylabel(plot_options[idx]['y_label'])
    plt.show()


def main() -> None:
    print(__doc__)

    # Import pickled finite-temperature vertex correction data.
    vertex_data = get_data()
    # print(json.dumps(vertex_data, indent=4))
    el_1, el_2 = vertex_data['el_vals']
    qx_vals = vertex_data['qx_vals']
    xi_vals = vertex_data['xi_vals']
    n_max = vertex_data['n_max']
    vertex_vals = vertex_data['vertex_vals']
    a_vals = vertex_vals.keys()

    # For testing:
    a_tmp = 2.5
    qx_tmp = 0
    zt_tmp = get_col(vertex_vals[a_tmp]['zero_temp_vals'], 0)
    ft_tmp_list = [get_col(val_list, 0) for val_list in vertex_vals[a_tmp]['finite_temp_vals']]
    ft_tmp_list[qx_tmp].append(-zt_tmp[qx_tmp])
    # print(ft_tmp_list[qx_tmp])
    print(sum(ft_tmp_list[qx_tmp]))
    print()

    # Append zero-temperature data values to corresponding sequences of
    # finite-temperature values.
    diff_vals_seq =\
        {a: v['finite_temp_vals'].copy() for (a, v) in vertex_vals.items()}
    for a in a_vals:
        zt_list = vertex_vals[a]['zero_temp_vals']
        minus_zt_list = [negate(zt) for zt in zt_list]
        ft_dbl_list = diff_vals_seq[a]
        for (minus_zt, ft_list) in zip(minus_zt_list, ft_dbl_list):
            ft_list.append(minus_zt)

    # Sum vertex correction values in quadrature.
    zeta_corrs = {a: zeta_function_correction(n_max, a) for a in a_vals}
    diff_vals = diff_vals_seq.copy()
    for (a, val_list_list) in diff_vals.items():
        summed = [sum_quadrature(val_list) for val_list in val_list_list]
        summed_with_zeta = [(s[0] + zeta_corrs[a], s[1]) for s in summed]
        diff_vals[a] = summed_with_zeta
        # diff_vals[a] = summed

    # For testing:
    print(diff_vals[a_tmp][qx_tmp])

    # Generate figure of plots.
    make_fig(qx_vals, diff_vals)


if __name__ == '__main__':
    main()
