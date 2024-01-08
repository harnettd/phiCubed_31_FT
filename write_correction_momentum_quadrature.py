"""Write to file data needed to plot the finite temperature correction.

Write to an external file the data needed to plot the finite temperature
correction to the 3-point vertex function as a function of qx for several
values of a.
"""
from numpy import arange
from pickle import dump

from finite_real_part import finite_real_part
from finite_temperature_vertex import dimless_vertex_sequence
from zero_temperature_vertex import dimless_vertex_use_psd


def set_q_eucl(ell: int, q_space: tuple) -> tuple:
    return ell, *q_space


def get_finite_real_parts(pair: tuple) -> tuple:
    return tuple([finite_real_part(c) for c in pair])


def main() -> None:
    print(__doc__)

    # Set the parameters for the vertex function correction.
    el_1 = 1
    el_2 = 1
    qx_list = list(arange(0, 10, 0.5))
    xi_vals = 1, 0.5, 0
    a_vals = [2.5, 5., 7.5, 10.]
    n_max = 75
    n_range = range(-n_max, n_max + 1)

    # Initialize data dictionaries.
    zero_temp_data = {}
    finite_temp_data = {}

    for a in a_vals:
        # Generate the zero temperature data.
        # First, generate data as complex Laurent series.
        data_zt = [dimless_vertex_use_psd((el_1, qx, 0, 0), (el_2, qx, 0, 0), xi_vals, a)
                   for qx in qx_list]
        # Second, separate finite real parts of values and uncertainties.
        zero_temp_data[a] =\
            [get_finite_real_parts(datum) for datum in data_zt]

        # Generate the finite temperature data.
        finite_temp_data[a] = []
        for qx in qx_list:
            data_ft = dimless_vertex_sequence(
                (el_1, qx, 0, 0), (el_2, qx, 0, 0), xi_vals, a, n_range)
            data_ft = [get_finite_real_parts(datum) for datum in data_ft]
            qx_dict = dict(zip(n_range, data_ft))
            finite_temp_data[a].append(qx_dict)

    directory = './data'
    filename = 'finite_temp_vertex_correction_momentum_quad_data.pkl'
    file = f'{directory}/{filename}'
    data = {
        'qx_vals': qx_list,
        'zero': zero_temp_data,
        'finite': finite_temp_data
    }
    with open(file, 'wb') as f:
        dump(obj=data, file=f)
    f.close()


if __name__ == '__main__':
    main()
