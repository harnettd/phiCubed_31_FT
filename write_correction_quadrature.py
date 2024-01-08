"""Write to file data needed to plot the finite temperature correction.

Write to an external file the data needed to plot the finite temperature
correction to the 3-point vertex function as a function of a for several
values of ell.
"""
from numpy import linspace
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
    ell_vals = range(4)
    q1_space = 0, 0, 0
    q2_space = 0, 0, 0
    xi_vals = 1, 0.5, 0
    a_vals = list(linspace(0.025, 10, 25))
    n_max = 75
    n_range = range(-n_max, n_max + 1)

    # Initialize data dictionaries.
    zero_temp_data = {}
    finite_temp_data = {}

    for ell in ell_vals:
        q1_eucl = set_q_eucl(ell, q1_space)
        q2_eucl = set_q_eucl(ell, q2_space)
        params = q1_eucl, q2_eucl, xi_vals

        # Generate the zero temperature data.
        # First, generate data as complex Laurent series.
        data_zt = [dimless_vertex_use_psd(*params, a) for a in a_vals]
        # Second, separate finite real parts of values and uncertainties.
        zero_temp_data[ell] =\
            [get_finite_real_parts(datum) for datum in data_zt]

        # Generate the finite temperature data.
        finite_temp_data[ell] = []
        for a in a_vals:
            data_ft = dimless_vertex_sequence(*params, a, n_range)
            data_ft = [get_finite_real_parts(datum) for datum in data_ft]
            a_dict = dict(zip(n_range, data_ft))
            finite_temp_data[ell].append(a_dict)

    directory = './data'
    filename = 'finite_temp_vertex_correction_quad_data.pkl'
    file = f'{directory}/{filename}'
    data = {
        'a_vals': a_vals,
        'zero': zero_temp_data,
        'finite': finite_temp_data
    }
    with open(file, 'wb') as f:
        dump(obj=data, file=f)
    f.close()


if __name__ == '__main__':
    main()
