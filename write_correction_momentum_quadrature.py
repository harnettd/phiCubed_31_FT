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
    qx_vals = list(arange(0, 10, 0.5))
    xi_vals = 1, 0.5, 0
    a_vals = [2.5, 5., 7.5, 10.]
    n_max = 76
    n_range = range(-n_max, n_max + 1)

    # Initialize data dictionary.
    vertex_data = {
        'el_vals': (el_1, el_2),
        'qx_vals': qx_vals,
        'xi_vals': xi_vals,
        'n_max': n_max,
        'vertex_vals': {}
    }

    for a in a_vals:
        vertex_data['vertex_vals'][a] = {
            'zero_temp_vals': [],
            'finite_temp_vals': []
        }

        for qx in qx_vals:
            q1_eucl = set_q_eucl(el_1, (qx, 0, 0))
            q2_eucl = set_q_eucl(el_2, (qx, 0, 0))

            # Generate the zero temperature data point.
            pair_raw = dimless_vertex_use_psd(q1_eucl, q2_eucl, xi_vals, a)
            pair = get_finite_real_parts(pair_raw)
            vertex_data['vertex_vals'][a]['zero_temp_vals'].append(pair)

            # Generate the finite temperature data point (a sequence).
            pair_list_raw = dimless_vertex_sequence(q1_eucl, q2_eucl, xi_vals, a, n_range)
            pair_list = [get_finite_real_parts(item) for item in pair_list_raw]
            vertex_data['vertex_vals'][a]['finite_temp_vals'].append(pair_list)

    # Pickle vertex_data
    directory = './data'
    filename = 'finite_temp_vertex_correction_momentum_quad_data.pkl'
    file = f'{directory}/{filename}'
    data = vertex_data
    with open(file, 'wb') as f:
        dump(obj=data, file=f)
    f.close()


if __name__ == '__main__':
    main()
