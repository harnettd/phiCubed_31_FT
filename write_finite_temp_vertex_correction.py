"""Write the data needed to plot the finite temperature vertex correction."""
import pickle
import numpy as np
import sympy as sp

import finite_temperature_vertex as ftv
from finite_real_part import finite_real_part
from sequence_tools import get_col
import zero_temperature_vertex as ztv


def append_to(result_matrix, col_num):
    """Append the real, finite part of a data_matrix col to result_matrix."""
    finite_real_parts =\
        [finite_real_part(_) for _ in get_col(data_matrix, col_num)]
    result_matrix.append(finite_real_parts)


eps = sp.symbols('eps')
el_list = list(np.arange(0, 5))
q1_space = 0, 0, 0
q2_space = 0, 0, 0
xis = 1, 1, 1
a_list = list(np.linspace(0.1, 2.5, 40))
n_max = 75
n_range = range(-n_max, n_max + 1)

zero_temp_val_matrix = []
zero_temp_err_matrix = []
finite_temp_val_matrix = []
finite_temp_err_matrix = []

for el in el_list:
    data_matrix = []
    q1_eucl = [el, *q1_space]
    q2_eucl = [el, *q2_space]

    for a in a_list:
        data_matrix_row =\
            (*ztv.dimless_vertex_use_psd(q1_eucl, q2_eucl, xis, a),
             *ftv.dimless_vertex_sum(q1_eucl, q2_eucl, xis, a, n_range, True))
        data_matrix.append(data_matrix_row)

    append_to(zero_temp_val_matrix, 0)
    append_to(zero_temp_err_matrix, 1)
    append_to(finite_temp_val_matrix, 2)
    append_to(finite_temp_err_matrix, 3)

data = {
    "el_list": el_list,
    "a_list": a_list,
    "zero_temp_val_matrix": zero_temp_val_matrix,
    "zero_temp_err_matrix": zero_temp_err_matrix,
    "finite_temp_val_matrix": finite_temp_val_matrix,
    "finite_temp_err_matrix": finite_temp_err_matrix
}

data_file = "data/finite_temp_vertex_correction_data.pkl"
with open(data_file, 'wb') as f:
    pickle.dump(obj=data, file=f)
f.close()
