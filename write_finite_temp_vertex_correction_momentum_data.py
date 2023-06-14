"""Write data to plot finite temperature vertex correction vs momentum."""
import pickle
import numpy as np

import finite_temperature_vertex as ftv
from finite_real_part import finite_real_part
from sequence_tools import get_col
import zero_temperature_vertex as ztv


def get_finite_real_parts(matrix, col):
    return [finite_real_part(_) for _ in get_col(matrix, col)]


el_1 = 1
el_2 = 1
qx_list = list(np.arange(0, 2.6, 0.1))
xis = 1, 1, 1
a_list = [0.25, 0.5, 0.75, 1]
n_max = 50
n_range = range(-n_max, n_max + 1)

zero_temp_val_matrix = []
zero_temp_err_matrix = []
finite_temp_val_matrix = []
finite_temp_err_matrix = []

for a in a_list:
    zero_temp_results =\
        [ztv.dimless_vertex_use_psd([el_1, qx, 0, 0], [el_2, qx, 0, 0], xis, a) for qx in qx_list]
    finite_temp_results =\
        [ftv.dimless_vertex_sum([el_1, qx, 0, 0], [el_2, qx, 0, 0], xis, a, n_range, True) for qx in qx_list]

    zero_temp_val_matrix.append(get_finite_real_parts(zero_temp_results, 0))
    zero_temp_err_matrix.append(get_finite_real_parts(zero_temp_results, 1))
    finite_temp_val_matrix.append(get_finite_real_parts(finite_temp_results, 0))
    finite_temp_err_matrix.append(get_finite_real_parts(finite_temp_results, 1))

data = {'a_list': a_list,
        'qx_list': qx_list,
        'zero_temp_val_matrix': zero_temp_val_matrix,
        'zero_temp_err_matrix': zero_temp_err_matrix,
        'finite_temp_val_matrix': finite_temp_val_matrix,
        'finite_temp_err_matrix': finite_temp_err_matrix,
}

filename = 'data/finite_temp_vertex_momentum_data.pkl'
with open(filename, 'wb') as f:
    pickle.dump(obj=data, file=f)
f.close()
