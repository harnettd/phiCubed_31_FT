"""Write data to plot zero-temperature vertex function vs momenta."""
import pickle
import numpy as np

from finite_real_part import finite_real_part
import zero_temperature_vertex as ztv

el_1 = 1
el_2 = 1
qx_list = list(np.arange(0, 2.6, 0.1))
a_list = [0.25, 0.5, 0.75, 1.]
xis = 1, 1, 1

vertex_val_data = []
vertex_err_data = []

for a in a_list:
    psd_results =\
        [ztv.dimless_vertex_use_psd([el_1, qx, 0, 0], [el_2, qx, 0, 0], xis, a) for qx in qx_list]
    psd_results_vals = [finite_real_part(result[0]) for result in psd_results]
    psd_results_errs = [finite_real_part(result[1]) for result in psd_results]
    vertex_val_data.append(psd_results_vals)
    vertex_err_data.append(psd_results_errs)

data = {
    'a_list': a_list,
    'qx_list': qx_list,
    'vertex_val_data': vertex_val_data,
    'vertex_err_data': vertex_err_data,
}

filename = 'data/zero_temp_vertex_momentum_data.pkl'
with open(filename, 'wb') as f:
    pickle.dump(data, f)
f.close()
