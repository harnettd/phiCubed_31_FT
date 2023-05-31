"""Write the data needed to plot the zero-temperature vertex function."""
import pickle
import numpy as np
import sympy as sp

import zero_temperature_vertex as ztv

eps = sp.symbols('eps')


def finite_real_part(expr):
    """Return the finite, real part of expr"""
    return complex(expr.subs(eps, 0)).real


q1_space = 0, 0, 0
q2_space = 0, 0, 0
xis = 1, 1, 1

el_list = list(np.arange(0, 2))
a_list = list(np.linspace(0.1, 2.1, 3))

vertex_val_data = []
vertex_err_data = []
for el in el_list:
    q1_eucl = [el, *q1_space]
    q2_eucl = [el, *q2_space]
    vertex_datum_raw =\
        [ztv.dimless_vertex_use_psd(q1_eucl, q2_eucl, xis, a) for a in a_list]
    vertex_val_datum = [finite_real_part(x[0]) for x in vertex_datum_raw]
    vertex_err_datum = [finite_real_part(x[1]) for x in vertex_datum_raw]
    vertex_val_data.append(vertex_val_datum)
    vertex_err_data.append(vertex_err_datum)

data = {
    "el_list": el_list,
    "a_list": a_list,
    "vertex_val_data": vertex_val_data,
    "vertex_err_data": vertex_err_data
}

FILENAME = 'data/zero_temp_vertex_data.pkl'
with open(FILENAME, 'wb') as f:
    pickle.dump(data, f)
f.close()
