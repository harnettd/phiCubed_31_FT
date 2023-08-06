"""Write partial sum sequence data with and without zeta-corrections."""

import pickle

from finite_real_part import finite_real_part
import finite_temperature_vertex as ftv

el_1 = 1
q1 = 0, 0, 0
el_2 = 1
q2 = 0, 0, 0
xis = 1, 0.5, 0
a_vals = 0.25, 1, 10, 25
nmax_list = [15, 21, 101, 251]

q1_eucl = [el_1, *q1]
q2_eucl = [el_2, *q2]

parsum_vals_arr = []
parsum_errs_arr = []
parsum_vals_zeta_arr = []
parsum_errs_zeta_arr = []


def get_finite_real_parts(seq):
    return [finite_real_part(s) for s in seq]


for a in a_vals:
    nmax = nmax_list[a_vals.index(a)]

    parsum_data =\
        ftv.dimless_vertex_partial_sum_sequence(q1_eucl, q2_eucl, xis, a, nmax,
                                                add_zeta_correction=False)

    parsum_vals_arr.append(get_finite_real_parts(parsum_data[0]))
    parsum_errs_arr.append(get_finite_real_parts(parsum_data[1]))

    parsum_data_zeta =\
        ftv.dimless_vertex_partial_sum_sequence(q1_eucl, q2_eucl, xis, a, nmax,
                                                add_zeta_correction=True)

    parsum_vals_zeta_arr.append(get_finite_real_parts(parsum_data_zeta[0]))
    parsum_errs_zeta_arr.append(get_finite_real_parts(parsum_data_zeta[1]))

data = {
    "a_vals": a_vals,
    "parsum_vals_arr": parsum_vals_arr,
    "parsum_errs_arr": parsum_errs_arr,
    "parsum_vals_zeta_arr": parsum_vals_zeta_arr,
    "parsum_errs_zeta_arr": parsum_errs_zeta_arr
}

data_file = 'data/convergence_data.pkl'
with open(data_file, 'wb') as f:
    pickle.dump(data, file=f)
f.close()
