import matplotlib.pyplot as plt
import numpy as np
import pickle

data_file = 'data/finite_temp_vertex_correction_data.pkl'
with open(data_file, 'rb') as f:
    data = pickle.load(file=f)
f.close()

el_list = data['el_list']
a_arr = np.array(data['a_list'])
zero_temp_val_arr = np.array(data['zero_temp_val_matrix'])
zero_temp_err_arr = np.array(data['zero_temp_err_matrix'])
finite_temp_val_arr = np.array(data['finite_temp_val_matrix'])
finite_temp_err_arr = np.array(data['finite_temp_err_matrix'])

correction_val_arr = finite_temp_val_arr - zero_temp_val_arr
correction_err_arr = finite_temp_err_arr + zero_temp_err_arr

params = {
    0: [-1, '.-k', 'upper right', None, r'$\tilde{\Gamma}_T - \tilde{\Gamma}_0$'],
    1: [-1, '.--k', 'upper right', None, None],
    2: [-1, '.-.k', 'lower right', r'$a$', r'$\tilde{\Gamma}_T - \tilde{\Gamma}_0$'],
    3: [-1, '.:k', 'lower right', r'$a$', None],
}

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False,
                       figsize=(6, 3.7), layout='constrained')


def add_plot(row, col, el_index):
    ax[row, col].errorbar(x=a_arr[:params[el_index][0]],
                          y=correction_val_arr[el_index, :params[el_index][0]],
                          yerr=correction_err_arr[el_index, :params[el_index][0]],
                          fmt=params[el_index][1],
                          label=r'$\ell =$ {}'.format(el_index))
    ax[row, col].set_title(r'$\ell =$ {}'.format(el_index))
    # ax[row, col].legend(loc=params[el_index][2])
    ax[row, col].set_xlabel(params[el_index][3])
    ax[row, col].set_xticks(np.arange(0, 12.5, 2.5))
    ax[row, col].set_ylabel(params[el_index][4])
    if row == 0 and col == 0:
        ax[row, col].set_yscale('log')

add_plot(0, 0, 0)
add_plot(0, 1, 1)
add_plot(1, 0, 2)
add_plot(1, 1, 3)

plt.show()
