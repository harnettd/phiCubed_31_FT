import matplotlib.pyplot as plt
import numpy as np
import pickle

filename = 'data/finite_temp_vertex_momentum_data.pkl'
with open(filename, 'rb') as f:
    data = pickle.load(file=f)
f.close()

a_arr = np.array(data['a_list'])
qx_arr = np.array(data['qx_list'])
zero_temp_val_arr = np.array(data['zero_temp_val_matrix'])
zero_temp_err_arr = np.array(data['zero_temp_err_matrix'])
finite_temp_val_arr = np.array(data['finite_temp_val_matrix'])
finite_temp_err_arr = np.array(data['finite_temp_err_matrix'])

correction_val_arr = finite_temp_val_arr - zero_temp_val_arr
correction_err_arr = finite_temp_err_arr + zero_temp_err_arr

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

fig, ax =\
    plt.subplots(nrows=2, ncols=2,
                 figsize=(6, 3.7), layout='constrained')


def add_plot(row, col, index):
    ax[row, col].errorbar(x=qx_arr, y=correction_val_arr[index],
                          yerr=correction_err_arr[index],
                          fmt=plot_options[index]['format'],
                          label=r'$a=$' + '{:4.2f}'.format(a_arr[index]))
    ax[row, col].set_title(r'$a$ = {:4.1f}'.format(a_arr[index]))
    ax[row, col].set_xlabel(plot_options[index]['x_label'])
    ax[row, col].set_ylabel(plot_options[index]['y_label'])
    # ax[row, col].legend(loc=plot_options[index]['legend_loc'])


add_plot(0, 0, 0)
add_plot(0, 1, 1)
add_plot(1, 0, 2)
add_plot(1, 1, 3)

plt.show()
