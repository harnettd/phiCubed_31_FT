"""Plot the zero-temperature vertex function."""
import pickle
import matplotlib.pyplot as plt
import numpy as np

filename = 'data/zero_temp_vertex_data.pkl'
with open(filename, 'rb') as f:
    data = pickle.load(f)
f.close()

el_arr = np.array(data['el_list'])
a_arr = np.array(data['a_list'])
vertex_val_arr = np.array(data['vertex_val_data'])
vertex_err_arr = np.array(data['vertex_err_data'])

colors = ['blue', 'orange', 'green', 'red']

fig, ax = plt.subplots(layout='constrained', figsize=(6, 3.71))
for p in range(len(el_arr) - 1):
    ax.errorbar(x=a_arr, y=vertex_val_arr[p],
                yerr=vertex_err_arr[p],
                fmt='.-', color=colors[p],
                label=r'$\ell=$ {}'.format(el_arr[p]))
    ax.set_xlabel(r'$a$')
    ax.set_ylabel(r'$\tilde{\Gamma}_{0}$')
    ax.legend(loc='lower right')
plt.show()
