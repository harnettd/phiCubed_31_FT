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

num_plots = len(el_arr)
scale = 1e3
fig, ax = plt.subplots()
for p in range(num_plots):
    ax.errorbar(x=a_arr, y=vertex_val_arr[p] * scale, 
                yerr=vertex_err_arr[p] * scale,
                fmt=',-',
                label=r'$\ell=$ {}'.format(el_arr[p]))
    ax.set_xlabel(r'$a$')
    h_label = r'$\tilde{\Gamma}_{0} \times$ ' + str(int(scale))
    ax.set_ylabel(h_label)
    ax.legend(loc='upper left')
plt.show()
