import pickle
import matplotlib.pyplot as plt
import numpy as np

filename = 'data/zero_temp_vertex_momentum_data.pkl'
with open(filename, 'rb') as f:
    data = pickle.load(f)
f.close()

a_arr = np.array(data['a_list'])
qx_arr = np.array(data['qx_list'])
vertex_val_arr = np.array(data['vertex_val_data'])
vertex_err_arr = np.array(data['vertex_err_data'])

colors = ['blue', 'orange', 'green', 'red']
fig, ax = plt.subplots(layout='constrained', figsize=(6, 3.71))
for index in range(len(a_arr)):
    ax.errorbar(x=qx_arr, y=vertex_val_arr[index], yerr=vertex_err_arr[index],
                fmt='.-', color=colors[index],
                label=r'$a=$' + '{0:4.2f}'.format(a_arr[index]))
    ax.legend(loc='upper right')
    ax.set_xlabel(r'$q^x$')
    ax.set_ylabel(r'$\tilde{\Gamma}_0$')
plt.show()