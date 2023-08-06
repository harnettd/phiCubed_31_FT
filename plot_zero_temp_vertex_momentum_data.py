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

formats = ['.-k', '.--k', '.-.k', '.:k']
fig, ax = plt.subplots(layout='constrained', figsize=(6, 3.71))
for index in range(len(a_arr)):
    ax.errorbar(x=qx_arr, y=vertex_val_arr[index], yerr=vertex_err_arr[index],
                fmt=formats[index],
                label=r'$a=$' + '{0:4.1f}'.format(a_arr[index]))
    ax.legend(loc='upper right')
    ax.set_xlabel(r'$q^1$')
    ax.set_ylabel(r'$\tilde{\Gamma}_0$')
plt.show()