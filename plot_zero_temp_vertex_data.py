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
fig, ax = plt.subplots()
for p in range(num_plots):
    ax.scatter(a_arr, vertex_val_arr[p])
plt.show()
