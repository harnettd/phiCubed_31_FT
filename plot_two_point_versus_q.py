"""Plot Siyuan's two-point function data versus q"""
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("./data/versus-q.csv", delimiter=",")
end = -1
step = 2
opts = {
    "fmt": [".-k", ".--k", ".-.k", ".:k"],
    "a": [0.1, 0.2, 0.3, 0.4],
}

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.7),
                       layout='constrained')
for p in range(4):
    data_row = 3 * p
    ax.errorbar(x=data[data_row, :end:step],
                y=data[data_row + 1, :end:step],
                yerr=data[data_row + 2, :end:step],
                fmt=opts["fmt"][p], label=r"$a = {:}$".format(opts["a"][p]))
ax.set_ylabel(r"$\tilde{\Pi}_s$")
ax.set_xlabel(r"$q^1$")
ax.legend(loc="upper right")
plt.show()
