"""Plot Siyuan's two-point function data versus a"""
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("./data/versus-a.csv", delimiter=",")
end = -3
step = 1
opts = {
    "fmt": [".-k", ".--k", ".-.k", ".:k"],
    "ell": [1, 2, 3, 4],
}

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.7),
                       layout='constrained')
for p in range(4):
    data_row = 3 * p
    ax.errorbar(x=data[data_row, :end:step], y=data[data_row + 1, :end:step],
                yerr=data[data_row + 2, :end:step], fmt=opts["fmt"][p],
                label=r"$\ell = {:}$".format(opts["ell"][p]))
ax.set_ylabel(r"$\tilde{\Pi}_s$")
ax.set_xlabel(r"$a$")
ax.legend(loc="upper right")
plt.show()
