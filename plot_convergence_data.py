"""Plot partial sum sequence data with and without zeta-corrections."""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.ticker import ScalarFormatter

data_file = 'data/convergence_data.pkl'
with open(data_file, 'rb') as f:
    data = pickle.load(file=f)
f.close()

a_vals = data["a_vals"]
parsum_vals_arr = data["parsum_vals_arr"]
parsum_errs_arr = data["parsum_errs_arr"]
parsum_vals_zeta_arr = data["parsum_vals_zeta_arr"]
parsum_errs_zeta_arr = data["parsum_errs_zeta_arr"]

nmax_list = [len(parsum_vals_arr[index]) for index in range(len(a_vals))]
nrows = 2
ncols = 2

xlabel = r'$n_{\mathrm{max}}$'
ylabel = r'Truncated $\tilde{\Gamma}_{T}$'

opts = {
    "xlabels": [[None, None], [xlabel, xlabel]],
    "xstart": [[2, 2], [35, 50]],
    "xend": [[10, 10], [101, 201]],
    "xstep": [[1, 1], [10, 25]],
    "ylabels": [[ylabel, None], [ylabel, None]],
    "legend_loc": [['lower right', 'lower right'], ['lower right', 'lower right']],
}

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False,
                         figsize=(6, 3.7), layout='constrained')

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 1))

for a_index in range(len(a_vals)):
    row = a_index // nrows
    col = a_index % ncols

    n_vals = np.arange(nmax_list[a_index])

    axes[row, col].errorbar(x=n_vals[opts["xstart"][row][col]:opts["xend"][row][col]:opts["xstep"][row][col]],
                            y=np.array(parsum_vals_zeta_arr[a_index][opts["xstart"][row][col]:opts["xend"][row][col]:opts["xstep"][row][col]]),
                            yerr=np.array(parsum_errs_zeta_arr[a_index][opts["xstart"][row][col]:opts["xend"][row][col]:opts["xstep"][row][col]]),
                            fmt='.-k', label="zeta-corrected")

    axes[row, col].errorbar(x=n_vals[opts["xstart"][row][col]:opts["xend"][row][col]:opts["xstep"][row][col]],
                            y=np.array(parsum_vals_arr[a_index][opts["xstart"][row][col]:opts["xend"][row][col]:opts["xstep"][row][col]]),
                            yerr=np.array(parsum_errs_arr[a_index][opts["xstart"][row][col]:opts["xend"][row][col]:opts["xstep"][row][col]]),
                            fmt='.--k', label="uncorrected")

    axes[row, col].legend(loc=opts["legend_loc"][row][col])
    axes[row, col].set_title('a = {:}'.format(a_vals[a_index]))
    axes[row, col].set_xlabel(opts["xlabels"][row][col])
    axes[row, col].set_xticks(np.arange(opts["xstart"][row][col],
                                        opts["xend"][row][col],
                                        opts["xstep"][row][col]))

    axes[row, col].yaxis.set_major_formatter(formatter)
    axes[row, col].set_ylabel(opts["ylabels"][row][col])

plt.show()
