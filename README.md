# phiCubed_31_FT
Compute the three-point, one-loop correlation function in phi^3 at finite temperature using pySecDec. 

## Setup
Install pySecDec.
Set the parameter normaliz_executable in both generate_phiCubed_31.py and generate_phiCubed_space_31.py.
Then, at the command line, enter

    python generate_phiCubed_31.py
    make -C phiCubed_31
    python generate_phiCubed_space_31.py
    make -C phiCubed_space_31

## Usage
At a Python prompt (such as in a Jupyter notebook), enter

    import correlator as corr

Then, to compute the zero temperature correlator, either enter

    corr.zero_temp_use_psd(p1, p2, mass_1, mass_2, mass_3)

to use pySecDec, or enter

    corr.zero_temp_use_trap(p1, p2, mass_1, mass_2, mass_3, k0_eucl_max, num_grid_pts)

to use the trapezoid rule for the temporal integral.

To compute the finite temperature correlator, enter

    corr.finite_temp(p1, p2, mass_1, mass_2, mass_3, beta, n_min, n_max)

where (in line with Python convention) n_min is included but n_max is excluded.

To compute the spatial integral directly, first enter

    import spatial_integral as spint

Then, enter

    spint.use_psd(p1_space, p2_space, Delta_1, Delta_2, Delta_3)

to use pySecDec, or enter

    spint.use_tplquad(p1_space, p2_space, Delta_1, Delta_2, Delta_3)

to use tplquad, a numerical triple integrator from scipy.integrate. 
