# phiCubed_31_FT
Compute the three-point, one-loop correlation function in phi^3^ at finite temperature using pySecDec. 

## Setup
Install pySecDec.
Set `normaliz_executable` in generate_phiCubed_31.py and generate_phiCubed_space_31.py.
Then,

    python generate_phiCubed_31.py
    make -C phiCubed_31
    python generate_phiCubed_space_31.py
    make -C phiCubed_space_31
