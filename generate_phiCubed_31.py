#! /usr/bin/env python

from pySecDec.loop_integral import loop_package
import pySecDec as psd
from sympy import sympify

li = psd.loop_integral.LoopIntegralFromGraph(
	internal_lines=[['m3', [1, 2]], ['m1', [2, 3]], ['m2', [3, 1]]],
	external_lines=[['p1', 1], ['p2', 2], ['p3', 3]],
    powerlist=[1, 1, 1],

	replacement_rules=[
        ('p1*p1', 'p1p1'),
        ('p1*p2', 'p1p2'),
        ('p1*p3', '-p1p1 - p1p2'),
        ('p2*p2', 'p2p2'),
        ('p2*p3', '-p1p2 - p2p2'),
        ('p3*p3', 'p1p1 + p2p2 + 2*p1p2'),
        ('m1**2', 'm1m1'),
        ('m2**2', 'm2m2'),
        ('m3**2', 'm3m3')
    ],
           
	regulator='eps',
	dimensionality='4 + 2*eps'
)

kinematics_symbols=['p1p1', 'p2p2', 'p1p2']
mass_symbols=['m1m1', 'm2m2', 'm3m3']

loop_package(

	name='phiCubed_31',

	loop_integral=li,

    additional_prefactor=sympify('1/(4*pi)**(2+eps)'),  # missing factor of I
    complex_parameters=kinematics_symbols + mass_symbols,

	# the highest order of the final epsilon expansion --> change this value to whatever you think is appropriate
	requested_order=0,

    contour_deformation=True,
    
	# the optimization level to use in FORM (can be 0, 1, 2, 3, 4)
	form_optimization_level=2,

	# the WorkSpace parameter for FORM
	form_work_space='2G',

	# the method to be used for the sector decomposition
	# valid values are ``iterative`` or ``geometric`` or ``geometric_ku``
	decomposition_method='geometric',
	# if you choose ``geometric[_ku]`` and 'normaliz' is not in your
	# $PATH, you can set the path to the 'normaliz' command-line
	# executable here
	normaliz_executable='/usr/bin/normaliz'

)
