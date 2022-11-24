#! /usr/bin/env python

from pySecDec.loop_integral import loop_package
import pySecDec as psd
from sympy import sympify

li = psd.loop_integral.LoopIntegralFromPropagators(
    propagators=['k**2 - M3**2', '(k + p2)**2 - M1**2', '(k - p1)**2 - M2**2'],
    loop_momenta=['k'],
    external_momenta=['p1', 'p2'],
    
	replacement_rules=[
        ('p1*p1', 'p1p1'),
        ('p1*p2', 'p1p2'),
        ('p2*p2', 'p2p2'),
        ('M1**2', 'M1M1'),
        ('M2**2', 'M2M2'),
        ('M3**2', 'M3M3')
    ],
           
	regulator='eps',
	dimensionality='3 + 2*eps'
)

kinematics_symbols=['p1p1', 'p2p2', 'p1p2']
mass_symbols=['M1M1', 'M2M2', 'M3M3']

loop_package(

	name='phiCubed_31_space',

	loop_integral=li,

    additional_prefactor=sympify('1/(4*pi)**(3/2+eps)'),  # missing (-1)
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
	normaliz_executable='/home/derek/normaliz-3.9.3/normaliz'

)
