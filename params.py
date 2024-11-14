'''Parameters for interior model'''

import numpy as np

#mass fractions: Fe, MgPv, H2O, remainder in H/He
mass_fractions = np.array([0.233,0.467,0.3])
#mass_fractions = np.array([0.0,0.0,0.0])

components = ['fe','mgpv','h2o','hhe']

#Pgrid = np.arange(4,22,0.01)#log P (Pa)
Pgrid = np.arange(2,15,0.01)

Rp_range = np.array([0.1,10.0]) #initial guesses for Rp (REarth)
#Rp_range = np.array([1.4,3.2]) #initial guesses for Rp (REarth)

#constants (SI units)
MEarth = 5.972e24
REarth = 6.371e6 #mean
G = 6.67e-11
sigma_SB = 5.6704e-08
