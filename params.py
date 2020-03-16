'''Parameters for interior model'''

import numpy as np

#MR curve settings (MEarth)
mass_upper = 10
mass_lower = 0.5
mass_step = 0.5

#mass fractions: Fe, MgPv, H2O, remainder in H/He
mass_fractions = np.array([0.233,0.467,0.3])
#mass_fractions = np.array([0.0,0.0,0.0])

P_0 = 1.0e7 #surface pressure (Pa)
T_0 = 300.0 #surface temperature (K)

#P_ad_hhe = 1.0e9 #pressure to switch to adiabatic H/He profile
Pad = 1.0e9 #pressure to switch to adiabatic temperature profile

#EOS information

components = ['fe','mgpv','h2o','hhe']

Pgrid = np.arange(4,22,0.01)#log P (Pa)

Rp_range = np.array([0.1,4.0]) #initial guesses for Rp (REarth)

#constants (SI units)
MEarth = 5.972e24
REarth = 6.371e6 #mean
G = 6.67e-11

eos_files = {}

#loads local files if using Mac OS
import sys
if sys.platform == 'darwin':
    eos_files['fe'] = '../eos_data/fe_eos.dat'
    eos_files['mgpv'] = '../eos_data/mgpv_eos.dat'
    eos_files['h2o'] = '../eos_data/h2o_eos.hdf5'
    eos_files['hhe'] = '../eos_data/hhe_eos.hdf5'

else:
    eos_files['fe'] = '/data/mn442/sunrise/shrek/eos/fe_eos.dat'
    eos_files['mgpv'] = '/data/mn442/sunrise/shrek/eos/mgpv_eos.dat'
    eos_files['h2o'] = '/data/mn442/sunrise/shrek/eos/h2o_eos.hdf5'
    eos_files['hhe'] = '/data/mn442/sunrise/shrek/eos/hhe_eos.hdf5'

