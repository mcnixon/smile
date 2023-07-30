'''Parameters for interior model'''

import numpy as np
import os

#MR curve settings (MEarth)
mass_upper = 3.05
mass_lower = 1.0
mass_step = 0.1

pt = 'isotherm-adiabat'

#mass fractions: Fe, MgPv, H2O, remainder in H/He
mass_fractions = np.array([0.233,0.467,0.3])
#mass_fractions = np.array([0.0,0.0,0.0])

P_0 = 1.0e4 #surface pressure (Pa)
T_0 = 300.0 #surface temperature (K)

T_file = None#'steam_gao_f.txt'

#P_ad_hhe = 1.0e9 #pressure to switch to adiabatic H/He profile
#Pad = 1.0e5 #pressure to switch to adiabatic temperature profile
#Pad = 2.206e7 #pressure to switch to adiabatic temperature profile
Pad = 1.0e4

#EOS information

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

eos_files = {}

eos_path = os.getenv("eos_path")

eos_files['fe'] = eos_path+'/fe_eos.dat'
eos_files['mgpv'] = eos_path+'/mgpv_eos.dat'
eos_files['h2o'] = eos_path+'/h2o_eos.hdf5'
eos_files['hhe'] = eos_path+'/hhe_eos.hdf5'

lv_file = eos_path+'/liquid_vapour_bd.txt'

entropy_files = {}
entropy_files['hhe'] = eos_path+'/hhe_entropy.hdf5'
entropy_files['h2o'] = eos_path+'/h2o_entropy.hdf5'
