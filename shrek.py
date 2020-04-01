'''Main script for executing models'''

import numpy as np
import params
import eos
import model

def mr_curve(P0=None, T0=None, x_w=None,steps=1.0e3):
    masses = np.arange(params.mass_lower,params.mass_upper,params.mass_step)
    radii = np.zeros_like(masses)
    for i,m in enumerate(masses):
        int_model = model.Model(m,P0,T0,x_w)
        print(m)
        radii[i] = int_model.find_Rp(steps=steps)

    return np.c_[masses,radii]

def single_model(mass, P0=None, T0=None, x_w=None,steps=2.0e4):
    '''Model a single planet.
    Inputs:
    - mass (Earth masses)
    - Ps, surface pressure (Pa)
    - Ts, surface temp (K)
    - x_w, H2O mass fraction
    '''

    int_model = model.Model(mass,P0,T0,x_w)
    
    return int_model.find_Rp(steps=steps)

def steps_test():
    int_model = model.Model(0.5)
    steps = np.logspace(2,5,20)
    radii = np.zeros_like(steps)
    for i in range(len(steps)):
        radii[i] = int_model.find_Rp(steps=int(steps[i]))

    return radii
