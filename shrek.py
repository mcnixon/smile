'''Main script for executing models'''

import numpy as np
import params
import eos
import model

def mr_curve(mode):
    masses = np.arange(params.mass_lower,params.mass_upper,params.mass_step)
    radii = np.zeros_like(masses)
    for i,m in enumerate(masses):
        int_model = model.Model(m,mode)
        print(m)
        radii[i] = int_model.find_Rp()

    return np.c_[masses,radii]

def steps_test():
    int_model = model.Model(0.5)
    steps = np.logspace(2,5,20)
    radii = np.zeros_like(steps)
    for i in range(len(steps)):
        radii[i] = int_model.find_Rp(steps=int(steps[i]))

    return radii
