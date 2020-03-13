'''Main script for executing models'''

import numpy as np
import params
import eos
import model

def mr_curve():
    masses = np.arange(params.mass_lower,params.mass_upper,params.mass_step)
    radii = np.zeros_like(masses)
    for i,m in enumerate(masses):
        print(m)
        radii[i] = model.find_Rp(m)

    return np.c_[masses,radii]
