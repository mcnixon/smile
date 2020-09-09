'''Main script for executing models'''

import numpy as np
import params
import eos
import model

def rogers_curve(P0=None, T0=None, x_w=None,x_g=None,mode='adaptive_new'):
    masses = np.arange(params.mass_lower,params.mass_upper,params.mass_step)
    radii = np.zeros_like(masses)
    for i,m in enumerate(masses):
        int_model = model.Model(m,P0,T0,x_w,x_g,mode,profiles=True)
        print(m)
        rad = int_model.find_Rp()
        j=0
        while rad == 'failed':
            j += 1
            masses[i] += 0.01*j
            int_model = model.Model(m+0.01*j,P0,T0,x_w,x_g,mode,profiles=True)
            rad = int_model.find_Rp()
        radii[i] = rad

    return np.c_[masses,radii]

def mr_curve(P0=None, T0=None, x_w=None,x_g=None,mode='adaptive_new',Pad=None,phases=False):
    masses = np.arange(params.mass_lower,params.mass_upper,params.mass_step)
    radii = np.zeros_like(masses)
    for i,m in enumerate(masses):
        int_model = model.Model(m,P0,T0,Pad,x_w,x_g,mode,profiles=True,phases=phases)
        print(m)
        radii[i] = int_model.find_Rp()

    return np.c_[masses,radii]

def get_r_liquid(mass,x_g,T0,mode='adaptive_new',phases=False):
    x_w = None
    P0 = 1.0e4
    Pad = 1.7e6
    int_model = model.Model(mass,P0,T0,Pad,x_w,x_g,mode,profiles=True,phases=phases,liquid=True,hhb=True)
    return int_model.find_Rp()

    return np.c_[masses,radii]

def get_radius(mass,P0=None, T0=None, Pad=None, x_w=None,x_g=None,mode='adaptive_new',profiles=True,phases=False):
    int_model = model.Model(mass,P0,T0,Pad,x_w,x_g,mode,profiles,phases=phases)
    return int_model.find_Rp()

def get_radius_hhb(mass,T0,x_g):
    int_model = model.Model(mass,P0=1.0e4,T0=T0,Pad=1.7e6,x_w=None,x_g=x_g,mode='adaptive_new',profiles=True,hhb=True)
    return int_model.find_Rp()

def steps_test():
    int_model = model.Model(0.5)
    steps = np.logspace(2,5,20)
    radii = np.zeros_like(steps)
    for i in range(len(steps)):
        radii[i] = int_model.find_Rp(steps=int(steps[i]))

    return radii
