'''Main script for executing models'''

import numpy as np
import params
import eos
import model

def get_radius(mass, P0=1.0e5, T0=300.0, Pad=1.0e5, x_si=None, x_w=0.0, x_g=0.0, pt='isotherm-adiabat', pt_file=None, mixed=False, profiles=False, phases=False):
    '''Find the radius of a planet.
    
    Output: planet radius in Earth radii

    Inputs:
    mass: planet mass (Earth masses)
    P0: photospheric pressure (Pa)
    T0: photospheric temperature (K)
    Pad: pressure at the radiative-convective boundary (Pa)
    x_si: silicate mass fraction (set to None for Earth-like nucleus)
    x_w: H2O mass fraction
    x_g: H/He mass fraction
    pt: pressure-temperature profile prescription. 'isotherm-adiabat' uses an isotherm starting at (P0, T0) switching to an adiabat at Pad, 'guillot' uses an analytic profile, 'file' uses the P-T profile set by pt_file
    pt_file: filename of user-specified pressure-temperature profile
    mixed: Boolean. If True, H/He/H2O are mixed throughout. If False, they are differentiated.
    profiles: Boolean. If True, saves mass, radius, temperature, density and pressure profiles for the final model
    phases: honestly I can't remember what this does, just leave False
    '''
    
    #Initialising the model
    int_model = model.Model(mass, P0=P0,T0=T0, Pad=Pad, x_si=x_si, x_w=x_w, x_g=x_g, pt=pt, pt_file=pt_file, mixed=mixed, profiles=profiles, phases=phases)

    #Solve for and return radius
    return int_model.find_Rp()

def mr_curve(mass_lower=None, mass_upper=None, mass_step=0.5, P0=None, T0=None, Pad=None, x_si=None, x_w=0.0, x_g=0.0, mixed=False, profiles=False, phases=False):
    '''Generate a mass-radius curve.

    Output: array in the format [masses,radii] (in Earth units)

    Inputs:
    mass_lower: lowest mass in array (Earth masses)
    mass_upper: highest mass in array (Earth masses)
    mass_step: mass step size (Earth masses)
    P0: photospheric pressure (Pa)
    T0: photospheric temperature (K)
    Pad: pressure at the radiative-convective boundary (Pa)
    x_w: H2O mass fraction
    x_g: H/He mass fraction
    mixed: Boolean. If True, H/He/H2O are mixed throughout. If False, they are differentiated.
    profiles: Boolean. If True, saves mass, radius, temperature, density and pressure profiles for the final model
    phases: honestly I can't remember what this does, just leave False
    '''

    #Generate mass array and empty radius array
    if mass_lower is None or mass_upper is None:
        masses = np.arange(params.mass_lower,params.mass_upper+1-e4,mass_step)
    else:
        masses = np.arange(mass_lower,mass_upper+1e-4,mass_step)
    radii = np.zeros_like(masses)

    #Solve for the radius at each mass
    for i,m in enumerate(masses):
        int_model = model.Model(m,P0=P0,T0=T0,Pad=Pad,x_si=x_si,x_w=x_w,x_g=x_g,mixed=mixed,profiles=profiles,phases=phases)
        print(m)
        radii[i] = int_model.find_Rp()

    return np.c_[masses,radii]

#Ignore everything below here for now!

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

def get_r_liquid(mass,x_g,T0,mode='adaptive_new',phases=False):
    x_w = None
    P0 = 1.0e4
    Pad = 1.7e6
    int_model = model.Model(mass,P0,T0,Pad,x_w,x_g,mode,profiles=True,phases=phases,liquid=True,hhb=True)
    return int_model.find_Rp()

    return np.c_[masses,radii]

def get_radius_hhb(mass,T0,x_g):
    int_model = model.Model(mass,P0=1.0e4,T0=T0,Pad=1.7e6,x_w=None,x_g=x_g,mode='adaptive_new',profiles=False,hhb=True)
    return int_model.find_Rp()

def steps_test():
    int_model = model.Model(0.5)
    steps = np.logspace(2,5,20)
    radii = np.zeros_like(steps)
    for i in range(len(steps)):
        radii[i] = int_model.find_Rp(steps=int(steps[i]))

    return radii
