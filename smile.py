'''Main script for executing models'''

import numpy as np
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
