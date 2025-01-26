'''Main script for executing models'''

import numpy as np
import eos
import model

def get_radius(mass, P0=1.0e5, T0=300.0, Pad=1.0e5, x_si=None, x_w=0.0, x_g=0.0, Rp_lower=0.1, Rp_upper=10.0, pt='isotherm-adiabat', pt_file=None, mixed=False, profiles=False, hhb=False):
    '''Find the radius of a planet.
    
    Output: planet radius in Earth radii (float)

    If 'hhb' = True, returns (planet radius in Earth radii, HHB pressure in Pa, HHB temperature in K) (tuple of floats)

    Inputs:
    mass: planet mass (Earth masses)
    P0: photospheric pressure (Pa)
    T0: photospheric temperature (K)
    Pad: pressure at the radiative-convective boundary (Pa)
    x_si: silicate mass fraction (set to None for Earth-like nucleus)
    x_w: H2O mass fraction
    x_g: H/He mass fraction
    Rp_lower: lower bound for Rp guess (Earth radii)
    Rp_upper: upper bound for Rp guess (Earth radii)
    pt: pressure-temperature profile prescription. 'isotherm-adiabat' uses an isotherm starting at (P0, T0) switching to an adiabat at Pad, 'guillot' uses an analytic profile, 'file' uses the P-T profile set by pt_file
    pt_file: filename of user-specified pressure-temperature profile
    mixed: Boolean. If True, H/He/H2O are mixed throughout. If False, they are differentiated.
    profiles: Boolean. If True, saves mass, radius, temperature, density and pressure profiles for the final model
    hhb: Boolean. If True, returns the pressure and temperature at the H/He/H2O boundary. Only use if mixed = False. This mode calculates the HHB more quickly than running the full model, but the radius may be less accurate.
    '''
    
    #Initialising the model
    int_model = model.Model(mass, P0=P0,T0=T0, Pad=Pad, x_si=x_si, x_w=x_w, x_g=x_g, Rp_lower=Rp_lower, Rp_upper=Rp_upper, pt=pt, pt_file=pt_file, mixed=mixed, profiles=profiles, hhb=hhb)

    #Solve for and return radius
    return int_model.find_Rp()
