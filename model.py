'''Single interior model solver'''

import params
import eos
import numpy as np

def find_Rp(mass):
    solved = False
    P_0 = params.P_0
    Rp_range = np.copy(params.Rp_range)

    while solved is False:
        Rp_choice = np.mean(Rp_range)

        soln = euler(mass*params.MEarth,Rp_choice*params.REarth)

        final_r = soln[0]
        final_m = soln[1]

        if final_r < 0:# or final_m > 0:
            Rp_range[0] = Rp_choice
            print(Rp_choice)
        elif final_r > 1.0e3:
            Rp_range[1] = Rp_choice
            print(Rp_choice)
        else:
            print('done')
            Rp_final = Rp_choice
            solved = True
    return Rp_final

def euler(mass,R0):

    P0 = params.P_0
    
    #choosing step sizes for each component to create mass grid

    mass_fractions = np.copy(params.mass_fractions)

    if len(params.components) > len(mass_fractions):
        mass_fractions = np.append(mass_fractions,1.0-np.sum(mass_fractions))

    mass_bds = np.cumsum(mass_fractions)*mass
    mass_bds = np.insert(mass_bds,0,0)

    mass_steps_dict = {}
    for i,component in enumerate(params.components):
        if component is 'hhe':
            nsteps = 2.0e4
        if component is 'h2o':
            nsteps = 2.0e4
        if component is 'mgpv':
            nsteps = 500
        if component is 'fe':
            nsteps = 500

        if mass_fractions[i] > 0:
            mass_steps_dict[i] = np.linspace(mass_bds[i],mass_bds[i+1],nsteps)
        else:
            mass_steps_dict[i] = np.array([])


    mass_steps = np.concatenate([mass_steps_dict[i] for i in range(len(mass_fractions))])
    mass_steps = mass_steps[::-1]

    #get EOS data

    pressure_grid = params.Pgrid

    eos_data = {}
    rho_dict = {}

    for component in params.components:
        eos_data[component] = eos.EOS(component)
        rho_dict[component] = eos_data[component].get_eos(pressure_grid,params.Pad,np.log10(params.T_0))

    #initialise pressure and radius arrays
    p = np.zeros_like(mass_steps)
    r = np.zeros_like(mass_steps)

    p[0] = P0
    r[0] = R0

    #think very hard about logs

    for i,m in enumerate(mass_steps[:-1]):
        dp_dm = -(params.G*m)/(4.0*np.pi*r[i]**4)
        #check which layer we are in
        component_idx = np.argmax(mass_bds[np.where(mass_bds<m)])
        comp = params.components[component_idx]
        rho = np.interp(np.log10(p[i]),pressure_grid,rho_dict[comp])
        rho = 10**rho
        dr_dm = 1.0/(4.0*np.pi*r[i]**2*rho)

        step = mass_steps[i+1] - m
        p[i+1] = p[i]+step*dp_dm
        r[i+1] = r[i]+step*dr_dm

    return(r[-1],mass_steps[-1])
