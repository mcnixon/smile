'''Single interior model solver'''

import params
import eos
import numpy as np
from rkf45 import *

from scipy.integrate import solve_ivp

class Model:
    '''Model parameters for a given mass'''

    def __init__(self,mass):

        self.P0 = params.P_0
    
        #choosing step sizes for each component to create mass grid

        self.mass = mass

        self.mass_fractions = np.copy(params.mass_fractions)

        if len(params.components) > len(self.mass_fractions):
            self.mass_fractions = np.append(self.mass_fractions,1.0-np.sum(self.mass_fractions))

        self.mass_bds = np.cumsum(self.mass_fractions)*self.mass
        self.mass_bds = np.insert(self.mass_bds,0,0)

        #get EOS data

        self.pressure_grid = params.Pgrid

        self.eos_data = {}
        self.rho_dict = {}

        for component in params.components:
            self.eos_data[component] = eos.EOS(component)
            self.rho_dict[component] = self.eos_data[component].get_eos(self.pressure_grid,np.log10(params.Pad),np.log10(params.T_0))

    def find_Rp(self):

        solved = False
        Rp_range = np.copy(params.Rp_range)

        while solved is False:
            Rp_choice = np.mean(Rp_range)
            print(Rp_choice)

            Yp0 = self.ode_sys(self.mass*params.MEarth,np.array([self.P0,Rp_choice*params.REarth]))

            #soln = r8_rkf45(self.ode_sys, 2, np.array([self.P0,Rp_choice*params.REarth]), Yp0, self.mass*params.MEarth, 0, 1.0e-2, 1.0e5, 1)

            soln = solve_ivp(self.ode_sys, (self.mass*params.MEarth,0), np.array([self.P0,Rp_choice*params.REarth]),max_step=1.0e20)

            #soln = euler(self.ode_sys, np.array([self.P0,Rp_choice*params.REarth]), self.mass*params.MEarth,0.0,2.0e4)

            #print(soln.t)
            #print(soln.y)

            #import matplotlib.pyplot as plt
            #plt.plot(soln.t,soln.y[1],'.')
            #plt.show()

            #quit()

            #for i in range(1000):

            #    soln = r8_rkf45(self.ode_sys, 2, soln[0], soln[1], soln[2], 0, 1.0e-2, 1.0e85, 1)

            #    print(soln)
            #quit()

            #final_r = soln[0][1]
            #final_m = soln[2]

            #final_p = soln.y[0,-1]
            final_r = soln.y[1,-1]
            final_m = soln.t[-1]

            #if final_m > 0:
            #    soln = euler(self.ode_sys, np.array([final_p,final_r]), final_m,0.0,5.0e2)
            #    final_r = soln[1][1]
            #    final_m = soln[0]

            #final_r = soln[1][1]
            #final_m = soln[0]

            print(final_r,final_m)


            if final_r < 0 or final_m > 0:
                Rp_range[0] = Rp_choice
            elif final_r > 1.0e3:
                Rp_range[1] = Rp_choice
            else:
                print('done')
                print('Rp = '+str(Rp_choice))
                quit()
                Rp_final = Rp_choice
                solved = True
        return Rp_final


    def ode_sys(self,mass,y):

        p = y[0]
        r = y[1]

        dp_dm = -(params.G*mass)/(4.0*np.pi*r**4)
        #check which layer we are in
        #print(self.mass_bds,mass)
        if mass > 0:
        #print(self.mass_bds,mass)
            component_idx = np.argmax(self.mass_bds[np.where(self.mass_bds*params.MEarth<mass)])
        else:
            component_idx = 0
        comp = params.components[component_idx]
        rho = np.interp(np.log10(p),self.pressure_grid,self.rho_dict[comp])
        rho = 10**rho
        dr_dm = 1.0/(4.0*np.pi*r**2*rho)

        return(np.array([dp_dm,dr_dm]))

def euler(f, y0, t0, t_end, nsteps):
    t = np.linspace(t0,t_end,nsteps)
    h = (t_end-t0)/nsteps
    y = np.zeros((len(t),len(y0)))
    y[0] = y0
    for i in range(len(t)-1):
        y[i+1] = y[i] + h*f(t[i],y[i])

    return(t[-1],y[-1])
    
