'''Single interior model solver'''

import params
import eos
import numpy as np
#from rkf45 import *
import time

from scipy.integrate import solve_ivp

class Model:
    '''Model parameters for a given mass. Additional parameters may be specified, otherwise default to params.py values'''

    def __init__(self,mass,P0=None,T0=None,x_w=None):

        if P0 is None:
            self.P0 = params.P_0
        else:
            self.P0 = P0

        if T0 is None:
            self.T0 = params.T_0
        else:
            self.T0 = T0
            
        #choosing step sizes for each component to create mass grid

        self.mass = mass

        if x_w is None:
            self.mass_fractions = np.copy(params.mass_fractions)
        else:
            x_fe = (1.0/3.0)*(1.0-x_w)
            x_si = 1.0 - (x_fe + x_w)
            self.mass_fractions = np.array([x_fe,x_si,x_w])

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
            self.rho_dict[component] = self.eos_data[component].get_eos(self.pressure_grid,np.log10(params.Pad),np.log10(self.T0))

    def find_Rp(self,steps=2.0e4,mode='python'):
        
        solved = False
        Rp_range = np.copy(params.Rp_range)

        while solved is False:
            start = time.time()
            Rp_choice = np.mean(Rp_range)
            #Rp_choice = 1.3663574218750003
            print(Rp_choice)

            #Yp0 = self.ode_sys(self.mass*params.MEarth,np.array([self.P0,Rp_choice*params.REarth]))

            #soln = r8_rkf45(self.ode_sys, 2, np.array([self.P0,Rp_choice*params.REarth]), Yp0, self.mass*params.MEarth, 0, 1.0e-2, 1.0e5, 1)

            #soln = solve_ivp(self.ode_sys, (self.mass*params.MEarth,0), np.array([self.P0,Rp_choice*params.REarth]),max_step=1.0e20)

            #soln = euler(self.ode_sys, np.array([self.P0,Rp_choice*params.REarth]), self.mass*params.MEarth,0.0,2.0e4)

            if mode=='python':
                print('Pre function call: '+str(time.time()-start))
                soln = rk4(self.ode_sys, np.array([self.P0,Rp_choice*params.REarth]), self.mass*params.MEarth,0.0,steps)
                print('Post function call: '+str(time.time()-start))
                final_r = soln[-1][-1]

            import ctypes
            solver = ctypes.CDLL('./solver.so')
            solver.rk4.restype = ctypes.c_double
            
            y0 = np.array([self.P0,Rp_choice*params.REarth])
            eos_c = None
            for comp in params.components:
                if eos_c is None:
                    eos_c = self.rho_dict[comp]
                else:
                    eos_c = np.c_[eos_c, self.rho_dict[comp]]
            if eos_c.shape == 3:
                eos_c = np.c_[eos_c,np.zeros_like(pressure_grid)]
            mass_bds_c = np.delete(self.mass_bds,0)
            if len(mass_bds_c) == 3:
                mass_bds_c = np.append(mass_bds_c,1.0)
            mass_bds_c *= params.MEarth


            #create C type arrays to pass to solver
            c_double_p = ctypes.POINTER(ctypes.c_double)
            y0 = y0.astype(np.double)
            y0_p = y0.ctypes.data_as(c_double_p)
            eos_c = eos_c.astype(np.double)
            eos_p = eos_c.ctypes.data_as(c_double_p)
            mass_bds_c = mass_bds_c.astype(np.double)
            mass_bds_p = mass_bds_c.ctypes.data_as(c_double_p)
            pgrid = self.pressure_grid.astype(np.double)
            pgrid_p = pgrid.ctypes.data_as(c_double_p)
            mass_c = self.mass*params.MEarth
            mass_p = ctypes.c_double(mass_c)#.ctypes.data_as(c_double_p)
            steps_p = ctypes.c_double(steps)

            #np.savetxt('c_test_eos.dat',eos_c)
            #np.savetxt('c_test_pgrid.dat',pgrid)
            #quit()
            if mode == 'c':
                print('Pre function call: '+str(time.time()-start))
                soln = solver.rk4(y0_p, mass_p,ctypes.c_double(0.0),steps_p,mass_bds_p,pgrid_p,eos_p)
                final_r = np.copy(soln)
                print('Post function call: '+str(time.time()-start))

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
            #final_r = soln.y[1,-1]
            #final_m = soln.t[-1]

            #if final_m > 0:
            #    soln = euler(self.ode_sys, np.array([final_p,final_r]), final_m,0.0,5.0e2)
            #    final_r = soln[1][1]
            #    final_m = soln[0]

            #final_r = soln[1][1]
            #final_m = soln[0]

            #final_r = np.copy(soln)
            final_m = 0.0
            print('Final R = '+str(final_r))
            #quit()

            if final_r < 0 or final_m > 0:
                Rp_range[0] = Rp_choice
            elif final_r > 1.0e3:
                Rp_range[1] = Rp_choice
            else:
                print('done')
                print('Rp = '+str(Rp_choice))
                #quit()
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
        lrho = np.interp(np.log10(p),self.pressure_grid,self.rho_dict[comp])
        rho = 10**lrho
        dr_dm = 1.0/(4.0*np.pi*r**2*rho)

        return((np.array([dp_dm,dr_dm]),lrho))

def euler(f, y0, t0, t_end, nsteps):
    t = np.linspace(t0,t_end,nsteps)
    h = (t_end-t0)/nsteps
    y = np.zeros((len(t),len(y0)))
    y[0] = y0
    for i in range(len(t)-1):
        y[i+1] = y[i] + h*f(t[i],y[i])

    return(t[-1],y[-1])

def rk4(f, y0, t0, t_end, nsteps):
    t = np.linspace(t0,t_end,nsteps+1)
    h = (t_end-t0)/nsteps
    y = np.zeros((len(t),len(y0)))
    y[0] = y0
    #print(str(t[0])+'\t'+str(y[0][1]))
    for i in range(len(t)-1):
        k1 = h*f(t[i],y[i])[0]
        k2 = h*f(t[i]+0.5*h,y[i]+0.5*k1)[0]
        k3 = h*f(t[i]+0.5*h,y[i]+0.5*k2)[0]
        k4 = h*f(t[i]+h,y[i]+k3)[0]

        y[i+1] = y[i] + (k1+2*k2+2*k3+k4)*(1.0/6.0)

        #print(str(t[i+1])+'\t'+str(y[i+1][1]))

    return(t[-1],y[-1])
