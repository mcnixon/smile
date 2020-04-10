'''Single interior model solver'''

import params
import eos
import numpy as np
#from rkf45 import *

from scipy.integrate import solve_ivp

class Model:
    '''Model parameters for a given mass'''

    def __init__(self,mass,P0=None,T0=None,x_w=None,mode='adaptive',profiles=False):

        self.mode = mode
        self.profiles = profiles
        self.x_w = x_w
        
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
        if self.profiles:
            self.T_dict = {}

        for component in params.components:
            self.eos_data[component] = eos.EOS(component)
            rho_T = self.eos_data[component].get_eos(self.pressure_grid,np.log10(params.Pad),np.log10(self.T0))
            self.rho_dict[component] = rho_T[0]
            if self.profiles:
                self.T_dict[component] = rho_T[1]

        #find H2O layer step size
        #R_c = 0.6*params.REarth
        #dlrho_dlP = np.gradient(self.rho_dict['h2o'],self.pressure_grid[1]-self.pressure_grid[0])
        #import matplotlib.pyplot as plt
        #plt.plot(self.pressure_grid,self.rho_dict['h2o'])
        #plt.show()
        #dlrho_dlP_max = 0.07#1.0#np.amax(dlrho_dlP)
        #dlrho_max = 0.1
        #print(dlrho_max,self.P0,R_c,self.mass*params.MEarth,dlrho_dlP_max)
        #self.dm_h2o = -dlrho_max*4.0*np.pi*self.P0*R_c**4/(params.G*self.mass*params.MEarth*dlrho_dlP_max)
        #print(self.dm_h2o)
        #quit()

    def find_Rp(self,steps=8.0e4):

        solved = False
        Rp_range = np.copy(params.Rp_range)
        self.mass_profile = None

        while solved is False:
            Rp_choice = np.mean(Rp_range)
            #Rp_choice = 1.3075939953518174#1.3075939953447233
            #Rp_choice = 1.0
            print(Rp_choice)

            #Yp0 = self.ode_sys(self.mass*params.MEarth,np.array([self.P0,Rp_choice*params.REarth]))

            #soln = r8_rkf45(self.ode_sys, 2, np.array([self.P0,Rp_choice*params.REarth]), Yp0, self.mass*params.MEarth, 0, 1.0e-2, 1.0e5, 1)

            #soln = solve_ivp(self.ode_sys, (self.mass*params.MEarth,0), np.array([self.P0,Rp_choice*params.REarth]),max_step=1.0e20)

            #soln = euler(self.ode_sys, np.array([self.P0,Rp_choice*params.REarth]), self.mass*params.MEarth,0.0,2.0e4)
            if self.mode == 'adaptive':
                soln = self.rk4_a(self.ode_sys, np.array([self.P0,Rp_choice*params.REarth]), self.mass*params.MEarth,0.0)

            if self.mode == 'adaptive_new':
                if self.mass_profile is None:
                    soln_init = self.rk4_a(self.ode_sys, np.array([self.P0,4.0*params.REarth]),self.mass*params.MEarth,0.0)
                    continue
                else:
                    soln = self.rk4(self.ode_sys, np.array([self.P0,Rp_choice*params.REarth]),self.mass*params.MEarth,0.0)

            #else:
            #    soln = rk4(self.ode_sys, np.array([self.P0,Rp_choice*params.REarth]), self.mass*params.MEarth,0.0,steps)

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

            final_r = soln[1][1]
            final_m = soln[0]

            print(final_r,final_m)
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

                #if self.profiles:
                    #np.savetxt('../mr_out/profiles_M'+str(self.mass)+'_P'+str(self.P0)+'_T'+str(self.T0)+'_xw'+str(self.x_w)+'.out',np.c_[self.mass_profile,self.radius_profile,self.pressure_profile,self.temperature_profile,self.density_profile,self.component_profile])
                
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

        if self.mode == 'adaptive' or self.mode == 'adaptive_new':
            return((np.array([dp_dm,dr_dm]),lrho))
        else:
            return np.array([dp_dm,dr_dm])

    def rk4_a(self,f, y0, t0, t_end):
        y = np.copy(y0)
        t = np.copy(t0)
        h = -t0*1.0e-3
        h_min = -t0*1.0e-5
        h_max = -t0*1.0e-2

        if self.profiles:
            self.mass_profile = np.array([t])
            self.pressure_profile = np.array([np.log10(y[0])])
            self.radius_profile = np.array([y[1]])
            self.density_profile = None
            self.temperature_profile = None
            self.component_profile = None
        
        no_increase = False
        force_step = False

        while np.logical_and(t > 0,y[1] > 0):

            if np.abs(h) > t:
                h = -t

            if t > 0:
                component_idx = np.argmax(self.mass_bds[np.where(self.mass_bds*params.MEarth<t)])
            else:
                component_idx = 0
            
            k1 = h*f(t,y)[0]
            k2 = h*f(t+0.5*h,y+0.5*k1)[0]
            k3 = h*f(t+0.5*h,y+0.5*k2)[0]
            k4 = h*f(t+h,y+k3)[0]

            lrho = f(t,y)[1]
            if self.profiles:
                if self.density_profile is None:
                    self.density_profile = np.array([lrho])
                    self.temperature_profile = np.interp(np.log10(y[0]),self.pressure_grid,self.T_dict[params.components[component_idx]])
                    self.component_profile = component_idx

            delta_y = (k1+2*k2+2*k3+k4)*(1.0/6.0)

            y_new = y + delta_y
            t_new = t + h

            if t_new > 0:
                component_idx_new = np.argmax(self.mass_bds[np.where(self.mass_bds*params.MEarth<t_new)])
            else:
                component_idx_new = 0

            lrho_new = f(t_new,y_new)[1]
            if self.profiles:
                lT_new = np.interp(np.log10(y_new[0]),self.pressure_grid,self.T_dict[params.components[component_idx_new]])

            delta_rho = np.abs(lrho_new-lrho)

            delta_rho_max = 1.0e-1
            delta_rho_min = 1.0e-3

            last = False
            step = 'not taken'
            if force_step:
                step = 'step taken'
                #print('boundary '+step)
                y = y_new
                t = t_new
                no_increase = False
                force_step = False
            elif component_idx ==0:
                step = 'step adjusted'
                h = -self.mass_bds[component_idx+1]*params.MEarth*0.001
                force_step = True
            elif component_idx_new != component_idx:
                bd_edge = self.mass_bds[component_idx]*params.MEarth+t0*5.0e-6
                if t <= bd_edge:
                    h = -t0*1.0e-5
                    force_step = True
                    #print('step fixed for boundary cross')
                else:
                    h = bd_edge - t
                    #print(h,bd_edge,t)
                    #print('boundary adjustment')
                    no_increase = True
            elif t_new == 0: #line up with elif
                no_increase = True
                if y_new[1] < 0:
                    #print('final overshot')
                    h *= 0.5
                else:
                    step = 'step taken'
                    #print('final '+step)
                    y = y_new
                    t = t_new
            #elif t < 0.01*t0:
            #    h = -0.0001*t0
            #    force_step = True
            elif np.abs(delta_y[1]) > 0.002*y0[1] and y_new[1] > 0:
                #print('singularity')
                no_increase = True
                h *= 0.5
            elif delta_rho > delta_rho_max and np.abs(h)>np.abs(h_min):
                #print('d rho too large')
                h *= 0.8
            elif no_increase is False and delta_rho < delta_rho_min and np.abs(h)<np.abs(h_max):
                #print('d rho too small')
                h /= 0.8
            else:
                step = 'step taken'
                #print(step)
                y = y_new
                t = t_new
                no_increase = False

            if step == 'step taken':
                if self.profiles:
                    self.mass_profile = np.append(self.mass_profile,t)
                    self.pressure_profile = np.append(self.pressure_profile,np.log10(y[0]))
                    self.radius_profile = np.append(self.radius_profile,y[1])
                    self.density_profile = np.append(self.density_profile,lrho_new)
                    self.temperature_profile = np.append(self.temperature_profile,lT_new)
                    self.component_profile = np.append(self.component_profile,component_idx_new)

        #np.savetxt('../R1_profile.txt',np.c_[self.mass_profile,self.radius_profile,self.pressure_profile,self.temperature_profile,self.density_profile,self.component_profile])
        return(t,y)


    def rk4(self,f, y0, t0, t_end):
        t = self.mass_profile
        y = np.zeros((len(t),len(y0)))
        y[0] = y0             
        h = t[1:]-t[:-1]
        
        if self.profiles:
            self.density_profile = np.zeros_like(t)
            self.temperature_profile = np.zeros_like(t)
            self.component_profile = np.zeros_like(t).astype(int)
            self.density_profile[0] = f(t[0],y[0])[1]
            self.component_profile[0] = np.argmax(self.mass_bds[np.where(self.mass_bds*params.MEarth<t0)])
            self.temperature_profile[0] = np.interp(np.log10(y0[0]),self.pressure_grid,self.T_dict[params.components[self.component_profile[0]]])
        
        for i in range(len(t)-1):
            k1 = h[i]*f(t[i],y[i])[0]
            k2 = h[i]*f(t[i]+0.5*h[i],y[i]+0.5*k1)[0]
            k3 = h[i]*f(t[i]+0.5*h[i],y[i]+0.5*k2)[0]
            k4 = h[i]*f(t[i]+h[i],y[i]+k3)[0]

            y[i+1] = y[i] + (k1+2*k2+2*k3+k4)*(1.0/6.0)

            if self.profiles:
                self.density_profile[i+1] = f(t[i+1],y[i+1])[1]
                if t[i+1] > 0:
                    self.component_profile[i+1] = np.argmax(self.mass_bds[np.where(self.mass_bds*params.MEarth<t[i+1])])
                else:
                    self.component_profile[i+1] = 0
                self.temperature_profile[i+1] = np.interp(np.log10(y[i+1,0]),self.pressure_grid,self.T_dict[params.components[self.component_profile[i+1]]])

        #if self.profiles:
            #np.savetxt('../mr_out/profiles_M'+str(self.mass)+'_P'+str(self.P0)+'_T'+str(self.T0)+'_xw'+str(self.x_w)+'.out',np.c_[t,y[:,1],np.log10(y[:,0]),self.temperature_profile,self.density_profile,self.component_profile])

        return(t[-1],y[-1])
    
def euler(f, y0, t0, t_end, nsteps):
    t = np.linspace(t0,t_end,nsteps)
    h = (t_end-t0)/nsteps
    y = np.zeros((len(t),len(y0)))
    y[0] = y0
    for i in range(len(t)-1):
        y[i+1] = y[i] + h*f(t[i],y[i])

    return(t[-1],y[-1])

def rkf(f, y0, t0, t_end):
    y = np.copy(y0)
    t = np.copy(t0)
    h = -t0*1.0e-3
    #print(t,y)

    a = np.array([0.0,1.0/5.0,3.0/10.0,3.0/5.0,1.0,7.0/8.0])
    b = np.array([[0.0,0.0,0.0,0.0,0.0],
                  [1.0/5.0,0.0,0.0,0.0,0.0],
                  [3.0/40.0,9.0/40.0,0.0,0.0,0.0],
                  [3.0/10.0,-9.0/10.0,6.0/5.0,0.0,0.0],
                  [-11.0/54.0,5.0/2.0,-70.0/27.0,35.0/27.0,0.0],
                  [1631.0/55296.0,175.0/512.0,575.0/13824.0,44275.0/110592.0,253.0/4096.0]])
    c = np.array([37.0/378.0,0.0,250.0/621.0,125.0/594.0,0.0,512.0/1771.0])
    c_s = np.array([2825.0/27648.0,0.0,18575.0/48384.0,13525.0/55296.0,277.0/14336.0,1.0/4.0])

    c = np.repeat(np.expand_dims(c,-1),2,axis=-1)
    c_s = np.repeat(np.expand_dims(c_s,-1),2,axis=-1)
    
    eps = 0.001

    count = 0

    #print(count,t0,h,y0[0],y0[1])
    
    while np.logical_and(t > 0,y[1] > 0):

        if np.abs(h) > t:
            h = -t

        k = np.zeros_like(c)
        k[0] = h*f(t,y)
        k[1] = h*f(t+a[1]*h,y+b[1,0]*k[0])
        k[2] = h*f(t+a[2]*h,y+b[2,0]*k[0]+b[2,1]*k[1])
        k[3] = h*f(t+a[3]*h,y+b[3,0]*k[0]+b[3,1]*k[1]+b[3,2]*k[2])
        k[4] = h*f(t+a[4]*h,y+b[4,0]*k[0]+b[4,1]*k[1]+b[4,2]*k[2]+b[4,3]*k[3])
        k[5] = h*f(t+a[5]*h,y+b[5,0]*k[0]+b[5,1]*k[1]+b[5,2]*k[2]+b[5,3]*k[3]+b[5,4]*k[4])

        dy = np.sum(c*k,axis=0)
        y_new = y + np.sum(c*k,axis=0)
        y_s = y + np.sum(c_s*k,axis=0)
        #print(t,y_new[1],y_s[1])

        delta = y_new-y_s
        delta = delta[1]

        #acc = y0[1]*(h/t0)
        #acc = 6371.0*(h/t0)
        #acc = eps*k[0][1]
        acc = 0.001*params.REarth*h/(t0)
        #print(acc,delta)
        
        #t += h
        #y = y_new

        #print('delta = ',delta,'acc = ',acc)
        
        if np.abs(delta) > np.abs(acc):
            step = 'not taken'
            print(h,step)
            h *= 0.95*np.abs((acc/delta))**0.25
        elif delta == 0:
            step = 'taken'
            print(h,step)
            y = y_new
            t += h
        else:
            step = 'taken'
            print(h,step)
            y = y_new
            t += h
            print(h,np.abs((acc/delta))**0.2,h*np.abs((acc/delta))**0.2)
            h *= 0.95*np.abs((acc/delta))**0.2
        #print(t,y[1])
        #t += h
        #y = y_new

        count += 1
        #print(count,t,h,y[0],y[1])
    
    return (t,y)
