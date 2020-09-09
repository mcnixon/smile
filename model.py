'''Single interior model solver'''

import params
import eos
import numpy as np
#from rkf45 import *

from scipy.integrate import solve_ivp

class Model:
    '''Model parameters for a given mass'''

    def __init__(self,mass,P0=None,T0=None,Pad=None,x_w=None,x_g=None,mode='adaptive',profiles=False,hhb=False,phases=False,liquid=False):

        self.mode = mode
        self.profiles = profiles
        self.x_w = x_w
        self.x_g = x_g
        self.hhb = hhb

        self.phases = phases

        self.tau_0 = 2.0/3.0
        
        if P0 is None:
            self.P0 = params.P_0
        else:
            self.P0 = P0
        if Pad is None:
            self.Pad = params.Pad
        else:
            self.Pad = Pad

        if T0 is None:
            self.T0 = params.T_0
        else:
            self.T0 = T0

        self.Lp = 10**(-10.5)*mass*params.MEarth
            
        #choosing step sizes for each component to create mass grid

        self.mass = mass

        #x_fe = 0.1*(1.0/3.0)
        #x_si = 0.1*(2.0/3.0)
        #x_w = 0.9 - x_g
        #self.mass_fractions = np.array([x_fe,x_si,x_w,x_g])
        #self.mass_fractions = np.array([0.167,0.333,0.5,0.0])
        
        if x_w is None:
            #self.mass_fractions = np.copy(params.mass_fractions)
            x_w = (1.0-x_g)/2.0
            x_fe = (1.0/3.0)*x_w
            x_si = (2.0/3.0)*x_w
            self.mass_fractions = np.array([x_fe,x_si,x_w,x_g])
        elif x_g is None:
            x_fe = (1.0/3.0)*(1.0-x_w)
            x_si = 1.0 - (x_fe + x_w)
            self.mass_fractions = np.array([x_fe,x_si,x_w])
        else:
            x_fe = (1.0/3.0)*(1.0-(x_w+x_g))
            x_si = 1.0 - (x_fe + x_w + x_g)
            self.mass_fractions = np.array([x_fe,x_si,x_w,x_g])

        if self.hhb:
            x_w = 0.9 - x_g
            x_fe = 0.1*(1/3)
            x_si = 1.0 - (x_w + x_g + x_fe)
            self.mass_fractions = np.array([x_fe,x_si,x_w,x_g])

            #x_w = 0.5 - x_g
            #x_fe = 0.5*(1/3)
            #x_si = 1.0 - (x_w + x_g + x_fe)
            #self.mass_fractions = np.array([x_fe,x_si,x_w,x_g])

        #if params.pt == 'Guillot':
        #    x_fe = (1.0-x_g)*0.1
        #    x_si = (1.0-x_g)*0.23
        #    x_w = (1.0-x_g)*0.67
        #    self.mass_fractions = np.array([x_fe,x_si,x_w,x_g])

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

        if params.pt == 'Guillot':
            self.f_r = 0.25

        self.hhe_check = len(params.components) == 4 and self.mass_fractions[-1] > 0.0
            
        for component in params.components:
            self.eos_data[component] = eos.EOS(component)
            if component == 'h2o' and self.hhe_check and self.eos_data[component].isothermal == False:
                rho_T = [None, None]
            else:
                rho_T = self.eos_data[component].get_eos(self.pressure_grid,np.log10(self.Pad),np.log10(self.T0))
            self.rho_dict[component] = rho_T[0]
            if self.profiles:
                self.T_dict[component] = rho_T[1]

        #H2O liquid-vapour phase boundary
        self.lv = np.loadtxt('../eos_data/liquid_vapour_bd.txt')
        self.lv = np.log10(self.lv)

    def find_Rp(self,steps=8.0e4):

        solved = False
        count = 0
        Rp_range = np.copy(params.Rp_range)
        self.mass_profile = None

        T_hhb = 0.0
        P_hhb = 20.0

        while solved is False:
            count += 1
            if count > 50:
                print('failed')
                return ['failed']
            #if Rp_range[0] > 3.1:
            #    return np.array([3.1])
            Rp_choice = np.mean(Rp_range)
            if self.mass_profile is None and self.mode == 'adaptive_new':
                Rp_choice = np.amax(Rp_range)
            #if self.mass_profile is not None:
                #Rp_choice = 2.2520408395677802
            #Rp_choice = 1.0
            print(Rp_choice)

            if self.hhe_check and self.eos_data['h2o'].isothermal == False:
                self.rho_dict['h2o'] = None

            if params.pt == 'Guillot':
                self.T_int = (self.Lp/(4.0*np.pi*(Rp_choice*params.REarth)**2*params.sigma_SB))**0.25
                self.T_eq = (self.T0**4-self.T_int**4)**0.25
                self.T_irr = self.T_eq*self.f_r**-0.25
                self.gamma = 0.6*(self.T_irr/2000.0)**0.5
                if params.pt == 'Guillot':
                    self.P0 = (params.G*self.mass*params.MEarth*1.68*0.67/((Rp_choice*params.REarth)**2*10**-7.32*self.T0**0.45))**(1/1.68)
                    self.Tsurf = ((3.0/4.0)*self.T_int**4*((2.0/3.0)+(2.0/3.0)) + (3.0/4.0)*self.T_irr**4*self.f_r*((2.0/3.0)+1.0/(self.gamma*3.0**0.5)+(self.gamma/(3.0**0.5)-1.0/(self.gamma*3.0**0.5))*np.exp(-self.gamma*(2.0/3.0)*3.0**0.5)))**0.25

                self.y0 = np.array([self.P0,Rp_choice*params.REarth,self.tau_0])
                    
            else:
                self.Tsurf = self.T0
                self.y0 = np.array([self.P0,Rp_choice*params.REarth])
                #define all the variables here

            if self.mode == 'adaptive':
                soln = self.rk4_a(self.ode_sys,self.y0, self.mass*params.MEarth,0.0)

            if self.mode == 'adaptive_new':
                if self.mass_profile is None:
                    soln_init = self.rk4_a(self.ode_sys,self.y0,self.mass*params.MEarth,0.0)
                    continue
                else:
                    soln = self.rk4(self.ode_sys,self.y0,self.mass*params.MEarth,0.0)

            else:
                self.mass_profile = np.linspace(self.mass*params.MEarth,0.0,3.0e5)
                #print(self.mass_profile)
                #quit()
                soln = self.rk4(self.ode_sys, np.array([self.P0,Rp_choice*params.REarth,self.tau_0,self.Tsurf]), self.mass*params.MEarth,0.0)

            final_r = soln[1][1]
            final_m = soln[0]

            if self.hhb:
                hhb_idx = np.argwhere(self.component_profile==3)[-1]
                P_hhb_new = self.pressure_profile[hhb_idx]
                T_hhb_new = self.temperature_profile[hhb_idx]
                print(np.log10(P_hhb_new),T_hhb_new)
                if np.abs(T_hhb_new - T_hhb) < 0.001 and np.abs(np.log10(P_hhb_new) - np.log10(P_hhb)) < 0.001 and count >= 10:

                    #np.savetxt('../mr_out/ia_profiles_T'+str(self.T0)+'.out',np.c_[self.mass_profile,self.radius_profile,self.pressure_profile,self.temperature_profile,self.density_profile,self.component_profile])
                    
                    #print(Rp_choice)
                    return(Rp_choice,P_hhb,T_hhb)
                else:
                    P_hhb = np.copy(P_hhb_new)
                    T_hhb = np.copy(T_hhb_new)
            
            #print(final_r,final_m)
            #quit()

            if final_r < 0 or final_m > 0:
                Rp_range[0] = Rp_choice
            elif final_r > 1.0e3:
                Rp_range[1] = Rp_choice
            else:
                #print('done')
                print('Rp = '+str(Rp_choice))
                #quit()
                Rp_final = Rp_choice

                #if self.profiles:
                #    np.savetxt('../mr_out/profile_M'+str(self.mass)+'_P'+str(self.P0)+'_T'+str(self.T0)+'_xw'+str(self.x_w)+'.out',np.c_[self.mass_profile,self.radius_profile,self.pressure_profile,self.temperature_profile,self.density_profile,self.component_profile])
                
                solved = True

        #hhb_idx = np.argwhere(self.component_profile==3)[-1]
        #P_hhb = self.pressure_profile[hhb_idx]
        #T_hhb = self.temperature_profile[hhb_idx]
        #print(P_hhb,T_hhb)

        if self.hhb:
            hhb_idx = np.argwhere(self.component_profile==3)[-1]
            P_hhb = self.pressure_profile[hhb_idx]
            T_hhb = self.temperature_profile[hhb_idx]
            print(P_hhb,T_hhb)
            return(Rp_final,P_hhb,T_hhb)
        #np.savetxt('../mr_out/new_h2o_profile_T'+str(self.T0)+'_Pad'+str(np.log10(params.Pad))+'.out',np.c_[self.mass_profile,self.radius_profile,self.pressure_profile,self.temperature_profile,self.density_profile,self.component_profile])
        return Rp_final

    def ode_sys(self,mass,y):

        p = y[0]
        r = y[1]

        if mass > self.mass_bds[3]*params.MEarth:
            component_idx = 3
        elif mass > self.mass_bds[2]*params.MEarth:
            component_idx = 2
        elif mass > self.mass_bds[1]*params.MEarth:
            component_idx = 1
        else:
            component_idx = 0

        if component_idx == 2 and self.rho_dict['h2o'] is None:
            lT = np.interp(np.log10(p),self.pressure_grid,self.T_dict['hhe'])
            #lT = fast_interp(np.log10(p),self.pressure_grid,self.T_dict['hhe'])
            rho_T = self.eos_data['h2o'].get_eos(self.pressure_grid,np.log10(self.Pad),lT,np.log10(p))
            self.rho_dict['h2o'] = rho_T[0]
            self.T_dict['h2o'] = rho_T[1]
            
        #if mass > 0:
        #    component_idx = np.argmax(self.mass_bds[np.where(self.mass_bds*params.MEarth<mass)])
        #else:
        #    component_idx = 0
        
        comp = params.components[component_idx]

        dp_dm = -(params.G*mass)/(4.0*np.pi*r**4)
        
        if params.pt == 'Guillot':
            tau = y[2]
            T = y[3]

            #find non-adiabatic temperature
            kappa = self.eos_data[comp].get_opacity(np.log10(p),np.log10(T))

            if component_idx == 3:
                dtau_dm = -kappa/(4.0*np.pi*r**2)
            else:
                dT_dm = 0.0
                dtau_dm = 0.0
     
        #check which layer we are in

        if params.pt == 'Guillot' and component_idx == 3:
            lrho = self.eos_data[comp].get_density(np.log10(p),np.log10(T))
            rho = 10**lrho
        else:    
            lrho = np.interp(np.log10(p),self.pressure_grid,self.rho_dict[comp])
            #lrho = fast_interp(np.log10(p),self.pressure_grid,self.rho_dict[comp])

            rho = 10**lrho
        dr_dm = 1.0/(4.0*np.pi*r**2*rho)

        if component_idx == 3 and params.pt == 'Guillot':
            N = 1000.0
            if tau < N*1.0/(self.gamma*3.0**0.5):
                dT_dtau = ((3.0*self.T_int**4/4.0)+((3.0*self.T_irr**4/4.0)*self.f_r*(1.0-self.gamma**2)*(np.exp(-self.gamma*tau*3.0**0.5))))/(4.0*T**3)
                dT_dm_n = dT_dtau*dtau_dm
            else:
                dT_dr = -3.0*kappa*self.Lp*rho/(16.0*4.0*params.sigma_SB*np.pi*(T)**3*r**2)
                dT_dm_n = dT_dr*dr_dm
            #find adiabatic temperature
            if component_idx == 3:
                dT_dm_ad = self.eos_data[comp].get_gradient(np.log10(p),np.log10(T))*dp_dm*(T/p)
                if np.abs(dT_dm_n) > np.abs(dT_dm_ad) and p>1.0e5:
                    self.T_prescription = 'adiabatic'

                if self.T_prescription == 'adiabatic':
                    dT_dm = dT_dm_ad
                else:
                    dT_dm = dT_dm_n

        if params.pt == 'Guillot':
            return((np.array([dp_dm,dr_dm,dtau_dm,dT_dm]),lrho,T))
        elif self.mode == 'adaptive' or self.mode == 'adaptive_new':
            return((np.array([dp_dm,dr_dm]),lrho))
        else:
            return((np.array([dp_dm,dr_dm]),lrho))

    def rk4_a(self,f, y0, t0, t_end):
        y = np.copy(y0)
        #print(np.insert(y0,0,np.array([t0])))
        t = np.copy(t0)
        h_min = -t0*1.0e-5
        h_max = -t0*1.0e-2

        #h = -self.mass_fractions[-1]*t0*0.1
        frac = self.mass_fractions[np.where(self.mass_fractions>0.0)]
        if frac[-1] > 1.0e-4:
            h = -t0*1.0e-6
        else:
            h = -frac[-1]*t0*0.01

        if np.abs(h)<np.abs(h_min):
            h_min = np.copy(h)
        

        if self.profiles:
            self.mass_profile = np.array([t])
            self.pressure_profile = np.array([np.log10(y[0])])
            self.radius_profile = np.array([y[1]])
            self.density_profile = None
            self.temperature_profile = np.array([np.log10(self.Tsurf)])
            self.component_profile = None
        
        no_increase = False
        force_step = False

        self.T_prescription = 'non-adiabat'

        f0 = None
        #h = -t0*1.0e-10
        while np.logical_and(t > 0,y[1] > 0):

            if np.abs(h) > t:
                h = -t

            if t > 0:
                if t > self.mass_bds[3]*params.MEarth:
                    component_idx = 3
                elif t > self.mass_bds[2]*params.MEarth:
                    component_idx = 2
                elif t > self.mass_bds[1]*params.MEarth:
                    component_idx = 1
                else:
                    component_idx = 0
            else:
                component_idx = 0

            #force_step = True
            #h *=1.01

            self.mass_step = np.copy(h)

            if f0 is None:
                f0 = f(t,y)
            k1 = h*f0[0]
            k2 = h*f(t+0.5*h,y+0.5*k1)[0]
            k3 = h*f(t+0.5*h,y+0.5*k2)[0]
            k4 = h*f(t+h,y+k3)[0]

            lrho = f0[1]
            lT = self.temperature_profile[-1]

            lT_bd = np.interp(np.log10(y[0]),self.lv[:,0],self.lv[:,1])
            #lT_bd = fast_interp(np.log10(y[0]),self.lv[:,0],self.lv[:,1])
            if lT > lT_bd:
                phase = 'v'
            else:
                phase = 'l'

            
            if self.profiles:
                if self.density_profile is None:
                    self.density_profile = np.array([lrho])
                    self.component_profile = component_idx

            delta_y = (k1+2*k2+2*k3+k4)*(1.0/6.0)

            #print(y,delta_y)

            y_new = y + delta_y
            t_new = t + h

            if t_new > self.mass_bds[3]*params.MEarth:
                component_idx_new = 3
            elif t_new > self.mass_bds[2]*params.MEarth:
                component_idx_new = 2
            elif t_new > self.mass_bds[1]*params.MEarth:
                component_idx_new = 1
            else:
                component_idx_new = 0
            
            #if t_new > 0:
            #    component_idx_new = np.argmax(self.mass_bds[np.where(self.mass_bds*params.MEarth<t_new)])
            #else:
            #    component_idx_new = 0

            f_new = f(t_new,y_new)
            lrho_new = f_new[1]
            if params.pt == 'Guillot' and component_idx_new == 3:
                lT_new = f_new[2]#np.log10(y_new[3])
            else:
                lT_new = np.interp(np.log10(y_new[0]),self.pressure_grid,self.T_dict[params.components[component_idx_new]])
                #lT_new = fast_interp(np.log10(y_new[0]),self.pressure_grid,self.T_dict[params.components[component_idx_new]])

            lT_bd = np.interp(np.log10(y[0]),self.lv[:,0],self.lv[:,1])
            if lT_new > lT_bd:
                phase_new = 'v'
            else:
                phase_new = 'l'

            delta_rho = np.abs(lrho_new-lrho)

            delta_rho_max = 1.0e-1
            delta_rho_min = 1.0e-3

            last = False
            new_eos = False
            step = 'not taken'
            if force_step:
                step = 'step taken'
                #print('boundary '+step)
                y = y_new
                t = t_new
                no_increase = False
                force_step = False
            elif np.logical_and(self.phases,component_idx == 2):
                step = 'step adjusted'
                h = -self.mass_bds[component_idx+1]*params.MEarth*(1.0/5000.0)
                force_step = True
            elif component_idx == 0:
                step = 'step adjusted'
                h = -self.mass_bds[component_idx+1]*params.MEarth*(1.0/1000.0)
                #h = -self.mass_bds[component_idx+1]*params.MEarth*(1.0/5000.0)
                force_step = True
            elif component_idx_new != component_idx:
                #if component_idx_new == 2.0:
                    #print(phase_new)
                #bd_edge = self.mass_bds[component_idx]*params.MEarth+t0*5.0e-6
                bd_edge = self.mass_bds[component_idx]*params.MEarth
                h = bd_edge - t
                force_step = True
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
            elif component_idx == 2.0 and phase_new != phase:
                force_step = True
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
                #print(t_new,y_new)
                no_increase = False

            if step == 'step taken':
                f0 = np.copy(f_new)
                #print(np.insert(y,0,np.array([t])))
                if self.profiles:
                    self.mass_profile = np.append(self.mass_profile,t)
                    self.pressure_profile = np.append(self.pressure_profile,np.log10(y[0]))
                    self.radius_profile = np.append(self.radius_profile,y[1])
                    self.density_profile = np.append(self.density_profile,lrho_new)
                    self.temperature_profile = np.append(self.temperature_profile,lT_new)
                    self.component_profile = np.append(self.component_profile,component_idx_new)
                    self.lT_current = np.copy(lT_new)

        #print('saving profile')
        #np.savetxt('../mr_out/test_profile.txt',np.c_[self.mass_profile,self.radius_profile,self.pressure_profile,self.temperature_profile,self.density_profile,self.component_profile])
        #quit()
        return(t,y)


    def rk4(self,f, y0, t0, t_end):
        t = self.mass_profile
        #t = np.linspace(t0,0,1.0e5)
        y = np.zeros((len(t),len(y0)))
        y[0] = y0             
        h = t[1:]-t[:-1]
        self.mass_step = h[0]

        self.T_prescription = 'non-adiabat'

        f0 = f(t[0],y[0])

        self.component_profile = np.zeros_like(t).astype(int)
        self.component_profile[np.where(t>self.mass_bds[1]*params.MEarth)] = 1
        self.component_profile[np.where(t>self.mass_bds[2]*params.MEarth)] = 2
        self.component_profile[np.where(t>self.mass_bds[3]*params.MEarth)] = 3
        
        if self.profiles:
            self.density_profile = np.zeros_like(t)
            self.temperature_profile = np.zeros_like(t)
            self.density_profile[0] = np.copy(f0[1])
            self.temperature_profile[0] = np.log10(self.Tsurf)
        
        for i in range(len(t)-1):
            self.mass_step = h[i]
            k1 = h[i]*f0[0]
            k2 = h[i]*f(t[i]+0.5*h[i],y[i]+0.5*k1)[0]
            k3 = h[i]*f(t[i]+0.5*h[i],y[i]+0.5*k2)[0]
            k4 = h[i]*f(t[i]+h[i],y[i]+k3)[0]

            delta_y = (k1+2*k2+2*k3+k4)*(1.0/6.0)
            y[i+1] = y[i] + delta_y

            #print(y[i][-1],self.component_profile[i],1.0/(self.gamma*3.0**0.5))
            f1 = f(t[i+1],y[i+1])
            if self.profiles:
                self.density_profile[i+1] = f1[1]
                if params.pt == 'Guillot':
                    self.temperature_profile[i+1] = f1[2]
                else:
                    #self.temperature_profile[i+1] = fast_interp(np.log10(y[i+1,0]),self.pressure_grid,self.T_dict[params.components[self.component_profile[i+1]]])
                    self.temperature_profile[i+1] = np.interp(np.log10(y[i+1,0]),self.pressure_grid,self.T_dict[params.components[self.component_profile[i+1]]])
                f0 = f1
                    

            if y[i+1,1] < 0.0:
                return(t[i+1],y[i+1])

        self.pressure_profile = y[:,0]

        #np.savetxt('../mr_out/new_h2o_profile_M'+str(self.mass)+'_T'+str(self.T0)+'_P0'+str(np.log10(self.P0))+'_h2o'+str(self.x_w)+'.out',np.c_[t,y[:,1],np.log10(y[:,0]),self.temperature_profile,self.density_profile,self.component_profile])
        #np.savetxt('../mr_out/liquid_profile_M'+str(self.mass)+'.out',np.c_[t,y[:,1],np.log10(y[:,0]),self.temperature_profile,self.density_profile,self.component_profile])

        #if self.profiles:
            #np.savetxt('../mr_out/test_profile5_M'+str(self.mass)+'.txt',np.c_[t,y[:,1],np.log10(y[:,0]),self.temperature_profile,self.density_profile,self.component_profile])
            #np.savetxt('../mr_out/newT_rogers_fixed_profiles_plaw_M'+str(self.mass)+'_T'+str(self.T0)+'.out',np.c_[t,y[:,1],np.log10(y[:,0]),np.log10(y[:,3]),self.density_profile,self.component_profile])

        return(t[-1],y[-1])

def fast_interp(x,xp,fp):
    if x >= xp[-1]:
        return fp[-1]
    elif x <= xp[0]:
        return fp[0]
    else:
        f_idx = (x-xp[0])/(xp[1]-xp[0])
        #idx = f_idx.astype(int)
        idx = int(f_idx)
        f = f_idx - idx
        return fp[idx] + f*(fp[idx+1]-fp[idx])
