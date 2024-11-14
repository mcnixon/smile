'''Single interior model solver'''

import params
import eos
import numpy as np

class Model:
    '''Model parameters for a given mass'''

    def __init__(self, mass, P0, T0, Pad, x_si, x_w, x_g, pt, pt_file, mixed, profiles, phases=False, hhb=False, liquid=False):

        self.profiles = profiles
        self.x_w = x_w
        self.x_g = x_g
        self.hhb = hhb

        self.pt = pt
        self.pt_file = pt_file

        self.mixed = mixed

        self.phases = phases

        self.tau_0 = 2.0/3.0

        self.P0 = P0
        self.Pad = Pad
        self.T0 = T0

        self.Lp = 10**(-10.5)*mass*params.MEarth
            
        #choosing step sizes for each component to create mass grid

        self.mass = mass

        if x_si is None:
            x_fe = (1.0/3.0)*(1.0-(x_w+x_g))
            x_si = 1.0 - (x_fe + x_w + x_g)
            
        else:
            x_fe = 1.0 - (x_si + x_w + x_g)
        self.mass_fractions = np.array([x_fe,x_si,x_w,x_g])

        if len(params.components) > len(self.mass_fractions):
            self.mass_fractions = np.append(self.mass_fractions,1.0-np.sum(self.mass_fractions))

        self.mass_bds = np.cumsum(self.mass_fractions)*self.mass
        self.mass_bds = np.insert(self.mass_bds,0,0)


        frac = self.mass_fractions[np.where(self.mass_fractions>0.0)]
        if frac[-1] > 1.0e-6:
            self.mass_profile = (1-np.logspace(-8,0,1000))*self.mass
        else:
            self.mass_profile = (1-np.logspace(np.log10(0.01*frac[-1]),0,1000))*self.mass
        self.mass_profile = np.insert(self.mass_profile,0,self.mass)
        self.mass_profile *= params.MEarth

        #get EOS data

        self.pressure_grid = params.Pgrid

        self.eos_data = {}
        self.rho_dict = {}
        self.T_dict = {}

        if self.pt == 'guillot':
            self.f_r = 0.25

        self.hhe_check = len(params.components) == 4 and self.mass_fractions[-1] > 0.0
        self.env_check = len(params.components) > 2 and self.mass_fractions[-1] > 0.0
            
        for component in params.components:
            self.eos_data[component] = eos.EOS(component,pt=self.pt,pt_file=self.pt_file)
            if component == 'h2o' and self.hhe_check and self.eos_data[component].isothermal == False:
                rho_T = [None, None]
            elif component == 'mgpv' and self.env_check and self.eos_data[component].isothermal == False:
                rho_T = [None,None]
            else:
                rho_T = self.eos_data[component].get_eos(self.pressure_grid,np.log10(self.Pad),np.log10(self.T0))
            self.rho_dict[component] = rho_T[0]
            self.T_dict[component] = rho_T[1]

        if self.mixed:
            self.env_data = eos.EOS('hhe',component2='h2o',pt=self.pt,pt_file=self.pt_file)
            rho_T = self.env_data.get_mixed_eos((self.x_w/(self.x_g+self.x_w)),self.pressure_grid,np.log10(self.Pad),np.log10(self.T0))
            self.rho_dict['env'] = rho_T[0]
            self.T_dict['env'] = rho_T[1]
            
        #H2O liquid-vapour phase boundary
        self.lv = np.loadtxt(eos.lv_file)
        self.lv = np.log10(self.lv)


    def find_Rp(self,steps=8.0e4):

        solved = False
        count = 0
        Rp_range = np.copy(params.Rp_range)
        #self.mass_profile = None

        T_hhb = 0.0
        P_hhb = 20.0

        while solved is False:
            count += 1
            if count > 50:
                print('failed')
                return 'failed'
            Rp_choice = np.mean(Rp_range)

            if self.hhe_check and self.eos_data['h2o'].isothermal == False:
                self.rho_dict['h2o'] = None
            if self.env_check and self.eos_data['mgpv'].isothermal == False:
                self.rho_dict['mgpv'] = None

            if self.pt == 'guillot':
                self.T_int = (self.Lp/(4.0*np.pi*(Rp_choice*params.REarth)**2*params.sigma_SB))**0.25
                self.T_eq = (self.T0**4-self.T_int**4)**0.25
                self.T_irr = self.T_eq*self.f_r**-0.25
                self.gamma = 0.6*(self.T_irr/2000.0)**0.5
                if self.pt == 'guillot':
                    self.P0 = (params.G*self.mass*params.MEarth*1.68*0.67/((Rp_choice*params.REarth)**2*10**-7.32*self.T0**0.45))**(1/1.68)
                    self.Tsurf = ((3.0/4.0)*self.T_int**4*((2.0/3.0)+(2.0/3.0)) + (3.0/4.0)*self.T_irr**4*self.f_r*((2.0/3.0)+1.0/(self.gamma*3.0**0.5)+(self.gamma/(3.0**0.5)-1.0/(self.gamma*3.0**0.5))*np.exp(-self.gamma*(2.0/3.0)*3.0**0.5)))**0.25

                self.y0 = np.array([self.P0,Rp_choice*params.REarth,self.tau_0])
                    
            else:
                self.Tsurf = self.T0
                self.y0 = np.array([self.P0,Rp_choice*params.REarth])
                #define all the variables here

            soln = self.rk4(self.ode_sys,self.y0,self.mass*params.MEarth,0.0)

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
                #print('Rp = '+str(Rp_choice))
                Rp_final = Rp_choice
                
                solved = True

        if self.hhb:
            hhb_idx = np.argwhere(self.component_profile==3)[-1]
            P_hhb = self.pressure_profile[hhb_idx]
            T_hhb = self.temperature_profile[hhb_idx]
            print(P_hhb,T_hhb)
            return(Rp_final,P_hhb,T_hhb)
        
        if self.profiles:
            np.savetxt('../mr_out/profile_M'+str(self.mass)+'_P0'+str(self.P0)+'_T'+str(self.T0)+'_Pad'+str(self.Pad)+'_xw'+str(self.x_w)+'_xg'+str(self.x_g)+'.out',np.c_[self.mass_profile,self.radius_profile,self.pressure_profile,self.temperature_profile,self.density_profile,self.component_profile])
        return Rp_final

    def ode_sys(self,mass,y):

        p = y[0]
        r = y[1]

        if mass > self.mass_bds[3]*params.MEarth:
            if self.mixed:
                component_idx = 2
            else:
                component_idx = 3
        elif mass > self.mass_bds[2]*params.MEarth:
            component_idx = 2
        elif mass > self.mass_bds[1]*params.MEarth:
            component_idx = 1
        else:
            component_idx = 0

        if component_idx == 2 and self.rho_dict['h2o'] is None:
            lT = np.interp(np.log10(p),self.pressure_grid,self.T_dict['hhe'])
            rho_T = self.eos_data['h2o'].get_eos(self.pressure_grid,np.log10(self.Pad),lT,np.log10(p))
            self.rho_dict['h2o'] = rho_T[0]
            self.T_dict['h2o'] = rho_T[1]
            
        if self.mixed:
            if component_idx > 1.0:
                comp = 'env'
            else:
                comp = params.components[component_idx]
        else:        
            comp = params.components[component_idx]

        if component_idx == 1 and self.rho_dict['mgpv'] is None:

            if self.mixed:
                lT = np.interp(np.log10(p),self.pressure_grid,self.T_dict['env'])
            elif self.mass_fractions[2] > 0.0:
                lT = np.interp(np.log10(p),self.pressure_grid,self.T_dict['h2o'])
            else:
                lT = np.interp(np.log10(p),self.pressure_grid,self.T_dict['hhe'])
            rho_T = self.eos_data['mgpv'].get_eos(self.pressure_grid,np.log10(self.Pad),lT,np.log10(p))
            self.rho_dict['mgpv'] = rho_T[0]
            self.T_dict['mgpv'] = rho_T[1]

        dp_dm = -(params.G*mass)/(4.0*np.pi*r**4)
        
        if self.pt == 'guillot':
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

        if self.pt == 'guillot' and component_idx == 3:
            lrho = self.eos_data[comp].get_density(np.log10(p),np.log10(T))
            rho = 10**lrho
        else:    
            lrho = np.interp(np.log10(p),self.pressure_grid,self.rho_dict[comp])

            rho = 10**lrho
        dr_dm = 1.0/(4.0*np.pi*r**2*rho)

        if component_idx == 3 and self.pt == 'guillot':
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

        if self.pt == 'guillot':
            return((np.array([dp_dm,dr_dm,dtau_dm,dT_dm]),lrho,T))
        else:
            return((np.array([dp_dm,dr_dm]),lrho))
    
    def rk4(self,f, y0, t0, t_end):
        t = self.mass_profile

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
        
        self.density_profile = np.zeros_like(t)
        self.temperature_profile = np.zeros_like(t)
        self.density_profile[0] = np.copy(f0[1])
        self.temperature_profile[0] = self.Tsurf
        
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
            self.density_profile[i+1] = f1[1]
            if self.pt == 'guillot':
                self.temperature_profile[i+1] = f1[2]
            else:
                    if self.mixed:
                        if self.component_profile[i+1] > 1.0:
                            self.temperature_profile[i+1] = np.interp(np.log10(y[i+1,0]),self.pressure_grid,self.T_dict['env'])
                        else:
                            self.temperature_profile[i+1] = np.interp(np.log10(y[i+1,0]),self.pressure_grid,self.T_dict[params.components[self.component_profile[i+1]]])
                    else:
                        self.temperature_profile[i+1] = np.interp(np.log10(y[i+1,0]),self.pressure_grid,self.T_dict[params.components[self.component_profile[i+1]]])
                
            f0 = f1
                    

            if y[i+1,1] < 0.0:
                return(t[i+1],y[i+1])

        self.pressure_profile = y[:,0]
        self.radius_profile = y[:,1]

        return(t[-1],y[-1])
