'''Single interior model solver'''

import eos
import numpy as np

class Model:
    '''
    Model parameters for a given mass.
    
    Attributes:
    mass (float): Planet mass (Earth masses)
    P0 (float): Pressure at the photosphere
    T0 (float): Temperature at the photosphere
    Pad (float): Pressure at the radiative-convective boundary (for isotherm-adiabat models)
    x_si (float): Silicate mass fraction
    x_w (float): Water mass fraction
    x_g (float): H/He mass fraction
    Rp_lower (float): Lower limit for Rp (Earth radii)
    Rp_upper (float): upper limit for Rp (Earth radii)
    pt (str): Pressure-temperature relation type
    pt_file (str): Pressure-temperature relation file
    mixed (bool): Flag for mixed H/He/H2O envelope
    profiles (bool): Flag to save interior profiles to text file
    hhb (bool): Flag to compute H/He/H2O boundary. Default is False
    output_dir (str): Location to save profiles, only used if profiles=True.
    '''

    def __init__(self, mass, P0, T0, Pad, x_si, x_w, x_g, Rp_lower, Rp_upper, pt, pt_file, mixed, profiles, hhb, output_dir):

        '''
        Initializes the Model class with the given parameters.
        '''

        self.mass = mass
        self.profiles = profiles
        self.x_w = x_w
        self.x_g = x_g

        self.initial_Rp_range = np.array([Rp_lower,Rp_upper])
        
        self.hhb = hhb

        self.pt = pt
        self.pt_file = pt_file

        self.mixed = mixed

        self.output_dir = output_dir

        #Optical depth of photosphere for Guillot P-T profile (default = 2/3)
        self.tau_0 = 2.0/3.0

        self.P0 = P0
        self.Pad = Pad
        self.T0 = T0

        #Constants (SI units)
        self.MEarth = 5.972e24
        self.REarth = 6.371e6
        self.G = 6.67e-11
        self.sigma_SB = 5.6704e-08

        #Component names
        self.components = ['fe','mgpv','h2o','hhe']

        #Planet luminosity for Guillot P-T profile (default = 10**-10.5 times planet mass, see Rogers et al 2010)
        self.Lp = 10**(-10.5)*self.mass*self.MEarth

        #Set mass fractions. If x_si is not provided, assume x_si = 2*x_fe
        if x_si is None:
            x_fe = (1.0/3.0)*(1.0-(x_w+x_g))
            x_si = 1.0 - (x_fe + x_w + x_g)
            
        else:
            x_fe = 1.0 - (x_si + x_w + x_g)
        self.mass_fractions = np.array([x_fe,x_si,x_w,x_g])

        if len(self.components) > len(self.mass_fractions):
            self.mass_fractions = np.append(self.mass_fractions,1.0-np.sum(self.mass_fractions))

        self.mass_bds = np.cumsum(self.mass_fractions)*self.mass
        self.mass_bds = np.insert(self.mass_bds,0,0)

        #Choose step sizes for each component to create mass grid
        frac = self.mass_fractions[np.where(self.mass_fractions>0.0)]
        if frac[-1] > 1.0e-6:
            self.mass_profile = (1-np.logspace(-8,0,1000))*self.mass
        else:
            self.mass_profile = (1-np.logspace(np.log10(0.01*frac[-1]),0,1000))*self.mass
        self.mass_profile = np.insert(self.mass_profile,0,self.mass)
        self.mass_profile *= self.MEarth

        #Get EOS data
        self.pressure_grid = np.arange(2,15,0.01)

        self.eos_data = {}
        self.rho_dict = {}
        self.T_dict = {}

        #Set redistribution factor for Guillot P-T profile (default = 0.25)
        if self.pt == 'guillot':
            self.f_r = 0.25

        #Check whether H/He is included as a separate layer
        self.hhe_check = len(self.components) == 4 and self.mass_fractions[-1] > 0.0

        #Check whether the planet has a gaseous envelope
        self.env_check = len(self.components) > 2 and self.mass_fractions[-1] > 0.0

        #Load EOS data
        for component in self.components:
            self.eos_data[component] = eos.EOS(component,pt=self.pt,pt_file=self.pt_file)
            if component == 'h2o' and self.hhe_check and self.eos_data[component].isothermal == False:
                rho_T = [None, None]
            elif component == 'mgpv' and self.env_check and self.eos_data[component].isothermal == False:
                rho_T = [None,None]
            else:
                rho_T = self.eos_data[component].get_eos(self.pressure_grid,np.log10(self.Pad),np.log10(self.T0))
            self.rho_dict[component] = rho_T[0]
            self.T_dict[component] = rho_T[1]

        #Load EOS data for mixed H/He/H2O envelope
        if self.mixed:
            self.env_data = eos.EOS('hhe',component2='h2o',pt=self.pt,pt_file=self.pt_file)
            rho_T = self.env_data.get_mixed_eos((self.x_w/(self.x_g+self.x_w)),self.pressure_grid,np.log10(self.Pad),np.log10(self.T0))
            self.rho_dict['env'] = rho_T[0]
            self.T_dict['env'] = rho_T[1]
            
        #Load H2O liquid-vapour phase boundary data
        self.lv = np.loadtxt(eos.lv_file)
        self.lv = np.log10(self.lv)


    def find_Rp(self):
        '''
        Finds planetary radius for a given set of input parameters

        Returns:
        float, tuple or str: If solved successfully, returns the planetary radius (float) or radius + pressure and temperature at the H/He/H2O boundary (HHB) if 'hhb' = True. Note that if 'hhb' = True, the radius may be less accurate.
        If not solved successfully (e.g. due to radius outside the range of initial guesses), returns 'Failed to converge.'
        '''

        solved = False
        count = 0
        Rp_range = np.copy(self.initial_Rp_range)

        #Initialising P and T at the HHB. Will always update before returning values.
        P_hhb = 20.0
        T_hhb = 0.0

        while solved is False:
            count += 1
            if count > 50:
                #Exits the loop if a solution is not found within 50 iterations
                if Rp_choice > np.mean(self.initial_Rp_range):
                    return 'Failed to converge. Try increasing Rp_upper.'
                else:
                    return 'Failed to converge. Try decreasing Rp_lower.'
            
            #Chooses the mean value of the selected Rp range
            Rp_choice = np.mean(Rp_range)

            #Resets water and silicate EOS for non-isothermal models
            if self.hhe_check and self.eos_data['h2o'].isothermal == False:
                self.rho_dict['h2o'] = None
            if self.env_check and self.eos_data['mgpv'].isothermal == False:
                self.rho_dict['mgpv'] = None

            #Sets intrinsic temperature, photospheric pressure and surface temperature for Guillot P-T profile
            if self.pt == 'guillot':
                self.T_int = (self.Lp/(4.0*np.pi*(Rp_choice*self.REarth)**2*self.sigma_SB))**0.25
                self.T_eq = (self.T0**4-self.T_int**4)**0.25
                self.T_irr = self.T_eq*self.f_r**-0.25
                self.gamma = 0.6*(self.T_irr/2000.0)**0.5
                self.P0 = (self.G*self.mass*self.MEarth*1.68*0.67/((Rp_choice*self.REarth)**2*10**-7.32*self.T0**0.45))**(1/1.68)
                self.Tsurf = ((3.0/4.0)*self.T_int**4*((2.0/3.0)+(2.0/3.0)) + (3.0/4.0)*self.T_irr**4*self.f_r*((2.0/3.0)+1.0/(self.gamma*3.0**0.5)+(self.gamma/(3.0**0.5)-1.0/(self.gamma*3.0**0.5))*np.exp(-self.gamma*(2.0/3.0)*3.0**0.5)))**0.25

                self.y0 = np.array([self.P0,Rp_choice*self.REarth,self.tau_0])

            #Sets surface temperature for non-Guillot P-T profile
            else:
                self.Tsurf = self.T0
                self.y0 = np.array([self.P0,Rp_choice*self.REarth])

            #Solve differential equation system
            soln = self.rk4(self.ode_sys,self.y0)

            final_r = soln[1][1]
            final_m = soln[0]

            if self.hhb:
                hhb_idx = np.argwhere(self.component_profile==3)[-1]
                P_hhb_new = self.pressure_profile[hhb_idx]
                T_hhb_new = self.temperature_profile[hhb_idx]
                print(np.log10(P_hhb_new),T_hhb_new)
                if np.abs(T_hhb_new - T_hhb) < 0.001 and np.abs(np.log10(P_hhb_new) - np.log10(P_hhb)) < 0.001 and count >= 10:
                    return(Rp_choice,P_hhb,T_hhb)
                else:
                    P_hhb = np.copy(P_hhb_new)
                    T_hhb = np.copy(T_hhb_new)

            #Update Rp range based on final mass and radius
            if final_r < 0 or final_m > 0:
                Rp_range[0] = Rp_choice
            elif final_r > 1.0e3:
                Rp_range[1] = Rp_choice
            else:
                #Convergence criteria met, set output Rp
                Rp_final = Rp_choice
                
                solved = True

        #Return P and T at the H/He/H2O boundary if chosen
        if self.hhb:
            hhb_idx = np.argwhere(self.component_profile==3)[-1]
            P_hhb = self.pressure_profile[hhb_idx]
            T_hhb = self.temperature_profile[hhb_idx]
            print(P_hhb,T_hhb)
            return(Rp_final,P_hhb,T_hhb)

        #Save mass, radius, pressure, temperature, density and component profiles if chosen
        if self.profiles:
            #Create output directory if needed
            import pathlib
            pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            if self.mixed:
                np.savetxt(self.output_dir + f"/profile_M_{self.mass:.2f}_P0_{self.P0:.2e}_T0_{self.T0:.2f}_Pad_{self.Pad:.2e}_xg_{self.mass_fractions[3]:.2e}_xw_{self.mass_fractions[2]:.2e}_xmgpv_{self.mass_fractions[1]:.2e}_mixed_env.out", np.c_[self.mass_profile,self.radius_profile,self.pressure_profile,self.temperature_profile,self.density_profile,self.component_profile], header='Mass (kg) | Radius (m) | Pressure (Pa) | Temperature (K) | Density (kg/m3) | Component_index')
            else:
                np.savetxt(self.output_dir + f"/profile_M_{self.mass:.2f}_P0_{self.P0:.2e}_T0_{self.T0:.2f}_Pad_{self.Pad:.2e}_xg_{self.mass_fractions[3]:.2e}_xw_{self.mass_fractions[2]:.2e}_xmgpv_{self.mass_fractions[1]:.2e}_unmixed_env.out", np.c_[self.mass_profile,self.radius_profile,self.pressure_profile,self.temperature_profile,self.density_profile,self.component_profile], header='Mass (kg) | Radius (m) | Pressure (Pa) | Temperature (K) | Density (kg/m3) | Component_index')

        #Return final Rp
        return Rp_final

    def ode_sys(self,mass,y):
        '''
        Calculate differential equation terms at a given point
        Parameters:
        mass: mass at current point
        y: (pressure, radius) at current point
        '''

        #Unpack y
        p = y[0]
        r = y[1]

        #Find current component
        if mass > self.mass_bds[3]*self.MEarth:
            if self.mixed:
                component_idx = 2
            else:
                component_idx = 3
        elif mass > self.mass_bds[2]*self.MEarth:
            component_idx = 2
        elif mass > self.mass_bds[1]*self.MEarth:
            component_idx = 1
        else:
            component_idx = 0

        #Generate new H2O EOS if necessary
        if component_idx == 2 and self.rho_dict['h2o'] is None:
            lT = np.interp(np.log10(p),self.pressure_grid,self.T_dict['hhe'])
            rho_T = self.eos_data['h2o'].get_eos(self.pressure_grid,np.log10(self.Pad),lT,np.log10(p))
            self.rho_dict['h2o'] = rho_T[0]
            self.T_dict['h2o'] = rho_T[1]
            
        if self.mixed:
            if component_idx > 1.0:
                comp = 'env'
            else:
                comp = self.components[component_idx]
        else:        
            comp = self.components[component_idx]
            
        #Generate new silicate EOS if necessary
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

        #Calculate pressure derivative
        dp_dm = -(self.G*mass)/(4.0*np.pi*r**4)

        #Find non-adiabatic optical depth gradient for Guillot P-T profile
        if self.pt == 'guillot':
            tau = y[2]
            T = y[3]

            kappa = self.eos_data[comp].get_opacity(np.log10(p),np.log10(T))

            if component_idx == 3:
                dtau_dm = -kappa/(4.0*np.pi*r**2)
            else:
                dT_dm = 0.0
                dtau_dm = 0.0

        #Get current density
        if self.pt == 'guillot' and component_idx == 3:
            lrho = self.eos_data[comp].get_density(np.log10(p),np.log10(T))
            rho = 10**lrho
        else:    
            lrho = np.interp(np.log10(p),self.pressure_grid,self.rho_dict[comp])

            rho = 10**lrho

        #Calculate radius derivative
        dr_dm = 1.0/(4.0*np.pi*r**2*rho)

        #Find temperature gradient for Guillot P-T profile
        if component_idx == 3 and self.pt == 'guillot':
            N = 1000.0
            if tau < N*1.0/(self.gamma*3.0**0.5):
                dT_dtau = ((3.0*self.T_int**4/4.0)+((3.0*self.T_irr**4/4.0)*self.f_r*(1.0-self.gamma**2)*(np.exp(-self.gamma*tau*3.0**0.5))))/(4.0*T**3)
                dT_dm_n = dT_dtau*dtau_dm
            else:
                dT_dr = -3.0*kappa*self.Lp*rho/(16.0*4.0*self.sigma_SB*np.pi*(T)**3*r**2)
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

        #Return current properties
        if self.pt == 'guillot':
            return((np.array([dp_dm,dr_dm,dtau_dm,dT_dm]),lrho,T))
        else:
            return((np.array([dp_dm,dr_dm]),lrho))
    
    def rk4(self,f, y0):
        '''
        Implements the 4th order Runge-Kutta method (RK4) for solving ordinary differential equations (ODEs)
        
        Parameters:
        f (function): The function defining the ODE system
        y0 (array): Initial conditions for the ODE system
        
        Returns:
        tuple: Final mass and final values of the dependent variables
        '''

        #Load mass profile (dependent variable)
        t = self.mass_profile

        #Initialise solution array and mass steps
        y = np.zeros((len(t),len(y0)))
        y[0] = y0             
        h = t[1:]-t[:-1]
        self.mass_step = h[0]

        self.T_prescription = 'non-adiabat'
        
        #Initial function evalution
        f0 = f(t[0],y[0])

        #Initialise profiles
        self.component_profile = np.zeros_like(t).astype(int)
        self.component_profile[np.where(t>self.mass_bds[1]*self.MEarth)] = 1
        self.component_profile[np.where(t>self.mass_bds[2]*self.MEarth)] = 2
        if self.mixed:
            self.component_profile[np.where(t>self.mass_bds[3]*self.MEarth)] = 2
        else:
            self.component_profile[np.where(t>self.mass_bds[3]*self.MEarth)] = 3
        
        self.density_profile = np.zeros_like(t)
        self.temperature_profile = np.zeros_like(t)
        self.density_profile[0] = 10**np.copy(f0[1])
        self.temperature_profile[0] = self.Tsurf

        #Iterate over mass steps using RK4
        for i in range(len(t)-1):
            self.mass_step = h[i]
            k1 = h[i]*f0[0]
            k2 = h[i]*f(t[i]+0.5*h[i],y[i]+0.5*k1)[0]
            k3 = h[i]*f(t[i]+0.5*h[i],y[i]+0.5*k2)[0]
            k4 = h[i]*f(t[i]+h[i],y[i]+k3)[0]

            delta_y = (k1+2*k2+2*k3+k4)*(1.0/6.0)
            y[i+1] = y[i] + delta_y
            f1 = f(t[i+1],y[i+1])
            self.density_profile[i+1] = 10**f1[1]

            #Update temperature profile
            if self.pt == 'guillot':
                self.temperature_profile[i+1] = 10**f1[2]
            else:
                    if self.mixed:
                        if self.component_profile[i+1] > 1.0:
                            self.temperature_profile[i+1] = 10**np.interp(np.log10(y[i+1,0]),self.pressure_grid,self.T_dict['env'])
                        else:
                            self.temperature_profile[i+1] = 10**np.interp(np.log10(y[i+1,0]),self.pressure_grid,self.T_dict[self.components[self.component_profile[i+1]]])
                    else:
                        self.temperature_profile[i+1] = 10**np.interp(np.log10(y[i+1,0]),self.pressure_grid,self.T_dict[self.components[self.component_profile[i+1]]])
                
            f0 = f1
                    

            #Return if radius becomes negative
            if y[i+1,1] < 0.0:
                return(t[i+1],y[i+1])

        self.pressure_profile = y[:,0]
        self.radius_profile = y[:,1]

        return(t[-1],y[-1])
