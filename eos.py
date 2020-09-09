import numpy as np
import params
import h5py
from scipy.interpolate import RegularGridInterpolator

cheat = False

class EOS:
    '''Equation of State for a given component of the planet.'''

    def __init__(self,component):
        '''
        Creates EOS object with all pressure/adiabatic gradient information.

        -component: name of species ('fe','mgpv','h2o','hhe')
        -datafile: location of EOS data
        -isothermal: whether to create isothermal or T-dependent EOS
        '''

        self.component = component

        self.datafile = params.eos_files[self.component]

        #check for hdf5 formatted (non-isothermal) EOS
        if self.datafile[-1] == '5':
            self.isothermal = False
        else:
            self.isothermal = True

        if self.isothermal:
            self.f = np.loadtxt(self.datafile,skiprows=1)
            self.pressure_data = np.copy(self.f[:,0])
            self.rho_data = np.copy(self.f[:,1])
        else:
            self.f = h5py.File(self.datafile,"r")
            self.pressure_data = np.copy(self.f['logP'])
            self.max_P_idx = np.argmax(self.pressure_data)
            self.T_data = np.copy(self.f['logT'])
            self.max_T_idx = np.argmax(self.T_data)
            self.rho_grid = np.copy(self.f['logrho'])
            self.grad_grid = np.copy(self.f['dlogT_dlogP'])
            
            self.grad_interp = RegularGridInterpolator((self.pressure_data,self.T_data),self.grad_grid,method='linear',bounds_error=False,fill_value=None)
            self.rho_interp = RegularGridInterpolator((self.pressure_data,self.T_data),self.rho_grid,method='linear',bounds_error=False,fill_value=None)

        if params.pt == 'Guillot':
            if self.component == 'hhe':
                self.gamma = 1.0
                self.C = -7.32
                self.A = 0.68
                self.B = 0.45
            elif self.component == 'h2o':
                self.gamma = 0.01
                self.C = -17.59
                self.A = 0.835
                self.B = 4.102
            else:
                self.gamma = 0.0
                self.C = 0.0
                self.A = 0.0
                self.B = 0.0

            self.opac_f = h5py.File("opac.hdf5","r")
            self.Pgrid_o = np.copy(self.opac_f["logP"])
            self.Tgrid_o = np.copy(self.opac_f["logT"])
            self.opac_grid = np.copy(self.opac_f["log_opac"])
            #self.opac_interp = RegularGridInterpolator((self.Pgrid_o,self.Tgrid_o),self.opac_grid,method='nearest',bounds_error=False,fill_value=None)

    if params.pt =='Guillot':
        def get_opacity(self,logP,logT):
            '''Compute opacity from power-law'''
            #log_opac = self.opac_interp(np.c_[logP,logT])[0]
            log_opac = self.C + self.A*logP + self.B*logT
            return 10**log_opac

    def get_gradient(self,logP,logT):
        '''Find nearest adiabatic gradient'''
        #logP_idx = (logP-self.pressure_data[0])/(self.pressure_data[1]-self.pressure_data[0])
        #logP_idx = np.around(logP_idx).astype(int)
        #logP_idx = logP_idx.astype(int)
        #if logP_idx > self.max_P_idx:
        #    logP_idx = self.max_P_idx
        #if self.T_data[-1] == 8.0:
        #    logT_idx = (logT-self.T_data[0])/(self.T_data[1]-self.T_data[0])
        #else:
        #    logT_idx = (10**logT-10**self.T_data[0])/(10**self.T_data[1]-10**self.T_data[0])
        #logT_idx = np.around(logT_idx).astype(int)
        #logT_idx = logT_idx.astype(int)
        #if logT_idx > self.max_T_idx:
        #    logT_idx = self.max_T_idx

        #if cheat:
        #    return self.grad_grid[logP_idx,logT_idx]

        #P_over = logP >= np.amax(self.pressure_data)
        #T_over = logT >= np.amax(self.T_data)

        #if P_over and T_over:
        #    return self.grad_grid[-1,-1]
        #elif P_over:
        #    return np.interp(logT,self.T_data,self.grad_grid[-1,:])
        #elif T_over:
        #    return np.interp(logP,self.pressure_data,self.grad_grid[:,-1])
        #else:

        #    f_P_T1 = np.interp(logP,self.pressure_data,self.grad_grid[:,logT_idx])
        #    f_P_T2 = np.interp(logP,self.pressure_data,self.grad_grid[:,logT_idx+1])

        #    return np.interp(logT,self.T_data[logT_idx:logT_idx+2],np.array([f_P_T1,f_P_T2]))
        #return self.grad_grid[logP_idx,logT_idx]
        return self.grad_interp(np.c_[logP,logT])

    def get_density(self,logP,logT):
        '''Find nearest density'''
        #logP_idx = (logP-self.pressure_data[0])/(self.pressure_data[1]-self.pressure_data[0])
        #logP_idx = np.around(logP_idx).astype(int)
        #logP_idx = logP_idx.astype(int)

        #logP_idx[np.where(logP_idx>self.max_P_idx)] = self.max_P_idx
        
        #if logP_idx > self.max_P_idx:
        #    logP_idx = self.max_P_idx
        #if self.T_data[-1] == 8.0:
        #    logT_idx = (logT-self.T_data[0])/(self.T_data[1]-self.T_data[0])
        #else:
        #    logT_idx = (10**logT-10**self.T_data[0])/(10**self.T_data[1]-10**self.T_data[0])
        #logT_idx = np.around(logT_idx).astype(int)
        #logT_idx = logT_idx.astype(int)
        #if logT_idx > self.max_T_idx:
        #    logT_idx = self.max_T_idx
        #logT_idx[np.where(logT_idx>self.max_T_idx)] = self.max_T_idx
        #logT_idx[np.where(logT_idx<0)] = 0

        #if cheat:
        #    return(self.rho_grid[logP_idx,logT_idx])
            
        #P_over = logP_idx >= np.argmax(self.pressure_data)
        #T_over = logT_idx >= np.argmax(self.T_data)

        #if P_over and T_over:
        #    rho = self.rho_grid[-1,1]
            #return self.rho_grid[-1,-1]
        #elif P_over:
        #    rho = np.interp(logT,self.T_data,self.rho_grid[-1,:])
            #return np.interp(logT,self.T_data,self.rho_grid[-1,:])
        #elif T_over:
        #    rho = np.interp(logP,self.pressure_data,self.rho_grid[:,-1])
            #return np.interp(logP,self.pressure_data,self.rho_grid[:,-1])
        #else:
            #print(logP,logT_idx,self.rho_grid[:,logT_idx].shape)
            #f_P_T1 = np.interp(logP,self.pressure_data,self.rho_grid[:,logT_idx])
            #f_P_T2 = np.interp(logP,self.pressure_data,self.rho_grid[:,logT_idx+1])
            #rho = np.interp(logT,self.T_data[logT_idx:logT_idx+2],np.array([f_P_T1,f_P_T2]))
            #return np.interp(logT,self.T_data[logT_idx:logT_idx+2],np.array([f_P_T1,f_P_T2]))
        #return rho
            
        #return self.rho_grid[logP_idx,logT_idx]
        #rho2 = self.rho_interp(np.c_[logP,logT])
        return self.rho_interp(np.c_[logP,logT])
   
    def get_eos_old(self,logPgrid,logPad=None,logTsurf=None,logP_hhb=None):
        '''Compute EOS given a pressure grid, radiative-convective boundary pressure (Pad) and surface T'''

        if self.isothermal:
            logrho = np.interp(logPgrid,self.pressure_data,self.rho_data)
            T_out = np.zeros_like(logrho)
        else:
            logrho = np.zeros_like(logPgrid)
            logT = logTsurf

            T_out = np.zeros_like(logrho) 
            step = logPgrid[1]-logPgrid[0]#requires even log-spaced grid

            for i in range(len(logPgrid)):
                logT_for_rho = np.copy(logT)

                T_out[i] = logT_for_rho

                #dealing with out of bounds temperatures
                if logT_for_rho > np.amax(self.T_data):
                    logT_for_rho = np.amax(self.T_data)
                if logT_for_rho < np.amin(self.T_data):
                    logT_for_rho = np.amin(self.T_data)

                #print(logPgrid[i],logT_for_rho)
                logrho[i] = self.get_density(logPgrid[i],T_out[i])

                if logP_hhb is None:
                    if logPgrid[i] >= logPad:
                        gradient = self.get_gradient(logPgrid[i],T_out[i])
                        logT += step*gradient
                else:
                    if logPgrid[i] >= logP_hhb:
                        gradient = self.get_gradient(logPgrid[i],T_out[i])
                        logT += step*gradient

        return (logrho, T_out)

    def get_eos(self,logPgrid,logPad=None,logTsurf=None,logP_hhb=None):
        '''Compute EOS given a pressure grid, radiative-convective boundary pressure (Pad) and surface T'''

        if self.isothermal:
            logrho = np.interp(logPgrid,self.pressure_data,self.rho_data)
            T_out = np.zeros_like(logrho)
        else:
            T_out = np.ones_like(logPgrid)*logTsurf
            step = logPgrid[1] - logPgrid[0]
            for i in range(len(logPgrid)-1):
                if logP_hhb is None:
                    if logPgrid[i] >= logPad:
                        gradient = self.get_gradient(logPgrid[i],T_out[i])
                        T_out[i+1] = T_out[i] + step*gradient
                else:
                    if logPgrid[i] >= logP_hhb:
                        gradient = self.get_gradient(logPgrid[i],T_out[i])
                        T_out[i+1] = T_out[i] + step*gradient
            logrho = self.get_density(logPgrid,T_out)
        return (logrho, T_out)
