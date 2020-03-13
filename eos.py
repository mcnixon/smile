import numpy as np
import params
import h5py
from scipy.interpolate import RegularGridInterpolator

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
            self.T_data = np.copy(self.f['logT'])
            self.rho_grid = np.copy(self.f['logrho'])
            self.grad_grid = np.copy(self.f['dlogT_dlogP'])
            
            self.grad_interp = RegularGridInterpolator((self.pressure_data,self.T_data),self.grad_grid,method='nearest')
            self.rho_interp = RegularGridInterpolator((self.pressure_data,self.T_data),self.rho_grid,method='nearest')

    def __get_gradient(self,logP,logT):
        '''Find nearest adiabatic gradient'''
        return self.grad_interp(np.c_[logP,logT])

    def __get_density(self,logP,logT):
        '''Find nearest density'''
        return self.rho_interp(np.c_[logP,logT])

    def get_eos(self,logPgrid,logPad=None,logTsurf=None):
        '''Compute EOS given a pressure grid, radiative-convective boundary pressure (Pad) and surface T'''

        if self.isothermal:
            logrho = np.interp(logPgrid,self.pressure_data,self.rho_data)
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
                logrho[i] = self.__get_density(logPgrid[i],logT_for_rho)

                if logPgrid[i] >= logPad:
                    gradient = self.__get_gradient(logPgrid[i],logT_for_rho)
                    logT += step*gradient

        return logrho
