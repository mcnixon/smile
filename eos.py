import numpy as np
import params
import h5py
from scipy.interpolate import RegularGridInterpolator

cheat = False

class EOS:
    '''Equation of State for a given component of the planet.'''

    def __init__(self,component,component2=None,pt='isotherm-adiabat',pt_file=None):
        '''
        Creates EOS object with all pressure/adiabatic gradient information.

        -component: name of species ('fe','mgpv','h2o','hhe')
        -datafile: location of EOS data
        -isothermal: whether to create isothermal or T-dependent EOS
        '''

        self.component = component

        self.component2 = component2

        self.pt = pt
        self.pt_file = pt_file

        self.datafile = params.eos_files[self.component]

        if self.component2 is not None:
            self.datafile2 = params.eos_files[self.component2]
            self.datafile_s = params.entropy_files[self.component]
            self.datafile2_s = params.entropy_files[self.component2]

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


        if self.component2 is not None:
            
            if self.datafile2[-1] == '5':
                self.isothermal2 = False
            else:
                self.isothermal2 = True

            if self.isothermal2:
                self.f2 = np.loadtxt(self.datafile2,skiprows=1)
                self.pressure_data2 = np.copy(self.f2[:,0])
                self.rho_data2 = np.copy(self.f2[:,1])
            else:
                self.f2 = h5py.File(self.datafile2,"r")
                self.pressure_data2 = np.copy(self.f2['logP'])
                self.max_P_idx2 = np.argmax(self.pressure_data2)
                self.T_data2 = np.copy(self.f2['logT'])
                self.max_T_idx2 = np.argmax(self.T_data2)
                self.rho_grid2 = np.copy(self.f2['logrho'])
                self.grad_grid2 = np.copy(self.f2['dlogT_dlogP'])
            
                self.grad_interp2 = RegularGridInterpolator((self.pressure_data2,self.T_data2),self.grad_grid2,method='linear',bounds_error=False,fill_value=None)
                self.rho_interp2 = RegularGridInterpolator((self.pressure_data2,self.T_data2),self.rho_grid2,method='linear',bounds_error=False,fill_value=None)

            self.f_s = h5py.File(self.datafile_s,'r')
            self.pressure_data_s = np.copy(self.f_s['logP'])
            self.T_data_s = np.copy(self.f_s['logT'])
            self.S_grid = np.copy(self.f_s['logS'])
            self.dlogS_dlogP_grid = np.copy(self.f_s['dlogS_dlogP'])
            self.dlogS_dlogT_grid = np.copy(self.f_s['dlogS_dlogT'])

            self.S_interp = RegularGridInterpolator((self.pressure_data_s,self.T_data_s),self.S_grid,method='linear',bounds_error=False,fill_value=None)
            self.dlogS_dlogP_interp = RegularGridInterpolator((self.pressure_data_s,self.T_data_s),self.dlogS_dlogP_grid,method='linear',bounds_error=False,fill_value=None)
            self.dlogS_dlogT_interp = RegularGridInterpolator((self.pressure_data_s,self.T_data_s),self.dlogS_dlogT_grid,method='linear',bounds_error=False,fill_value=None)

            self.f2_s = h5py.File(self.datafile2_s,'r')
            self.pressure_data2_s = np.copy(self.f2_s['logP'])
            self.T_data2_s = np.copy(self.f2_s['logT'])
            self.S_grid2 = np.copy(self.f2_s['logS'])
            self.dlogS_dlogP_grid2 = np.copy(self.f2_s['dlogS_dlogP'])
            self.dlogS_dlogT_grid2 = np.copy(self.f2_s['dlogS_dlogT'])

            self.S_interp2 = RegularGridInterpolator((self.pressure_data2_s,self.T_data2_s),self.S_grid2,method='linear',bounds_error=False,fill_value=None)
            self.dlogS_dlogP_interp2 = RegularGridInterpolator((self.pressure_data2_s,self.T_data2_s),self.dlogS_dlogP_grid2,method='linear',bounds_error=False,fill_value=None)
            self.dlogS_dlogT_interp2 = RegularGridInterpolator((self.pressure_data2_s,self.T_data2_s),self.dlogS_dlogT_grid2,method='linear',bounds_error=False,fill_value=None)

                
        if self.pt == 'guillot':
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

    def get_density2(self,logP,logT):
        if self.component2 is not None:
            return self.rho_interp2(np.c_[logP,logT])
        else:
            return None

    def get_S(self,logP,logT):
        if self.component2 is not None:
            return self.S_interp(np.c_[logP,logT])
        else:
            return None
        
    def get_S2(self,logP,logT):
        if self.component2 is not None:
            return self.S_interp2(np.c_[logP,logT])
        else:
            return None

    def get_dlogS_dlogP(self,logP,logT):
        if self.component2 is not None:
            return self.dlogS_dlogP_interp(np.c_[logP,logT])
        else:
            return None
        
    def get_dlogS_dlogP2(self,logP,logT):
        if self.component2 is not None:
            return self.dlogS_dlogP_interp2(np.c_[logP,logT])
        else:
            return None

    def get_dlogS_dlogT(self,logP,logT):
        if self.component2 is not None:
            return self.dlogS_dlogT_interp(np.c_[logP,logT])
        else:
            return None
        
    def get_dlogS_dlogT2(self,logP,logT):
        if self.component2 is not None:
            return self.dlogS_dlogT_interp2(np.c_[logP,logT])
        else:
            return None
            
    #if self.pt =='guillot':
    #    def get_opacity(self,logP,logT):
    #        '''Compute opacity from power-law'''
            #log_opac = self.opac_interp(np.c_[logP,logT])[0]
    #        log_opac = self.C + self.A*logP + self.B*logT
    #        return 10**log_opac

    def get_gradient(self,logP,logT):
        '''Find nearest adiabatic gradient'''
        return self.grad_interp(np.c_[logP,logT])

    def get_density(self,logP,logT):
        '''Find nearest density'''
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

            if self.pt == 'file':
                PT_file = np.loadtxt(self.pt_file)
                logPfile = np.log10(PT_file[:,0])
                logTfile = np.log10(PT_file[:,1])
                T_out = np.interp(logPgrid,logPfile,logTfile)
                logPad = np.amax(logPfile)
            
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

    def get_mixed_eos(self,x2,logPgrid,logPad=None,logTsurf=None,logP_hhb=None):
        '''Compute mixed EOS given component 2 mass fraction, pressure grid, radiative-convective boundary pressure (Pad) and surface T'''

        T_out = np.ones_like(logPgrid)*logTsurf
        step = logPgrid[1] - logPgrid[0]

        if self.pt == 'file':
            PT_file = np.loadtxt(self.pt_file)
            logPfile = np.log10(PT_file[:,0])
            logTfile = np.log10(PT_file[:,1])
            T_out = np.interp(logPgrid,logPfile,logTfile)
            logPad = np.amax(logPfile)

        for i in range(len(logPgrid)-1):
            if logPgrid[i] >= logPad:
                num = (1.0-x2)*(10**self.get_S(logPgrid[i],T_out[i]))*self.get_dlogS_dlogP(logPgrid[i],T_out[i]) + x2*(10**self.get_S2(logPgrid[i],T_out[i]))*self.get_dlogS_dlogP2(logPgrid[i],T_out[i])
                denom = (1.0-x2)*(10**self.get_S(logPgrid[i],T_out[i]))*self.get_dlogS_dlogT(logPgrid[i],T_out[i]) + x2*10**(self.get_S2(logPgrid[i],T_out[i]))*self.get_dlogS_dlogT2(logPgrid[i],T_out[i])
                gradient = -num/denom
                #if logPgrid[i] > 11.0 and logPgrid[i] < 11.2:
                #    print(logPgrid[i],(10**self.get_S2(logPgrid[i],T_out[i])),self.get_dlogS_dlogP2(logPgrid[i],T_out[i]))
                T_out[i+1] = T_out[i] + step*gradient
        logrho_1 = self.get_density(logPgrid,T_out)
        logrho_2 = self.get_density2(logPgrid,T_out)

        rho = 1.0/((1-x2)/(10**(logrho_1)) + x2/(10**logrho_2))
        logrho = np.log10(rho)

        np.savetxt('PTrho.txt',np.c_[logPgrid,logrho,T_out])
        
        return (logrho,T_out)

