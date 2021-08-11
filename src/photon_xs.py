# Get the total photon absorption cross-section by element
import numpy as np
from numpy import genfromtxt, interp, log10

from constants import *



"""
Returns the total photon absorption cross-section in cm2 as a function of E in MeV.
material: string specifying the target material, e.g. "Ge", "CsI", etc.
"""
class AbsCrossSection:
    def __init__(self, material):
        #self.pe_data = np.empty()
        self.xs_dim = 1e-24  # barns to cm2
        if material == "H":
            self.pe_data = np.genfromtxt("../data/photon_absorption/photon_abs_H.txt", skip_header=3)
        elif material == "Be":
            self.pe_data = np.genfromtxt("../data/photon_absorption/photon_abs_Be.txt", skip_header=3)
        elif material == "C":
            self.pe_data = np.genfromtxt("../data/photon_absorption/photon_abs_C.txt", skip_header=3)
        elif material == "Si":
            self.pe_data = np.genfromtxt("../data/photon_absorption/photon_abs_Si.txt", skip_header=3)
        elif material == "Ge":
            self.pe_data = np.genfromtxt("../data/photon_absorption/photon_abs_Ge.txt", skip_header=3)
        elif material == "Ar":
            self.pe_data = np.genfromtxt("../data/photon_absorption/photon_abs_Ar.txt", skip_header=3)
        elif material == "Xe":
            self.pe_data = np.genfromtxt("../data/photon_absorption/photon_abs_Xe.txt", skip_header=3)
        elif material == "NaI":
            self.xs_dim = 149.89 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
            self.pe_data = np.genfromtxt("../data/photon_absorption/photon_abs_NaI.txt", skip_header=3)
        elif material == "CsI":
            self.xs_dim = 259.81 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
            self.pe_data = np.genfromtxt("../data/photon_absorption/photon_abs_CsI.txt", skip_header=3)
        else:
            print("Material unknown or not in library.")
        
        self.cleanPEData()
    
    def cleanPEData(self):
        self.pe_data = self.pe_data[np.unique(self.pe_data[:, 0], return_index=True)[1]]

    def sigma(self, E):
        return 10**np.interp(log10(E), log10(self.pe_data[:,0]), log10(self.xs_dim * self.pe_data[:,1]))
    
    def mu(self, E, n):  # atomic number density in cm^-3
        return self.sigma(E) * n
