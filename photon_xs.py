# Get the total photon absorption cross-section by element

import pkg_resources

from .materials import Material
from .constants import *
from .fmath import *



"""
Returns the total photon absorption cross-section in cm2 as a function of E in MeV.
material: Material class specifying the material, e.g. "Ge", "CsI", etc.
Data taken from NIST XCOM database.
"""


class AbsCrossSection:
    def __init__(self, material: Material):
        self.xs_dim = 1e-24  # barns to cm2
        self.mat_name = material.mat_name
        self.path_prefix = "data/photon_absorption/photon_abs_"
        self.file_extension = ".txt"
        fpath = pkg_resources.resource_filename(__name__, self.path_prefix + self.mat_name + self.file_extension)
        self.pe_data = np.genfromtxt(fpath, skip_header=3)

        if self.mat_name == "NaI":
            self.xs_dim = 149.89 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "CsI":
            self.xs_dim = 259.81 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "CH2":
            self.xs_dim = 14.027 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "N2":
            self.xs_dim = 28.0 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "O2":
            self.xs_dim = 32.0 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "TeO2":
            self.xs_dim = 32.0 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        
        self.cleanPEData()
    
    def cleanPEData(self):
        self.pe_data = self.pe_data[np.unique(self.pe_data[:, 0], return_index=True)[1]]

    def sigma_cm2(self, E):
        return 10**np.interp(log10(E), log10(self.pe_data[:,0]), log10(self.xs_dim * self.pe_data[:,1]), left=0.0, right=0.0)
    
    def sigma_mev(self, E):
        return 10**np.interp(log10(E), log10(self.pe_data[:,0]), log10(self.xs_dim * self.pe_data[:,1]), left=0.0, right=0.0) / MEV2_CM2
    
    def mu(self, E, n):  # atomic number density in cm^-3
        return self.sigma_cm2(E) * n



class PairProdutionCrossSection:
    def __init__(self, material: Material):
        self.xs_dim = 1e-24  # barns to cm2
        self.mat_name = material.mat_name
        self.path_prefix = "data/photon_pair_production/pair_production_xs_"
        self.file_extension = ".txt"
        fpath = pkg_resources.resource_filename(__name__, self.path_prefix + self.mat_name + self.file_extension)
        self.xs_data = np.genfromtxt(fpath, skip_header=3)

        if self.mat_name == "NaI":
            self.xs_dim = 149.89 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "CsI":
            self.xs_dim = 259.81 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "CH2":
            self.xs_dim = 14.027 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "TeO2":
            self.xs_dim = 14.027 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        
        self.cleanPEData()
    
    def cleanPEData(self):
        self.xs_data = self.xs_data[np.unique(self.xs_data[:, 0], return_index=True)[1]]

    def sigma_cm2(self, E):
        return heaviside(E-2*M_E,0.0) * \
            power(10, np.interp(log10(E), log10(self.xs_data[:,0]),
                    log10(self.xs_dim * self.xs_data[:,1]), left=-np.inf))
    
    def sigma_mev(self, E):
        return heaviside(E-2*M_E,0.0) * \
            power(10, np.interp(log10(E), log10(self.xs_data[:,0]), 
                    log10(self.xs_dim * self.xs_data[:,1] / MEV2_CM2), left=-np.inf))
    
    def mu(self, E, n):  # atomic number density in cm^-3
        return self.sigma_cm2(E) * n




class ALPPairProdutionCrossSection:
    def __init__(self, material: Material, ma=0.1):
        self.xs_dim = 1e-24  # barns to cm2
        self.mat_name = material.mat_name
        # takes in data in MeV, barns
        self.path_prefix = "data/alp_pair_production/pair_production_xs_table_"
        
        mass_extension = ["100keV", "400keV", "500keV", "600keV", "700keV", "800keV", "900keV", "1MeV", "1.1MeV"]
        mass_control_points = [0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        mass_idx = np.arange(0, 9, 1)
        idx_closest = np.clip(int(np.interp(ma, mass_control_points, mass_idx)), a_min=0, a_max=9)
        mass_str = mass_extension[idx_closest]

        self.argon_rescale = power(material.z[0]/18, 2)
        
        self.file_extension = ".txt"
        fpath = pkg_resources.resource_filename(__name__, self.path_prefix + mass_str + self.file_extension)
        self.xs_data = np.genfromtxt(fpath)

    def sigma_cm2(self, E):
        return heaviside(E-2*M_E,0.0) * \
            power(10, np.interp(log10(E), log10(self.xs_data[:,0]),
                    log10(self.argon_rescale * self.xs_dim * self.xs_data[:,1]), left=-np.inf))
    
    def sigma_mev(self, E):
        return heaviside(E-2*M_E,0.0) * \
            power(10, np.interp(log10(E), log10(self.xs_data[:,0]), 
                    log10(self.argon_rescale * self.xs_dim * self.xs_data[:,1] / MEV2_CM2), left=-np.inf))
    
    def mu(self, E, n):  # atomic number density in cm^-3
        return self.sigma_cm2(E) * n




class ComptonCrossSection:
    def __init__(self, material: Material):
        self.xs_dim = 1e-24  # barns to cm2
        self.mat_name = material.mat_name
        self.path_prefix = "data/photon_compton/compton_xs_"
        self.file_extension = ".txt"
        fpath = pkg_resources.resource_filename(__name__, self.path_prefix + self.mat_name + self.file_extension)
        self.xs_data = np.genfromtxt(fpath, skip_header=3)
        
        self.cleanPEData()
    
    def cleanPEData(self):
        self.xs_data = self.xs_data[np.unique(self.xs_data[:, 0], return_index=True)[1]]

    def sigma_cm2(self, E):
        return 10**np.interp(log10(E), log10(self.xs_data[:,0]), log10(self.xs_dim * self.xs_data[:,1]))
    
    def sigma_mev(self, E):
        return 10**np.interp(log10(E), log10(self.xs_data[:,0]), log10(self.xs_dim * self.xs_data[:,1] / MEV2_CM2))
    
    def mu(self, E, n):  # atomic number density in cm^-3
        return self.sigma_cm2(E) * n




class PECrossSection:
    def __init__(self, material: Material):
        self.xs_dim = 1e-24  # barns to cm2
        self.mat_name = material.mat_name
        self.path_prefix = "data/photoelectric/pe_xs_"
        self.file_extension = ".txt"
        fpath = pkg_resources.resource_filename(__name__, self.path_prefix + self.mat_name + self.file_extension)
        self.pe_data = np.genfromtxt(fpath, skip_header=3)

        if self.mat_name == "NaI":
            self.xs_dim = 149.89 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "CsI":
            self.xs_dim = 259.81 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "CH2":
            self.xs_dim = 14.027 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "N2":
            self.xs_dim = 28.0 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        elif self.mat_name == "O2":
            self.xs_dim = 32.0 / AVOGADRO  # (cm2 / g  * g / mol  * mol / N)
        
        self.cleanPEData()
    
    def cleanPEData(self):
        self.pe_data = self.pe_data[np.unique(self.pe_data[:, 0], return_index=True)[1]]

    def sigma_cm2(self, E):
        return 10**np.interp(log10(E), log10(self.pe_data[:,0]), log10(self.xs_dim * self.pe_data[:,1]), left=0.0, right=0.0)
    
    def sigma_mev(self, E):
        return 10**np.interp(log10(E), log10(self.pe_data[:,0]), log10(self.xs_dim * self.pe_data[:,1]), left=0.0, right=0.0) / MEV2_CM2
    
    def mu(self, E, n):  # atomic number density in cm^-3
        return self.sigma_cm2(E) * n