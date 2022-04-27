# Compute Borrmann effect parameters for Crystallographic Scattering

import pkg_resources

from .constants import *
from .fmath import *
from .photon_xs import AbsCrossSection
from .crystal import *


# read in the ff data




ZjEtaj_L1 = 0.407
ZjEtaj_L23 = 0.555
ZjEtaj_M23 = 0.090






"""
material: string specifying the material/type of crystal, e.g. "Ge", "CsI", etc.
cell_density: No. unit cells per volume cm^-3
abs_coeff: absorption coefficient in cm^-1
"""
class Borrmann:
    def __init__(self, material: Material, verbose=False):
        ge_l1_path = pkg_resources.resource_filename(__name__, "data/borrmann/Ge_L1_f.txt")
        ge_l23_path = pkg_resources.resource_filename(__name__, "data/borrmann/Ge_L23_f.txt")
        ge_m23_path = pkg_resources.resource_filename(__name__, "data/borrmann/Ge_M23_f.txt")
        self.ge_l1 = np.genfromtxt(ge_l1_path, delimiter=",")
        self.ge_l23 = np.genfromtxt(ge_l23_path, delimiter=",")
        self.ge_m23 = np.genfromtxt(ge_m23_path, delimiter=",")
        self.n = material.ndensity # cm^-3
        self.abs_xs = AbsCrossSection(material)
        self.crystal = get_crystal(material.mat_name, volume=1000)
        self.verbose = verbose

    def imff(self, h, k, l):
        energy = self.crystal.energy(h, k, l)
        sigma = self.abs_xs.sigma_cm2(energy*1e-3)
        imff = energy * sigma / (2 * HC * R_E)
        
        if self.verbose == True:
            print("    imff = ", imff)
        return imff

    def debye_waller(self):
        return 1.0

    def sf_ratio(self, h, k, l):  # structure function ratio
        return self.crystal.sfunc(h, k, l)/self.crystal.sfunc(0, 0, 0)

    def zj_etaj_sum(self, energy):
        lam = HC / energy
        mu = self.n * self.abs_xs.sigma_cm2(energy*1e-3)
        ZjEtaj = (M_E * mu) / (2 * HBARC * ALPHA * lam * (self.n/4))

        if self.verbose == True:
            print("    mu = ", mu)
            print("    sum(ZjEtaj) = ", ZjEtaj)
        return ZjEtaj
    
    def f_L1(self, sinThetaByLambda):
        return np.interp(sinThetaByLambda, self.ge_l1[:,0], self.ge_l1[:,1])

    def f_L23(self, sinThetaByLambda):
        return np.interp(sinThetaByLambda, self.ge_l23[:,0], self.ge_l23[:,1])

    def f_M23(self, sinThetaByLambda):
        return np.interp(sinThetaByLambda, self.ge_m23[:,0], self.ge_m23[:,1])

    def epsilon(self, energy, h, k, l):
        gvec = self.crystal.G(h, k, l)
        sinThetaByLambda = sqrt(np.dot(gvec, gvec))/4/pi
        debye = 0.981
        l1 =  ZjEtaj_L1*self.f_L1(sinThetaByLambda)
        l23 = ZjEtaj_L23*self.f_L23(sinThetaByLambda)
        m23 = ZjEtaj_M23*self.f_M23(sinThetaByLambda)
        denominator = ZjEtaj_L1 + ZjEtaj_L23 + ZjEtaj_M23
        epsilon = debye * (l1 + l23 + m23)/denominator
        return epsilon

    #def epsilon(self, energy, h, k, l):
    #    return self.sf_ratio(h, k, l) * self.debye_waller() * self.imff(h, k, l) / self.zj_etaj_sum(energy)

    def anomalous_abs(self, energy, h, k, l):
        mu = self.n * self.abs_xs.sigma_cm2(1e-3*energy)
        return mu * (1 - self.epsilon(energy, h, k, l))

    def anomalous_depth(self, energy, h, k, l):
        return 1/self.anomalous_abs(energy, h, k, l)


# Class for calculating the atomic form factors, shell-by-shell
class HydrogenicWaveFunction:
    def __init__(self, n=0, l=0, m=0):
        pass

    def radial_wf(self, r, n=0, l=0):
        pass

    def spherical_harmonic(self, theta, phi, l=0, m=0):
        pass

    def integral(self):
        pass




