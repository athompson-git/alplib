# Compute Borrmann effect parameters for Crystallographic Scattering

from constants import *
from photon_xs import AbsCrossSection
import crystal

"""
material: string specifying the material/type of crystal, e.g. "Ge", "CsI", etc.
cell_density: No. unit cells per volume cm^-3
abs_coeff: absorption coefficient in cm^-1
"""
class Borrmann:
    def __init__(self, material, density, abs_coeff, verbose=False):
        self.n = density
        self.abs_xs = AbsCrossSection(material)
        self.mu = abs_coeff
        self.crystal = crystal.get_crystal(material)
        self.verbose = verbose

    def imff(self, h, k, l):
        energy = self.crystal.energy(h, k, l)
        sigma = self.abs_xs.sigma(energy)
        imff = energy * sigma / (2 * kHC * kRE)
        
        if self.verbose == True:
            print("    imff = ", imff)
        return imff

    def debye_waller(self):
        return 1.0

    def sf_ratio(self, h, k, l):  # structure function ratio
        return self.crystal.sfunc(h, k, l)/self.crystal.sfunc(0, 0, 0)

    def zj_etaj_sum(self, energy):
        lam = kHC / energy
        mu = self.n * self.abs_xs.sigma(energy)
        ZjEtaj = (kME * mu) / (2 * kHBARC * kALPHA * lam * (self.n/4))

        if self.verbose == True:
            print("    mu = ", mu)
            print("    sum(ZjEtaj) = ", ZjEtaj)
        return ZjEtaj

    def epsilon(self, energy, h, k, l):
        return self.sf_ratio(h, k, l) * self.debye_waller() * self.imff(h, k, l) / self.zj_etaj_sum(energy)

    def anomalous_abs(self, energy, h, k, l):
        mu = self.n * self.abs_xs.sigma(energy)
        return mu / (1 - self.epsilon(energy, h, k, l))

    def anomalous_depth(self, energy, h, k, l):
        return 1/self.anomalous_abs(energy, h, k, l)

