# Compute Borrmann effect parameters for Crystallographic Scattering

from constants import *
from photon_xs import AbsCrossSection
import crystal

"""
material: string specifying the material/type of crystal, e.g. "Ge", "CsI", etc.
cell_density: No. unit cells per volume
abs_coeff: absorption coefficient in cm^-1
"""
class Borrmann:
    def __init__(self, material, cell_density, abs_coeff):
        self.n = cell_density
        self.abs_xs = AbsCrossSection(material)
        self.mu = abs_coeff
        self.crystal = crystal.get_crystal(material)

    def imff(self, h, k, l):
        energy = self.crystal.energy(h, k, l)
        sigma = self.abs_xs.sigma(energy)
        return energy * sigma / (2 * kHBARC * kRE)

    def debye_waller(self):
        return 1.0

    def sf_ratio(self, h, k, l):  # structure function ratio
        return self.crystal.sfunc(h, k, l)/self.crystal.sfunc(0, 0, 0)

    def zj_etaj_sum(self, h, k, l):
        lam = self.crystal.wavelength(h,k,l)
        return (kME * self.mu) / (2 * kHBARC * kALPHA * lam * self.n)

    def epsilon(self, h, k, l):
        return self.sf_ratio(h, k, l) * self.debye_waller() * self.imff(h, k, l) / self.zj_etaj_sum(h, k, l)

    def anomalous_abs(self, h, k, l):
        return self.mu / (1 - self.epsilon(h, k, l))

    def anomalous_depth(self, h, k, l):
        return 1/self.anomalous_abs(h, k, l)

