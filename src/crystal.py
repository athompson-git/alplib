# Class for setting and getting crystal parameters, lattice vectors, and structure functions

import numpy as np
from numpy import pi, cross, dot, exp, sum, sqrt

from constants import *


"""
lattice_const: lattice constant in Angstroms
primitives: an array of the N primitive basis vectors in format [alpha0, alpha1,...]
        for each alpha = [#,#,#] as a 3-list
a1, a2, a3: Bravais lattice vectors as 3-lists [#,#,#]
"""
class Crystal:
    def __init__(self, lattice_const, primitives, a1, a2, a3):
        self.a = lattice_const * cm_per_ang
        self.alpha = self.a * np.array(primitives)
        self.a0 = self.a * np.array([0,0,0])
        self.a1 = self.a * np.array(a1)
        self.a2 = self.a * np.array(a2)
        self.a3 = self.a * np.array(a3)
        self.basis = np.array([self.a0, self.a1, self.a2, self.a3])

        self.b1 = 2*pi*cross(self.a2, self.a3) / (dot(self.a1, cross(self.a2, self.a3)))
        self.b2 = 2*pi*cross(self.a3, self.a1) / (dot(self.a1, cross(self.a2, self.a3)))
        self.b3 = 2*pi*cross(self.a1, self.a2) / (dot(self.a1, cross(self.a2, self.a3)))
    
    def r(self, n1, n2, n3):
        return n1 * self.a1 + n2 * self.a2 + n3 * self.a3
    
    def G(self, h, k, l):
        return h * self.b1 + k * self.b2 + l * self.b3
    
    def wavelength(self, h, k, l):
        return 2*pi/sqrt(dot(self.G(h, k, l), self.G(h, k, l)))
    
    def energy(self, h, k, l):
        return kHBARC * sqrt(dot(self.G(h, k, l), self.G(h, k, l)))
    
    def miller(self, h, k, l):
        return np.array([h, k, l])
    
    def sfunc(self, h, k, l):
        return abs((1+exp(-1j * dot(self.alpha[1], self.G(h, k, l)))) \
            * sum([exp(-2*pi*1j*dot(self.miller(h, k, l), avec/self.a)) for avec in self.basis]))



# Namelist
cryslist = ["Ge"]


def get_crystal(name):
    if name not in cryslist:
        print("Specified material not in library. Supported crystals:\n", cryslist)
        return
    
    if name == "Ge":
        # Diamond cubic, germanium
        GeAlpha0 = [0.0, 0.0, 0.0]
        GeAlpha1 = [0.25, 0.25, 0.25]
        GePrimitives = [GeAlpha0, GeAlpha1]
        GeA1 = [0.0, 0.5, 0.5]
        GeA2 = [0.5, 0.0, 0.5]
        GeA3 = [0.5, 0.5, 0.0]
        GeLatticeConst = 5.6585  # angstroms
        return Crystal(GeLatticeConst, GePrimitives, GeA1, GeA2, GeA3)
    
    return




