# Class for setting and getting crystal parameters, lattice vectors, and structure functions

from .materials import Material
from .constants import *
from .fmath import *


"""
Crystal class and methods
Inherits attributes from Material class
lattice_const: lattice constant in Angstroms
primitives: an array of the N primitive basis vectors in format [alpha0, alpha1,...]
        for each alpha = [#,#,#] as a 3-list
a1, a2, a3: Bravais lattice vectors as 3-lists [#,#,#]
"""
class Crystal(Material):
    def __init__(self, material, primitives, a1, a2, a3, volume=1.0, density=1.0,
                fiducial_mass=1.0):
        super().__init__(material_name=material, fiducial_mass=fiducial_mass, volume=volume, density=density)
        self.a = self.lattice_const
        self.alpha = self.a * np.array(primitives)
        self.a0 = self.a * np.array([0,0,0])
        self.a1 = self.a * np.array(a1)
        self.a2 = self.a * np.array(a2)
        self.a3 = self.a * np.array(a3)
        self.basis = np.array([self.a0, self.a1, self.a2, self.a3])

        self.b1 = 2*pi*cross(self.a2, self.a3) / (dot(self.a1, cross(self.a2, self.a3)))
        self.b2 = 2*pi*cross(self.a3, self.a1) / (dot(self.a1, cross(self.a2, self.a3)))
        self.b3 = 2*pi*cross(self.a1, self.a2) / (dot(self.a1, cross(self.a2, self.a3)))

        self.norm = [0.0, 0.0, 1.0]
        self.lx = power(volume, 1/3)
        self.ly = power(volume, 1/3)
        self.lz = power(volume, 1/3)
    
    def r(self, n1, n2, n3):
        return n1 * self.a1 + n2 * self.a2 + n3 * self.a3
    
    def G(self, h, k, l):
        return h * self.b1 + k * self.b2 + l * self.b3
    
    def wavelength(self, h, k, l):
        return 2*pi/sqrt(dot(self.G(h, k, l), self.G(h, k, l)))
    
    def energy(self, h, k, l):
        return HBARC * sqrt(dot(self.G(h, k, l), self.G(h, k, l)))
    
    def miller(self, h, k, l):
        return np.array([h, k, l])
    
    def sfunc(self, h, k, l):
        return abs((1+exp(-1j * dot(self.alpha[1], self.G(h, k, l)))) \
            * sum([exp(-2*pi*1j*dot(self.miller(h, k, l), avec/self.a)) for avec in self.basis]))



# Namelist
cryslist = ["Ge", "Si", "NaI", "CsI", "TeO2"]


def get_crystal(name, volume):
    if name not in cryslist:
        print("Specified material not in library. Supported crystals:\n", cryslist)
        return
    
    if name == "Ge":
        # Diamond cubic, germanium
        GeAlpha0 = [0.0, 0.0, 0.0]
        GeAlpha1 = [0.25, 0.25, 0.25]
        Primitives = [GeAlpha0, GeAlpha1]
        A1 = [0.0, 0.5, 0.5]
        A2 = [0.5, 0.0, 0.5]
        A3 = [0.5, 0.5, 0.0]
        return Crystal(name, Primitives, A1, A2, A3, volume=volume)
    
    if name == "Si":
        # Diamond cubic, sodium iodide
        Alpha0 = [0.0, 0.0, 0.0]
        Alpha1 = [0.25, 0.25, 0.25]
        Primitives = [Alpha0, Alpha1]
        A1 = [0.0, 0.5, 0.5]
        A2 = [0.5, 0.0, 0.5]
        A3 = [0.5, 0.5, 0.0]
        return Crystal(name, Primitives, A1, A2, A3)
    
    if name == "NaI":
        # Diamond cubic, sodium iodide
        Alpha0 = [0.0, 0.0, 0.0]
        Alpha1 = [0.5, 0.5, 0.5]
        Primitives = [Alpha0, Alpha1]
        A1 = [0.0, 0.5, 0.5]
        A2 = [0.5, 0.0, 0.5]
        A3 = [0.5, 0.5, 0.0]
        return Crystal(name, Primitives, A1, A2, A3)

    if name == "CsI":
        # Diamond cubic, cesium iodide
        Alpha0 = [0.0, 0.0, 0.0]
        Alpha1 = [0.5, 0.5, 0.5]
        Primitives = [Alpha0, Alpha1]
        A1 = [0.0, 0.5, 0.5]
        A2 = [0.5, 0.0, 0.5]
        A3 = [0.5, 0.5, 0.0]
        return Crystal(name, Primitives, A1, A2, A3, volume=volume)

    if name == "TeO2":
        # D4(422) space group
        Primatives_O = [[0.998, 0.916, 1.480], [1.481, 3.396, 3.386], [1.399, 3.314, 0.426], [0.916, 0.998, 6.145],
                        [3.797, 3.879, 5.293], [3.314, 1.399, 7.199], [3.396, 1.481, 4.239], [3.879, 3.797, 2.332]]
        Primitives_Te = [[0.093, 0.093, 0.000],[4.702, 4.702, 3.813], [2.304, 2.491 , 1.906], [2.491, 2.304, 5.719]]
        A1 = [0.0, 0.5, 0.5]
        A2 = [0.5, 0.0, 0.5]
        A3 = [0.5, 0.5, 0.0]
        return Crystal(name, Primitives_Te+Primatives_O, A1, A2, A3, volume=volume)

    return

