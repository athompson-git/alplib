"""
Define form factors
"""

from alplib.materials import Material
from .constants import *
from .fmath import *
from scipy.special import spherical_jn




def nuclear_ff(t, m, z, a):
    # Parameterization of the coherent nuclear form factor (Tsai 1986, B49)
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    # a: number of nucleons
    return (2*m*z**2) / (1 + t / 164000*np.power(a, -2/3))**2




def atomic_elastic_ff(t, z):
    # Coherent atomic form factor parameterization (Tsai 1986, B38)
    # Fit based on Thomas-Fermi model
    # t: MeV
    # z: atomic number
    a = 184.15*np.power(2.718, -1/2)*np.power(z, -1/3) / M_E
    return (z*t*a**2)**2 / (1 + t*a**2)**2



class AtomicElasticFF:
    """
    square of the form factor
    """
    def __init__(self, material: Material):
        self.z = material.z
        self.frac = material.frac

    def __call__(self, q):
        t = q**2
        a = 184.15*np.power(2.718, -1/2)*np.power(self.z, -1/3) / M_E
        return np.dot(self.frac, power(self.z*(t*a**2) / (1 + t*a**2), 2))





class ElectronElasticFF:
    """
    square of the form factor
    """
    def __init__(self, material: Material):
        self.z = material.z
        self.frac = material.frac

    def __call__(self, q):
        t = q**2
        a = 184.15*np.power(2.718, -1/2)*np.power(self.z, -1/3) / M_E
        return np.dot(self.frac, power(self.z*(t*a**2) / (1 + t*a**2) - self.z, 2))





class NuclearHelmFF:
    """
    square of the form factor
    """
    def __init__(self, material: Material):
        self.rn = 4.7*((material.n[0]+material.z[0])/133)**(1/3)
        self.z = material.z[0]
        self.frac = material.frac

    def __call__(self, q):
        r = self.rn * (10 ** -15) / METER_BY_MEV
        s = 0.9 * (10 ** -15) / METER_BY_MEV
        r0 = sqrt(5 / 3 * (r ** 2) - 5 * (s ** 2))
        # TODO: incorporate full sum over elements in compound materials
        #return np.dot(self.frac, (self.z * 3*spherical_jn(1, q*r0) / (q*r0) * exp((-(q*s)**2)/2))**2)
        return (self.z * 3*spherical_jn(1, q*r0) / (q*r0) * exp((-(q*s)**2)/2))**2




class ProtonFF:
    """
    Square of the proton form factor F1
    """
    def __init__(self):
        pass

    def __call__(self, t):
        g_e = power(1 - t/0.71e6, -2)
        return power((g_e - t/(4*M_P**2))/(1 - t/(4*M_P**2)), 2)




def _screening(e, ma):
    if ma == 0:
        return 0
    r0 = 1/0.001973  # 0.001973 MeV A -> 1 A (Ge) = 1/0.001973
    x = (r0 * ma**2 / (4*e))**2
    numerator = 2*log(2*e/ma) - 1 - exp(-x) * (1 - exp(-x)/2) + (x + 0.5)*exp1(2*x) - (1+x)*exp1(x)
    denomenator = 2*log(2*e/ma) - 1
    return numerator / denomenator