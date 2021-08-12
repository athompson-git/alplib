# Detection Cross Sections
# All cross sections in cm^2
# All energies in MeV
from .constants import *
from .fmath import *

import multiprocessing as multi
from matplotlib.pyplot import hist2d


# Define ALP DETECTION cross-sections

#### Photon Coupling ####

def primakoff_scattering_diffxs(theta, ea, g, ma, z, r0):
    # inverse-Primakoff scattering differential xs by theta
    # r0: screening parameter
    if ea < ma:
        return 0.0
    prefactor = (g * z)**2 / (2*137)
    q2 = -2*ea**2 + ma**2 + 2*ea*sqrt(ea**2 - ma**2)*cos(theta)
    beta = sqrt(ea**2 - ma**2)/ea
    return prefactor * (1 - exp(q2 * r0**2 / 4))**2 * (beta * sin(theta)**3)/(1+beta**2 - 2*beta*cos(theta))**2




def primakoff_scattering_xs_ntotal(ea, g, ma, z, r0):
    # inverse-Primakoff scattering total xs (numerically integrated)
    # r0: screening parameter
    return quad(primakoff_scattering_diffxs, 0, pi, args=(ea,g,ma,z,r0))[0]




def primakoff_scattering_xs(ea, g, ma, z, r0):
    # inverse-Primakoff scattering total xs (Creswick et al)
    # r0: screening parameter
    if ea < ma:
        return 0.0
    prefactor = (g * z)**2 / (2*137)
    eta2 = r0**2 * (ea**2 - ma**2)
    return prefactor * (((2*eta2 + 1)/(4*eta2))*log(1+4*eta2) - 1)




#### Electron Coupling ####

def axioelectric_xs(pe_xs, energy, z, a, g, ma):
    # Axio-electric total cross section for ionization
    pe = np.interp(energy, pe_xs[:,0], pe_xs[:,1])*1e-24 / (100*METER_BY_MEV)**2
    beta = sqrt(energy**2 - ma**2)
    return 137 * 3 * g**2 * pe * energy**2 * (1 - np.power(beta, 2/3)/3) / (16*pi*M_E**2 * beta)




def compton_scattering_xs(ea, g):
    # Total cross section (a + e- -> \gamma + e-)
    a = 1 / 137
    aa = g ** 2 / 4 / pi
    prefact = a * aa * pi / 2 / M_E / ea**2
    sigma = prefact * 2 * ea * (-(2*ea * (3*ea + M_E)/(2 * ea + M_E)**2) + np.log(2 * ea / M_E + 1))
    return sigma




def compton_scattering_he_dSdEt(ea, et, g):
    # Differential cross section by electroin recoil, HE approximation (a + e- -> \gamma + e-)
    a = 1 / 137
    aa = g ** 2 / 4 / pi
    prefact = a * aa * pi / 2 / M_E
    sigma = prefact * (et ** 2) / ((et ** 3) * (ea - et))
    return sigma




def compton_scattering_dSdEt(ea, et, g, ma):
    # Differential cross section by electron recoil (a + e- -> \gamma + e-)
    # dSigma / dEt   electron kinetic energy
    # ea: axion energy
    # et: transferred electron energy = E_e - m_e.
    y = 2 * M_E * ea + ma ** 2
    prefact = (1/137) * g ** 2 / (4 * M_E ** 2)
    pa = np.sqrt(ea ** 2 - ma ** 2)
    eg = ea - et
    return -(prefact / pa) * (1 - (8 * M_E * eg / y) + (12 * (M_E * eg / y) ** 2)
                                - (32 * M_E * (pa * ma) ** 2) * eg / (3 * y ** 3))




# Define form factors

def _nuclear_ff(t, m, z, a):
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    # a: number of nucleons
    return (2*m*z**2) / (1 + t / 164000*np.power(a, -2/3))**2



def _atomic_elastic_ff(t, m, z):
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    b = 184*np.power(2.718, -1/2)*np.power(z, -1/3) / M_E
    return (z*t*b**2)**2 / (1 + t*b**2)**2



def _atomic_elastic_ff(t, m, z):
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    b = 1194*np.power(2.718, -1/2)*np.power(z, -2/3) / M_E
    return (z*t*b**2)**2 / (1 + t*b**2)**2



def _screening(e, ma):
    if ma == 0:
        return 0
    r0 = 1/0.001973  # 0.001973 MeV A -> 1 A (Ge) = 1/0.001973
    x = (r0 * ma**2 / (4*e))**2
    numerator = 2*log(2*e/ma) - 1 - exp(-x) * (1 - exp(-x)/2) + (x + 0.5)*exp1(2*x) - (1+x)*exp1(x)
    denomenator = 2*log(2*e/ma) - 1
    return numerator / denomenator
