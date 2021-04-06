# Detection Cross Sections
# All cross sections in cm^2
# All energies in MeV
import sys
from constants import *

import math
import multiprocessing as multi

from matplotlib.pyplot import hist2d

from numpy import log, log10, pi, exp, sin, cos, sin, sqrt, arccos, heaviside
from scipy.special import exp1
from scipy.integrate import quad, dblquad, cumtrapz
from scipy.optimize import fsolve

import mpmath as mp
from mpmath import mpmathify, fsub, fadd
mp.dps = 15



# Define ALP production cross-sections
def dSigmadt_primakoff_free(t, s, ma, M, g):
    num = kALPHA * g**2 * (t*(M**2 + s)*ma**2 - (M * ma**2)**2 - t*((s-M**2)**2 + s*t) - t*(t-ma**2)/2)
    denom = 4*t**2 * ((M + ma)**2 - s)*((M - ma)**2 - s)
    return heaviside(num/denom, 0.0) * (num / denom)


def primakoff_scattering_diffxs(theta, ea, g, ma, z, r0):
    if ea < ma:
        return 0.0
    prefactor = (g * z)**2 / (2*137)
    q2 = -2*ea**2 + ma**2 + 2*ea*sqrt(ea**2 - ma**2)*cos(theta)
    beta = sqrt(ea**2 - ma**2)/ea
    return prefactor * (1 - exp(q2 * r0**2 / 4))**2 * (beta * sin(theta)**3)/(1+beta**2 - 2*beta*cos(theta))**2



def primakoff_scattering_xs_ntotal(ea, g, ma, z, r0):
    return quad(primakoff_scattering_diffxs, 0, pi, args=(ea,g,ma,z,r0))[0]



def primakoff_scattering_xs(ea, g, ma, z, r0):
    if ea < ma:
        return 0.0
    prefactor = (g * z)**2 / (2*137)
    eta2 = r0**2 * (ea**2 - ma**2)
    return prefactor * (((2*eta2 + 1)/(4*eta2))*log(1+4*eta2) - 1)



def axioelectric_xs(pe_xs, energy, z, a, ma, ge):
    pe = np.interp(energy, pe_xs[:,0], pe_xs[:,1])*1e-24 / (100*meter_by_mev)**2
    beta = sqrt(energy**2 - ma**2)
    return 137 * 3 * ge**2 * pe * energy**2 * (1 - np.power(beta, 2/3)/3) / (16*pi*me**2 * beta)



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
    b = 184*np.power(2.718, -1/2)*np.power(z, -1/3) / me
    return (z*t*b**2)**2 / (1 + t*b**2)**2



def _atomic_elastic_ff(t, m, z):
    # t: MeV
    # m: nucleus mass
    # z: atomic number
    b = 1194*np.power(2.718, -1/2)*np.power(z, -2/3) / me
    return (z*t*b**2)**2 / (1 + t*b**2)**2



def _screening(e, ma):
    if ma == 0:
        return 0
    r0 = 1/0.001973  # 0.001973 MeV A -> 1 A (Ge) = 1/0.001973
    x = (r0 * ma**2 / (4*e))**2
    numerator = 2*log(2*e/ma) - 1 - exp(-x) * (1 - exp(-x)/2) + (x + 0.5)*exp1(2*x) - (1+x)*exp1(x)
    denomenator = 2*log(2*e/ma) - 1
    return numerator / denomenator
    


