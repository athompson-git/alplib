# Production Cross Sections
# All cross sections in cm^2
# All energies in MeV

from .constants import *
from .fmath import *

#### Photon coupling ####

def free_primakoff_dsigma_dt(t, s, ma, M, g):
    num = ALPHA * g**2 * (t*(M**2 + s)*ma**2 - (M * ma**2)**2 - t*((s-M**2)**2 + s*t) - t*(t-ma**2)/2)
    denom = 4*t**2 * ((M + ma)**2 - s)*((M - ma)**2 - s)
    return heaviside(num/denom, 0.0) * (num / denom)




def primakoff_dsigma_dtheta(theta, energy, z, ma, g=1):
    # Primakoff scattering production diffxs by theta (γ + A -> a + A)
    if energy < ma:
        return 0
    pa = sqrt(energy**2 - ma**2)
    t = 2*energy*(pa*cos(theta) - energy) + ma**2
    ff = 1 #_nuclear_ff(t, ma, z, 2*z)
    return ALPHA * (g * z * ff * pa**2 / t)**2 * sin(theta)**3 / 4




def primakoff_nsigma(energy, z, ma, g=1):
    # Primakoff production total xs, numerical eval. (γ + A -> a + A)
    return quad(primakoff_dsigma_dtheta, 0, pi, args=(energy,z,ma,g), limit=3)[0]




def primakoff_sigma(energy, z, a, ma, g):
    # Primakoff production total xs (γ + A -> a + A)
    # Tsai, '86 (ma << E)
    if energy < ma:
        return 0
    M_E = 0.511
    prefactor = (1 / 137 / 4) * (g ** 2)
    return prefactor * ((z ** 2) * (log(184 * power(z, -1 / 3)) \
        + log(403 * power(a, -1 / 3) / M_E)) \
        + z * log(1194 * power(z, -2 / 3)))




#### Electron coupling ####

def compton_dsigma_dea(ea, eg, g, ma):
    # Differential cross-section dS/dE_a. (γ + e- > a + e-)
    a = 1 / 137
    aa = g ** 2 / 4 / pi
    s = 2 * M_E * eg + M_E ** 2
    x = ((ma**2 / (2*eg*M_E)) - ea / eg + 1)

    xmin = ((s - M_E**2)*(s - M_E**2 + ma**2) 
            - (s - M_E**2)*sqrt((s - M_E**2 + ma**2)**2 - 4*s*ma**2))/(2*s*(s-M_E**2))
    xmax = ((s - M_E**2)*(s - M_E**2 + ma**2) 
            + (s - M_E**2)*sqrt((s - M_E**2 + ma**2)**2 - 4*s*ma**2))/(2*s*(s-M_E**2))

    thresh = heaviside(s > (M_E + ma)**2, 0.0)*heaviside(x-xmin,0.0)*heaviside(xmax-x,0.0)
    return thresh * (1 / eg) * pi * a * aa / (s - M_E ** 2) * (x / (1 - x) * (-2 * ma ** 2 / (s - M_E ** 2) ** 2
                                                                * (s - M_E ** 2 / (1 - x) - ma ** 2 / x) + x))




def brem_dsigma_dea_domega(Ea, thetaa, Ee, g, ma, z):
    # Differential cross section dSigma/dE_a for ALP bremsstrahlung (e- Z -> e- Z a)
    theta_max = max(sqrt(ma*M_E)/Ee, power(ma/Ee, 3/2))
    x = Ea / Ee
    l = (Ee * thetaa / M_E)**2
    U = l*x*M_E**2 + x*M_E**2 + ((1-x)*M_E**2) / x
    tmin = (U / (2*Ee*(1-x)))**2
    a = 111*power(z, -1/3)/M_E
    aPrime = 773*power(z, -2/3)/M_E
    chi = z**2 * (log((a*M_E*(1+l))**2 / (a**2 * tmin + 1)) - 1)

    prefactor = heaviside(theta_max - thetaa, 0.0) * ((ALPHA * g)**2 / (4*pi**2)) * Ee / U**2
    return chi * prefactor * (x**3 - 2*(ma*x)**2 * (1-x)/U  \
                                + 2*(ma/U)**2 * (x*(ma*(1-x))**2 + M_E**2 * x**3 * (1-x)))




def resonance_sigma(ea, ee, g, ma):
    # Resonant production cross section (e- e+ -> a)
    pass
