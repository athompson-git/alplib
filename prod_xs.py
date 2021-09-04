# Production Cross Sections
# All cross sections in cm^2
# All energies in MeV

from .constants import *
from .fmath import *
from .decay import *

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

def compton_sigma_v2(eg, g, ma, z=1):
    # Compton scattering total cross section (γ + e- > a + e-)
    # Taken from 0807.2926. Validated.
    s = 2*eg*M_E + M_E**2
    p0 = 0.5*(2*eg*M_E + ma**2)/sqrt(s)
    k0 = (eg*M_E + M_E**2)/sqrt(s)
    p = sqrt(p0**2 - ma**2)
    k = sqrt(s) - k0
    
    prefactor = (z*ALPHA*g**2 / (8*s)) * (p/k)
    return prefactor * (-3 + (M_E**2 - ma**2)/s + s*power(ma / (2*eg*M_E),2) \
                        + (1 - (ma**2 / (eg*M_E)) + (ma**2 * (ma**2 - 2*M_E**2)/(2*power(eg*M_E,2)))) \
                            * (sqrt(s)/p)*log((2*p0*k0 + 2*p*k - ma**2)/(2*p0*k0 - 2*p*k - ma**2)))




def compton_dsigma_dea(ea, eg, g, ma, z=1):
    # Differential cross-section dS/dE_a. (γ + e- > a + e-)
    a = 1 / 137
    aa = g ** 2 / 4 / pi
    s = 2 * M_E * eg + M_E ** 2
    x = ((ma**2 / (2*eg*M_E)) - ea / eg + 1)

    xmin = ((s - M_E**2)*(s - M_E**2 + ma**2) 
            - (s - M_E**2)*sqrt((s - M_E**2 + ma**2)**2 - 4*s*ma**2))/(2*s*(s-M_E**2))
    xmax = ((s - M_E**2)*(s - M_E**2 + ma**2) 
            + (s - M_E**2)*sqrt((s - M_E**2 + ma**2)**2 - 4*s*ma**2))/(2*s*(s-M_E**2))

    thresh = heaviside(s - (M_E + ma)**2, 0.0)*heaviside(x-xmin,0.0)*heaviside(xmax-x,0.0)
    return z * thresh * (1 / eg) * pi * a * aa / (s - M_E ** 2) * (x / (1 - x) * (-2 * ma ** 2 / (s - M_E ** 2) ** 2
                                                                * (s - M_E ** 2 / (1 - x) - ma ** 2 / x) + x))




def brem_dsigma_dea_domega(Ea, thetaa, Ee, g, ma, z):
    # Differential cross section d^2 Sigma/(dE_a dOmega) for ALP bremsstrahlung (e- Z -> e- Z a)
    # Tsai, 1986
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




def brem_dsigma_dea(Ea, Ee, g, ma, z):
    # Differential cross section dSigma/dE_a for ALP bremsstrahlung (e- Z -> e- Z a)
    # Tsai, 1986
    r0 = ALPHA / M_E
    x = Ea / Ee
    f = power(ma / (x * M_E), 2) * (1 - x)
    ln_el = log(184*power(z, -1/3))
    ln_inel = log(1194*power(z, -2/3))

    prefactor = 2 * r0**2 * g**2 / 4 / pi / Ee  # divide by Ee to change dsigma/dx into dsigma/dEa
    phase_space = ((x * (1 + f/1.5)/power(1+f, 2)) * (z**2 * ln_el + z * ln_inel) \
                        + x * (z**2 + z) * ((1+f)*log(1+f)/(3*f**2) - (1 + 4*f + 2*f**2)/(3 * f * power(1+f, 2))))
    return prefactor * phase_space * heaviside(phase_space, 0.0)




def brem_sigma(Ee, g, ma, z=1):
    # Total axion bremsstrahlung production cross section (e- Z -> e- Z a)
    # Tsai 1986
    #ea_max = Ee * (1 - max(power(M_E/ma, 2), power(ma/Ee, 2)))
    ea_max = Ee * (1 - power(ma/Ee, 2))
    return heaviside(Ee-ma,0.0)*quad(brem_dsigma_dea, ma, ea_max, args=(Ee, g, ma, z,))[0]


def brem_sigma_v2(Ee, g, ma, z=1):
    # Total axion bremsstrahlung production cross section (e- Z -> e- Z a)
    # Tsai 1986
    return heaviside(Ee-ma,0.0)*quad(brem_dsigma_dea, ma, Ee*0.9999, args=(Ee, g, ma, z,))[0]


def brem_sigma_mc(Ee, g, ma, z=1, nsamples=100):
    ea_max = Ee * (1 - power(ma/Ee, 2))
    ea_rnd = np.random.uniform(ma, ea_max, nsamples)
    mc_vol = (Ee - ma)/nsamples
    return mc_vol * np.sum(brem_dsigma_dea(ea_rnd, Ee, g, ma, z))



def resonance_sigma(ee, ma, g):
    # Resonant production cross section (e- e+ -> a)
    s = 2*M_E*ee
    return (12 * pi / ma**2) * (power(W_ee(g, ma)/2, 2)/((sqrt(s) - ma)**2 + power(W_ee(g, ma)/2, 2)))




def resonance_peak(g):
    # Returns the peak value of the resonance production cross section (e- e+ -> a)
    return pi * g**2 / (2 * M_E)