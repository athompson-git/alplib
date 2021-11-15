# Decay widths and lifetimes
# All lifetimes in s
# All decay widths in MeV

from .constants import *
from .fmath import *


def W_gg(g_agamma, ma):
    # a -> gamma gamma
    # g_agamma in MeV^-1
    return g_agamma**2 * ma**3 / (64*pi)




def W_ee(g_ae, ma):
    # a -> e+ e-
    return g_ae**2 * ma * sqrt(1 - (2 * M_E / ma)**2) / (8 * pi) \
        if 1 - 4 * (M_E / ma) ** 2 > 0 else 0.0




def W_aprime_gamma_phi(g_gauge, m_aprime, m_phi):
    # Aprime -> gamma + phi (scalar)
    return power(g_gauge, 2) * power((m_aprime**2 - m_phi**2)/m_aprime, 3) / (128*pi)




def W_aprime_gamma_a(g_gauge, m_aprime, m_phi):
    # Aprime -> gamma + a (pseudoscalar)
    pass




def Tau(width):
    # Get the lifetime in the rest frame in s
    pass




def Tau_lab(width, va):
    # Get the lifetime in the lab frame in s
    pass




def p_survive(p, m, tau, l):
    # Probability that a particle will survive a distance l from production site
    # momentum in lab frame p
    # lifetime tau in seconds
    # mass of decaying particle m
    # distance from source l in meters
    energy = sqrt(p**2 + m**2)
    boost = energy / m
    v = p / energy
    prob = exp(-l/(METER_BY_MEV*v*boost*tau/HBAR))
    return prob




def p_decay(p, m, tau, l):
    # Probability that a particle will decay before reaching a distance l
    # momentum in lab frame p
    # lifetime tau in seconds
    # mass of decaying particle m
    # distance from source l in meters
    energy = sqrt(p**2 + m**2)
    boost = energy / m
    v = p / energy
    prob = exp(-l/(METER_BY_MEV*v*boost*tau/HBAR))
    return (1 - prob)




def p_decay_in_region(p, m, tau, l, dl):
    # Probability that the particle will decay within a region (l, l + dl)
    # momentum in lab frame p
    # lifetime tau in seconds
    # mass of decaying particle m
    # l and dl in meters
    energy = sqrt(p**2 + m**2)
    boost = energy / m
    v = p / energy
    prob = exp(-l/(METER_BY_MEV*v*boost*tau/HBAR)) * (1 - exp(-dl/(METER_BY_MEV*v*boost*tau/HBAR)))
    return prob




def decay_quantile(u, p, m, width_gamma):
    # Quantile/PPF function to generate decay positions for a given lifetime and momentum.
    # momentum in lab frame p
    # decay width width_gamma in MeV
    # mass of decaying particle m
    # l and dl in meters
    return (METER_BY_MEV * p / width_gamma / m) * log((sqrt(u) + 1)/(1 - u))