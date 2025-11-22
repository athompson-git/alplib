# Decay widths and lifetimes
# All lifetimes in s
# All decay widths in MeV

from .constants import *
from .fmath import *

import pkg_resources

def W_gg(g_agamma, ma):
    # a -> gamma gamma
    # g_agamma in MeV^-1
    return g_agamma**2 * ma**3 / (64*pi)




def W_ff(g, mf, ma):
    # a -> f f   generic decay
    return g**2 * ma * sqrt(1 - (2 * mf / ma)**2) / (8 * pi) \
        if 1 - 4 * (mf / ma) ** 2 > 0 else 0.0




def W_ee(g_ae, ma):
    # a -> e+ e-
    return g_ae**2 * ma * sqrt(1 - (2 * M_E / ma)**2) / (8 * pi) \
        if 1 - 4 * (M_E / ma) ** 2 > 0 else 0.0




# Loop functions
def fp2(tau):
    return arcsin(1/sqrt(tau))**2 if tau >= 1 \
        else pi**2 / 4  + log((1+sqrt(1-tau))/(1-sqrt(1-tau)))**2 / 4




def b1(tau):
    return 1 - tau*fp2(tau)




def W_gg_loop(g_af, ma, mf):
    # a -> gamma gamma via f loop
    #g_agamma = (g_af * ALPHA / (pi * mf)) * (1 - power(2*mf*arcsin(ma/(2*mf))/ma,2))
    g_agamma_eff = g_af * ALPHA * b1(power(2*mf/ma, 2)) / (4*pi)
    return W_gg(g_agamma_eff, ma)




def W_aprime_gamma_phi(g_gauge, m_aprime, m_phi):
    # Aprime -> gamma + phi (scalar)
    return power(g_gauge, 2) * power((m_aprime**2 - m_phi**2)/m_aprime, 3) / (128*pi)




# Gluon dominance data for decay width [m_a (GeV), Gamma (eV)] for f_a = 1e6 GeV
fpath_gludom_data = pkg_resources.resource_filename(__name__, 'data/gluon_dominance_widths_nogamma.txt')
gludom_data = np.genfromtxt(fpath_gludom_data)
def W_agg_hadronic(f_a, ma):
    # a -> hadrons total decay width
    eV_fact = 1e-9
    Lambda_scale = (32 * pi**2 * f_a / 1e6)**2  # data is normalized to EFT scale of 1 TeV

    return np.interp(ma*1e-3, gludom_data[:,0], gludom_data[:,1]) * eV_fact / Lambda_scale 




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