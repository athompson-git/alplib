"""
Helper functions to track QCD axion parameter space and model-dependent things.
"""

from .constants import *
from .fmath import *


# DFSZ and KSVZ parameter relations from 2003.01600



def Cae(ma, tanbeta, dfsz_type):  # ma in eV
    alpha = 1/137
    fa = 5.7e6 / ma
    if dfsz_type == "DFSZI":
        EbyN = 8/3
        return (1/3)*sin(arctan(tanbeta))**2 + (3*alpha**2)/(4*pi**2) * (EbyN * log(fa/(0.511e-3)) - 1.92 * log(1/(0.511e-3)))
    if dfsz_type == "DFSZII":
        EbyN = 2/3
        return -(1/3)*cos(arctan(tanbeta))**2 + (3*alpha**2)/(4*pi**2) * (EbyN * log(fa/(0.511e-3)) - 1.92 * log(1/(0.511e-3)))




def f_a(ma):
    # ma in eV, returns f_a in GeV
    return 5.691e6 / ma




def gae_DFSZ(ma, tanbeta, dfsz_type):
    # ma in eV
    # return g_ae as a function of m_a, \tan\beta, and the DFSZ model (I or II)
    return abs((M_E*1e-3) * Cae(ma, tanbeta, dfsz_type) / f_a(ma))




def gagamma_KSVZ(ma, eByN):
    # return g_{a\gamma} as a function of m_a and E/N for the KSVZ models
    # ma in eV
    return (0.203*eByN - 0.39)*ma*1e-9




def gagamma_DFSZI(ma):
    # ma in eV
    return (0.203*8/3 - 0.39)*ma*1e-9




def gagamma_DFSZII(ma):
    # ma in eV
    return (0.203*2/3 - 0.39)*ma*1e-9




def gamma_loop(gf, mf, ma):
    tau = 4*power(mf/ma, 2)
    bf = 1 - tau*power(arcsin(1/sqrt(tau)), 2) if tau >= 1 \
        else 1 - tau*(pi/2 + 1j*log((1+sqrt(1-tau))/(1-sqrt(1-tau))))*(pi/2 - 1j*log((1+sqrt(1-tau))/(1-sqrt(1-tau))))
    return ALPHA * gf * bf / pi
