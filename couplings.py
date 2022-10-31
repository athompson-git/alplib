"""
Helper functions to track QCD axion parameter space and model-dependent things.
"""

from .constants import *
from .fmath import *


# DFSZ and KSVZ parameter relations from 2003.01100:
# "The landscape of QCD axion models", Di Luzio, Giannotti, Nardi, Visinelli




############  UNIVERSAL RELATIONS  ############

def f_a(ma):
    # ma in eV, returns f_a in GeV
    return 5.691e6 / ma



############  D F S Z    A X I O N S  ############

# Fermion couplings
def Cae(ma, tanbeta, dfsz_type):
    # ma in eV
    # electron coupling coefficient for DFSZ models
    fa = f_a(ma)
    if dfsz_type == "DFSZI":
        EbyN = 8/3
        return -(1/3)*sin(arctan(tanbeta))**2 + (3*ALPHA**2)/(4*pi**2) * (EbyN * log(fa/(M_E*1e-3)) - 1.92 * log(1/(M_E*1e-3)))
    if dfsz_type == "DFSZII":
        EbyN = 2/3
        return (1/3)*cos(arctan(tanbeta))**2 + (3*ALPHA**2)/(4*pi**2) * (EbyN * log(fa/(M_E*1e-3)) - 1.92 * log(1/(M_E*1e-3)))




def Camu(ma, tanbeta, dfsz_type):
    # ma in eV
    # muon coupling coefficient for DFSZ models
    fa = f_a(ma)
    if dfsz_type == "DFSZI":
        EbyN = 8/3
        return -(1/3)*sin(arctan(tanbeta))**2 + (3*ALPHA**2)/(4*pi**2) * (EbyN * log(fa/(M_MU*1e-3)) - 1.92 * log(1/(M_MU*1e-3)))
    if dfsz_type == "DFSZII":
        EbyN = 2/3
        return (1/3)*cos(arctan(tanbeta))**2 + (3*ALPHA**2)/(4*pi**2) * (EbyN * log(fa/(M_MU*1e-3)) - 1.92 * log(1/(M_MU*1e-3)))




def Catau(ma, tanbeta, dfsz_type):
    # ma in eV
    # tau coupling coefficient for DFSZ models
    fa = f_a(ma)
    if dfsz_type == "DFSZI":
        EbyN = 8/3
        return -(1/3)*sin(arctan(tanbeta))**2 + (3*ALPHA**2)/(4*pi**2) * (EbyN * log(fa/(M_TAU*1e-3)) - 1.92 * log(1/(M_TAU*1e-3)))
    if dfsz_type == "DFSZII":
        EbyN = 2/3
        return (1/3)*cos(arctan(tanbeta))**2 + (3*ALPHA**2)/(4*pi**2) * (EbyN * log(fa/(M_TAU*1e-3)) - 1.92 * log(1/(M_TAU*1e-3)))




def Can(tanbeta):
    # neutron coupling coefficient for DFSZ models
    cu0 = -cos(arctan(tanbeta))**2 / 3
    cd0 = -sin(arctan(tanbeta))**2 / 3
    csea = 0.038*cd0 + 0.012*cu0 + 0.009*cd0 + 0.0035*cu0
    return -0.023 + 0.88*cd0 - 0.39*cu0 - csea




def Cap(tanbeta):
    # proton coupling coefficient for DFSZ models
    cu0 = -cos(arctan(tanbeta))**2 / 3
    cd0 = -sin(arctan(tanbeta))**2 / 3
    csea = 0.038*cd0 + 0.012*cu0 + 0.009*cd0 + 0.0035*cu0
    return -0.47 + 0.88*cu0 - 0.39*cd0 - csea




def Capi0(tanbeta):
    # pion coupling for DFSZ models
    cu0 = -cos(arctan(tanbeta))**2 / 3
    cd0 = -sin(arctan(tanbeta))**2 / 3
    return 0.12 + (cd0 - cu0)/3




def gae_DFSZ(ma, tanbeta, dfsz_type):
    # ma in eV
    # return g_ae as a function of m_a, \tan\beta, and the DFSZ model (I or II)
    return abs((M_E*1e-3) * Cae(ma, tanbeta, dfsz_type) / f_a(ma))




def gamu_DFSZ(ma, tanbeta, dfsz_type):
    # ma in eV
    # return g_amu as a function of m_a, \tan\beta, and the DFSZ model (I or II)
    return abs((M_MU*1e-3) * Camu(ma, tanbeta, dfsz_type) / f_a(ma))




def gatau_DFSZ(ma, tanbeta, dfsz_type):
    # ma in eV
    # return g_atau as a function of m_a, \tan\beta, and the DFSZ model (I or II)
    return abs((M_TAU*1e-3) * Catau(ma, tanbeta, dfsz_type) / f_a(ma))




def gan1_DFSZ(ma, tanbeta):
    cu0 = -cos(arctan(tanbeta))**2 / 3
    cd0 = -sin(arctan(tanbeta))**2 / 3
    csea = 0.038*cd0 + 0.012*cu0 + 0.009*cd0 + 0.0035*cu0
    can = -0.023 + 0.88*cd0 - 0.39*cu0 - csea
    cap = -0.47 + 0.88*cu0 - 0.39*cd0 - csea
    caN = abs((cap - can)/2)
    return caN*(M_P*1e6)*ma / 5.691e15




def gan0_DFSZ(ma, tanbeta):
    cu0 = -cos(arctan(tanbeta))**2 / 3
    cd0 = -sin(arctan(tanbeta))**2 / 3
    csea = 0.038*cd0 + 0.012*cu0 + 0.009*cd0 + 0.0035*cu0
    can = -0.023 + 0.88*cd0 - 0.39*cu0 - csea
    cap = -0.47 + 0.88*cu0 - 0.39*cd0 - csea
    caN = abs((cap + can)/2)
    return caN*(M_P*1e6)*ma / 5.691e15




def gagamma_DFSZI(ma):
    # ma in eV
    return (0.203*8/3 - 0.39)*ma*1e-9




def gagamma_DFSZII(ma):
    # ma in eV
    return (0.203*2/3 - 0.39)*ma*1e-9




def gangae_DFSZ(ma, tanbeta, dfsz_type="DFSZI"):
    # Combined product of couplings gan * gae
    cu0 = -cos(arctan(tanbeta))**2 / 3
    cd0 = -sin(arctan(tanbeta))**2 / 3
    csea = 0.038*cd0 + 0.012*cu0 + 0.009*cd0 + 0.0035*cu0
    cae = Cae(ma, tanbeta, dfsz_type)
    can = -0.023 + 0.88*cd0 - 0.39*cu0 - csea
    cap = -0.47 + 0.88*cu0 - 0.39*cd0 - csea
    caN = (cap - can)/2
    return abs(caN*cae*(M_P*1e6)*(M_E*1e6)*ma**2 / power(5.691e15,2))




def gangagamma_DFSZ(ma, tanbeta, dfsz_type="DFSZI"):
    # Combined product of couplings gan * gagamma
    if dfsz_type == "DFSZII":
        EbyN = 2/3
    if dfsz_type == "DFSZI":
        EbyN = 8/3
    return gan1_DFSZ(ma, tanbeta) * gagamma_KSVZ(ma, EbyN)




############  K S V Z    A X I O N S  ############

def gagamma_KSVZ(ma, eByN):
    # return g_{a\gamma} as a function of m_a and E/N for the KSVZ models
    # ma in eV
    return abs(0.203*eByN - 0.39)*ma*1e-9




# Generic couplings at loop level


def gamma_loop(gf, mf, ma):
    tau = 4*power(mf/ma, 2)
    bf = 1 - tau*power(arcsin(1/sqrt(tau)), 2) if tau >= 1 \
        else 1 - tau*(pi/2 + 1j*log((1+sqrt(1-tau))/(1-sqrt(1-tau))))*(pi/2 - 1j*log((1+sqrt(1-tau))/(1-sqrt(1-tau))))
    return abs(ALPHA * (2*gf/mf) * bf / pi)