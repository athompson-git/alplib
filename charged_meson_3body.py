# Classes and constants for axion production and detection from 3-body decay of charged mesons

from alplib.materials import Efficiency
from .constants import *
from .fmath import *
from .prod_xs import *
from .det_xs import *
from .decay import *
from .couplings import *
from .efficiency import Efficiency
from .cross_section_mc import *
from .matrix_element import *

# Proton total cross section
def sigmap(p):
    A = 307.8
    B = 0.897
    C = -2.598
    D = -4.973
    n = 0.003
    return A + B*power(p,n) + C*log(p)*log(p) + D*log(p)




# pi+ decay probability
def pi_decay(p_pi):
    e_pi = sqrt(p_pi**2 + M_PI**2)
    boost = e_pi / M_PI
    v_pi = p_pi / e_pi
    prob = exp(-50 / (METER_BY_MEV*v_pi*boost*2.6e-8 / HBAR))
    return (1 - prob)




def kaon_decay(p):
    energy = sqrt(p**2 + M_K**2)
    boost = energy / M_K
    v = p / energy
    prob = exp(-50 / (METER_BY_MEV*v*boost*2.6e-8 / HBAR))
    return (1 - prob)




def charged_meson_flux_mc(meson_type, p_min, p_max, theta_min, theta_max,
                            n_samples=1000, p_proton=8.89, n_pot=18.75e20):
    # Charged meson monte carlo flux simulation
    # Based on the Sanford-Wang and Feynman scaling parameterized proton prodution cross sections
    # momentum from [p_min, p_max] in GeV

    if meson_type not in ["pi_plus", "pi_minus", "k_plus", "K0S"]:
        raise Exception("meson_type not in list of available fluxes")
    
    meson_mass=M_PI
    meson_lifetime = PION_LIFETIME
    if meson_type == "k_plus" or meson_type == "K0S":
        meson_mass = M_K
        meson_lifetime = KAON_LIFETIME
    
    p_list = np.random.uniform(p_min, p_max, n_samples)
    theta_list = np.random.uniform(theta_min, theta_max, n_samples)

    xs_wgt = meson_production_d2SdpdOmega(p_list, theta_list, p_proton, meson_type=meson_type) * sin(theta_list)
    probability_decay = p_decay(p_list*1e3, meson_mass, meson_lifetime, 50)
    pi_plus_wgts = probability_decay * (2*pi*(theta_max-theta_min) * (p_max-p_min)) * n_pot * xs_wgt / n_samples / sigmap(p_proton)
    return np.array([p_list*1000.0, theta_list, pi_plus_wgts]).transpose()




# Charged pion production double-differential cross section on Be target
def meson_production_d2SdpdOmega(p, theta, p_proton, meson_type="pi_plus"):
    pB = p_proton
    mt = M_P
    # Sanford-Wang Parameterization
    if meson_type == "pi_plus":
        c1 = 220.7
        c2 = 1.080
        c3 = 1.0
        c4 = 1.978
        c5 = 1.32
        c6 = 5.572
        c7 = 0.0868
        c8 = 9.686
        c9 = 1.0
        #c1, c2, c3, c4, c5, c6, c7, c8, c9 = 1.20245,1.08, 2.15, 2.31,1.98,5.73,0.137,24.1, 1.0
        prefactor = c1 * power(p, c2) * (1 - p/(pB - c9))
        exponential = exp(-c3*power(p,c4)/power(pB,c5) - c6*theta*(p-c7*pB*power(cos(theta),c8)))
        return prefactor * exponential
    elif meson_type == "pi_minus":
        c1 = 213.7
        c2 = 0.9379
        c3 = 5.454
        c4 = 1.210
        c5 = 1.284
        c6 = 4.781
        c7 = 0.07338
        c8 = 8.329
        c9 = 1.0
        prefactor = c1 * power(p, c2) * (1 - p/(pB - c9))
        exponential = exp(-c3*power(p,c4)/power(pB,c5) - c6*theta*(p-c7*pB*power(cos(theta),c8)))
        return prefactor * exponential
    elif meson_type == "k_plus":
        pT = p*sin(theta)
        pL = p*cos(theta)
        beta = pB / (mt*1e-3 + sqrt(pB**2 + (M_P*1e-3)**2))
        gamma = power(1-beta**2, -0.5)
        pLstar = gamma*(pL - sqrt(pL**2 + pT**2 + (M_K*1e-3)**2)*beta)
        s = (M_P*1e-3)**2 + (mt*1e-3)**2 + 2*sqrt(pB**2 + (M_P*1e-3)**2)*mt*1e-3
        xF = abs(2*pLstar/sqrt(s))

        c1 = 11.70
        c2 = 0.88
        c3 = 4.77
        c4 = 1.51
        c5 = 2.21
        c6 = 2.17
        c7 = 1.51
        prefactor = c1 * p**2 / sqrt(p**2 + (M_K*1e-3)**2)
        return prefactor * (1 - xF) * exp(-c2*pT - c3*power(xF, c4) - c5*pT**2 - c7*power(pT*xF, c6))
    elif meson_type == "k0S":
        c1 = 15.130
        c2 = 1.975
        c3 = 4.084
        c4 = 0.928
        c5 = 0.731
        c6 = 4.362
        c7 = 0.048
        c8 = 13.300
        c9 = 1.278
        prefactor = c1 * power(p, c2) * (1 - p/(pB - c9))
        exponential = exp(-c3*power(p,c4)/power(pB,c5) - c6*theta*(p-c7*pB*power(cos(theta),c8)))
        return prefactor * exponential




class ChargedPionFluxMiniBooNE:
    def __init__(self, proton_energy=8000.0):
        self.n_samples = 10000
        self.ep = proton_energy
        self.x0 = np.array([])
        self.y0 = np.array([])
        self.z0 = np.array([])
        self.px0 = np.array([])
        self.py0 = np.array([])
        self.pz0 = np.array([])

    def sigmap(self, p):
        A = 307.8
        B = 0.897
        C = -2.598
        D = -4.973
        n = 0.003
        return A + B*power(p,n) + C*log(p)*log(p) + D*log(p)

    def d2SdpdOmega_SW(self):
        pass

    def simulate_beam_spot(self):
        r1 = norm.rvs(size=self.n_samples)
        r2 = norm.rvs(size=self.n_samples)
        r3 = norm.rvs(size=self.n_samples)
        r4 = norm.rvs(size=self.n_samples)

        sigma_x = 1.51e-1  # cm
        sigma_y = 0.75e-1  # cm
        sigma_theta_x = 0.66e-3  # mrad
        sigma_theta_y = 0.40e-3  # mrad

        self.x0 = r1*sigma_x
        self.y0 = r2*sigma_y
        self.z0 = -10.0

        self.px0 = sqrt(self.ep**2 - M_P**2)*r3*sigma_theta_x
        self.py0 = sqrt(self.ep**2 - M_P**2)*r4*sigma_theta_y
        self.pz0 = sqrt(self.ep**2 - M_P**2 - self.px0**2 - self.py0**2)

    def B(self, r):
        # B field in T for r in cm
        return heaviside(r - 2.2, 0.0) * (4*pi*1e-2) * 170 / (2*pi*r)

    def focus_pions(self):
        pass
