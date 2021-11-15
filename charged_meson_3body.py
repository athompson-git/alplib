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
from .matrix_element import M2DarkPrimakoff

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
    probability_decay = p_decay(p_list, meson_mass, meson_lifetime, 50)
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



# Compton cross section dSigmadOmega
def compton_dsigma_domega(theta, Ea, ma, ge):
    y = 2*M_E*Ea + ma**2
    pa = sqrt(Ea**2 - ma**2)
    e_gamma = 0.5*y/(M_E + Ea - pa*cos(theta))

    prefactor = ge**2 * ALPHA * e_gamma / (4*pi*2*pa*M_E**2)
    return prefactor * (1 + 4*(M_E*e_gamma/y)**2 - 4*M_E*e_gamma/y - 4*M_E*e_gamma*(ma*pa*sin(theta))**2 / y**3)




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




# Convolve flux with axion branching ratio and generate ALP flux
class ChargedMeson3BodyDecay:
    def __init__(self, meson_flux, axion_mass=0.1, coupling=1.0, n_samples=50,
                 meson_mass=M_PI, ckm=V_UD, fM=F_PI, m_lepton=M_MU, boson_type="P", energy_cut=140.0):
        self.meson_flux = meson_flux
        self.mm = meson_mass
        self.ckm = ckm
        self.fM = fM
        self.m_lepton = m_lepton
        self.rep = boson_type
        self.ma = axion_mass
        self.gmu = coupling
        self.det_dist = 541
        self.dump_dist = 50
        self.det_length = 12
        self.det_sa = cos(arctan(self.det_length/(self.det_dist-self.dump_dist)/2))
        self.solid_angles = []
        self.energy_cut = energy_cut
        self.nsamples = n_samples
        self.energies = []
        self.cosines = []
        self.decay_pos = []
        self.weights = []
        self.decay_weight = []
        self.scatter_weight = []
    
    def lifetime(self, gagamma):
        return 1/W_gg(gagamma, self.ma)

    def dGammadEa(self, Ea):
        m212 = self.mm**2 + self.ma**2 - 2*self.mm*Ea
        e2star = (m212 - self.m_lepton**2)/(2*sqrt(m212))
        e3star = (self.mm**2 - m212 - self.ma**2)/(2*sqrt(m212))

        if self.ma > e3star:
            return 0.0

        m223Max = (e2star + e3star)**2 - (sqrt(e2star**2) - sqrt(e3star**2 - self.ma**2))**2
        m223Min = (e2star + e3star)**2 - (sqrt(e2star**2) + sqrt(e3star**2 - self.ma**2))**2
    
        def MatrixElement2P(m223):
            ev = (m212 + m223 - self.m_lepton**2 - self.ma**2)/(2*self.mm)
            emu = (self.mm**2 - m223 + self.m_lepton**2)/(2*self.mm)
            q2 = self.mm**2 - 2*self.mm*ev

            prefactor = heaviside(e3star-self.ma,0.0)*(self.gmu*G_F*self.fM*self.ckm/(q2 - self.m_lepton**2))**2
            return prefactor*((2*self.mm*emu*q2 * (q2 - self.m_lepton**2) - (q2**2 - (self.m_lepton*self.mm)**2)*(q2 + self.m_lepton**2 - self.ma**2)) - (2*q2*self.m_lepton**2 * (self.mm**2 - q2)))
        
        def MatrixElement2S(m223):
            ev = (m212 + m223 - self.m_lepton**2 - self.ma**2)/(2*self.mm)
            emu = (self.mm**2 - m223 + self.m_lepton**2)/(2*self.mm)
            q2 = self.mm**2 - 2*self.mm*ev

            prefactor = heaviside(e3star-self.ma,0.0)*(self.gmu*G_F*self.fM*self.ckm/(q2 - self.m_lepton**2))**2
            return prefactor*((2*self.mm*emu*q2 * (q2 - self.m_lepton**2) - (q2**2 - (self.m_lepton*self.mm)**2)*(q2 + self.m_lepton**2 - self.ma**2)) + (2*q2*self.m_lepton**2 * (self.mm**2 - q2)))

        def MatrixElement2V(m223):
            q2 = self.mm**2 - 2*self.mm*(m212 + m223 - self.m_lepton**2 - self.ma**2)/(2*self.mm)

            prefactor = heaviside(e3star-self.ma,0.0)*8*power(G_F*self.fM*self.ckm/(q2 - self.m_lepton**2)/self.ma, 2)

            lq = (m212 - self.m_lepton**2)/2
            lp = (self.mm**2 - m212 - m223)/2
            kq = (m212 + m223 - self.m_lepton**2 - self.ma**2)/2
            pq = (m223 - self.ma**2)/2
            kl = (self.mm**2 + self.m_lepton**2 - m223)/2
            kp = (self.mm**2 + self.ma**2 - m212)/2

            cr = self.gmu
            cl = self.gmu

            # Dmu(self.mm/kl)*
            return -prefactor * ((power(cr*self.mm*self.m_lepton,2) - power(cl*q2,2)) * (lq*self.ma**2 + 2*lp*pq) \
                - 2*cr*self.m_lepton**2 * kq * (cr*self.ma**2 * kl + 2*cr*kp*lp - 3*cl*q2*self.ma**2))
        
        if self.rep == "P":
            return (2*self.mm)/(32*power(2*pi*self.mm, 3))*quad(MatrixElement2P, m223Min, m223Max)[0]
        
        if self.rep == "S":
            return (2*self.mm)/(32*power(2*pi*self.mm, 3))*quad(MatrixElement2S, m223Min, m223Max)[0]

        if self.rep == "V":
            return (2*self.mm)/(32*power(2*pi*self.mm, 3))*quad(MatrixElement2V, m223Min, m223Max)[0]

    def gamma_sm(self):
        return (G_F*self.fM*self.m_lepton*self.ckm)**2 * self.mm * (1-(self.m_lepton/self.mm)**2)**2 / (4*pi)

    def total_br(self):
        EaMax = (self.mm**2 + self.ma**2 - self.m_lepton**2)/(2*self.mm)
        EaMin = self.ma
        return quad(self.dGammadEa, EaMin, EaMax)[0] / self.gamma_sm()
    
    def simulate_single(self, meson_p, pion_wgt, cut_on_solid_angle=True, solid_angle_cosine=0.0):
        ea_min = self.ma
        ea_max = (self.mm**2 + self.ma**2 - self.m_lepton**2)/(2*self.mm)

        # Draw random variate energies and angles in the pion rest frame
        energies = np.random.uniform(ea_min, ea_max, self.nsamples)
        momenta = sqrt(energies**2 - self.ma**2)
        cosines = np.random.uniform(-1, 1, self.nsamples)
        pz = momenta*cosines

        # Boost to lab frame
        beta = meson_p / sqrt(meson_p**2 + self.mm**2)
        boost = power(1-beta**2, -0.5)
        e_lab = boost*(energies + beta*pz)
        pz_lab = boost*(pz + beta*energies)
        cos_theta_lab = pz_lab / sqrt(e_lab**2 - self.ma**2)

        # Jacobian for transforming d2Gamma/(dEa * dOmega) to lab frame:
        jacobian = sqrt(e_lab**2 - self.ma**2) / momenta
        # Monte Carlo volume, making sure to use the lab frame energy range
        mc_vol = (max(e_lab) - min(e_lab))

        # Draw weights from the PDF
        # isotropic in rest frame, angular MC volume factors cancel
        weights = np.array([pion_wgt*mc_vol*self.dGammadEa(ea)/self.gamma_sm()/self.nsamples \
            for ea in energies])

        for i in range(self.nsamples):
            solid_angle_acceptance = heaviside(cos_theta_lab[i] - solid_angle_cosine, 0.0)
            if solid_angle_acceptance == 0.0 and cut_on_solid_angle:
                continue
            self.energies.append(e_lab[i])
            self.cosines.append(cos_theta_lab[i])
            self.weights.append(weights[i]*jacobian[i]*heaviside(e_lab[i]-self.energy_cut,1.0))
            self.solid_angles.append(solid_angle_cosine)
    
    def simulate(self, cut_on_solid_angle=True):
        self.energies = []
        self.cosines = []
        self.weights = []
        self.scatter_weight = []
        self.decay_weight = []
        self.decay_pos = []
        self.solid_angles = []

        if self.ma > self.mm - self.m_lepton:
            # Kinematically forbidden beyond Meson mass - muon mass difference
            return

        for i, p in enumerate(self.meson_flux):
            # Simulate decay positions between target and dump
            # The quantile is truncated at the dump position via umax
            decay_l = METER_BY_MEV * p[0] / self.gamma_sm() / self.mm
            umax = exp(-2*self.dump_dist/decay_l) * power(exp(self.dump_dist/decay_l) - 1, 2) \
                if decay_l > 1.0 else 1.0
            try:
                u = np.random.uniform(0.0, min(umax, 1.0))
            except:
                print("umax = ", umax, " decay l = ", decay_l, p[0])
            x = decay_quantile(u, p[0], self.mm, self.gamma_sm())
            
            # Append decay positions and solid angle cosines for the geometric acceptance of each meson decay
            self.decay_pos.append(x)
            solid_angle_cosine = cos(arctan(self.det_length/(self.det_dist-x)/2))

            # Simulate decays for each charged meson
            self.simulate_single(p[0], p[2], cut_on_solid_angle, solid_angle_cosine)
        

    def propagate(self, gagamma=None):  # propagate to detector
        e_a = np.array(self.energies)
        wgt = np.array(self.weights)
        if gagamma is not None:
            # Decay via loop-induced gamma coupling
            # Get axion Lorentz transformations and kinematics
            p_a = sqrt(e_a**2 - self.ma**2)
            v_a = p_a / e_a
            axion_boost = e_a / self.ma

            surv_prob = exp(-self.det_dist / METER_BY_MEV / v_a / (axion_boost * self.lifetime(gagamma)))
            decay_prob = 1.0 - exp(-self.det_length / METER_BY_MEV / v_a / (axion_boost * self.lifetime(gagamma)))
            
            self.decay_weight = np.asarray(wgt * surv_prob * decay_prob, dtype=np.float64)
            self.scatter_weight = np.asarray(wgt * surv_prob, dtype=np.float64)
        else:
            # Do not decay
            self.decay_weight = np.asarray(wgt*0.0, dtype=np.float64)
            self.scatter_weight = np.asarray(wgt, dtype=np.float64)
    
    def scatter_dark_primakoff(self, gZN, gaGZ, mZp, n_e, cosine_bins, evis_bins=None, eff=Efficiency()):
        binned_events = np.zeros(cosine_bins.shape[0]-1)
        centers = (cosine_bins[1:] + cosine_bins[:-1])/2
        for i in range(self.scatter_weight.shape[0]):
            rcos = np.random.uniform(-1, 1, self.nsamples)
            xs = 4*pi*dark_iprim_dsigma_dcostheta(rcos, self.energies[i], gZN, gaGZ, self.ma, mZp)/self.nsamples
            wgts = eff(self.energies[i]) * 4.75 * self.scatter_weight[i]*n_e*power(METER_BY_MEV*100, 2)*xs  # ad hoc coherency factor 4.75
            h, hbins = np.histogram(rcos, weights=wgts, bins=cosine_bins)
            binned_events += h
        return binned_events, centers
    
    def evis_dark_primakoff(self, gZN, gaGZ, mZp, n_e, evis_bins, eff=Efficiency()):
        binned_events = np.zeros(evis_bins.shape[0]-1)
        centers = (evis_bins[1:] + evis_bins[:-1])/2
        # Simulate using the MatrixElement method
        m2_dp = M2DarkPrimakoff(self.ma, 37e3, mZp)

        # Declare initial vectors
        pa_mu = LorentzVector(0.0, 0.0, 0.0, 0.0)
        PM_mu = LorentzVector(0.0, 0.0, 0.0, 0.0)
        mc = Scatter2to2MC(m2_dp, pa_mu, PM_mu, n_samples=1)
        
        for i in range(self.nsamples):
            Ea0 = self.energies[i]
            mc.lv_p1 = LorentzVector(Ea0, 0.0, 0.0, np.sqrt(Ea0**2 - self.ma**2))
            mc.lv_p2 = LorentzVector(37e3, 0.0, 0.0, 0.0)
            mc.scatter_sim()
            wgts = power(gZN * gaGZ, 2)*eff(self.energies[i]) * 4.75 * self.scatter_weight[i]*n_e*power(METER_BY_MEV*100, 2)*mc.get_e3_lab_weights()
            h, hbins = np.histogram([Ea0], weights=wgts, bins=evis_bins)
            binned_events += h
        return binned_events, centers

    def decay_gamma_cosines(self, cosine_bins):
        centers = (cosine_bins[1:] + cosine_bins[:-1])/2
        h, hbins = np.histogram(self.cosines, weights=self.decay_weight, bins=cosine_bins)
        return h, centers


