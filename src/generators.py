from .constants import *
from .fmath import *
from .det_xs import *
from .decay import *
from .prod_xs import *
from .fluxes import *

import multiprocessing as multi

from numpy import log, pi, exp, sin, cos, sin, sqrt, \
                    arccos, arcsin, heaviside, power
from scipy.special import exp1
from scipy.integrate import quad

import mpmath as mp
from mpmath import mpmathify, fsub
mp.dps = 15



# Define ALP PRODUCTION cross-sections

#### Photon coupling ####


def primakoff_production_diffxs(theta, energy, z, ma, g=1):
    # Primakoff scattering production diffxs by theta (γ + A -> a + A)
    if energy < ma:
        return 0
    pa = sqrt(energy**2 - ma**2)
    t = 2*energy*(pa*cos(theta) - energy) + ma**2
    ff = 1 #_nuclear_ff(t, ma, z, 2*z)
    return ALPHA * (g * z * ff * pa**2 / t)**2 * sin(theta)**3 / 4




def primakoff_production_xs_ntotal(energy, z, ma, g=1):
    # Primakoff production total xs, numerical eval. (γ + A -> a + A)
    return quad(primakoff_production_diffxs, 0, pi, args=(energy,z,ma,g), limit=3)[0]




def primakoff_production_xs(energy, z, a, ma, g):
    # Primakoff production total xs (γ + A -> a + A)
    #Tsai, '86 (ma << E)
    if energy < ma:
        return 0
    M_E = 0.511
    prefactor = (1 / 137 / 4) * (g ** 2)
    return prefactor * ((z ** 2) * (np.log(184 * np.power(z, -1 / 3)) \
        + np.log(403 * np.power(a, -1 / 3) / M_E)) \
        + z * np.log(1194 * np.power(z, -2 / 3)))




def dark_prim_dSdt(t, s, gZN, gaGZ, ma, mZ, M):
    # Priamkoff with massive vector mediator
    prefactor = (gZN*gaGZ)**2 / (16*pi) / ((M + ma)**2 - s) / ((M - ma)**2 - s)
    return (ma**2 * t * (M**2 + s)- (M*ma**2)**2 - t((s-M**2)**2 + s*t) - t*(t-ma**2)/2) / (t-mZ**2)**2




#### Electron coupling ####

def compton_production_dSdEa(ea, eg, g, ma):
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




def brem_prod_d2SdEadOmega(Ea, thetaa, Ee, g, ma, z):
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




def resonant_production(ea, ee, g, ma):
    # Resonant production cross section (e- e+ -> a)
    pass





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
    


# Directional axion production from beam-produced photon distribution
# The beam lies in the z-direction with detector cross-section lying on-axis perpendicular to beam
# We compute the energy and polar angle (w.r.t. beam axis) of the produced axion flux
class PrimakoffAxionFromBeam:
    def __init__(self, photon_rates=[1.,1.,0.], target_z=90, target_photon_cross=15e-24,
                 detector_distance=4., detector_length=0.2, detector_area=0.04, det_z=18,
                 axion_mass=0.1, axion_coupling=1e-3, nsamples=10000):
        self.photon_rates = photon_rates  # per second
        self.axion_mass = axion_mass  # MeV
        self.axion_coupling = axion_coupling  # MeV^-1
        self.target_z = target_z
        self.target_photon_cross = target_photon_cross  # cm^2
        self.det_dist = detector_distance  # meter
        self.det_back = detector_length + detector_distance
        self.det_length = detector_length
        self.det_area = detector_area
        self.det_z = det_z
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        self.gamma_sep_angle = []
        self.nsamples = nsamples
        self.theta_edges = np.logspace(-8, np.log10(pi), nsamples + 1)
        self.thetas = exp(np.random.uniform(-12, np.log(pi), nsamples))
        self.theta_widths = self.theta_edges[1:] - self.theta_edges[:-1]
        self.phis = np.random.uniform(-pi,pi, nsamples)
        self.support = np.ones(nsamples)
        self.hist, self.binx, self.biny = np.histogram2d([0], [0], weights=[0],
                                                         bins=[np.logspace(-1,5,65),np.logspace(-8,np.log10(pi),65)])
    
    def det_sa(self):
        return np.arctan(sqrt(self.det_area / pi) / self.det_dist)

    def branching_ratio(self, energy):
        cross_prim = primakoff_production_xs(energy, self.target_z, 2*self.target_z,
                                             self.axion_mass, self.axion_coupling)
        return cross_prim / (cross_prim + (self.target_photon_cross / (100 * METER_BY_MEV) ** 2))
    
    def get_beaming_angle(self, v):
        return np.arcsin(sqrt(1-v**2))
    
    def theta_z(self, theta, cosphi, theta_gamma):
        return abs(arccos(sin(theta_gamma)*cosphi*sin(theta) + cos(theta_gamma)*cos(theta)))
    
    
    # Simulate the 2D differential angular-energy axion flux.
    def simulate_kinematics_single(self, photon):
        if photon[0] < self.axion_mass:
            return np.histogram2d([0], [0], weights=[0],
                                  bins=[np.logspace(-1,5,65), np.logspace(-5,np.log10(pi),65)])[0]
        rate = photon[2]
        e_gamma = photon[0]
        theta_gamma = photon[1]
        
        # Get axion Lorentz transformations and kinematics
        p_a = sqrt(e_gamma**2 - self.axion_mass**2)
        v_a = p_a / e_gamma
        axion_boost = e_gamma / self.axion_mass
        tau = 64 * pi / (self.axion_coupling ** 2 * self.axion_mass ** 3) * axion_boost

        # Get decay and survival probabilities
        surv_prob = np.exp(-self.det_dist / METER_BY_MEV / v_a / tau)
        decay_prob = 1.0000000000-np.exp(-self.det_length / METER_BY_MEV / v_a / tau)
        decay_weight = surv_prob * decay_prob
        br = 1/(self.target_photon_cross / (100 * METER_BY_MEV) ** 2)
        
        weight = rate * br * decay_weight * self.axion_coupling**2
        
        def integrand(theta, phi):
            return primakoff_production_diffxs(theta, e_gamma, self.target_z, self.axion_mass)
        
        thetas_z = arccos(cos(self.thetas)*cos(theta_gamma) + cos(self.phis)*sin(self.thetas)*sin(theta_gamma))
        
        convolution = np.vectorize(integrand)
        return np.histogram2d(e_gamma*self.support, thetas_z, weights=weight*2*pi*convolution(self.thetas, self.phis)*self.theta_widths,
                              bins=[np.logspace(-1,5,65), np.logspace(-8,np.log10(pi),65)])[0]
        

    # Simulate the angular-integrated energy flux.
    def simulate_int(self, photon):
        data_tuple = ([], [], [], [])

        if photon[0] < self.axion_mass:
            return data_tuple
        rate = photon[2]
        e_gamma = photon[0]
        theta_gamma = abs(photon[1])

        # Simulate
        def heaviside(theta, phi):
            return self.det_sa() > arccos(cos(theta)*cos(theta_gamma) \
                                   + cos(phi)*sin(theta)*sin(theta_gamma))

        def integrand(theta, phi):
            return heaviside(theta, phi) * \
                   primakoff_production_diffxs(theta, e_gamma, self.target_z, self.axion_mass)
        
        convolution = np.vectorize(integrand)
        integral = 2*pi*(log(pi/exp(-12))/self.nsamples) * np.sum(convolution(self.thetas, self.phis) * self.thetas)

        # Get the branching ratio (numerator already contained in integrand func)
        br = 1/(self.target_photon_cross / (100 * METER_BY_MEV) ** 2)

        axion_p = sqrt(e_gamma** 2 - self.axion_mass ** 2)
        axion_v = axion_p / e_gamma

        # Push back lists and weights
        data_tuple[0].extend([e_gamma]) # elastic limit
        data_tuple[1].extend([theta_gamma])
        data_tuple[2].extend([rate * br * integral])  # scatter weights
        data_tuple[3].extend([np.arcsin(sqrt(1-axion_v**2))]) # beaming formula for iso decay
        return data_tuple
    
    def simulate(self, nsamples=10, multicore=False):  # simulate the ALP flux
        #t1 = time.time()
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.gamma_sep_angle = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        
        if multicore == True:
            print("Running NCPU = ", max(1, multi.cpu_count()-1))
            
            with multi.Pool(max(1, multi.cpu_count()-1)) as pool:
                ntuple = pool.map(self.simulate_int, [f for f in self.photon_rates])
                pool.close()
            
            for tup in ntuple:
                self.axion_energy.extend(tup[0])
                self.axion_angle.extend(tup[1])
                self.axion_flux.extend(tup[2])
                self.gamma_sep_angle.extend(tup[3])
        else:
            for f in self.photon_rates:
                tup = self.simulate_int(f)
                self.axion_energy.extend(tup[0])
                self.axion_angle.extend(tup[1])
                self.axion_flux.extend(tup[2])
                self.gamma_sep_angle.extend(tup[3])

    
    def simulate_kinematics(self, nsamples=10):
        #t1 = time.time()
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.gamma_sep_angle = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        
        print("Running NCPU = ", max(1, multi.cpu_count()-1))
        
        with multi.Pool(max(1, multi.cpu_count()-1)) as pool:
            ntuple = pool.map(self.simulate_kinematics_single, [f for f in self.photon_rates])
            pool.close()
        
        for tup in ntuple:
            self.hist += tup
    
    def propagate(self):  # propagate to detector
        g = self.axion_coupling
        e_a = np.array(self.axion_energy)
        wgt = np.array(self.axion_flux)

        # Get axion Lorentz transformations and kinematics
        p_a = sqrt(e_a**2 - self.axion_mass**2)
        v_a = p_a / e_a
        axion_boost = e_a / self.axion_mass
        tau = 64 * pi / (g ** 2 * self.axion_mass ** 3) * axion_boost

        # Get decay and survival probabilities
        surv_prob = np.array([mp.exp(-self.det_dist / METER_BY_MEV / v_a[i] / tau[i]) \
                     for i in range(len(v_a))])
        decay_prob = np.array([fsub(1,mp.exp(-self.det_length / METER_BY_MEV / v_a[i] / tau[i])) \
                      for i in range(len(v_a))])
        # TODO: remove g**2 multiplication here (was ad hoc to speed up / modularize)
        self.decay_axion_weight = np.asarray(g**2 * wgt * surv_prob * decay_prob, dtype=np.float64)
        self.scatter_axion_weight = np.asarray(g**2 * wgt * surv_prob, dtype=np.float64)
    
    def decay_events(self, detection_time, threshold, efficiency=None):
        res = 0
        for i in range(len(self.decay_axion_weight)):
            if self.axion_energy[i] >= threshold:
                if efficiency is not None:
                    self.decay_axion_weight[i] *= detection_time * efficiency(self.axion_energy[i])
                    res += self.decay_axion_weight[i]
                else:
                    self.decay_axion_weight[i] *= detection_time
                    res += self.decay_axion_weight[i]
            else:
                self.decay_axion_weight[i] = 0.0
        return res

    def scatter_events(self, detector_number, detector_z, detection_time, threshold, efficiency=None):
        res = 0
        r0 = 2.2e-10 / METER_BY_MEV
        for i in range(len(self.scatter_axion_weight)):
            if self.axion_energy[i] >= threshold:
                if efficiency is not None:
                    self.scatter_axion_weight[i] *= primakoff_scattering_xs(self.axion_energy[i], self.axion_coupling, 
                                                                            self.axion_mass, detector_z, r0) \
                        * efficiency(self.axion_energy[i]) * detection_time * detector_number * METER_BY_MEV ** 2
                    res += self.scatter_axion_weight[i]
                else:
                    self.scatter_axion_weight[i] *= primakoff_scattering_xs(self.axion_energy[i], self.axion_coupling, 
                                                                            self.axion_mass, detector_z, r0) \
                                                   * detection_time * detector_number * METER_BY_MEV ** 2
                    res += self.scatter_axion_weight[i]
            else:
                self.scatter_axion_weight[i] = 0.0
        return res



class IsotropicAxionFromPrimakoff:
    def __init__(self, photon_rates=[1,1], axion_mass=0.1, axion_coupling=1e-4,
                 target_mass=240e3, target_z=90, target_photon_cross=15e-24, detector_distance=4,
                 detector_length=0.2, detector_area=20):
        self.photon_rates = photon_rates  # per second
        self.axion_mass = axion_mass  # MeV
        self.axion_coupling = axion_coupling  # MeV^-1
        self.target_mass = target_mass  # MeV
        self.target_z = target_z
        self.target_photon_cross = target_photon_cross  # cm^2
        self.detector_distance = detector_distance  # meters
        self.detector_length = detector_length
        self.det_area = detector_area
        self.axion_energy = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        self.axion_velocity = []
        self.simulate()


    def branching_ratio(self, energy, coupling=1.0):
        cross_prim = primakoff_scattering_xs(energy, coupling, self.axion_mass, self.target_z, 2.2e-10 / METER_BY_MEV)/2
        return cross_prim / (cross_prim + (self.target_photon_cross / (100 * METER_BY_MEV) ** 2))

    # Convolute axion production and decay rates with a photon flux
    def simulate_single(self, energy, rate):
        if energy <= self.axion_mass:
            return
        br = mpmathify(self.branching_ratio(energy, self.axion_coupling))
        axion_p = mp.sqrt(energy ** 2 - self.axion_mass ** 2)
        axion_v = mpmathify(axion_p / energy)

        axion_boost = mpmathify(energy / self.axion_mass)
        tau = mpmathify(64 * pi / (self.axion_coupling ** 2 * self.axion_mass ** 3) * axion_boost)
        surv_prob =  mp.exp(-self.detector_distance / METER_BY_MEV / axion_v / tau)
        decay_in_detector = fsub(1,mp.exp(-self.detector_length / METER_BY_MEV / axion_v / tau))

        self.axion_velocity.append(axion_v)
        self.axion_energy.append(energy)
        self.axion_flux.append(rate * br / (4*pi*self.detector_distance ** 2))
        self.decay_axion_weight.append(rate * br * surv_prob * decay_in_detector / (4*pi*self.detector_distance ** 2))
        self.scatter_axion_weight.append(surv_prob * rate * br / (4*pi*self.detector_distance ** 2))
    
    def propagate(self):  # TODO: deprecate for isotropic generation
        g = self.axion_coupling
        e_a = np.array(self.axion_energy)
        wgt = np.array(self.axion_flux)

        # Get axion Lorentz transformations and kinematics
        p_a = sqrt(e_a**2 - self.axion_mass**2)
        v_a = p_a / e_a
        axion_boost = e_a / self.axion_mass
        tau = 64 * pi / (g ** 2 * self.axion_mass ** 3) * axion_boost

        # Get decay and survival probabilities
        surv_prob = np.array([mp.exp(-self.detector_distance / METER_BY_MEV / v_a[i] / tau[i]) \
                     for i in range(len(v_a))])
        decay_prob = np.array([fsub(1,mp.exp(-self.detector_length / METER_BY_MEV / v_a[i] / tau[i])) \
                      for i in range(len(v_a))])
        self.decay_axion_weight = np.asarray(g**2 * wgt * surv_prob * decay_prob, dtype=np.float64)
        self.scatter_axion_weight = np.asarray(g**2 * wgt * surv_prob, dtype=np.float64)

    # Loops over photon flux and fills the photon and axion energy arrays.
    def simulate(self):
        self.axion_energy = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        self.axion_velocity = []
        for f in self.photon_rates:
            self.simulate_single(f[0], f[1])

    def decay_events(self, detection_time, threshold):
        res = 0
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                self.decay_axion_weight[i] *= detection_time * self.det_area
                res += self.decay_axion_weight[i]
            else:
                self.decay_axion_weight[i] = 0.0
        return res * detection_time * self.det_area

    def scatter_events(self, detector_number, detector_z, detection_time, threshold):
        res = 0
        r0 = 2.2e-10 / METER_BY_MEV
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                wgt = self.scatter_axion_weight[i]
                xs = primakoff_scattering_xs(self.axion_energy[i], self.axion_coupling, self.axion_mass, detector_z, r0)
                self.scatter_axion_weight[i] = wgt * xs * detection_time * detector_number * METER_BY_MEV ** 2
                res += self.scatter_axion_weight[i]
            else:
                self.scatter_axion_weight[i] = 0.0
        return res

    def photon_events_binned(self, detector_area, detection_time, threshold):
        res = np.zeros(len(self.decay_axion_weight))
        scale = detection_time * detector_area
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                res[i] = self.decay_axion_weight[i]
        return res * scale

    def scatter_events_binned(self, detector_number, detector_z, detection_time, threshold):
        res = np.zeros(len(self.scatter_axion_weight))
        r0 = 2.2e-10 / METER_BY_MEV
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                res[i] = self.scatter_axion_weight[i] \
                        * primakoff_scattering_xs(self.axion_energy[i], self.axion_coupling, self.axion_mass, detector_z, r0) \
                        * detection_time * detector_number * METER_BY_MEV ** 2
        return res 



# Directional axion production from beam-produced photon distribution
# The beam lies in the z-direction with detector cross-section lying on-axis perpendicular to beam
# We compute the energy and polar angle (w.r.t. beam axis) of the produced axion flux
class ComptonAxionFromBeam:
    def __init__(self, photon_rates=[1.,1.,0.], target_z=90, target_photon_cross=15e-24,
                 detector_distance=4., detector_length=0.2, detector_area=0.04, det_z=18,
                 axion_mass=0.1, axion_coupling=1e-5, nsamples=100):
        self.photon_rates = photon_rates  # per second
        self.axion_mass = axion_mass  # MeV
        self.axion_coupling = axion_coupling  # MeV^-1
        self.target_z = target_z
        self.target_photon_cross = target_photon_cross  # cm^2
        self.det_dist = detector_distance  # meter
        self.det_back = detector_length + detector_distance
        self.det_length = detector_length
        self.det_area = detector_area
        self.det_z = det_z
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.decay_weight = []
        self.scatter_weight = []
        self.decay_sep_angle = []
        self.nsamples = nsamples
        self.theta_edges = np.logspace(-8, np.log10(pi), nsamples + 1)
        self.thetas = exp(np.random.uniform(-12, np.log(pi), nsamples))
        self.theta_widths = self.theta_edges[1:] - self.theta_edges[:-1]
        self.phis = np.random.uniform(-pi,pi, nsamples)
        self.support = np.ones(nsamples)
        self.hist, self.binx, self.biny = np.histogram2d([0], [0], weights=[0],
                                                         bins=[np.logspace(-1,5,65),np.logspace(-8,np.log10(pi),65)])
    
    def det_sa(self):
        return np.arctan(sqrt(self.det_area / pi) / self.det_dist)

    def branching_ratio(self, energy):
        cross_prim = primakoff_production_xs(energy, self.target_z, 2*self.target_z,
                                             self.axion_mass, self.axion_coupling)
        return cross_prim / (cross_prim + (self.target_photon_cross / (100 * METER_BY_MEV) ** 2))
    
    def get_beaming_angle(self, v):
        return np.arcsin(sqrt(1-v**2))
    
    def theta_z(self, theta, cosphi, theta_gamma):
        return abs(arccos(sin(theta_gamma)*cosphi*sin(theta) + cos(theta_gamma)*cos(theta)))
    
    def lifetime(self):
        if 1 < 4 * (M_E / self.axion_mass) ** 2:
            return np.inf
        return (8 * pi) / (self.axion_coupling ** 2 * self.axion_mass * np.power(1 - 4 * (M_E / self.axion_mass) ** 2, 1 / 2))

    # Simulate the angular-integrated energy flux.
    def simulate_single(self, photon):
        data_tuple = ([], [], [], [])
        if photon[0] < self.axion_mass:
            return data_tuple
        if photon[1] > self.det_sa():
            print(self.det_sa())
            return data_tuple
        
        e_gamma = photon[0]
        theta_gamma = abs(photon[1])
        rate = photon[2]

        s = 2 * M_E * e_gamma + M_E ** 2
        if s < (M_E + self.axion_mass)**2:
            return data_tuple

        axion_energies = np.linspace(self.axion_mass, e_gamma, self.nsamples) # version 2
        de = (axion_energies[-1] - axion_energies[0]) / (self.nsamples - 1)
        axion_energies = (axion_energies[1:] + axion_energies[:-1]) / 2
        dde = compton_production_dSdEa(axion_energies, e_gamma, 1.0, self.axion_mass) * de
        axion_p = sqrt(axion_energies** 2 - self.axion_mass ** 2)
        axion_v = axion_p / e_gamma

        flux_wgts = []
        # Both photons and axions decrease with decay_prob, since we assume e+e- does not make it to the detector.
        for i in range(self.nsamples - 1):
            br = dde[i] * self.target_z / (dde[i] + (self.target_photon_cross / (100 * METER_BY_MEV) ** 2))
            flux_wgts.append(br * rate)
        
        # Push back lists and weights
        data_tuple[0].extend(axion_energies) 
        data_tuple[1].extend(theta_gamma*np.ones(self.nsamples-1))
        data_tuple[2].extend(flux_wgts)
        data_tuple[3].extend(np.arcsin(sqrt(1-axion_v**2)))
        return data_tuple
    
    def simulate(self, multicore=False):  # simulate the ALP flux
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.decay_sep_angle = []
        self.decay_weight = []
        self.scatter_weight = []
        
        if multicore == True:
            print("Running NCPU = ", max(1, multi.cpu_count()-1))
            
            with multi.Pool(max(1, multi.cpu_count()-1)) as pool:
                ntuple = pool.map(self.simulate_single, [f for f in self.photon_rates])
                pool.close()
            
            for tup in ntuple:
                self.axion_energy.extend(tup[0])
                self.axion_angle.extend(tup[1])
                self.axion_flux.extend(tup[2])
                self.decay_sep_angle.extend(tup[3])
        else:
            for f in self.photon_rates:
                tup = self.simulate_single(f)
                self.axion_energy.extend(tup[0])
                self.axion_angle.extend(tup[1])
                self.axion_flux.extend(tup[2])
                self.decay_sep_angle.extend(tup[3])
    
    def propagate(self):  # propagate to detector
        g = self.axion_coupling
        e_a = np.array(self.axion_energy)
        wgt = np.array(self.axion_flux)

        # Get axion Lorentz transformations and kinematics
        p_a = sqrt(e_a**2 - self.axion_mass**2)
        v_a = p_a / e_a
        axion_boost = e_a / self.axion_mass

        # Get decay and survival probabilities
        #surv_prob = np.array([mp.exp(-self.det_dist / METER_BY_MEV / v_a[i] / (axion_boost[i] * self.lifetime())) \
          #           for i in range(len(v_a))])
        #decay_prob = np.array([fsub(1,mp.exp(-self.det_length / METER_BY_MEV / v_a[i] / (axion_boost[i] * self.lifetime()))) \
         #             for i in range(len(v_a))])
        surv_prob = np.exp(-self.det_dist / METER_BY_MEV / v_a / (axion_boost * self.lifetime()))
        decay_prob = 1.0 - np.exp(-self.det_length / METER_BY_MEV / v_a / (axion_boost * self.lifetime()))
        
        # TODO: remove g**2 multiplication here (was ad hoc to speed up / modularize)
        self.decay_weight = np.asarray(g**2 * wgt * surv_prob * decay_prob, dtype=np.float64)
        self.scatter_weight = np.asarray(g**2 * wgt * surv_prob, dtype=np.float64)
    
    def decay_events(self, detector_area, detection_time, threshold):
        res = 0
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                self.decay_weight[i] *= detection_time * detector_area
                res += self.decay_weight[i]
            else:
                self.decay_weight[i] = 0.0
        return res
    
    def scatter_events(self, detector_number, detector_z, detection_time, threshold):
        res = 0
        xs = [compton_scattering_xs(this_energy, self.axion_coupling) for this_energy in self.axion_energy]
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                self.scatter_weight[i] *= xs[i] * METER_BY_MEV**2 * detection_time * detector_number * detector_z
                res += self.scatter_weight[i] # approx scatter_xs = prod_xs
            else:
                self.scatter_weight[i] = 0.0
        return res 



class IsotropicAxionFromCompton:
    def __init__(self, photon_rates=[1,1], axion_mass=0.1, axion_coupling=1e-4,
                 target_z=90, target_photon_cross=15e-24,
                 detector_distance=4, detector_length=0.2, detector_area=20):
        self.photon_rates = photon_rates  # per second
        self.axion_mass = axion_mass  # MeV
        self.axion_coupling = axion_coupling  # MeV^-1
        self.target_z = target_z
        self.target_photon_cross = target_photon_cross  # cm^2
        self.detector_distance = detector_distance  # meter
        self.detector_length = detector_length
        self.det_area = detector_area
        self.axion_energy = []
        self.scatter_weight = []
        self.decay_weight = []
        self.photon_energy = []
        self.photon_weight = []
        self.electron_energy = []
        self.electron_weight = []
        self.axion_scatter_cross = []
        self.epem_angle = []
        self.emgamma_angle = []
        self.simulate()


    def simulate_single(self, eg, rate):
        s = 2 * M_E * eg + M_E ** 2
        aa = self.axion_coupling ** 2 / 4 / pi
        a = 1 / 137
        ma = self.axion_mass
        if s < (M_E + self.axion_mass)**2:
            return

        ne = 100
        axion_energies = np.linspace(ma, eg, ne) # version 2
        de = (axion_energies[-1] - axion_energies[0]) / (ne - 1)
        axion_energies = (axion_energies[1:] + axion_energies[:-1]) / 2
        dde = compton_production_dSdEa(axion_energies, eg, self.axion_coupling, self.axion_mass) * de
        cross_scatter = compton_scattering_xs(axion_energies, self.axion_coupling)

        # Both photons and axions decrease with decay_prob, since we assume e+e- does not make it to the detector.
        for i in range(ne - 1):
            axion_prob = dde[i] * self.target_z / (dde[i] + (self.target_photon_cross / (100 * METER_BY_MEV) ** 2))
            surv_prob = self.AxionSurvProb(axion_energies[i])
            decay_prob = self.AxionDecayProb(axion_energies[i])
            axion_v = sqrt(axion_energies[i] ** 2 - self.axion_mass ** 2) / axion_energies[i]

            if np.isnan(axion_prob):
                print("dde = ", dde[i])
                print(axion_energies[i])
            
            self.axion_energy.append(axion_energies[i])
            self.scatter_weight.append(surv_prob * rate * axion_prob / (4 * pi * self.detector_distance ** 2))
            self.decay_weight.append(decay_prob * rate * axion_prob / (4 * pi * self.detector_distance ** 2))
            self.axion_scatter_cross.append(cross_scatter[i])
            self.epem_angle.append(arcsin(sqrt(1-axion_v**2)))

    def AxionDecayProb(self, ea):
        # Decay the axions in flight to e+ e-.
        # Returns probability that it will decay inside the detector volume.
        if ea <= self.axion_mass:
            return 0.0
        axion_p = np.sqrt(ea ** 2 - self.axion_mass ** 2)
        axion_v = axion_p / ea
        axion_boost = ea / self.axion_mass
        tau = (8 * pi) / (self.axion_coupling ** 2 * self.axion_mass
                             * np.power(1 - 4 * (M_E / self.axion_mass) ** 2, 1 / 2)) \
              if 1 - 4 * (M_E / self.axion_mass) ** 2 > 0 else np.inf  # lifetime for a -> gamma gamma
        tau *= axion_boost
        return np.exp(-self.detector_distance / METER_BY_MEV / axion_v / tau) \
               * (1.0 - np.exp(-self.detector_length / METER_BY_MEV / axion_v / tau))
    
    def AxionSurvProb(self, ea):
        # Decay the axions in flight to e+ e-.
        # Returns probability that it will decay inside the detector volume.
        if ea <= self.axion_mass:
            return 0.0
        axion_p = np.sqrt(ea ** 2 - self.axion_mass ** 2)
        axion_v = axion_p / ea
        axion_boost = ea / self.axion_mass
        tau = (8 * pi) / (self.axion_coupling ** 2 * self.axion_mass
                             * np.power(1 - 4 * (M_E / self.axion_mass) ** 2, 1 / 2)) \
              if 1 - 4 * (M_E / self.axion_mass) ** 2 > 0 else np.inf  # lifetime for a -> gamma gamma
        tau *= axion_boost
        return mp.exp(-self.detector_distance / METER_BY_MEV / axion_v / tau)

    def simulate(self, nsamplings=1000):
        self.photon_energy = []
        self.photon_weight = []
        self.axion_energy = []
        self.electron_energy = []
        self.electron_weight = []
        self.decay_weight = []
        self.scatter_weight = []
        self.epem_angle = []
        self.emgamma_angle = []
        for f in self.photon_rates:
          self.simulate_single(f[0], f[1])

    def photon_events(self, detector_area, detection_time, threshold):
        res = 0
        for i in range(len(self.photon_energy)):
          if self.photon_energy[i] >= threshold:
            res += self.photon_weight[i]
        return res * detection_time * detector_area

    def electron_events_binned(self, nbins, detector_number, detector_z, detection_time, threshold):
        self.electron_energy = []
        self.electron_weight = []
        self.photon_weight = []
        self.photon_energy = []
        for i in range(len(self.axion_energy) - 1): # integrate over E_a
            Et_max = 2 * np.max(self.axion_energy[i]) ** 2 / (M_E + 2 * np.max(self.axion_energy[i]))
            Et = np.linspace(0, Et_max, nbins)  # electron energies
            delta_Et = (Et[-1] - Et[0]) / (nbins - 1)
            Et = (Et[1:] + Et[:-1]) / 2  # get bin centers

            # Get differential scattering rate
            dSigma_dEt = compton_scattering_he_dSdEt(self.axion_energy[i], Et, self.axion_coupling)
            if np.any(dSigma_dEt < 0):
                continue

            sigma = np.sum(dSigma_dEt) * delta_Et  # total cross-section

            # Fill in electrons..
            for j in range(Et.shape[0]-1): # Integrate over E_t
                if Et[j] < threshold:
                    continue
                if Et[j] > Et_max:
                    continue

                scatter_rate = self.scatter_weight[i] * (dSigma_dEt[j] / sigma) * self.axion_scatter_cross[i] * delta_Et
                exposure = METER_BY_MEV ** 2 * detection_time * detector_number * detector_z
                self.electron_weight.append(scatter_rate * exposure)
                self.electron_energy.append(Et[j])
                self.photon_weight.append(scatter_rate * exposure)
                self.photon_energy.append(self.axion_energy[i]-Et[j])

        return np.sum(self.electron_weight), np.sum(self.photon_weight)


    def scatter_events(self, detector_number, detector_z, detection_time, threshold):
        res = 0
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                self.scatter_weight[i] *= self.axion_scatter_cross[i]*METER_BY_MEV**2 * detection_time * detector_number * detector_z
                res += self.scatter_weight[i] # approx scatter_xs = prod_xs
            else:
                self.scatter_weight[i] = 0.0
        return res 

    def scatter_events_binned(self, detector_number, detector_z, detection_time, threshold):
        res = np.zeros(len(self.scatter_weight))
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                res[i] = self.scatter_weight[i] * self.axion_scatter_cross[i]  # approx scatter_xs = prod_xs
        return res * METER_BY_MEV ** 2 * detection_time * detector_number * detector_z

    def decay_events(self, detector_area, detection_time, threshold):
        res = 0
        for i in range(len(self.axion_energy)):
            if self.axion_energy[i] >= threshold:
                self.decay_weight[i] *= detection_time * detector_area
                res += self.decay_weight[i]
            else:
                self.decay_weight[i] = 0.0
        return res




class BremAxionFromLepton:
    def __init__(self, electron_flux=[1.,1.,0.], target_z=90, sm_electron_xs=15e-24,
                 detector_distance=4., detector_length=0.2, detector_area=0.04, det_z=18,
                 axion_mass=0.1, axion_coupling=1e-3, nsamples=10000):
        self.electron_flux = electron_flux  # per second
        self.axion_mass = axion_mass  # MeV
        self.axion_coupling = axion_coupling  # MeV^-1
        self.target_z = target_z
        self.sm_electron_xs = sm_electron_xs  # cm^2
        self.det_dist = detector_distance  # meter
        self.det_back = detector_length + detector_distance
        self.det_length = detector_length
        self.det_area = detector_area
        self.det_z = det_z
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        self.gamma_sep_angle = []
        self.nsamples = nsamples
        self.theta_edges = np.logspace(-8, np.log10(pi), nsamples + 1)
        self.thetas = exp(np.random.uniform(-12, np.log(pi), nsamples))
        self.theta_widths = self.theta_edges[1:] - self.theta_edges[:-1]
        self.phis = np.random.uniform(-pi,pi, nsamples)
        self.support = np.ones(nsamples)
        self.hist, self.binx, self.biny = np.histogram2d([0], [0], weights=[0],
                                                         bins=[np.logspace(-1,5,65),np.logspace(-8,np.log10(pi),65)])
    
    def det_sa(self):
        return np.arctan(sqrt(self.det_area / pi) / self.det_dist)

    def branching_ratio(self, energy):
        cross_prim = primakoff_production_xs(energy, self.target_z, 2*self.target_z,
                                             self.axion_mass, self.axion_coupling)
        return cross_prim / (cross_prim + (self.sm_electron_xs / (100 * METER_BY_MEV) ** 2))
    
    def get_beaming_angle(self, v):
        return np.arcsin(sqrt(1-v**2))
    
    def theta_z(self, theta, cosphi, theta_gamma):
        return abs(arccos(sin(theta_gamma)*cosphi*sin(theta) + cos(theta_gamma)*cos(theta)))
    
    
    # Simulate the 2D differential angular-energy axion flux.
    def simulate_kinematics_single(self, photon):
        if photon[0] < self.axion_mass:
            return np.histogram2d([0], [0], weights=[0],
                                  bins=[np.logspace(-1,5,65), np.logspace(-5,np.log10(pi),65)])[0]
        rate = photon[2]
        e_gamma = photon[0]
        theta_gamma = photon[1]
        
        # Get axion Lorentz transformations and kinematics
        p_a = sqrt(e_gamma**2 - self.axion_mass**2)
        v_a = p_a / e_gamma
        axion_boost = e_gamma / self.axion_mass
        tau = 64 * pi / (self.axion_coupling ** 2 * self.axion_mass ** 3) * axion_boost

        # Get decay and survival probabilities
        surv_prob = np.exp(-self.det_dist / METER_BY_MEV / v_a / tau)
        decay_prob = 1.0000000000-np.exp(-self.det_length / METER_BY_MEV / v_a / tau)
        decay_weight = surv_prob * decay_prob
        br = 1/(self.sm_electron_xs / (100 * METER_BY_MEV) ** 2)
        
        weight = rate * br * decay_weight * self.axion_coupling**2
        
        def integrand(theta, phi):
            return primakoff_production_diffxs(theta, e_gamma, self.target_z, self.axion_mass)
        
        thetas_z = arccos(cos(self.thetas)*cos(theta_gamma) + cos(self.phis)*sin(self.thetas)*sin(theta_gamma))
        
        convolution = np.vectorize(integrand)
        return np.histogram2d(e_gamma*self.support, thetas_z, weights=weight*2*pi*convolution(self.thetas, self.phis)*self.theta_widths,
                              bins=[np.logspace(-1,5,65), np.logspace(-8,np.log10(pi),65)])[0]
        

    # Simulate the angular-integrated energy flux.
    def flux_integral(self, electron):
        data_tuple = ([], [], [], [])

        if electron[0] < self.axion_mass:
            return data_tuple
        rate = electron[2]
        Ee = electron[0]
        thetae = abs(electron[1])

        # Simulate
        def heaviside(theta, phi):
            return self.det_sa() > arccos(cos(theta)*cos(thetae) \
                                   + cos(phi)*sin(theta)*sin(thetae))

        thetaa_list = np.random.uniform(0, pi, self.nsamples)
        ea_list = np.random.uniform(self.axion_mass, Ee, self.nsamples)

        def integrand(theta, phi):
            return heaviside(theta, phi) * \
                   primakoff_production_diffxs(theta, Ee, self.target_z, self.axion_mass)
        
        convolution = np.vectorize(integrand)
        integral = 2*pi*(log(pi/exp(-12))/self.nsamples) * np.sum(convolution(self.thetas, self.phis) * self.thetas)

        # Get the branching ratio (numerator already contained in integrand func)
        br = 1/(self.sm_electron_xs / (100 * METER_BY_MEV) ** 2)

        axion_p = sqrt(Ee** 2 - self.axion_mass ** 2)
        axion_v = axion_p / Ee

        # Push back lists and weights
        data_tuple[0].extend([e_gamma]) # elastic limit
        data_tuple[1].extend([theta_gamma])
        data_tuple[2].extend([rate * br * integral])  # scatter weights
        data_tuple[3].extend([np.arcsin(sqrt(1-axion_v**2))]) # beaming formula for iso decay
        return data_tuple
    
    def simulate(self, nsamples=10, multicore=False):  # simulate the ALP flux
        #t1 = time.time()
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.gamma_sep_angle = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        
        if multicore == True:
            print("Running NCPU = ", max(1, multi.cpu_count()-1))
            
            with multi.Pool(max(1, multi.cpu_count()-1)) as pool:
                ntuple = pool.map(self.flux_integral, [f for f in self.electron_flux])
                pool.close()
            
            for tup in ntuple:
                self.axion_energy.extend(tup[0])
                self.axion_angle.extend(tup[1])
                self.axion_flux.extend(tup[2])
                self.gamma_sep_angle.extend(tup[3])
        else:
            for f in self.photon_rates:
                tup = self.simulate_int(f)
                self.axion_energy.extend(tup[0])
                self.axion_angle.extend(tup[1])
                self.axion_flux.extend(tup[2])
                self.gamma_sep_angle.extend(tup[3])
    
    def propagate(self):  # propagate to detector
        g = self.axion_coupling
        e_a = np.array(self.axion_energy)
        wgt = np.array(self.axion_flux)

        # Get axion Lorentz transformations and kinematics
        p_a = sqrt(e_a**2 - self.axion_mass**2)
        v_a = p_a / e_a
        axion_boost = e_a / self.axion_mass
        tau = 64 * pi / (g ** 2 * self.axion_mass ** 3) * axion_boost

        # Get decay and survival probabilities
        surv_prob = np.array([mp.exp(-self.det_dist / METER_BY_MEV / v_a[i] / tau[i]) \
                     for i in range(len(v_a))])
        decay_prob = np.array([fsub(1,mp.exp(-self.det_length / METER_BY_MEV / v_a[i] / tau[i])) \
                      for i in range(len(v_a))])
        # TODO: remove g**2 multiplication here (was ad hoc to speed up / modularize)
        self.decay_axion_weight = np.asarray(g**2 * wgt * surv_prob * decay_prob, dtype=np.float64)
        self.scatter_axion_weight = np.asarray(g**2 * wgt * surv_prob, dtype=np.float64)
    
    def decay_events(self, detection_time, threshold, efficiency=None):
        res = 0
        for i in range(len(self.decay_axion_weight)):
            if self.axion_energy[i] >= threshold:
                if efficiency is not None:
                    self.decay_axion_weight[i] *= detection_time * efficiency(self.axion_energy[i])
                    res += self.decay_axion_weight[i]
                else:
                    self.decay_axion_weight[i] *= detection_time
                    res += self.decay_axion_weight[i]
            else:
                self.decay_axion_weight[i] = 0.0
        return res

    def scatter_events(self, detector_number, detector_z, detection_time, threshold, efficiency=None):
        res = 0
        r0 = 2.2e-10 / METER_BY_MEV
        for i in range(len(self.scatter_axion_weight)):
            if self.axion_energy[i] >= threshold:
                if efficiency is not None:
                    self.scatter_axion_weight[i] *= primakoff_scattering_xs(self.axion_energy[i], self.axion_coupling, 
                                                                            self.axion_mass, detector_z, r0) \
                        * efficiency(self.axion_energy[i]) * detection_time * detector_number * METER_BY_MEV ** 2
                    res += self.scatter_axion_weight[i]
                else:
                    self.scatter_axion_weight[i] *= primakoff_scattering_xs(self.axion_energy[i], self.axion_coupling, 
                                                                            self.axion_mass, detector_z, r0) \
                                                   * detection_time * detector_number * METER_BY_MEV ** 2
                    res += self.scatter_axion_weight[i]
            else:
                self.scatter_axion_weight[i] = 0.0
        return res

