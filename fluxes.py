# ALP Fluxes, DM Fluxes
# All fluxes in cm^-2 s^-1 or cm^-2 s^-1 MeV^-1


from .constants import *
from .fmath import *
from .materials import *
from .decay import *
from .prod_xs import *
from .det_xs import *
from .photon_xs import *


class AxionFlux:
    # Generic superclass for constructing fluxes
    def __init__(self, axion_mass, target: Material, detector: Material,
                    det_dist, det_length, det_area, nsamples=1000):
        self.ma = axion_mass
        self.target_z = target.z[0]  # TODO: take z array for compound mats
        self.det_z = detector.z[0]
        self.det_dist = det_dist  # meters
        self.det_length = det_length  # meters
        self.det_area = det_area  # square meters
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        self.nsamples = nsamples
    
    def det_sa(self):
        return arctan(sqrt(self.det_area / pi) / self.det_dist)
    
    def propagate(self, decay_width, rescale_factor=1.0):
        e_a = np.array(self.axion_energy)
        wgt = np.array(self.axion_flux)

        # Get axion Lorentz transformations and kinematics
        p_a = sqrt(e_a**2 - self.ma**2)
        v_a = p_a / e_a
        boost = e_a / self.ma
        tau = boost / decay_width if decay_width > 0.0 else np.inf * np.ones_like(boost)

        # Get decay and survival probabilities
        surv_prob = np.array([mp.exp(-self.det_dist / METER_BY_MEV / v_a[i] / tau[i]) \
                     for i in range(len(v_a))])
        decay_prob = np.array([fsub(1, mp.exp(-self.det_length / METER_BY_MEV / v_a[i] / tau[i])) \
                      for i in range(len(v_a))])

        self.decay_axion_weight = np.asarray(rescale_factor * wgt * surv_prob * decay_prob, dtype=np.float32)  # removed g^2
        self.scatter_axion_weight = np.asarray(rescale_factor * wgt * surv_prob, dtype=np.float32)  # removed g^2




class FluxPrimakoff(AxionFlux):
    # Generator for ALP flux from 2D photon spectrum (E_\gamma, \theta_\gamma)
    def __init__(self, axion_mass, target_z, det_z, det_dist, det_length, det_area, nsamples=1000):
        super().__init__(axion_mass, target_z, det_z, det_dist, det_length, det_area, nsamples)
    
    def decay_width(self):
        pass

    def simulate_single(self):
        pass

    def simulate(self):
        pass




class FluxPrimakoffIsotropic(AxionFlux):
    # Generator for ALP flux from 2D photon spectrum (E_\gamma, \theta_\gamma)
    def __init__(self, axion_mass, target: Material, detector: Material,
                    det_dist, det_length, det_area, nsamples=1000):
        super().__init__(axion_mass, target, detector, det_dist, det_length, det_area, nsamples)
    
    def decay_width(self):
        pass

    def simulate_single(self):
        pass

    def simulate(self):
        pass




class FluxCompton(AxionFlux):
    # Generator for ALP flux from 2D photon spectrum (E_\gamma, \theta_\gamma)
    def __init__(self, axion_mass, target: Material, detector: Material,
                    det_dist, det_length, det_area, nsamples=1000):
        super().__init__(axion_mass, target, detector, det_dist, det_length, det_area, nsamples)
    
    def decay_width(self):
        pass

    def simulate_single(self):
        pass

    def simulate(self):
        pass




class FluxComptonIsotropic(AxionFlux):
    """
    Generator for axion-bremsstrahlung flux
    Takes in a flux of el
    """
    def __init__(self, photon_flux=[1,1], target=Material("W"), detector=Material("Ar"), det_dist=4.,
                    det_length=0.2, det_area=0.04, axion_mass=0.1, axion_coupling=1e-3, nsamples=100):
        super().__init__(axion_mass, target, detector, det_dist, det_length, det_area)
        self.photon_flux = photon_flux
        self.ge = axion_coupling
        self.nsamples = nsamples
        self.target_photon_xs = AbsCrossSection(target)
    
    def decay_width(self, ge, ma):
        return W_ee(ge, ma)

    def simulate_single(self, photon):
        gamma_energy = photon[0]
        gamma_wgt = photon[1]

        s = 2 * M_E * gamma_energy + M_E ** 2
        if s < (M_E + self.ma)**2:
            return

        ea_rnd = np.random.uniform(self.ma, gamma_energy, self.nsamples)
        mc_xs = (gamma_energy - self.ma) * compton_dsigma_dea(ea_rnd, gamma_energy, self.ge, self.ma, self.target_z) / self.nsamples
        diff_br = mc_xs / self.target_photon_xs.sigma_mev(gamma_energy)

        for i in range(self.nsamples):
            self.axion_energy.append(ea_rnd[i])
            self.axion_flux.append(gamma_wgt * diff_br[i])

    def simulate(self):
        self.axion_energy = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        for i, el in enumerate(self.photon_flux):
            self.simulate_single(el)
    
    def propagate(self, new_coupling=None):
        if new_coupling is not None:
            super().propagate(W_ee(new_coupling, self.ma), rescale_factor=power(new_coupling/self.ge, 2))
        else:
            super().propagate(W_ee(self.ge, self.ma))
        geom_accept = self.det_area / (4*pi*self.det_dist**2)
        self.decay_axion_weight *= geom_accept
        self.scatter_axion_weight *= geom_accept




def track_length_prob(Ei, Ef, t):
    b = 4/3
    return heaviside(Ei-Ef, 0.0) * abs(power(log(Ei/Ef), b*t - 1) / (Ei * gamma(b*t)))




class FluxBremIsotropic(AxionFlux):
    """
    Generator for axion-bremsstrahlung flux
    Takes in a flux of el
    """
    def __init__(self, electron_flux=[1.,0.], target=Material("W"), detector=Material("Ar"), target_density=19.3,
                 target_radiation_length=6.76, target_length=10.0, det_dist=4., det_length=0.2,
                 det_area=0.04, axion_mass=0.1, axion_coupling=1e-3, nsamples=100):
        super().__init__(axion_mass, target, detector, det_dist, det_length, det_area)
        # TODO: make flux take in a Detector class and a Target class (possibly Material class?)
        # Replace A = 2*Z with real numbers of nucleons
        self.electron_flux = electron_flux
        self.positron_flux = electron_flux
        self.ge = axion_coupling
        self.target_density = target_density  # g/cm3
        self.target_radius = target_length  # cm
        self.ntargets_by_area = target_length * target_density * AVOGADRO / (2*target.z[0])  # N_T / cm^2
        self.ntarget_area_density = target_radiation_length * AVOGADRO / (2*target.z[0])
        self.nsamples = nsamples
    
    def decay_width(self):
        return W_ee(self.ge, self.ma)
    
    def electron_flux_dN_dE(self, energy):
        # TODO: remove ad hoc factor of 2 and use actual electron/positron flux
        return 2*np.interp(energy, self.electron_flux[:,0], self.electron_flux[:,1], left=0.0, right=0.0)
    
    def electron_flux_attenuated(self, t, E0, E1):
        return self.electron_flux_dN_dE(E0) * track_length_prob(E0, E1, t)

    def simulate_single(self, electron):
        el_energy = electron[0]
        el_wgt = electron[1]

        ea_max = el_energy * (1 - power(self.ma/el_energy, 2))
        #ea_max = el_energy
        if ea_max < self.ma:
            return

        ea_rnd = np.random.uniform(self.ma, ea_max, self.nsamples)
        mc_vol = (ea_max - self.ma)/self.nsamples
        diff_br = (self.ntarget_area_density * HBARC**2) * mc_vol * brem_dsigma_dea(ea_rnd, el_energy, self.ge, self.ma, self.target_z)

        for i in range(self.nsamples):
            self.axion_energy.append(ea_rnd[i])
            self.axion_flux.append(el_wgt * diff_br[i])

    def simulate(self):
        self.axion_energy = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        for i, el in enumerate(self.electron_flux):
            self.simulate_single(el)
    
    def propagate(self, new_coupling=None):
        if new_coupling is not None:
            super().propagate(W_ee(new_coupling, self.ma), rescale_factor=power(new_coupling/self.ge, 2))
        else:
            super().propagate(W_ee(self.ge, self.ma))
        geom_accept = self.det_area / (4*pi*self.det_dist**2)
        self.decay_axion_weight *= geom_accept
        self.scatter_axion_weight *= geom_accept




class FluxResonanceIsotropic(AxionFlux):
    """
    Generator for e+ e- resonant ALP production flux
    Takes in a flux of el
    """
    def __init__(self, positron_flux=[1.,0.], target=Material("W"), detector=Material("Ar"), target_density=19.3, target_length=10.0,
                 target_radiation_length=6.76, det_dist=4., det_length=0.2, det_area=0.04,
                 axion_mass=0.1, axion_coupling=1e-3, nsamples=100):
        # TODO: make flux take in a Detector class and a Target class (possibly Material class?)
        # Replace A = 2*Z with real numbers of nucleons
        super().__init__(axion_mass, target, detector, det_dist, det_length, det_area, nsamples)
        self.positron_flux = positron_flux  # differential positron energy flux dR / dE+ / s
        self.positron_flux_bin_widths = positron_flux[1:,0] - positron_flux[:-1,0]
        self.ge = axion_coupling
        self.target_density = target_density  # g/cm3
        self.target_radius = target_length  # cm
        self.ntargets_by_area = target_length * target_density * AVOGADRO / (2*target.z[0])  # N_T / cm^2
        self.ntarget_area_density = target_radiation_length * AVOGADRO / (2*target.z[0])
    
    def decay_width(self):
        return W_ee(self.ge, self.ma)
    
    def positron_flux_dN_dE(self, energy):
        return np.interp(energy, self.positron_flux[:,0], self.positron_flux[:,1], left=0.0, right=0.0)
    
    def positron_flux_attenuated(self, t, energy_pos, energy_res):
        return self.positron_flux_dN_dE(energy_pos) * track_length_prob(energy_pos, energy_res, t)

    def simulate(self):
        self.axion_energy = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        resonant_energy = self.ma**2 / (2 * M_E)
        if resonant_energy + M_E < self.ma:
            return
        
        if resonant_energy > max(self.positron_flux[:,0]):
            return
        
        e_rnd = np.random.uniform(resonant_energy, max(self.positron_flux[:,0]), self.nsamples)
        t_rnd = np.random.uniform(0.0, 5.0, self.nsamples)
        mc_vol = (5.0 - 0.0)*(max(self.positron_flux[:,0]) - resonant_energy)

        attenuated_flux = mc_vol*np.sum(self.positron_flux_attenuated(t_rnd, e_rnd, resonant_energy))/self.nsamples
        wgt = (self.ntarget_area_density * HBARC**2) * resonance_peak(self.ge) * attenuated_flux
        
        self.axion_energy.append(resonant_energy + M_E)
        self.axion_flux.append(wgt)
    
    def propagate(self, new_coupling=None):
        if new_coupling is not None:
            super().propagate(W_ee(new_coupling, self.ma), rescale_factor=power(new_coupling/self.ge, 2))
        else:
            super().propagate(W_ee(self.ge, self.ma))
        geom_accept = self.det_area / (4*pi*self.det_dist**2)
        self.decay_axion_weight *= geom_accept
        self.scatter_axion_weight *= geom_accept




class ElectronEventGenerator:
    """
    Takes in an AxionFlux at the detector (N/s) and gives scattering / decay rates (# events)
    """
    def __init__(self, flux: AxionFlux, detector: Material):
        self.flux = flux
        self.det_z = detector.z[0]
        self.axion_energy = np.array(flux.axion_energy)
        self.decay_weights = np.zeros_like(flux.decay_axion_weight)
        self.scatter_weights = np.zeros_like(flux.scatter_axion_weight)
        self.pair_weights = np.zeros_like(flux.scatter_axion_weight)
        self.efficiency = None  # TODO: add efficiency info
        self.energy_threshold = None  # TODO: add threshold as member var
        self.pair_xs = PairProdutionCrossSection(detector)

    def pair_production(self, ge, ma, ntargets, days_exposure, threshold):
        # TODO: remove this ad hoc XS and replace with real calc
        self.pair_weights += days_exposure * S_PER_DAY * (ntargets / self.flux.det_area) \
            * (2 * ge**2 / ALPHA)*self.pair_xs.sigma_mev(sqrt(self.axion_energy**2 - ma**2)) \
                * METER_BY_MEV**2 * self.flux.scatter_axion_weight * heaviside(self.axion_energy - threshold, 0.0)
        res = np.sum(self.pair_weights)
        return res

    def compton(self, ge, ma, ntargets, days_exposure, threshold):
        self.scatter_weights += days_exposure * S_PER_DAY * (ntargets / self.flux.det_area) \
            * icompton_sigma(self.axion_energy, ma, ge, self.det_z) \
                * METER_BY_MEV**2 * self.flux.scatter_axion_weight * heaviside(self.axion_energy - threshold, 0.0)
        res = np.sum(self.scatter_weights)
        return res
    
    def decays(self, days_exposure, threshold):
        res = 0
        self.decay_weights = days_exposure * S_PER_DAY * self.flux.decay_axion_weight * heaviside(self.axion_energy - threshold, 0.0)
        res = np.sum(self.decay_weights)
        return res




class PhotonEventGenerator:
    def __init__(self, flux: AxionFlux):
        self.flux = flux

    def inverse_primakoff(self):
        pass

    def decay(self):
        pass



##### DARK MATTER FLUXES #####
rho_chi = 0.4e6 #(* keV / cm^3 *)
vesc = 544.0e6
v0 = 220.0e6
ve = 244.0e6
nesc = erf(vesc/v0) - 2*(vesc/v0) * exp(-(vesc/v0)**2) * sqrt(pi)

def fv(v):  # Velocity profile ( v ~ [0,1] )
    return (1.0 / (nesc * np.power(pi,3/2) * v0**3)) * exp(-((v + ve)**2 / v0**2))

def DMFlux(v, m):  # DM differential flux as a function of v~[0,1] and the DM mass
    return heaviside(v + v0 - vesc) * 4*pi*C_LIGHT*(rho_chi / m) * (C_LIGHT*v)**3 * fv(C_LIGHT*v)


##### NUCLEAR COUPLING FLUXES #####

def Fe57SolarFlux(gp): 
    # Monoenergetic flux at 14.4 keV from the Sun
    return (4.56e23) * gp**2


##### ELECTRON COUPLING FLUXES #####


##### PHOTON COUPLING #####

def PrimakoffSolarFlux(Ea, gagamma):
    # Solar ALP flux
    # input Ea in keV, gagamma in GeV-1
    return (gagamma * 1e8)**2 * (5.95e14 / 1.103) * (Ea / 1.103)**3 / (exp(Ea / 1.103) - 1)