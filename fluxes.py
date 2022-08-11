# ALP Fluxes, DM Fluxes
# All fluxes in cm^-2 s^-1 or cm^-2 s^-1 MeV^-1


from .constants import *
from .fmath import *
from .materials import *
from .decay import *
from .prod_xs import *
from .det_xs import *
from .photon_xs import *
from .matrix_element import *
from .cross_section_mc import *




class AxionFlux:
    # Generic superclass for constructing fluxes
    def __init__(self, axion_mass, target: Material, det_dist, det_length, det_area, n_samples=1000):
        self.ma = axion_mass
        self.target_z = target.z[0]  # TODO: take z array for compound mats
        self.target_a = target.z[0] + target.n[0]
        self.target_density = target.density
        self.det_dist = det_dist  # meters
        self.det_length = det_length  # meters
        self.det_area = det_area  # square meters
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        self.n_samples = n_samples
    
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
        surv_prob = np.array([np.exp(-self.det_dist / METER_BY_MEV / v_a[i] / tau[i]) \
                     for i in range(len(v_a))])
        decay_prob = np.array([(1 - np.exp(-self.det_length / METER_BY_MEV / v_a[i] / tau[i])) \
                      for i in range(len(v_a))])

        self.decay_axion_weight = np.asarray(rescale_factor * wgt * surv_prob * decay_prob, dtype=np.float32)
        self.scatter_axion_weight = np.asarray(rescale_factor * wgt * surv_prob, dtype=np.float32)
    
    def propagate_iso_vol_int(self, geom: DetectorGeometry, decay_width, rescale_factor=1.0):
        e_a = np.array(self.axion_energy)
        wgt = np.array(self.axion_flux)

        # Get axion Lorentz transformations and kinematics
        p_a = sqrt(e_a**2 - self.ma**2)
        v_a = p_a / e_a
        boost = e_a / self.ma
        tau = boost / decay_width if decay_width > 0.0 else np.inf * np.ones_like(boost)

        surv_prob = np.array([np.exp(-self.det_dist / METER_BY_MEV / v_a[i] / tau[i]) \
                     for i in range(len(v_a))])

        vol_integrals = []
        for i in range(len(e_a)):
            # Integrate over the detector volume with weight function P_decay / (4*pi*l^2)
            # where l is the distance to the source from the volume element
            f = lambda x : (1 / METER_BY_MEV / v_a[i] / tau[i]) \
                * np.exp(-x / METER_BY_MEV / v_a[i] / tau[i]) / (4*pi*x**2)
            vol_integrals.append(geom.integrate(f_int=f))
        
        self.decay_axion_weight = np.asarray(rescale_factor * wgt * np.array(vol_integrals), dtype=np.float32)
        self.scatter_axion_weight = np.asarray(rescale_factor * wgt * surv_prob, dtype=np.float32)




class FluxPrimakoff(AxionFlux):
    # Generator for ALP flux from 2D photon spectrum (E_\gamma, \theta_\gamma)
    def __init__(self, axion_mass, target_z, det_z, det_dist, det_length, det_area, n_samples=1000):
        super().__init__(axion_mass, target_z, det_z, det_dist, det_length, det_area, n_samples)
    
    def decay_width(self):
        pass

    def simulate_single(self):
        pass

    def simulate(self):
        pass




class FluxPrimakoffIsotropic(AxionFlux):
    """
    Generator for Primakoff-produced axion flux
    Takes in a flux of photons
    """
    def __init__(self, photon_flux=[1,1], target=Material("W"), det_dist=4.0, det_length=0.2,
                    det_area=0.04, axion_mass=0.1, axion_coupling=1e-3, n_samples=1000):
        super().__init__(axion_mass, target, det_dist, det_length, det_area)
        self.photon_flux = photon_flux
        self.gagamma = axion_coupling
        self.n_samples = n_samples
        self.target_photon_xs = AbsCrossSection(target)

    def decay_width(self, gagamma, ma):
        return W_gg(gagamma, ma)
    
    def photon_flux_dN_dE(self, energy):
        return np.interp(energy, self.photon_flux[:,0], self.photon_flux[:,1], left=0.0, right=0.0)

    def simulate_single(self, photon):
        gamma_energy = photon[0]
        gamma_wgt = photon[1]
        if gamma_energy < self.ma:
            return

        xs = primakoff_sigma(gamma_energy, self.gagamma, self.ma, self.target_z)
        br = xs / self.target_photon_xs.sigma_mev(gamma_energy)
        self.axion_energy.append(gamma_energy)
        self.axion_flux.append(gamma_wgt * br)

    def simulate(self):
        self.axion_energy = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        for i, el in enumerate(self.photon_flux):
            self.simulate_single(el)
    
    def propagate(self, new_coupling=None, is_isotropic=True):
        if new_coupling is not None:
            rescale=power(new_coupling/self.gagamma, 2)
            super().propagate(W_gg(new_coupling, self.ma), rescale)
        else:
            super().propagate(W_gg(self.gagamma, self.ma))
        if is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept
    
    def propagate_iso_vol_int(self, geom: DetectorGeometry, new_coupling=None):
        if new_coupling is not None:
            rescale=power(new_coupling/self.gagamma, 2)
            super().propagate_iso_vol_int(geom, W_gg(new_coupling, self.ma), rescale)
        else:
            super().propagate_iso_vol_int(geom, W_gg(self.gagamma, self.ma))




class FluxCompton(AxionFlux):
    # Generator for ALP flux from 2D photon spectrum (E_\gamma, \theta_\gamma)
    def __init__(self, axion_mass, target: Material, det_dist, det_length, det_area, n_samples=1000):
        super().__init__(axion_mass, target, det_dist, det_length, det_area, n_samples)

    def simulate_single(self):
        pass

    def simulate(self):
        pass




class FluxComptonIsotropic(AxionFlux):
    """
    Generator for axion flux from compton-like scattering
    Takes in a flux of photons
    """
    def __init__(self, photon_flux=[1,1], target=Material("W"), det_dist=4.,
                    det_length=0.2, det_area=0.04, axion_mass=0.1, axion_coupling=1e-3, n_samples=100,
                    is_isotropic=True):
        super().__init__(axion_mass, target, det_dist, det_length, det_area)
        self.photon_flux = photon_flux
        self.ge = axion_coupling
        self.n_samples = n_samples
        self.target_photon_xs = AbsCrossSection(target)
        self.is_isotropic = is_isotropic
    
    def decay_width(self, ge, ma):
        return W_ee(ge, ma)

    def simulate_single(self, photon):
        gamma_energy = photon[0]
        gamma_wgt = photon[1]

        s = 2 * M_E * gamma_energy + M_E ** 2
        if s < (M_E + self.ma)**2:
            return

        ea_rnd = np.random.uniform(self.ma, gamma_energy, self.n_samples)
        mc_xs = (gamma_energy - self.ma) * compton_dsigma_dea(ea_rnd, gamma_energy, self.ge, self.ma, self.target_z) / self.n_samples
        diff_br = mc_xs / self.target_photon_xs.sigma_mev(gamma_energy)

        for i in range(self.n_samples):
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
        
        if self.is_isotropic:
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
    def __init__(self, electron_flux=[1.,0.], positron_flux=[1.,0.], target=Material("W"),
                    target_density=19.3, target_radiation_length=6.76, target_length=10.0, det_dist=4., det_length=0.2,
                    det_area=0.04, axion_mass=0.1, axion_coupling=1e-3, n_samples=100, is_isotropic=True):
        super().__init__(axion_mass, target, det_dist, det_length, det_area)
        # TODO: Replace A = 2*Z with real numbers of nucleons
        self.electron_flux = electron_flux
        self.positron_flux = positron_flux
        self.ge = axion_coupling
        self.target_density = target_density  # g/cm3
        self.target_radius = target_length  # cm
        self.ntargets_by_area = target_length * target_density * AVOGADRO / (2*target.z[0])  # N_T / cm^2
        self.ntarget_area_density = target_radiation_length * AVOGADRO / (2*target.z[0])
        self.n_samples = n_samples
        self.is_isotropic = is_isotropic
    
    def decay_width(self):
        return W_ee(self.ge, self.ma)
    
    def electron_flux_dN_dE(self, energy):
        return np.interp(energy, self.electron_flux[:,0], self.electron_flux[:,1], left=0.0, right=0.0)
    
    def positron_flux_dN_dE(self, energy):
        return np.interp(energy, self.positron_flux[:,0], self.positron_flux[:,1], left=0.0, right=0.0)
    
    def electron_flux_attenuated(self, t, E0, E1):
        return (self.electron_flux_dN_dE(E0) + self.positron_flux_dN_dE(E0)) * track_length_prob(E0, E1, t)

    def simulate_single(self, electron):
        el_energy = electron[0]
        el_wgt = electron[1]

        ea_max = el_energy * (1 - power(self.ma/el_energy, 2))
        if ea_max < self.ma:
            return

        ea_rnd = np.random.uniform(self.ma, ea_max, self.n_samples)
        mc_vol = (ea_max - self.ma)/self.n_samples
        diff_br = (self.ntarget_area_density * HBARC**2) * mc_vol * brem_dsigma_dea(ea_rnd, el_energy, self.ge, self.ma, self.target_z)

        for i in range(self.n_samples):
            self.axion_energy.append(ea_rnd[i])
            self.axion_flux.append(el_wgt * diff_br[i])

    def simulate(self):
        self.axion_energy = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        """
        ep_min = max((self.ma**2 - M_E**2)/(2*M_E), M_E)
        for i, el in enumerate(self.electron_flux):
            t_depth = np.random.uniform(0.0, 5.0)
            new_energy = np.random.uniform(ep_min, el[0])
            flux_weight = self.electron_flux_attenuated(t_depth, el[0], new_energy) * 5.0 * (el[0] - ep_min)
            self.simulate_single([new_energy, flux_weight])
        """
        for i, el in enumerate(self.electron_flux):
            self.simulate_single(el)
    
    def propagate(self, new_coupling=None):
        if new_coupling is not None:
            super().propagate(W_ee(new_coupling, self.ma), rescale_factor=power(new_coupling/self.ge, 2))
        else:
            super().propagate(W_ee(self.ge, self.ma))
        
        if self.is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept




class FluxResonanceIsotropic(AxionFlux):
    """
    Generator for e+ e- resonant ALP production flux
    Takes in a flux of positrons
    """
    def __init__(self, positron_flux=[1.,0.], target=Material("W"), target_length=10.0,
                 target_radiation_length=6.76, det_dist=4., det_length=0.2, det_area=0.04,
                 axion_mass=0.1, axion_coupling=1e-3, n_samples=100, is_isotropic=True):
        # TODO: make flux take in a Detector class and a Target class (possibly Material class?)
        # Replace A = 2*Z with real numbers of nucleons
        super().__init__(axion_mass, target, det_dist, det_length, det_area, n_samples)
        self.positron_flux = positron_flux  # differential positron energy flux dR / dE+ / s
        self.positron_flux_bin_widths = positron_flux[1:,0] - positron_flux[:-1,0]
        self.ge = axion_coupling
        self.target_radius = target_length  # cm
        self.ntarget_area_density = target_radiation_length * AVOGADRO / (2*target.z[0])  # N_T / cm^2
        self.is_isotropic = is_isotropic
    
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

        resonant_energy = -M_E + self.ma**2 / (2 * M_E)
        if resonant_energy + M_E < self.ma:
            return
        
        if resonant_energy < M_E:
            return
        
        if resonant_energy > max(self.positron_flux[:,0]):
            return
        
        e_rnd = np.random.uniform(resonant_energy, max(self.positron_flux[:,0]), self.n_samples)
        t_rnd = np.random.uniform(0.0, 5.0, self.n_samples)
        mc_vol = (5.0 - 0.0)*(max(self.positron_flux[:,0]) - resonant_energy)

        attenuated_flux = mc_vol*np.sum(self.positron_flux_attenuated(t_rnd, e_rnd, resonant_energy))/self.n_samples
        wgt = self.target_z * (self.ntarget_area_density * HBARC**2) * resonance_peak(self.ge) * attenuated_flux
        
        self.axion_energy.append(self.ma**2 / (2 * M_E))
        self.axion_flux.append(wgt)
    
    def propagate(self, new_coupling=None):
        if new_coupling is not None:
            super().propagate(W_ee(new_coupling, self.ma), rescale_factor=power(new_coupling/self.ge, 2))
        else:
            super().propagate(W_ee(self.ge, self.ma))
        
        if self.is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept




class FluxPairAnnihilationIsotropic(AxionFlux):
    """
    Generator associated production via electron-positron annihilation
    (e+ e- -> a gamma)
    Takes in a flux of positrons
    """
    def __init__(self, positron_flux=[1.,0.], target=Material("W"),
                 target_radiation_length=6.76, det_dist=4., det_length=0.2, det_area=0.04,
                 axion_mass=0.1, axion_coupling=1e-3, n_samples=100, is_isotropic=True):
        # TODO: make flux take in a Detector class and a Target class (possibly Material class?)
        # Replace A = 2*Z with real numbers of nucleons
        super().__init__(axion_mass, target, det_dist, det_length, det_area, n_samples)
        self.positron_flux = positron_flux  # differential positron energy flux dR / dE+ / s
        self.positron_flux_bin_widths = positron_flux[1:,0] - positron_flux[:-1,0]
        self.ge = axion_coupling
        self.ntarget_area_density = target_radiation_length * AVOGADRO / (2*target.z[0])
        self.is_isotropic = is_isotropic
    
    def decay_width(self):
        return W_ee(self.ge, self.ma)
    
    def simulate_single(self, positron):
        ep_lab = positron[0]
        pos_wgt = positron[1]

        if ep_lab < max((self.ma**2 - M_E**2)/(2*M_E), M_E):
            # Threshold check
            return

        # Simulate ALPs produced in the CM frame
        cm_cosines = np.random.uniform(-1, 1, self.n_samples)
        cm_wgts = (self.ntarget_area_density * HBARC**2) * associated_dsigma_dcos_CM(cm_cosines, ep_lab, self.ma, self.ge, self.target_z)

        # Boost the ALPs to the lab frame and multiply weights by jacobian for the boost
        jacobian_cm_to_lab = power(2, 1.5) * power(1 + cm_cosines, 0.5)
        ea_cm = sqrt(M_E * (ep_lab + M_E) / 2)
        paz_cm = sqrt(M_E * (ep_lab + M_E) / 2 - self.ma**2) * cm_cosines
        beta = sqrt(ep_lab**2 - M_E**2) / (M_E + ep_lab)
        gamma = power(1-beta**2, -0.5)

        # Get the lab frame energy distribution
        ea_lab = gamma*(ea_cm + beta*paz_cm)
        mc_volume = 2 / self.n_samples  # we integrated over cosThetaLab from -1 to 1

        for i in range(self.n_samples):
            self.axion_energy.append(ea_lab[i])
            self.axion_flux.append(pos_wgt * jacobian_cm_to_lab[i] * cm_wgts[i] * mc_volume)

    
    def simulate(self):
        self.axion_energy = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        for i, el in enumerate(self.positron_flux):
            self.simulate_single(el)

    def propagate(self, new_coupling=None):
        if new_coupling is not None:
            super().propagate(W_ee(new_coupling, self.ma), rescale_factor=power(new_coupling/self.ge, 2))
        else:
            super().propagate(W_ee(self.ge, self.ma))
        
        if self.is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept




class FluxNuclearIsotropic(AxionFlux):
    """
    Takes in a rate (#/s) of nuclear decays for a specified nuclear transition
    Produces the associated ALP flux from a given branching ratio
    """
    def __init__(self, transition_rates=np.array([[1.0, 0.0, 1.0, 0.5]]), target=Material("W"),
                 det_dist=4., det_length=0.2, det_area=0.04, is_isotropic=True,
                 axion_mass=0.1, gae=1.0e-5, gann0=1e-3, gann1=1e-3, n_samples=100):
        super().__init__(axion_mass, target, det_dist, det_length, det_area, n_samples)
        self.rates = transition_rates
        self.gann0 = gann0
        self.gann1 = gann1
        self.gae = gae
        self.is_isotropic = is_isotropic

    def br(self, energy, j=1, delta=0.0, beta=1.0, eta=0.5):
        mu0 = 0.88
        mu1 = 4.71
        return ((j/(j+1)) / (1 + delta**2) / pi / ALPHA) \
            * power(sqrt(energy**2 - self.ma**2)/energy, 2*j + 1) \
                * power((self.gann0 * beta + self.gann1)/((mu0-0.5)*beta + (mu1 - eta)), 2)

    def simulate(self, j=1, delta=0.0):
        self.axion_energy = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        for i in range(self.rates.shape[0]):
            if self.rates[i,0] > self.ma:
                self.axion_energy.append(self.rates[i,0])
                self.axion_flux.append(self.rates[i,1] * self.br(self.rates[i,0], j, delta, self.rates[i,2], self.rates[i,3]))

    def propagate(self, new_coupling=None):
        if new_coupling is not None:
            rescale=power(new_coupling/self.gae, 2)
            super().propagate(W_ee(new_coupling, self.ma), rescale)
        else:
            super().propagate(W_ee(self.gae, self.ma))
        
        if self.is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept
    
    def propagate_iso_vol_int(self, geom: DetectorGeometry, new_coupling=None):
        if new_coupling is not None:
            rescale=power(new_coupling/self.gae, 2)
            super().propagate_iso_vol_int(geom, W_ee(new_coupling, self.ma), rescale)
        else:
            super().propagate_iso_vol_int(geom, W_ee(self.gae, self.ma))




class FluxChargedMeson3BodyDecay(AxionFlux):
    def __init__(self, meson_flux, boson_mass=0.1, coupling=1.0, n_samples=50, meson_type="pion",
                 interaction_model="scalar_ib1", energy_cut=140.0, det_dist=541, det_length=12,
                 det_area=36*pi, c0=-0.95, lepton_masses=[M_E, M_MU]):
        super().__init__(boson_mass, Material("Be"), det_dist, det_length, det_area, n_samples)
        self.meson_flux = meson_flux
        param_dict = {
            "pion": [M_PI, V_UD, F_PI, PION_WIDTH],
            "kaon": [M_K, V_US, F_K, KAON_WIDTH]
        }
        decay_params = param_dict[meson_type]
        self.mm = decay_params[0]
        self.ckm = decay_params[1]
        self.fM = decay_params[2]
        self.total_width = decay_params[3]
        self.lepton_masses = lepton_masses
        self.gmu = coupling
        self.dump_dist = 50
        self.det_sa = cos(arctan(self.det_length/(self.det_dist-self.dump_dist)/2))
        self.solid_angles = []
        self.energy_cut = energy_cut
        self.cosines = []
        self.decay_pos = []
        self.c0 = c0  # contact model parameter, 0 by default
        self.m2_e = M2Meson3BodyDecay(boson_mass, meson_type, M_E, interaction_model)
        self.m2_mu = M2Meson3BodyDecay(boson_mass, meson_type, M_MU, interaction_model)

    def set_ma(self, ma):
        self.ma = ma
        self.m2_mu.m3 = self.ma
        self.m2_e.m3 = self.ma

    def dGammadEa(self, Ea, ml=M_MU):
        m212 = self.mm**2 + self.ma**2 - 2*self.mm*Ea
        e2star = (m212 - ml**2)/(2*sqrt(m212))
        e3star = (self.mm**2 - m212 - self.ma**2)/(2*sqrt(m212))

        if self.ma > e3star:
            return 0.0
        
        m223Max = (e2star + e3star)**2 - (sqrt(e2star**2) - sqrt(e3star**2 - self.ma**2))**2
        m223Min = (e2star + e3star)**2 - (sqrt(e2star**2) + sqrt(e3star**2 - self.ma**2))**2

        if ml == M_MU:
            def MatrixElement2(m223):
                return self.m2_mu(m212, m223, c0=self.c0, coupling=self.gmu)
            
            return (2*self.mm)/(32*power(2*pi*self.mm, 3))*quad(MatrixElement2, m223Min, m223Max)[0]
        elif ml == M_E:
            def MatrixElement2(m223):
                    return self.m2_e(m212, m223, c0=self.c0, coupling=self.gmu)
                
            return (2*self.mm)/(32*power(2*pi*self.mm, 3))*quad(MatrixElement2, m223Min, m223Max)[0]
        else:
            return 0.0
    
    def total_br(self):
        ea_max_mu = (self.mm**2 + self.ma**2 - M_MU**2)/(2*self.mm)
        ea_max_e = (self.mm**2 + self.ma**2 - M_E**2)/(2*self.mm)
        return (quad(self.dGammadEa, self.ma, ea_max_e, args=(M_E,))[0] \
            + quad(self.dGammadEa, self.ma, ea_max_mu, args=(M_MU,))[0]) / self.total_width
    
    def simulate_single(self, meson_p, pion_wgt, cut_on_solid_angle=True, solid_angle_cosine=0.0, ml=M_E):
        ea_min = self.ma
        ea_max = (self.mm**2 + self.ma**2 - ml**2)/(2*self.mm)

        # Boost to lab frame
        beta = meson_p / sqrt(meson_p**2 + self.mm**2)
        boost = power(1-beta**2, -0.5)

        min_cm_cos = cos(min(boost * arccos(solid_angle_cosine), pi))
        # Draw random variate energies and angles in the pion rest frame
        energies = np.random.uniform(ea_min, ea_max, self.n_samples)
        momenta = sqrt(energies**2 - self.ma**2)
        cosines = np.random.uniform(min_cm_cos, 1, self.n_samples)
        pz = momenta*cosines

        e_lab = boost*(energies + beta*pz)
        pz_lab = boost*(pz + beta*energies)
        cos_theta_lab = pz_lab / sqrt(e_lab**2 - self.ma**2)

        # Jacobian for transforming d2Gamma/(dEa * dOmega) to lab frame:
        #jacobian = sqrt(e_lab**2 - self.ma**2) / momenta
        # Monte Carlo volume, making sure to use the lab frame energy range
        mc_vol = (ea_max - ea_min)*(1-min_cm_cos)
        #mc_vol_lab = boost*(ea_max - sqrt(ea_max**2 - self.ma**2)*beta) - boost*(ea_min + sqrt(ea_min**2 - self.ma**2)*beta)

        # Draw weights from the PDF
        weights = np.array([pion_wgt*mc_vol*self.dGammadEa(ea, ml)/self.total_width/self.n_samples \
            for ea in energies])
        #weights = np.array([pion_wgt*mc_vol_lab*self.dGammadEa(ea)/self.gamma_sm()/self.n_samples \
        #    for ea in energies])*jacobian

        for i in range(self.n_samples):
            solid_angle_acceptance = heaviside(cos_theta_lab[i] - solid_angle_cosine, 0.0)
            if solid_angle_acceptance == 0.0 and cut_on_solid_angle:
                continue
            self.axion_energy.append(e_lab[i])
            self.cosines.append(cos_theta_lab[i])
            self.axion_flux.append(weights[i]*heaviside(e_lab[i]-self.energy_cut,1.0))
            self.solid_angles.append(solid_angle_cosine)
    
    def simulate(self, cut_on_solid_angle=True):
        self.axion_energy = []
        self.cosines = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []
        self.decay_pos = []
        self.solid_angles = []

        for i, p in enumerate(self.meson_flux):
            # Simulate decay positions between target and dump
            # The quantile is truncated at the dump position via umax
            decay_l = METER_BY_MEV * p[0] / self.total_width / self.mm
            umax = exp(-2*self.dump_dist/decay_l) * power(exp(self.dump_dist/decay_l) - 1, 2) \
                if decay_l > 1.0 else 1.0
            try:
                u = np.random.uniform(0.0, min(umax, 1.0))
            except:
                print("umax = ", umax, " decay l = ", decay_l, p[0])
            x = decay_quantile(u, p[0], self.mm, self.total_width)
            
            # Append decay positions and solid angle cosines for the geometric acceptance of each meson decay
            self.decay_pos.append(x)
            solid_angle_cosine = cos(arctan(self.det_length/(self.det_dist-x)/2))

            # Simulate decays for each charged meson
            for ml in self.lepton_masses:
                if self.ma > self.mm - ml:
                    continue
                self.simulate_single(p[0], p[2], cut_on_solid_angle, solid_angle_cosine, ml)
        

    def propagate(self, gagamma=None):  # propagate to detector
        wgt = np.array(self.axion_flux)
        # Do not decay
        self.decay_axion_weight = np.asarray(wgt*0.0, dtype=np.float64)
        self.scatter_axion_weight = np.asarray(wgt, dtype=np.float64)




class FluxChargedMeson3BodyIsotropic(AxionFlux):
    def __init__(self, meson_flux=[[0.0, 0.0259]], boson_mass=0.1, coupling=1.0, meson_type="pion",
                 interaction_model="scalar_ib1", det_dist=20, det_length=2, det_area=2,
                 target=Material("W"), n_samples=50, c0=-0.97, lepton_masses=[M_E, M_MU]):
        super().__init__(boson_mass, target, det_dist, det_length, det_area, n_samples)
        self.meson_flux = meson_flux
        param_dict = {
            "pion": [M_PI, V_UD, F_PI, PION_WIDTH],
            "kaon": [M_K, V_US, F_K, KAON_WIDTH]
        }
        self.lepton_masses = lepton_masses
        decay_params = param_dict[meson_type]
        self.mm = decay_params[0]
        self.ckm = decay_params[1]
        self.fM = decay_params[2]
        self.total_width = decay_params[3]
        self.gmu = coupling
        self.n_samples = n_samples
        self.c0 = c0  # contact model parameter, 0 by default
        self.m2_e = M2Meson3BodyDecay(boson_mass, meson_type, M_E, interaction_model)
        self.m2_mu = M2Meson3BodyDecay(boson_mass, meson_type, M_MU, interaction_model)
    
    def lifetime(self, gagamma):
        return 1/W_gg(gagamma, self.ma)
    
    def set_ma(self, ma):
        self.ma = ma
        self.m2_mu.m3 = self.ma
        self.m2_e.m3 = self.ma

    def dGammadEa(self, Ea, ml=M_MU):
        m212 = self.mm**2 + self.ma**2 - 2*self.mm*Ea
        e2star = (m212 - ml**2)/(2*sqrt(m212))
        e3star = (self.mm**2 - m212 - self.ma**2)/(2*sqrt(m212))

        if self.ma > e3star:
            return 0.0

        m223Max = (e2star + e3star)**2 - (sqrt(e2star**2) - sqrt(e3star**2 - self.ma**2))**2
        m223Min = (e2star + e3star)**2 - (sqrt(e2star**2) + sqrt(e3star**2 - self.ma**2))**2

        if ml == M_MU:
            def MatrixElement2(m223):
                return self.m2_mu(m212, m223, c0=self.c0, coupling=self.gmu)
            
            return (2*self.mm)/(32*power(2*pi*self.mm, 3))*quad(MatrixElement2, m223Min, m223Max)[0]
        elif ml == M_E:
            def MatrixElement2(m223):
                    return self.m2_e(m212, m223, c0=self.c0, coupling=self.gmu)
                
            return (2*self.mm)/(32*power(2*pi*self.mm, 3))*quad(MatrixElement2, m223Min, m223Max)[0]
        else:
            return 0.0

    def total_br(self):
        return np.sum([quad(self.dGammadEa, self.ma,  (self.mm**2 + self.ma**2 - ml**2)/(2*self.mm), args=(ml,))[0] for ml in self.lepton_masses]) / self.total_width
    
    def diff_br(self):
        ea_min = self.ma
        ea_max = (self.mm**2 + self.ma**2 - self.m_lepton**2)/(2*self.mm)
        mc_vol = ea_max - ea_min
        energies = np.random.uniform(ea_min, ea_max, self.n_samples)
        weights = np.array([mc_vol*self.dGammadEa(ea)/self.total_width/self.n_samples \
            for ea in energies])
        return energies, weights
    
    def simulate_single(self, meson_p, pion_wgt, ml=M_E):
        ea_min = self.ma
        ea_max = (self.mm**2 + self.ma**2 - ml**2)/(2*self.mm)

        # Draw random variate energies and angles in the pion rest frame
        energies = np.random.uniform(ea_min, ea_max, self.n_samples)
        momenta = sqrt(energies**2 - self.ma**2)
        cosines = np.random.uniform(-1, 1, self.n_samples)
        pz = momenta*cosines

        # Boost to lab frame
        beta = meson_p / sqrt(meson_p**2 + self.mm**2)
        boost = power(1-beta**2, -0.5)
        e_lab = boost*(energies + beta*pz)
        #pz_lab = boost*(pz + beta*energies)

        # Jacobian for transforming d2Gamma/(dEa * dOmega) to lab frame:
        #jacobian = sqrt(e_lab**2 - self.ma**2) / momenta
        # Monte Carlo volume, making sure to use the lab frame energy range

        mc_vol = ea_max - ea_min
        weights = np.array([pion_wgt*mc_vol*self.dGammadEa(ea, ml)/self.total_width/self.n_samples \
            for ea in energies])

        for i in range(self.n_samples):
            self.axion_energy.append(e_lab[i])
            self.axion_flux.append(weights[i])
    
    def simulate(self):
        self.axion_energy = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []

        for i, p in enumerate(self.meson_flux):
            for ml in self.lepton_masses:
                if self.ma > self.mm - ml:
                    continue
                # Simulate decays for each charged meson
                self.simulate_single(p[0], p[1], ml)

    def propagate(self):  # propagate to detector
        wgt = np.array(self.axion_flux)
        self.decay_axion_weight = np.asarray(wgt*0.0, dtype=np.float64)
        self.scatter_axion_weight = np.asarray(self.det_area / (4*pi*self.det_dist**2) * wgt, dtype=np.float64)





class FluxPi0Isotropic(AxionFlux):
    def __init__(self, pi0_rate=0.0259, boson_mass=0.1, coupling=1.0,
                 det_dist=20.0, det_area=2.0, det_length=1.0,
                 target=Material("W"), n_samples=1000):
        super().__init__(boson_mass, target, det_dist, det_length, det_area)
        self.meson_rate = pi0_rate
        self.n_samples = n_samples
        self.g = coupling
        self.boson_mass = boson_mass
        self.mm = M_PI0
    
    def br(self):
        return 2 * (self.g)**2 * (1 - power(self.ma / M_PI0, 2))**3 / sqrt(4*pi*ALPHA)
    
    def simulate_flux(self, pi0_flux, energy_cut=0.0, angle_cut=np.pi):
        # pi0_flux = momenta array, normalized to pi0 rate

        self.axion_energy = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []

        if self.ma > self.mm:
            # Kinematically forbidden
            return

        # Simulate decays for each charged meson
        p_cm = (M_PI0**2 - self.ma**2)/(2*M_PI0)
        e1_cm = sqrt(p_cm**2 + self.ma**2)
        cos_rnd = np.random.uniform(-1, 1, pi0_flux.shape[0])

        for i, p in enumerate(pi0_flux):
            beta = p / sqrt(p**2 + self.mm**2)
            boost = power(1-beta**2, -0.5)
            e_lab = boost*(e1_cm + beta*p_cm*cos_rnd[i])
            pz_lab = boost*(p_cm*cos_rnd[i] + beta*e1_cm)
            angle_lab = arccos(pz_lab / sqrt(e_lab**2 - self.ma**2))
            if e_lab < energy_cut:
                continue
            if angle_lab > angle_cut:
                continue
            self.axion_energy.append(e_lab)
            self.axion_flux.append(self.meson_rate*self.br()/pi0_flux.shape[0])
    
    def simulate(self):
        self.axion_energy = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []

        if self.ma > self.mm:
            # Kinematically forbidden
            return

        # Simulate decays for each charged meson
        p_cm = (M_PI0**2 - self.ma**2)/(2*M_PI0)
        e1_cm = sqrt(p_cm**2 + self.ma**2)

        self.axion_energy.extend(e1_cm*np.ones(self.n_samples))
        self.axion_flux.extend(self.meson_rate * self.br() * np.ones(self.n_samples)/np.sum(self.n_samples))
        

    def propagate(self, is_isotropic=True):
        if is_isotropic:
            wgt = np.array(self.axion_flux)
            self.decay_axion_weight = np.asarray(wgt*0.0, dtype=np.float64)
            self.scatter_axion_weight = np.asarray(self.det_area / (4*pi*self.det_dist**2) * wgt, dtype=np.float64)
        else:
            wgt = np.array(self.axion_flux)
            self.decay_axion_weight = np.asarray(wgt*0.0, dtype=np.float64)
            self.scatter_axion_weight = np.asarray(wgt, dtype=np.float64)





class FluxPairAnnihilationGamma(AxionFlux):
    """
    Generator associated production via electron-positron annihilation into gamma + ALP
    (e+ e- -> a gamma) via virtual photon
    Takes in a flux of positrons
    """
    def __init__(self, positron_flux=[1.,0.], target=Material("W"),
                 target_radiation_length=6.76, det_dist=4., det_length=0.2, det_area=0.04,
                 axion_mass=0.1, axion_coupling=1e-3, n_samples=100, is_isotropic=True):
        super().__init__(axion_mass, target, det_dist, det_length, det_area, n_samples)
        self.positron_flux = positron_flux  # differential positron energy flux dR / dE+ / s
        self.positron_flux_bin_widths = positron_flux[1:,0] - positron_flux[:-1,0]
        self.positron_flux_bin_centers = (positron_flux[1:,0] + positron_flux[:-1,0])/2
        self.gagamma = axion_coupling
        self.ntarget_area_density = target_radiation_length * AVOGADRO / (2*target.z[0])  # todo: change 2*Z
        self.is_isotropic = is_isotropic
        self.positron_flux_smeared_wgts = []
        self.positron_flux_smeared_energies = []
    
    def track_length_prob(self, Ei, Ef, t):
        b = 4/3
        return heaviside(Ei-Ef, 0.0) * abs(power(log(Ei/Ef), b*t - 1) / (Ei * gamma(b*t)))
    
    def positron_flux_dN_dE(self, energy):
        return np.interp(energy, self.positron_flux[:,0], self.positron_flux[:,1], left=0.0, right=0.0)
    
    def positron_flux_attenuated(self, t, energy_pos, energy_res):
        return self.positron_flux_dN_dE(energy_pos) * self.track_length_prob(energy_pos, energy_res, t)
    
    def simulate_positron_flux(self):
        self.positron_flux_smeared_energies = []
        self.positron_flux_smeared_wgts = []
        ep_min = max((self.ma**2 - M_E**2)/(2*M_E), M_E)
        print("ep min = ", ep_min)
        for i in range(self.positron_flux_bin_centers.shape[0]):
            ep_lab = self.positron_flux_bin_centers[i]
            if ep_lab < ep_min:
                continue  # Threshold check

            # Simulate ALPs produced in the CM frame
            t_depths = np.random.uniform(0.0, 5.0, self.n_samples)
            smeared_energies = np.random.uniform(ep_min, ep_lab, self.n_samples)

            flux_wgts = self.positron_flux_attenuated(t_depths, ep_lab, smeared_energies)
            self.positron_flux_smeared_wgts.extend(self.positron_flux_bin_widths[i] * 5.0 * (ep_lab-ep_min) * flux_wgts / self.n_samples)
            self.positron_flux_smeared_energies.extend(smeared_energies)
    
    def simulate(self):
        self.axion_energy = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        self.simulate_positron_flux()

        for i in range(len(self.positron_flux_smeared_wgts)):
            ep = self.positron_flux_smeared_energies[i]
            pos_flux = self.positron_flux_smeared_wgts[i]
            beta = sqrt(ep**2 - M_E**2) / (M_E + ep)
            gamma = power(1-beta**2, -0.5)
            s = 2 * M_E * (M_E + ep)
            ps_cm = sqrt((s - self.ma**2)**2 / (4*s))
            es_cm = sqrt(ps_cm**2 + self.ma**2)

            es_min = gamma*(es_cm - ps_cm*beta)
            es_max = gamma*(es_cm + ps_cm*beta)
            alp_energies = np.random.uniform(es_min, es_max, 1)
            mc_volume_diffxs = 2 * gamma * beta * ps_cm
            
            cm_wgts = (self.ntarget_area_density * HBARC**2) \
                * epem_to_alp_photon_dsigma_de(alp_energies, ep, self.gagamma, self.ma, self.target_z)

            
            self.axion_energy.extend(alp_energies)
            self.axion_flux.extend(cm_wgts * pos_flux * mc_volume_diffxs)

    def propagate(self, new_coupling=None):
        if new_coupling is not None:
            super().propagate(W_gg(new_coupling, self.ma), rescale_factor=power(new_coupling/self.gagamma, 2))
        else:
            super().propagate(W_gg(self.gagamma, self.ma))
        
        if self.is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept
