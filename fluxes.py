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

import multiprocessing as multi



class AxionFlux:
    # Generic superclass for constructing fluxes
    def __init__(self, axion_mass, target: Material, det_dist, det_length, det_area, n_samples=1000,
                 off_axis_angle=0.0, timing_window_ns=np.inf):
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
        self.off_axis_angle = off_axis_angle
        self.timing_window = 1e-9 * timing_window_ns  # acceptable time delay from speed of light particles

    def det_sa(self):
        return arctan(sqrt(self.det_area / pi) / self.det_dist)

    def propagate(self, decay_width, rescale_factor=1.0, time_of_flight_cut=None):
        e_a = np.array(self.axion_energy)
        wgt = np.array(self.axion_flux)

        # Get axion Lorentz transformations and kinematics
        p_a = sqrt(e_a**2 - self.ma**2)
        v_a = p_a / e_a
        boost = e_a / self.ma
        tau = boost / decay_width if decay_width > 0.0 else np.inf * np.ones_like(boost)

        # Calculate time of flight
        if time_of_flight_cut is not None:
            tof = self.det_dist / (v_a * 1e-2 * C_LIGHT)
            delta_tof = abs((self.det_dist / (1e-2 * C_LIGHT)) - tof)
            in_timing_window_wgt = delta_tof < self.timing_window
            wgt *= in_timing_window_wgt

        # Get decay and survival probabilities
        surv_prob = np.exp(-self.det_dist / METER_BY_MEV / v_a / tau)
        decay_prob = (1 - np.exp(-self.det_length / METER_BY_MEV / v_a / tau))

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
    Takes in a flux of photons: this should be a list of lists or a numpy array of shape (2,n)
    Each element in the list should be a list of length 2 carrying [photon_energy, photon_weight]
    where the weight is in units of counts/second inside the target
    One may also take in photon flux from a text file of columnated data.
    """
    def __init__(self, photon_flux=[[1,1]], target=Material("W"), det_dist=4.0, det_length=0.2,
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
    def __init__(self, photon_flux=[[1,1]], target=Material("W"), det_dist=4.,
                    det_length=0.2, det_area=0.04, axion_mass=0.1, axion_coupling=1e-3, n_samples=100,
                    is_isotropic=True, loop_decay=False):
        super().__init__(axion_mass, target, det_dist, det_length, det_area)
        self.photon_flux = photon_flux
        self.ge = axion_coupling
        self.n_samples = n_samples
        self.target_photon_xs = AbsCrossSection(target)
        self.is_isotropic = is_isotropic
        self.loop_decay = loop_decay

    def decay_width(self, ge, ma):
        if self.loop_decay:
            return W_ee(ge, ma) + W_gg_loop(ge, ma, M_E)
        else:
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
            super().propagate(self.decay_width(new_coupling, self.ma),
                rescale_factor=power(new_coupling/self.ge, 2))
        else:
            super().propagate(self.decay_width(self.ge, self.ma))

        if self.is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept
    
    def propagate_iso_vol_int(self, geom: DetectorGeometry, new_coupling=None):
        if new_coupling is not None:
            rescale=power(new_coupling/self.ge, 2)
            super().propagate_iso_vol_int(geom, W_ee(new_coupling, self.ma), rescale)
        else:
            super().propagate_iso_vol_int(geom, W_ee(self.ge, self.ma))




def track_length_prob(Ei, Ef, t):
    b = 4/3
    return heaviside(Ei-Ef, 1.0) * abs(power(log(Ei/Ef), b*t - 1) / (Ei * gamma(b*t)))




class FluxBremIsotropic(AxionFlux):
    """
    Generator for axion-bremsstrahlung flux
    Takes in a flux of el
    """
    def __init__(self, electron_flux=[[1.,0.]], positron_flux=[1.,0.], target=Material("W"),
                    target_length=10.0, det_dist=4., det_length=0.2, det_area=0.04,
                    axion_mass=0.1, axion_coupling=1e-3, n_samples=100,
                    is_isotropic=True, loop_decay=False, boson_type="pseudoscalar",
                    max_track_length=5.0, is_monoenergetic=False, **kwargs):
        super().__init__(axion_mass, target, det_dist, det_length, det_area)
        # TODO: Replace A = 2*Z with real numbers of nucleons
        self.electron_flux = electron_flux
        self.positron_flux = positron_flux
        self.boson_type = boson_type
        self.ge = axion_coupling
        self.target_density = target.density  # g/cm3
        self.target_radius = target_length  # cm
        self.ntargets_by_area = target_length * target.density * AVOGADRO / (2*target.z[0])  # N_T / cm^2
        self.ntarget_area_density = target.rad_length * AVOGADRO / (2*target.z[0])
        self.n_samples = n_samples
        self.is_isotropic = is_isotropic
        self.loop_decay = loop_decay
        self.max_t = max_track_length
        self.is_monoenergetic = is_monoenergetic

    def decay_width(self, ge, ma):
        if self.loop_decay:
            return W_ee(ge, ma) + W_gg_loop(ge, ma, M_E)
        else:
            return W_ee(ge, ma)

    def electron_flux_dN_dE(self, energy):
        return np.interp(energy, self.electron_flux[:,0], self.electron_flux[:,1], left=0.0, right=0.0)

    def positron_flux_dN_dE(self, energy):
        return np.interp(energy, self.positron_flux[:,0], self.positron_flux[:,1], left=0.0, right=0.0)

    def electron_positron_flux_attenuated(self, t, E0, E1):
        if self.is_monoenergetic:
            el_flux_att = np.array([self.electron_flux[i,1] * track_length_prob(self.electron_flux[i,0], E1, t) \
                            for i in range(self.electron_flux.shape[0])])
            return el_flux_att
        return (self.electron_flux_dN_dE(E0) + self.positron_flux_dN_dE(E0)) * track_length_prob(E0, E1, t)

    def simulate_single(self, electron):
        el_energy = electron[0]
        el_wgt = electron[1]

        ea_max = el_energy * (1 - power(self.ma/el_energy, 2))
        if ea_max <= self.ma:
            return
        
        ea_rnd = power(10, np.random.uniform(np.log10(self.ma), np.log10(ea_max), self.n_samples))

        if self.boson_type == "pseudoscalar":
            mc_vol = np.log(10) * ea_rnd *  (np.log10(ea_max) - np.log10(self.ma)) / self.n_samples
            diff_br = (self.ntarget_area_density * HBARC**2) * mc_vol * brem_dsigma_dea(ea_rnd, el_energy, self.ge, self.ma, self.target_z)
        elif self.boson_type == "vector":
            x_rnd = ea_rnd / el_energy
            mc_vol = np.log(10) * x_rnd *  (np.log10(ea_max/el_energy) - np.log10(self.ma/el_energy)) / self.n_samples
            diff_br = (self.ntarget_area_density * HBARC**2) * mc_vol * brem_dsigma_dx_vector(x_rnd, self.ge, self.ma, self.target_z)

        self.axion_energy.extend(ea_rnd)
        self.axion_flux.extend(el_wgt * diff_br)

    def simulate(self, use_track_length=True):
        self.axion_energy = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        if use_track_length:
            ep_min = max(self.ma, M_E) #max((self.ma**2 - M_E**2)/(2*M_E), M_E)
            for i, el in enumerate(self.electron_flux):
                if el[0] < ep_min:
                    continue
                t_depth = 10**np.random.uniform(-3, np.log10(self.max_t), 5)
                new_energy = np.random.uniform(ep_min, el[0], 5)
                for i in range(5):
                    flux_weight = self.electron_positron_flux_attenuated(t_depth[i], el[0], new_energy[i]) \
                        * np.log(10) * t_depth[i] * (np.log10(self.max_t*3)) * (el[0] - ep_min) / 5
                    self.simulate_single([new_energy[i], flux_weight])
            
        else:
            for i, el in enumerate(self.electron_flux):
                self.simulate_single(el)

    def propagate(self, new_coupling=None):
        if new_coupling is not None:
            super().propagate(self.decay_width(new_coupling, self.ma),
                rescale_factor=power(new_coupling/self.ge, 2))
        else:
            super().propagate(self.decay_width(self.ge, self.ma))

        if self.is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept
    
    def propagate_iso_vol_int(self, geom: DetectorGeometry, new_coupling=None):
        if new_coupling is not None:
            rescale=power(new_coupling/self.ge, 2)
            super().propagate_iso_vol_int(geom, W_ee(new_coupling, self.ma), rescale)
        else:
            super().propagate_iso_vol_int(geom, W_ee(self.ge, self.ma))




class FluxResonanceIsotropic(AxionFlux):
    """
    Generator for e+ e- resonant ALP production flux
    Takes in a flux of positrons
    """
    def __init__(self, positron_flux=[[1.,0.]], target=Material("W"), target_length=10.0,
                 det_dist=4., det_length=0.2, det_area=0.04, axion_mass=0.1, axion_coupling=1e-3,
                 n_samples=100, is_isotropic=True, loop_decay=False, boson_type="pseudoscalar",
                 max_track_length=5.0, **kwargs):
        # TODO: make flux take in a Detector class and a Target class (possibly Material class?)
        # Replace A = 2*Z with real numbers of nucleons
        super().__init__(axion_mass, target, det_dist, det_length, det_area, n_samples)
        self.positron_flux = positron_flux  # differential positron energy flux dR / dE+ / s
        self.positron_flux_bin_widths = positron_flux[1:,0] - positron_flux[:-1,0]
        self.ge = axion_coupling
        self.target_radius = target_length  # cm
        self.ntarget_area_density = target.rad_length * AVOGADRO / (target.z[0] + target.n[0])  # N_T / cm^2
        self.is_isotropic = is_isotropic
        self.loop_decay = loop_decay
        self.boson_type = boson_type
        self.max_t = max_track_length

    def decay_width(self, ge, ma):
        if self.loop_decay:
            return W_ee(ge, ma) + W_gg_loop(ge, ma, M_E)
        else:
            return W_ee(ge, ma)

    def positron_flux_dN_dE(self, energy):
        return np.interp(energy, self.positron_flux[:,0], self.positron_flux[:,1], left=0.0, right=0.0)

    def positron_flux_attenuated(self, t, energy_pos, energy_res):
        return self.positron_flux_dN_dE(energy_pos) * track_length_prob(energy_pos, energy_res, t)
    
    def resonance_peak(self):
        if self.boson_type == "pseudoscalar":
            return 2*pi*M_E*power(self.ge / self.ma, 2) / sqrt(1 - power(2*M_E/self.ma, 2))
        elif self.boson_type == "vector":
            # based on hand calculation equating 1802.04756 formula in the delta func limit
            return 12*pi*self.ge**2 * ALPHA / M_E / 6

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
        t_rnd = np.random.uniform(0.0, self.max_t, self.n_samples)
        mc_vol = self.max_t*(max(self.positron_flux[:,0]) - resonant_energy)

        attenuated_flux = mc_vol*np.sum(self.positron_flux_attenuated(t_rnd, e_rnd, resonant_energy))/self.n_samples
        wgt = self.target_z * (self.ntarget_area_density * HBARC**2) * self.resonance_peak() * attenuated_flux

        self.axion_energy.append(self.ma**2 / (2 * M_E))
        self.axion_flux.append(wgt)

    def propagate(self, new_coupling=None):
        if new_coupling is not None:
            super().propagate(self.decay_width(new_coupling, self.ma),
                rescale_factor=power(new_coupling/self.ge, 2))
        else:
            super().propagate(self.decay_width(self.ge, self.ma))

        if self.is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept
    
    def propagate_iso_vol_int(self, geom: DetectorGeometry, new_coupling=None):
        if new_coupling is not None:
            rescale=power(new_coupling/self.ge, 2)
            super().propagate_iso_vol_int(geom, W_ee(new_coupling, self.ma), rescale)
        else:
            super().propagate_iso_vol_int(geom, W_ee(self.ge, self.ma))




class FluxPairAnnihilationIsotropic(AxionFlux):
    """
    Generator associated production via electron-positron annihilation
    (e+ e- -> a gamma)
    Takes in a flux of positrons
    """
    def __init__(self, positron_flux=[[1.,0.]], target=Material("W"),
                 det_dist=4., det_length=0.2, det_area=0.04, axion_mass=0.1,
                 axion_coupling=1e-3, n_samples=100, is_isotropic=True,
                 loop_decay=False, **kwargs):
        # TODO: make flux take in a Detector class and a Target class (possibly Material class?)
        # Replace A = 2*Z with real numbers of nucleons
        super().__init__(axion_mass, target, det_dist, det_length, det_area, n_samples)
        self.positron_flux = positron_flux  # differential positron energy flux dR / dE+ / s
        self.positron_flux_bin_widths = positron_flux[1:,0] - positron_flux[:-1,0]
        self.ge = axion_coupling
        self.ntarget_area_density = target.rad_length * AVOGADRO / (2*target.z[0])
        self.is_isotropic = is_isotropic
        self.loop_decay = loop_decay

        self.mc = Scatter2to2MC(M2AssociatedProduction(axion_mass), p2=LorentzVector(M_E, 0.0, 0.0, 0.0), n_samples=n_samples)

    def decay_width(self, ge, ma):
        if self.loop_decay:
            return W_ee(ge, ma) + W_gg_loop(ge, ma, M_E)
        else:
            return W_ee(ge, ma)

    def p1_cm(self, s):
        return np.sqrt((np.power(s - 2*M_E**2, 2) - np.power(2*M_E**2, 2))/(4*s))

    def p3_cm(self, s):
        return np.sqrt((np.power(s - self.ma**2, 2))/(4*s))

    def simulate_single(self, positron):
        ep_lab = positron[0]
        pos_wgt = positron[1]

        if ep_lab < max((self.ma**2 - M_E**2)/(2*M_E), M_E):
            # Threshold check
            # ATTN: USING 20% CUTOFF TO CURB IR DIVERGENCE
            return

        # Simulate ALPs produced in the CM frame
        self.mc.lv_p1 = LorentzVector(ep_lab, 0.0, 0.0, np.sqrt(ep_lab**2 - M_E**2))
        self.mc.scatter_sim()
        ea_lab, lab_weights = self.mc.get_e3_lab_weights()

        self.axion_energy.extend(ea_lab)
        self.axion_flux.extend(pos_wgt * self.target_z * self.ge**2 * self.ntarget_area_density * HBARC**2 * lab_weights)


    def simulate(self):
        self.axion_energy = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        self.mc = Scatter2to2MC(M2AssociatedProduction(self.ma), p2=LorentzVector(M_E, 0.0, 0.0, 0.0), n_samples=self.n_samples)

        for i, el in enumerate(self.positron_flux):
            self.simulate_single(el)

    def propagate(self, new_coupling=None):
        if new_coupling is not None:
            super().propagate(self.decay_width(new_coupling, self.ma),
                rescale_factor=power(new_coupling/self.ge, 2))
        else:
            super().propagate(self.decay_width(self.ge, self.ma))

        if self.is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept
    
    def propagate_iso_vol_int(self, geom: DetectorGeometry, new_coupling=None):
        if new_coupling is not None:
            rescale=power(new_coupling/self.ge, 2)
            super().propagate_iso_vol_int(geom, W_ee(new_coupling, self.ma), rescale)
        else:
            super().propagate_iso_vol_int(geom, W_ee(self.ge, self.ma))




class FluxNuclearIsotropic(AxionFlux):
    """
    Takes in a rate (#/s) of nuclear decays for a specified nuclear transition
    Produces the associated ALP flux from a given branching ratio
    """
    def __init__(self, transition_rates=np.array([[1.0, 0.0, 1.0, 0.5]]), target=Material("W"),
                 det_dist=4., det_length=0.2, det_area=0.04, is_isotropic=True,
                 axion_mass=0.1, gann0=1e-3, gann1=1e-3,  gae=0.0, gagamma=0.0, n_samples=100):
        super().__init__(axion_mass, target, det_dist, det_length, det_area, n_samples)
        self.rates = transition_rates
        self.gann0 = gann0
        self.gann1 = gann1
        self.gae = gae
        self.gagamma = gagamma
        self.is_isotropic = is_isotropic

    def br(self, energy, j=1, delta=0.0, beta=1.0, eta=0.5):
        mu0 = 0.88
        mu1 = 4.71
        return ((j/(j+1)) / (1 + delta**2) / pi / ALPHA) \
            * power(sqrt(energy**2 - self.ma**2)/energy, 2*j + 1) \
                * power((self.gann0 * beta + self.gann1)/((mu0-0.5)*beta + (mu1 - eta)), 2)

    def decay_width(self):
        return W_ee(self.gae, self.ma) + W_ee(self.gagamma, self.ma)

    def simulate(self, j=1, delta=0.0):
        self.axion_energy = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        for i in range(self.rates.shape[0]):
            if self.rates[i,0] > self.ma:
                self.axion_energy.append(self.rates[i,0])
                self.axion_flux.append(self.rates[i,1] * self.br(self.rates[i,0], j, delta, self.rates[i,2], self.rates[i,3]))

    def propagate(self, ):
        super().propagate(W_ee(self.gae, self.ma))

        if self.is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept

    def propagate_iso_vol_int(self, geom: DetectorGeometry):
        super().propagate_iso_vol_int(geom, self.decay_width())




class FluxChargedMeson3BodyDecay(AxionFlux):
    def __init__(self, meson_flux, boson_mass=0.1, coupling=1.0, n_samples=50, meson_type="pion",
                 interaction_model="scalar_ib1", energy_cut=140.0, det_dist=541, det_length=10.0,
                 det_area=25.0*pi, c0=-0.95, lepton_masses=[M_E, M_MU], verbose=False):
        super().__init__(boson_mass, Material("Be"), det_dist, det_length, det_area, n_samples=n_samples)
        self.meson_flux = meson_flux
        param_dict = {
            "pion": [M_PI, V_UD, F_PI, PION_WIDTH],
            "kaon": [M_K, V_US, F_K, KAON_WIDTH]
        }
        decay_params = param_dict[meson_type]
        self.model = interaction_model
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
        self.verbose = verbose

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

    def simulate_single(self, meson, cut_on_solid_angle=True, decay_pos=0.0, ml=M_E, multicore=False):
        # check valid decay position
        if (decay_pos > 52.0):
            if multicore:
                return 0.0, 0.0, 0.0, 0.0, decay_pos, 0.0
            else:
                return

        # takes in meson = [p, theta, wgt]
        ea_min = self.ma
        ea_max = (self.mm**2 + self.ma**2 - ml**2)/(2*self.mm)

        # Boost to lab frame
        beta = meson[0] / sqrt(meson[0]**2 + self.mm**2)
        boost = power(1-beta**2, -0.5)

        # compute solid angle acceptance for this decay position
        solid_angle_cosine = cos(arctan(self.det_length/(self.det_dist-decay_pos)))
        min_cm_cos = cos(min(boost * arccos(solid_angle_cosine), pi))
        # Draw random variate energies and angles in the pion rest frame
        energies = np.random.uniform(ea_min, ea_max, self.n_samples)
        momenta = sqrt(energies**2 - self.ma**2)
        cosines = np.random.uniform(min_cm_cos, 1, self.n_samples)
        pz = momenta*cosines

        # get energy, momentum, and angles along pion direction
        e_lab = boost*(energies + beta*pz)
        pz_lab = boost*(pz + beta*energies)
        theta_lab = arccos(pz_lab / sqrt(e_lab**2 - self.ma**2))
        phi_lab = np.random.uniform(0, 2*pi, self.n_samples)

        # get theta along beam direction
        thetas_z = arccos(cos(theta_lab)*cos(meson[1]) + cos(phi_lab)*sin(theta_lab)*sin(meson[1]))

        # Monte Carlo volume, making sure to use the lab frame energy range
        # V = 2pi * (1-min_cm_cos) * (Ea^max - Ea^min) / N
        mc_vol_lab = (1.0-min_cm_cos) * boost * beta * sqrt(e_lab**2 - self.ma**2)  # uses jacobian factor

        # Draw weights from the PDF
        weights = np.array([meson[2]*mc_vol_lab[i]*self.dGammadEa(energies[i], ml)/self.total_width/self.n_samples \
            for i in range(energies.shape[0])])

        if multicore:
            return e_lab, cos(theta_lab), thetas_z, solid_angle_cosine*np.ones(self.n_samples), \
                    decay_pos*np.ones(self.n_samples), weights*heaviside(e_lab-self.energy_cut,1.0)
        else:
            for i in range(self.n_samples):
                solid_angle_acceptance = heaviside(cos(theta_lab[i]) - solid_angle_cosine, 0.0)
                if solid_angle_acceptance == 0.0 and cut_on_solid_angle:
                    continue
                self.axion_energy.append(e_lab[i])
                self.cosines.append(cos(theta_lab[i]))
                self.axion_flux.append(weights[i]*heaviside(e_lab[i]-self.energy_cut,1.0))
                self.axion_angle.append(thetas_z[i])
                self.solid_angles.append(solid_angle_cosine)
                self.decay_pos.append(decay_pos)

    def simulate(self, cut_on_solid_angle=True, verbose=False, multicore=False):
        self.axion_energy = []
        self.cosines = []
        self.axion_flux = []    
        self.scatter_axion_weight = []
        self.decay_axion_weight = []
        self.decay_pos = []
        self.solid_angles = []

        decay_positions = []
        for i, p in enumerate(self.meson_flux):
            # Simulate decay positions between target and dump
            # The quantile is truncated at the dump position via umax
            x = expon.rvs(scale=(p[0]*PION_LIFETIME*0.01*C_LIGHT/M_PI))
            decay_positions.append(x)

            if multicore == False:
                if verbose:
                    print("Simulating meson decay for i={}".format(i))
                for ml in self.lepton_masses:
                    if self.ma > self.mm - ml:
                        continue
                    self.simulate_single(p, cut_on_solid_angle, x, ml, multicore)
        
        if multicore == True:
            for ml in self.lepton_masses:
                if self.ma > self.mm - ml:
                    continue
                print("Running NCPU = ", max(1, multi.cpu_count()-1))
                
                with multi.Pool(max(1, multi.cpu_count()-1)) as pool:
                    ntuple = [pool.apply_async(self.simulate_single,
                                               args=(self.meson_flux[i], cut_on_solid_angle, decay_positions[i], ml, multicore))
                                                            for i in range(self.meson_flux.shape[0])]                    
                    for tup in ntuple:
                        sim_data = tup.get()
                        self.axion_energy.extend(sim_data[0])
                        self.cosines.extend(sim_data[1])
                        self.axion_angle.extend(sim_data[2])
                        self.solid_angles.extend(sim_data[3])
                        self.decay_pos.extend(sim_data[4])
                        self.axion_flux.extend(sim_data[5])

                    pool.close()


        #for i, p in enumerate(self.meson_flux):
        #    if verbose:
        #        print("Simulating meson decay i={}".format(i))
        #    # Simulate decay positions between target and dump
        #    # The quantile is truncated at the dump position via umax
        #    decay_l = METER_BY_MEV * p[0] / self.total_width / self.mm
        #    umax = exp(-2*self.dump_dist/decay_l) * power(exp(self.dump_dist/decay_l) - 1, 2) \
        #        if decay_l > 1.0 else 1.0
        #    try:
        #        u = np.random.uniform(0.0, min(umax, 1.0))
        #    except:
        #        print("umax = ", umax, " decay l = ", decay_l, p[0])
        #    x = decay_quantile(u, p[0], self.mm, self.total_width)

        #    # Simulate decays for each charged meson
        #    for ml in self.lepton_masses:
        #        if self.ma > self.mm - ml:
        #            continue
        #        self.simulate_single(p, cut_on_solid_angle, x, ml)

    def propagate(self, decay=None, new_coupling=None):  # propagate to detector
        # decay options: 'photon', 'electron'
        wgt = np.array(self.axion_flux)
        if decay:
            if new_coupling is not None:
                width = np.sum([W_ff(new_coupling, mf, self.ma) for mf in self.lepton_masses])
                rescale=power(new_coupling/self.gmu, 2)
                super().propagate(width, rescale)
            else:
                width = np.sum([W_ff(self.gmu, mf, self.ma) for mf in self.lepton_masses])
                super().propagate(width)
        else:
            # Do not decay
            self.decay_axion_weight = np.asarray(wgt*0.0, dtype=np.float64)
            self.scatter_axion_weight = np.asarray(wgt, dtype=np.float64)




class FluxChargedMeson3BodyIsotropic(AxionFlux):
    def __init__(self, meson_flux=[[0.0, 0.0259]], boson_mass=0.1, coupling=1.0, meson_type="pion",
                 interaction_model="scalar_ib1", det_dist=20, det_length=2, det_area=2,
                 target=Material("W"), n_samples=50, c0=-0.97, lepton_masses=[M_E, M_MU], abd=(0,0,0)):
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
        self.m2_e = M2Meson3BodyDecay(boson_mass, meson_type, M_E, interaction_model, abd)
        self.m2_mu = M2Meson3BodyDecay(boson_mass, meson_type, M_MU, interaction_model, abd)

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
        
        self.axion_energy.extend(e_lab)
        self.axion_flux.extend(weights)

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




class FluxNeutralMeson2BodyDecay(AxionFlux):
    def __init__(self, meson_flux, flux_weight, boson_mass=0.1, coupling=1.0, meson_mass=M_PI0,
                 det_dist=20.0, det_area=2.0, det_length=1.0, apply_angle_cut=False,
                 n_samples=10, off_axis_angle=0.0):
        super().__init__(boson_mass, Material("Be"), det_dist, det_length, det_area)
        self.meson_flux = meson_flux  # Take nx4 array of 4-vectors (px, py, pz, E)
        self.m_meson = meson_mass
        self.n_samples = n_samples
        self.g = coupling
        self.boson_mass = boson_mass
        self.mm = M_PI0
        self.apply_angle_cut = apply_angle_cut
        self.flux_weight = flux_weight

        # Get detector solid angle
        self.off_axis_angle = off_axis_angle
        self.phi_range = 2*pi
        if off_axis_angle > 0.0:
            self.phi_range = 2*self.det_sa()

    def br(self):
        if self.ma > self.m_meson:
            return 0.0
        return 2 * (self.g)**2 * (1 - power(self.ma / self.m_meson, 2))**3 / (4*pi*ALPHA)

    def simulate(self):
        for m in self.meson_flux:
            pi0_p4 = LorentzVector(m[3], m[0], m[1], m[2])
            mc = Decay2Body(pi0_p4, m1=self.boson_mass, m2=0.0, n_samples=self.n_samples)
            mc.decay()

            ap_energies = np.array([lv.energy() for lv in mc.p1_lab_4vectors])
            ap_thetas = np.array([lv.theta() for lv in mc.p1_lab_4vectors])
            weights = self.br() * mc.weights * self.flux_weight

            if self.apply_angle_cut:
                theta_mask = (ap_thetas < self.det_sa() + self.off_axis_angle) \
                    * (ap_thetas > self.off_axis_angle - self.det_sa())
                ap_energies = ap_energies[theta_mask]
                weights = weights[theta_mask] * self.phi_range/(2*pi)  # Assume azimuthal symmetry
                ap_thetas = ap_thetas[theta_mask]
            else:
                weights = self.det_area * weights / (4*pi*self.det_dist**2)

            self.axion_flux.extend(weights)
            self.axion_energy.extend(ap_energies)
            self.axion_angle.extend(ap_thetas)

    def propagate(self):
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
