import sys
sys.path.append("../")

from alplib.fluxes import *
from alplib.generators import *

import matplotlib.pyplot as plt

# NA64 simulation for electron coupling

# missing energy search EOT
beam_energy = 1e5  # 100 GeV
eot = 2.84e11
days = 365

na64_dump = Material("W")
na64_ecal = Material("Pb")
ecal_length = 1.0  # meters
hcal_length = 6.5  # meters
hcal_area = 0.6*0.6  # 60cm across
ecal_dist = 43.3  # meters: beam d
ecal_area = 0.23  # ecal area in m^2; 43cm across
dump_length = 2.0  # meters
ecal_thresh = 1.0e3  # check
na64_e_flux = np.array([[beam_energy, eot/S_PER_DAY/days]])
det_am = na64_ecal.m[0] # mass of target atom in MeV
ecal_n = 100.0 * MEV_PER_KG / det_am


def get_events(ma, g, use_loop=False):
    flux_brem = FluxBremIsotropic(na64_e_flux, target=na64_dump, det_dist=ecal_dist, det_length=ecal_length,
                                    det_area=ecal_area, target_length=dump_length, axion_mass=ma,
                                    axion_coupling=g, loop_decay=use_loop, is_isotropic=False, n_samples=10000)
    
    flux_brem.simulate()
    flux_brem.propagate()
    
    events_gen = ElectronEventGenerator(flux_brem, na64_ecal)
    events_gen.compton(ge=g, ma=ma, ntargets=ecal_n, days_exposure=days, threshold=ecal_thresh)
    events_gen.decays(days_exposure=days, threshold=ecal_thresh)
    return events_gen.axion_energy, events_gen.decay_weights + events_gen.scatter_weights