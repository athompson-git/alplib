# Class to hold detector-specific constants, dimensions, and responses

from .constants import *
from .fmath import *

import json
import pkg_resources


class Detector:
    """
    detector class
    """
    def __init__(self, det_type, fiducial_mass=1.0, volume=1.0, density=1.0, efficiency=None):
        """
        initializing Detector,
        it reads ./det_params.json for detector information,
        :param det_type: name of the detector
        all units in MeV, kg, cm, unless otherwise stated (e.g. lattice params)
        """
        self.det_type = det_type
        self.efficiency = efficiency
        fpath = pkg_resources.resource_filename(__name__, '../data/det_params.json')
        f = open(fpath, 'r')
        det_file = json.load(f)
        f.close()
        if det_type.lower() in det_file:
            det_info = det_file[det_type.lower()]
            self.iso = det_info['iso']
            self.z = np.array(det_info['z'])
            self.n = np.array(det_info['n'])
            self.m = np.array(det_info['m'])
            self.frac = np.array(det_info['frac'])
            self.lattice_const = np.array([det_info['lattice_const']])  # Angstroms
            self.cell_volume = np.array([det_info['cell_volume']])  # Angstroms^3
            self.r0 = np.array([det_info['atomic_radius']])  # Angstroms
            self.er_min = det_info['er_min']
            self.er_max = det_info['er_max']
            self.bg = det_info['bg']
            self.bg_un = det_info['bg_un']
            self.density = det_info['density']  # g/cm^3
            self.fid_mass = fiducial_mass  # kg
            self.volume = volume  # cm^3
            self.ntargets = density * volume / (np.sum(self.m) / MEV_PER_KG / 1e-3)
        else:
            raise Exception("No such detector in det_params.json.")
